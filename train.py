"""
WavePriorNet Training Script

Recommended for A6000 48GB:
    python train.py --data_root /path/to/REVIDE_indoor --batch_size 8 --crop_size 512
    python train.py --data_root /path/to/REVIDE_indoor --batch_size 16 --crop_size 512 --amp

每次训练自动在 --save_dir 下创建带时间戳的 run 目录，例如：
    checkpoints/run_20240427_143022/
        config.json          训练超参数
        train_log.csv        每 epoch：loss / PSNR / LR / epoch_time
        train.log            完整终端输出
        best.pth             最优 checkpoint
        latest.pth           最新 checkpoint
        tb_logs/             TensorBoard 事件文件
        summary.json         训练结束后的汇总（最优 PSNR、总时长等）

Loss design (MAP-Net map.py verbatim):
    loss_main  = L1(pred, gt)
    loss_phy_j = sum_s [ 0.2 / 2^s * L1(aux_j[s], gt) ]
    loss_phy_i = sum_s [ 0.2 / 2^s * L1(aux_i[s], hazy_in) ]
    loss_perc  = L1(VGG(pred), VGG(gt))
    loss_chamf = ChamferColor(pred, gt)
    total = loss_main + loss_phy_j + loss_phy_i
          + lambda_perc * loss_perc + lambda_chamfer * loss_chamf

LR schedule: linear warmup (10% of epochs) → CosineAnnealingLR
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16

from core.dataset import REVIDEDataset
from model_wavepriornet import WavePriorNet


# ---------------------------------------------------------------------------
# Tee: 同时写终端和文件
# ---------------------------------------------------------------------------

class Tee:
    """将 stdout 同时写入文件。"""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', buffering=1, encoding='utf-8')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ---------------------------------------------------------------------------
# Perceptual Loss  –  VGG16 relu2_2
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='DEFAULT').features[:9]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(pred), self.vgg(target))


# ---------------------------------------------------------------------------
# Chamfer Color Loss  –  color-alignment-in-diffusion (subsampled)
# ---------------------------------------------------------------------------

def chamfer_color_loss(pred, gt, num_points=512):
    N, C, H, W = pred.shape
    HW = H * W
    p = pred.view(N, C, -1).permute(0, 2, 1)
    g = gt.view(N, C, -1).permute(0, 2, 1)
    idx = torch.randperm(HW, device=pred.device)[:num_points]
    p, g = p[:, idx, :], g[:, idx, :]
    p2 = (p ** 2).sum(-1, keepdim=True)
    g2 = (g ** 2).sum(-1, keepdim=True)
    dist2 = (p2 + g2.transpose(1, 2) - 2 * torch.bmm(p, g.transpose(1, 2))).clamp(min=0)
    return dist2.min(dim=2).values.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_psnr   = ckpt.get('best_psnr', 0.0)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  [ckpt] loaded {path}  epoch={ckpt.get('epoch','?')}  best_psnr={best_psnr:.2f}")
    return start_epoch, best_psnr


def psnr_metric(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(max_val ** 2 / mse)


# ---------------------------------------------------------------------------
# Warmup + Cosine scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr, lr_min):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr            = lr
        self.lr_min        = lr_min

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """每 epoch 追加一行到 train_log.csv。"""

    FIELDS = ['epoch', 'lr', 'loss_total', 'loss_main', 'loss_phy_j',
              'loss_phy_i', 'loss_perc', 'loss_chamfer',
              'val_psnr', 'epoch_time_s']

    def __init__(self, path):
        self.path = path
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def write(self, row: dict):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction='ignore')
            writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                             for k, v in row.items()})


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(output, gt_flat, perc_loss_fn, args, device):
    pred   = output['out']
    img_in = output['img_in']

    loss_main  = F.l1_loss(pred, gt_flat)
    loss_phy_j = torch.tensor(0.0, device=device)
    loss_phy_i = torch.tensor(0.0, device=device)
    lambda_phy = 0.2

    if 'aux_j' in output:
        for s, aux_j in enumerate(output['aux_j']):
            loss_phy_j = loss_phy_j + (lambda_phy / 2 ** s) * F.l1_loss(aux_j, gt_flat)
    if 'aux_i' in output:
        for s, aux_i in enumerate(output['aux_i']):
            loss_phy_i = loss_phy_i + (lambda_phy / 2 ** s) * F.l1_loss(aux_i, img_in)

    loss_perc = (perc_loss_fn(pred, gt_flat)
                 if perc_loss_fn is not None and args.lambda_perc > 0
                 else torch.tensor(0.0, device=device))
    loss_chamfer = (chamfer_color_loss(pred, gt_flat)
                    if args.lambda_chamfer > 0
                    else torch.tensor(0.0, device=device))

    total = (loss_main + loss_phy_j + loss_phy_i
             + args.lambda_perc * loss_perc
             + args.lambda_chamfer * loss_chamfer)

    return total, {
        'main':    loss_main.item(),
        'phy_j':   loss_phy_j.item(),
        'phy_i':   loss_phy_i.item(),
        'perc':    loss_perc.item(),
        'chamfer': loss_chamfer.item(),
    }


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, perc_loss_fn, device, args, epoch, writer):
    model.train()
    total_loss  = 0.0
    accum       = {k: 0.0 for k in ('main', 'phy_j', 'phy_i', 'perc', 'chamfer')}
    t0          = time.time()

    accum_steps = args.accum_steps
    optimizer.zero_grad()

    for step, (hazy, color, gt) in enumerate(loader):
        hazy  = hazy.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)
        gt    = gt.to(device, non_blocking=True)

        B, T, C, H, W = gt.shape
        gt_flat = gt.view(B * T, C, H, W)

        with autocast(enabled=args.amp):
            output = model(hazy, color)
            loss, loss_dict = compute_loss(output, gt_flat, perc_loss_fn, args, device)

        # normalize loss so gradients are averaged across accumulation steps
        loss_scaled = loss / accum_steps

        if args.amp:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_update_step = (step + 1) % accum_steps == 0 or (step + 1) == len(loader)
        if is_update_step:
            if args.amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        for k in accum:
            accum[k] += loss_dict[k]

        global_step = epoch * len(loader) + step
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  Ep{epoch:03d} [{step:4d}/{len(loader)}]  "
                f"loss={loss.item():.4f}  "
                f"main={loss_dict['main']:.4f}  "
                f"phy_j={loss_dict['phy_j']:.4f}  "
                f"phy_i={loss_dict['phy_i']:.4f}  "
                f"perc={loss_dict['perc']:.4f}  "
                f"chamf={loss_dict['chamfer']:.5f}  "
                f"({elapsed:.1f}s)"
            )
            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
                writer.add_scalar('train/total', loss.item(), global_step)

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in accum.items()}, time.time() - t0


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr, count = 0.0, 0
    for hazy, color, gt in loader:
        hazy  = hazy.to(device)
        color = color.to(device)
        gt    = gt.to(device)
        pred  = model(hazy, color)['out']
        B, T, C, H, W = gt.shape
        total_psnr += psnr_metric(pred.clamp(0, 1), gt.view(B * T, C, H, W))
        count += 1
    return total_psnr / max(count, 1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',      type=str, required=True)
    p.add_argument('--num_frames',     type=int,   default=5)
    p.add_argument('--crop_size',      type=int,   default=512)
    p.add_argument('--hidden',         type=int,   default=128)
    p.add_argument('--stack_num',      type=int,   default=6)
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--warmup_epochs',  type=int,   default=10)
    p.add_argument('--batch_size',     type=int,   default=8)
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--lr_min',         type=float, default=1e-6)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--lambda_perc',    type=float, default=0.1)
    p.add_argument('--lambda_chamfer', type=float, default=0.05)
    p.add_argument('--amp',            action='store_true', default=False)
    p.add_argument('--accum_steps',    type=int,   default=1,
                   help='gradient accumulation steps (effective_batch = batch_size * accum_steps)')
    p.add_argument('--num_workers',    type=int,   default=8)
    p.add_argument('--save_dir',       type=str,   default='checkpoints')
    p.add_argument('--log_every',      type=int,   default=20)
    p.add_argument('--val_every',      type=int,   default=5)
    p.add_argument('--resume',         type=str,   default=None)
    p.add_argument('--no_tb',          action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 创建带时间戳的 run 目录 ----
    run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_dir  = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 同时写终端和 train.log ----
    tee = Tee(run_dir / 'train.log')
    sys.stdout = tee

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    # ---- 保存 config.json ----
    config = vars(args)
    config['run_dir']    = str(run_dir)
    config['start_time'] = datetime.now().isoformat()
    config['device']     = str(device)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {run_dir / 'config.json'}")

    # ---- datasets ----
    train_ds = REVIDEDataset(
        args.data_root, split='Train',
        num_frames=args.num_frames,
        crop_size=(args.crop_size, args.crop_size),
        augment=True,
    )
    val_ds = REVIDEDataset(
        args.data_root, split='Test',
        num_frames=args.num_frames,
        crop_size=(args.crop_size, args.crop_size),
        augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # ---- model ----
    model = WavePriorNet(
        num_frames=args.num_frames,
        hidden=args.hidden,
        stack_num=args.stack_num,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # ---- losses ----
    perc_loss_fn = PerceptualLoss().to(device) if args.lambda_perc > 0 else None

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.lr, args.lr_min
    )
    scaler = GradScaler(enabled=args.amp)

    # ---- resume ----
    start_epoch, best_psnr = 0, 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(args.resume, model, optimizer)

    # ---- TensorBoard ----
    writer = None if args.no_tb else SummaryWriter(log_dir=str(run_dir / 'tb_logs'))

    # ---- CSV logger ----
    csv_logger = CSVLogger(run_dir / 'train_log.csv')

    # ---- 训练配置打印 ----
    print(f"\n{'='*55}")
    print(f"  WavePriorNet  –  A6000 48GB  |  {run_name}")
    print(f"  batch={args.batch_size}  accum={args.accum_steps}  effective_batch={args.batch_size * args.accum_steps}  T={args.num_frames}  crop={args.crop_size}  AMP={args.amp}")
    print(f"  epochs={args.epochs}  warmup={args.warmup_epochs}  steps/ep={len(train_loader)}")
    print(f"  lr {args.lr}→{args.lr_min}  wd={args.weight_decay}")
    print(f"  params={n_params:.1f}M")
    print(f"  loss: L1 + phy_j(0.2/2^s) + phy_i(0.2/2^s) + {args.lambda_perc}*perc + {args.lambda_chamfer}*chamfer")
    print(f"{'='*55}\n")

    # ---- 训练循环 ----
    train_start = time.time()
    best_epoch  = -1

    for epoch in range(start_epoch, args.epochs):
        lr_now = scheduler.step(epoch)
        print(f"\nEpoch {epoch:03d}/{args.epochs}  lr={lr_now:.2e}")

        avg_loss, avg_loss_dict, epoch_time = train_one_epoch(
            model, train_loader, optimizer, scaler,
            perc_loss_fn, device, args, epoch, writer
        )

        if writer:
            writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            writer.add_scalar('train/lr', lr_now, epoch)

        # ---- validation ----
        val_psnr = float('nan')
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_psnr = validate(model, val_loader, device)
            print(f"  [val] PSNR={val_psnr:.2f} dB  best={best_psnr:.2f}  epoch_time={epoch_time:.1f}s")
            if writer:
                writer.add_scalar('val/psnr', val_psnr, epoch)

            ckpt = {
                'epoch':      epoch,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'best_psnr':  best_psnr,
                'args':       vars(args),
            }
            save_checkpoint(ckpt, run_dir / 'latest.pth')

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                ckpt['best_psnr'] = best_psnr
                save_checkpoint(ckpt, run_dir / 'best.pth')
                print(f"  [best] PSNR={best_psnr:.2f} dB  @ epoch {epoch}")

        # ---- CSV 记录 ----
        csv_logger.write({
            'epoch':         epoch,
            'lr':            lr_now,
            'loss_total':    avg_loss,
            'loss_main':     avg_loss_dict['main'],
            'loss_phy_j':    avg_loss_dict['phy_j'],
            'loss_phy_i':    avg_loss_dict['phy_i'],
            'loss_perc':     avg_loss_dict['perc'],
            'loss_chamfer':  avg_loss_dict['chamfer'],
            'val_psnr':      val_psnr if not math.isnan(val_psnr) else '',
            'epoch_time_s':  round(epoch_time, 1),
        })

    # ---- 训练结束：写 summary.json ----
    total_time = time.time() - train_start
    summary = {
        'run_name':          run_name,
        'run_dir':           str(run_dir),
        'finish_time':       datetime.now().isoformat(),
        'total_train_time_s': round(total_time, 1),
        'total_train_time_h': round(total_time / 3600, 2),
        'best_psnr_db':      round(best_psnr, 4),
        'best_epoch':        best_epoch,
        'total_epochs':      args.epochs,
        'params_M':          round(n_params, 2),
        'batch_size':        args.batch_size,
        'crop_size':         args.crop_size,
        'num_frames':        args.num_frames,
        'amp':               args.amp,
        'lr':                args.lr,
        'best_ckpt':         str(run_dir / 'best.pth'),
        'latest_ckpt':       str(run_dir / 'latest.pth'),
        'train_log_csv':     str(run_dir / 'train_log.csv'),
        'config_json':       str(run_dir / 'config.json'),
    }
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if writer:
        writer.close()

    print(f"\n{'='*55}")
    print(f"  Training done.")
    print(f"  Total time   : {total_time/3600:.2f} h  ({total_time:.0f} s)")
    print(f"  Best PSNR    : {best_psnr:.2f} dB  @ epoch {best_epoch}")
    print(f"  Run dir      : {run_dir}")
    print(f"  Summary      : {run_dir / 'summary.json'}")
    print(f"  Train log    : {run_dir / 'train_log.csv'}")
    print(f"{'='*55}")

    sys.stdout = tee.terminal
    tee.close()


if __name__ == '__main__':
    main()
