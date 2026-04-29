"""
WavePriorNet Test / Evaluation Script

Usage:
    # 在 REVIDE_indoor Test split 上评估，计算 PSNR/SSIM，保存结果
    python test.py --data_root /path/to/REVIDE_indoor --ckpt checkpoints/run_xxx/best.pth

    # 推理单个视频文件夹（无 GT）
    python test.py --hazy_dir /path/to/hazy --color_dir /path/to/color_ref \\
                   --ckpt checkpoints/run_xxx/best.pth --save_dir results/

结果保存在 --save_dir 下（默认与 ckpt 同目录下的 eval_<timestamp>/）：
    eval_results.json    总体 PSNR / SSIM / 推理时间
    per_frame.csv        逐帧 PSNR / SSIM
    pred/                预测图像（--save_images 时）
    gt/                  GT 图像（--save_images 时）
    hazy/                输入有雾图（--save_images 时）
"""

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from core.dataset import REVIDEDataset
from model_wavepriornet import WavePriorNet


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(max_val ** 2 / mse)


def ssim_single(pred, gt, window_size=11):
    """SSIM for a single [3, H, W] pair in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pred = pred.unsqueeze(0)
    gt   = gt.unsqueeze(0)

    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(3, 1, 1, 1).to(pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=3)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=3) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, save_dir=None):
    model.eval()
    all_psnr, all_ssim = [], []
    per_frame_rows = []
    infer_times = []

    for batch_idx, (hazy, color, gt, seq_names, frame_names_batch) in enumerate(loader):
        hazy  = hazy.to(device)
        color = color.to(device)
        gt    = gt.to(device)

        t0 = time.time()
        output = model(hazy, color)
        infer_times.append(time.time() - t0)

        pred = output['out'].clamp(0, 1)
        B, T, C, H, W = gt.shape
        gt_flat    = gt.view(B * T, C, H, W)
        hazy_flat  = hazy.view(B * T, C, H, W)
        color_flat = color.view(B * T, C, H, W)

        # frame_names_batch: list of T lists, each length B (DataLoader collate 转置了)
        # 转成 B*T 个 (seq, fname) 对
        flat_meta = []
        for b in range(B):
            seq = seq_names[b]
            for t in range(T):
                flat_meta.append((seq, frame_names_batch[t][b]))

        for i, (seq, fname) in enumerate(flat_meta):
            p = psnr(pred[i:i+1], gt_flat[i:i+1])
            s = ssim_single(pred[i], gt_flat[i])
            all_psnr.append(p)
            all_ssim.append(s)
            per_frame_rows.append({
                'seq':   seq,
                'frame': fname,
                'psnr':  round(p, 4),
                'ssim':  round(s, 4),
            })

            if save_dir:
                for sub in ('pred', 'gt', 'hazy', 'color'):
                    (Path(save_dir) / sub / seq).mkdir(parents=True, exist_ok=True)
                save_image(pred[i],       str(Path(save_dir) / 'pred'  / seq / f'{fname}.png'))
                save_image(gt_flat[i],    str(Path(save_dir) / 'gt'    / seq / f'{fname}.png'))
                save_image(hazy_flat[i],  str(Path(save_dir) / 'hazy'  / seq / f'{fname}.png'))
                save_image(color_flat[i], str(Path(save_dir) / 'color' / seq / f'{fname}.png'))

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}]  "
                  f"PSNR={sum(all_psnr)/len(all_psnr):.2f}  "
                  f"SSIM={sum(all_ssim)/len(all_ssim):.4f}")

    avg_psnr     = sum(all_psnr) / len(all_psnr)
    avg_ssim     = sum(all_ssim) / len(all_ssim)
    avg_infer_ms = sum(infer_times) / len(infer_times) * 1000

    return avg_psnr, avg_ssim, avg_infer_ms, per_frame_rows


# ---------------------------------------------------------------------------
# Single-folder inference (no GT)
# ---------------------------------------------------------------------------

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}


@torch.no_grad()
def infer_folder(model, hazy_dir, color_dir, save_dir, device, num_frames, crop_size):
    model.eval()
    to_tensor   = transforms.ToTensor()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    hazy_paths  = sorted([p for p in Path(hazy_dir).iterdir()  if p.suffix in IMG_EXTENSIONS])
    color_paths = sorted([p for p in Path(color_dir).iterdir() if p.suffix in IMG_EXTENSIONS])
    n = min(len(hazy_paths), len(color_paths))
    print(f"Found {n} frames in {hazy_dir}")

    for start in range(0, n - num_frames + 1, num_frames):
        hazy_imgs  = [Image.open(hazy_paths[start + i]).convert('RGB')  for i in range(num_frames)]
        color_imgs = [Image.open(color_paths[start + i]).convert('RGB') for i in range(num_frames)]

        if crop_size:
            from torchvision.transforms import functional as TF
            hazy_imgs  = [TF.center_crop(f, crop_size) for f in hazy_imgs]
            color_imgs = [TF.center_crop(f, crop_size) for f in color_imgs]

        hazy  = torch.stack([to_tensor(f) for f in hazy_imgs]).unsqueeze(0).to(device)
        color = torch.stack([to_tensor(f) for f in color_imgs]).unsqueeze(0).to(device)

        pred = model(hazy, color)['out'].clamp(0, 1)

        for i in range(num_frames):
            fname = hazy_paths[start + i].stem
            save_image(pred[i], str(Path(save_dir) / f'{fname}_pred.png'))

    print(f"Results saved to {save_dir}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--data_root',   type=str, default=None)
    p.add_argument('--split',       type=str, default='Test')
    p.add_argument('--hazy_dir',    type=str, default=None)
    p.add_argument('--color_dir',   type=str, default=None)
    p.add_argument('--save_dir',    type=str, default=None,
                   help='结果保存目录。默认为 ckpt 同级目录下的 eval_<timestamp>/')
    p.add_argument('--num_frames',  type=int, default=5)
    p.add_argument('--crop_size',   type=int, default=None)
    p.add_argument('--hidden',      type=int, default=128)
    p.add_argument('--stack_num',   type=int, default=6)
    p.add_argument('--batch_size',  type=int, default=1)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_images', action='store_true',
                   help='保存 pred / gt / hazy 图像到 save_dir')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 确定结果保存目录 ----
    if args.save_dir is None:
        ckpt_parent = Path(args.ckpt).parent
        ts = datetime.now().strftime('eval_%Y%m%d_%H%M%S')
        save_dir = ckpt_parent / ts
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # ---- 加载模型 ----
    model = WavePriorNet(
        num_frames=args.num_frames,
        hidden=args.hidden,
        stack_num=args.stack_num,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    ckpt_epoch    = ckpt.get('epoch', '?')
    ckpt_best_psnr = ckpt.get('best_psnr', 0.0)
    print(f"Loaded: {args.ckpt}  (epoch={ckpt_epoch}, best_psnr={ckpt_best_psnr:.2f})")

    # ---- 数据集评估 ----
    if args.data_root is not None:
        crop = (args.crop_size, args.crop_size) if args.crop_size else None
        ds = REVIDEDataset(
            args.data_root, split=args.split,
            num_frames=args.num_frames,
            crop_size=crop,
            augment=False,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"Evaluating on {args.split} split ({len(ds)} samples)...")

        img_save_dir = str(save_dir) if args.save_images else None
        t_start = time.time()
        avg_psnr, avg_ssim, avg_infer_ms, per_frame_rows = evaluate(
            model, loader, device, save_dir=img_save_dir
        )
        total_eval_time = time.time() - t_start

        # ---- 打印结果 ----
        print(f"\n{'='*45}")
        print(f"  PSNR  : {avg_psnr:.4f} dB")
        print(f"  SSIM  : {avg_ssim:.4f}")
        print(f"  Avg inference time : {avg_infer_ms:.1f} ms/batch")
        print(f"  Total eval time    : {total_eval_time:.1f} s")
        print(f"{'='*45}")

        # ---- 保存 eval_results.json ----
        result_dict = {
            'ckpt':              args.ckpt,
            'ckpt_epoch':        ckpt_epoch,
            'ckpt_best_psnr':    ckpt_best_psnr,
            'eval_time':         datetime.now().isoformat(),
            'split':             args.split,
            'num_samples':       len(ds),
            'num_frames':        args.num_frames,
            'crop_size':         args.crop_size,
            'avg_psnr_db':       round(avg_psnr, 4),
            'avg_ssim':          round(avg_ssim, 4),
            'avg_infer_ms':      round(avg_infer_ms, 2),
            'total_eval_time_s': round(total_eval_time, 1),
            'save_dir':          str(save_dir),
        }
        json_path = save_dir / 'eval_results.json'
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"  eval_results.json → {json_path}")

        # ---- 保存 per_frame.csv ----
        csv_path = save_dir / 'per_frame.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['seq', 'frame', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(per_frame_rows)
        print(f"  per_frame.csv      → {csv_path}")

    # ---- 单文件夹推理 ----
    elif args.hazy_dir is not None and args.color_dir is not None:
        infer_folder(
            model, args.hazy_dir, args.color_dir,
            str(save_dir), device, args.num_frames, args.crop_size,
        )
    else:
        print("请提供 --data_root 或 (--hazy_dir 和 --color_dir)。")


if __name__ == '__main__':
    main()
