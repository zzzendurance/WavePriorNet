"""
单张推理脚本：给定一张有雾图和一张 gt_color_shift 图，输出去雾预测图。

用法示例：
    python infer_single.py \
        --ckpt  checkpoints/run_20260428_005635/best.pth \
        --hazy  /path/to/REVIDE_indoor/Test/hazy/C005/00000.JPG \
        --color /path/to/REVIDE_indoor/Test/gt_color_shift/C005/00000.JPG \
        --out   pred_00000.png

可选参数：
    --num_frames  5        模型需要 T 帧输入，不足时用同一张图复制填充（默认 5）
    --crop_size   512      center_crop 到指定大小（不填则保持原图，显存可能不足）
    --hidden      128      模型通道数，需与训练时一致
    --stack_num   6        Transformer 层数，需与训练时一致
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from model_wavepriornet import WavePriorNet


def load_image(path, crop_size):
    img = Image.open(path).convert('RGB')
    if crop_size:
        img = TF.center_crop(img, crop_size)
    return transforms.ToTensor()(img)   # [3, H, W], [0,1]


@torch.no_grad()
def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ---- 加载模型 ----
    model = WavePriorNet(
        num_frames=args.num_frames,
        hidden=args.hidden,
        stack_num=args.stack_num,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    best_psnr = ckpt.get('best_psnr', 0.0)
    print(f"Loaded : {args.ckpt}  (epoch={epoch}, best_psnr={best_psnr:.2f})")

    # ---- 读取图像 ----
    hazy_t  = load_image(args.hazy,  args.crop_size)   # [3,H,W]
    color_t = load_image(args.color, args.crop_size)   # [3,H,W]
    print(f"Hazy   : {args.hazy}  →  {tuple(hazy_t.shape)}")
    print(f"Color  : {args.color}  →  {tuple(color_t.shape)}")

    # 模型需要 T 帧，用同一张图复制填充
    T = args.num_frames
    hazy_seq  = hazy_t.unsqueeze(0).repeat(T, 1, 1, 1)    # [T,3,H,W]
    color_seq = color_t.unsqueeze(0).repeat(T, 1, 1, 1)   # [T,3,H,W]

    # 加 batch 维度：[1, T, 3, H, W]
    hazy_input  = hazy_seq.unsqueeze(0).to(device)
    color_input = color_seq.unsqueeze(0).to(device)

    # ---- 推理 ----
    output = model(hazy_input, color_input)
    # 取第 0 帧（所有帧相同，取任意一帧即可）
    pred = output['out'][0].clamp(0, 1).cpu()   # [3,H,W]

    # ---- 保存 ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred, str(out_path))
    print(f"Saved  : {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       type=str, required=True,  help='模型权重路径 (best.pth)')
    p.add_argument('--hazy',       type=str, required=True,  help='有雾输入图路径')
    p.add_argument('--color',      type=str, required=True,  help='gt_color_shift 图路径')
    p.add_argument('--out',        type=str, default='pred.png', help='输出图保存路径')
    p.add_argument('--num_frames', type=int, default=5,      help='模型帧数 T（复制填充）')
    p.add_argument('--crop_size',  type=int, default=None,   help='center_crop 尺寸，不填保持原图')
    p.add_argument('--hidden',     type=int, default=128)
    p.add_argument('--stack_num',  type=int, default=6)
    return p.parse_args()


if __name__ == '__main__':
    infer(parse_args())
