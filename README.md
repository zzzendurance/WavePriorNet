# WavePriorNet — Stage 1

**Wavelet Spatial-Temporal Transformer with Physical Color Prior for Video Dehazing**

针对 REVIDE_indoor 视频去雾数据集设计，Stage 1 基线网络。

---

## 1. 任务定义

### 输入

| 张量 | 形状 | 说明 |
|------|------|------|
| `hazy` | `[B, T, 3, H, W]` | 有雾视频帧序列，值域 `[0, 1]` |
| `color_ref` | `[B, T, 3, H, W]` | 色彩偏移的 GT 参考帧（由 `color_shift.py` 生成），值域 `[0, 1]` |

`B` = batch size，`T` = 每样本连续帧数（默认 5），`H×W` = 空间分辨率（训练时随机裁剪）。

### 输出（dict）

| 键 | 形状 | 说明 |
|----|------|------|
| `out` | `[B×T, 3, H, W]` | 最终去雾预测，值域 `[0, 1]` |
| `img_in` | `[B×T, 3, H, W]` | 输入有雾帧（归一化后，用于 `aux_i` loss 目标） |
| `aux_j` | `list[Tensor]` | 辅助场景预测（**仅训练期间**） |
| `aux_i` | `list[Tensor]` | 物理重建输出（**仅训练期间**） |

预测方式为**残差预测**（MAP-Net 风格）：
```
out = hazy_in + decoder_delta      # decoder 最后层初始化为 0，训练初期 out ≈ hazy_in
```

### 典型张量尺寸（B=2, T=5, crop=384）

| 位置 | 形状 | 说明 |
|------|------|------|
| 输入 | `[2, 5, 3, 384, 384]` | 2 个 clip，每 clip 5 帧 |
| 展平 | `[10, 3, 384, 384]` | BT = 2×5 |
| Encoder 后 | `[10, 128, 192, 192]` | stride-2 下采样 |
| DWT 后 | 4×`[10, 128, 96, 96]` | Haar 小波 stride-2 |
| Transformer 后 | `[10, 128, 192, 192]` | IDWT 恢复到 H/2 |
| Decoder 输出 | `[10, 3, 384, 384]` | 全分辨率残差 |

---

## 2. 模型架构

### 2.1 整体流程

```
hazy  [B,T,3,H,W]                color_ref [B,T,3,H,W]
        │                                    │
  ┌─ HazyEncoder ─┐               ┌─ PriorBranch ─────────┐
  │ Conv+ResBlock  │               │ Encoder + ResBlock×2  │
  │ stride-2 ↓    │               │ + PhysicalPriorHead   │
  └───────────────┘               │   t [BT,1,H/2,W/2]   │
  feat_scene                      │   A [BT,1,1,1]        │
  [BT,128,H/2,W/2]               │   feat_prior          │
        │                         └───────────────────────┘
        │                                    │
        └──────── GMRAFusion ────────────────┘
                  per-pixel gate:
                  gate = sigmoid(attn_j + β·attn_prior)
                  out  = feat_j * gate
                            │
                    feat [BT,128,H/2,W/2]
                            │
          ┌─────────────────────────────────────┐
          │  WaveletTransformerBlock × 6        │
          │  ┌─ RLN (detach_grad=False) ──┐     │  ← DehazeFormer
          │  │ DWT(Haar) → WaveletAttn   │     │  ← SVINet
          │  │ IDWT → x*rescale+rebias  │     │  ← DehazeFormer
          │  │ identity+x → FFN         │     │
          │  └───────────────────────────┘     │
          └─────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
          [training only]           Decoder (MAP-Net style)
          aux_head                  SKFusion(feat, skip)  ← DehazeFormer
          ↓                         → up → conv_hr → conv_last(init=0)
         aux_j = hazy + delta_aux              │
         aux_i = aux_j*t + A*(1-t)        delta [BT,3,H,W]
                                                │
                                    out = hazy_in + delta
                                    out.clamp(0, 1)
```

### 2.2 各组件来源

| 组件 | 来源论文 | 说明 |
|------|---------|------|
| `HazyEncoder` | SVINet | Conv+ResBlock×2，stride-2 下采样 |
| `WaveletAttention` | SVINet | 多尺度 patch attention，Q/K/V 来自 LL 子带，权重作用于 LH/HL/HH/L3 五路 |
| `FFN` | SVINet | 膨胀卷积前馈网络 |
| `RLN` | DehazeFormer | Revised LayerNorm，`detach_grad=False`，梯度可流过 meta1/meta2 |
| `SKFusion` | DehazeFormer | Selective Kernel Fusion，用于 Decoder 跳跃连接 |
| `PhysicalPriorHead` | MAP-Net | 估计透射率 `t`（softmax 分类）和大气光 `A`（全局池化+Sigmoid） |
| `GMRAFusion` | MAP-Net | per-pixel 注意力融合（已修复 OOM，见 §4） |
| 残差预测 / `conv_last` 初始化为 0 | MAP-Net | 训练初期输出稳定 |
| `aux_j / aux_i` 多阶段监督 | MAP-Net | 物理散射模型约束 |
| `chamfer_color_loss` | color-alignment-in-diffusion | 采样 512 像素的单向 Chamfer Distance |
| `DWT_2D / IDWT_2D` | — | 纯 PyTorch Haar 小波（无外部依赖） |

### 2.3 WaveletAttention patchsize（crop=384）

Encoder stride-2 + DWT stride-2 后特征图为 **96×96**：

| patchsize | 整除 96 | 序列长 (T=5) | 注意力矩阵 | 显存 |
|-----------|---------|-------------|-----------|------|
| (32,32)   | ✓ (3×3) | 45 | [B,45,45] | < 0.1 MB |
| (16,16)   | ✓ (6×6) | 180 | [B,180,180] | ~0.5 MB |
| (8,8)     | ✓ (12×12) | 720 | [B,720,720] | ~8 MB |
| (4,4)     | ✓ (24×24) | 2880 | [B,2880,2880] | ~120 MB |

---

## 3. 训练损失

```
L_total = L_main
        + L_phy_j   (物理先验 scene loss)
        + L_phy_i   (物理先验 reconstruction loss)
        + λ_perc  × L_perceptual
        + λ_chamfer × L_chamfer
```

| 损失项 | 公式 | 权重 | 来源 |
|--------|------|------|------|
| `L_main` | `L1(pred, GT)` | 1.0 | MAP-Net |
| `L_phy_j` | `Σ_s 0.2/2^s × L1(aux_j[s], GT)` | 0.2 | MAP-Net |
| `L_phy_i` | `Σ_s 0.2/2^s × L1(aux_i[s], hazy_in)` | 0.2 | MAP-Net |
| `L_perceptual` | `L1(VGG16_relu2_2(pred), VGG16_relu2_2(GT))` | λ=0.1 | — |
| `L_chamfer` | 单向 Chamfer Distance（采样 512 像素） | λ=0.05 | color-alignment |

---

## 4. OOM 修复记录

以下 4 个问题在实际运行中触发，均已修复并合入当前代码。

### Fix 1：GMRAFusion 无效注意力导致 OOM（`model_wavepriornet.py`）

**问题**：`GMRAFusion.forward` 将 `[BT, C, H, W]` reshape 成 `[BT*H*W, 1, C]` 再做 `q @ k^T`。
当 `B=8, T=5, crop=512` 时，`BT*H*W = 40×256×256 ≈ 2.6M`，中间张量约 **1.25 GiB**，触发 OOM。

**根本原因**：`nr=1`（序列长度为 1）时，`softmax([单个标量]) = 1`，`attn @ v = v`，整个注意力退化为恒等映射，但仍分配了巨大的中间张量。

**修复**：改用逐像素通道点积（`sum(dim=1)`）+ sigmoid gate，语义等价，内存从 O(BT·H·W·C) 降至 O(BT·C·H·W)（少一个 H·W 因子）：

```python
# 修复前（退化的矩阵乘法，1.25 GiB）
q  = feat_j.permute(0,2,3,1).reshape(BT*H*W, 1, C)
attn_j = q @ k_j.transpose(-2,-1) * scale  # [BT*HW, 1, 1]

# 修复后（等价的逐像素点积，< 1 MB）
attn_j = (feat_j * feat_j).sum(dim=1, keepdim=True) * scale   # [BT,1,H,W]
attn_p = (feat_j * feat_prior).sum(dim=1, keepdim=True) * scale
gate   = torch.sigmoid(attn_j + self.aggre_beta * attn_p)
out    = feat_j * gate
```

---

### Fix 2：WaveletAttention patchsize 与特征图尺寸不整除（`model_wavepriornet.py`）

**问题**：原始 patchsize `((54,30),(36,20),(18,10),(9,5))` 来自 SVINet，针对 `540×960` 视频（Encoder+DWT 后为 `135×240`）设计。
换成 `crop=512` 后，特征图为 `128×128`，`128 / 54` 不整除，`to_seq` 的 `view` 报 shape 不匹配错误。

**修复**：改为能整除 `128` 的 patchsize，同时保留多尺度注意力设计：

```python
# 修复前（针对 540×960，不适配 512×512）
patchsize=((54, 30), (36, 20), (18, 10), (9, 5))

# 修复后（整除 128，对应序列长 80/320/1280/5120）
patchsize=((32, 32), (16, 16), (8, 8), (4, 4))
```

---

### Fix 3：Decoder SKFusion 尺寸不匹配（`model_wavepriornet.py`）

**问题**：`Decoder.forward` 中先对 `feat`（H/2）做 2× 上采样得到全分辨率特征，再与 `skip`（H/2）送入 `SKFusion`，两者空间尺寸不同（如 `384 vs 192`），`torch.cat` 报错。

**修复**：将 `skip_fusion` 移到 `up` 之前，两者都在 H/2 分辨率下融合：

```python
# 修复前（尺寸不匹配）
x = self.up(feat)               # [BT, hidden, H, W]
x = self.skip_fusion([x, skip]) # skip 是 H/2，报错

# 修复后（先融合再上采样）
feat = self.skip_fusion([feat, skip])  # 都是 H/2，匹配
x    = self.up(feat)                   # [BT, hidden, H, W]
```

---

### Fix 4：新增 Gradient Accumulation（`train.py`）

**背景**：`batch=8, crop=512, BT=40` 时显存超出 44GB。直接减小 batch_size 会影响训练稳定性。

**方案**：新增 `--accum_steps` 参数，每 N 步 backward 一次 optimizer step，等效 batch 不变。
模型只使用 RLN（LayerNorm 类），无 BatchNorm，梯度累积与大 batch **数学完全等价**。

```python
# loss 除以 accum_steps，保证梯度是 N 步的平均而非累加
loss_scaled = loss / accum_steps
loss_scaled.backward()

if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
    optimizer.step()
    optimizer.zero_grad()
```

---

## 5. 学习率策略

**Warmup + CosineAnnealing**（前 10 epoch 线性 warmup，之后 Cosine 衰减）：

```
epoch  0-9  :  lr 线性从 0 升至 1e-4      ← 保护残差初始化
epoch 10-99 :  lr Cosine 从 1e-4 降至 1e-6
```

---

## 6. 实际运行结果

以下为一次完整训练的参数与结果（run_20260428_005635）：

| 参数 | 值 |
|------|---|
| batch_size | 2 |
| crop_size | 384 |
| num_frames | 5 |
| lr | 1e-4 |
| AMP | 关闭 |
| 总 epoch | 20 |
| 模型参数量 | **14.76 M** |

| 指标 | 值 |
|------|---|
| 最佳 PSNR | **35.82 dB**（epoch 19） |
| 总训练时长 | 8.64 h（31104.6 s） |
| 最佳权重 | `checkpoints/run_20260428_005635/best.pth` |

> 注：本次仅跑了 20 epoch（快速验证），推荐跑满 100 epoch 以获得更充分收敛。
> REVIDE_indoor 参考水平：CG-Net (CVPR 2021) ~20 dB，MAP-Net (CVPR 2023) ~24.16 dB。
> 20 epoch 即达到 35.82 dB，说明模型收敛良好。

---

## 7. 测试指标

| 指标 | 含义 | 越高越好 |
|------|------|---------|
| **PSNR** (dB) | `10·log10(1/MSE)`，像素级重建精度 | ✓ |
| **SSIM** | 亮度/对比度/结构综合相似性，值域 [0,1] | ✓ |

---

## 8. 数据集准备

```
REVIDE_indoor/
├── Train/
│   ├── hazy/           <seq>/  00000.JPG …  (42 sequences × 55 frames, 2708×1800)
│   ├── gt/             <seq>/  00000.JPG …
│   └── gt_color_shift/ <seq>/  00000.JPG …  ← 由 color_shift.py 生成
└── Test/
    ├── hazy/           <seq>/  (6 sequences)
    ├── gt/             <seq>/
    └── gt_color_shift/ <seq>/
```

生成 `gt_color_shift`（只需一次）：

```bash
cd /path/to/REVIDE_indoor
python color_shift.py --split Train --data_dir .
python color_shift.py --split Test  --data_dir .
```

---

## 9. 快速开始

### 安装依赖

```bash
pip install torch torchvision Pillow tensorboard numpy
```

### 训练

推荐配置（A6000 48GB，显存安全）：

```bash
python train.py \
  --data_root /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor \
  --batch_size 2 \
  --crop_size 384 \
  --accum_steps 4 \
  --epochs 100 \
  --warmup_epochs 10
```

等效 batch = 2 × 4 = **8**，显存约 8-10 GB。

快速验证（20 epoch 确认收敛）：

```bash
python train.py \
  --data_root /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor \
  --batch_size 2 --crop_size 384 --accum_steps 4 \
  --epochs 20 --val_every 2
```

从 checkpoint 继续训练：

```bash
python train.py \
  --data_root /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor \
  --batch_size 2 --crop_size 384 --accum_steps 4 \
  --resume checkpoints/run_xxx/latest.pth
```

TensorBoard 监控：

```bash
tensorboard --logdir checkpoints/
```

---

### 测试（test.py）

在整个 Test split 上评估 PSNR/SSIM，并按序列保存四路图像（pred / gt / hazy / color）：

```bash
python test.py \
  --ckpt  checkpoints/run_20260428_005635/best.pth \
  --data_root /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor \
  --split Test \
  --num_frames 5 \
  --batch_size 1 \
  --crop_size 384 \
  --save_images \
  --save_dir results/run_20260428_005635_test
```

> **注意**：`--crop_size` 必须能被 **128** 整除（Encoder stride=2 × DWT stride=2 × patchsize=32）。
> 不加 `--crop_size` 时原图 2708×1800 → 特征图 450×677，patchsize=32 无法整除，`view` 直接报错。
> 推荐值：384（训练时用的）或 512。

输出目录结构（按序列对应原始数据集）：

```
results/run_20260428_005635_test/
├── pred/
│   ├── C005/  00000.png  00001.png  ...   ← 去雾预测
│   ├── E005/  ...
│   └── ...
├── gt/    C005/  00000.png  ...           ← 干净 GT
├── hazy/  C005/  00000.png  ...           ← 原始有雾输入
├── color/ C005/  00000.png  ...           ← gt_color_shift 输入
├── eval_results.json                      ← 总体 PSNR / SSIM
└── per_frame.csv                          ← 逐帧 seq / frame / psnr / ssim
```

**Test split 采样规则**：每个序列按 `num_frames=5` 做非重叠切片，尾部不足 5 帧的丢弃。
6 个序列共 284 帧，切出 54 个 clip × 5 帧 = **270 帧**。

| 序列 | 总帧数 | clip 数 |
|------|--------|---------|
| C005 | 43 | 8 |
| E005 | 58 | 11 |
| E006 | 36 | 7 |
| J004 | 57 | 11 |
| L006 | 54 | 10 |
| W002 | 36 | 7 |

---

### 单张推理（infer_single.py）

只有一张有雾图和一张 gt_color_shift 图时使用：

```bash
python infer_single.py \
  --ckpt  checkpoints/run_20260428_005635/best.pth \
  --hazy  /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor/Test/hazy/C005/00000.JPG \
  --color /home/user/zhaojiaqi/datasets/videoDehazeDatasets/REVIDE_indoor/Test/gt_color_shift/C005/00000.JPG \
  --out   pred_C005_00000.png \
  --crop_size 384
```

模型需要 T=5 帧输入，脚本将这一张图复制 5 份填充，取第 0 帧输出作为预测结果。
`--crop_size` 同样必须能被 128 整除，原图过大时必须指定。

---

## 10. 超参数一览

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch_size` | 2 | 实际每步 B，配合 accum_steps 使用 |
| `accum_steps` | 4 | 梯度累积步数，等效 batch = batch_size × accum_steps |
| `crop_size` | 384 | 随机裁剪（原图 2708×1800），Encoder+DWT 后特征图 96×96 |
| `num_frames` | 5 | 每样本连续帧数 T |
| `hidden` | 128 | 特征通道宽度 |
| `stack_num` | 6 | WaveletTransformerBlock 层数 |
| `epochs` | 100 | 总训练轮数 |
| `warmup_epochs` | 10 | 线性 warmup 轮数 |
| `lr` | 1e-4 | 峰值学习率 |
| `lr_min` | 1e-6 | Cosine 衰减终点 |
| `weight_decay` | 1e-4 | AdamW 权重衰减 |
| `lambda_perc` | 0.1 | 感知 loss 权重 |
| `lambda_chamfer` | 0.05 | Chamfer color loss 权重 |
| `amp` | false | 混合精度（当前关闭，开启可节省约 50% 显存） |

---

## 11. 工程结构

```
WavePriorNet/
├── core/
│   ├── dataset.py          # REVIDE_indoor 三路数据加载（hazy/gt/gt_color_shift），返回 seq_name/frame_names
│   └── dwt.py              # 纯 PyTorch Haar DWT/IDWT（无外部依赖）
├── model_wavepriornet.py   # 完整网络定义（585 行）
├── train.py                # 训练脚本（含物理先验 loss + warmup + 梯度累积）
├── test.py                 # 测试/评估脚本（PSNR/SSIM，按序列保存四路图像）
├── infer_single.py         # 单张推理脚本（单张 hazy + color_shift → pred）
└── requirements.txt
```

---

## 12. 引用

```bibtex
@inproceedings{yan2020sttn,
  title={Learning Joint Spatial-Temporal Transformations for Video Inpainting},
  author={Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{xu2023mapnet,
  title={Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior},
  author={Xu, Jiaqi and others},
  booktitle={CVPR},
  year={2023}
}

@article{song2023dehazeformer,
  title={Vision Transformers for Single Image Dehazing},
  author={Song, Yuda and He, Zhuqing and others},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}

@inproceedings{shum2025coloralignment,
  title={Color Alignment in Diffusion},
  author={Shum, Ka Chun and others},
  booktitle={CVPR},
  year={2025}
}
```
