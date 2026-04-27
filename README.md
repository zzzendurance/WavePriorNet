# WavePriorNet — Stage 1

**Wavelet Spatial-Temporal Transformer with Physical Color Prior for Video Dehazing**

针对 REVIDE_indoor 视频去雾数据集设计，Stage 1 基线网络。

---

## 1. 任务定义

| 输入 | 张量形状 | 说明 |
|------|----------|------|
| `hazy` | `[B, T, 3, H, W]` | 有雾视频帧序列，值域 `[0,1]` |
| `color_ref` | `[B, T, 3, H, W]` | 色彩偏移的 GT 参考帧（由 `color_shift.py` 生成），值域 `[0,1]` |

| 输出（dict） | 说明 |
|-------------|------|
| `out` | `[B×T, 3, H, W]` 最终去雾预测，值域 `[0,1]` |
| `aux_j` | `list[Tensor]` 辅助场景预测（仅训练期间） |
| `aux_i` | `list[Tensor]` 物理重建输出（仅训练期间） |
| `img_in` | `[B×T, 3, H, W]` 输入有雾帧（用于 `aux_i` loss 目标） |

预测方式为**残差预测**（MAP-Net 风格）：
```
out = hazy_in + decoder_delta      # decoder 最后层初始化为 0，训练初期 out ≈ hazy_in
```

---

## 2. 引用的四个仓库及对应模块

### 2.1 SVINet（本地 `model_SVINet_10.py`）

**引用模块**：`Encoder`、`WaveletAttention`、`FFN`、`WaveletTransformerBlock` 主体

| 类/函数 | 对应原始代码 | 说明 |
|---------|-------------|------|
| `Encoder` | `model_SVINet_10.py:78-101` | 两级 Conv+ResBlock，stride-2 下采样 |
| `WaveletAttention` | `model_SVINet_10.py:128-242` | 多尺度 patch attention，Q/K/V 来自 LL 子带，权重同时作用于 LH/HL/HH/L3 五路；保留原版 `self.w = nn.Parameter(torch.ones(2))` |
| `FFN` | `model_SVINet_10.py:219-230` | 膨胀卷积 FFN |

**关键细节**：`WaveletAttention` 完全按原版实现，包括 `to_seq`/`from_seq` 的 reshape 逻辑和每个子带独立的 `output_linear_*` 投影层。

---

### 2.2 DehazeFormer（`DehazeFormer/models/dehazeformer.py`）

**引用模块**：`RLN`、`SKFusion`、`TransformerBlock` 应用模式

| 类/函数 | 对应原始代码 | 说明 |
|---------|-------------|------|
| `RLN` | `dehazeformer.py:10-41` **完全照搬** | Revised LayerNorm，`detach_grad=False`（默认），梯度可流过 `meta1/meta2`，保留雾气全局统计量 |
| `SKFusion` | `dehazeformer.py:349-376` **完全照搬** | Selective Kernel Fusion，用于 Decoder 跳跃连接 |
| RLN 应用模式 | `dehazeformer.py:261-273` | `identity=x → norm → attn → x*rescale+rebias → identity+x → FFN` |

**关键修复**：旧版将 `rescale/rebias` 硬编码为 `detach()`，切断了梯度；现在改为 `detach_grad=False`，与原版一致。`rescale/rebias` 在 attention 之后、残差相加之前乘回，完全按 DehazeFormer `TransformerBlock` 的模式。

---

### 2.3 MAP-Net（`MAP-Net/mmedit/models/`）

**引用模块**：`PhysicalPriorHead`、`GMRAFusion`、残差预测方式、`aux_j/aux_i` 多阶段监督

| 类/函数 | 对应原始代码 | 说明 |
|---------|-------------|------|
| `PhysicalPriorHead` | `mapnet_net.py:18-94` | 估计透射率 `t`（softmax 分类→加权求和）和大气光 `A`（全局池化→Sigmoid） |
| `GMRAFusion` | `mapnet_net.py:197-221` | `attn = attn_scene + β * attn_prior`，per-pixel 注意力；`guide_conv` 对应 L117，`aggre_conv` 对应 L129 |
| 残差预测 | `mapnet_net.py:381` | `out = img_01 + decoder_out` |
| `conv_last` 初始化为 0 | `map_modules.py:95-96` | 训练初期 decoder 输出为 0，等价于直接输出输入图，避免训练初期大幅震荡 |
| `aux_j / aux_i` 物理重建 | `mapnet_net.py:388-398` | `aux_i = aux_j * t + A * (1-t)`（大气散射模型） |
| 物理先验 loss 权重 | `map.py:93-98` | `λ_phy=0.2`，多阶段权重 `0.2/2^s` |

---

### 2.4 color-alignment-in-diffusion（`color-alignment-in-diffusion/train.py`）

**引用模块**：Chamfer Color Loss

| 函数 | 对应原始代码 | 说明 |
|------|-------------|------|
| `chamfer_color_loss` | `train.py:cross_forward` 中的 chamfer_distance 投影 | 将 pred/GT 展平为 RGB 点云，计算单向 Chamfer Distance；为避免 OOM 随机采样 512 像素（原版全像素） |

---

## 3. 完整架构

```
hazy  [B,T,3,H,W]                color_ref [B,T,3,H,W]
        │                                    │
  HazyEncoder (SVINet)             PriorBranch
  Conv+ResBlock×2, stride-2        Encoder + ResBlock×2
        │                          + PhysicalPriorHead (MAP-Net)
  feat_scene [BT,128,H/2,W/2]     feat_prior [BT,128,H/2,W/2]
                                   t [BT,1,H/2,W/2]  A [BT,1,1,1]
        │                                    │
        └──────── GMRAFusion (MAP-Net) ───────┘
                  guide_conv([prior, scene])
                  attn_scene + β·attn_prior
                  aggre_conv([aggregated, query])
                            │
                    feat [BT,128,H/2,W/2]
                            │
          ┌─────────────────────────────────┐
          │  WaveletTransformerBlock × 6    │
          │  (SVINet backbone)              │
          │  RLN(detach_grad=False)         │  ← DehazeFormer
          │  → DWT(Haar) → WaveletAttn     │
          │  → IDWT → x*rescale+rebias     │  ← DehazeFormer pattern
          │  → identity+x → FFN            │
          └─────────────────────────────────┘
                            │
                 ┌──────────┴──────────┐
            [training]           Decoder (MAP-Net style)
            aux_head              up(feat) + SKFusion(skip) ← DehazeFormer
            ↓                     body → conv_hr → conv_last(init=0)
           aux_j = hazy + delta_aux               │
           aux_i = aux_j*t + A*(1-t)         delta [BT,3,H,W]
                                                   │
                                        out = hazy_in + delta  (残差预测)
                                        out.clamp(0,1)
```

---

## 4. 训练损失

完全按照 MAP-Net `map.py:85-131` 的 loss 设计：

```
L_total = L_main
        + L_phy_j   (物理先验 scene loss)
        + L_phy_i   (物理先验 reconstruction loss)
        + λ_perc  × L_perceptual
        + λ_chamfer × L_chamfer
```

| 损失项 | 公式 | 权重 | 来源 |
|--------|------|------|------|
| `L_main` | `L1(pred, GT)` | 1.0 | MAP-Net L88 |
| `L_phy_j` | `Σ_s 0.2/2^s × L1(aux_j[s], GT)` | 0.2 | MAP-Net L93-98 |
| `L_phy_i` | `Σ_s 0.2/2^s × L1(aux_i[s], hazy_in)` | 0.2 | MAP-Net L93-98 |
| `L_perceptual` | `L1(VGG16_relu2_2(pred), VGG16_relu2_2(GT))` | λ=0.1 | — |
| `L_chamfer` | 单向 Chamfer Distance（采样512像素） | λ=0.05 | color-alignment |

---

## 5. 学习率策略

**Warmup + CosineAnnealing**（前 10 epoch 线性 warmup，之后 Cosine 衰减）：

```
epoch  0-9  :  lr 线性从 0 升至 1e-4      ← 保护残差初始化
epoch 10-99 :  lr Cosine 从 1e-4 降至 1e-6
```

---

## 6. 测试指标

| 指标 | 含义 | 越高越好 |
|------|------|---------|
| **PSNR** (dB) | `10·log10(1/MSE)`，像素级重建精度 | ✓ |
| **SSIM** | 亮度/对比度/结构综合相似性，值域 [0,1] | ✓ |

REVIDE_indoor 参考水平：CG-Net (CVPR 2021) ~20 dB，MAP-Net (CVPR 2023) ~24.16 dB。

---

## 7. 数据集准备

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

## 8. 快速开始

### 安装依赖

```bash
pip install torch torchvision Pillow tensorboard numpy
```

### 训练（A6000 48GB，fp32，默认配置 ~28GB）

```bash
cd WavePriorNet
python train.py \
  --data_root /path/to/REVIDE_indoor \
  --batch_size 8 \
  --crop_size 512 \
  --epochs 100 \
  --warmup_epochs 10
```

fp16 AMP（最大吞吐量，~14GB，可将 batch_size 提高到 16）：

```bash
python train.py \
  --data_root /path/to/REVIDE_indoor \
  --batch_size 16 \
  --crop_size 512 \
  --amp
```

TensorBoard 监控：

```bash
tensorboard --logdir checkpoints/tb_logs
```

### 测试（计算 PSNR / SSIM，保存预测图像）

```bash
python test.py \
  --data_root /path/to/REVIDE_indoor \
  --ckpt checkpoints/best.pth \
  --save_images \
  --save_dir results/
```

### 单视频推理（无 GT）

```bash
python test.py \
  --hazy_dir /path/to/hazy_frames/ \
  --color_dir /path/to/color_ref_frames/ \
  --ckpt checkpoints/best.pth \
  --save_dir results/
```

---

## 9. 超参数一览（A6000 48GB 默认）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | B×T=40 帧/step |
| `crop_size` | 512 | 随机裁剪（原图 2708×1800） |
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

---

## 10. 工程结构

```
WavePriorNet/
├── core/
│   ├── dataset.py          # REVIDE_indoor 三路数据加载（hazy/gt/gt_color_shift）
│   └── dwt.py              # 纯 PyTorch Haar DWT/IDWT（无外部依赖）
├── model_wavepriornet.py   # 完整网络定义
├── train.py                # 训练脚本（含物理先验 loss + warmup）
├── test.py                 # 测试/评估脚本（PSNR/SSIM）
└── requirements.txt
```

---

## 11. 引用

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
