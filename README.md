# 🖼️ Deep Learning CV — CIFAR-10 Image Classification

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-00C28B?style=flat)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-mewadaatharva14-181717?style=flat&logo=github)](https://github.com/mewadaatharva14)

> Two models trained on CIFAR-10 and compared head-to-head:
> a Custom CNN built from scratch vs ResNet-50 transfer learning with two-stage fine-tuning.

---

## 📌 Overview

This repository implements and compares two approaches to image classification on CIFAR-10
(60,000 RGB images, 10 classes, 32×32):

**Model 1 — Custom CNN from scratch:** Six convolutional layers in three blocks with
BatchNorm, Dropout2d, and MaxPooling. Trained entirely from random initialization —
no pretrained weights.

**Model 2 — ResNet-50 Transfer Learning:** ImageNet-pretrained ResNet-50 with the
classification head replaced. Trained in two stages — feature extraction first
(fc only), then fine-tuning (layer3 + layer4 + fc) with a lower learning rate.

---

## 🗂️ Project Structure

```
deep-learning-cv/
├── src/
│   ├── __init__.py
│   ├── cnn/
│   │   ├── model.py       ← 3 ConvBlocks, BatchNorm, Dropout2d
│   │   └── trainer.py     ← training loop, curves, confusion matrix
│   └── resnet/
│       ├── model.py       ← ResNet-50, freeze/unfreeze, fc replacement
│       └── trainer.py     ← two-stage training, stage boundary plot
│
├── notebooks/
│   ├── 01_cnn_cifar10.ipynb
│   └── 02_resnet50_transfer_learning.ipynb
│
├── configs/
│   ├── cnn_config.yaml
│   └── resnet_config.yaml
│
├── assets/                ← training curves and confusion matrices
├── data/                  ← CIFAR-10 downloads here automatically
├── checkpoints/           ← saved model weights (gitignored)
├── requirements.txt
├── LICENSE
├── README.md
└── train.py
```

---

## 🏗️ Model 1 — Custom CNN

### Architecture

| Layer | Operation | Output Shape |
|---|---|---|
| Input | RGB image | (B, 3, 32, 32) |
| ConvBlock 1 | Conv(3→32) + BN + ReLU ×2 + MaxPool + Dropout2d | (B, 32, 16, 16) |
| ConvBlock 2 | Conv(32→64) + BN + ReLU ×2 + MaxPool + Dropout2d | (B, 64, 8, 8) |
| ConvBlock 3 | Conv(64→128) + BN + ReLU ×2 + MaxPool + Dropout2d | (B, 128, 4, 4) |
| Flatten | — | (B, 2048) |
| Linear + ReLU + Dropout | 2048→512 | (B, 512) |
| Linear | 512→10 | (B, 10) |

**Total parameters:** ~1.2M

### Why BatchNorm?

Normalizes activations within each mini-batch, eliminating internal covariate shift:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Benefits: faster convergence, allows higher learning rates, acts as regularizer.

### Why Dropout2d over Dropout?

Standard Dropout zeros individual pixels — spatially correlated activations in conv
feature maps mean neighboring pixels carry the same information, making standard
Dropout ineffective. Dropout2d drops entire feature maps, forcing the network
to learn redundant representations across channels.

---

## 🏗️ Model 2 — ResNet-50 Transfer Learning

### Residual Connection

ResNet solves vanishing gradients by learning the residual instead of the full mapping:

$$H(x) = \mathcal{F}(x) + x$$

If the optimal mapping is close to identity, learning $\mathcal{F}(x) \approx 0$
is much easier than learning $H(x) \approx x$ directly.

### Two-Stage Training

| Stage | Layers Trained | Learning Rate | Epochs |
|---|---|---|---|
| Feature Extraction | fc only | 1e-3 | 5 |
| Fine-Tuning | layer3 + layer4 + fc | 1e-4 | 10 |

**Stage 1** — backbone frozen, only the new classification head trains.
Quickly adapts the output distribution to CIFAR-10 classes.

**Stage 2** — layer3 + layer4 unfrozen. High-level features adapt to CIFAR-10
with a 10× smaller learning rate to avoid destroying pretrained ImageNet weights.

### Why layer3 + layer4 specifically?

ResNet-50 layers learn features at increasing abstraction levels:

| Layer | Features Learned | Fine-tune? |
|---|---|---|
| layer1 | Edges, colors, gradients | ❌ Keep frozen |
| layer2 | Textures, simple patterns | ❌ Keep frozen |
| layer3 | Object parts, mid-level shapes | ✅ Fine-tune |
| layer4 | Class-specific, high-level | ✅ Fine-tune |
| fc | Classification head | ✅ Always retrain |

---

## 📊 Results

### Model Comparison

| Model | Parameters | Test Accuracy | Epochs | Training Time (CPU) |
|---|---|---|---|---|
| Custom CNN | ~1.2M | — | 15 | — |
| ResNet-50 Transfer | ~25M (23M frozen S1) | — | 15 (5+10) | — |

> Run `python train.py --model cnn` and `python train.py --model resnet` to fill this table.

### Training Curves

*CNN and ResNet training curves — add after training*

### Confusion Matrices

*Per-class accuracy heatmaps — add after training*

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/mewadaatharva14/deep-learning-cv.git
cd deep-learning-cv
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Train**
```bash
# Train Custom CNN
python train.py --model cnn

# Train ResNet-50
python train.py --model resnet

# Custom config
python train.py --model cnn --config configs/cnn_config.yaml
```

CIFAR-10 (~170MB) downloads automatically on first run.

---

## 📓 Notebooks

```bash
jupyter notebook notebooks/
```

| Notebook | Contents |
|---|---|
| `01_cnn_cifar10.ipynb` | Math derivation, model summary, training, curves, confusion matrix |
| `02_resnet50_transfer_learning.ipynb` | ResNet math, two-stage training, stage boundary plot, CNN vs ResNet comparison |

---

## 🔑 Key Implementation Details

**Why resize CIFAR-10 to 224×224 for ResNet-50:**
ResNet-50 was designed for ImageNet (224×224). Feeding 32×32 images directly
collapses the spatial dimensions too early — the first 7×7 conv + MaxPool would
reduce 32×32 to just 7×7 before the residual blocks even start. Resizing to
224×224 lets the full architecture operate as designed.

**Why lower learning rate for fine-tuning:**
Pretrained weights encode millions of ImageNet training steps. A large learning
rate would overwrite these features instantly — catastrophic forgetting. A 10×
smaller lr gently nudges the weights toward CIFAR-10 without destroying
the ImageNet knowledge already encoded.

**Why CosineAnnealingLR:**
Step decay drops the lr abruptly — the model can oscillate around minima before
the drop then converge too fast after. Cosine annealing reduces lr smoothly,
allowing the optimizer to settle into sharp minima gradually.

**Why `filter(lambda p: p.requires_grad, model.parameters())` in optimizer:**
When backbone is frozen, passing all parameters to the optimizer wastes memory
and compute tracking gradients for frozen layers. Filtering ensures only
trainable parameters are passed — the optimizer only updates what needs updating.

---

## 📚 References

| Resource | Link |
|---|---|
| ResNet Paper | [He et al. 2015](https://arxiv.org/abs/1512.03385) |
| Batch Normalization | [Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167) |
| CIFAR-10 Dataset | [Krizhevsky 2009](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Transfer Learning Survey | [Zhuang et al. 2020](https://arxiv.org/abs/1911.02685) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/mewadaatharva14">mewadaatharva14</a>
</p>