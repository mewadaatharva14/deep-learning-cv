"""
ResNet-50 Trainer
==================
Two-stage training:
  Stage 1 — feature extraction (fc only, 5 epochs)
  Stage 2 — fine-tuning (layer3 + layer4 + fc, 10 epochs)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

from src.resnet.model import ResNet50Transfer


class ResNetTrainer:
    def __init__(self, config: dict):
        self.config  = config
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = ResNet50Transfer(
            num_classes = config["model"]["num_classes"],
            dropout     = config["model"]["dropout"],
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.train_losses  = []
        self.val_losses    = []
        self.train_accs    = []
        self.val_accs      = []
        self.stage_boundary = 0     # epoch where fine-tuning starts

        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    def _get_dataloaders(self):
        # ResNet-50 expects 224x224
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=14),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR10(
            root=self.config["data"]["data_dir"],
            train=True, download=True, transform=transform_train,
        )
        val_dataset = datasets.CIFAR10(
            root=self.config["data"]["data_dir"],
            train=False, download=True, transform=transform_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size  = self.config["training"]["batch_size"],
            shuffle     = True,
            num_workers = self.config["data"]["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size  = self.config["training"]["batch_size"],
            shuffle     = False,
            num_workers = self.config["data"]["num_workers"],
        )
        return train_loader, val_loader

    def _build_optimizer(self, lr: float):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr           = lr,
            weight_decay = self.config["training"]["weight_decay"],
        )

    def _run_epoch(self, loader: DataLoader, optimizer, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in tqdm(loader, leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if train:
                    optimizer.zero_grad()

                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / total, correct / total

    def train(self):
        train_loader, val_loader = self._get_dataloaders()
        ckpt = self.config["training"]["checkpoint_dir"]
        os.makedirs(ckpt, exist_ok=True)

        fe_cfg = self.config["model"]["feature_extraction"]
        ft_cfg = self.config["model"]["fine_tuning"]

        best_val_acc = 0.0

        # ── Stage 1: Feature Extraction ──────────────────────────
        print("\n" + "="*55)
        print("  Stage 1 — Feature Extraction (fc only)")
        print(f"  Trainable params: {self.model.count_parameters():,}")
        print("="*55)

        optimizer_fe  = self._build_optimizer(fe_cfg["learning_rate"])
        scheduler_fe  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fe, T_max=fe_cfg["epochs"]
        )

        for epoch in range(1, fe_cfg["epochs"] + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer_fe, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   optimizer_fe, train=False)
            scheduler_fe.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"[S1] Epoch [{epoch:02d}/{fe_cfg['epochs']}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}%  "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{ckpt}/resnet_best.pth")

        self.stage_boundary = fe_cfg["epochs"]

        # ── Stage 2: Fine-Tuning ──────────────────────────────────
        print("\n" + "="*55)
        print(f"  Stage 2 — Fine-Tuning {ft_cfg['unfreeze_layers']}")
        self.model.unfreeze_layers(ft_cfg["unfreeze_layers"])
        print(f"  Trainable params: {self.model.count_parameters():,}")
        print("="*55)

        optimizer_ft = self._build_optimizer(ft_cfg["learning_rate"])
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=ft_cfg["epochs"]
        )

        for epoch in range(1, ft_cfg["epochs"] + 1):
            train_loss, train_acc = self._run_epoch(train_loader, optimizer_ft, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   optimizer_ft, train=False)
            scheduler_ft.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"[S2] Epoch [{epoch:02d}/{ft_cfg['epochs']}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}%  "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{ckpt}/resnet_best.pth")

        print(f"\nBest Val Accuracy: {best_val_acc*100:.2f}%")
        self._save_loss_plot()

    def evaluate(self, loader: DataLoader = None):
        if loader is None:
            _, loader = self._get_dataloaders()

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(self.device)
                preds  = self.model(imgs).argmax(dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Test Accuracy: {acc*100:.2f}%")

        cm = confusion_matrix(all_labels, all_preds)
        self._save_confusion_matrix(cm)
        return acc, cm

    def _save_loss_plot(self):
        os.makedirs("assets", exist_ok=True)
        epochs     = list(range(1, len(self.train_losses) + 1))
        boundary   = self.stage_boundary

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for ax, train_data, val_data, title in zip(
            [ax1, ax2],
            [self.train_losses, [a * 100 for a in self.train_accs]],
            [self.val_losses,   [a * 100 for a in self.val_accs]],
            ["Loss", "Accuracy (%)"],
        ):
            ax.plot(epochs, train_data, label="Train")
            ax.plot(epochs, val_data,   label="Val")
            ax.axvline(x=boundary, color="red", linestyle="--",
                       alpha=0.7, label="Fine-tuning starts")
            ax.set_title(f"ResNet-50 — {title}")
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle("ResNet-50 Transfer Learning — CIFAR-10")
        plt.tight_layout()
        plt.savefig("assets/resnet_training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → assets/resnet_training_curves.png")

    def _save_confusion_matrix(self, cm: np.ndarray):
        os.makedirs("assets", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=self.classes, yticklabels=self.classes, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("ResNet-50 Transfer Learning — Confusion Matrix")
        plt.tight_layout()
        plt.savefig("assets/resnet_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → assets/resnet_confusion_matrix.png")

    def get_history(self) -> dict:
        return {
            "train_losses":   self.train_losses,
            "val_losses":     self.val_losses,
            "train_accs":     self.train_accs,
            "val_accs":       self.val_accs,
            "stage_boundary": self.stage_boundary,
        }