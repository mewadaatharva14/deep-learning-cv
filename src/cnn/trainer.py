"""
CNN Trainer
============
Training loop, evaluation, confusion matrix, loss curves.
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

from src.cnn.model import CustomCNN


class CNNTrainer:
    def __init__(self, config: dict):
        self.config   = config
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = CustomCNN(
            num_classes = config["model"]["num_classes"],
            dropout     = config["model"]["dropout"],
        ).to(self.device)

        self.criterion  = nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.Adam(
            self.model.parameters(),
            lr           = config["training"]["learning_rate"],
            weight_decay = config["training"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max = config["training"]["epochs"],
        )

        self.train_losses  = []
        self.val_losses    = []
        self.train_accs    = []
        self.val_accs      = []

        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]

    def _get_dataloaders(self):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        transform_val = transforms.Compose([
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

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in tqdm(loader, leave=False):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += imgs.size(0)

        return total_loss / total, correct / total

    def train(self):
        train_loader, val_loader = self._get_dataloaders()
        epochs = self.config["training"]["epochs"]
        ckpt   = self.config["training"]["checkpoint_dir"]
        os.makedirs(ckpt, exist_ok=True)

        print(f"\nDevice     : {self.device}")
        print(f"Parameters : {self.model.count_parameters():,}")
        print(f"Epochs     : {epochs}\n")

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(
                f"Epoch [{epoch:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}%  "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f"{ckpt}/cnn_best.pth")

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses, label="Train")
        ax1.plot(self.val_losses,   label="Val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot([a * 100 for a in self.train_accs], label="Train")
        ax2.plot([a * 100 for a in self.val_accs],   label="Val")
        ax2.set_title("Accuracy (%)")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle("Custom CNN — CIFAR-10 Training Curves")
        plt.tight_layout()
        plt.savefig("assets/cnn_training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → assets/cnn_training_curves.png")

    def _save_confusion_matrix(self, cm: np.ndarray):
        os.makedirs("assets", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.classes, yticklabels=self.classes, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Custom CNN — Confusion Matrix")
        plt.tight_layout()
        plt.savefig("assets/cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved → assets/cnn_confusion_matrix.png")

    def get_history(self) -> dict:
        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "train_accs":   self.train_accs,
            "val_accs":     self.val_accs,
        }