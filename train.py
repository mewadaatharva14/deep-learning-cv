"""
Training Entry Point
=====================
Train CustomCNN or ResNet50 on CIFAR-10.

Usage:
    python train.py --model cnn
    python train.py --model resnet
    python train.py --model cnn    --config configs/cnn_config.yaml
    python train.py --model resnet --config configs/resnet_config.yaml
"""

import argparse
import os
import yaml

DEFAULT_CONFIGS = {
    "cnn":    "configs/cnn_config.yaml",
    "resnet": "configs/resnet_config.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN or ResNet-50 on CIFAR-10.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "resnet"],
        help="Model to train: cnn or resnet",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: configs/<model>_config.yaml)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\nConfig file not found: {config_path}\n"
            f"Run from the repo root directory."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args        = parse_args()
    config_path = args.config or DEFAULT_CONFIGS[args.model]
    config      = load_config(config_path)

    print(f"\n{'='*55}")
    print(f"  Model  : {args.model.upper()}")
    print(f"  Config : {config_path}")
    print(f"{'='*55}")

    if args.model == "cnn":
        from src.cnn.trainer import CNNTrainer
        trainer = CNNTrainer(config)

    elif args.model == "resnet":
        from src.resnet.trainer import ResNetTrainer
        trainer = ResNetTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()