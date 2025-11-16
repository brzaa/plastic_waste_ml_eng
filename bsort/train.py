"""
Training module for YOLO11n bottle cap detector.

Optimized for:
- Raspberry Pi 5 deployment (BCM2712 SoC)
- INT8 quantization for 5-10ms inference latency
- Weights & Biases experiment tracking
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import wandb
import yaml
from ultralytics import YOLO

from bsort.config import Config

logger = logging.getLogger(__name__)


class BottleCapTrainer:
    """Trainer for YOLO11n bottle cap detection model."""

    def __init__(self, config: Config):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.project_name = config.get("wandb.project", "bottle-cap-sorting")
        self.data_yaml_path = None

    def _create_data_yaml(self, data_dir: Path) -> Path:
        """
        Create YOLO data.yaml configuration file.

        Args:
            data_dir: Root data directory

        Returns:
            Path to created data.yaml
        """
        train_dir = data_dir / "train" / "images"
        val_dir = data_dir / "val" / "images"

        data_config = {
            "path": str(data_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",  # Optional
            "nc": 3,  # Number of classes
            "names": self.config.get("classes.names", ["light_blue", "dark_blue", "others"]),
        }

        yaml_path = data_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f)

        logger.info(f"Created data.yaml at {yaml_path}")
        return yaml_path

    def _setup_wandb(self, run_name: Optional[str] = None):
        """
        Initialize Weights & Biases tracking.

        Args:
            run_name: Optional custom run name
        """
        wandb_config = {
            "project": self.project_name,
            "config": {
                "model": self.config.get("training.model"),
                "input_size": self.config.get("training.input_size"),
                "epochs": self.config.get("training.epochs"),
                "batch_size": self.config.get("training.batch_size"),
                "optimizer": self.config.get("training.optimizer"),
                "device": self.config.get("training.device"),
            },
        }

        if run_name:
            wandb_config["name"] = run_name

        entity = self.config.get("wandb.entity")
        if entity:
            wandb_config["entity"] = entity

        # Ultralytics integrates with W&B automatically if wandb is imported
        wandb.init(**wandb_config)
        logger.info(f"Initialized W&B project: {self.project_name}")

    def train(
        self,
        data_dir: str,
        output_dir: str = "runs/train",
        resume: bool = False,
        wandb_run_name: Optional[str] = None,
    ) -> Path:
        """
        Train YOLO11n model on bottle cap dataset.

        Args:
            data_dir: Directory containing train/val/test splits
            output_dir: Directory to save training outputs
            resume: Resume from last checkpoint
            wandb_run_name: Custom W&B run name

        Returns:
            Path to best model weights
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Create data.yaml
        self.data_yaml_path = self._create_data_yaml(data_path)

        # Setup W&B
        if self.config.get("wandb.log_model", True):
            self._setup_wandb(wandb_run_name)

        # Load YOLO11n model
        model_name = self.config.get("training.model", "yolo11n.pt")
        logger.info(f"Loading model: {model_name}")
        self.model = YOLO(model_name)

        # Training arguments
        train_args = {
            "data": str(self.data_yaml_path),
            "epochs": self.config.get("training.epochs", 100),
            "imgsz": self.config.get("training.input_size", 640),
            "batch": self.config.get("training.batch_size", 16),
            "device": self.config.get("training.device", "cpu"),
            "workers": self.config.get("training.workers", 4),
            "project": output_dir,
            "name": "bottle_cap_detector",
            "exist_ok": True,
            # Optimizer
            "optimizer": self.config.get("training.optimizer", "AdamW"),
            "lr0": self.config.get("training.lr0", 0.001),
            "lrf": self.config.get("training.lrf", 0.01),
            "momentum": self.config.get("training.momentum", 0.937),
            "weight_decay": self.config.get("training.weight_decay", 0.0005),
            # Augmentation
            "hsv_h": self.config.get("training.augmentation.hsv_h", 0.015),
            "hsv_s": self.config.get("training.augmentation.hsv_s", 0.4),
            "hsv_v": self.config.get("training.augmentation.hsv_v", 0.3),
            "degrees": self.config.get("training.augmentation.degrees", 5.0),
            "translate": self.config.get("training.augmentation.translate", 0.1),
            "scale": self.config.get("training.augmentation.scale", 0.3),
            "flipud": self.config.get("training.augmentation.flipud", 0.0),
            "fliplr": self.config.get("training.augmentation.fliplr", 0.5),
            "mosaic": self.config.get("training.augmentation.mosaic", 1.0),
            # Early stopping
            "patience": self.config.get("training.patience", 20),
            # Misc
            "plots": True,
            "save": True,
            "verbose": True,
        }

        if resume:
            train_args["resume"] = True

        # Train
        logger.info("Starting training...")
        logger.info(f"Training arguments: {train_args}")

        results = self.model.train(**train_args)

        # Get best model path
        best_model_path = Path(output_dir) / "bottle_cap_detector" / "weights" / "best.pt"

        if not best_model_path.exists():
            logger.error("Best model not found after training!")
            raise FileNotFoundError(f"Expected model at {best_model_path}")

        logger.info(f"Training complete! Best model: {best_model_path}")

        # Log model to W&B
        if self.config.get("wandb.log_model", True):
            artifact = wandb.Artifact(
                name="bottle-cap-detector",
                type="model",
                description="YOLO11n bottle cap detector",
            )
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
            logger.info("Logged model artifact to W&B")

        return best_model_path

    def export_onnx(
        self,
        model_path: str,
        output_path: str,
        quantize: bool = True,
        simplify: bool = True,
    ) -> Path:
        """
        Export trained model to ONNX format with INT8 quantization.

        Critical for RPi5 performance: INT8 reduces inference from ~30ms to ~10ms.

        Args:
            model_path: Path to trained .pt model
            output_path: Path to save .onnx model
            quantize: Enable INT8 quantization
            simplify: Simplify ONNX graph

        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting model to ONNX: {model_path}")

        model = YOLO(model_path)

        # Export arguments
        export_args = {
            "format": "onnx",
            "imgsz": self.config.get("training.input_size", 640),
            "opset": self.config.get("export.opset", 12),
            "simplify": simplify,
            "dynamic": self.config.get("export.dynamic", False),  # Static for optimization
        }

        # Export to ONNX
        onnx_path = model.export(**export_args)
        logger.info(f"Exported ONNX model: {onnx_path}")

        # Move to desired location
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if Path(onnx_path) != output_path:
            shutil.move(onnx_path, output_path)
            logger.info(f"Moved ONNX model to: {output_path}")

        # INT8 Quantization
        # Note: Ultralytics YOLO doesn't directly support INT8 export in the same call.
        # For production INT8, we need to use ONNX Runtime quantization tools.
        if quantize:
            logger.info("Applying INT8 quantization...")
            quantized_path = self._quantize_onnx(output_path)
            return quantized_path

        return output_path

    def _quantize_onnx(self, onnx_path: Path) -> Path:
        """
        Apply INT8 quantization to ONNX model using ONNX Runtime.

        This is THE critical optimization for hitting 5-10ms on RPi5.
        - Reduces model size 4x (FP32 -> INT8)
        - Reduces memory bandwidth 4x
        - Enables INT8 SIMD instructions on Cortex-A76

        Args:
            onnx_path: Path to FP32 ONNX model

        Returns:
            Path to quantized INT8 model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            logger.error("onnxruntime not installed. Skipping quantization.")
            return onnx_path

        quantized_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"

        # Dynamic quantization (no calibration data needed)
        # For static quantization, we'd need calibration images
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8,  # or QInt8
            optimize_model=True,
        )

        logger.info(f"INT8 quantized model saved: {quantized_path}")

        # Compare file sizes
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        compression_ratio = original_size / quantized_size

        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Quantized size: {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")

        return quantized_path


def train_model(
    config_path: str = "settings.yaml",
    data_dir: str = "data/processed",
    output_dir: str = "runs/train",
    export_onnx: bool = True,
    quantize: bool = True,
) -> dict:
    """
    Main training pipeline.

    Args:
        config_path: Path to configuration file
        data_dir: Directory containing processed dataset
        output_dir: Directory for training outputs
        export_onnx: Export to ONNX after training
        quantize: Apply INT8 quantization

    Returns:
        Dictionary with paths to trained model and exports
    """
    from bsort.config import load_config

    config = load_config(config_path)
    trainer = BottleCapTrainer(config)

    # Train
    best_model_path = trainer.train(data_dir=data_dir, output_dir=output_dir)

    result = {"pytorch_model": str(best_model_path)}

    # Export to ONNX
    if export_onnx:
        onnx_output = Path(config.get("data.processed_dir", "data/processed")) / "best.onnx"
        onnx_path = trainer.export_onnx(
            model_path=str(best_model_path),
            output_path=str(onnx_output),
            quantize=quantize,
        )
        result["onnx_model"] = str(onnx_path)

    return result
