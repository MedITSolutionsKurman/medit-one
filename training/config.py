#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The MedIT Solutions Kurman i Wspolnicy Sp. z o. o. Team.
# https://meditsolutions.pl

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import os
import yaml
import torch
from dacite import from_dict, Config as DaciteConfig


@dataclass
class ModelConfig:
    name: str = "one"  # Model type ('one' supported)
    checkpoint: Optional[str] = None  # Path to model checkpoint or null
    trainer_checkpoint: Optional[str] = None  # Path to trainer checkpoint or null
    tokenizer_name: str = "gpt2"  # Tokenizer to use


@dataclass
class ArchitectureConfig:
    num_layers: int = 24  # Number of transformer layers
    num_heads: int = 24  # Number of attention heads
    hidden_size: int = 1024  # Hidden dimension size
    transformer_dim: int = 4096  # Transformer dimension
    intermediate_size: int = 11008  # Intermediate size for FFN
    max_model_length: int = 2048  # Maximum sequence length supported by model
    torch_dtype: str = "bfloat16"  # Model dtype: "float32", "float16", "bfloat16"
    use_flash_attention: bool = True  # Whether to use flash attention
    use_single_token: bool = False  # Whether to use single token optimization


@dataclass
class DatasetConfig:
    train: str  # Dataset identifier
    config_name: Optional[str] = None  # Dataset name/version if applicable
    cache_dir: Optional[str] = None  # Dataset cache directory
    max_length: int = 512  # Maximum sequence length for training
    train_test_split: float = 0.1  # Fraction for test split
    skip: int = 0  # Skip this many examples
    take: Optional[int] = None  # Take only this many examples (null for all)
    num_proc: int = 4  # Number of preprocessing workers


@dataclass
class TrainingConfig:
    batch_size: int = 32  # Batch size per device
    accumulation_iter: int = 128  # Gradient accumulation steps
    epochs: int = 1  # Number of training epochs
    lr: float = 5.0e-5  # Learning rate
    warmup_steps: int = 500  # Learning rate warmup steps
    weight_decay: float = 0.01  # Weight decay
    save_steps: int = 500  # Save checkpoint every N steps
    neftune_noise_alpha: float = 0.1  # NEFTune noise alpha
    save_total_limit: int = 1  # Maximum number of checkpoints to keep
    betas: List[float] = field(
        default_factory=lambda: [0.9, 0.95]
    )  # Adam optimizer betas
    eps: float = 1.0e-8  # Adam optimizer epsilon


@dataclass
class DirectoriesConfig:
    logging_dir: str = "./logs"  # Directory for logs
    output_dir: str = "./results"  # Directory for model outputs

    def __post_init__(self):
        # Convert relative paths to absolute
        if not os.path.isabs(self.logging_dir):
            self.logging_dir = os.path.abspath(self.logging_dir)
        if not os.path.isabs(self.output_dir):
            self.output_dir = os.path.abspath(self.output_dir)

        # Create directories if they don't exist
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class SystemConfig:
    seed: int = 42  # Random seed
    device: str = "gpu"  # "cpu" or "gpu"

    def __post_init__(self):
        if self.device not in ["cpu", "gpu"]:
            raise ValueError("device must be either 'cpu' or 'gpu'")


@dataclass
class TrainingConfiguration:
    model: ModelConfig
    architecture: ArchitectureConfig
    dataset: DatasetConfig
    training: TrainingConfig
    directories: DirectoriesConfig
    system: SystemConfig

    @staticmethod
    def from_yaml(path: str) -> "TrainingConfiguration":
        """Load configuration from YAML file.

        Args:
            path: Path to the configuration file

        Returns:
            Configuration object

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        try:
            config = from_dict(
                data_class=TrainingConfiguration,
                data=config_dict,
                config=DaciteConfig(
                    type_hooks={
                        Tuple: lambda x: tuple(x),
                    },
                    strict=True,
                ),
            )
            return config
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")

    def get_torch_dtype(self) -> torch.dtype:
        """Get PyTorch dtype from configuration.

        Returns:
            PyTorch dtype object
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.architecture.torch_dtype.lower(), torch.float32)

    def save(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the configuration file
        """
        # Convert to dictionary
        config_dict = {
            "model": self.model.__dict__,
            "architecture": self.architecture.__dict__,
            "dataset": self.dataset.__dict__,
            "training": {
                **self.training.__dict__,
                "betas": list(self.training.betas),  # Convert tuple to list for YAML
            },
            "directories": self.directories.__dict__,
            "system": self.system.__dict__,
        }

        # Save to YAML
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def load_config(path: str) -> TrainingConfiguration:
    """Helper function to load configuration from YAML file.

    Args:
        path: Path to the configuration file

    Returns:
        Configuration object
    """
    return TrainingConfiguration.from_yaml(path)
