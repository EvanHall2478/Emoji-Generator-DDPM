"""Configuration definitions for this assignment.

Modify this file to set the allowed hyperparameters and options you believe are
appropriate for your experiments.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import utils


@dataclasses.dataclass
class Config:
    """Configuration class for this assignment.

    We will use Python dataclasses to define and manage all configuration parameters.

    Usage:
        >>> cfg = Configs()
        >>> print(cfg.epochs)
        100
    """
    # yapf: disable
    # ---------------------------------------- [you may modify this]
    # DDPM hyperparameters
    timesteps: int = 500  # Number of diffusion timesteps
    beta_start: float = 0.0001  # Starting noise variance
    beta_end: float = 0.02  # Ending noise variance

    # Training hyperparameters
    batch_size: int = 32  # Batch size
    epochs: int = 300  # Number of training epochs
    lr: float = 1e-4  # Learning rate
    save_interval: int = 25  # Save checkpoint every N epochs

    # Model architecture
    time_emb_dim: Optional[int] = 128  # Time embedding dimension
    text_emb_dim: Optional[int] = 384  # Text embedding dimension. Set to 384 for text conditioning, None for unconditional

    # UNet architecture
    block_out_ch: Tuple[int, ...] = (64, 128, 256)  # Number of output channels at each resolution level
    num_layers_per_block: int = 2  # Number of convolutional layers in each UNet block
    # ----------------------------------------
    # yapf: enable

    # Experiment settings
    run_id: str = f"{int(time.time() * 1000)}"
    quick_test: bool = False
    hf_root_dir: str = "/mnt/hf_cache" if utils.is_cluster_node() else "a2/hf_cache"

    # Data parameters
    dataset_name: str = f"{hf_root_dir}/datasets/valhalla/emoji-dataset"
    img_size: int = 48

    # Generation parameters
    num_samples: int = 16
    sample_prompts: List[str] = dataclasses.field(default_factory=lambda: [
        'man bowing deeply with fairy wings', 'house building', 'male sleuth',
        'woman judge with curly hair', 'pouting cat face', 'merwoman light skin tone',
        'ferris wheel', 'man gesturing not ok with folded hands',
        'older woman raising hand', 'adult fairy', 'male judge bowing deeply',
        'older woman with fairy wings', 'shrugging fairy', 'mountain bicyclist',
        'male judge', 'female mechanic'
    ])

    # sample_prompts: List[str] = dataclasses.field(default_factory=lambda: [
    #     'merman', 'john burtt, fluffy grey cat', 'happy cowboy', 'engineering student',
    #     'sleeping guy', 'guy that is rock climbing', 'guy that is falling in the sky',
    #     'astornaut on the moon', 'older man eating candy'
    # ])

    # Text encoding parameters
    model_name: str = f"{hf_root_dir}/models/sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 80

    # Paths
    checkpoint_path: str = "a2/runs/{run_id}/checkpoint.pth"
    samples_path: str = "a2/runs/{run_id}/samples/{epoch}.png"
    configs_path: str = "a2/runs/{run_id}/configs.json"

    # UNet architecture
    num_ch: int = 3
    num_groups: int = 8

    #
    # Helper methods defined below.
    def get_path(self, key, **kwargs):
        if "run_id" not in kwargs:
            kwargs["run_id"] = self.run_id
        path = getattr(self, key).format(**kwargs)
        dir_name = os.path.dirname(path) if "." in path else path
        os.makedirs(dir_name, exist_ok=True)
        return path

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--quick_test",
            action="store_true",
            help="run a fast, lightweight configuration for testing",
        )
        parser.add_argument(
            "--run_id",
            type=str,
            help="Only use for inference (using infer.py)",
        )
        return parser.parse_args()

    def apply_quick_test_config(self, quick_test=False):
        args = self.parse_args()
        if args.quick_test or quick_test:
            self.quick_test = True
            self.run_id = "quick_test"
            self.epochs = 1
            self.batch_size = 4
            self.block_out_ch = (2, 4)
            self.num_layers_per_block = 1
            self.timesteps = 10
            self.save_interval = 1
            self.num_groups = 2

    def apply_infer_config(self):
        args = self.parse_args()
        if not args.run_id:
            raise ValueError(f"Need pass run_id of trained model in '--args.run_id'")
        self.run_id = args.run_id

    def save(self):
        with open(self.get_path("configs_path"), "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


if __name__ == "__main__":
    sys.exit("Intended for import.")
