"""This module contains a class for loading and processing the dataset.

Run this file to visualize emojis."""

import base64
import io
import os
import random

from configs import Config
import datasets
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import utils


class EmojiDataset(data.Dataset):
    """Dataset class for loading emoji images."""

    def __init__(self, cfg):
        dataset = datasets.load_dataset(cfg.dataset_name, split="train")

        self.image_data = []
        for item in dataset:
            text = item['text'].strip()
            if text.endswith(" type 1 2"):
                text = text[:-len(" type 1 2")]
            if "type" not in text:
                self.image_data.append({'image': item['image'], 'text': text})

        if cfg.quick_test:
            self.image_data = random.sample(self.image_data, k=20)
        print(f"Dataset size: {len(self.image_data)} images")

        self.transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        item = self.image_data[idx]
        image = item['image']

        # Convert to RGB if needed (handle RGBA images)
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations
        image = self.transform(image)
        return image, item['text']


def main():
    """Main function to demonstrate the emoji dataset with grid visualization"""
    cfg = Config()
    cfg.apply_quick_test_config(quick_test=True)

    utils.set_seeds()

    dataset = EmojiDataset(cfg)
    random_indices = random.sample(range(len(dataset)), 16)
    samples = []
    prompts = []
    for idx in random_indices:
        sample, prompt = dataset[idx]
        samples.append(sample)
        prompts.append(prompt)
    samples_path = cfg.get_path("samples_path", epoch="demo")
    utils.plot_sample_grid(samples, prompts, samples_path)


if __name__ == "__main__":
    main()
