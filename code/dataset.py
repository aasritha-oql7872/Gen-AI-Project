"""
CelebA dataset loader.

"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_transform():
    return transforms.Compose([
        transforms.CenterCrop(178),       #  RGB images are 178x218, crop to square
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # scale to [-1, 1]
    ])


class CelebADataset(torch.utils.data.Dataset):
    """Wraps CelebA to return only selected attributes as a float vector."""

    def __init__(self, split="train"):
        self.dataset = datasets.CelebA(
            root="./data",
            split=split,
            target_type="attr",
            transform=get_transform(),
            download=True,
        )
       
        all_attr_names = self.dataset.attr_names
        self.attr_indices = [
            all_attr_names.index(name) for name in config.SELECTED_ATTRS
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, attrs = self.dataset[idx]
        # Pick only selected attributes, convert from {0,1} to float
        selected = attrs[self.attr_indices].float()
        return image, selected


def get_dataloader(split="train"):
    dataset = CelebADataset(split=split)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
  
    loader = get_dataloader("train")
    images, attrs = next(iter(loader))
    print(f"Images: {images.shape}")    # (B, 3, 64, 64)
    print(f"Attrs:  {attrs.shape}")     # (B, 10)
    print(f"Attr values: {attrs[0]}")   # e.g., [1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
