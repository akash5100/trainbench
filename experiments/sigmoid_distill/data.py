"""MS-COCO 2017 multi-label data parsing and lightweight DataLoaderLite."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


@dataclass
class CocoSample:
  """Coco Sample for multi-class classification."""

  id: int
  image_path: Path
  category_id: list[int]


def load_coco_metadata(data_root: Path, split: str = "train2017") -> tuple[list[CocoSample], np.ndarray, list[str]]:
  """Loads the coco split metadata from disk."""
  anno_file = data_root / "annotations" / f"instances_{split}.json"
  img_dir = data_root / split

  with open(anno_file, encoding="utf-8") as f:
    data = json.load(f)

  # 1. Map 80 categories deterministically to 0..79 indices
  categories = sorted(data["categories"], key=lambda c: c["id"])
  class_names = [c["name"] for c in categories]
  cat_to_idx = {c["id"]: idx for idx, c in enumerate(categories)}

  # 2. Group all category_ids by image_id (one image has multiple objects/classes)
  image_to_categories = defaultdict(set)
  for anno in data["annotations"]:
    image_to_categories[anno["image_id"]].add(anno["category_id"])

  # 3. Create one CocoSample per unique image in data["images"]
  sample_buffer: list[CocoSample] = []
  for img in data["images"]:
    img_id = img["id"]
    img_path = img_dir / img["file_name"]
    cat_ids = sorted(list(image_to_categories[img_id]))
    sample_buffer.append(
      CocoSample(
        id=img_id,
        image_path=img_path,
        category_id=cat_ids,
      )
    )

  # 4. Build multi-hot target matrix [num_images, 80]
  labels = np.zeros((len(sample_buffer), len(class_names)), dtype=np.float32)
  for row_idx, sample in enumerate(sample_buffer):
    for cat_id in sample.category_id:
      labels[row_idx, cat_to_idx[cat_id]] = 1.0

  return sample_buffer, labels, class_names


class CocoDataset(Dataset):
  """MS-COCO PyTorch Dataset yielding (image_tensor, multi_hot_target, image_id)."""

  def __init__(self, samples: list[CocoSample], labels: np.ndarray, transform=None):
    """Initializes dataset with sample metadata, ground-truth labels, and optional transform."""
    self.samples = samples
    self.labels = labels
    self.transform = transform

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    sample = self.samples[index]

    # Read image from disk directly into a uint8 PyTorch Tensor [C, H, W]
    image = read_image(str(sample.image_path))

    # Channel sanity: repeat grayscale to 3 channels, strip alpha from 4-channel RGBA
    if image.shape[0] == 1:
      image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
      image = image[:3, :, :]

    # Convert uint8 [0..255] to float32 [0.0..1.0]
    image = image.float() / 255.0

    if self.transform is not None:
      image = self.transform(image)

    target = torch.from_numpy(self.labels[index])
    return image, target, sample.id


# ━━━━━━━━ DataLoader Factory ━━━━━━━━


def build_dataloaders(
  data_root: str | Path,
  batch_size: int = 16,
  processor=None,
  transform=None,
  num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[str]]:
  """Builds train and val PyTorch DataLoaders for MS-COCO 2017.

  Accepts either a model `processor` or explicit torchvision `transform`.
  """
  root = Path(data_root)

  train_samples, train_labels, class_names = load_coco_metadata(root, split="train2017")
  val_samples, val_labels, _ = load_coco_metadata(root, split="val2017")

  if processor is not None:
    active_transform = getattr(processor, "transform", processor)
  elif transform is not None:
    active_transform = transform
  else:
    raise ValueError("Either `processor` or `transform` must be explicitly provided to build_dataloaders.")

  train_ds = CocoDataset(train_samples, train_labels, transform=active_transform)
  val_ds = CocoDataset(val_samples, val_labels, transform=active_transform)

  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
  )

  return train_loader, val_loader, class_names


# ━━━━━━━━ CLI Sanity Check ━━━━━━━━


if __name__ == "__main__":
  import argparse

  from trainbench.models.torch.lit_classifier import LitClassifierConfig, LitClassifierProcessor

  parser = argparse.ArgumentParser(description="Inspect MS-COCO 2017 multi-label DataLoaders.")
  parser.add_argument("--data-root", type=str, required=True, help="Path to COCO 2017 root directory.")
  parser.add_argument("--batch-size", type=int, default=8, help="Batch size for sanity test.")
  parser.add_argument("--img-size", type=int, default=224, help="Image resolution.")
  args = parser.parse_args()

  config = LitClassifierConfig(image_size=args.img_size)
  processor = LitClassifierProcessor(config=config)

  train_loader, val_loader, class_names = build_dataloaders(
    data_root=args.data_root,
    batch_size=args.batch_size,
    processor=processor,
  )

  print(f"Train batches: {len(train_loader)} (total samples: {len(train_loader.dataset)})")
  print(f"Val batches:   {len(val_loader)} (total samples: {len(val_loader.dataset)})")
  print(f"Classes count: {len(class_names)}")

  # Test batch iteration
  batch_imgs, batch_targets, batch_ids = next(iter(train_loader))
  print(f"Batch images shape:  {batch_imgs.shape} (Expected: [{args.batch_size}, 3, {args.img_size}, {args.img_size}])")
  print(f"Batch targets shape: {batch_targets.shape} (Expected: [{args.batch_size}, 80])")
  print(f"Batch sample IDs:    {batch_ids}")
