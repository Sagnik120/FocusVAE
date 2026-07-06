"""CelebA dataset with (image, query, region-box) triples."""
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# original aligned CelebA crops are 178 x 218
ORIG_W, ORIG_H = 178, 218


class CelebAQueryDataset(Dataset):
    def __init__(self, root, image_size=64, split="train", split_frac=(0.9, 0.05, 0.05), seed=0):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.queries = pd.read_csv(self.root / "queries.csv")
        self.boxes = pd.read_csv(self.root / "region_boxes.csv")
        self.boxes = self.boxes.set_index(["image_id", "region"])

        image_ids = sorted(self.queries["image_id"].unique())
        rng = random.Random(seed)
        rng.shuffle(image_ids)
        n = len(image_ids)
        n_train = int(n * split_frac[0])
        n_val = int(n * split_frac[1])
        if split == "train":
            self.image_ids = image_ids[:n_train]
        elif split == "val":
            self.image_ids = image_ids[n_train:n_train + n_val]
        else:
            self.image_ids = image_ids[n_train + n_val:]

        self.queries = self.queries[self.queries["image_id"].isin(self.image_ids)]
        self.queries_by_image = {
            img_id: g.to_dict("records")
            for img_id, g in self.queries.groupby("image_id")
        }

        self.image_size = image_size
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.image_ids)

    def _normalized_box(self, image_id, region):
        row = self.boxes.loc[(image_id, region)]
        x1 = max(row["x1"], 0) / ORIG_W
        y1 = max(row["y1"], 0) / ORIG_H
        x2 = min(row["x2"], ORIG_W) / ORIG_W
        y2 = min(row["y2"], ORIG_H) / ORIG_H
        return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(self.img_dir / image_id).convert("RGB")
        img = self.tf(img)

        record = random.choice(self.queries_by_image[image_id])
        query = record["query"]
        region = record["region"]
        label = record["label"]
        box = self._normalized_box(image_id, region)

        return {
            "image": img,
            "query": query,
            "region": region,
            "label": torch.tensor(label, dtype=torch.float32),
            "box": box,
            "image_id": image_id,
        }


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "query": [b["query"] for b in batch],
        "region": [b["region"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "box": torch.stack([b["box"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
    }
