from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.core.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)
from albumentations.pytorch.transforms import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from ultralytics.data.converter import coco91_to_coco80_class

from src.utils import recover_bounding_boxes


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Adjusted handling for bounding boxes and labels
    boxes = [target["boxes"] if target["boxes"].nelement() > 0 else torch.empty((0, 4)) for target in targets]
    labels = [
        target["labels"] if target["labels"].nelement() > 0 else torch.empty((0,), dtype=torch.int64)
        for target in targets
    ]

    # Pad the sequences if there are any boxes or labels, else create appropriate empty tensors
    if any(b.nelement() > 0 for b in boxes):
        boxes_padded = pad_sequence(boxes, batch_first=True, padding_value=0)
    else:
        boxes_padded = torch.zeros((len(images), 0, 4), dtype=torch.float32)

    if any(label.nelement() > 0 for label in labels):
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    else:
        labels_padded = torch.zeros((len(images), 0), dtype=torch.int64)

    # Stack all images to create a single tensor
    images = torch.stack(images)

    targets_padded = {"boxes": boxes_padded, "labels": labels_padded}
    return images, targets_padded


def convert_bboxes(
    bboxes: list[list[float]], source_format: str, target_format: str, rows: int, cols: int
) -> list[list[float]]:
    bboxes = convert_bboxes_to_albumentations(bboxes, source_format=source_format, rows=rows, cols=cols)
    return convert_bboxes_from_albumentations(bboxes, target_format=target_format, rows=rows, cols=cols)


DATA_PATH = Path("~/data/coco8").expanduser()


@dataclass
class Config:
    data_path: Path = DATA_PATH
    train_batch_size: int = 32
    val_batch_size: int = 1
    num_workers: int = 1
    learning_rate: float = 0.001
    max_epochs: int = 10
    image_size: int = 512


SIZE = 640


def get_train_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=SIZE, p=1),
            A.PadIfNeeded(
                min_height=SIZE,
                min_width=SIZE,
                p=1,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], clip=True),
    )


def get_val_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=SIZE, p=1),
            A.PadIfNeeded(
                min_height=SIZE,
                min_width=SIZE,
                p=1,
                position="top_left",
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            ),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255, p=1),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], clip=True),
    )


def coco_to_yolo(bboxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Convert a list of bounding boxes from COCO format to unnormalized YOLO format.

    Args:
        bboxes (List[Tuple[float, float, float, float]]): List of bounding boxes in
            COCO format (x_min, y_min, width, height).

    Returns:
        List[Tuple[float, float, float, float]]: List of bounding boxes in
            YOLO format (x_center, y_center, width, height).
    """
    yolo_bboxes = []

    for x_min, y_min, w, h in bboxes:
        x_center = x_min + w / 2
        y_center = y_min + h / 2
        yolo_bboxes.append((x_center, y_center, w, h))

    return yolo_bboxes


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path, mode: Literal["train", "val"], transforms: A.Compose):
        self.data_path = data_path
        self.transforms = transforms
        self.class_mapping = coco91_to_coco80_class()

        ids = [x.stem for x in (self.data_path / "images" / mode).glob("*.jpg")]

        self.data = [
            {
                "image_file_name": self.data_path / "images" / mode / f"{x}.jpg",
                "image_id": x,
                "annotation_file_name": self.data_path / "labels" / mode / f"{x}.txt",
            }
            for x in ids
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx]

        image = cv2.imread(str(data["image_file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the corresponding label file
        label_path = data["annotation_file_name"]

        with label_path.open() as f:
            labels_and_bboxes = f.read()

        boxes_and_class_label = recover_bounding_boxes(labels_and_bboxes, image.shape)

        bboxes = [x[:-1] for x in boxes_and_class_label]
        labels = [x[-1] for x in boxes_and_class_label]

        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)

        transformed_image = transformed["image"]

        transformed_bboxes = coco_to_yolo(transformed["bboxes"])

        return transformed_image, {
            "boxes": torch.tensor(transformed_bboxes, dtype=torch.float32),
            "labels": torch.tensor(transformed["labels"], dtype=torch.int64),
        }


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = config.data_path
        self.train_transforms = get_train_transforms()
        self.val_transforms = get_val_transforms()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.coco_train = CocoDataset(self.config.data_path, "train", self.train_transforms)

        self.coco_val = CocoDataset(self.config.data_path, "val", self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.coco_train,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.coco_val,
            batch_size=self.config.val_batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            shuffle=False,
            persistent_workers=True,
        )
