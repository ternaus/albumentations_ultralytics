import argparse

import pytorch_lightning as pl
import torch
import torchmetrics
from ultralytics.models import YOLO

from src.data_module import CocoDataModule, Config
from src.utils import cxcywh2xyxy, non_max_suppression


class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = YOLO("yolov8n.pt").model
        self.config = config
        self.val_map = torchmetrics.detection.mean_ap.MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=None,  # Common IoU threshold for object detection
            class_metrics=True,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, targets = batch

        # Convert targets to the format expected by the model
        yolo_targets = []
        for i in range(len(targets["boxes"])):
            boxes = targets["boxes"][i]
            labels = targets["labels"][i]
            if boxes.nelement() == 0:
                yolo_targets.append(
                    {"labels": torch.empty((0,), dtype=torch.int64), "boxes": torch.empty((0, 4), dtype=torch.float32)}
                )
            else:
                yolo_targets.append({"labels": labels, "boxes": boxes})

        # Forward pass
        outputs = self.model(images, yolo_targets)

        # Compute loss
        loss = outputs["loss"]

        # Log the loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        with torch.inference_mode():
            outputs = self.forward(images)

        raw_preds = non_max_suppression(outputs[0], conf_thres=0.25, iou_thres=0.5, max_det=200, max_wh=640)

        # Fill preds with actual predictions where available
        preds = []
        for pred in raw_preds:
            preds += [{"boxes": pred[..., :4], "scores": pred[..., 4], "labels": pred[..., 5].int()}]

        target_formatted = []
        for idx in range(len(targets["boxes"])):
            valid_mask = targets["labels"][idx] != -1
            if valid_mask.any():
                valid_boxes = cxcywh2xyxy(targets["boxes"][idx][valid_mask])
                valid_labels = targets["labels"][idx][valid_mask]
                target_formatted.append({"boxes": valid_boxes, "labels": valid_labels})
            else:
                target_formatted.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32, device=images.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=images.device),
                    }
                )

        self.val_map.update(preds, target_formatted)

    def on_validation_epoch_end(self):
        map_result = self.val_map.compute()
        self.log("val_mAP", map_result["map_50"])  # Log overall mAP
        self.val_map.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train or validate the model")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="Mode to run the model in")

    args = parser.parse_args()

    # Usage
    config = Config()
    coco_data_module = CocoDataModule(config)
    yolo_model = ObjectDetectionModel(config)

    # Trainer setup
    trainer = pl.Trainer(max_epochs=config.max_epochs, precision="16-true")

    if args.mode == "train":
        trainer.fit(yolo_model, coco_data_module)
    elif args.mode == "val":
        trainer.validate(model=yolo_model, datamodule=coco_data_module, verbose=True)
