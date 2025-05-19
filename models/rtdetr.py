import time

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from transformers import (AutoImageProcessor, DFineForObjectDetection,
                          RTDetrForObjectDetection, RTDetrImageProcessor)

import wandb


class RTDETR(pl.LightningModule):
    """
    PyTorch Lightning wrapper for RT-DETR model
    """
    def __init__(
            self, 
            num_classes: int = 91,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            model_checkpoint: str = "PekingU/rtdetr_r18vd",
            fine_tune: bool = True
        ):
        super().__init__()
        self.save_hyperparameters({
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fine_tune": fine_tune
        })
        
        self.model = RTDetrForObjectDetection.from_pretrained(
        # self.model = DFineForObjectDetection.from_pretrained(
            model_checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        if not fine_tune:
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False
        
        self.processor = RTDetrImageProcessor.from_pretrained(model_checkpoint)
        self.map_metric = MeanAveragePrecision(box_format='xywh', iou_type="bbox")
    
    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def training_step(self, batch, batch_idx):
        start_time = time.time()

        outputs = self.model(**batch)
        
        loss = outputs.loss
        elapsed = time.time() - start_time
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/step_time", elapsed, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True)
        
        predictions = []
        targets = []
        
        predictions = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.5,
            target_sizes=torch.stack([label_dict["orig_size"] for label_dict in batch["labels"]])
        )

        for label_dict in batch["labels"]:
            converted_boxes = []
            for box in label_dict["boxes"]:
                converted_boxes.append(
                    torch.tensor([box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]]).to("cuda")
                )
                converted_boxes[-1] *= 640
            
            targets.append({
                "boxes": torch.stack(converted_boxes),
                "labels": label_dict["class_labels"],
            })
        
        self.map_metric.update(predictions, targets)
        
        metrics = self.map_metric.compute()
        self.log("val/mAP50", metrics["map_50"], prog_bar=True)
        self.log("val/mAP", metrics["map"], prog_bar=True)
        self.log("val/precision", metrics["mar_100"], prog_bar=True)
        self.log("val/recall", metrics["mar_100"], prog_bar=True)
        
        # Visualize first image in batch with predictions (only for batch_idx=0)
        # if batch_idx == 0:
        #     # Get the first image
        #     image = batch["pixel_values"][0]
        #     # Denormalize and convert to numpy array
        #     image = (image * 255).byte().permute(1, 2, 0).cpu().numpy().copy()
            
        #     # Get predictions for the first image
        #     boxes = predictions[0]["boxes"]
        #     scores = predictions[0]["scores"]
        #     labels = predictions[0]["labels"]
            
        #     # Draw boxes using OpenCV
        #     for box, score, label in zip(boxes, scores, labels):
        #         cy, cx, h, w = box.cpu().numpy().astype(int)
                
        #         cv2.rectangle(image, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 255, 0), 2)
                
        #         text = f'Class {label}: {score:.2f}'
                
        #         (text_width, text_height), baseline = cv2.getTextSize(
        #             text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        #         )
                
        #         cv2.putText(
        #             image, 
        #             text, 
        #             (cx + h//2, cy - w // 2 - baseline), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.5, 
        #             (0, 0, 0), 
        #             1
        #         )
            
        #     self.logger.experiment.log(
        #         {"val/detection_example": wandb.Image(image, caption=f"Validation predictions (epoch {self.current_epoch})")}
        #     )
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=self.hparams.learning_rate / 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        inputs = self.processor(images=image, return_tensors="pt")
        
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.5,
            target_sizes=[(image.height, image.width)]
        )[0]
        
        return results

