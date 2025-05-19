import os

import hydra
import pytorch_lightning as pl
import super_gradients as sg
import torch
from omegaconf import DictConfig, OmegaConf
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets.coco_format_detection import \
    COCOFormatDetectionDataset
from super_gradients.training.losses import PPYoloELoss, YoloXFastDetectionLoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import \
    PPYoloEPostPredictionCallback
from super_gradients.training.models.detection_models.yolo_base import \
    YoloXPostPredictionCallback
from super_gradients.training.transforms.transforms import (
    DetectionHorizontalFlip, DetectionHSV, DetectionPaddedRescale,
    DetectionRandomAffine, DetectionStandardize,
    DetectionTargetsFormatTransform)
from super_gradients.training.utils.collate_fn import DetectionCollateFN
from torch.utils.data import DataLoader

from dataset import (WOBDataset, categories, detection_collate_fn,
                     get_transforms)
from models import RTDETR

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(42)

DECI_MODELS = {
    "yolonas": {
        "s": {
            "model": Models.YOLO_NAS_S,
            "loss": PPYoloELoss(use_static_assigner=False, num_classes=len(categories), reg_max=16),
            "post_prediction_callback": PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
        },
        "m": {
            "model": Models.YOLO_NAS_M,
            "loss": PPYoloELoss(use_static_assigner=False, num_classes=len(categories), reg_max=16),
            "post_prediction_callback": PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
        },
        "l": {
            "model": Models.YOLO_NAS_L,
            "loss": PPYoloELoss(use_static_assigner=False, num_classes=len(categories), reg_max=16),
            "post_prediction_callback": PPYoloEPostPredictionCallback(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
        },
    },
    "yolox": {
        "s": {
            "model": Models.YOLOX_S,
            "loss": YoloXFastDetectionLoss(strides=[8, 16, 32], num_classes=len(categories)),
            "post_prediction_callback": YoloXPostPredictionCallback(conf=0.001, iou=0.6)
        }, 
        "m": {
            "model": Models.YOLOX_M,
            "loss": YoloXFastDetectionLoss(strides=[8, 16, 32], num_classes=len(categories)),
            "post_prediction_callback": YoloXPostPredictionCallback(conf=0.001, iou=0.6)
        },
        "l": {
            "model": Models.YOLOX_L,
            "loss": YoloXFastDetectionLoss(strides=[8, 16, 32], num_classes=len(categories)),
            "post_prediction_callback": YoloXPostPredictionCallback(conf=0.001, iou=0.6)
        },
    }
}


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.model.name == "rtdetr" or cfg.model.name == "dfine":
        train_dataset = WOBDataset(
            base_dir=cfg.data.base_dir,
            annotations_file=os.path.join(cfg.data.base_dir, cfg.data.train_annotations),
            transform=get_transforms(train=True)
        )
        
        val_dataset = WOBDataset(
            base_dir=cfg.data.base_dir,
            annotations_file=os.path.join(cfg.data.base_dir, cfg.data.val_annotations),
            transform=get_transforms(train=False)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=True, 
            num_workers=cfg.training.num_workers, 
            collate_fn=detection_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.training.batch_size, 
            shuffle=False, 
            num_workers=cfg.training.num_workers,
            collate_fn=detection_collate_fn, 
            drop_last=True
        )
        model = RTDETR(
            model_checkpoint=cfg.model.model_checkpoint,
            num_classes=len(categories),
            learning_rate=cfg.training.learning_rate,
            fine_tune=False
        ).train()
        wandb_logger = pl.loggers.WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            log_model=cfg.logging.log_model
        )

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator="gpu",
            logger=wandb_logger,
        )
        trainer.fit(model, train_loader, val_loader)
    elif cfg.model.name.startswith("yolo"):
        trainer = sg.Trainer(
            experiment_name=cfg.logging.run_name,
            ckpt_root_dir=f"/root/cv/project/ckpt/{cfg.model.name}_{cfg.model.size}"
        )

        train_dataset = COCOFormatDetectionDataset(
            data_dir=cfg.data.base_dir,
            images_dir="images/train",
            json_annotation_file=cfg.data.train_annotations,
            input_dim=(640, 640),
            ignore_empty_annotations=False,
            with_crowd=False,
            all_classes_list=[category["name"] for category in categories],
            transforms=[
                DetectionRandomAffine(degrees=0.0, scales=(0.5, 1.5), shear=0.0, target_size=(640, 640), filter_box_candidates=False, border_value=128),
                DetectionHSV(prob=1.0, hgain=5, vgain=30, sgain=30),
                DetectionHorizontalFlip(prob=0.5),
                DetectionPaddedRescale(input_dim=(640, 640)),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH"),
            ],
        )
        val_dataset = COCOFormatDetectionDataset(
            data_dir=cfg.data.base_dir,
            images_dir="images/val",
            json_annotation_file=cfg.data.val_annotations,
            input_dim=(640, 640),
            ignore_empty_annotations=False,
            with_crowd=False,
            all_classes_list=[category["name"] for category in categories],
            transforms=[
                DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
                DetectionStandardize(max_value=255),
                DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH"),
            ],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            collate_fn=DetectionCollateFN(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            collate_fn=DetectionCollateFN(),
        )
        train_params = {
            "sg_logger": "wandb_sg_logger",
            "sg_logger_params":
            {
                "project_name": cfg.logging.project_name,
                "experiment_name": cfg.logging.run_name,
                "save_checkpoints_remote": cfg.logging.log_model,
                "save_tensorboard_remote": True,
                "save_logs_remote": True
            },
            "warmup_initial_lr": 1e-5,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.5,
            "optimizer": "AdamW",
            "zero_weight_decay_on_bias_and_bn": True,
            "lr_warmup_epochs": 1,
            "warmup_mode": "LinearEpochLRWarmup",
            "optimizer_params": {"weight_decay": 0.0001},
            "ema": False,
            "average_best_models": False,
            "ema_params": {"beta": 25, "decay_type": "exp"},
            "max_epochs": cfg.training.max_epochs,
            "mixed_precision": True,
            "loss": DECI_MODELS[cfg.model.name][cfg.model.size]["loss"],
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(categories),
                    normalize_targets=True,
                    include_classwise_ap=True,
                    class_names=[category["name"] for category in categories],
                    post_prediction_callback=DECI_MODELS[cfg.model.name][cfg.model.size]["post_prediction_callback"],
                )
            ],
            "metric_to_watch": "mAP@0.50",
        }
        model = models.get(
            DECI_MODELS[cfg.model.name][cfg.model.size]["model"],
            num_classes=len(categories), 
            pretrained_weights="coco")
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_loader,
            valid_loader=val_loader,
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")

if __name__ == "__main__":
    main()
