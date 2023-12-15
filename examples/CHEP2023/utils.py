import os
from pathlib import Path
from importlib import import_module
import re
import sys

import torch

from src.models import *

try:
    import wandb
except ImportError:
    wandb = None
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger


class MetricThresholdStopping(Callback):
    def __init__(self, monitor_metric, threshold_value, mode):
        super().__init__()
        self.monitor_metric = monitor_metric
        self.threshold_value = threshold_value
        self.mode = mode
        assert mode in ["min", "max"], "Mode must be 'min' or 'max'"

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        metric_value = logs.get(self.monitor_metric)
        if metric_value is not None:
            if self.mode == "min" and metric_value <= self.threshold_value:
                trainer.should_stop = True
            elif self.mode == "max" and metric_value >= self.threshold_value:
                trainer.should_stop = True


def get_default_root_dir():
    if (
        "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
    ):
        return os.path.join(".", os.environ["SLURM_JOB_ID"])
    else:
        return None


def load_config_and_checkpoint(config_path, default_root_dir):
    # Check if there is a checkpoint to load
    checkpoint = (
        find_latest_checkpoint(default_root_dir)
        if default_root_dir is not None
        else None
    )
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        return (
            torch.load(checkpoint, map_location=torch.device("cpu"))[
                "hyper_parameters"
            ],
            checkpoint,
        )
    else:
        print("No checkpoint found, loading config from file")
        with open(config_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader), None


def get_trainer(config, default_root_dir):
    job_id = (
        os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ
        and "SLURM_JOB_QOS" in os.environ
        and "interactive" not in os.environ["SLURM_JOB_QOS"]
        and "jupyter" not in os.environ["SLURM_JOB_QOS"]
        else None
    )

    logger = (
        WandbLogger(
            project=config["project"],
            save_dir=config["artifact_dir"],
            id=job_id,
            group=config.get("group", None),
        )
        if wandb is not None and config.get("log_wandb", True)
        else CSVLogger(save_dir=config["artifact_dir"])
    )

    checkpoint_metric_to_monitor = (
        config["metric_to_monitor"] if "metric_to_monitor" in config else "val_loss"
    )
    checkpoint_metric_mode = config["metric_mode"] if "metric_mode" in config else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["artifact_dir"], "artifacts"),
        filename="best",
        monitor=checkpoint_metric_to_monitor,
        mode=checkpoint_metric_mode,
        save_top_k=1,
        save_last=True,
    )

    filename_suffix = (
        str(logger.experiment.id)
        if (
            hasattr(logger, "experiment")
            and hasattr(logger.experiment, "id")
            and logger.experiment.id is not None
        )
        else ""
    )
    filename = (
        "best-" + filename_suffix + "-{" + checkpoint_metric_to_monitor + ":5f}-{epoch}"
    )

    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last-{filename_suffix}"

    callbacks = [checkpoint_callback]

    if config.get("stop_on_threshold", None) is not None:
        metric_threshold_callback = MetricThresholdStopping(
            monitor_metric=config["stop_on_threshold"]["metric_to_monitor"],
            threshold_value=config["stop_on_threshold"]["threshold_value"],
            mode=config["stop_on_threshold"]["metric_mode"],
        )
        callbacks.append(metric_threshold_callback)

    gpus = config.get("gpus", 0)
    accelerator = "gpu" if gpus else "cpu"
    devices = gpus or 1
    torch.set_float32_matmul_precision("medium")

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        num_nodes=config["nodes"],
        max_epochs=config["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        default_root_dir=default_root_dir,
    )


def get_module(config, checkpoint_path=None):
    if config["model"] in globals():
        model_class = globals()[config["model"]]
    else:
        raise ValueError(f"Model name {config['model']} not found in globals")

    default_root_dir = get_default_root_dir()

    # First check if we need to load a checkpoint
    if checkpoint_path is not None:
        stage_module, config = load_module(checkpoint_path, model_class, config)
    elif default_root_dir is not None and find_latest_checkpoint(
        default_root_dir, "*.ckpt"
    ):
        checkpoint_path = find_latest_checkpoint(default_root_dir, "*.ckpt")
        stage_module, config = load_module(checkpoint_path, model_class, config)
    else:
        stage_module = model_class(config)
    return stage_module, config, default_root_dir


def load_module(checkpoint_path, stage_module_class, config):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["hyper_parameters"]
    stage_module = stage_module_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path
    )
    stage_module._hparams = {**stage_module._hparams, **config}
    return stage_module, config


def find_latest_checkpoint(checkpoint_base, templates=None):
    if templates is None:
        templates = ["*.ckpt"]
    elif isinstance(templates, str):
        templates = [templates]
    checkpoint_paths = []
    for template in templates:
        checkpoint_paths = checkpoint_paths or [
            str(path) for path in Path(checkpoint_base).rglob(template)
        ]
    return max(checkpoint_paths, key=os.path.getctime) if checkpoint_paths else None
