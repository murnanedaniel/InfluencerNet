import os
import yaml
import click
import sys
sys.path.append("../")

import torch
try:
    # from pytorch_lightning.loggers import WandbLogger
    from lightning.pytorch.loggers import WandbLogger
    import wandb
    use_wandb = True
except Exception:
    print("Wandb not installed")
    use_wandb = False

# use_wandb = False

# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

from lightning_modules import *

# from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.strategies import DDPStrategy

@click.command()
@click.argument('config', type=str, required=True)
@click.option('--root_dir', default=None)
@click.option('--checkpoint', "-c", default=None)

def main(config, root_dir=None, checkpoint=None):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['root_dir'] = root_dir
    train(config, checkpoint)

def train(config, checkpoint):

    if checkpoint is not None:
        loaded_configs = torch.load(checkpoint)["hyper_parameters"]
        config.update(loaded_configs)

    model_name = config["model"] + config["backbone"]
    if model_name in globals():
        model = globals()[model_name](config)
    else:
        raise ValueError(f"Model name {model_name} not found in globals")

    os.makedirs(config["artifacts"], exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["artifacts"],
        filename='best',
        monitor="val_loss", 
        mode="min", 
        save_top_k=1, 
        save_last=True
    )
    
    if use_wandb:
        logger = WandbLogger(
            project=config["project"],
            save_dir=config["artifacts"],
        )
        filename_suffix = logger.experiment.id
    else:
        logger = None
        filename_suffix = ""

    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last-{filename_suffix}"

    if config["root_dir"] is None:
        if 'SLURM_JOB_ID' in os.environ:
            default_root_dir = os.path.join(".", os.environ['SLURM_JOB_ID'])
        else:
            default_root_dir = '.'
    else:
        default_root_dir = os.path.join(".", config["root_dir"])

    accelerator = "gpu" if torch.cuda.is_available() else None

    gradient_clip = config["gradient_clip"] if "gradient_clip" in config else None

    trainer = Trainer(
        accelerator = accelerator,
        devices=config["gpus"],
        num_nodes=config["nodes"],
        # auto_select_gpus=True,
        max_epochs=config["max_epochs"],
        logger=logger,
        strategy=DDPStrategy(static_graph=True),
        callbacks=[checkpoint_callback],
        default_root_dir=default_root_dir,
        num_sanity_val_steps=0,
        gradient_clip_val=gradient_clip,
    )

    print("Fitting with checkpoint: ", checkpoint)
    trainer.fit(model, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
