import sys
import yaml
import time

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from lightning_modules.jetGNN.submodels.gravnet import GravNet as GravNet_jet
from lightning_modules.jetGNN.submodels.multiclass_gravnet import MultiClassGravNet as MultiClassGravNet_jet

import wandb

def main():
    print("Running main")
    print(time.ctime())

    default_config_path = "default_config.yaml"

    with open(default_config_path) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=default_configs, project=default_configs["project"])
    config = wandb.config

    print("Initialising model")
    print(time.ctime())
    model_name = eval(default_configs["model"])
    model = model_name(dict(config))

    checkpoint_callback = ModelCheckpoint(
        monitor="auc", mode="max", save_top_k=2, save_last=True
    )
    
    logger = WandbLogger(
        project=config["project"],
        save_dir=config["artifacts"],
    )
        
    accelerator = "gpu" if torch.cuda.is_available() else None

    trainer = Trainer(
        accelerator = accelerator,
        devices=default_configs["gpus"],
        num_nodes=default_configs["nodes"],
        auto_select_gpus=True,
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


if __name__ == "__main__":

    main()
