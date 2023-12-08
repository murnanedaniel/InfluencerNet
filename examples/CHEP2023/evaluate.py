import os
import sys
import yaml
import click

try:
    import wandb
except ImportError:
    wandb = None

from .utils import get_module, get_trainer, find_latest_checkpoint

@click.command()
@click.argument("config_file")
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
def main(config_file, checkpoint):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    evaluate(config_file, checkpoint)

def evaluate(config_file, checkpoint=None):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(yaml.dump(config))

    # setup stage
    os.makedirs(config["artifact_dir"], exist_ok=True)

    checkpoint_path = (
        checkpoint
        if checkpoint
        else find_latest_checkpoint(
            config["stage_dir"], templates=["best*.ckpt", "*.ckpt"]
        )
    )
    if not checkpoint_path:
        print("No checkpoint found")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")

    module, config, default_root_dir = get_module(
        config, checkpoint_path=checkpoint_path
    )

    # setup stage
    module.setup(stage="test")

    # run training, depending on whether we are using a Lightning trainable model or not
    trainer = get_trainer(config, default_root_dir)
    with torch.inference_mode():
        trainer.test(module)
    

if __name__ == "__main__":
    main()
