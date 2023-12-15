import os
import yaml
import click

try:
    import wandb
except ImportError:
    wandb = None

from utils import get_module, get_trainer

@click.command()
@click.argument("config_file")
@click.option("--checkpoint", "-c", default=None, help="Checkpoint to use for training")
def main(config_file, checkpoint):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    train(config_file, checkpoint)


# Refactoring to allow for auto-resume and manual resume of training
# 1. We cannot init a model before we know if we are resuming or not
# 2. First check if the module is a lightning module


def train(config_file, checkpoint=None):
    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(yaml.dump(config))

    # setup stage
    os.makedirs(config["artifacts"], exist_ok=True)

    module, config, default_root_dir = get_module(
        config, checkpoint_path=checkpoint
    )

    # run training, depending on whether we are using a Lightning trainable model or not
    trainer = get_trainer(config, default_root_dir)
    trainer.fit(module)
    

if __name__ == "__main__":
    main()
