import os
import shutil
import yaml
import click
from itertools import product
import subprocess

@click.command()
@click.argument('config_file', type=str, required=True)

def main(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get the path to the directory containing the config
    config_path = os.path.dirname(config_file)

    sweep(config, config_path, config_file)

def sweep(config, config_path, config_file):
    """
    A simple grid scan config for manual sweep
    """

    # Run combo_config on the config file
    config_list = combo_config(config)

    for run_config in config_list:

        # Save the run_config as a yaml file in the config_path / sweep using shutil
        run_config_path = os.path.join(config_path, "sweeps",  os.path.basename(config_file))
        if os.path.exists(run_config_path):
            i = 1
            while os.path.exists(run_config_path):
                run_config_path = os.path.join(config_path, "sweeps", os.path.basename(config_file).split('.')[0] + str(i) + '.yaml')
                i += 1

        with open(run_config_path, 'w') as f:
            yaml.dump(run_config, f)

        # Then submit the job using the submit function
        submit(run_config_path)

def submit(config_path):
    """
    Submit a batch job with shutil
    """

    subprocess.run(['sbatch', 'PM_submit_job_requeue.sh', config_path])

def combo_config(config):  # C
    total_list = {k: (v if type(v) == list else [v]) for (k, v) in config.items()}
    keys, values = zip(*total_list.items())

    # Build list of config dictionaries
    config_list = []
    [config_list.append(dict(zip(keys, bundle))) for bundle in product(*values)]

    return config_list


if __name__ == "__main__":
    main()
