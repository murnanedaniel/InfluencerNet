import click

@click.command()
@click.argument('configs', nargs=-1, type=click.Path(exists=True), min=1)
def main(configs):
    for config in configs:
        # do something with each config file
        print(f"Processing config file: {config}")

if __name__ == '__main__':
    main()