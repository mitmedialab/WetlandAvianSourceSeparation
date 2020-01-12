"""__main__.py

The file contains all the main interactions with the library:
    - Training
    - Optimization @TODO
    - Inference @TODO
"""
import argparse

from wass.train import Solver, TrainingConfig


"""Arguments Setup
"""
parser = argparse.ArgumentParser(
    prog="wass",
    description="""
Wetland Avian Source Separation (WASS) -- Scripts
\tSource Code:
\t\thttps://github.com/mitmedialab/WetlandAvianSourceSeparation
\tAuthors:
\t\thttps://github.com/FelixMichaud
\t\thttps://github.com/yliess86
\t\thttps://github.com/slash6475
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "-t", "--train", action="store_true", help="training procedure"
)
parser.add_argument(
    "-c", "--config", type=str, help="training configuration file path"
)
parser.add_argument(
    "-g",
    "--gpu",
    nargs="+",
    type=int,
    help="cuda devices for gpu acceleration",
)

args = parser.parse_args()


"""Training Procedure
"""
if args.train:
    if args.config is None:
        parser.error("--train requires --config to be provided.")

    config = TrainingConfig.load(args.config)
    print(config, "\n")

    cuda_devices = args.gpu
    solver = (
        Solver(config, cuda=False)
        if not cuda_devices
        else Solver(config, cuda=True, cuda_devices=cuda_devices)
    )
    solver()

    exit(0)


"""No Arguments Provided
"""
print("You must provide valid arguments. Run 'wass -h' for more infos.")
