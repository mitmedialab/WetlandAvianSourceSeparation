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
    "-t", "--train", help="training procedure", action="store_true"
)
parser.add_argument("-c", "--config", help="training configuration file path")
parser.add_argument(
    "-g", "--gpu", help="activate cuda gpu acceleration", action="store_true"
)

args = parser.parse_args()


"""Training Procedure
"""
if args.train:
    if args.config is None:
        parser.error("--train requires --config to be provided.")

    config = TrainingConfig.load(args.config)
    print(config, "\n")

    solver = Solver(config, cuda=args.gpu)
    solver()

    exit(0)


"""No Arguments Provided
"""
print("You must provide valid arguments. Run 'wass -h' for more infos.")
