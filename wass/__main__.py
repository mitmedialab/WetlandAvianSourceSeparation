"""__main__.py

The file contains all the main interactions with the library:
    - Training
    - Optimization @TODO
    - Inference
"""
import argparse

from wass.train import Solver, TrainingConfig
from wass.demo import inference_demo


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

parser.add_argument("-t", "--train", action="store_true", help="training")
parser.add_argument("--config", type=str, help="training config file path")
parser.add_argument("--gpu", nargs="+", type=int, help="cuda devices")

parser.add_argument("-d", "--demo", action="store_true", help="demonstration")
parser.add_argument("--model", type=str, help="path to a trained model")
parser.add_argument("--mixture", type=str, help="path to a mixture audio file")
parser.add_argument("--dest", type=str, help="path to save separated sources")

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


"""Inference Demonstration
"""
if args.demo:
    if args.model is None:
        parser.error("--demo requires --model to be provided.")
    if args.mixture is None:
        parser.error("--demo requires --mixture to be provided.")
    if args.dest is None:
        parser.error("--demo requires --dest to be provided.")

    inference_demo(args.model, args.mixture, args.dest)

    exit(0)


"""No Arguments Provided
"""
print("You must provide valid arguments. Run 'wass -h' for more infos.")
exit(1)
