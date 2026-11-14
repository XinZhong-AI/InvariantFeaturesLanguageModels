import argparse

from dataset import load_groups
from models import load_model

from intervention import run_intervention
from energy import run_energy

from seed import set_seed
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "info"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "intervention",
            "energy"
        ],
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=32
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True
    )

    return parser.parse_args()


def main():

    args = parse_args()

    set_seed(args.seed)

    groups = load_groups(args.dataset)

    model, tokenizer, device = load_model(
        args.model
    )

    if args.experiment == "intervention":

        run_intervention(
            model=model,
            tokenizer=tokenizer,
            device=device,
            groups=groups,
            k=args.k,
            alpha=args.alpha,
            eps=args.eps,
            output_path=args.output
        )

    elif args.experiment == "energy":

        run_energy(
            model=model,
            tokenizer=tokenizer,
            device=device,
            groups=groups,
            k=args.k,
            eps=args.eps,
            output_path=args.output
        )


if __name__ == "__main__":
    main()