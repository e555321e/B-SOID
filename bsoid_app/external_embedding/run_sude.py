import argparse
import numpy as np

from sude import sude


def parse_args():
    parser = argparse.ArgumentParser(description="Run SUDE embedding for B-SOiD.")
    parser.add_argument("--input_npy", required=True, help="Path to input feature .npy file")
    parser.add_argument("--output_npy", required=True, help="Path to output embedding .npy file")
    parser.add_argument("--no_dims", type=int, default=3, help="Embedding dimensions")
    parser.add_argument("--k1", type=int, default=20, help="Number of nearest neighbors")
    parser.add_argument("--normalize", type=int, default=1, help="Whether to normalize input (1/0)")
    parser.add_argument("--large", type=int, default=0, help="Use large mode (1/0)")
    parser.add_argument("--initialize", type=str, default="le", help="Initialization method")
    parser.add_argument("--agg_coef", type=float, default=1.2, help="Aggregation coefficient")
    parser.add_argument("--T_epoch", type=int, default=50, help="Max epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    x = np.load(args.input_npy)
    y = sude(
        x,
        no_dims=args.no_dims,
        k1=args.k1,
        normalize=bool(args.normalize),
        large=bool(args.large),
        initialize=args.initialize,
        agg_coef=args.agg_coef,
        T_epoch=args.T_epoch,
    )
    np.save(args.output_npy, y)


if __name__ == "__main__":
    main()