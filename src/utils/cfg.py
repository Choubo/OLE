import os
import argparse
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default='./clipn_checkpoint.pt',
        help="pretrain weight path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='ViT-B-16',
        help="model type of CLIP encoder",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default='./outlier_label/nonus.npy',
        help="path of the outlier class labels",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default='exp_1',
        help="name of experiment",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=50.,
        help="scale of classifier",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.1,
        help="quantile of refinement",
    )
    parser.add_argument(
        "--n_prototypes",
        type=int,
        default=500,
        help="number of prototypes."
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args