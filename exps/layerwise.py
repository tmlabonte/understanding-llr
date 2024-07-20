"""Main file for layer-wise model training."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
import os.path as osp
import pickle

# Imports Python packages.
from configargparse import Parser
import numpy as np

# Imports PyTorch packages.
import torch
from pytorch_lightning import Trainer

# Imports milkshake packages.
from exps.finetune import *
from exps.llr import *
from milkshake.args import add_input_args
from milkshake.main import main
from milkshake.utils import to_np


def experiment(args, model_class, datamodule_class):
    """Runs main training and evaluation procedure."""

    args.no_test = True

    # Sets class weights for loss-based class-balancing.
    # MultiNLI is class-balanced a priori, so we do not include it here.
    if args.balance_erm == "upweighting":
        if args.datamodule == "celeba":
            args.class_weights = [1, 5.71]
        elif args.datamodule == "civilcomments":
            args.class_weights = [1, 7.85]
        elif args.datamodule == "waterbirds":
            args.class_weights = [1, 3.31]

    # Creates results dict if it does not exist.
    if not osp.isfile(args.results_pkl):
        load_results(args)

    args.num_classes = 2 if not args.datamodule == "multinli" else 3
    args.num_groups = args.num_classes * 2
    model = model_class(args)

    for p in model.model.fc.parameters():
        p.requires_grad = False
    model.hparams.train_type = "erm"
    model, _, _ = main(args, model, datamodule_class)

    new_args = set_llr_args(args, "llr")
    train_fc_only(model)
    model.hparams.train_type = "llr"
    main(new_args, model, datamodule_class)
    
if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Arguments imported from retrain.py.
    parser.add("--balance_erm", choices=["mixture", "none", "subsetting", "upsampling", "upweighting"], default="none",
               help="Which type of class-balancing to perform during ERM training.")
    parser.add("--balance_retrain", choices=["mixture", "none", "subsetting", "upsampling", "upweighting"], default="none",
               help="Which type of class-balancing to perform during retraining.")
    parser.add("--mixture_ratio", type=float, default=1,
               help="The largest acceptable class imbalance ratio for the mixture balancing strategy.")
    parser.add("--save_retrained_model", action="store_true",
               help="Whether to save the retrained model outputs.")
    parser.add("--split", choices=["combined", "train"], default="train",
               help="The split to train on; either the train set or the combined train and held-out set.")
    parser.add("--train_pct", default=100, type=int,
               help="The percentage of the train set to utilize (for ablations)")

    datamodules = {
        "celeba": CelebARetrain,
        "civilcomments": CivilCommentsRetrain,
        "multinli": MultiNLIRetrain,
        "waterbirds": WaterbirdsRetrain,
    }
    models = {
        "bert": BERTWithLogging,
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": ResNetWithLogging,
    }

    args = parser.parse_args()
    args.train_type = "erm"
    args.results_pkl = f"{args.datamodule}_{args.model}_layerwise.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
