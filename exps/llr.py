"""Main file for last-layer retraining and deep feature reweighting."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from copy import deepcopy
from glob import glob
import os.path as osp
import pickle

# Imports Python packages.
from configargparse import Parser
from distutils.util import strtobool
import numpy as np

# Imports PyTorch packages.
import torch
from torch.utils.data import DataLoader, Subset
from pytorch_lightning import Trainer

# Imports milkshake packages.
from exps.finetune import *
from milkshake.args import add_input_args
from milkshake.main import main, load_weights


def reset_fc_hook(model):
    """Resets model classifier parameters."""

    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        try:
            model.model.fc.reset_parameters()
        except:
            pass

    try:
        for layer in model.model.classifier:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        try:
            model.model.classifier.reset_parameters()
        except:
            pass

def train_fc_only(model):
    """Freezes model parameters except for last layer."""

    for p in model.model.parameters():
        p.requires_grad = False

    try:
        for p in model.model.fc.parameters():
            p.requires_grad = True
    except:
        for p in model.model.classifier.parameters():
            p.requires_grad = True

def set_llr_args(args, train_type):
    """Sets args for last-layer retraining."""

    new_args = deepcopy(args)

    # Note that LLR is run for the same amount of epochs as ERM.
    new_args.ckpt_every_n_epochs = new_args.max_epochs + 1
    new_args.check_val_every_n_epoch = new_args.max_epochs
    new_args.lr = 1e-2
    new_args.lr_scheduler = "step"
    new_args.lr_steps = []
    new_args.optimizer = "sgd"
    new_args.weight_decay = 0

    if train_type == "llr":
        new_args.train_type = "llr"
        new_args.retrain_type = "group-unbalanced retraining"
    elif train_type == "dfr":
        new_args.train_type = "dfr"
        new_args.retrain_type = "group-balanced retraining"

    return new_args

def find_erm_weights(args):
    """Retrieves ERM weights from pickle file based on model config."""

    args.train_type = "erm"

    if not osp.isfile(args.results_pkl):
        raise ValueError(f"Results file {args.results_pkl} not found.")

    results = load_results(args)

    s = args.seed

    if args.model == "bert":
        v = args.bert_version
    elif args.model == "convnextv2":
        v = args.convnextv2_version
    elif args.model == "resnet":
        v = args.resnet_version

    p = args.split
    c = args.balance_erm
    if "mixture" in c:
        c += str(args.mixture_ratio)
    e = args.max_epochs

    wandb_version = results[s][v][p][c]["erm"][e]["version"]
    if not wandb_version:
        raise ValueError(f"Model version {wandb_version} not found.")

    # Finds model weights in wandb dir.
    fpath = "epoch=" + f"{e - 1:02d}" + "*"
    ckpt_path = osp.join(
        args.wandb_dir, "lightning_logs", wandb_version, "checkpoints", fpath)
    args.weights = glob(ckpt_path)[0]

def experiment(args, model_class, datamodule_class):
    """Runs main training and evaluation procedure."""

    args.no_test = True
    find_erm_weights(args)
    
    datamodule = datamodule_class(args)
    datamodule.setup()

    args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    model = model_class(args)
    model = load_weights(args, model)

    if args.retrain_type in ["llr", "both"]:
        # Performs LLR.
        new_args = set_llr_args(args, "llr")
        model.hparams.train_type = "llr" # Used for dumping results
        train_fc_only(model)
        main(new_args, model, datamodule_class, model_hooks=[reset_fc_hook])

    if args.retrain_type in ["dfr", "both"]:
        # Performs DFR.
        new_args = set_llr_args(args, "dfr")
        model.hparams.train_type = "dfr" # Used for dumping results
        train_fc_only(model)
        main(new_args, model, datamodule_class, model_hooks=[reset_fc_hook])

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
    parser.add("--heldout", default=True, type=lambda x: bool(strtobool(x)),
               help="Whether to perform LLR on a held-out set or the training set.")
    parser.add("--mixture_ratio", type=float, default=1,
               help="The largest acceptable class imbalance ratio for the mixture balancing strategy.")
    parser.add("--save_retrained_model", action="store_true",
               help="Whether to save the retrained model outputs.")
    parser.add("--split", choices=["combined", "train"], default="train",
               help="The split to train on; either the train set or the combined train and held-out set.")
    parser.add("--train_pct", default=100, type=int,
               help="The percentage of the train set to utilize (for ablations)")

    parser.add("--retrain_type", choices=["llr", "dfr", "both"], default="llr",
               help="Whether to perform LLR, DFR, or both.")

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
    args.results_pkl = f"{args.datamodule}_{args.model}.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
