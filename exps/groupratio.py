"""DataModule for group ratio experiments."""

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
import numpy as np

# Imports milkshake packages.
from exps.finetune import *
from exps.llr import *
from milkshake.args import add_input_args
from milkshake.datamodules.retrain import Retrain
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.main import main, load_weights


class GroupRatio(Retrain):
    def __init__(self, args, maj_data_pct, *xargs):
        super().__init__(args, *xargs)
        self.maj_data_pct = maj_data_pct

    # TODO: Change eval dataset

    def make_group_ratio_subset(self, dataset):
        # Uses groups 1 and 2 as minority groups.
        indices = dataset.train_indices
        groups = dataset.targets[indices][:, 1] # Removes class dimension
        indices, groups = self._shuffle_in_unison(indices, groups)
        num = len(dataset.groups)

        desired = np.unique(targets, return_counts=True)[1]
        desired[0] *= self.maj_data_pct / 100
        desired[3] *= self.maj_data_pct / 100

        subset = []
        counts = [0] * len_groups
        for idx, group in zip(indices, groups):
            if counts[group] < desired:
                subset.append(idx)
                counts[group] += 1
        print(f"Data subset: {counts}")

        return subset

    def group_unbalanced_dataloader(self, balance):
        # Only meant to use subsetting or none.
        if balance == "subsetting":
            self.dataset_train = self.make_balanced_subset(self.dataset_train)
        self.dataset_train = self.make_group_ratio_subset(self.dataset_train)

        return super().train_dataloader()

class WaterbirdsGroupRatio(Waterbirds, GroupRatio):
    """DataModule for the WaterbirdsGroupRatio dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

def experiment(args, model_class, datamodule_class):
    """Runs main training and evaluation procedure."""

    # Only meant to use subsetting or none.
    if args.balance_erm not in ["none", "subsetting"] \
        or args.balance_retrain not in ["none", "subsetting"]:
        return NotImplementedError()

    args.no_test = True

    # Creates results dict if it does not exist.
    if not osp.isfile(args.results_pkl):
        load_results(args)

    main(args, model_class, datamodule_class)

    # TODO: Add LLR


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
    parser.add("--num_classes", default=None, type=int,
               help="The number of outputs produced by the model.")

    parser.add("--maj_data_pct", type=int)

    datamodules = {
        "waterbirds": WaterbirdsGroupRatio,
    }
    models = {
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": ResNetWithLogging,
    }

    args = parser.parse_args()
    args.train_type = "erm"
    args.results_pkl = f"{args.datamodule}_{args.model}_groupbalance.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
