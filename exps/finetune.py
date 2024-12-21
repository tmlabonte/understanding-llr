"""Main file for ERM finetuning under spurious correlations."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
import os.path as osp
import pickle

# Imports Python packages.
from configargparse import Parser
from distutils.util import strtobool
import numpy as np

# Imports PyTorch packages.
import torch
from pytorch_lightning import Trainer

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.civilcomments import CivilComments
from milkshake.datamodules.multinli import MultiNLI
from milkshake.datamodules.retrain import Retrain
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.main import main
from milkshake.models.bert import BERT
from milkshake.models.convnextv2 import ConvNeXtV2
from milkshake.models.resnet import ResNet
from milkshake.utils import to_np


HELDOUTS = [True, False]
METRICS = ["test_aa", "test_acc_by_class", "test_acc_by_group",
            "test_wca", "test_wga", "train_aa", "train_acc_by_class",
            "train_acc_by_group", "train_wca", "train_wga", "version"]
SEEDS = [1, 2, 3]
SPLITS = ["combined", "train"]
TRAIN_TYPES = ["erm", "llr", "dfr"]

# Defines class-balancing methods by dataset.
base_methods = ["none", "subsetting", "upsampling", "upweighting"]
def mixtures(ratios):
    return [f"mixture{j}" for j in ratios]
CLASS_BALANCING = {
    "celeba":        base_methods + mixtures([2., 4.]),
    "civilcomments": base_methods + mixtures([3., 5.]),
    "multinli":      base_methods,
    "waterbirds":    base_methods + mixtures([2.]),
}

# Defines training epochs by dataset and pretraining type.
EPOCHS = {
    "celeba": list(range(2, 21, 2)),
    "civilcomments": list(range(2, 21, 2)),
    "multinli": list(range(2, 21, 2)),
    "waterbirds": list(range(10, 101, 10)),
}

# Defines parameters for preset model sizes.
VERSIONS = {
    "bert": ["tiny", "mini", "small", "medium", "base"],
    "convnextv2": ["atto", "femto", "pico", "nano", "tiny", "base"],
    "resnet": [18, 34, 50, 101, 152],
}

class CelebARetrain(CelebA, Retrain):
    """DataModule for the CelebARetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CivilCommentsRetrain(CivilComments, Retrain):
    """DataModule for the CivilCommentsRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class MultiNLIRetrain(MultiNLI, Retrain):
    """DataModule for the MultiNLIRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
class CelebARetrain(CelebA, Retrain):
    """DataModule for the CelebARetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.balance_erm_type = args.balance_erm_type  # Save balance_erm_type from args

    def dataloader_for_retraining(self):
        # Logic to handle class vs. group balancing
        if self.retrain_type == "group-unbalanced retraining":
            return self.group_unbalanced_dataloader(self.balance_retrain)
        elif self.retrain_type == "group-balanced retraining":
            return self.group_balanced_dataloader(self.balance_retrain)
        else:  # For ERM training
            if self.balance_erm_type == "class":
                return self.group_unbalanced_dataloader(self.balance_erm)
            elif self.balance_erm_type == "group":
                return self.group_balanced_dataloader(self.balance_erm)

class MultiNLIRetrain(MultiNLI, Retrain):
    """DataModule for the MultiNLIRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.balance_erm_type = args.balance_erm_type  # Save balance_erm_type from args

    def dataloader_for_retraining(self):
        # Logic to handle class vs. group balancing
        if self.retrain_type == "group-unbalanced retraining":
            return self.group_unbalanced_dataloader(self.balance_retrain)
        elif self.retrain_type == "group-balanced retraining":
            return self.group_balanced_dataloader(self.balance_retrain)
        else:  # For ERM training
            if self.balance_erm_type == "class":
                return self.group_unbalanced_dataloader(self.balance_erm)
            elif self.balance_erm_type == "group":
                return self.group_balanced_dataloader(self.balance_erm)

class CivilCommentsRetrain(CivilComments, Retrain):
    """DataModule for the CivilCommentsRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.balance_erm_type = args.balance_erm_type  # Save balance_erm_type from args

    def dataloader_for_retraining(self):
        # Logic to handle class vs. group balancing
        if self.retrain_type == "group-unbalanced retraining":
            return self.group_unbalanced_dataloader(self.balance_retrain)
        elif self.retrain_type == "group-balanced retraining":
            return self.group_balanced_dataloader(self.balance_retrain)
        else:  # For ERM training
            if self.balance_erm_type == "class":
                return self.group_unbalanced_dataloader(self.balance_erm)
            elif self.balance_erm_type == "group":
                return self.group_balanced_dataloader(self.balance_erm)

class WaterbirdsRetrain(Waterbirds, Retrain):
    """DataModule for the WaterbirdsRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.balance_erm_type = args.balance_erm_type  # Save balance_erm_type from args

    def dataloader_for_retraining(self):
        # Example logic to handle class vs. group balancing
        if self.retrain_type == "group-unbalanced retraining":
            return self.group_unbalanced_dataloader(self.balance_retrain)
        elif self.retrain_type == "group-balanced retraining":
            return self.group_balanced_dataloader(self.balance_retrain)
        else:  # For ERM training
            if self.balance_erm_type == "class":
                return self.group_unbalanced_dataloader(self.balance_erm)
            elif self.balance_erm_type == "group":
                return self.group_balanced_dataloader(self.balance_erm)

def log_results_helper(
    args,
    epoch,
    version,
    validation_step_outputs,
    weight_aa_by_proportion=False,
):
    """Exports validation accuracies to dict."""

    results = {m: {} for m in METRICS}

    train_group_proportions = []
    for idx, step_output in enumerate(validation_step_outputs):
        prefix = "train_" if idx == 0 else "test_"

        def collate_and_sum(name):
            results = [result[name] for result in step_output]
            return torch.sum(torch.stack(results), 0)

        # Computes average and group accuracies. For Waterbirds ONLY,
        # compute the test average accuracy as a weighted sum of the group
        # accuracies with respect to the training distribution proportions.
        total_by_group = collate_and_sum("total_by_group")
        total_by_class = collate_and_sum("total_by_class")  # Assuming you have this key in step_output
        if idx == 0:
            train_group_proportions = total_by_group / sum(total_by_group)
        correct_by_group = collate_and_sum("correct_by_group")
        correct_by_class = collate_and_sum("correct_by_class")  # Assuming you have this key in step_output

        acc_by_group = correct_by_group / total_by_group
        acc_by_class = correct_by_class / total_by_class  # Compute accuracy by class

        worst_group_acc = min(acc_by_group).item()
        worst_class_acc = min(acc_by_class).item()  # Compute the worst class accuracy

        if idx == 1 and weight_aa_by_proportion:
            average_acc = correct_by_group / total_by_group
            average_acc = sum(average_acc * train_group_proportions).item()
        else:
            average_acc = (sum(correct_by_group) / sum(total_by_group)).item()

        # Adds metrics to results dict.
        results[prefix + "aa"] = average_acc
        results[prefix + "wga"] = worst_group_acc
        results[prefix + "wca"] = worst_class_acc  # Add worst class accuracy to results
        results[prefix + "acc_by_group"] = list(to_np(acc_by_group))
        results[prefix + "acc_by_class"] = list(to_np(acc_by_class))  # Optionally add class accuracies to results

    results["version"] = version

    return results

def log_results(
    args,
    epoch,
    version,
    validation_step_outputs,
    weight_aa_by_proportion=False,
):
    """Exports validation accuracies to dict and dumps to disk.

    This setup makes it easy to overwrite dump_results in another exp
    if one wants to add a new key to the results dict.
    """

    results = log_results_helper(
        args,
        epoch,
        version,
        validation_step_outputs,
        weight_aa_by_proportion=weight_aa_by_proportion,
    )
    dump_results(args, epoch, results)

class BERTWithLogging(BERT):
    """Quick and dirty extension of BERT with metrics exported to dict."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        log_results(
            self.hparams,
            self.current_epoch + 1,
            self.trainer.logger.version,
            validation_step_outputs,
        )

class ConvNeXtV2WithLogging(ConvNeXtV2):
    """Quick and dirty extension of ConvNeXtV2 with metrics exported to dict."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        log_results(
            self.hparams,
            self.current_epoch + 1,
            self.trainer.logger.version,
            validation_step_outputs,
            weight_aa_by_proportion=self.hparams.datamodule == "waterbirds",
        )

class ResNetWithLogging(ResNet):
    """Quick and dirty extension of ResNet with metrics exported to dict."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        log_results(
            self.hparams,
            self.current_epoch + 1,
            self.trainer.logger.version, 
            validation_step_outputs,
            weight_aa_by_proportion=self.hparams.datamodule == "waterbirds",
        )

def load_results(args):
    """Loads results file or creates it if it does not exist."""

    if osp.isfile(args.results_pkl):
        with open(args.results_pkl, "rb") as f:
            results = pickle.load(f)
    else: 
        results = {}
        for s in SEEDS:
            results[s] = {}
            for v in VERSIONS[args.model]:
                results[s][v] = {}
                for p in SPLITS:
                    results[s][v][p] = {}
                    for c in CLASS_BALANCING[args.datamodule]:
                        results[s][v][p][c] = {}
                        for t in TRAIN_TYPES:
                            results[s][v][p][c][t] = {}
                            epochs = EPOCHS[args.datamodule]
                            if t == "erm":
                                for e in epochs:
                                    results[s][v][p][c][t][e] = {}
                                    for m in METRICS:
                                        results[s][v][p][c][t][e][m] = {}
                            else:
                                for h in HELDOUTS:
                                    results[s][v][p][c][t][h] = {}
                                    for d in CLASS_BALANCING[args.datamodule]:
                                        results[s][v][p][c][t][h][d] = {}
                                        results[s][v][p][c][t][h][d][epochs[-1]] = {}
                                        for m in METRICS:
                                            results[s][v][p][c][t][h][d][epochs[-1]][m] = {}

        with open(args.results_pkl, "wb") as f:
            pickle.dump(results, f)

    return results

def dump_results(args, curr_epoch, curr_results):
    """Saves metrics in curr_results to the results file."""

    s = args.seed

    if args.model == "bert":
        v = args.bert_version
    elif args.model == "convnextv2":
        v = args.convnextv2_version
    elif args.model == "resnet":
        v = args.resnet_version
        
    p = args.split
    c = args.balance_erm
    d = args.balance_retrain
    if "mixture" in c:
        c += str(args.mixture_ratio)
    if "mixture" in d:
        d += str(args.mixture_ratio)
    t = args.train_type
    h = args.heldout
    e = curr_epoch

    # VERY important to load results right before dumping. Otherwise, we may
    # overwrite results saved by different experiments.
    results = load_results(args)
    if t == "erm":
        for m in METRICS:
            if m in curr_results:
                results[s][v][p][c][t][e][m] = curr_results[m]
    else:
        for m in METRICS:
            if m in curr_results:
                results[s][v][p][c][t][h][d][e][m] = curr_results[m]
    
    with open(args.results_pkl, "wb") as f:
        pickle.dump(results, f)

def experiment(args, model_class, datamodule_class):
    """Runs main training and evaluation procedure."""

    args.no_test = True

    # Sets class weights for loss-based class-balancing.
    # MultiNLI is class-balanced a priori, so we do not include it here.

    if args.balance_erm == "upweighting":
        if args.balance_erm_type == "class":
            # Class-based weights
            if args.datamodule == "celeba":
                args.class_weights = [1, 5.71]
            elif args.datamodule == "civilcomments":
                args.class_weights = [1, 7.85]
            elif args.datamodule == "waterbirds":
                args.class_weights = [1, 3.31]
        elif args.balance_erm_type == "group":
            # Group-based weights (hypothetical example, adjust based on dataset needs)
            if args.datamodule == "celeba":
                args.group_weights = [0.4, 0.6]  
            elif args.datamodule == "civilcomments":
                args.group_weights = [0.3, 0.7]
            elif args.datamodule == "waterbirds":
                args.group_weights = [0.5, 0.5]
    # Creates results dict if it does not exist.
    if not osp.isfile(args.results_pkl):
        load_results(args)

    main(args, model_class, datamodule_class)

def weighted_loss(logits, targets, group_labels, group_weights=None, balance_erm_type="class"):
    loss = F.cross_entropy(logits, targets, reduction="none")
    if balance_erm_type == "group" and group_weights is not None:
        sample_weights = group_weights[group_labels]
        loss = loss * sample_weights
    return loss.mean()


if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # Arguments imported from retrain.py.
    # Extend the argument parser to include balance_erm_type
    parser.add("--balance_erm_type", choices=["class", "group"], default="class",
            help="Specify whether class or group balancing is used during ERM training.")
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
    args.results_pkl = f"{args.datamodule}_{args.model}.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
