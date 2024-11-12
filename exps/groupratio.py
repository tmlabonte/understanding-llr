"""DataModule and experiment code for group ratio experiments."""

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

# Imports PyTorch packages.
from torch.utils.data import WeightedRandomSampler

# Imports milkshake packages.
from exps.finetune import *
from exps.llr import *
from milkshake.args import add_input_args
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.dataset import Subset
from milkshake.datamodules.retrain import Retrain
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.main import main, load_weights


GROUP_RATIOS = {
    "celeba": np.arange(0.05, 1.05, 0.05), # About 0.061 is standard
    "waterbirds": np.arange(0.05, 1.05, 0.05), # About 0.053 is standard
}


class GroupRatio(Retrain):
    def __init__(self, args, *xargs):
        super().__init__(args, *xargs)
        self.erm_group_ratio = args.erm_group_ratio
        self.retrain_group_ratio = args.retrain_group_ratio
        self.orig_datamodule = args.datamodule

    def make_group_ratio_sampler(self, dataset, group_ratio):
        """Returns a WeightedRandomSampler with the specified group ratio.

        Class-balancing is always used so that the probability of sampling
        a point from each class is equal. However, within each class, if there
        is a minority group then it will be sampled according to the specified
        group ratio. If there is no minority group in the class, each group
        will be sampled with equal probability.
        """

        indices = dataset.train_indices
        classes, groups = dataset.targets[indices].T
        group_totals = np.unique(groups, return_counts=True)[1]

        minority_pct = group_ratio / (group_ratio + 1)
        if self.orig_datamodule == "celeba":
            # Ex. group_ratio 50 => group weights [0.25, 0.25, 0.333, 0.167]
            weights_by_group = [
                0.5,
                0.5,
                1 - minority_pct,
                minority_pct,
            ]
        elif self.orig_datamodule == "waterbirds":
            # Ex. group_ratio 50 => group weights [0.333, 0.167, 0.167, 0.333]
            weights_by_group = [
                1 - minority_pct,
                minority_pct,
                minority_pct,
                1 - minority_pct,
            ]

        weights = [0] * len(groups)
        for j, group in enumerate(groups):
            # Normalizes weights of each individual datum so that the sum
            # of weights for each group is weights_by_group.
            weights[j] = weights_by_group[group] / (2 * group_totals[group])
        print(f"Weights by group: {weights_by_group}")

        return WeightedRandomSampler(weights, len(weights))

    def make_group_ratio_subset(self, dataset, group_ratio):
        """Returns a Subset with the specified group ratio.

        Class-balancing is always used so that the number of data from each
        class is equal. However, within each class, if there is a minority group
        then it will be subsetted according to the specified group ratio. If
        there is no minority group in the class, each group will have the same
        number of data.
        """

        indices = dataset.train_indices
        classes, groups = dataset.targets[indices].T
        group_totals = np.unique(groups, return_counts=True)[1]
        class_totals = np.full(self.num_classes, 2 * group_totals.min())

        minority_pct = group_ratio / (group_ratio + 1)
        if self.orig_datamodule == "celeba":
            nums = [
                class_totals[0] / 2,
                class_totals[0] / 2,
                class_totals[1] - class_totals[1] * minority_pct,
                class_totals[1] * minority_pct,
            ]
        if self.orig_datamodule == "waterbirds":
            nums = [
                class_totals[0] - class_totals[0] * minority_pct,
                class_totals[0] * minority_pct,
                class_totals[1] * minority_pct,
                class_totals[1] - class_totals[1] * minority_pct
            ]

        subset = []
        counts = [0] * len(group_totals)
        indices, groups = self._shuffle_in_unison(indices, groups)
        for idx, group in zip(indices, groups):
            if counts[group] < nums[group]:
                subset.append(idx)
                counts[group] += 1
        print(f"Subsets by group: {counts}")

        return Subset(dataset, subset)

    def _initialize_datasets_no_aug(self, dataset_val):
        """Initializes datasets with no augmentations (for evaluation)."""
        
        super()._initialize_datasets_no_aug(dataset_val)

        if self.retrain_type == "erm":
            if self.balance_erm == "subsetting":
                self.dataset_train_no_aug = self.make_group_ratio_subset(
                    self.dataset_train_no_aug,
                    self.erm_group_ratio,
                )
        else:
            if self.balance_retrain == "subsetting":
                self.dataset_retrain_no_aug = self.make_group_ratio_subset(
                    self.dataset_retrain_no_aug,
                    self.retrain_group_ratio,
                )

    def group_unbalanced_dataloader(self, balance):
        if self.retrain_type == "erm":
            group_ratio = self.erm_group_ratio
        else:
            group_ratio = self.retrain_group_ratio

        if balance == "upsampling":
            sampler = self.make_group_ratio_sampler(
                    self.dataset_train, group_ratio)
            return DataModule._data_loader(
                    self, self.dataset_train, sampler=sampler)
        elif balance == "subsetting":
            self.dataset_train = self.make_group_ratio_subset(
                    self.dataset_train, group_ratio)
            return DataModule.train_dataloader(self)
        else:
            raise NotImplementedError()

class WaterbirdsGroupRatio(Waterbirds, GroupRatio):
    """DataModule for the WaterbirdsGroupRatio dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CelebAGroupRatio(CelebA, GroupRatio):
    """DataModule for the CelebAGroupRatio dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

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
                for c in CLASS_BALANCING[args.datamodule]:
                    results[s][v][c] = {}
                    for g in GROUP_RATIOS[args.datamodule]:
                        results[s][v][c][g] = {}
                        for t in TRAIN_TYPES:
                            results[s][v][c][g][t] = {}
                            epochs = EPOCHS[args.datamodule]
                            if t == "erm":
                                for e in epochs:
                                    results[s][v][c][g][t][e] = {}
                                    for m in METRICS:
                                        results[s][v][c][g][t][e][m] = {}
                            else:
                                for d in CLASS_BALANCING[args.datamodule]:
                                    results[s][v][c][g][t][d] = {}
                                    for f in GROUP_RATIOS[args.datamodule]:
                                        results[s][v][c][g][t][d][f] = {}
                                        results[s][v][c][g][t][d][f][epochs[-1]] = {}
                                        for m in METRICS:
                                            results[s][v][c][g][t][d][f][epochs[-1]][m] = {}

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
        
    c = args.balance_erm
    d = args.balance_retrain
    if "mixture" in c:
        c += str(args.mixture_ratio)
    if "mixture" in d:
        d += str(args.mixture_ratio)
    g = args.erm_group_ratio
    f = args.retrain_group_ratio
    t = args.train_type
    e = curr_epoch

    # VERY important to load results right before dumping. Otherwise, we may
    # overwrite results saved by different experiments.
    results = load_results(args)
    if t == "erm":
        for m in METRICS:
            if m in curr_results:
                results[s][v][c][g][t][e][m] = curr_results[m]
    else:
        for m in METRICS:
            if m in curr_results:
                results[s][v][c][g][t][d][f][e][m] = curr_results[m]
    
    with open(args.results_pkl, "wb") as f:
        pickle.dump(results, f)

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

def find_erm_weights(args):
    """Retrieves ERM weights from pickle file based on model config."""

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

    c = args.balance_erm
    d = args.balance_retrain
    if "mixture" in c:
        c += str(args.mixture_ratio)
    if "mixture" in d:
        d += str(args.mixture_ratio)
    g = args.erm_group_ratio
    e = args.max_epochs

    wandb_version = results[s][v][c][g]["erm"][e]["version"]
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

    # Creates results dict if it does not exist.
    if not osp.isfile(args.results_pkl):
        load_results(args)

    if args.train_type == "erm":
        main(args, model_class, datamodule_class)
    elif args.train_type == "llr":
        find_erm_weights(args)
        datamodule = datamodule_class(args)
        datamodule.setup()

        args.num_classes = datamodule.num_classes
        args.num_groups = datamodule.num_groups

        model = model_class(args)
        model = load_weights(args, model)

        # Performs LLR.
        new_args = set_llr_args(args, "llr")
        train_fc_only(model)
        model, _, _ = main(
            new_args, model, datamodule_class, model_hooks=[reset_fc_hook])

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

    parser.add("--erm_group_ratio", default=1, type=float,
               help="The ratio of minority to majority group data in ERM.")
    parser.add("--retrain_group_ratio", default=1, type=float,
               help="The ratio of minority to majority group data in LLR.")
    parser.add("--train_type", choices=["erm", "llr"], default="erm",
               help="Whether to perform ERM or LLR.")

    datamodules = {
        "waterbirds": WaterbirdsGroupRatio,
        "celeba": CelebAGroupRatio,
    }
    models = {
        # "convnextv2": ConvNeXtV2WithLogging,
        "resnet": ResNetWithLogging,
    }

    args = parser.parse_args()
    args.retrain_type = args.train_type
    args.results_pkl = f"{args.datamodule}_{args.model}_groupratio.pkl"

    # Checks legitimacy of group ratio arguments.
    if args.erm_group_ratio > 1 or args.retrain_group_ratio > 1:
        raise ValueError("Cannot have more minority group data than majority group data.")

    experiment(args, models[args.model], datamodules[args.datamodule])
