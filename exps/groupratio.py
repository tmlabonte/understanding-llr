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
    "waterbirds": np.arange(5, 105, 5), # About 5.3 is standard
    "celeba": np.arange(5, 105, 5), # About 6.1 is standard
}

class GroupRatio(Retrain):
    def __init__(self, args, *xargs):
        super().__init__(args, *xargs)
        self.erm_group_ratio = args.erm_group_ratio
        self.retrain_group_ratio = args.retrain_group_ratio
        self.orig_datamodule = args.datamodule

    def make_group_ratio_subset(self, dataset):
        if self.retrain_type == "erm":
            group_ratio = self.erm_group_ratio
            balance = self.balance_erm
        else:
            group_ratio = self.retrain_group_ratio
            balance = self.balance_retrain

        indices = dataset.train_indices
        groups = dataset.targets[indices][:, 1] # Removes class dimension
        indices, groups = self._shuffle_in_unison(indices, groups)
        num = len(dataset.groups)

        total_by_group = np.unique(groups, return_counts=True)[1]
        if balance == "none":
            total_by_class = [
                2 * min(total_by_group[:2]),
                2 * min(total_by_group[2:]),
            ]
        elif balance == "subsetting":
            total_by_class = [
                2 * min(total_by_group),
                2 * min(total_by_group),
            ]

        if self.orig_datamodule == "waterbirds":
            nums = [
                (group_ratio * total_by_class[0]) / (group_ratio + 100),
                (group_ratio * total_by_class[1]) / (group_ratio + 100),
            ]
            desired = [
                total_by_class[0] - nums[0],
                nums[0],
                nums[1],
                total_by_class[1] - nums[1],
            ]
        elif self.orig_datamodule == "celeba":
            nums = [
                total_by_class[0] / 2,
                (group_ratio * total_by_class[1]) / (group_ratio + 100),
            ]
            desired = [
                nums[0],
                nums[0],
                total_by_class[1] - nums[1],
                nums[1],
            ]

        subset = []
        counts = [0] * num
        for idx, group in zip(indices, groups):
            if counts[group] < desired[group]:
                subset.append(idx)
                counts[group] += 1
        print(f"Data subset: {counts}")

        return Subset(dataset, subset)

    def _initialize_datasets_no_aug(self, dataset_val):
        """Initializes datasets with no augmentations (for evaluation)."""
        
        super()._initialize_datasets_no_aug(dataset_val)
        if self.retrain_type == "erm":
            self.dataset_train_no_aug = self.make_group_ratio_subset(self.dataset_train_no_aug)
        else:
            self.dataset_retrain_no_aug = self.make_group_ratio_subset(self.dataset_retrain_no_aug)

    def group_unbalanced_dataloader(self, balance):
        # Only meant to use subsetting or none.
        self.dataset_train = self.make_group_ratio_subset(self.dataset_train)

        return DataModule.train_dataloader(self)

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

    # Only meant to use subsetting or none.
    if args.balance_erm not in ["none", "subsetting"] \
        or args.balance_retrain not in ["none", "subsetting"]:
        return NotImplementedError()

    args.no_test = True

    # Creates results dict if it does not exist.
    if not osp.isfile(args.results_pkl):
        load_results(args)

    if args.train_type == "erm":
        main(args, model_class, datamodule_class)
    elif args.train_type == "llr":
        find_erm_weights(args)
        print(args)
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

    parser.add("--erm_group_ratio", default=100, type=int,
               help="The percent of minority group data to include in ERM.")
    parser.add("--retrain_group_ratio", default=100, type=int,
               help="The percent of minority group data to include in LLR.")
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
    experiment(args, models[args.model], datamodules[args.datamodule])
