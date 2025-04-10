"""Main file for sample-ordering experiments."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
import os.path as osp
import pickle

# Imports Python packages.
from configargparse import Parser
import numpy as np
from numpy.random import default_rng

# Imports PyTorch packages.
import torch
from pytorch_lightning import Trainer

# Imports milkshake packages.
from exps.finetune import *
from milkshake.args import add_input_args
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.dataset import Subset
from milkshake.main import main
from milkshake.utils import to_np

ORDER_RATIOS = [0.0, 0.5, 1.0]

def log_results(
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
            #self.current_epoch + 1,
            (self.current_epoch + 1) * 10,
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
            for r in ORDER_RATIOS:
                results[s][r] = {}
                for v in VERSIONS[args.model]:
                    results[s][r][v] = {}
                    for c in CLASS_BALANCING[args.datamodule]:
                        results[s][r][v][c] = {}
                        for e in EPOCHS[args.datamodule]:
                            results[s][r][v][c][e] = {}
                            for m in METRICS:
                                results[s][r][v][c][e][m] = {}

        with open(args.results_pkl, "wb") as f:
            pickle.dump(results, f)

    return results

def dump_results(args, curr_epoch, curr_results):
    """Saves metrics in curr_results to the results file."""

    s = args.seed
    r = args.order_ratio

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
    e = curr_epoch

    # VERY important to load results right before dumping. Otherwise, we may
    # overwrite results saved by different experiments.
    results = load_results(args)
    for m in METRICS:
        if m in curr_results:
            results[s][r][v][c][e][m] = curr_results[m]
    
    with open(args.results_pkl, "wb") as f:
        pickle.dump(results, f)

class WaterbirdsRetrainOrder(WaterbirdsRetrain):
    """DataModule for manipulating sample order during training."""
    
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Number between 0 and 1 controlling order of maj/min class samples.
        # 0.0: All class 0 first
        # 0.5: Uniformly random
        # 1.0: All class 1 first
        self.order_ratio = args.order_ratio

    def order_helper(self, indices, targets):
        """Sets sample order as desired. Only works with two classes for now."""

        indices, targets = super()._shuffle_in_unison(indices, targets)
        
        class0 = indices[targets == 0]
        class1 = indices[targets == 1]

        if self.order_ratio == 0.0:
            return np.concatenate((class0, class1))
        elif self.order_ratio == 0.5:
            return indices
        elif self.order_ratio == 1.0:
            return np.concatenate((class1, class0))
        else:
            raise NotImplementedError()

        return subset

    def train_dataloader(self):
        # Makes class-balanced subset if specified.
        self.balanced_sampler = False
        self.shuffle = False
        if self.balance_erm == "subsetting":
            self.dataset_train = self.make_balanced_subset(self.dataset_train)

        indices = self.dataset_train.train_indices
        targets = self.dataset_train.targets[indices][:, 0] # Removes group dimension

        subset = self.order_helper(indices, targets)

        self.dataset_train = Subset(self.dataset_train, subset) # Just a re-ordering
        return DataModule.train_dataloader(self)

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

    main(args, model, datamodule_class)
    
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

    parser.add("--order_ratio", type=float, default=0.5)

    datamodules = {
        #"celeba": CelebARetrainOrder,
        #"civilcomments": CivilCommentsRetrainOrder,
        "waterbirds": WaterbirdsRetrainOrder,
    }
    models = {
        "bert": BERTWithLogging,
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": ResNetWithLogging,
    }

    args = parser.parse_args()
    args.train_type = "erm"
    args.results_pkl = f"{args.datamodule}_{args.model}_order.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
