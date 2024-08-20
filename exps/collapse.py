"""Main file for ERM neural collapse."""

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


METRICS = ["test_aa", "test_acc_by_class", "test_acc_by_group",
            "test_wca", "test_wga", "train_aa", "train_acc_by_class",
            "train_acc_by_group", "train_wca", "train_wga", "version",
            "global_cov", "inter_class_cov", "intra_class_cov", 
            "inter_group_cov", "intra_group_cov", "class_trace", "group_trace"]
SEEDS = [1, 2, 3]
TRAIN_TYPES = ["erm", "llr", "dfr"]

# Defines class-balancing methods by dataset.
base_methods = ["none", "subsetting", "upsampling", "upweighting"]
def mixtures(ratios):
    return [f"mixture{j}" for j in ratios]
CLASS_BALANCING = {
    "celeba":        base_methods + mixtures([2., 4.]),
    "civilcomments": base_methods + mixtures([3., 5.]),
    "multinli":      ["none"],
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

class WaterbirdsRetrain(Waterbirds, Retrain):
    """DataModule for the WaterbirdsRetrain dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)


def get_vectorized_features(targets, features, num_classes, num_groups):
    """Gets vectorized features for each group, class, and training split.

    Args:
        targets: A torch.Tensor of classification targets.
        features: A torch.Tensor of model features.
        num_classes: The total number of classes.
        num_groups: The total number of groups.

    Returns:
        A dictionary of vectorized features broken down by class and group.
    """

    # Splits apart targets and groups if necessary.
    groups = None
    if num_groups:
        groups = targets[:, 1]
        targets = targets[:, 0]

    # Vectorizes the batch features.
    batch_size = features.size(0)
    batch_vectorized_features = features.view(batch_size, -1)#.to(torch.float16)

    # Collates class features.
    features_by_class = {}
    for j in range(num_classes):
        class_features = batch_vectorized_features[targets == j]
        if j in features_by_class:
            features_by_class[j] = torch.concat((features_by_class[j], class_features))
        else:
            features_by_class[j] = class_features

    # Collates group features.
    if num_groups:
        features_by_group = {}
        for j in range(num_groups):
            group_features = batch_vectorized_features[groups == j]
            if j in features_by_group:
                features_by_group[j] = torch.concat((features_by_group[j], group_features))
            else:
                features_by_group[j] = group_features

    vectorized_features = {
        "features_by_class": features_by_class,
        "features_by_group": features_by_group,
        "features": batch_vectorized_features
    }

    return vectorized_features

class FeatureCollapseResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        self.features = None
        
    @torch.no_grad()
    def get_global_feature_info(self):
        """Gets number of samples and feature means.

        Returns:
            num_samples: The number of samples in the training dataset.
            global_mean: The mean of all training sample features.
            class_means: The means of the features of each class.
            group_means: The means of the features of each group.
        """

        num_samples = 0
        num_samples_per_class = [0] * self.hparams.num_classes
        num_samples_per_group = [0] * self.hparams.num_groups
        global_mean = 0
        class_means = [0] * self.hparams.num_classes
        group_means = [0] * self.hparams.num_groups
        for idx, batch in enumerate(self.trainer.train_dataloader):
            batch[0] = batch[0].cuda()
            batch[1] = batch[1].cuda()
            result = self.step(batch, idx)

            features = get_vectorized_features(
                result["targets"],
                self.features,
                self.hparams.num_classes,
                self.hparams.num_groups,
            )

            num_samples += len(batch[0])
            global_mean += torch.sum(features["features"], dim=0)

            for j in range(self.hparams.num_classes):
                num_samples_per_class[j] += sum([1 for t in result["targets"][:, 0] if t == j])
                class_means[j] += torch.sum(features["features_by_class"][j], dim=0)

            for j in range(self.hparams.num_groups):
                num_samples_per_group[j] += sum([1 for t in result["targets"][:, 1] if t == j])
                group_means[j] += torch.sum(features["features_by_group"][j], dim=0)

        global_mean /= num_samples

        for j in range(self.hparams.num_classes):
            class_means[j] /= num_samples_per_class[j]
        for j in range(self.hparams.num_groups):
            group_means[j] /= num_samples_per_group[j]
        class_means = torch.stack(class_means)
        group_means = torch.stack(group_means)

        return num_samples, global_mean, class_means, group_means

    @torch.no_grad()
    def compute_covariance_matrices(self, num_samples, global_mean, mode_means, mode=None):
        """Performs computation of covariance matrices for feature collapse.

        TODO: Clean this up, is "mode" the best way or make multiple methods?

        Args:
            num_samples: The number of samples in the training dataset.
            global_mean: The mean of all training sample features.
            mode_means: Either the class means or group means.
            mode: Either "total", "class", or "group".

        Returns:
            cov_norm: The Frobenius norm of the covariance matrix.
            trace: The trace of the inter/intra matmul (signal-to-noise ratio).
        """

        if mode == "total":
            total_cov = 0
        elif mode == "class":
            inter_class_cov = 0
            intra_class_cov = 0
        elif mode == "group":
            inter_group_cov = 0
            intra_group_cov = 0

        for idx, batch in enumerate(self.trainer.train_dataloader):
            batch[0] = batch[0].cuda()
            batch[1] = batch[1].cuda()
            result = self.step(batch, idx)

            # TODO: Perhaps compute only the features we need based on mode.
            features = get_vectorized_features(
                result["targets"],
                self.features,
                self.hparams.num_classes,
                self.hparams.num_groups,
            )

            if mode == "total":
                features = features["features"] - global_mean
                for sample_features in features:
                    total_cov += torch.outer(sample_features, sample_features)
            elif mode == "class":
                class_features = []
                for j in range(self.hparams.num_classes):
                    class_features.append(features["features_by_class"][j] - mode_means[j])
                class_features = torch.cat(class_features)

                for sample_features in class_features:
                    intra_class_cov += torch.outer(sample_features, sample_features)
            elif mode == "group":
                group_features = []
                for j in range(self.hparams.num_groups):
                    group_features.append(features["features_by_group"][j] - mode_means[j])
                group_features = torch.cat(group_features)

                for sample_features in group_features:
                    intra_group_cov += torch.outer(sample_features, sample_features)

        if mode == "total":
            total_cov /= num_samples
            total_cov_norm = torch.frobenius_norm(total_cov)
            return total_cov_norm
        elif mode == "class":
            centered_class_means = mode_means - global_mean
            for class_mean in centered_class_means:
                inter_class_cov += torch.outer(class_mean, class_mean)

            inter_class_cov /= self.hparams.num_classes
            intra_class_cov /= num_samples

            inter_class_cov_norm = torch.frobenius_norm(inter_class_cov)
            intra_class_cov_norm = torch.frobenius_norm(intra_class_cov)

            class_trace = (intra_class_cov * torch.linalg.pinv(inter_class_cov, hermitian=True).t()).sum() / self.hparams.num_classes
            return inter_class_cov_norm, intra_class_cov_norm, class_trace
        elif mode == "group":
            centered_group_means = mode_means - global_mean
            for group_mean in centered_group_means:
                inter_group_cov += torch.outer(group_mean, group_mean)

            inter_group_cov /= self.hparams.num_groups
            intra_group_cov /= num_samples

            inter_group_cov_norm = torch.frobenius_norm(inter_group_cov)
            intra_group_cov_norm = torch.frobenius_norm(intra_group_cov)

            group_trace = (intra_group_cov * torch.linalg.pinv(inter_group_cov, hermitian=True).t()).sum() / self.hparams.num_groups
            return inter_group_cov_norm, intra_group_cov_norm, group_trace

    def compute_collapse_metrics(self):
        """Performs feature collapse metrics computation.

        Returns:
            collapse_metrics: The dictionary of matrix norms for collapse.
        """

        # Registers model hook to get ResNet features.
        def get_features():
            def hook(model, input, output):
                self.features = output.detach()
            return hook
        handle = self.model.layer4.register_forward_hook(get_features())

        # Run 4 separate epoch inferences and compute collapse metrics.
        # If there is enough VRAM, we can combine the separate inferences.
        num_samples, global_mean, class_means, group_means = self.get_global_feature_info()
        total_cov = self.compute_covariance_matrices(num_samples, global_mean, None, mode="total")
        inter_class_cov, intra_class_cov, class_trace = self.compute_covariance_matrices(num_samples, global_mean, class_means, mode="class")
        #inter_class_cov, intra_class_cov = self.compute_covariance_matrices(num_samples, global_mean, class_means, mode="class")
        inter_group_cov, intra_group_cov, group_trace = self.compute_covariance_matrices(num_samples, global_mean, group_means, mode="group")
        #inter_group_cov, intra_group_cov = self.compute_covariance_matrices(num_samples, global_mean, group_means, mode="group")

        # De-registers model feature hook.
        handle.remove()

        collapse_metrics = {
            "global_cov": total_cov,
            "inter_class_cov": inter_class_cov,
            "intra_class_cov": intra_class_cov,
            "inter_group_cov": inter_group_cov,
            "intra_group_cov": intra_group_cov,
            "class_trace": class_trace,
            "group_trace": group_trace,
        }

        return collapse_metrics

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Here, performs covariance matrix calculation for feature collapse.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        collapse_metrics = self.compute_collapse_metrics()

        dataloader_idx = training_step_outputs[0]["dataloader_idx"]

        self.collate_metrics(training_step_outputs, "train")

        self.log_metrics(collapse_metrics, "train", dataloader_idx)

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        log_results(
            self.hparams,
            self.current_epoch + 1,
            self.trainer.logger.version, 
            validation_step_outputs,
            weight_aa_by_proportion=self.hparams.datamodule == "waterbirds",
        )


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
                    for t in TRAIN_TYPES:
                        results[s][v][c][t] = {}
                        epochs = EPOCHS[args.datamodule]
                        if t == "erm":
                            for e in epochs:
                                results[s][v][c][t][e] = {}
                                for m in METRICS:
                                    results[s][v][c][t][e][m] = {}
                        else:
                            for d in CLASS_BALANCING[args.datamodule]:
                                results[s][v][c][t][d] = {}
                                results[s][v][c][t][d][epochs[-1]] = {}
                                for m in METRICS:
                                    results[s][v][c][t][d][epochs[-1]] = {}

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
    t = args.train_type
    e = curr_epoch

    # VERY important to load results right before dumping. Otherwise, we may
    # overwrite results saved by different experiments.
    results = load_results(args)
    if t == "erm":
        for m in METRICS:
            if m in curr_results:
                results[s][v][c][t][e][m] = curr_results[m]
    else:
        for m in METRICS:
            if m in curr_results:
                results[s][v][c][t][d][e][m] = curr_results[m]
    
    with open(args.results_pkl, "wb") as f:
        pickle.dump(results, f)

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

    main(args, model_class, datamodule_class)
    
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
        "resnet": FeatureCollapseResNet,
    }

    args = parser.parse_args()
    args.train_type = "erm"
    args.results_pkl = f"{args.datamodule}_{args.model}.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])