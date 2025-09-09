"""Main file for ERM neural collapse."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
import os.path as osp
import pickle
from copy import deepcopy

# Imports Python packages.
from configargparse import Parser
import numpy as np
import math
from distutils.util import strtobool

# Imports PyTorch packages.
import torch
from pytorch_lightning import Trainer
import torch.nn.functional as F

# Imports Sklearn packages.
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse.linalg import lsqr

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.civilcomments import CivilComments
from milkshake.datamodules.multinli import MultiNLI
from milkshake.datamodules.retrain import Retrain
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.main import main, load_weights
from milkshake.models.bert import BERT
from milkshake.models.convnextv2 import ConvNeXtV2
from milkshake.models.resnet import ResNet
from milkshake.imports import valid_models_and_datamodules
from milkshake.main import main
from milkshake.utils import compute_accuracy, get_weights, to_np
from milkshake.utils import compute_accuracy


METRICS = ["test_aa", "test_acc_by_class", "test_acc_by_group",
            "test_wca", "test_wga", "train_aa", "train_acc_by_class",
            "train_acc_by_group", "train_wca", "train_wga", "version",
            "global_cov", "inter_class_cov", "intra_class_cov", 
            "inter_group_cov", "intra_group_cov", "class_trace", "group_trace",
            "max_margin", "min_margin", "avg_margin", "weight_norm", "min_margin_by_group",
            "max_margin_by_group", "avg_margin_by_group"]
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
    "celeba": list(range(20, 61, 20)),
    "civilcomments": list(range(5, 61, 5)),
    "multinli": list(range(20, 61, 20)),
    "waterbirds": list(range(100, 301, 100)),
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


def get_unfrozen_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())

def compute_norm_of_unfrozen_weights(model):
    unfrozen_params = get_unfrozen_params(model)
    total_norm = torch.norm(torch.cat([p.view(-1) for p in unfrozen_params]), 2)
    return total_norm.item()


def get_vectorized_features(targets, features, num_classes, num_groups):
    """Gets vectorized features for each group, class, and training split."""
    groups = None
    if num_groups:
        groups = targets[:, 1]
        targets = targets[:, 0]

    batch_size = features.size(0)
    batch_vectorized_features = features.view(batch_size, -1)

    features_by_class = {}
    for j in range(num_classes):
        class_features = batch_vectorized_features[targets == j]
        if j in features_by_class:
            features_by_class[j] = torch.cat((features_by_class[j], class_features), dim=0)
        else:
            features_by_class[j] = class_features

    features_by_group = {}
    if num_groups:
        for j in range(num_groups):
            group_features = batch_vectorized_features[groups == j]
            if j in features_by_group:
                features_by_group[j] = torch.cat((features_by_group[j], group_features), dim=0)
            else:
                features_by_group[j] = group_features

    vectorized_features = {
        "features_by_class": features_by_class,
        "features_by_group": features_by_group,
        "features": batch_vectorized_features
    }

    return vectorized_features


def conjugate_gradient_torch(A_func, b, tol=1e-5, maxiter=None):
    """
    Pure PyTorch conjugate gradient solver for symmetric positive-definite linear operators.
    A_func: function that maps vector -> vector (same shape as b)
    b: 1-D torch tensor
    """
    # ensure b is 1-D
    assert b.dim() == 1, "b must be 1-D"
    device = b.device
    dtype = b.dtype
    n = b.shape[0]
    if maxiter is None:
        maxiter = n

    x = torch.zeros_like(b, device=device, dtype=dtype)
    r = b - A_func(x)
    if r.norm() < tol:
        return x, 0

    p = r.clone()
    rTr = torch.dot(r, r)
    k = 0

    while k < maxiter:
        Ap = A_func(p)
        pAp = torch.dot(p, Ap)
        if torch.abs(pAp) < 1e-30:
            # breakdown
            return x, k + 1

        alpha = rTr / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        if r.norm() < tol:
            return x, 0

        rnewTrnew = torch.dot(r, r)
        beta = rnewTrnew / rTr
        p = r + beta * p
        rTr = rnewTrnew
        k += 1

    return x, maxiter  # did not converge within maxiter


class FeatureCollapseMarginResNet(ResNet):
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
                if j in features["features_by_class"]:
                    num_samples_per_class[j] += len(features["features_by_class"][j])
                    class_means[j] += torch.sum(features["features_by_class"][j], dim=0)

            for j in range(self.hparams.num_groups):
                if j in features["features_by_group"]:
                    num_samples_per_group[j] += len(features["features_by_group"][j])
                    group_means[j] += torch.sum(features["features_by_group"][j], dim=0)

        global_mean /= num_samples

        for j in range(self.hparams.num_classes):
            if num_samples_per_class[j] > 0:
                class_means[j] /= num_samples_per_class[j]
            else:
                class_means[j] = torch.zeros_like(global_mean)
                
        for j in range(self.hparams.num_groups):
            if num_samples_per_group[j] > 0:
                group_means[j] /= num_samples_per_group[j]
            else:
                group_means[j] = torch.zeros_like(global_mean)
                
        class_means = torch.stack(class_means)
        group_means = torch.stack(group_means)

        return num_samples, global_mean, class_means, group_means

    @torch.no_grad()
    def compute_covariance_matrices(self, num_samples, global_mean, mode_means, mode=None):
        """Performs memory-efficient computation of covariance matrices for feature collapse.

        Args:
            num_samples: The number of samples in the training dataset.
            global_mean: The mean of all training sample features.
            mode_means: Either the class means or group means (tensor of shape [M, d]).
            mode: Either "total", "class", or "group".

        Returns:
            Various norms and traces depending on mode.
        """
        torch.cuda.empty_cache()

        feature_dim = global_mean.shape[0]
        num_modes = mode_means.shape[0] if mode != "total" else 0
        device, dtype = global_mean.device, global_mean.dtype

        if mode != "total":
            features_by_mode = "features_by_class" if mode == "class" else "features_by_group"

        num_random_vecs = 1

        # === Inter-covariance pseudoinverse solver (M x M trick) ===
        # if mode != "total" and num_modes > 0:
        #     C = (mode_means - global_mean.to(device=device, dtype=dtype))  # (M, d)
        #     M = C.shape[0]

        #     # small Gram matrix
        #     G = (C @ C.t()) / float(M)   # (M, M)

        #     # regularization relative to spectrum
        #     eigvals = torch.linalg.eigvalsh(G)
        #     max_eig = eigvals.max().clamp(min=0.0)
        #     eps = max(1e-12, 1e-6 * float(max_eig))
        #     G_reg = G + eps * torch.eye(M, device=device, dtype=dtype)

        #     # factorize once
        #     try:
        #         L = torch.linalg.cholesky(G_reg)
        #         def solve_G(b):
        #             y = torch.cholesky_solve(b.unsqueeze(-1), L)  # (M,1)
        #             return y.squeeze(-1)
        #     except RuntimeError:
        #         G_inv = torch.linalg.inv(G_reg)
        #         def solve_G(b):
        #             return G_inv @ b

        #     def solve_inter_pinv(z):
        #         # Computes x ≈ B^+ z without ever forming d x d
        #         Cz = C @ z                     # (M,)
        #         y = solve_G(Cz)                # (M,)
        #         x = float(M) * (C.t() @ y)     # (d,)
        #         return x
        # else:
        #     def solve_inter_pinv(z):
        #         return torch.zeros_like(z)
            

        # def solve_inter_pinv(z):
        #     M = num_modes
        #     C = (mode_means - global_mean.to(device=device, dtype=dtype))  # (M, d)
        #     Cz = C @ z                                                     # (M,)
        #     G = (C @ C.t())                                                # (M, M)
        #     G_pinv = torch.linalg.pinv(G)
        #     Btz = float(M) * (C.t() @ G_pinv @ Cz)
        #     return Btz

        # ---- inter-covariance: B v = (1/M) sum_c ( (mu_c - mu) ( (mu_c - mu)^T v ) )
        def apply_inter_cov(v):
            centered_means = mode_means - global_mean  # (M, d)
            v_torch = torch.from_numpy(v).to(centered_means.device).to(centered_means.dtype) # (d,)
            Cv = centered_means @ v_torch                     # (M,)
            Bv = centered_means.t().mv(Cv)          # (d,)
            return (Bv / float(num_modes)).cpu().numpy()


        # === Apply intra covariance (streaming) ===
        def apply_intra_cov(v):
            Wv = torch.zeros((feature_dim,), device=device, dtype=dtype)
            for idx, batch in enumerate(self.trainer.train_dataloader):
                batch[0], batch[1] = batch[0].cuda(), batch[1].cuda()
                result = self.step(batch, idx)
                features = get_vectorized_features(
                    result["targets"], self.features,
                    self.hparams.num_classes, self.hparams.num_groups,
                )
                centered_feats_list = []
                for j in range(num_modes):
                    if j in features[features_by_mode] and len(features[features_by_mode][j]) > 0:
                        centered_feats_list.append(features[features_by_mode][j] - mode_means[j])
                if len(centered_feats_list) == 0:
                    continue
                centered_feats = torch.cat(centered_feats_list, dim=0)  # (n_j, d)
                Cv = centered_feats @ v                                # (n_j,)
                Wv += centered_feats.t() @ Cv                          # (d,)
            Wv /= num_samples
            return Wv

        # === Apply total covariance (streaming) ===
        def apply_total_cov(v):
            Tv = torch.zeros((feature_dim,), device=device, dtype=dtype)
            for idx, batch in enumerate(self.trainer.train_dataloader):
                batch[0], batch[1] = batch[0].cuda(), batch[1].cuda()
                result = self.step(batch, idx)
                features = get_vectorized_features(
                    result["targets"], self.features,
                    self.hparams.num_classes, self.hparams.num_groups,
                )
                centered_feats = features["features"] - global_mean
                Cv = centered_feats @ v
                Tv += centered_feats.t() @ Cv
            Tv /= num_samples
            return Tv

        # === Norm/trace estimation ===
        if mode == "total":
            norm_estimate = 0
            for _ in range(num_random_vecs):
                z = torch.randint(0, 2, (feature_dim,), device=device, dtype=dtype) * 2 - 1
                Tz = apply_total_cov(z)
                norm_estimate += torch.dot(z, Tz)
            norm_estimate /= num_random_vecs
            norm_estimate = torch.sqrt(norm_estimate)
            return norm_estimate
        else:
            trace_estimate, inter_norm_estimate, intra_norm_estimate = 0, 0, 0
            for _ in range(num_random_vecs):
                z = torch.randint(0, 2, (feature_dim,), device=device, dtype=dtype) * 2 - 1
                # inter norm
                x_inter = (mode_means - global_mean)  # just to reuse
                Bz = x_inter.t() @ (x_inter @ z) / num_modes  # same as apply_inter_cov
                inter_norm_estimate += torch.dot(Bz, Bz)
                # intra norm
                Wz = apply_intra_cov(z)
                intra_norm_estimate += torch.dot(Wz, Wz)
                # trace via pseudoinverse solve
                z_numpy = z.cpu().numpy()

                B = LinearOperator(
                    shape=(feature_dim, feature_dim),
                    matvec=apply_inter_cov,
                    rmatvec=apply_inter_cov
                )

                x_numpy  = lsqr(B, z_numpy)[0]
                x_torch = torch.from_numpy(x_numpy).float().cuda()
                y = apply_intra_cov(x_torch)
                trace_estimate += torch.dot(z, y)

            inter_norm_estimate = torch.sqrt(inter_norm_estimate / num_random_vecs)
            intra_norm_estimate = torch.sqrt(intra_norm_estimate / num_random_vecs)
            trace_estimate /= num_random_vecs
            return inter_norm_estimate, intra_norm_estimate, trace_estimate

        

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


        if not hasattr(self.trainer, 'train_dataloader') or self.trainer.train_dataloader is None:
            # Get dataloader directly from datamodule for eval-only mode
            dataloader = self.trainer.datamodule.train_dataloader()
            # Temporary assign it to make rest of the code work
            self.trainer.train_dataloader = dataloader

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
    
    @torch.no_grad()
    def log_margin_metrics(self, dataloader_idx):
        """Performs margin metrics computation.

        Returns:
            margin_metrics: The dictionary of margins and weight norms.
        """
        
        self.eval()

        # Compute the L2 norm of the network weights 
        weight_norm = compute_norm_of_unfrozen_weights(self)

        # Initialize lists to store correct class scores and maximum incorrect class scores
        correct_class_scores_list = []
        max_incorrect_class_scores_list = []
        group_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(self.trainer.train_dataloader):
                data = batch[0].cuda()
                labels = batch[1].cuda()

                group_labels_batch = labels[:, 1].tolist()
                group_labels.extend(group_labels_batch)

                # Forward pass
                output_scores = self(data)

                # Computes loss and prediction probabilities.
                if self.hparams.num_classes == 1:
                    output_scores = torch.sigmoid(output_scores)
                else:
                    output_scores = F.softmax(output_scores, dim=1)                

                # Iterate through examples in the batch
                for i in range(len(labels)):
                    class_label = labels[i, 0]

                    # Extract scores for the correct class
                    correct_class_score = output_scores[i, class_label]

                    # Extract scores for all other classes
                    incorrect_class_scores = output_scores[i][labels[i, 0] != torch.arange(output_scores.size(1)).cuda()]

                    # Calculate the maximum incorrect class score
                    max_incorrect_class_score = torch.max(incorrect_class_scores)

                    # Store correct and incorrect class scores
                    correct_class_scores_list.append(correct_class_score.item())
                    max_incorrect_class_scores_list.append(max_incorrect_class_score.item())

        # Convert lists to tensors
        correct_class_scores_tensor = torch.tensor(correct_class_scores_list)
        max_incorrect_class_scores_tensor = torch.tensor(max_incorrect_class_scores_list)
        group_labels = torch.tensor(group_labels)

        # Get unique group labels
        unique_groups = torch.unique(group_labels)

        # Calculate margins for each point
        margins = correct_class_scores_tensor - max_incorrect_class_scores_tensor
        normalized_margins = margins / weight_norm

        # Create a mask for the positive margins
        positive_margin_mask = normalized_margins > 0
        positive_margins = normalized_margins[positive_margin_mask]

        # Calculates max, min, average, and per group margins.
        if positive_margins.numel() > 0:
            min_margin = torch.min(positive_margins)
        else:
            min_margin = 0
        max_margin = torch.max(normalized_margins)
        avg_margin = torch.sum(normalized_margins) / len(normalized_margins)

        margin_metrics = {
            "max_margin": max_margin,
            "min_margin": min_margin,
            "avg_margin": avg_margin,
            "weight_norm": weight_norm
        }

        self.log_metrics(margin_metrics, "train", dataloader_idx)

        if "min_margin_by_group" in self.hparams.metrics or\
            "max_margin_by_group" in self.hparams.metrics or\
            "avg_margin_by_group" in self.hparams.metrics:

            names = []
            values = []

            min_margins_by_group = []
            max_margins_by_group = []
            avg_margins_by_group = []

            # Iterate over unique group labels
            for group_label in unique_groups:
                # Mask to filter margins for the current group
                mask = (group_labels == group_label)
                
                # Filter margins for the current group
                normalized_margins_for_group = normalized_margins[mask]

                # Create a mask for the positive margins
                positive_group_margin_mask = normalized_margins_for_group > 0
                positive_group_margins = normalized_margins_for_group[positive_group_margin_mask]
                
                # Calculate the margins for the current group
                if positive_group_margins.numel() > 0:
                    min_margin_for_group = torch.min(positive_group_margins)
                else:
                    min_margin_for_group = torch.tensor(0)

                max_margin_for_group = torch.max(normalized_margins_for_group)
                avg_margin_for_group = torch.sum(normalized_margins_for_group) / len(normalized_margins_for_group)

                
                # Append the result to the list
                min_margins_by_group.append(min_margin_for_group.item())
                max_margins_by_group.append(max_margin_for_group.item())
                avg_margins_by_group.append(avg_margin_for_group.item())

            if "min_margin_by_group" in self.hparams.metrics:
                names.extend([f"min_margin_group{group}" for group in unique_groups])
                values.extend(list(min_margins_by_group))
            if "max_margin_by_group" in self.hparams.metrics:
                names.extend([f"max_margin_group{group}" for group in unique_groups])
                values.extend(list(max_margins_by_group))
            if "avg_margin_by_group" in self.hparams.metrics:
                names.extend([f"avg_margin_group{group}" for group in unique_groups])
                values.extend(list(avg_margins_by_group))

            self.log_helper2(names, values, dataloader_idx)

    # def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
    #     """Performs a step, then computes and logs metrics.

    #     Args:
    #         batch: A tuple containing the inputs and targets as torch.Tensor.
    #         idx: The index of the given batch.
    #         dataloader_idx: The index of the current dataloader.
    #         stage: "train", "val", or "test".

    #     Returns:
    #         A dictionary containing the loss, prediction probabilities, targets, and metrics.
    #     """

    #     result = self.step(batch, idx)

    #     accs = compute_accuracy(
    #         result["probs"],
    #         result["targets"],
    #         self.hparams.num_classes,
    #         self.hparams.num_groups,
    #     )

    #     self.add_metrics_to_result(result, accs, dataloader_idx)

    #     self.log_metrics(result, stage, dataloader_idx)

    #     # New addition to log margin metrics during the first epoch.
    #     if idx % 2500 == 0:
    #         collapse_metrics = self.compute_collapse_metrics()
    #         self.log_margin_metrics(dataloader_idx)
    #         self.log_metrics(collapse_metrics, "train", dataloader_idx)

    #     return result

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Here, performs covariance matrix calculation for feature collapse.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        collapse_metrics = self.compute_collapse_metrics()

        dataloader_idx = training_step_outputs[0]["dataloader_idx"]

        self.log_margin_metrics(dataloader_idx)

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

        collapse_metrics = self.compute_collapse_metrics()

        dump_results(args, self.current_epoch + 1, collapse_metrics)

class FeatureCollapseMarginBERT(BERT):
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

        torch.cuda.empty_cache()  # Good practice, but might need more aggressive memory management

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
            total_cov_norm = torch.norm(total_cov, p="fro")
            return total_cov_norm
        elif mode == "class":
            centered_class_means = mode_means - global_mean
            for class_mean in centered_class_means:
                inter_class_cov += torch.outer(class_mean, class_mean)

            inter_class_cov /= self.hparams.num_classes
            intra_class_cov /= num_samples

            inter_class_cov_norm = torch.norm(inter_class_cov, p="fro")
            intra_class_cov_norm = torch.norm(intra_class_cov, p="fro")

            # Debugging the matrices
            print("intra_class_cov shape:", intra_class_cov.shape)
            print("inter_class_cov shape:", inter_class_cov.shape)
            print("Any NaN in intra_class_cov:", torch.isnan(intra_class_cov).any())
            print("Any NaN in inter_class_cov:", torch.isnan(inter_class_cov).any())

            # Handle NaN values if found
            intra_class_cov = torch.nan_to_num(intra_class_cov, nan=0.0)
            inter_class_cov = torch.nan_to_num(inter_class_cov, nan=0.0)

            # The problematic line
            # class_trace = (intra_class_cov * torch.linalg.pinv(inter_class_cov, hermitian=True).t()).sum() / self.hparams.num_classes
            epsilon = 1e-3  # Adjust as needed
            inter_class_cov_reg = inter_class_cov + torch.eye(inter_class_cov.shape[0], device=inter_class_cov.device) * epsilon
            class_trace = (intra_class_cov * torch.linalg.pinv(inter_class_cov_reg, hermitian=True).t()).sum() / self.hparams.num_classes
            
            return inter_class_cov_norm, intra_class_cov_norm, class_trace
        elif mode == "group":
            centered_group_means = mode_means - global_mean
            for group_mean in centered_group_means:
                inter_group_cov += torch.outer(group_mean, group_mean)

            inter_group_cov /= self.hparams.num_groups
            intra_group_cov /= num_samples

            inter_group_cov_norm = torch.norm(inter_group_cov, p="fro")
            intra_group_cov_norm = torch.norm(intra_group_cov, p="fro")

            # Debugging the matrices
            print("intra_group_cov shape:", intra_group_cov.shape)
            print("inter_group_cov shape:", inter_group_cov.shape)
            print("Any NaN in intra_group_cov:", torch.isnan(intra_group_cov).any())
            print("Any NaN in inter_group_cov:", torch.isnan(inter_group_cov).any())

            # Handle NaN values if found
            intra_group_cov = torch.nan_to_num(intra_group_cov, nan=0.0)
            inter_group_cov = torch.nan_to_num(inter_group_cov, nan=0.0)

            # The problematic line
            # group_trace = (intra_group_cov * torch.linalg.pinv(inter_group_cov, hermitian=True).t()).sum() / self.hparams.num_groups
            epsilon = 1e-3  # Adjust as needed
            inter_group_cov_reg = inter_group_cov + torch.eye(inter_group_cov.shape[0], device=inter_group_cov.device) * epsilon
            group_trace = (intra_group_cov * torch.linalg.pinv(inter_group_cov_reg, hermitian=True).t()).sum() / self.hparams.num_groups



            return inter_group_cov_norm, intra_group_cov_norm, group_trace

    def compute_collapse_metrics(self):
        """Performs feature collapse metrics computation.

        Returns:
            collapse_metrics: The dictionary of matrix norms for collapse.
        """

        # Registers model hook to get ResNet features.
        def get_features():
            def hook(model, input, output):
                self.features = output[0].detach()
            return hook
        handle = self.model.bert.encoder.layer[-1].register_forward_hook(get_features())


        if not hasattr(self.trainer, 'train_dataloader') or self.trainer.train_dataloader is None:
            # Get dataloader directly from datamodule for eval-only mode
            dataloader = self.trainer.datamodule.train_dataloader()
            # Temporary assign it to make rest of the code work
            self.trainer.train_dataloader = dataloader

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
    
    @torch.no_grad()
    def log_margin_metrics(self, dataloader_idx):
        """Performs margin metrics computation.

        Returns:
            margin_metrics: The dictionary of margins and weight norms.
        """
        
        self.eval()

        # Compute the L2 norm of the network weights 
        weight_norm = compute_norm_of_unfrozen_weights(self)

        # Initialize lists to store correct class scores and maximum incorrect class scores
        correct_class_scores_list = []
        max_incorrect_class_scores_list = []
        group_labels = []

        with torch.no_grad():
            for idx, batch in enumerate(self.trainer.train_dataloader):
                data = batch[0].cuda()
                labels = batch[1].cuda()

                group_labels_batch = labels[:, 1].tolist()
                group_labels.extend(group_labels_batch)

                # Forward pass
                output_scores = self(data)

                # Computes loss and prediction probabilities.
                if self.hparams.num_classes == 1:
                    output_scores = torch.sigmoid(output_scores)
                else:
                    output_scores = F.softmax(output_scores, dim=1)                

                # Iterate through examples in the batch
                for i in range(len(labels)):
                    class_label = labels[i, 0]

                    # Extract scores for the correct class
                    correct_class_score = output_scores[i, class_label]

                    # Extract scores for all other classes
                    incorrect_class_scores = output_scores[i][labels[i, 0] != torch.arange(output_scores.size(1)).cuda()]

                    # Calculate the maximum incorrect class score
                    max_incorrect_class_score = torch.max(incorrect_class_scores)

                    # Store correct and incorrect class scores
                    correct_class_scores_list.append(correct_class_score.item())
                    max_incorrect_class_scores_list.append(max_incorrect_class_score.item())

        # Convert lists to tensors
        correct_class_scores_tensor = torch.tensor(correct_class_scores_list)
        max_incorrect_class_scores_tensor = torch.tensor(max_incorrect_class_scores_list)
        group_labels = torch.tensor(group_labels)

        # Get unique group labels
        unique_groups = torch.unique(group_labels)

        # Calculate margins for each point
        margins = correct_class_scores_tensor - max_incorrect_class_scores_tensor
        normalized_margins = margins / weight_norm

        # Create a mask for the positive margins
        positive_margin_mask = normalized_margins > 0
        positive_margins = normalized_margins[positive_margin_mask]

        # Calculates max, min, average, and per group margins.
        if positive_margins.numel() > 0:
            min_margin = torch.min(positive_margins)
        else:
            min_margin = 0
        max_margin = torch.max(normalized_margins)
        avg_margin = torch.sum(normalized_margins) / len(normalized_margins)

        margin_metrics = {
            "max_margin": max_margin,
            "min_margin": min_margin,
            "avg_margin": avg_margin,
            "weight_norm": weight_norm
        }

        self.log_metrics(margin_metrics, "train", dataloader_idx)

        if "min_margin_by_group" in self.hparams.metrics or\
            "max_margin_by_group" in self.hparams.metrics or\
            "avg_margin_by_group" in self.hparams.metrics:

            names = []
            values = []

            min_margins_by_group = []
            max_margins_by_group = []
            avg_margins_by_group = []

            # Iterate over unique group labels
            for group_label in unique_groups:
                # Mask to filter margins for the current group
                mask = (group_labels == group_label)
                
                # Filter margins for the current group
                normalized_margins_for_group = normalized_margins[mask]

                # Create a mask for the positive margins
                positive_group_margin_mask = normalized_margins_for_group > 0
                positive_group_margins = normalized_margins_for_group[positive_group_margin_mask]
                
                # Calculate the margins for the current group
                if positive_group_margins.numel() > 0:
                    min_margin_for_group = torch.min(positive_group_margins)
                else:
                    min_margin_for_group = torch.tensor(0)

                max_margin_for_group = torch.max(normalized_margins_for_group)
                avg_margin_for_group = torch.sum(normalized_margins_for_group) / len(normalized_margins_for_group)

                
                # Append the result to the list
                min_margins_by_group.append(min_margin_for_group.item())
                max_margins_by_group.append(max_margin_for_group.item())
                avg_margins_by_group.append(avg_margin_for_group.item())

            if "min_margin_by_group" in self.hparams.metrics:
                names.extend([f"min_margin_group{group}" for group in unique_groups])
                values.extend(list(min_margins_by_group))
            if "max_margin_by_group" in self.hparams.metrics:
                names.extend([f"max_margin_group{group}" for group in unique_groups])
                values.extend(list(max_margins_by_group))
            if "avg_margin_by_group" in self.hparams.metrics:
                names.extend([f"avg_margin_group{group}" for group in unique_groups])
                values.extend(list(avg_margins_by_group))

            self.log_helper2(names, values, dataloader_idx)

    # def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
    #     """Performs a step, then computes and logs metrics.

    #     Args:
    #         batch: A tuple containing the inputs and targets as torch.Tensor.
    #         idx: The index of the given batch.
    #         dataloader_idx: The index of the current dataloader.
    #         stage: "train", "val", or "test".

    #     Returns:
    #         A dictionary containing the loss, prediction probabilities, targets, and metrics.
    #     """

    #     result = self.step(batch, idx)

    #     accs = compute_accuracy(
    #         result["probs"],
    #         result["targets"],
    #         self.hparams.num_classes,
    #         self.hparams.num_groups,
    #     )

    #     self.add_metrics_to_result(result, accs, dataloader_idx)

    #     self.log_metrics(result, stage, dataloader_idx)

    #     # New addition to log margin metrics during the first epoch.
    #     if idx % 3000 == 0:
    #         collapse_metrics = self.compute_collapse_metrics()
    #         self.log_margin_metrics(dataloader_idx)
    #         self.log_metrics(collapse_metrics, "train", dataloader_idx)

    #     return result

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Here, performs covariance matrix calculation for feature collapse.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        collapse_metrics = self.compute_collapse_metrics()

        dataloader_idx = training_step_outputs[0]["dataloader_idx"]

        self.log_margin_metrics(dataloader_idx)

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

        collapse_metrics = self.compute_collapse_metrics()

        dump_results(args, self.current_epoch + 1, collapse_metrics)


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
        "bert": FeatureCollapseMarginBERT,
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": FeatureCollapseMarginResNet,
    }

    args = parser.parse_args()
    args.train_type = "erm"
    args.results_pkl = f"{args.datamodule}_{args.model}_collapse_margin_seed_1_epochs_20_40.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
