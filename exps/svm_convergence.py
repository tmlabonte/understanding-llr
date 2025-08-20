# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from copy import deepcopy
from glob import glob
import os.path as osp
import pickle
from PIL import Image


# Imports Python packages.
from configargparse import Parser
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from torchvision import models, transforms
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from milkshake.utils import ignore_warnings
from milkshake.main import main, load_weights
from milkshake.datamodules.datamodule import DataModule
from distutils.util import strtobool


# Imports milkshake packages.
from exps.finetune import *
from milkshake.args import add_input_args
from milkshake.main import main, load_weights

ignore_warnings()


METRICS = ["test_aa", "test_acc_by_class", "test_acc_by_group",
            "test_wca", "test_wga", "train_aa", "train_acc_by_class",
            "train_acc_by_group", "train_wca", "train_wga", "version",
            "cosine_similarity", "directional_error"]
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
    "waterbirds": list(range(10, 501, 10)),
}

# Defines parameters for preset model sizes.
VERSIONS = {
    "bert": ["tiny", "mini", "small", "medium", "base"],
    "convnextv2": ["atto", "femto", "pico", "nano", "tiny", "base"],
    "resnet": [18, 34, 50, 101, 152],
}

class SVMResNet(ResNet):
    def __init__(self, args):
        super().__init__(args)
        self.features = None
        self.svm_weights_normalized = None
        self.svm_trained = False

        # Registers model hook to get ResNet features.
        def get_features():
            def hook(model, input, output):
                self.features = output.detach()
            return hook
        handle = self.model.avgpool.register_forward_hook(get_features())
    
    @torch.no_grad()
    def log_svm_metrics(self, dataloader_idx):
        """Performs margin metrics computation.

        Returns:
            margin_metrics: The dictionary of margins and weight norms.
        """
        
        self.eval()

        # Extract the last fully connected layer weights for later comparison
        nn_fc_weights = self.model.fc[1].weight.data.clone()
        nn_fc_bias = self.model.fc[1].bias.data.clone()

        nn_fc_weights_cpu = nn_fc_weights.cpu()  # First move to CPU
        nn_fc_weights_normalized = nn_fc_weights_cpu / np.linalg.norm(nn_fc_weights_cpu)

        if self.svm_trained == False:

            print("Extracting Features")

            features = []
            labels = []

            with torch.no_grad():
                for batch in self.trainer.train_dataloader:
                    data, target = batch
                    data = data.to(self.device)
                    output = self(data)

                    features.append(self.features)
                    labels.append(target)  

            labels = torch.cat(labels, axis=0)
            features = torch.cat(features, axis=0).squeeze()

            print("Shape of Features: ", features.shape)
            print("Shape of Labels: ", labels.shape)
            
            # Train your SVM

            print("Fitting SVM")

            svm = LinearSVC(fit_intercept=False, C=1e5)
            features = features.cpu()
            labels = labels.cpu()
            svm.fit(features, labels[:, 0])

            svm_weights = svm.coef_
            svm_weights_normalized = svm_weights / np.linalg.norm(svm_weights)

            self.svm_weights_normalized = svm_weights_normalized

            self.svm_trained = True

            print("SVM Trained")

        # Similarity Metrics
        cosine_similarity = np.dot(self.svm_weights_normalized.flatten(), nn_fc_weights_normalized.flatten())
        directional_error = np.linalg.norm(nn_fc_weights_normalized - self.svm_weights_normalized)

        svm_metrics = {
            "cosine_similarity": cosine_similarity,
            "directional_error": directional_error
        }

        self.log_metrics(svm_metrics, "train", dataloader_idx)

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Here, performs covariance matrix calculation for feature collapse.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        dataloader_idx = training_step_outputs[0]["dataloader_idx"]

        self.log_svm_metrics(dataloader_idx)

        self.collate_metrics(training_step_outputs, "train")

class SVMBERT(BERT):
    def __init__(self, args):
        super().__init__(args)
        self.features = None
        self.svm_weights_normalized = None
        self.svm_trained = False
        self.num_classes = args.num_classes

        # Registers model hook to get BERT features.
        def get_features():
            def hook(module, input, output):
                self.features = input[0].detach()  # input to the classifier
            return hook

        handle = self.model.classifier.register_forward_hook(get_features())
    
    @torch.no_grad()
    def log_svm_metrics(self, dataloader_idx):
        """Performs margin metrics computation.

        Returns:
            margin_metrics: The dictionary of margins and weight norms.
        """
        
        self.eval()

        # Extract the last fully connected layer weights for later comparison
        nn_fc_weights = self.model.classifier.weight.data.clone()
        nn_fc_bias = self.model.classifier.bias.data.clone()

        nn_fc_weights_cpu = nn_fc_weights.cpu()  # First move to CPU
        nn_fc_weights_normalized = nn_fc_weights_cpu / np.linalg.norm(nn_fc_weights_cpu)

        if self.svm_trained == False:

            print("Extracting Features")

            features = []
            labels = []

            with torch.no_grad():
                for batch in self.trainer.train_dataloader:
                    data, target = batch
                    data = data.to(self.device)
                    output = self(data)

                    features.append(self.features)
                    labels.append(target)  

            labels = torch.cat(labels, axis=0)
            features = torch.cat(features, axis=0).squeeze()

            print("Shape of Features: ", features.shape)
            print("Shape of Labels: ", labels.shape)
            
            # Train your SVM

            print("Fitting SVM")

            svm = LinearSVC(fit_intercept=False, C=1e5)
            features = features.cpu()
            labels = labels.cpu()
            svm.fit(features, labels[:, 0])

            if self.num_classes == 1:
                svm_weights = svm.coef_
                svm_weights_normalized = svm_weights / np.linalg.norm(svm_weights)

                self.svm_weights_normalized = svm_weights_normalized
            else:
                svm_weights = svm.coef_  # shape: [num_classes, feature_dim]
                svm_weights_normalized = svm_weights / np.linalg.norm(svm_weights, axis=1, keepdims=True)

                self.svm_weights_normalized = svm_weights_normalized

            self.svm_trained = True

            print("SVM Trained")

        # Similarity Metrics

        if self.num_classes == 1:
            cosine_similarity = np.dot(self.svm_weights_normalized.flatten(), nn_fc_weights_normalized.flatten())
            directional_error = np.linalg.norm(nn_fc_weights_normalized - self.svm_weights_normalized)
        else:
            # Normalize the NN weights per class
            nn_fc_weights_normalized = nn_fc_weights_cpu / np.linalg.norm(nn_fc_weights_cpu, axis=1, keepdims=True)
            nn_fc_weights_normalized = nn_fc_weights_normalized.cpu().numpy()

            # Cosine similarities and directional errors per class
            cosine_similarities = np.sum(self.svm_weights_normalized * nn_fc_weights_normalized, axis=1)
            directional_errors = np.linalg.norm(nn_fc_weights_normalized - self.svm_weights_normalized, axis=1)

            # Aggregate metrics
            cosine_similarity = np.mean(cosine_similarities)
            directional_error = np.mean(directional_errors)

        svm_metrics = {
            "cosine_similarity": cosine_similarity,
            "directional_error": directional_error
        }

        self.log_metrics(svm_metrics, "train", dataloader_idx)

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.

        Here, performs covariance matrix calculation for feature collapse.

        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """

        dataloader_idx = training_step_outputs[0]["dataloader_idx"]

        self.log_svm_metrics(dataloader_idx)

        self.collate_metrics(training_step_outputs, "train")


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

    c = args.balance_erm
    d = args.balance_retrain
    if "mixture" in c:
        c += str(args.mixture_ratio)
    if "mixture" in d:
        d += str(args.mixture_ratio)
    e = args.max_epochs

    wandb_version = results[s][v][c]["erm"][e]["version"]
    if not wandb_version:
        raise ValueError(f"Model version {wandb_version} not found.")

    # Finds model weights in wandb dir.
    fpath = "epoch=" + f"{e - 1:02d}" + "*"
    ckpt_path = osp.join(
        args.wandb_dir, "lightning_logs", wandb_version, "checkpoints", fpath)
    args.weights = glob(ckpt_path)[0]

def experiment(args, model_class, datamodule_class):
    """Runs main training and evaluation procedure."""
   
    datamodule = datamodule_class(args)
    datamodule.setup()

    if args.num_classes == None:
        args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    model = model_class(args)
    model = load_weights(args, model)

    # Performs LLR.
    new_args = set_llr_args(args, "llr")
    model.hparams.train_type = "llr" # Used for dumping results
    train_fc_only(model)
    model, _, _ = main(
        new_args, model, datamodule_class, model_hooks=[reset_fc_hook])

    """
    # Performs DFR.
    new_args = set_llr_args(args, "dfr")
    model.hparams.train_type = "dfr" # Used for dumping results
    train_fc_only(model)
    model, _, _ = main(
        new_args, model, datamodule_class, model_hooks=[reset_fc_hook])
    """

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
        "bert": SVMBERT,
        "convnextv2": ConvNeXtV2WithLogging,
        "resnet": SVMResNet,
    }

    args = parser.parse_args()
    args.results_pkl = f"{args.datamodule}_{args.model}_svm_llr.pkl"
    experiment(args, models[args.model], datamodules[args.datamodule])
