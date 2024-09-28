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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from torchvision import models, transforms
from sklearn.svm import SVC
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

class ConvNeXtV2WithFeatures(ConvNeXtV2):
    def __init__(self, args):
        super().__init__(args)
        self.features = None

        # Registers model hook to get ResNet features.
        def get_features():
            def hook(model, input, output):
                self.features = output.detach()
            return hook
        handle = self.model.convnextv2.layernorm.register_forward_hook(get_features())

class ResNetWithFeatures(ResNet):
    def __init__(self, args):
        super().__init__(args)
        self.features = None

        # Registers model hook to get ResNet features.
        def get_features():
            def hook(model, input, output):
                self.features = output.detach()
            return hook
        handle = self.model.avgpool.register_forward_hook(get_features())

def find_erm_weights(args):
    """Retrieves ERM weights from pickle file based on model config."""
    
    ### Get the wandb version from pickle file.

    if not osp.isfile(args.results_pkl):
        raise ValueError(f"Results file {args.results_pkl} not found.")

    results = load_results(args)

    args.train_type = "erm"
    s = args.seed
    dataset = args.datamodule

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

    args.no_test = True
    find_erm_weights(args)

    datamodule = datamodule_class(args)
    datamodule.setup()


    if args.num_classes == None:
        args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    nn_model = model_class(args)

    nn_model = load_weights(args, nn_model)

    # Extract the last fully connected layer weights for later comparison

    if args.model == "convnextv2":
        original_fc_weights = nn_model.model.classifier.weight.data.clone()
        original_fc_bias = nn_model.model.classifier.bias.data.clone()
    elif args.model == "resnet":
        original_fc_weights = nn_model.model.fc[1].weight.data.clone()
        original_fc_bias = nn_model.model.fc[1].bias.data.clone()
    else:
        original_fc_weights = nn_model.model.fc.weight.data.clone()
        original_fc_bias = nn_model.model.fc.bias.data.clone()

    print(original_fc_weights.shape)

    print("Extracting Features")

    features = []
    labels = []
    nn_normalized_min_margin = None

    with torch.no_grad():
        for batch in datamodule.train_dataloader():
            data, target = batch
            output = nn_model(data)

            # correct_class_logits = output[target[:, 0]]
            # incorrect_class_logits = output[1 - target[:, 0]]

            # correct_class_weights = original_fc_weights[target[:, 0]]  # Shape: [batch_size, 512]
            # weight_norms = torch.norm(correct_class_weights, p=2, dim=1)

            # margins = correct_class_logits - incorrect_class_logits
            # normalized_margins = margins / weight_norms
            # if nn_normalized_min_margin == None or (len(normalized_margins[normalized_margins > 0]) > 0 and torch.min(normalized_margins[normalized_margins > 0]) < nn_normalized_min_margin):
            #     nn_normalized_min_margin = torch.min(normalized_margins[normalized_margins > 0])

            features.append(nn_model.features)
            labels.append(target)  

    labels = torch.cat(labels, axis=0)
    features = torch.cat(features, axis=0).squeeze()

    print(features.shape)
    print(labels.shape)
    # print(nn_normalized_min_margin)
      
    # Train your SVM

    print("Fitting SVM")

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(features, labels[:, 0])

    # # Get the support vectors
    # support_vectors = svm.support_vectors_

    # # Calculate the distance from the support vectors to the hyperplane
    # distances = svm.decision_function(support_vectors)

    # # Find the minimum distance
    # svm_min_margin = min(abs(distances))

    # Compare SVM weights with the original model's FC layer weights
    svm_weights = svm.coef_
    svm_bias = svm.intercept_

    # # Normalize the min margin by the norm of the weights
    # svm_normalized_min_margin = svm_min_margin /  np.linalg.norm(svm_weights)

    # # This line doubles the SVM weights to have weights for both classes
    # svm_weights = np.vstack([svm_weights, -svm_weights])

    # Normalize SVM weights
    svm_weights_normalized = svm_weights / np.linalg.norm(svm_weights)

    # Original FC layer weights
    original_fc_weights_normalized = original_fc_weights / np.linalg.norm(original_fc_weights)

    # Similarity Metrics
    cosine_similarity = np.dot(svm_weights_normalized.flatten(), original_fc_weights_normalized.flatten())
    directional_error = np.linalg.norm(original_fc_weights_normalized - svm_weights_normalized)
    print(f'Cosine Similarity between SVM and original FC weights: {cosine_similarity:.4f}')
    print(f'Directional Error between SVM and original FC weights: {directional_error:.4f}')

    # print(f'SVM Margin: {svm_normalized_min_margin:.4f}')
    # print(f'NN Last Layer Margin: {nn_normalized_min_margin:.4f}')


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
    parser.add("--layerwise", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether or not the neural network model was trained layerwise.")

    datamodules = {
        "celeba": CelebARetrain,
        "civilcomments": CivilCommentsRetrain,
        "multinli": MultiNLIRetrain,
        "waterbirds": WaterbirdsRetrain,
    }
    models = {
        "bert": BERTWithLogging,
        "convnextv2": ConvNeXtV2WithFeatures,
        "resnet": ResNetWithFeatures,
    }

    args = parser.parse_args()
    args.train_type = "erm"

    if args.layerwise == True:
        args.results_pkl = f"{args.datamodule}_{args.model}_layerwise.pkl"
    else:
        args.results_pkl = f"{args.datamodule}_{args.model}.pkl"
    print(args.results_pkl)
    experiment(args, models[args.model], datamodules[args.datamodule])