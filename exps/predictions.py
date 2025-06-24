import torch
import numpy as np
from copy import deepcopy
from glob import glob
import os.path as osp
from configargparse import Parser
from distutils.util import strtobool
from pytorch_lightning import Trainer
from milkshake.utils import ignore_warnings
from milkshake.main import main, load_weights
from milkshake.args import add_input_args
from exps.finetune import *

ignore_warnings()

def reset_fc_hook(model):
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
    for p in model.model.parameters():
        p.requires_grad = False
    try:
        for p in model.model.fc.parameters():
            p.requires_grad = True
    except:
        for p in model.model.classifier.parameters():
            p.requires_grad = True

def set_llr_args(args, train_type):
    new_args = deepcopy(args)
    new_args.ckpt_every_n_epochs = new_args.max_epochs + 1
    new_args.check_val_every_n_epoch = new_args.max_epochs
    new_args.lr = 1e-2
    new_args.lr_scheduler = "step"
    new_args.lr_steps = []
    new_args.optimizer = "sgd"
    new_args.weight_decay = 0
    new_args.train_type = train_type
    new_args.retrain_type = "group-balanced retraining" if train_type == "dfr" else "group-unbalanced retraining"
    return new_args

def find_erm_weights(args):
    args.train_type = "erm"

    if args.model == "bert":
        v = args.bert_version
    elif args.model == "convnextv2":
        v = args.convnextv2_version
    elif args.model == "resnet":
        v = args.resnet_version
        
    if not osp.isfile(args.results_pkl):
        raise ValueError(f"Results file {args.results_pkl} not found.")
    results = load_results(args)
    s = args.seed
    v = getattr(args, f"{args.model}_version")
    p = args.split
    c = args.balance_erm
    if "mixture" in c:
        c += str(args.mixture_ratio)
    e = args.max_epochs
    wandb_version = results[s][v][p][c]["erm"][e]["version"]
    if not wandb_version:
        raise ValueError(f"Model version {wandb_version} not found.")
    fpath = "epoch=" + f"{e - 1:02d}" + "*"
    ckpt_path = osp.join(args.wandb_dir, "lightning_logs", wandb_version, "checkpoints", fpath)
    args.weights = glob(ckpt_path)[0]

def ensemble_predictions(models, dataloader, device):
    preds = []
    for model in models:
        model.to(device)
        model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(device)
                logits = model(x)
                all_logits.append(logits.cpu())
        preds.append(torch.cat(all_logits))  #Store model's logit across the test set
    avg_logits = torch.mean(torch.stack(preds), dim=0)  #Compute average prediction over models
    return avg_logits, torch.argmax(avg_logits, dim=1)

def experiment(args, model_class, datamodule_class):
    args.no_test = True
    args.class_weights = None
    args.group_weights = None
    if args.balance_retrain == "upweighting":
        if args.datamodule == "celeba":
            args.class_weights = [1, 5.71]
        elif args.datamodule == "civilcomments":
            args.class_weights = [1, 7.85]
        elif args.datamodule == "multinli":
            args.class_weights = [1, 1, 1]
        elif args.datamodule == "waterbirds":
            args.class_weights = [1, 3.31]

    

    find_erm_weights(args)
    datamodule = datamodule_class(args)
    datamodule.setup()
    args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups
    model = model_class(args)
    model = load_weights(args, model)

    retrained_models = []
    for i in range(15):
        new_args = set_llr_args(args, "dfr")
        if args.datamodule == "civilcomments":
            new_args.lr = 1e-3
        new_args.seed = args.seed + i  # ensure different validation splits
        model_i = deepcopy(model)
        model_i.hparams.train_type = "dfr"
        train_fc_only(model_i)
        main(new_args, model_i, datamodule_class, model_hooks=[reset_fc_hook])
        if args.datamodule == "civilcomments":
            retrained_models.append(deepcopy(model_i.state_dict()))
        else:
            retrained_models.append(deepcopy(model_i))


    test_loader = datamodule.test_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.datamodule == "civilcomments":
        models_for_eval = []
        for state_dict in retrained_models:
            m = model_class(args)
            m.load_state_dict(state_dict)
            m.to(device)
            m.eval()
            models_for_eval.append(m)
        avg_logits, predictions = ensemble_predictions(models_for_eval, test_loader, device)
    else:
        avg_logits, predictions = ensemble_predictions(retrained_models, test_loader, device)

    # Evaluate ensemble classifier accuracy
    y_true = torch.cat([batch[1] for batch in test_loader])  # assumes batch[1] contains labels
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true.numpy(), predictions.numpy())
    print(f"Ensemble Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = Parser(args_for_setting_config_path=["-c", "--cfg", "--config"], config_arg_is_required=True)
    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--balance_erm_type", choices=["class", "group"], default="class")
    parser.add("--balance_erm", choices=["mixture", "none", "subsetting", "upsampling", "upweighting"], default="none")
    parser.add("--balance_retrain", choices=["mixture", "none", "subsetting", "upsampling", "upweighting"], default="none")
    parser.add("--heldout", default=True, type=lambda x: bool(strtobool(x)))
    parser.add("--mixture_ratio", type=float, default=1)
    parser.add("--save_retrained_model", action="store_true")
    parser.add("--split", choices=["combined", "train"], default="train")
    parser.add("--train_pct", default=100, type=int)
    parser.add("--retrain_type", choices=["llr", "dfr", "both"], default="both")

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
