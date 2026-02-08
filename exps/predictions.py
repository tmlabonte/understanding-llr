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
import hashlib, json
from itertools import islice
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn



def _as_int(x, default=-1):
    try: return int(x)
    except: return default



def rebuild_head(m, num_classes):
    # BERT
    if hasattr(m.model, "classifier") and hasattr(m.model.classifier, "in_features"):
        in_f = m.model.classifier.in_features
        m.model.classifier = nn.Linear(in_f, num_classes)
        return ("classifier", in_f, num_classes)
    # ResNet/ConvNeXt
    if hasattr(m.model, "fc") and hasattr(m.model.fc, "in_features"):
        in_f = m.model.fc.in_features
        m.model.fc = nn.Linear(in_f, num_classes)
        return ("fc", in_f, num_classes)
    return ("unknown", None, None)

def _head_shape(m):
    try:
        return tuple(m.model.classifier.weight.shape)
    except Exception:
        try:
            return tuple(m.model.fc.weight.shape)
        except Exception:
            return None




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
    """Retrieves ERM weights from pickle file based on model config, with debug prints."""
    args.train_type = "erm"

    if not osp.isfile(args.results_pkl):
        raise ValueError(f"Results file {args.results_pkl} not found.")

    results = load_results(args)

    # Resolve keys
    s = args.seed
    if args.model == "bert":
        v = args.bert_version
    elif args.model == "convnextv2":
        v = args.convnextv2_version
    elif args.model == "resnet":
        v = args.resnet_version
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    p = args.split
    c = args.balance_erm
    if isinstance(c, str) and "mixture" in c:
        c += str(args.mixture_ratio)
    e = args.max_epochs

    # Debug print of the exact lookup tuple
    print(f"[ERM] lookup keys -> seed(s)={s}  version(v)={v}  split(p)={p}  balance(c)={c}  epochs(e)={e}")

    try:
        wandb_version = results[s][v][p][c]["erm"][e]["version"]
    except KeyError as ke:
        # Extra context to help you see what's present when something mismatches
        def _keys(x): 
            return list(x.keys()) if isinstance(x, dict) else type(x).__name__
        msg = [
            "[ERM] KeyError during results lookup.",
            f"  available seeds: {_keys(results)}",
        ]
        try: msg.append(f"  available versions for seed {s}: {_keys(results[s])}")
        except: pass
        try: msg.append(f"  available splits for (s={s}, v={v}): {_keys(results[s][v])}")
        except: pass
        try: msg.append(f"  available balances for (s={s}, v={v}, p={p}): {_keys(results[s][v][p])}")
        except: pass
        try: msg.append(f"  available train_types for (.., c={c}): {_keys(results[s][v][p][c])}")
        except: pass
        try: msg.append(f"  available epochs for (.., 'erm'): {_keys(results[s][v][p][c]['erm'])}")
        except: pass
        raise KeyError("\n".join(msg)) from ke

    if not wandb_version:
        raise ValueError(f"[ERM] Model version not found for keys s={s}, v={v}, p={p}, c={c}, e={e}.")

    # Resolve checkpoint path and print it
    fpath = "epoch=" + f"{e - 1:02d}" + "*"
    ckpt_path = osp.join(args.wandb_dir, "lightning_logs", wandb_version, "checkpoints", fpath)
    matches = glob(ckpt_path)
    if not matches:
        raise FileNotFoundError(f"[ERM] No checkpoint matched {ckpt_path}")
    args.weights = matches[0]
    print(f"[ERM] resolved -> wandb_version={wandb_version}  ckpt={args.weights}")


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

def dump_hparams(tag, args, trainer_args=None):
    keys = ["seed","max_epochs","batch_size","lr","optimizer","weight_decay",
            "lr_scheduler","lr_steps","train_type","retrain_type",
            "balance_erm","balance_erm_type","balance_retrain","split","train_pct"]
    msg = {k: getattr(args, k, None) for k in keys}
    if trainer_args:
        msg.update({f"trainer.{k}": trainer_args.get(k) for k in trainer_args})
    h = hashlib.sha256(json.dumps(msg, sort_keys=True).encode()).hexdigest()
    print(f"[HP] {tag}: {msg}\n[HP] hash={h}")
    return h

def _collect_trainer_args(args):
    # Pull a few trainer-related fields if present on args (won’t fail if missing)
    fields = ["deterministic","accelerator","devices","precision","gradient_clip_val",
              "accumulate_grad_batches","max_epochs","log_every_n_steps"]
    out = {}
    for f in fields:
        if hasattr(args, f):
            out[f] = getattr(args, f)
    return out

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

    # Build base datamodule once to discover counts
    base_dm = datamodule_class(args)
    base_dm.setup()
    args.num_classes = base_dm.num_classes
    args.num_groups  = base_dm.num_groups

    trainer_args0 = _collect_trainer_args(args)


    for i in range(5):
        # 1) derive per-run args (with new seed)
        new_args = set_llr_args(args, "dfr")
        if args.datamodule == "civilcomments":
            new_args.lr = 1e-3
        new_args.seed = args.seed + i

        # keep dataset-dependent counts in new_args
        new_args.num_classes = args.num_classes
        new_args.num_groups  = args.num_groups

        dump_hparams(f"retrain_run_{i}", new_args, trainer_args=trainer_args0)

        # 2) re-resolve ERM checkpoint for THIS seed/split/balance/epochs
        find_erm_weights(new_args)                 # prints s,v,p,c,e and resolves ckpt
        print(f"[ENC] run {i} ERM ckpt: {new_args.weights}")

        # 3) build a fresh model and load the run’s ERM encoder
        model_i = model_class(new_args)
        model_i = load_weights(new_args, model_i)

        # 4) ensure head matches dataset; freeze encoder
        _ = rebuild_head(model_i, args.num_classes)
        print(f"[HEAD] run {i} head shape={_head_shape(model_i)}  num_classes={args.num_classes}")
        model_i.hparams.train_type = "dfr"
        model_i.hparams.num_classes = args.num_classes
        model_i.hparams.num_groups  = args.num_groups
        train_fc_only(model_i)

        # 5) build per-run datamodule (seed affects splits), fingerprint
        dm_check = datamodule_class(new_args)
        dm_check.setup()
        _ = peek_and_fingerprint_datamodule(dm_check, peek=5)

        # 6) train last layer
        main(new_args, model_i, datamodule_class, model_hooks=[reset_fc_hook])






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

    parser.add("--erm_ckpt_override", type=str, default=None,
        help="Absolute path to a fixed ERM checkpoint to initialize the encoder.")

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
