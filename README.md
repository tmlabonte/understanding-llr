# On the Unreasonable Effectiveness of Last-layer Retraining
## Official codebase for the ICLR 2025 SCSL workshop paper
### Installation
```
conda update -n base -c defaults conda
conda create -n milkshake python==3.10
conda activate milkshake
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -e .
```
### Instructions
To run an experiment, specify the config with `-c`. For example,
```
python exps/finetune.py -c cfgs/waterbirds.yaml
```

By default, the program will run ERM finetuning with no class-balancing. Here is an example of a run with a different class-balancing method:
```
python exps/finetune.py -c cfgs/waterbirds.yaml --balance_erm subsetting
```

After models are finetuned, run last-layer retraining with `exps/llr.py`.

### Citation and License
This codebase uses [Milkshake](https://github.com/tmlabonte/milkshake) as a template and inherits its MIT License. Please consider using the following citation:
```
@inproceedings{hill2025unreasonable,
  author={John C. Hill and Tyler LaBonte and Xinchen Zhang and Vidya Muthukumar},
  title={On the Unreasonable Effectiveness of Last-layer Retraining},
  booktitle={International Conference on Learning Representations (ICLR) Workshop on Spurious Correlations and Shortcut Learning (SCSL)},
  year={2025},
}
```
