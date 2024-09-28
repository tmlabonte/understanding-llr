import math
import os
import os.path as osp
import pickle
import random
import string
import sys

import matplotlib.pyplot as plt
import numpy as np

# Imports Python packages.
from configargparse import Parser
from scipy.stats import pearsonr

# Imports milkshake packages.
from milkshake.args import add_input_args

# Import PyTorch packages.
from pytorch_lightning import Trainer

plt.rcParams.update({"font.size": 14})
seeds = [1, 2, 3]

balance_erm = ["none", "upsampling", "subsetting"]
balance_retrain = ["none", "upsampling", "subsetting"]

colors = {
   "erm": "gray",      # Gray for ERM without LLR
   "none": "C1",    
   "upsampling": "C1",  # Orange
   "subsetting": "C1"    
}

labels = {
   "erm": "ERM",
   "none": "No CB",
   "upsampling": "Upsampling",
   "subsetting": "Subsetting"
}

def stats(x):
   mean = np.nanmean(x)
   std = np.nanstd(x)
   return mean, std

def get_y_limits(dataset, title, means, stds, means_no_llr, stds_no_llr):
    """Determines the y-axis limits based on the dataset and title."""

    if dataset == "civilcomments_bert" and "WGA" in title:
        return 0.50, 0.85
    elif dataset == "civilcomments_bert" and "WCA" in title:
        return 0.50, 0.9
    elif dataset == "waterbirds_convnextv2" and "WGA" in title:
        return 0.50, 0.85
    elif dataset == "waterbirds_resnet" and "WGA" in title:
        return 0.40, 0.85
    elif dataset == "celeba_convnextv2" and "WGA" in title:
        return 0.40, 0.8
    elif dataset == "celeba_convnextv2" and "WCA" in title:
        return 0.7, 1
    else:
        y_min = max(0, min([m - s for m,s in zip(means + means_no_llr,stds + stds_no_llr)]) - 0.05)
        y_max = min(1.0, max([m + s for m,s in zip(means + means_no_llr,stds+  stds_no_llr)]) + 0.05)
        return y_min, y_max

def plot_subplots(data, errors, categories, ylabel, title, output_file, dataset):
    """Plots subplots with bars and error bars for multiple balancing methods.

    This functions plots the 12 bars graph for each dataset.
    Different balance erm and different balance retrain as orange bars. (9 bars total)
    Different erms as grey bars. (3 total)

    Args:
       data (dict): Dictionary containing the means of the metrics to plot.
       errors (dict): Dictionary containing the standard deviations/errors corresponding to the data.
       categories (list): List of category labels for the x-axis.
       ylabel (str): The label for the y-axis.
       title (str): The title of the plot.
       output_file (str): Name of the file to save the figure.
       dataset (str): Dataset name, used to customize y-axis limits based on the dataset and title.
    """
   
    x = np.arange(len(categories) + 1)  # Adding one position for the leftmost ERM bar
    width = 0.63  # Reduced width to accommodate additional bars

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(title)

    # Loop through each balancing method (ERM)
    for ii, erm in enumerate(balance_erm):
        ax = axes[ii]
        means = data[erm]
        stds = errors[erm]

        means_no_llr = data[f"{erm}_no_llr"]
        stds_no_llr = errors[f"{erm}_no_llr"]

        # Use the helper function to determine y-axis limits
        y_min, y_max = get_y_limits(dataset, title, means, stds, means_no_llr, stds_no_llr)

        # Plot ERM bars
        ax.bar(x[0] - width/2, means_no_llr[0], width, yerr=stds_no_llr[0], 
            color=colors['erm'], capsize=5, label='ERM')

        # Plot balance categories
        for idx, (mean, std, cat) in enumerate(zip(means, stds, categories)):
            ax.bar(x[idx + 1] - width/2, mean, width, yerr=std, 
                color=colors[cat], capsize=5, label=labels[cat])

        # Set plot titles and labels
        ax.set_title(f"Balance ERM: {labels[erm]}")
        ax.set_xticks(x - 0.3)
        ax.set_xticklabels(["ERM"] + categories, rotation=45)
        ax.set_ylim(y_min, y_max)

        # Set y-ticks
        y_ticks = np.arange(y_min, y_max + 0.01, 0.03)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks])

        # Add grid and labels
        ax.grid(alpha=0.5, axis='y')
        if ii == 0:
            ax.set_ylabel(ylabel)  # Add y-axis label to the first subplot
        ax.set_xlabel("Balance LLR")  # Add x-axis label
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-axis numbers are shown for all subplots

    # Add legend
    handles = [
        plt.Line2D([0], [0], color=colors["erm"], lw=6, label="ERM"),
        plt.Line2D([0], [0], color=colors["none"], lw=6, label="LLR"),  
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)

    # Create output directory and save the figure
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches="tight", dpi=600)
    plt.show()
    plt.clf()

def process_dataset(epoch_for_dataset, results, dataset, categories, output_subplots, metric):
   """Processes a dataset, computes metrics, and generates bar plots or scatter plots.

    This function process the data for the 12 bar graphs,
    we find the test_aa, train_aa, test_wga, test_wca for each dataset.
    Then generates 12 bar graphs for each plot.

    Args:
       dataset (str): The name of the dataset to process.
       categories (list): List of category labels for the bar plot x-axis.
       output_subplots (str): The filename to save the bar plot figure.
       metric (str): The specific metric to evaluate. Can be 'train_aa', 'test_wca', 'test_wga', or 'test_aa'.

    Returns:
       plot_subplots() this functions plots the 12 bars graph for each dataset.
    """

   model_key = 50 if dataset in ["celeba_resnet", "waterbirds_resnet"] else "base"
   data = {erm: [] for erm in balance_erm}
   errors = {erm: [] for erm in balance_erm}
   data_no_llr = {erm: [] for erm in balance_erm}
   errors_no_llr = {erm: [] for erm in balance_erm}
   test_aa_data, test_wga_data, colors_list, labels_list = [], [], [], []
   
   for erm in balance_erm:
       erm_means, erm_stds, erm_means_no_llr, erm_stds_no_llr = [], [], [], []
       for retrain in balance_retrain:
           metric_values, metric_values_no_llr, aa_values, wga_values = [], [], [], []
           for s in seeds:
               try:
                    metric_value = results[s][model_key][erm]["llr"][retrain][epoch_for_dataset][metric]
                    metric_value_no_llr = results[s][model_key][erm]["erm"][epoch_for_dataset][metric]
                    metric_values.append(metric_value)
                    metric_values_no_llr.append(metric_value_no_llr)
                 
               except KeyError:
                   print(f"Missing data for seed {s}, model {model_key}, {erm}.")
                   metric_values.append(np.nan)
                   metric_values_no_llr.append(np.nan)
                   aa_values.append(np.nan)
                   wga_values.append(np.nan)   

           metric_mean, metric_std = stats(metric_values)      # Store the means and stds for bar plots
           metric_mean_no_llr, metric_std_no_llr = stats(metric_values_no_llr)
           erm_means.append(metric_mean)
           erm_stds.append(metric_std)
           erm_means_no_llr.append(metric_mean_no_llr)
           erm_stds_no_llr.append(metric_std_no_llr)

           if metric == "wga":    # Collect data for scatterplot
               test_aa_data.extend(aa_values)  # Adding all values for scatterplot
               test_wga_data.extend(wga_values)  # Adding all values for scatterplot
               # Set the color and label for this combination (for scatterplot)
               colors_list.extend([colors[erm]] * len(aa_values)) 
               labels_list.extend([f"{labels[erm]} - {labels[retrain]}"] * len(aa_values))

       data[erm] = erm_means   # Store data for bar plots
       errors[erm] = erm_stds
       data_no_llr[erm] = erm_means_no_llr
       errors_no_llr[erm] = erm_stds_no_llr

   combined_data = {erm: data[erm] for erm in balance_erm}
   combined_errors = {erm: errors[erm] for erm in balance_erm}

   for erm in balance_erm:
    combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
    combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

    if metric != "aa_wga":  # Plot the bar graphs
        combined_data = {erm: data[erm] for erm in balance_erm}
        combined_errors = {erm: errors[erm] for erm in balance_erm}
        for erm in balance_erm:
            combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
            combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

    ylabel = {"train_aa": "Train AA", "wca": "WCA", "test_aa": "Test AA"}.get(metric, "Test WGA")
    title = f"{dataset.replace('_', ' ').capitalize()} {ylabel}"
    plot_subplots(combined_data, combined_errors, categories, 
        ylabel, title, output_subplots, dataset)

def plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file): 
   """Plots a scatterplot comparing metrics for two models (e.g., ResNet vs ConvNeXt) with different retraining methods.

    This function takes in the llr and erm data with version convnext and version resnet, then graph it as scatterplot.
    Color_map sets the different color for each data, and markers sets different shapes for different balance erm.
    Custom_legend_labels can be used to set different legend bars.

    Args:
       x_data (list or array): The data to plot on the x-axis (e.g., ConvNeXt metrics).
       y_data (list or array): The data to plot on the y-axis (e.g., ResNet metrics).
       labels_list (list of str): List of labels indicating the ERM and retrain methods for each point.
       title (str): The title for the scatterplot.
       output_file (str): The filename to save the scatterplot.

    Returns:
        None
    """

   # Resnet vs Convnext
   fig, ax = plt.subplots(figsize=(10, 6))

   color_map = {       # Color and marker mappings
       ("none", "none"): "orange",
       ("none", "subsetting"): "orange",
       ("none", "upsampling"): "orange",
       ("upsampling", "none"): "green",
       ("upsampling", "subsetting"): "green",
       ("upsampling", "upsampling"): "green",
       ("subsetting", "none"): "blue",
       ("subsetting", "subsetting"): "blue",
       ("subsetting", "upsampling"): "blue" }

   markers = {"none": "o", "upsampling": "v", "subsetting": "s"}

   for ii, label in enumerate(labels_list):       # Plot points with annotations
       erm_method, retrain_method = label.split()[0].lower(), 
        label.split()[1].lower() if "(LLR)" in label else "none"
       color = color_map.get((erm_method, retrain_method), 'gray')
       marker = markers.get(retrain_method, 'o')
       ax.scatter(x_data[ii], y_data[ii], color=color, marker=marker, s=100)
       ax.annotate(retrain_method.capitalize(), (x_data[ii], y_data[ii]), 
        textcoords="offset points", xytext=(0, 8), ha='center', fontsize=6)

   pearson_corr, _ = pearsonr(x_data, y_data)      # Add Pearson correlation
   ax.set_title(f"{title} (Pearson: {pearson_corr:.3f})")
   ax.set_xlabel('ConvNeXt Metric')
   ax.set_ylabel('ResNet Metric')
   ax.grid(True)

   custom_legend_labels = [   # Legend setup
       ("ERM", "gray", "o"), ("LLR None", "orange", "o"),
       ("LLR Upsampling", "green", "v"), ("LLR Subsetting", "blue", "s")  ]

   for label, color, marker in custom_legend_labels:
       ax.scatter([], [], color=color, marker=marker, label=label)

   ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
    ncol=4, fontsize='small', markerscale=0.7)
   output_dir = "out"       # Save and show plot
   os.makedirs(output_dir, exist_ok=True)

   plt.savefig(os.path.join(output_dir, output_file), bbox_inches='tight', dpi=600)
   plt.show()
   plt.clf()
   plt.close()

def process_dataset_comparison(epoch_for_dataset, dataset_convnext, 
    dataset_resnet, categories,metric, title_prefix, output_prefix):
   """Processes and compares two datasets (e.g., ConvNeXt and ResNet) by extracting metrics and generating a scatterplot.

    This function takes the input from same dataset with version convnext and version resnet,
    then it process the data to find llr and erm for both versions.

   Args:
       dataset_convnext (str): The name of the ConvNeXt dataset file (without extension).
       dataset_resnet (str): The name of the ResNet dataset file (without extension).
       categories (list): List of category labels for the comparison (not used in the current function).
       metric (str): The metric to be compared (e.g., 'test_wga', 'test_aa').
       title_prefix (str): A prefix for the plot title.
       output_prefix (str): A prefix for the output filename of the scatterplot.

   Returns:
       plot_comparison_scatterplot() to plot the different versions(convnext vs resnet) for same dataset.
   """

   try:   # Resnet vs Convnext
       results_convnext = pickle.load(open(f"{dataset_convnext}.pkl", "rb"))
       results_resnet = pickle.load(open(f"{dataset_resnet}.pkl", "rb"))
       print(f"Successfully loaded {dataset_convnext}.pkl and {dataset_resnet}.pkl")
   except FileNotFoundError:
       print(f"File not found: {dataset_convnext}.pkl or {dataset_resnet}.pkl")
       return

   model_keys = {"convnext": "base", "resnet": 50}         # Define model keys and epoch rules
  
   # Iterate over metrics
   x_data, y_data, labels_list = [], [], []
   for erm in balance_erm:
       for retrain in balance_retrain:
           x_values_llr, y_values_llr, x_values_erm, y_values_erm = [], [], [], []
           for s in seeds:
               try:     # Extract LLR and ERM values
                   convnext_llr = results_convnext[s][model_keys["convnext"]][erm]["llr"][retrain][epoch_for_dataset][metric]
                   resnet_llr = results_resnet[s][model_keys["resnet"]][erm]["llr"][retrain][epoch_for_dataset][metric]
                   convnext_erm = results_convnext[s][model_keys["convnext"]][erm]["erm"][epoch_for_dataset][metric]
                   resnet_erm = results_resnet[s][model_keys["resnet"]][erm]["erm"][epoch_for_dataset][metric]
                   if not np.isnan(convnext_llr) and not np.isnan(resnet_llr):       # Collect non-NaN values
                       x_values_llr.append(convnext_llr)
                       y_values_llr.append(resnet_llr)
                   if not np.isnan(convnext_erm) and not np.isnan(resnet_erm):
                       x_values_erm.append(convnext_erm)
                       y_values_erm.append(resnet_erm)
               except KeyError:
                   print(f"Missing data for seed {s}, {erm}/{retrain}")
                   continue

           if x_values_llr and y_values_llr:       # Append averaged values to plot data
               x_data.append(np.nanmean(x_values_llr))
               y_data.append(np.nanmean(y_values_llr))
               labels_list.append(f"{retrain.capitalize()} {erm.capitalize()} (LLR)")

           if x_values_erm and y_values_erm:
               x_data.append(np.nanmean(x_values_erm))
               y_data.append(np.nanmean(y_values_erm))
               labels_list.append(f"{erm.capitalize()} (ERM)")

       title = f"{title_prefix}: ConvNeXt vs ResNet - {metric.replace('_', ' ').capitalize()}"     # Plot for each metric
       output_file = f"{output_prefix}_{metric}_comparison_scatterplot.png"
       plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file)

def plot_metric_comparison(epoch_for_dataset, results,dataset, metric_x, metric_y, 
    title, output_file, x_label, y_label, x_limit=(0.5, 1.0), y_limit=(0.5, 1.0)):
   """Plots a comparison of two metrics (e.g., AA vs WCA, AA vs WGA, or WGA vs WCA) for a given dataset.

    This function processes the llr and erm datas for given dataset.
    Then find the average for llr and erm for AA vs WCA, AA vs WGA, or WGA vs WCA plots.

   Args:
       dataset (str): The name of the dataset file (without extension) to load and analyze.
       metric_x (str): The name of the first metric (for the x-axis).
       metric_y (str): The name of the second metric (for the y-axis).
       title (str): The title for the scatterplot.
       output_file (str): The filename to save the scatterplot.
       x_label (str): The label for the x-axis.
       y_label (str): The label for the y-axis.
       x_limit (tuple, optional): The limit for the x-axis as a tuple (min, max). Default is (0.5, 1.0).
       y_limit (tuple, optional): The limit for the y-axis as a tuple (min, max). Default is (0.5, 1.0).

   Returns:
      plot_metric_scatterplot() to plot the combined plots for AA vs WCA, AA vs WGA, or WGA vs WCA.
   """

   model_key = 50 if "resnet" in dataset else "base"
   metric_x_data, metric_y_data = [], []
   colors_list, labels_list = [], []

   for erm in balance_erm:
       for retrain in balance_retrain:
           metric_x_llr, metric_y_llr, metric_x_erm, metric_y_erm = [], [], [], []
           for s in seeds:    # Collect data across seeds
               try:
                   metric_x_llr.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset][metric_x])
                   metric_y_llr.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset][metric_y])
                   metric_x_erm.append(results[s][model_key][erm]["erm"][epoch_for_dataset][metric_x])
                   metric_y_erm.append(results[s][model_key][erm]["erm"][epoch_for_dataset][metric_y])
               except KeyError:
                   print(f"Missing data for seed {s}, model {model_key}, {erm}.")
                   metric_x_llr.append(np.nan)
                   metric_y_llr.append(np.nan)
                   metric_x_erm.append(np.nan)
                   metric_y_erm.append(np.nan)

           metric_x_data.extend([np.nanmean(metric_x_llr), np.nanmean(metric_x_erm)])  # Compute averages for LLR and ERM
           metric_y_data.extend([np.nanmean(metric_y_llr), np.nanmean(metric_y_erm)])
           colors_list.extend([colors[erm], 'gray'])  # LLR color and ERM in gray
           labels_list.extend([f"{labels[erm]} / {labels[retrain]}", f"{labels[erm]} (ERM)"])

   plot_metric_scatterplot(metric_x_data, metric_y_data, colors_list, labels_list, 
    title, output_file, x_label, y_label, x_limit, y_limit)   # Plot the combined metrics

def plot_metric_scatterplot(x_data, y_data, colors_list, labels_list, 
    title, output_file, x_label, y_label, x_limit, y_limit):
   """Plots a scatterplot comparing two metrics with custom labels and colors, and calculates Pearson correlation.

    This function generates the plots for AA vs WCA and AA vs WGA and WGA vs WCA graphs.
    We can adjust the axis lim and title here.
    Also prints out the pearson correlation.

   Args:
       x_data (list or array): Data for the x-axis.
       y_data (list or array): Data for the y-axis.
       colors_list (list): List of colors for each point.
       labels_list (list): List of labels for each point, used for both plotting and annotations.
       title (str): The title for the scatterplot.
       output_file (str): The filename to save the scatterplot.
       x_label (str): The label for the x-axis.
       y_label (str): The label for the y-axis.
       x_limit (tuple): The limits for the x-axis (min, max).
       y_limit (tuple): The limits for the y-axis (min, max).

   Returns:
       None
   """

   fig, ax = plt.subplots(figsize=(15, 9))
   ax.set_xlim(x_limit)
   ax.set_ylim(y_limit)

   for ii in range(len(x_data)):  # Plot each point and label
       ax.scatter(x_data[ii], y_data[ii], color=colors_list[ii], label=labels_list[ii], s=180)
       ax.annotate(f'{labels_list[ii]}', (x_data[ii], y_data[ii]), textcoords="offset points",
                   xytext=(0, 8), ha='center', fontsize=6)
   pearson_corr, _ = pearsonr(x_data, y_data)
   print(f"Pearson correlation between {x_label} and {y_label}: {pearson_corr:.3f}")

   ax.set_title(f"{title} (Pearson: {pearson_corr:.3f})")    # Set title and labels
   ax.set_xlabel(x_label)
   ax.set_ylabel(y_label)
   ax.grid(True)

   output_dir = "out"   # Save plot
   os.makedirs(output_dir, exist_ok=True)
   plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=600)
   plt.show()

def main(args):
   """Main function for running metric comparisons and dataset processing.

   This function compares metrics between models (e.g., ConvNeXt and ResNet) and processes datasets to generate
   bar plots and scatter plots for different evaluation metrics.

   Args:
       args (Namespace): Command-line arguments passed by the user. These include:
           - datamodule (str): The name of the data module.
           - model (str): The model type to evaluate (e.g., "convnext", "resnet").
           - balance_erm (str): The ERM balancing strategy to use during training.
           - balance_retrain (str): The retraining balancing strategy to use.
           - mixture_ratio (float): Class imbalance ratio for mixture balancing strategy.
           - save_retrained_model (bool): Whether to save the retrained model outputs.
           - split (str): Which dataset split to use for training.
           - train_pct (int): Percentage of the training set to utilize.

   Returns:
       None
   """
   dataset = f"{args.datamodule}_{args.model}"
   
   try:        #Plot AA vs WCA and AA vs WGA and WGA vs WCA  
       results = pickle.load(open(f"{dataset}.pkl", "rb"))
       print(f"Successfully loaded {dataset}.pkl")
   except FileNotFoundError:
       print(f"File not found: {dataset}.pkl")
       return
   epoch_for_dataset = 100 if "waterbirds" in dataset else 20

   for metrics in (("test_aa", "test_wga"), ("test_aa", "test_wca"), ("test_wca", "test_wga")):
       plot_metric_comparison(epoch_for_dataset, results, f"{args.datamodule}_{args.model}",
            metrics[0], metrics[1], f"{args.datamodule}_{args.model}: {metrics[0]} VS {metrics[1]}",
            f"{args.datamodule}_{args.model}: {metrics[0]} VS {metrics[1]}.png", 
            f"{metrics[0]}", f"{metrics[1]}")
   for metric in ("test_aa", "test_wca", "test_wga","train_aa"):

       categories = ["none","upsampling","subsetting"]
       process_dataset(epoch_for_dataset, results,f"{args.datamodule}_{args.model}", 
            categories, f"{args.datamodule}_{args.model}: {metric}.png", metric)
       process_dataset_comparison(epoch_for_dataset, f"{args.datamodule}_convnextv2", 
            f"{args.datamodule}_resnet", categories, metric,
            f"{args.datamodule} Convnext VS Resnet 50", 
            f"{args.datamodule} Convnext VS Resnet 50.png")

if __name__ == "__main__":
   parser = Parser(
       args_for_setting_config_path=["-c", "--cfg", "--config"],
       config_arg_is_required=True,)
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
   args = parser.parse_args()
   main(args)
