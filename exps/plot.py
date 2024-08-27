#python exps/plot.py train_aa
# python exps/plot.py -c cfgs/waterbirds.yaml --model resnet --resnet_version 50 --exp llr
# python exps/plot.py -c cfgs/waterbirds.yaml --exp llr
# python exps/plot.py -c cfgs/waterbirds.yaml --model resnet --resnet_version 50 --exp layerwise
# --exp spectral
# --exp margin
# def layerwise():
# plot("wga")
# plot("aa")
#python exps/plot.py wga
#python exps/plot.py wca
# python exps/plot.py layerwise --all
# python exps/plot.py 
# --model resnet --resnet_version 50

import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import string
import math
  

plt.rcParams.update({"font.size": 14})

# Configuration
seeds = [1, 2, 3]
balance_erm = ["none", "upsampling", "subsetting"]
balance_retrain = ["none", "upsampling", "subsetting"]
epochs = np.arange(2, 21, 2)
colors = {
    "erm": "gray",      # Gray for ERM without LLR
    "none": "C1",     
    "upsampling": "C1",  # Orange
    "subsetting": "C1"    
}
labels = {
    "erm": "ERM",
    "none": "None",
    "upsampling": "Upsampling",
    "subsetting": "Subsetting"
}

def stats(x):
    mean = np.nanmean(x)
    std = np.nanstd(x)
    return mean, std


def plot_subplots(data, errors, categories, ylabel, title, output_file, dataset):
    x = np.arange(len(categories) + 1)  # Adding one position for the leftmost ERM bar
    width = 0.63  # Reduced width to accommodate additional bars

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(title)

    for i, erm in enumerate(balance_erm):
        ax = axes[i]
        means = data[erm]
        stds = errors[erm]
        means_no_llr = data[f"{erm}_no_llr"]
        stds_no_llr = errors[f"{erm}_no_llr"]

        # Apply specific y-axis range for Civil Comments WGA plot
        if dataset == "civilcomments_bert" and "WGA" in title:
            y_min, y_max = 0.50, 0.85  # Customize this range as per your requirements
        elif dataset == "waterbirds_convnextv2" and "WGA" in title:
            y_min, y_max = 0.50, 0.85
        elif dataset == "waterbirds_resnet" and "WGA" in title:
            y_min, y_max = 0.40, 0.85
        elif dataset == "celeba_convnextv2" and "WGA" in title:
            y_min, y_max = 0.40, 0.8
        else:
            # Calculate dynamic y-limits based on mean ± std with a ±0.05 margin
            y_min = max(0, min([m - s for m, s in zip(means + means_no_llr, stds + stds_no_llr)]) - 0.05)
            y_max = min(1.0, max([m + s for m, s in zip(means + means_no_llr, stds + stds_no_llr)]) + 0.05)

        ax.bar(x[0] - width/2, means_no_llr[0], width, yerr=stds_no_llr[0], color=colors['erm'], capsize=5, label='ERM')
        for idx, (mean, std, cat) in enumerate(zip(means, stds, categories)):
            ax.bar(x[idx + 1] - width/2, mean, width, yerr=std, color=colors[cat], capsize=5, label=labels[cat])

        ax.set_title(f"Balance ERM: {labels[erm]}")
        ax.set_xticks(x-0.3)
        ax.set_xticklabels(["ERM"] + categories, rotation=45)

        # Set dynamic y-axis limits
        ax.set_ylim(y_min, y_max)

        # Set y-axis ticks at every 0.05 interval within the dynamic range
        y_ticks = np.arange(y_min, y_max + 0.01, 0.03)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks])

        ax.grid(alpha=0.5, axis='y')

        if i == 0:
            ax.set_ylabel(ylabel)  # Add y-axis label to the first subplot
        ax.set_xlabel("Balance LLR")  # Add x-axis label
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-axis numbers are shown for all subplots

    handles = [
        plt.Line2D([0], [0], color=colors['erm'], lw=6, label='ERM'),
        plt.Line2D([0], [0], color=colors['none'], lw=6, label='LLR'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

def process_dataset(dataset, categories, output_subplots, metric):
    try:
        with open(f"{dataset}.pkl", "rb") as f:
            results = pickle.load(f)
        print(f"Successfully loaded {dataset}.pkl")
    except FileNotFoundError:
        print(f"File not found: {dataset}.pkl")
        return

    model_key = 50 if dataset in ["celeba_resnet", "waterbirds_resnet"] else "base"

    # Prepare data for subplots and scatterplot
    data = {erm: [] for erm in balance_erm}
    errors = {erm: [] for erm in balance_erm}
    data_no_llr = {erm: [] for erm in balance_erm}
    errors_no_llr = {erm: [] for erm in balance_erm}

    test_aa_data = []
    test_wga_data = []
    colors_list = []
    labels_list = []

    epoch_for_dataset = 100 if dataset in ["waterbirds_resnet", "waterbirds_convnextv2"] else 20

    # Loop over balance ERM and balance retrain combinations
    for erm in balance_erm:
        erm_means = []
        erm_stds = []
        erm_means_no_llr = []
        erm_stds_no_llr = []

        for retrain in balance_retrain:
            metric_values = []
            metric_values_no_llr = []
            aa_values = []
            wga_values = []

            for s in seeds:
                try:
                    # Get metric values for the current configuration
                    if metric == "train_aa":
                        metric_value = results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["train_aa"]
                        metric_value_no_llr = results[s][model_key][erm]["erm"][epoch_for_dataset]["train_aa"]
                    elif metric == "wca":
                        metric_value = results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_acc_by_class"]
                        metric_value_no_llr = results[s][model_key][erm]["erm"][epoch_for_dataset]["test_acc_by_class"]
                    elif metric == "wga":
                        metric_value = results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_wga"]
                        metric_value_no_llr = results[s][model_key][erm]["erm"][epoch_for_dataset]["test_wga"]
                    elif metric == "test_aa":
                        metric_value = results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_aa"]
                        metric_value_no_llr = results[s][model_key][erm]["erm"][epoch_for_dataset]["test_aa"]

                    metric_values.append(metric_value)
                    metric_values_no_llr.append(metric_value_no_llr)

                    # Also populate data for scatterplots (test_wga and test_aa)
                    if metric == "wga":
                        aa_values.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_acc"])
                        wga_values.append(metric_value)

                except KeyError:
                    print(f"Missing data for seed {s}, model {model_key}, {erm}.")
                    metric_values.append(np.nan)
                    metric_values_no_llr.append(np.nan)
                    aa_values.append(np.nan)
                    wga_values.append(np.nan)

            # Store the means and stds for bar plots
            metric_mean, metric_std = stats(metric_values)
            metric_mean_no_llr, metric_std_no_llr = stats(metric_values_no_llr)

            erm_means.append(metric_mean)
            erm_stds.append(metric_std)
            erm_means_no_llr.append(metric_mean_no_llr)
            erm_stds_no_llr.append(metric_std_no_llr)

            # Collect data for scatterplot
            if metric == "wga":
                test_aa_data.extend(aa_values)  # Adding all values for scatterplot
                test_wga_data.extend(wga_values)  # Adding all values for scatterplot

                # Set the color and label for this combination (for scatterplot)
                colors_list.extend([colors[erm]] * len(aa_values))
                labels_list.extend([f"{labels[erm]} - {labels[retrain]}"] * len(aa_values))

        # Store data for bar plots
        data[erm] = erm_means
        errors[erm] = erm_stds
        data_no_llr[erm] = erm_means_no_llr
        errors_no_llr[erm] = erm_stds_no_llr

    combined_data = {erm: data[erm] for erm in balance_erm}
    combined_errors = {erm: errors[erm] for erm in balance_erm}
    for erm in balance_erm:
        combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
        combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

    ylabel = "Train AA" if metric == "train_aa" else "WCA" if metric == "wca" else "Test AA" if metric == "test_aa" else "Test WGA" 
    title = f"{dataset.replace('_', ' ').capitalize()} {ylabel}"
    plot_subplots(combined_data, combined_errors, categories, ylabel, title, output_subplots, dataset)

    # Plot the bar graphs
    if metric != "aa_wga":
        combined_data = {erm: data[erm] for erm in balance_erm}
        combined_errors = {erm: errors[erm] for erm in balance_erm}
        for erm in balance_erm:
            combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
            combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

        ylabel = "Train AA" if metric == "train_aa" else "WCA" if metric == "wca" else "Test AA" if metric == "test_aa" else "Test WGA"
        title = f"{dataset.replace('_', ' ').capitalize()} {ylabel}"
        plot_subplots(combined_data, combined_errors, categories, ylabel, title, output_subplots, dataset)

    # Plot the scatterplot for aa_wga
    if metric == "aa_wga":
        plot_scatterplot(test_wga_data, test_aa_data, colors_list, labels_list, f"{dataset} Test WGA vs AA", f"{dataset}_scatterplot.png")

def process_dataset_for_scatterplot(dataset, categories, metric):
    try:
        with open(f"{dataset}.pkl", "rb") as f:
            results = pickle.load(f)
        print(f"Successfully loaded {dataset}.pkl")
    except FileNotFoundError:
        print(f"File not found: {dataset}.pkl")
        return

    model_key = 50 if dataset in ["celeba_resnet", "waterbirds_resnet"] else "base"

    test_aa_data = []
    test_wga_data = []
    colors_list = []
    labels_list = []

    epoch_for_dataset = 100 if dataset in ["waterbirds_resnet", "waterbirds_convnextv2"] else 20

    # Loop over balance ERM and balance retrain combinations
    for erm in balance_erm:
        for retrain in balance_retrain:
            aa_values_llr = []
            wga_values_llr = []
            aa_values_erm = []
            wga_values_erm = []

            for s in seeds:
                try:
                    # Collect LLR data for each seed
                    aa_values_llr.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_aa"])
                    wga_values_llr.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_wga"])
                    
                    # Collect ERM data for each seed
                    aa_values_erm.append(results[s][model_key][erm]["erm"][epoch_for_dataset]["test_aa"])
                    wga_values_erm.append(results[s][model_key][erm]["erm"][epoch_for_dataset]["test_wga"])

                except KeyError:
                    print(f"Missing data for seed {s}, model {model_key}, {erm}.")
                    aa_values_llr.append(np.nan)
                    wga_values_llr.append(np.nan)
                    aa_values_erm.append(np.nan)
                    wga_values_erm.append(np.nan)

            # Compute average across the 3 seeds for LLR
            aa_mean_llr = np.nanmean(aa_values_llr)
            wga_mean_llr = np.nanmean(wga_values_llr)

            # Compute average across the 3 seeds for ERM
            aa_mean_erm = np.nanmean(aa_values_erm)
            wga_mean_erm = np.nanmean(wga_values_erm)

            # Collect the averaged LLR data for scatterplot
            test_aa_data.append(aa_mean_llr)
            test_wga_data.append(wga_mean_llr)
            colors_list.append(colors[erm])
            labels_list.append(f"{labels[erm]} / {labels[retrain]}")

            # Collect the averaged ERM data for scatterplot
            test_aa_data.append(aa_mean_erm)
            test_wga_data.append(wga_mean_erm)
            colors_list.append('gray')  # Different color for ERM
            labels_list.append(f"{labels[erm]}")  # Updated label for ERM

    # Call the plot function with the correct number of arguments
    plot_scatterplot(test_wga_data, test_aa_data, colors_list, labels_list, f"{dataset} Test WGA vs AA (Average)", f"{dataset}_scatterplot.png")




def plot_scatterplot(test_wga_data, test_aa_data, colors_list, labels_list, title, output_file):
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot each point and label
    for i in range(len(test_wga_data)):
        ax.scatter(test_wga_data[i], test_aa_data[i], color=colors_list[i], label=labels_list[i], s=180)
        ax.annotate(f'{labels_list[i]}', (test_wga_data[i], test_aa_data[i]), textcoords="offset points", 
                    xytext=(0, 8), ha='center', fontsize=6)

    ax.set_title(title)
    ax.set_xlabel('Test WGA (Worst-Group Accuracy)')
    ax.set_ylabel('Test AA (Average Accuracy)')

    # Avoid duplicate labels in the legend
    #handles, labels = ax.get_legend_handles_labels()
    #unique_labels = dict(zip(labels, handles))
    #ax.legend(unique_labels.values(), unique_labels.keys())

    ax.grid(True)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()

def plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the color map based on (ERM, LLR) combinations
    color_map = {
        ("none", "none"): "orange",       # none/none -> Orange
        ("none", "subsetting"): "orange", # none/subsetting -> Orange
        ("none", "upsampling"): "orange", # none/upsampling -> Orange
        ("upsampling", "none"): "green",  # upsampling/none -> Green
        ("upsampling", "subsetting"): "green", # upsampling/subsetting -> Green
        ("upsampling", "upsampling"): "green", # upsampling/upsampling -> Green
        ("subsetting", "none"): "blue",   # subsetting/none -> Blue
        ("subsetting", "subsetting"): "blue", # subsetting/subsetting -> Blue
        ("subsetting", "upsampling"): "blue", # subsetting/upsampling -> Blue
    }

    # Markers for LLR methods
    markers = {"none": "o", "upsampling": "v", "subsetting": "s"}

    # Track legend entries to avoid duplicates
    legend_entries = {}

    # Plot each point and label
    for i in range(len(x_data)):
        label = labels_list[i]
        
        if "(LLR)" in label:
            # Extract the retrain and erm methods from the label
            retrain_method = label.split()[0].lower()  # Extract the retrain method
            erm_method = label.split()[1].lower()  # Extract the erm method

            # Use the (erm_method, retrain_method) combination for color
            color = color_map.get((erm_method, retrain_method), 'orange')  # Default to orange if not found
            marker = markers.get(retrain_method, 'o')  # Use the marker based on the retrain method
            
            # Plot with specific marker and color
            ax.scatter(x_data[i], y_data[i], color=color, marker=marker, s=100)
            
            # Add the method to the legend if it's not already there
            if (erm_method, retrain_method) not in legend_entries:
                legend_entries[(erm_method, retrain_method)] = ax.scatter([], [], color=color, marker=marker, label=f"ERM {erm_method.capitalize()} / LLR {retrain_method.capitalize()}")
            
            # Annotate LLR dots with the correct retrain method
            ax.annotate(f'{retrain_method.capitalize()}', (x_data[i], y_data[i]), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=6)
        else:
            # ERM points, colored gray
            ax.scatter(x_data[i], y_data[i], color="gray", s=100)
            
            # Annotate ERM dots with 'ERM'
            ax.annotate(f'ERM', (x_data[i], y_data[i]), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=6)

            # Add ERM to the legend if it's not already there
            if "ERM" not in legend_entries:
                legend_entries["ERM"] = ax.scatter([], [], color="gray", marker='o', label="ERM")

    ax.set_title(title)
    ax.set_xlabel('ConvNeXt Metric')
    ax.set_ylabel('ResNet Metric')

    ax.grid(True)

    # Create a smaller legend at the top of the plot
    ax.legend(handles=legend_entries.values(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,
              fontsize='small', markerscale=0.6)  # Adjust fontsize and markerscale for a smaller legend

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()

    # Clear the figure to free memory
    plt.clf()
    plt.close()


def process_dataset_comparison(dataset_convnext, dataset_resnet, categories, metrics, title_prefix, output_prefix):
    try:
        with open(f"{dataset_convnext}.pkl", "rb") as f:
            results_convnext = pickle.load(f)
        with open(f"{dataset_resnet}.pkl", "rb") as f:
            results_resnet = pickle.load(f)
        print(f"Successfully loaded {dataset_convnext}.pkl and {dataset_resnet}.pkl")
    except FileNotFoundError:
        print(f"File not found: {dataset_convnext}.pkl or {dataset_resnet}.pkl")
        return

    model_key_convnext = "base"
    model_key_resnet = 50

    epoch_for_dataset = 100 if dataset_convnext in ["waterbirds_resnet", "waterbirds_convnextv2"] else 20

    # Loop over the 4 metrics to create individual scatterplots
    for metric in metrics:
        x_data = []
        y_data = []
        labels_list = []

        # Loop over balance ERM and balance retrain combinations
        for erm in balance_erm:
            for retrain in balance_retrain:
                x_values_llr = []
                y_values_llr = []
                x_values_erm = []
                y_values_erm = []

                for s in seeds:
                    try:
                        # Collect ConvNeXt and ResNet LLR data for each seed
                        convnext_value_llr = results_convnext[s][model_key_convnext][erm]["llr"][retrain][epoch_for_dataset][metric]
                        resnet_value_llr = results_resnet[s][model_key_resnet][erm]["llr"][retrain][epoch_for_dataset][metric]

                        # Collect ConvNeXt and ResNet ERM data for each seed
                        convnext_value_erm = results_convnext[s][model_key_convnext][erm]["erm"][epoch_for_dataset][metric]
                        resnet_value_erm = results_resnet[s][model_key_resnet][erm]["erm"][epoch_for_dataset][metric]

                        # Only add values if data is not missing
                        if not np.isnan(convnext_value_llr) and not np.isnan(resnet_value_llr):
                            x_values_llr.append(convnext_value_llr)
                            y_values_llr.append(resnet_value_llr)
                        else:
                            print(f"Missing LLR data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")

                        if not np.isnan(convnext_value_erm) and not np.isnan(resnet_value_erm):
                            x_values_erm.append(convnext_value_erm)
                            y_values_erm.append(resnet_value_erm)
                        else:
                            print(f"Missing ERM data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")

                    except KeyError:
                        print(f"KeyError: Missing data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")
                        continue

                # Compute average across the 3 seeds for both LLR and ERM if data is available
                if x_values_llr and y_values_llr:
                    x_mean_llr = np.nanmean(x_values_llr)
                    y_mean_llr = np.nanmean(y_values_llr)

                    # Collect the averaged LLR data for scatterplot
                    x_data.append(x_mean_llr)
                    y_data.append(y_mean_llr)
                    labels_list.append(f"{retrain.capitalize()} {erm.capitalize()} (LLR)")  # Use the retrain and erm method for LLR labeling
                
                if x_values_erm and y_values_erm:
                    x_mean_erm = np.nanmean(x_values_erm)
                    y_mean_erm = np.nanmean(y_values_erm)

                    # Collect the averaged ERM data for scatterplot
                    x_data.append(x_mean_erm)
                    y_data.append(y_mean_erm)
                    labels_list.append(f"{erm.capitalize()} (ERM)")  # Use the erm method for ERM labeling

        # Generate the scatterplot for the current metric
        title = f"{title_prefix}: ConvNeXt vs ResNet - {metric.replace('_', ' ').capitalize()}"
        output_file = f"{output_prefix}_{metric}_comparison_scatterplot.png"
        plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file)


def plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file):
    

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the color map based on (ERM, LLR) combinations
    color_map = {
        ("none", "none"): "orange",       # none/none -> Orange
        ("none", "subsetting"): "orange", # none/subsetting -> Orange
        ("none", "upsampling"): "orange", # none/upsampling -> Orange
        ("upsampling", "none"): "green",  # upsampling/none -> Green
        ("upsampling", "subsetting"): "green", # upsampling/subsetting -> Green
        ("upsampling", "upsampling"): "green", # upsampling/upsampling -> Green
        ("subsetting", "none"): "blue",   # subsetting/none -> Blue
        ("subsetting", "subsetting"): "blue", # subsetting/subsetting -> Blue
        ("subsetting", "upsampling"): "blue", # subsetting/upsampling -> Blue
    }

    # Markers for LLR methods
    markers = {"none": "o", "upsampling": "v", "subsetting": "s"}

    # Plot each point and label
    for i in range(len(x_data)):
        label = labels_list[i]
        
        if "(LLR)" in label:
            # Extract the retrain and erm methods from the label
            retrain_method = label.split()[0].lower()  # Extract the retrain method
            erm_method = label.split()[1].lower()  # Extract the erm method

            # Use the (erm_method, retrain_method) combination for color
            color = color_map.get((erm_method, retrain_method), 'orange')  # Default to orange if not found
            marker = markers.get(retrain_method, 'o')  # Use the marker based on the retrain method
            
            # Plot with specific marker and color
            ax.scatter(x_data[i], y_data[i], color=color, marker=marker, s=100)
            
            # Annotate LLR dots with the correct retrain method
            ax.annotate(f'{retrain_method.capitalize()}', (x_data[i], y_data[i]), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=6)
        else:
            # ERM points, colored gray
            erm_method = label.split()[0].lower()  # Extract the erm method from the label
            ax.scatter(x_data[i], y_data[i], color="gray", s=100)
            
            # Annotate ERM dots with 'none', 'subsetting', or 'upsampling'
            ax.annotate(f'{erm_method.capitalize()}', (x_data[i], y_data[i]), textcoords="offset points", 
                        xytext=(0, 8), ha='center', fontsize=6)  # Label ERM dots with their specific balance_erm method

    ax.set_title(title)
    ax.set_xlabel('ConvNeXt Metric')
    ax.set_ylabel('ResNet Metric')

    ax.grid(True)

    # Create a custom legend manually
    custom_legend_labels = [
       # ("ERM None", "gray", "o"),          # Custom entry for ERM None
       # ("ERM Subsetting", "gray", "s"),    # Custom entry for ERM Subsetting
       # ("ERM Upsampling", "gray", "v"),    # Custom entry for ERM Upsampling
        ("ERM ", "gray", "o"),
        ("LLR None", "orange", "o"),        # Custom entry for LLR None
        ("LLR Upsampling", "green", "o"),   # Custom entry for LLR Upsampling
        ("LLR Subsetting", "blue", "o")     # Custom entry for LLR Subsetting
    ]

    # Manually add legend based on custom_legend_labels
    for label, color, marker in custom_legend_labels:
        ax.scatter([], [], color=color, marker=marker, label=label)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize='small', markerscale=0.7)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()

    # Clear the figure to free memory
    plt.clf()
    plt.close()


def process_dataset_comparison(dataset_convnext, dataset_resnet, categories, metrics, title_prefix, output_prefix):
    try:
        with open(f"{dataset_convnext}.pkl", "rb") as f:
            results_convnext = pickle.load(f)
        with open(f"{dataset_resnet}.pkl", "rb") as f:
            results_resnet = pickle.load(f)
        print(f"Successfully loaded {dataset_convnext}.pkl and {dataset_resnet}.pkl")
    except FileNotFoundError:
        print(f"File not found: {dataset_convnext}.pkl or {dataset_resnet}.pkl")
        return

    model_key_convnext = "base"
    model_key_resnet = 50

    epoch_for_dataset = 100 if dataset_convnext in ["waterbirds_resnet", "waterbirds_convnextv2"] else 20

    # Loop over the 4 metrics to create individual scatterplots
    for metric in metrics:
        x_data = []
        y_data = []
        labels_list = []

        # Loop over balance ERM and balance retrain combinations
        for erm in balance_erm:
            for retrain in balance_retrain:
                x_values_llr = []
                y_values_llr = []
                x_values_erm = []
                y_values_erm = []

                for s in seeds:
                    try:
                        # Collect ConvNeXt and ResNet LLR data for each seed
                        convnext_value_llr = results_convnext[s][model_key_convnext][erm]["llr"][retrain][epoch_for_dataset][metric]
                        resnet_value_llr = results_resnet[s][model_key_resnet][erm]["llr"][retrain][epoch_for_dataset][metric]

                        # Collect ConvNeXt and ResNet ERM data for each seed
                        convnext_value_erm = results_convnext[s][model_key_convnext][erm]["erm"][epoch_for_dataset][metric]
                        resnet_value_erm = results_resnet[s][model_key_resnet][erm]["erm"][epoch_for_dataset][metric]

                        # Only add values if data is not missing
                        if not np.isnan(convnext_value_llr) and not np.isnan(resnet_value_llr):
                            x_values_llr.append(convnext_value_llr)
                            y_values_llr.append(resnet_value_llr)
                        else:
                            print(f"Missing LLR data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")

                        if not np.isnan(convnext_value_erm) and not np.isnan(resnet_value_erm):
                            x_values_erm.append(convnext_value_erm)
                            y_values_erm.append(resnet_value_erm)
                        else:
                            print(f"Missing ERM data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")

                    except KeyError:
                        print(f"KeyError: Missing data for seed {s}, model {model_key_convnext}/{model_key_resnet}, {erm}/{retrain}")
                        continue

                # Compute average across the 3 seeds for both LLR and ERM if data is available
                if x_values_llr and y_values_llr:
                    x_mean_llr = np.nanmean(x_values_llr)
                    y_mean_llr = np.nanmean(y_values_llr)

                    # Collect the averaged LLR data for scatterplot
                    x_data.append(x_mean_llr)
                    y_data.append(y_mean_llr)
                    labels_list.append(f"{retrain.capitalize()} {erm.capitalize()} (LLR)")  # Use the retrain and erm method for LLR labeling
                
                if x_values_erm and y_values_erm:
                    x_mean_erm = np.nanmean(x_values_erm)
                    y_mean_erm = np.nanmean(y_values_erm)

                    # Collect the averaged ERM data for scatterplot
                    x_data.append(x_mean_erm)
                    y_data.append(y_mean_erm)
                    labels_list.append(f"{erm.capitalize()} (ERM)")  # Use the erm method for ERM labeling

        # Generate the scatterplot for the current metric
        title = f"{title_prefix}: ConvNeXt vs ResNet - {metric.replace('_', ' ').capitalize()}"
        output_file = f"{output_prefix}_{metric}_comparison_scatterplot.png"
        plot_comparison_scatterplot(x_data, y_data, labels_list, title, output_file)




def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <metric>")
        print("Where <metric> is either 'train_aa', 'test_aa', 'wca', 'wga', 'conv_resnet', or 'aa_wga'")
        return

    metric = sys.argv[1]
    
    if metric == 'aa_wga':
        # Scatterplots with averaged seeds for various datasets
        process_dataset_for_scatterplot("celeba_resnet", ["none", "upsampling", "subsetting"], metric)
        process_dataset_for_scatterplot("celeba_convnextv2", ["none", "upsampling", "subsetting"], metric)
        process_dataset_for_scatterplot("waterbirds_resnet", ["none", "upsampling", "subsetting"], metric)
        process_dataset_for_scatterplot("waterbirds_convnextv2", ["none", "upsampling", "subsetting"], metric)
        process_dataset_for_scatterplot("civilcomments_bert", ["none", "upsampling", "subsetting"], metric)

    elif metric == 'conv_resnet':
        # Comparison scatterplots for ConvNeXt vs ResNet
        process_dataset_comparison("waterbirds_convnextv2", "waterbirds_resnet", ["none", "upsampling", "subsetting"], 
                           ['test_aa', 'test_wga', 'test_wca', 'train_aa'], "Waterbirds: ConvNeXt vs ResNet", "waterbirds")
        process_dataset_comparison("celeba_convnextv2", "celeba_resnet", ["none", "upsampling", "subsetting"], 
                           ['test_aa', 'test_wga', 'test_wca', 'train_aa'], "Celeba: ConvNeXt vs ResNet", "celeba")

    else:
        # Bar plots and subplots for various metrics across datasets
        process_dataset("celeba_resnet", ["none", "upsampling", "subsetting"], 
                        f"celeba_Resnet {'TrainAA' if metric == 'train_aa'  else 'WCA' if metric == 'wca' else 'Test AA' if metric == 'test_aa' else 'WGA'}", metric)
        process_dataset("waterbirds_resnet", ["none", "upsampling", "subsetting"], 
                        f"Waterbirds_Resnet {'TrainAA' if metric == 'train_aa'  else 'WCA' if metric == 'wca' else 'Test AA' if metric == 'test_aa' else 'WGA'}", metric)
        process_dataset("civilcomments_bert", ["none", "upsampling", "subsetting"], 
                        f"CivilComments_Bert {'TrainAA' if metric == 'train_aa' else  'WCA' if metric == 'wca' else 'Test AA' if metric == 'test_aa' else 'WGA'}", metric)
        process_dataset("waterbirds_convnextv2", ["none", "upsampling", "subsetting"], 
                        f"Waterbirds_ConvNeXtV2 {'TrainAA' if metric == 'train_aa' else   'WCA' if metric == 'wca' else 'Test AA' if metric == 'test_aa' else 'WGA'}", metric)
        process_dataset("celeba_convnextv2", ["none", "upsampling", "subsetting"], 
                        f"Celeba_ConvNeXtV2 {'TrainAA' if metric == 'train_aa'  else 'WCA' if metric == 'wca' else 'Test AA' if metric == 'test_aa' else 'WGA'}", metric)

if __name__ == "__main__":
    main()
