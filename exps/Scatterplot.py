import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({"font.size": 16})

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

def plot_scatterplot(data_aa, data_wga, categories, title, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Iterate through each category (e.g., none, upsampling, subsetting)
    for idx, category in enumerate(categories):
        test_aa = data_aa[category]
        test_wga = data_wga[category]
        
        ax.scatter(test_aa, test_wga, label=labels[category], s=100)
        
        # Annotate points
        for i, (aa, wga) in enumerate(zip(test_aa, test_wga)):
            ax.annotate(f'Seed {i+1}', (aa, wga), textcoords="offset points", xytext=(0, 10), ha='center')
    
    ax.set_title(title)
    ax.set_xlabel('Test AA (Average Accuracy)')
    ax.set_ylabel('Test WGA (Worst-Group Accuracy)')
    ax.legend()
    ax.grid(True)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()
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
        print(f"Successfully loaded {dataset}.pkl")  # Debug print
    except FileNotFoundError:
        print(f"File not found: {dataset}.pkl")
        return

    # Determine the key to use for the model version
    model_key = 50 if dataset in ["celeba_resnet", "waterbirds_resnet"] else "base"

    # Prepare data for subplots
    data = {erm: [] for erm in balance_erm}
    errors = {erm: [] for erm in balance_erm}
    data_no_llr = {erm: [] for erm in balance_erm}
    errors_no_llr = {erm: [] for erm in balance_erm}

    test_aa_data = {erm: [] for erm in balance_erm}
    test_wga_data = {erm: [] for erm in balance_erm}

    epoch_for_dataset = 100 if dataset in ["waterbirds_resnet", "waterbirds_convnextv2"] else 20

    for erm in balance_retrain:
        erm_means = []
        erm_stds = []
        erm_means_no_llr = []
        erm_stds_no_llr = []
        for retrain in balance_retrain:
            metric_values = []
            metric_values_no_llr = []
            for s in seeds:
                try:
                    if metric == "train_aa":
                        metric_values.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["train_aa"])
                        metric_values_no_llr.append(results[s][model_key][erm]["erm"][epoch_for_dataset]["train_aa"])
                    elif metric == "wca":
                        metric_values.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_acc_by_class"])
                        metric_values_no_llr.append(results[s][model_key][erm]["erm"][epoch_for_dataset]["test_acc_by_class"])
                    elif metric == "wga":
                        metric_values.append(results[s][model_key][erm]["llr"][retrain][epoch_for_dataset]["test_wga"])
                        metric_values_no_llr.append(results[s][model_key][erm]["erm"][epoch_for_dataset]["test_wga"])
                except KeyError:
                    metric_values.append(np.nan)
                    metric_values_no_llr.append(np.nan)
            metric_mean, metric_std = stats(metric_values)
            metric_mean_no_llr, metric_std_no_llr = stats(metric_values_no_llr)
            erm_means.append(metric_mean)
            erm_stds.append(metric_std)
            erm_means_no_llr.append(metric_mean_no_llr)
            erm_stds_no_llr.append(metric_std_no_llr)

            # Gather test AA and test WGA data for scatterplot
            if metric == "train_aa":
                test_aa_data[erm].append(metric_mean)
            elif metric == "wga":
                test_wga_data[erm].append(metric_mean)

        data[erm] = erm_means
        errors[erm] = erm_stds
        data_no_llr[erm] = erm_means_no_llr
        errors_no_llr[erm] = erm_stds_no_llr

    combined_data = {erm: data[erm] for erm in balance_erm}
    combined_errors = {erm: errors[erm] for erm in balance_erm}
    for erm in balance_erm:
        combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
        combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

    ylabel = "Train AA" if metric == "train_aa" else "WCA" if metric == "wca" else "Test WGA"
    title = f"{dataset.replace('_', ' ').capitalize()} {ylabel}"
    plot_subplots(combined_data, combined_errors, categories, ylabel, title, output_subplots, dataset)

    # Plot scatterplot of Test AA vs WGA
    if metric == "wga":
        plot_scatterplot(test_aa_data, test_wga_data, categories, f"{dataset} Test AA vs WGA", f"{dataset}_scatterplot.png")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <metric>")
        print("Where <metric> is either 'train_aa', 'wca', or 'wga'")
        return

    metric = sys.argv[1]
    if metric not in ['train_aa', 'wca', 'wga']:
        print("Invalid metric. Use 'train_aa', 'wca', or 'wga'.")
        return

    process_dataset("celeba_resnet", ["none", "upsampling", "subsetting"], 
                    f"celeba_Resnet {'TrainAA' if metric == 'train_aa' else 'WCA' if metric == 'wca' else 'WGA'}", metric)
    process_dataset("waterbirds_resnet", ["none", "upsampling", "subsetting"], 
                    f"Waterbirds_Resnet {'TrainAA' if metric == 'train_aa' else 'WCA' if metric == 'wca' else 'WGA'}", metric)
    process_dataset("civilcomments_bert", ["none", "upsampling", "subsetting"], 
                    f"CivilComments_Bert {'TrainAA' if metric == 'train_aa' else 'WCA' if metric == 'wca' else 'WGA'}", metric)
    process_dataset("waterbirds_convnextv2", ["none", "upsampling", "subsetting"], 
                    f"Waterbirds_ConvNeXtV2 {'TrainAA' if metric == 'train_aa' else 'WCA' if metric == 'wca' else 'WGA'}", metric)
    process_dataset("celeba_convnextv2", ["none", "upsampling", "subsetting"], 
                    f"Celeba_ConvNeXtV2 {'TrainAA' if metric == 'train_aa' else 'WCA' if metric == 'wca' else 'WGA'}", metric)

if __name__ == "__main__":
    main()
