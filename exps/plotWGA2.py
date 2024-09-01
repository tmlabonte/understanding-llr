import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
        print(f"Plotting data for Balance ERM: {labels[erm]}")
        print(f"Means: {means}")
        print(f"Stds: {stds}")
        ax.bar(x[0] - width/2, means_no_llr[0], width, yerr=stds_no_llr[0], color=colors['erm'], capsize=5, label='ERM')
        for idx, (mean, std, cat) in enumerate(zip(means, stds, categories)):
            ax.bar(x[idx + 1] - width/2, mean, width, yerr=std, color=colors[cat], capsize=5, label=labels[cat])

        ax.set_title(f"Balance ERM: {labels[erm]}")
        ax.set_xticks(x-0.3)
        ax.set_xticklabels(["ERM"] + categories, rotation=45)

        if dataset == "celeba_resnet":
            ax.set_yticks(np.arange(0.3, 0.9, 0.05))  # Adjusted y-axis ticks for CelebA
            ax.set_ylim(0.3, 0.8)  # Set y-axis range for CelebA
        else:
            ax.set_yticks(np.arange(0.5, 1, 0.05))  # Adjusted y-axis ticks for Waterbirds
            ax.set_ylim(0.5, 0.9)  # Set y-axis range for Waterbirds

        ax.grid(alpha=0.5, axis='y')

        if i == 0:
            ax.set_ylabel(ylabel)  # Add y-axis label to the first subplot
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


"""
def plot_line_graphs(dataset, results, balance_erm, balance_retrain, epochs, title, output_file):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title)
    
    for i, erm in enumerate(balance_erm):
        ax = axes[i]
        print(f"Processing Balance ERM: {erm}")  # Debug print
        for retrain in balance_retrain:
            label = f"Retrain: {labels[retrain]}"
            wga = np.zeros(len(epochs))
            stds = np.zeros(len(epochs))
            for j, e in enumerate(epochs):
                wga_values = []
                for s in seeds:
                    try:
                        wga_values.append(results[50][erm]["llr"][retrain][e]["test_wga"])
                    except KeyError:
                        continue
                wga[j], stds[j] = stats(wga_values)
            ax.plot(epochs, wga, label=label, linewidth=3)
            ax.fill_between(epochs, wga - stds, wga + stds, alpha=0.3)

        ax.set_xlabel("Epochs")
        if i == 0:
            ax.set_ylabel("Test WGA")
        ax.set_title(f"Balance ERM: {labels[erm]}")
        ax.grid(alpha=0.5, axis='y')
        ax.set_yticks(np.arange(0.3, 0.9, 0.05))  # Adjust y-axis ticks for better granularity
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-axis numbers are shown for all subplots

    # Set y-limits manually for each subplot
    if dataset == "celeba":
        axes[0].set_ylim(0.4, 0.65)  # For the first subplot
        axes[1].set_ylim(0.4, 0.65)  # For the second subplot
        axes[2].set_ylim(0.65, 0.8)  # For the third subplot
    elif dataset == "civilcomments":
        axes[0].set_ylim(0.3, 0.85)  # For the first subplot
        axes[1].set_ylim(0.3, 0.85)  # For the second subplot
        axes[2].set_ylim(0.3, 0.85)  # For the third subplot


    handles = [
        plt.Line2D([0], [0], color='C0', lw=6, label=labels[balance_retrain[0]]),
        plt.Line2D([0], [0], color='C1', lw=6, label=labels[balance_retrain[1]]),
        plt.Line2D([0], [0], color='C2', lw=6, label=labels[balance_retrain[2]])
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()
"""

def process_dataset(dataset, categories, output_subplots, output_line_graphs):
    try:
        with open(f"{dataset}.pkl", "rb") as f:
            results = pickle.load(f)
        print(f"Successfully loaded {dataset}.pkl")  # Debug print
    except FileNotFoundError:
        print(f"File not found: {dataset}.pkl")
        return

    # Prepare data for subplots
    data = {erm: [] for erm in balance_erm}
    errors = {erm: [] for erm in balance_erm}
    data_no_llr = {erm: [] for erm in balance_erm}
    errors_no_llr = {erm: [] for erm in balance_erm}

    epoch_for_dataset = 20 if dataset == "celeba_resnet" else 100

    for erm in balance_retrain:
        erm_means = []
        erm_stds = []
        erm_means_no_llr = []
        erm_stds_no_llr = []
        for retrain in balance_retrain:
            wga_values = []
            wga_values_no_llr = []
            for s in seeds:
                try:
                    wga_values.append(results[s][50][erm]["llr"][retrain][epoch_for_dataset]["test_wga"])
                    wga_values_no_llr.append(results[s][50][erm]["erm"][epoch_for_dataset]["test_wga"])
                except KeyError:
                    wga_values.append(np.nan)
                    wga_values_no_llr.append(np.nan)
            wga_mean, wga_std = stats(wga_values)
            wga_mean_no_llr, wga_std_no_llr = stats(wga_values_no_llr)
            erm_means.append(wga_mean)
            erm_stds.append(wga_std)
            erm_means_no_llr.append(wga_mean_no_llr)
            erm_stds_no_llr.append(wga_std_no_llr)

        data[erm] = erm_means
        errors[erm] = erm_stds
        data_no_llr[erm] = erm_means_no_llr
        errors_no_llr[erm] = erm_stds_no_llr

    combined_data = {erm: data[erm] for erm in balance_erm}
    combined_errors = {erm: errors[erm] for erm in balance_erm}
    for erm in balance_erm:
        combined_data[f"{erm}_no_llr"] = data_no_llr[erm]
        combined_errors[f"{erm}_no_llr"] = errors_no_llr[erm]

    plot_subplots(combined_data, combined_errors, categories, "Test WGA", f"{dataset.replace('_', ' ').capitalize()} WGA", output_subplots, dataset)


    # Plot line graphs for Test WGA
  #  plot_line_graphs(dataset, results, , balance_erm, balance_retrain, epochs, f"Test WGA over Epochs - {dataset.capitalize()} Dataset", output_line_graphs)

# Process both datasets
process_dataset("celeba_resnet", ["none", "upsampling", "subsetting"], "celeba_Resnet WGA", "celeba_resnet50_llr_test_wga.png")
process_dataset("waterbirds_resnet", ["none", "upsampling", "subsetting"], "Waterbirds_Resnet WGA", "waterbirds_llr_test_wga.png")


#process_dataset("civilcomments_bert", ["none", "upsampling", "subsetting"], "civilcomments_llr_erm_subplots.png", "civilcomments_llr_test_wga.png")

