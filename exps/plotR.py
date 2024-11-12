import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

plt.rcParams.update({"font.size": 16})

# Define configurations
seeds = [1]
group_ratios = [5, 10, 20, 50, 100]  # The group ratios for both ERM and LLR

# Colors for the different ERM group ratios
colors = ["C0", "C1", "C2", "C3", "C4"]
labels = {ratio: f"ERM group ratio {ratio}" for ratio in group_ratios}

# Helper functions
def stats(x):
    mean = np.round(np.mean(x, axis=1) * 100, 1)
    std = np.round(np.std(x, axis=1) * 100, 1)
    return mean, std

def plot(ax, x, y, label, color, linestyle="solid", marker=None):
    # Plot the line
    ax.plot(x, y, linestyle=linestyle, linewidth=3, color=color, label=label, marker=marker)

def err(ax, x, y, err, color):
    ax.fill_between(x, y - err, y + err, facecolor=color, alpha=0.2)

# Load results for Waterbirds and CelebA
with open("waterbirds_resnet_groupratio.pkl", "rb") as f:
    waterbirds_results = pickle.load(f)

with open("celeba_resnet_groupratio.pkl", "rb") as f:
    celeba_results = pickle.load(f)

# Function to create plots for ERM and LLR across different ERM group ratios
def plot_erm_llr(dataset_name, results, balance_methods, erm_epochs, filename):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))  # 1 row, 2 plots side by side

    def plot_data(ax, balance_method):
        # For LLR lines: Iterate over ERM group ratios
        for idx, erm_group_ratio in enumerate(group_ratios):
            llr_values = np.zeros((len(group_ratios), len(seeds)))  # Rows for LLR ratios, columns for seeds

            # For each LLR ratio, calculate the corresponding LLR values
            for i, llr_group_ratio in enumerate(group_ratios):
                for j, s in enumerate(seeds):
                    try:
                        # Access the llr test_wga for the given ERM and LLR group ratio
                        llr_value = results[s][18][balance_method][erm_group_ratio]["llr"].get(balance_method, {}).get(llr_group_ratio, {}).get(erm_epochs, {})
                        print(f"LLR value for seed {s}, ERM ratio {erm_group_ratio}, LLR ratio {llr_group_ratio}: {llr_value}")  # Debug print
                        
                        # Check if llr_value is a dictionary or a float
                        if isinstance(llr_value, dict):
                            llr_values[i, j] = llr_value.get("test_wga", np.nan)  # Extract 'test_wga' if it's a dictionary
                        elif isinstance(llr_value, (float, int)):  # In case it's directly a number
                            llr_values[i, j] = llr_value
                        else:
                            llr_values[i, j] = np.nan
                    except KeyError as e:
                        print(f"KeyError: {e}")
                        llr_values[i, j] = np.nan

            # Plot each line for ERM group ratio changing over LLR group ratios
            llr_mean, llr_std = stats(llr_values)
            plot(ax, group_ratios, llr_mean, f"LLR - {labels[erm_group_ratio]}", color=colors[idx], marker=None)
            err(ax, group_ratios, llr_mean, llr_std, color=colors[idx])

        # For ERM triangles: Fixed points for each ERM group ratio
        erm_values = np.zeros((len(group_ratios), len(seeds)))  # Rows for ERM group ratios, columns for seeds
        for idx, erm_group_ratio in enumerate(group_ratios):
            for j, s in enumerate(seeds):
                try:
                    # Access the erm test_wga for the given ERM group ratio
                    erm_value = results[s][18][balance_method][erm_group_ratio]["erm"].get(erm_epochs, {})
                    if isinstance(erm_value, dict):
                        erm_values[idx, j] = erm_value.get("test_wga", np.nan)
                    elif isinstance(erm_value, (float, int)):
                        erm_values[idx, j] = erm_value
                    else:
                        erm_values[idx, j] = np.nan
                except KeyError as e:
                    print(f"KeyError: {e}")
                    erm_values[idx, j] = np.nan

        # Plot triangles for ERM
        erm_mean, erm_std = stats(erm_values)
        plot(ax, group_ratios, erm_mean, f"ERM - {balance_method}", color="black", marker="^", linestyle="None")
        err(ax, group_ratios, erm_mean, erm_std, color="black")

        # Set ticks and grid
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_yticks(np.arange(0, 110, 10))
        ax.grid(True, alpha=0.5)

        ax.set_xlabel("LLR Group Ratio")
        ax.set_ylabel("Test_WGA")
        ax.legend(loc="best")

    # Plot for each balance method side by side
    plot_data(axs[0], balance_methods[0])  # Plot for `none`
    axs[0].set_title(f"{dataset_name} - None")

    plot_data(axs[1], balance_methods[1])  # Plot for `subsetting`
    axs[1].set_title(f"{dataset_name} - Subsetting")

    # Adjust layout and save the plot
    plt.tight_layout()

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=600)

    plt.show()

# Plot for Waterbirds with None and Subsetting side by side
# Using ERM epoch 100 for Waterbirds
plot_erm_llr("Waterbirds", waterbirds_results, ["none", "subsetting"], 100, "Waterbirds_None_Subsetting")

# Plot for CelebA with None and Subsetting side by side
# Using ERM epoch 20 for CelebA
plot_erm_llr("CelebA", celeba_results, ["none", "subsetting"], 20, "CelebA_None_Subsetting")
