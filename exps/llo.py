import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

plt.rcParams.update({"font.size": 16})

# Define configurations
seeds = [1, 2, 3]
model_versions = [18, 34, 50, 101, 152]

# Define different balance methods for the datasets
balance_erms_waterbirds = ["none", "upsampling", "subsetting", "upweighting", "mixture2.0"]
balance_erms_other = ["none", "sampler", "subset", "loss", "mixture2.0"]

# Colors and labels for the balance_erm configurations
colors = {
    "none": "C0",
    "upsampling": "C2",
    "subsetting": "C1",
    "upweighting": "C4",
    "mixture2.0": "C3",
    "sampler": "C2",
    "subset": "C1",
    "loss": "C4",
}
labels = {
    "none": "No class-balancing",
    "upsampling": "Upsampling",
    "subsetting": "Subsetting",
    "upweighting": "Upweighting",
    "mixture2.0": "Mixture 2:1",
    "sampler": "Upsampling",
    "subset": "Subsetting",
    "loss": "Upweighting",
}

# Helper functions
def stats(x):
    mean = np.round(np.mean(x, 1) * 100, 1)
    std = np.round(np.std(x, 1) * 100, 1)
    return mean, std

def plot(ax, x, y, c, linestyle="solid", label=None, color=None, is_version_plot=False):
    # Interpolation setup: create finer grid of x-values for smoother line
    x_interp = np.linspace(min(x), max(x), 500)
    y_interp = np.interp(x_interp, x, y)  # Linear interpolation, or you can use a cubic interpolation like this:
    # y_interp = interp1d(x, y, kind='cubic')(x_interp)
    
    # Plot the interpolated line
    if is_version_plot:
        ax.plot(x_interp, y_interp, linestyle=linestyle, linewidth=3, label=labels[c], color=colors[c])
    else:
        label = label if label else labels[c]
        color = color if color else colors[c]
        ax.plot(x_interp, y_interp, linestyle=linestyle, linewidth=3, color=color, label=label)

def err(ax, x, y, err, c, color=None):
    color = color if color else colors[c]
    ax.fill_between(x, y - err, y + err, facecolor=color, alpha=0.2)

def save(fname, xlabel, ylabel=None, legend="upper center", xticks=None, grid_axis="both"):
    plt.grid(alpha=0.5, axis=grid_axis)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(loc=legend)

    fname = f"{fname}.png"
    fpath = osp.join("out", fname)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)
    plt.clf()

# Load results for Waterbirds dataset (standard)
with open("waterbirds_resnet.pkl", "rb") as f:
    waterbirds_results = pickle.load(f)

# Load results for Waterbirds (all scales)
with open("waterbirds_resnet_allscales.pkl", "rb") as f:
    waterbirds_allscales_results = pickle.load(f)

# Load results for CelebA dataset (all scales)
with open("celeba_resnet_allscales.pkl", "rb") as f:
    celeba_results = pickle.load(f)

# Define epochs
waterbirds_epochs = np.arange(10, 101, 10)  # 100 epochs for Waterbirds
celeba_epochs = np.arange(2, 21, 2)  # 20 epochs for CelebA

# Function to create plots
def generate_plots(waterbirds_results, waterbirds_allscales_results, celeba_results):
    # Create a 1-row, 3-column grid for the plots
    fig, axs = plt.subplots(1, 3, figsize=(30, 8))  # 3 plots side by side

    # Helper function to plot the data
    def plot_data(results, ax, balance_erms, title, epochs, use_allscales=False):
        model_indices = np.arange(len(model_versions))  # Equally spaced indices [0, 1, 2, 3, 4]

        for balance_erm in balance_erms:
            wga = np.zeros((len(model_versions), len(seeds)))
            for i, model_version in enumerate(model_versions):
                for j, s in enumerate(seeds):
                    try:
                        # For allscales files, use True key in the dictionary
                        if use_allscales:
                            value = results[s][model_version][True][balance_erm]["erm"][epochs[-1]]["train_aa"]
                        else:
                            value = results[s][model_version][balance_erm]["erm"][epochs[-1]]["train_aa"]
                        wga[i, j] = value
                    except KeyError:
                        print(f"Missing data for seed {s}, model version {model_version}, balance {balance_erm}")
                        continue
                    except TypeError:
                        print(f"Missing data for seed {s}, model version {model_version}, balance {balance_erm}")
                        continue
            wga_mean, wga_std = stats(wga)
            plot(ax, model_indices, wga_mean, balance_erm, is_version_plot=True)
            err(ax, model_indices, wga_mean, wga_std, balance_erm)

        # Set the equally spaced ticks and label them with the model versions
        ax.set_xticks(model_indices)
        ax.set_xticklabels(model_versions)
        ax.grid(True)

    # Plot 1: Waterbirds (standard)
    plot_data(waterbirds_results, axs[0], balance_erms_waterbirds, "Waterbirds", waterbirds_epochs)

    # Plot 2: Waterbirds (all scales) with black dashed line at x=18
    plot_data(waterbirds_allscales_results, axs[1], balance_erms_other, "Waterbirds (All Scales)", waterbirds_epochs, use_allscales=True)

    # Add vertical black dashed line at x=18 in the 2nd plot
    axs[1].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Interpolation Threshold')

    # Plot 3: CelebA with 20 epochs and different balance methods
    plot_data(celeba_results, axs[2], balance_erms_other, "CelebA", celeba_epochs, use_allscales=True)

    # Adjust layout and save the plots
    plt.tight_layout()

    # Get handles and labels from one of the axes
    handles, labels = axs[1].get_legend_handles_labels()

    # Add one legend bar for the entire figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(handles))

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, "Waterbirds_CelebA_Comparison_AA.png"), bbox_inches='tight', dpi=600)
    plt.show()

# Generate the plots
generate_plots(waterbirds_results, waterbirds_allscales_results, celeba_results)
