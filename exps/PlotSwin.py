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
seeds = [1, 2, 3]
model_versions = [18,34,50,101,152]
balance_erms = ["none", "sampler", "subset", "loss", "mixture2.0"]

# Colors and labels for the balance_erm configurations
colors = {
    "none": "C0",
    "subset": "C1",
    "sampler": "C2",
    "loss": "C4",
    "mixture2.0": "C3",
}
labels = {
    "none": "No class-balancing",
    "subset": "Subsetting",
    "sampler": "Upsampling",
    "loss": "Upweighting",
    "mixture2.0": "Mixture 2:1",
}

# Helper functions
def stats(x):
    mean = np.round(np.mean(x, 1) * 100, 1)
    std = np.round(np.std(x, 1) * 100, 1)
    return mean, std

def plot(ax, x, y, c, linestyle="solid", label=None, color=None, is_version_plot=False):
    if is_version_plot:
        ax.plot(x, y, linestyle=linestyle, linewidth=3, label=labels[c], color=colors[c])
    else:
        label = label if label else labels[c]
        color = color if color else colors[c]
        ax.plot(x, y, linestyle=linestyle, linewidth=3, color=color, label=label)

def err(ax, x, y, err, c, color=None):
    color = color if color else colors[c]
    ax.fill_between(x, y - err, y + err, facecolor=color, alpha=0.2)

def save(fname, xlabel, ylabel=None, legend="upper center", xticks=None, grid_axis="both"):
    plt.grid(alpha=0.5, axis=grid_axis)
    if ylabel:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(loc=legend, bbox_to_anchor=(0.5, 1.2), ncol=len(balance_erms))

    fname = f"{fname}.png"
    fpath = osp.join("out", fname)
    plt.savefig(fpath, bbox_inches="tight", dpi=600)
    plt.clf()
    


# Load results for Waterbirds dataset
with open("waterbirds_resnet_allscales.pkl", "rb") as f:
    waterbirds_results = pickle.load(f)

# Load results for CelebA dataset
with open("celeba_resnet_allscales.pkl", "rb") as f:
    celeba_swin_results = pickle.load(f)

# Debug: Check if data is loaded correctly
print("Waterbirds Results Keys:", list(waterbirds_results.keys()))
print("CelebA Results Keys:", list(celeba_swin_results.keys()))

# Define epochs
waterbirds_epochs = np.arange(10, 101, 10)
celeba_epochs = np.arange(2, 21, 2)



# Function to create plots
def generate_plots(waterbirds_results, celeba_swin_results, waterbirds_epochs, celeba_epochs):
    # Plot 1: Test WGA over Epochs
    fig, axes = plt.subplots(1, 2, figsize=(20,7))
    
    # Waterbirds plot
    for balance_erm in balance_erms:
        wga = np.zeros((len(waterbirds_epochs), len(seeds)))
        for i, e in enumerate(waterbirds_epochs):
            for j, s in enumerate(seeds):
                try:
                    value = waterbirds_results[s][50][True][balance_erm]["erm"][e]["test_wga"]
                    wga[i, j] = value
                except KeyError:
                    continue
                except TypeError:
                    continue
        wga_mean, wga_std = stats(wga)
        plot(axes[0], waterbirds_epochs, wga_mean, balance_erm)
        err(axes[0], waterbirds_epochs, wga_mean, wga_std, balance_erm)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test WGA")
    axes[0].grid(True)

    # CelebA plot
    for balance_erm in balance_erms:
        wga = np.zeros((len(celeba_epochs), len(seeds)))
        for i, e in enumerate(celeba_epochs):
            for j, s in enumerate(seeds):
                try:
                    value = celeba_swin_results[s][50][True][balance_erm]["erm"][e]["test_wga"]
                    wga[i, j] = value
                except KeyError:
                    continue
                except TypeError:
                    continue
        wga_mean, wga_std = stats(wga)
        plot(axes[1], celeba_epochs, wga_mean, balance_erm)
        err(axes[1], celeba_epochs, wga_mean, wga_std, balance_erm)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test WGA")
    axes[1].grid(True)

    # Set the legend above the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(balance_erms))

    # Save the first plot
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, "resnet_allscales_over_epochs.png"), bbox_inches='tight', dpi=600)
    plt.show()

    # Plot 2: Test WGA over Model Versions with interpolation threshold
    fig, axes = plt.subplots(1, 2, figsize=(20,7))

    # Initialize waterbirds_interpolation_threshold
    waterbirds_interpolation_threshold = None

    # Find interpolation threshold for Waterbirds
    for i, model_version in enumerate(model_versions):
        for s in seeds:
            try:
                train_acc = waterbirds_results[s][model_version][True]["none"]["erm"][100]["train_aa"]
                if train_acc == 1.0:
                    waterbirds_interpolation_threshold = i
                    break
            except KeyError:
                continue
        if waterbirds_interpolation_threshold is not None:
            break

    # Plot for Waterbirds
    x_axis_values = ["18", "34", "50", "101", "152"]
    for balance_erm in balance_erms:
        wga = np.zeros((len(model_versions), len(seeds)))
        for i, model_version in enumerate(model_versions):
            for j, s in enumerate(seeds):
                try:
                    value = waterbirds_results[s][model_version][True][balance_erm]["erm"][100]["test_wga"]
                    wga[i, j] = value
                except KeyError:
                    continue
                except TypeError:
                    continue
        wga_mean, wga_std = stats(wga)
        plot(axes[0], x_axis_values, wga_mean, balance_erm, is_version_plot=True)
        err(axes[0], x_axis_values, wga_mean, wga_std, balance_erm)

    # Add interpolation threshold line if found
    if waterbirds_interpolation_threshold is not None:
        axes[0].axvline(x=x_axis_values[waterbirds_interpolation_threshold], color='black', linestyle='dashed', linewidth=2, label="Interpolation Threshold")

    # Set x-ticks to the correct model versions
    axes[0].set_xticks(x_axis_values)
    axes[0].set_ylabel("Test WGA")
    axes[0].grid(True)

    # Initialize celeba_interpolation_threshold
    celeba_interpolation_threshold = None

    # Find interpolation threshold for CelebA
    for i, model_version in enumerate(model_versions):
        for s in seeds:
            try:
                train_acc = celeba_swin_results[s][model_version][True][balance_erm]["erm"][20]["train_aa"]
                if train_acc == 1.0:
                    celeba_interpolation_threshold = i
                    break
            except KeyError:
                continue
        if celeba_interpolation_threshold is not None:
            break

    # Plot for CelebA
    for balance_erm in balance_erms:
        wga = np.zeros((len(model_versions), len(seeds)))
        for i, model_version in enumerate(model_versions):
            for j, s in enumerate(seeds):
                try:
                    value = celeba_swin_results[s][model_version][True][balance_erm]["erm"][20]["test_wga"]
                    wga[i, j] = value
                except KeyError:
                    continue
                except TypeError:
                    continue
        wga_mean, wga_std = stats(wga)
        plot(axes[1], x_axis_values, wga_mean, balance_erm, is_version_plot=True)
        err(axes[1], x_axis_values, wga_mean, wga_std, balance_erm)

    if celeba_interpolation_threshold is not None:
        axes[1].axvline(x=x_axis_values[celeba_interpolation_threshold], color='black', linestyle='dashed', linewidth=2, label="Interpolation Threshold")

    # Set x-ticks to the correct model versions
    axes[1].set_xticks(x_axis_values)
    axes[1].set_ylabel("Test WGA")
    axes[1].grid(True)

    # Set the legend above the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(balance_erms)+1)

    # Save the second plot
    plt.savefig(osp.join(output_dir, "resnet_allscales_model_versions.png"), bbox_inches='tight', dpi=600)
    plt.show()




# Generate the plots
generate_plots(waterbirds_results, celeba_swin_results, waterbirds_epochs, celeba_epochs)

# Print mean Test WGA for CelebA model versions at Epoch 20
celeba_epoch_20_wga = {}
x_axis_values = ["18", "34", "50", "101", "152"]
 
for balance_erm in balance_erms:
    wga = np.zeros(len(seeds))
    for j, s in enumerate(seeds):
        try:
            value = celeba_swin_results[s][50][True][balance_erm]["erm"][20]["test_wga"]
            wga[j] = value
        except KeyError as ke:
            print(f"Missing data for seed {s}, balance {balance_erm}, epoch 20")
            continue
        except TypeError as te:
            print(f"Type error for seed {s}, balance {balance_erm}, epoch 20: {te}")
            print(f"Actual value: {celeba_swin_results[s][50][True][balance_erm]['erm'][20]}")
            continue

    print(f"Accumulated WGA values for {balance_erm}: {wga}")  # Debug print
    wga_mean = np.round(np.mean(wga) * 100, 1)
    celeba_epoch_20_wga[balance_erm] = wga_mean

print("\nMean Test WGA for CelebA model versions at Epoch 20:")
for balance_erm, mean_value in celeba_epoch_20_wga.items():
    print(f"{labels[balance_erm]}: {mean_value}%")

# Print mean Test WGA for Waterbirds base model version for each balancing method at Epoch 100
base_model_means_epoch_100 = {}

for balance_erm in balance_erms:
    wga = np.zeros(len(seeds))
    for j, s in enumerate(seeds):
        try:
            value = waterbirds_results[s][50][True][balance_erm]["erm"][100]["test_wga"]
            wga[j] = value
        except KeyError as ke:
            print(f"Missing data for seed {s}, balance {balance_erm}, epoch 100")
            continue
        except TypeError as te:
            print(f"Type error for seed {s}, balance {balance_erm}, epoch 100: {te}")
            print(f"Actual value: {waterbirds_results[s]['base']['imagenet1k'][balance_erm]['erm'][100]}")
            continue

    print(f"Accumulated WGA values for {balance_erm}: {wga}")  # Debug print
    wga_mean = np.round(np.mean(wga) * 100, 1)
    base_model_means_epoch_100[balance_erm] = wga_mean

print("\nMean Test WGA for Waterbirds base model version for each balancing method at Epoch 100:")
for balance_erm, mean_value in base_model_means_epoch_100.items():
    print(f"{labels[balance_erm]}: {mean_value}%")
