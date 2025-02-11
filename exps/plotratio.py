import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

plt.rcParams.update({"font.size": 16})

# Define configurations
seeds = [1,2,3]
group_ratios = [5, 10, 20, 50, 100]  # The group ratios for LLR
ratios = [0.05,  0.1, 0.2, 0.5, 1.0]  # Ratios for ERM to plot on x-axis
# Colors for the different ERM group ratios
colors = ["C0", "C1", "C2", "C3", "C4"]
labels = {ratio: f"ERM group ratio {ratio}" for ratio in group_ratios}
# Update labels to include keys for both group_ratios and ratios
labels.update({ratio: f"LLR ratio {ratio}" for ratio in ratios})

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

with open("civilcomments_bert_groupratio.pkl", "rb") as f:
    civilcomments_results = pickle.load(f)

with open("multinli_bert_groupratio.pkl", "rb") as f:
    multinli_results = pickle.load(f)



def plot_all_datasets_with_correlation(results_dict, balance_method, filename):
    """
    Combines plots for Waterbirds, CelebA, Civilcomments, and MultiNLI datasets in a single figure.
    Computes and displays correlation coefficients between ERM and LLR lines.
    
    Args:
        results_dict: A dictionary with dataset names as keys and results as values.
        balance_method: The balance method to use (e.g., "upsampling").
        filename: The filename for saving the output plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # 2 rows, 2 columns for 4 datasets
    dataset_names = list(results_dict.keys())
    axes = axs.flatten()

    for i, dataset_name in enumerate(dataset_names):
        results = results_dict[dataset_name]
        # Set specific epochs based on the dataset
        erm_epochs = 100 if dataset_name == "Waterbirds" else 20

        # Plotting function
        def plot_data(ax):
            # For ERM triangles: Fixed points for each ERM group ratio
            erm_values = np.zeros((len(ratios), len(seeds)))  # Rows for ERM ratios, columns for seeds
            for i, r in enumerate(ratios):
                for j, s in enumerate(seeds):
                    try:
                        erm_value = results[s]["base" if dataset_name in ["Civilcomments", "MultiNLI"] else 50]["upsampling" if dataset_name == "Waterbirds" else "subsetting"][r]["erm"].get(erm_epochs, {})
                        if isinstance(erm_value, dict):
                            erm_values[i, j] = erm_value.get("test_wga", np.nan)
                        elif isinstance(erm_value, (float, int)):
                            erm_values[i, j] = erm_value
                        else:
                            erm_values[i, j] = np.nan
                    except KeyError as e:
                        print(f"KeyError: {e}")
                        erm_values[i, j] = np.nan

            # Plot triangles for ERM
            erm_mean, erm_std = stats(erm_values)
            plot(ax, ratios, erm_mean, f"ERM - {balance_method}", color="black", marker="^", linestyle="None")
            err(ax, ratios, erm_mean, erm_std, color="black")

            # For LLR lines: Iterate over ERM group ratios and calculate correlations
            correlations = []
            for idx, erm_group_ratio in enumerate(ratios):
                llr_values = np.zeros((len(ratios), len(seeds)))  # Rows for LLR ratios, columns for seeds
                for i, r in enumerate(ratios):
                    for j, s in enumerate(seeds):
                        try:
                            llr_value = results[s]["base" if dataset_name in ["Civilcomments", "MultiNLI"] else 50]["upsampling" if dataset_name == "Waterbirds" else "subsetting"][erm_group_ratio]["llr"].get(balance_method, {}).get(r, {}).get(erm_epochs, {})
                            if isinstance(llr_value, dict):
                                llr_values[i, j] = llr_value.get("test_wga", np.nan)
                            elif isinstance(llr_value, (float, int)):
                                llr_values[i, j] = llr_value
                            else:
                                llr_values[i, j] = np.nan
                        except KeyError as e:
                            print(f"KeyError: {e}")
                            llr_values[i, j] = np.nan

                llr_mean, llr_std = stats(llr_values)
                plot(ax, ratios, llr_mean, f"{labels[erm_group_ratio]}", color=colors[idx], marker=None)
                err(ax, ratios, llr_mean, llr_std, color=colors[idx])

                # Calculate and store the Pearson correlation coefficient
                if not np.isnan(erm_mean).all() and not np.isnan(llr_mean).all():
                    corr, _ = pearsonr(erm_mean, llr_mean)
                    correlations.append((erm_group_ratio, corr))

            ax.set_xscale("log")
            ax.set_xticks(ratios)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_ylim(35, 90)
            ax.grid(True, alpha=0.5)

            ax.set_xlabel("ERM Group Ratio (Log Scale)")
            ax.set_ylabel("Test_WGA")
            ax.legend(loc="best")

            # Display correlation coefficients on the plot
            correlation_text = "\n".join([f"LLR ratio: {grp}, r={corr:.2f}" for grp, corr in correlations])
            ax.text(0.95, 0.05, correlation_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8))

        plot_data(axes[i])
        axes[i].set_title(f"{dataset_name} - Subsetting(ERM)_Upsampling(LLR)")

    plt.tight_layout()

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, f"{filename}.png"), bbox_inches='tight', dpi=600)
    plt.show()

# Dictionary containing results for all datasets
results_dict = {
    "Waterbirds": waterbirds_results,
    "CelebA": celeba_results,
    "Civilcomments": civilcomments_results,
    "MultiNLI": multinli_results,
}


# Call the function with correlation coefficients
plot_all_datasets_with_correlation(results_dict, "upsampling", "GroupRatio_with_Correlation")



