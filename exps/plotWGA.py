import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 14})

# Configuration
seeds = [1, 2, 3]
balance_retrain = ["none", "subsetting", "upsampling"]
datasets = {
    "civilcomments": "civilcomments_bert_layerwise.pkl",
    "celeba": "celeba_convnextv2_layerwise.pkl",
    "waterbirds" : "waterbirds_convnextv2_layerwise.pkl"
}
colors = {
    "none": "C0",     
    "subsetting": "C1",  # Orange (subsetting)
    "upsampling": "C2",   # Green (upsampling)
    "erm": "C3"  # Red for ERM
}
labels = {
    "none": "None",
    "subsetting": "Subsetting",
    "upsampling": "Upsampling",
    "erm": "ERM"
}

# Key mapping for different directories
retrain_key_mapping = {
    "/home/xzhang941/revisiting-finetuning": {
        "none": "none",
        "subsetting": "subset",
        "upsampling": "sampler"
    },
    "/home/xzhang941/understanding-llr": {
        "none": "none",
        "subsetting": "subsetting",
        "upsampling": "upsampling"
    }
}

def stats(x):
    mean = np.nanmean(x)
    std = np.nanstd(x)
    return mean, std

def load_results(filepath):
    try:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        print(f"Successfully loaded {filepath}")  # Debug print
        return results
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def plot_layerwise_bars(data, errors, categories, ylabel, title, output_file, ylim):
    width = 0.2  # Width of the bars
    offset = 0.25 # Offset for center bars

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(title)

    for i, retrain in enumerate(categories):
        ax = axes[i]
        layerwise_means = data[retrain]['layerwise']
        layerwise_stds = errors[retrain]['layerwise']
        llr_means = data[retrain]['llr']
        llr_stds = errors[retrain]['llr']
        erm_means = data[retrain]['erm']
        erm_stds = errors[retrain]['erm']
        
        x = np.arange(3)  # Three bars per subplot (Layerwise, LLR, and ERM)
        
        ax.bar(-offset, layerwise_means, width, yerr=layerwise_stds, color=colors[retrain], capsize=5, label=f"{labels[retrain]} - Representation")
        ax.bar(0, llr_means, width, yerr=llr_stds, color='red', capsize=5, label=f"{labels[retrain]} - Last Layer")
        ax.bar(offset, erm_means, width, yerr=erm_stds, color='grey', capsize=5, label=f"{labels[retrain]} - ERM")

        ax.set_title(f"Class Balance: {labels[retrain]}")
        ax.set_xticks([-offset, 0, offset])
        ax.set_xticklabels(['Representation', 'Last-Layer', 'ERM'])
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.05, 0.05))  # Adjusted y-axis ticks based on the limit
        ax.set_ylim(ylim)  # Set y-axis range according to the input limits
        ax.grid(alpha=0.5, axis='y')

        if i == 0:
            ax.set_ylabel(ylabel)  # Add y-axis label to the first subplot
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-axis numbers are shown for all subplots

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

def calculate_statistics(values):
    mean_val, std_val = stats(values)
    return mean_val, std_val

def process_values(results, erm_results, llr_results, retrain_key_erm, retrain_key_llr, dataset, seeds):
    wga_layerwise_values, wga_llr_values, wga_erm_values = [], [], []
    aa_layerwise_values, aa_llr_values, aa_erm_values = [], [], []

    # Set the test epoch based on the dataset
    test_epoch = 100 if dataset == "waterbirds" else 20

    for s in seeds:
        try:
            # WGA Values
            wga_layerwise_values.append(llr_results[s]["base"][retrain_key_llr]["erm"][test_epoch]["test_wga"])
            if not(s == 3 and retrain_key_llr == "subsetting" and dataset == "waterbirds"): # Hack because that value is missing?
                wga_llr_values.append(llr_results[s]["base"][retrain_key_llr]["llr"]["none"][test_epoch]["test_wga"])
            else:
                wga_llr_values.append(np.nan)
            wga_erm_values.append(erm_results[s]["base"][retrain_key_erm]["erm"][test_epoch]["test_wga"])
            
            # AA Values
            aa_layerwise_values.append(llr_results[s]["base"][retrain_key_llr]["erm"][test_epoch]["test_aa"])
            if not (s == 3 and retrain_key_llr == "subsetting" and dataset == "waterbirds"): # Same hack as above
                aa_llr_values.append(llr_results[s]["base"][retrain_key_llr]["llr"]["none"][test_epoch]["test_aa"])
            else:
                aa_llr_values.append(np.nan)
            aa_erm_values.append(erm_results[s]["base"][retrain_key_erm]["erm"][test_epoch]["test_aa"])

        except KeyError as e:
            print(f"KeyError for {dataset} - {retrain_key_erm} with seed {s}: {e}")
            # print(llr_results[s]["base"][retrain_key_llr]["erm"][test_epoch])
            # print(llr_results[s]["base"][retrain_key_llr]["llr"]["none"][test_epoch])
            # print(erm_results[s]["base"][True if dataset == "civilcomments" else 'imagenet1k'][retrain_key_erm]["erm"]["none"][test_epoch])
            wga_layerwise_values.append(np.nan)
            wga_llr_values.append(np.nan)
            wga_erm_values.append(np.nan)
            aa_layerwise_values.append(np.nan)
            aa_llr_values.append(np.nan)
            aa_erm_values.append(np.nan)

    return wga_layerwise_values, wga_llr_values, wga_erm_values, aa_layerwise_values, aa_llr_values, aa_erm_values


def process_layerwise_models(dataset, erm_filepath, llr_filepath, categories):
    # Initialize data and error storage
    data_wga = {retrain: {key: [] for key in ['layerwise', 'llr', 'erm']} for retrain in balance_retrain}
    errors_wga = {retrain: {key: [] for key in ['layerwise', 'llr', 'erm']} for retrain in balance_retrain}
    data_aa = {retrain: {key: [] for key in ['layerwise', 'llr', 'erm']} for retrain in balance_retrain}
    errors_aa = {retrain: {key: [] for key in ['layerwise', 'llr', 'erm']} for retrain in balance_retrain}

    filepath = datasets[dataset]

    # Load results for the dataset
    results = load_results(filepath)
    if results is None:
        return

    # Load ERM and LLR results
    erm_results = load_results(erm_filepath)
    llr_results = load_results(llr_filepath)
    if erm_results is None or llr_results is None:
        return

    # Prepare data for processing
    for retrain in balance_retrain:
        retrain_key_erm = retrain_key_mapping[osp.dirname(erm_filepath)][retrain]
        retrain_key_llr = retrain_key_mapping[osp.dirname(llr_filepath)][retrain]

        # Process values for WGA and AA
        wga_layerwise_values, wga_llr_values, wga_erm_values, aa_layerwise_values, aa_llr_values, aa_erm_values = process_values(
            results, erm_results, llr_results, retrain_key_erm, retrain_key_llr, dataset, seeds)

        # Calculate WGA statistics
        layerwise_mean_wga, layerwise_std_wga = calculate_statistics(wga_layerwise_values)
        llr_mean_wga, llr_std_wga = calculate_statistics(wga_llr_values)
        erm_mean_wga, erm_std_wga = calculate_statistics(wga_erm_values)

        # Calculate AA statistics
        layerwise_mean_aa, layerwise_std_aa = calculate_statistics(aa_layerwise_values)
        llr_mean_aa, llr_std_aa = calculate_statistics(aa_llr_values)
        erm_mean_aa, erm_std_aa = calculate_statistics(aa_erm_values)

        # Store WGA data
        data_wga[retrain]['layerwise'] = layerwise_mean_wga
        errors_wga[retrain]['layerwise'] = layerwise_std_wga
        data_wga[retrain]['llr'] = llr_mean_wga
        errors_wga[retrain]['llr'] = llr_std_wga
        data_wga[retrain]['erm'] = erm_mean_wga
        errors_wga[retrain]['erm'] = erm_std_wga

        # Store AA data
        data_aa[retrain]['layerwise'] = layerwise_mean_aa
        errors_aa[retrain]['layerwise'] = layerwise_std_aa
        data_aa[retrain]['llr'] = llr_mean_aa
        errors_aa[retrain]['llr'] = llr_std_aa
        data_aa[retrain]['erm'] = erm_mean_aa
        errors_aa[retrain]['erm'] = erm_std_aa


    print(f"{dataset} WGA Data: {data_wga}, WGA Errors: {errors_wga}")  # Debug print
    print(f"{dataset} AA Data: {data_aa}, AA Errors: {errors_aa}")  # Debug print

    # Plot WGA and AA layerwise bars
    plot_layerwise_bars(data_wga, errors_wga, categories, "Test WGA", f"Layerwise Convnext Comparison - {dataset.capitalize()} (WGA)", f"{dataset}_layerwise_convnext_comparison_wga.png", ylim = (0.3, 0.85))
    plot_layerwise_bars(data_aa, errors_aa, categories, "Test AA", f"Layerwise Convnext Comparison - {dataset.capitalize()} (AA)", f"{dataset}_layerwise_convnext_comparison_aa.png", ylim = (0.5,1))

# Process each dataset separately
#process_layerwise_models("celeba", "/home/xzhang941/revisiting-finetuning/celeba.pkl", "/home/xzhang941/understanding-llr/celeba_convnextv2_layerwise.pkl", ["none", "subsetting", "upsampling"])
#process_layerwise_models("civilcomments", "/home/xzhang941/revisiting-finetuning/civilcomments.pkl", "/home/xzhang941/understanding-llr/civilcomments_bert_layerwise.pkl", ["none", "subsetting", "upsampling"])
process_layerwise_models("waterbirds", "/home/xzhang941/understanding-llr/waterbirds_convnextv2_layerwise.pkl", "/home/xzhang941/understanding-llr/waterbirds_convnextv2_layerwise.pkl", ["none", "subsetting", "upsampling"])


