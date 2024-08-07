import os
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 16})

# Configuration
seeds = [1, 2, 3]
balance_retrain = ["none", "subsetting", "upsampling"]
datasets = {
    "civilcomments": "civilcomments_bert_layerwise.pkl",
    "celeba": "celeba_convnextv2_layerwise.pkl"
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

def plot_layerwise_bars(data, errors, categories, ylabel, title, output_file):
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

        ax.set_title(f"Retrain: {labels[retrain]}")
        ax.set_xticks([-offset, 0, offset])
        ax.set_xticklabels(['Layerwise', 'LLR', 'ERM'])
        ax.set_yticks(np.arange(0.3, 0.9, 0.05))  # Adjusted y-axis ticks for better granularity
        ax.set_ylim(0.3, 0.8)  # Adjusted y-axis range to ensure visibility of bars
        ax.grid(alpha=0.5, axis='y')

        if i == 0:
            ax.set_ylabel(ylabel)  # Add y-axis label to the first subplot
        ax.yaxis.set_tick_params(labelleft=True)  # Ensure y-axis numbers are shown for all subplots

        # Add x-axis label
      #  ax.set_xlabel('Model Type')

    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(osp.join(output_dir, output_file), bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()

def load_results(filepath):
    try:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        print(f"Successfully loaded {filepath}")  # Debug print
        return results
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def process_layerwise_models(dataset, erm_filepath, llr_filepath, categories, output_file):
    # Initialize data and error storage
    data = {retrain: {'layerwise': [], 'llr': [], 'erm': []} for retrain in balance_retrain}
    errors = {retrain: {'layerwise': [], 'llr': [], 'erm': []} for retrain in balance_retrain}

    filepath = datasets[dataset]

    # Load results for the dataset
    results = load_results(filepath)
    if results is None:
        return

    # Load ERM results
    erm_results = load_results(erm_filepath)
    if erm_results is None:
        return

    # Load LLR and layerwise results
    llr_results = load_results(llr_filepath)
    if llr_results is None:
        return

    # Verify data structure
    print(f"Data structure for {dataset}:")
    for key in results.keys():
        print(f"  {key}: {type(results[key])}")

    # Prepare data for bar plot
    for retrain in balance_retrain:
        retrain_key_erm = retrain_key_mapping[osp.dirname(erm_filepath)][retrain]
        retrain_key_llr = retrain_key_mapping[osp.dirname(llr_filepath)][retrain]
        
        wga_layerwise_values = []
        wga_llr_values = []
        wga_erm_values = []
        for s in seeds:
            try:
                wga_layerwise_values.append(llr_results[s]["base"][retrain_key_llr]["erm"][20]["test_wga"])
                wga_llr_values.append(llr_results[s]["base"][retrain_key_llr]["llr"]["none"][20]["test_wga"])
                if dataset == "celeba":
                    wga_erm_values.append(erm_results[s]["base"]['imagenet1k'][retrain_key_erm]["erm"]["none"][20]["test_wga"])
                else:
                    wga_erm_values.append(erm_results[s]["base"][True][retrain_key_erm]["erm"]["none"][20]["test_wga"])
            
            except KeyError as e:
                print(f"KeyError for {dataset} - {retrain} with seed {s}: {e}")
                wga_layerwise_values.append(np.nan)
                wga_llr_values.append(np.nan)
                wga_erm_values.append(np.nan)
        
        layerwise_mean, layerwise_std = stats(wga_layerwise_values)
        llr_mean, llr_std = stats(wga_llr_values)
        erm_mean, erm_std = stats(wga_erm_values)

        # Detailed debugging to understand the structure of loaded files
        print(f"{dataset} - {retrain} Layerwise WGA Values: {wga_layerwise_values}")
        print(f"{dataset} - {retrain} LLR WGA Values: {wga_llr_values}")
        print(f"{dataset} - {retrain} ERM WGA Values: {wga_erm_values}")
        print(f"{dataset} - {retrain} Layerwise Mean: {layerwise_mean}, Std: {layerwise_std}")  # Debug print
        print(f"{dataset} - {retrain} LLR Mean: {llr_mean}, Std: {llr_std}")  # Debug print
        print(f"{dataset} - {retrain} ERM Mean: {erm_mean}, Std: {erm_std}")  # Debug print
        
        data[retrain]['layerwise'] = layerwise_mean
        errors[retrain]['layerwise'] = layerwise_std
        data[retrain]['llr'] = llr_mean
        errors[retrain]['llr'] = llr_std
        data[retrain]['erm'] = erm_mean
        errors[retrain]['erm'] = erm_std
    print(f"{dataset} Data: {data}, Errors: {errors}")  # Debug print

    # Plot layerwise bars
    plot_layerwise_bars(data, errors, categories, "Test WGA", f"Layerwise Models Comparison - {dataset.capitalize()}", output_file)

# Process each dataset separately
process_layerwise_models("celeba", "/home/xzhang941/revisiting-finetuning/celeba.pkl", "/home/xzhang941/understanding-llr/celeba_convnextv2_layerwise.pkl", ["none", "subsetting", "upsampling"], "celeba_layerwise_models_comparison.png")
process_layerwise_models("civilcomments", "/home/xzhang941/revisiting-finetuning/civilcomments.pkl", "/home/xzhang941/understanding-llr/civilcomments_bert_layerwise.pkl", ["none", "subsetting", "upsampling"], "civilcomments_layerwise_models_comparison.png")
