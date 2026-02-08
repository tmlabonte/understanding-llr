import os
import os.path as osp
import pickle
import numpy as np

# Configuration
seeds = [1, 2, 3]
balance_retrain = ["upsampling"]

datasets = {
    "celeba": "celeba_resnet.pkl",
    "waterbirds": "waterbirds_resnet.pkl",
    "civilcomments": "civilcomments_bert.pkl"
}
retrain_key_mapping = {
    "/home/xzhang941/understanding-llr": {
        "none": "none",
        "subsetting": "subsetting",
        "upsampling": "upsampling",
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
        return results
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def process_values(results, erm_results, llr_results, dfr_results, retrain_key_erm, retrain_key_llr, retrain_key_dfr, dataset, seeds):
    wga_llr_values, wga_erm_values, wga_dfr_values = [], [], []
    aa_llr_values, aa_erm_values, aa_dfr_values = [], [], []

    test_epoch = 100 if dataset == "waterbirds" else 20
    version = "base" if dataset =='civilcomments' else 50
    for s in seeds:
        try:
            # WGA Values
            wga_llr_values.append(llr_results[s][version]["train"]["subsetting"]["llr"][False]["upsampling"][test_epoch]["test_wga"])
            wga_erm_values.append(erm_results[s][version]["train"]["subsetting"]["erm"][test_epoch]["test_wga"])
            wga_dfr_values.append(dfr_results[s][version]["train"]["subsetting"]["dfr"][False]["upsampling"][test_epoch]["test_wga"])

            # AA Values
            aa_llr_values.append(llr_results[s][version]["train"]["subsetting"]["llr"][False]["upsampling"][test_epoch]["test_aa"])
            aa_erm_values.append(erm_results[s][version]["train"]["subsetting"]["erm"][test_epoch]["test_aa"])
            aa_dfr_values.append(dfr_results[s][version]["train"]["subsetting"]["dfr"][False]["upsampling"][test_epoch]["test_aa"])

        except KeyError as e:
            print(f"KeyError for {dataset} - {retrain_key_erm} with seed {s}: {e}")
            wga_llr_values.append(np.nan)
            wga_erm_values.append(np.nan)
            wga_dfr_values.append(np.nan)
            aa_llr_values.append(np.nan)
            aa_erm_values.append(np.nan)
            aa_dfr_values.append(np.nan)

    return wga_llr_values, wga_erm_values, wga_dfr_values, aa_llr_values, aa_erm_values, aa_dfr_values

def calculate_statistics(values):
    mean_val, std_val = stats(values)
    return mean_val, std_val

def process_layerwise_models(dataset, erm_filepath, llr_filepath, dfr_filepath):
    filepath = datasets[dataset]

    results = load_results(filepath)
    if results is None:
        return

    erm_results = load_results(erm_filepath)
    llr_results = load_results(llr_filepath)
    dfr_results = load_results(dfr_filepath)
    if erm_results is None or llr_results is None or dfr_results is None:
        return

    for retrain in balance_retrain:
        retrain_key_erm = retrain_key_mapping[osp.dirname(erm_filepath)][retrain]
        retrain_key_llr = retrain_key_mapping[osp.dirname(llr_filepath)][retrain]
        retrain_key_dfr = retrain_key_mapping[osp.dirname(dfr_filepath)][retrain]

        wga_llr_values, wga_erm_values, wga_dfr_values, aa_llr_values, aa_erm_values, aa_dfr_values = process_values(
            results, erm_results, llr_results, dfr_results, retrain_key_erm, retrain_key_llr, retrain_key_dfr, dataset, seeds)

        # Calculate and print statistics
        llr_mean_wga, llr_std_wga = calculate_statistics(wga_llr_values)
        erm_mean_wga, erm_std_wga = calculate_statistics(wga_erm_values)
        dfr_mean_wga, dfr_std_wga = calculate_statistics(wga_dfr_values)

        llr_mean_aa, llr_std_aa = calculate_statistics(aa_llr_values)
        erm_mean_aa, erm_std_aa = calculate_statistics(aa_erm_values)
        dfr_mean_aa, dfr_std_aa = calculate_statistics(aa_dfr_values)

        print(f"{dataset} - {retrain}")
        print(f"WGA: LLR {llr_mean_wga * 100:.2f}% +/- {llr_std_wga * 100:.2f}%, ERM {erm_mean_wga * 100:.2f}% +/- {erm_std_wga * 100:.2f}%, DFR {dfr_mean_wga * 100:.2f}% +/- {dfr_std_wga * 100:.2f}%")
        print(f"AA: LLR {llr_mean_aa * 100:.2f}% +/- {llr_std_aa * 100:.2f}%, ERM {erm_mean_aa * 100:.2f}% +/- {erm_std_aa * 100:.2f}%, DFR {dfr_mean_aa * 100:.2f}% +/- {dfr_std_aa * 100:.2f}%")


# Process each dataset
#process_layerwise_models("celeba", "/home/xzhang941/understanding-llr/celeba_resnet.pkl", "/home/xzhang941/understanding-llr/celeba_resnet.pkl", "/home/xzhang941/understanding-llr/celeba_resnet.pkl")
process_layerwise_models("waterbirds", "/home/xzhang941/understanding-llr/waterbirds_resnet.pkl", "/home/xzhang941/understanding-llr/waterbirds_resnet.pkl", "/home/xzhang941/understanding-llr/waterbirds_resnet.pkl")
#process_layerwise_models("civilcomments", "/home/xzhang941/understanding-llr/civilcomments_bert.pkl", "/home/xzhang941/understanding-llr/civilcomments_bert.pkl", "/home/xzhang941/understanding-llr/civilcomments_bert.pkl")
