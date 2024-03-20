import pickle
import torch
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import math


def read_pkl_file(file_path_info):
    try:
        # Open the file in binary read mode
        with open(file_path_info, 'rb') as file:
            # Use pickle to load the file's contents
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"Error reading the .pkl file: {e}")
        return None

def main():
    # List of pickle file names
    file_names = [
        "pv_rcnn_active_crb_stage_2_stage_1",
        "pv_rcnn_active_crb_record_static",
        "pv_rcnn_active_crb_uncertainty_Regression_confi",
        "pv_rcnn_active_crb_uncertainty_Regression_confi_with_predected_least_car",
        "facility_location_active_crb_stage_2_stage_1",
        "max_coverage_active_crb_stage_2_stage_1"
    ]

    # Initialize lists to hold the ratios from all files
    all_true_ratios = []
    all_predicted_ratios = []

    # Load the data from each file
    for file_name in file_names:
        data = read_pkl_file(file_name)
        all_true_ratios.append(data['true_ratio'])
        all_predicted_ratios.append(data['predicted_ratio'])

        # Plotting
    plt.figure(figsize=(14, 6))

    # Plot all true_ratios
    plt.subplot(1, 2, 1)
    for i, true_ratio in enumerate(all_true_ratios):
        plt.plot(true_ratio, label=f'File {i + 1}: {file_names[i].replace(".pkl", "")}')
    plt.title('True Ratios from All Files')
    plt.xlabel('Sample Index')
    plt.ylabel('Ratio')
    plt.legend()

    # Save the true ratios plot
    plt.savefig('true_ratios_plot.png')

    # Plot all predicted_ratios
    plt.subplot(1, 2, 2)
    for i, predicted_ratio in enumerate(all_predicted_ratios):
        plt.plot(predicted_ratio, label=f'File {i + 1}: {file_names[i].replace(".pkl", "")}')
    plt.title('Predicted Ratios from All Files')
    plt.xlabel('Sample Index')
    plt.ylabel('Ratio')
    plt.legend()

    # Save the predicted ratios plot
    plt.savefig('predicted_ratios_plot.png')

    plt.tight_layout()

if __name__ == "__main__":
    main()