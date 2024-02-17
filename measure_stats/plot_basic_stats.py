import pickle
import torch
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import math
# Define the path to the .pkl file

# Function to read and return the contents of the .pkl file
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

def calcuate_mean_median_variance(infomation):
    predicted_label = infomation['pred_labels']
    density = infomation['pred_box_unique_density']

    # Initialize dictionaries to hold density values for each class
    densities = {1: [], 2: [], 3: []}

    # Populate the dictionaries based on predicted labels
    for label, dens in zip(predicted_label, density):
        densities[label.item()].append(dens)

    # Initialize lists to store the calculated statistics for each class
    means = []
    medians = []
    variances = []

    # Calculate statistics for each class
    for label in [1, 2, 3]:  # Iterate through class labels
        if densities[label]:  # Check if there are any elements for the class
            class_densities = torch.stack(densities[label])  # Convert list to tensor
            means.append(torch.mean(class_densities).item())
            medians.append(torch.median(class_densities).item())
            variances.append(torch.var(class_densities).item())
        else:
            # Append 0 if there are no elements for the class
            means.append(0)
            medians.append(0)
            variances.append(0)

    return [means, medians, variances]

def calcualte_mean(density_mean_true):
    density_mean_true = [[0 if math.isnan(element) else element for element in sublist] for sublist in density_mean_true]
    density_mean_true = [[elem.item() if isinstance(elem, torch.Tensor) else elem for elem in sublist] for sublist in density_mean_true]
    position_means = []
    for position in range(3):  # Assuming each sublist has 3 positions
        # Extract values for the current position from each sublist, converting tensors to scalars
        position_values = [
            sublist[position].item() if isinstance(sublist[position], torch.Tensor) else sublist[position]
            for sublist in density_mean_true
        ]

        # Calculate the mean of these values
        position_mean = sum(position_values) / len(position_values)
        position_means.append(position_mean)
    return position_means


def plot_true_predicted_entropy(true_entropy, pred_entropy):
    # Assuming true_entropy and pred_entropy are lists of tensors on the GPU
    # Assuming true_entropy and pred_entropy are lists containing tensors and integers
    true_entropy_cpu = [tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor for tensor in true_entropy]
    pred_entropy_cpu = [tensor.cpu().numpy() if hasattr(tensor, 'cpu') else tensor for tensor in pred_entropy]
    true_entropy_cpu = [arr.item() for arr in true_entropy_cpu]
    pred_entropy_cpu = [arr.item() if isinstance(arr, np.ndarray) else arr for arr in pred_entropy_cpu]
    # Generate x-values in the style 0, 1, 2, 3, 4, 5, 6, 7...
    x_values = range(len(true_entropy_cpu))

    # Plotting
    differences = [true - pred for true, pred in zip(true_entropy_cpu, pred_entropy_cpu)]

    plt.figure(figsize=(12, 6))
    plt.plot(differences, label='Difference (True - Pred)')
    plt.xlabel('Index')
    plt.ylabel('Difference in Entropy')
    plt.title('Difference Between True and Predicted Entropy')
    plt.legend()
    plt.savefig('plot.png', dpi=300)
    plt.show()

def collect_info(data):
    true_entropy = []
    pred_entropy = []
    density_mean_true = []
    density_median_true = []
    density_variance_true = []
    density_mean_pred = []
    density_median_pred = []
    density_variance_pred = []
    total_num_car = torch.zeros(1).to('cuda:0')
    total_num_cyclist = torch.zeros(1).to('cuda:0')
    total_num_pedestrain = torch.zeros(1).to('cuda:0')

    total_num_car_select = torch.zeros(1).to('cuda:0')
    total_num_cyclist_select = torch.zeros(1).to('cuda:0')
    total_num_pedestrain_select = torch.zeros(1).to('cuda:0')
    selected_frame = []

    for element in data:
        for infomation in element:



            total_num_car += infomation['num_bbox']['Car']
            total_num_pedestrain += infomation['num_bbox']['Pedestrian']
            total_num_cyclist += infomation['num_bbox']['Cyclist']
            # prediceted_entropy
            value, counts = torch.unique(infomation['pred_labels'], return_counts=True)
            if len(value) == 0:
                entropy = 0
                pred_entropy.append(entropy)
            else:
                # calculates the shannon entropy of the predicted labels of bounding boxes
                unique_proportions = torch.ones(3).cuda()
                unique_proportions[value - 1] = counts.float()
                entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                pred_entropy.append(entropy)
            # true_entropy
            counts = torch.zeros(1).cuda()
            unique_proportions = torch.ones(3).cuda()
            car = infomation['num_bbox']['Car']
            Pedestrain = infomation['num_bbox']['Pedestrian']
            Cyclist = infomation['num_bbox']['Cyclist']
            counts += car + Pedestrain + Cyclist
            if car != 0:
                unique_proportions[0] = car
            if Pedestrain != 0:
                unique_proportions[1] = Pedestrain
            if Cyclist != 0:
                unique_proportions[2] = Cyclist
            entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
            true_entropy.append(entropy)
            density_mean_true.append([infomation['mean_points']['Car'], infomation['mean_points']['Pedestrian'], infomation['mean_points']['Cyclist']])
            density_median_true.append([infomation['median_points']['Car'], infomation['median_points']['Pedestrian'], infomation['median_points']['Cyclist']])
            density_variance_true.append([infomation['variance_points']['Car'], infomation['variance_points']['Pedestrian'], infomation['variance_points']['Cyclist']])
            pred_stat = calcuate_mean_median_variance(infomation)
            density_mean_pred.append(pred_stat[0])
            density_median_pred.append(pred_stat[1])
            density_variance_pred.append(pred_stat[2])

    density_mean_true = calcualte_mean(density_mean_true)
    density_median_true = calcualte_mean(density_median_true)
    density_variance_true = calcualte_mean(density_variance_true)
    density_mean_pred = calcualte_mean(density_mean_pred)
    density_median_pred = calcualte_mean(density_median_pred)
    density_variance_pred = calcualte_mean(density_variance_pred)
    pred_entropy = [elem.item() if isinstance(elem, torch.Tensor) else float(elem) for elem in pred_entropy]
    mean_pred_entropy_pred = sum(pred_entropy) / len(pred_entropy)
    true_entropy = [elem.item() if isinstance(elem, torch.Tensor) else float(elem) for elem in true_entropy]
    mean_pred_entropy_true = sum(true_entropy) / len(true_entropy)
    return [density_mean_true, density_median_true, density_variance_true, density_mean_pred, density_median_pred,density_variance_pred, mean_pred_entropy_pred, mean_pred_entropy_true]
# Use the function to read the .pkl file
def main():
    base_path = '/home/012/r/rx/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/output/active-kitti_models/pv_rcnn_active_crb_1/select-100/active_label/'
    file_template = 'selected_all_info_epoch_{}_rank_0.pkl'

    # Initialize a dictionary to hold the data from each epoch


    # Loop through the epochs 40, 80, 120, ..., 240
      # Start at 40, end at 240, step by 40
    file_path = base_path + file_template.format(40)
    data = read_pkl_file(file_path)
    new_data = collect_info(data)
    with open("240_statistics.pkl", "wb") as file:
        pickle.dump(new_data, file)




    base_path = '/home/012/r/rx/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/measure_stats/'
    file_template = '{}_statistics.pkl'
    overall_data = []
    for i in range(1, 7):

        file_path = base_path + file_template.format(i * 40)
        data = read_pkl_file(file_path)
        overall_data.append(data)
    print(overall_data)
    density_mean_true_car = []
    density_median_true_car = []
    density_variance_true_car = []
    density_mean_true_Pedestrian = []
    density_median_true_Pedestrian = []
    density_variance_true_Pedestrian = []
    density_mean_true_Cyclist = []
    density_median_true_Cyclist = []
    density_variance_true_Cyclist = []

    density_mean_predicted_car = []
    density_median_predicted_car = []
    density_variance_predicted_car = []
    density_mean_predicted_Pedestrian = []
    density_median_predicted_Pedestrian = []
    density_variance_predicted_Pedestrian = []
    density_mean_predicted_Cyclist = []
    density_median_predicted_Cyclist = []
    density_variance_predicted_Cyclist = []

    predected_entropy = []
    true_entropy = []
    for element in overall_data:
        density_mean_true_car.append(element[0][0])
        density_median_true_car.append(element[1][0])
        density_variance_true_car.append(element[2][0])
        density_mean_true_Pedestrian.append(element[0][1])
        density_median_true_Pedestrian.append(element[1][1])
        density_variance_true_Pedestrian.append(element[2][1])
        density_mean_true_Cyclist.append(element[0][2])
        density_median_true_Cyclist.append(element[1][2])
        density_variance_true_Cyclist.append(element[2][2])

        density_mean_predicted_car.append(element[3][0])
        density_median_predicted_car.append(element[4][0])
        density_variance_predicted_car.append(element[5][0])
        density_mean_predicted_Pedestrian.append(element[3][1])
        density_median_predicted_Pedestrian.append(element[4][1])
        density_variance_predicted_Pedestrian.append(element[5][1])
        density_mean_predicted_Cyclist.append(element[3][2])
        density_median_predicted_Cyclist.append(element[4][2])
        density_variance_predicted_Cyclist.append(element[5][2])

        predected_entropy.append(element[6])
        true_entropy.append(element[7])

    n_groups = len(density_mean_true_car)

    # Create subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))  # 3x3 grid for mean, median, variance of each class
    fig.suptitle('Comparison of True and Predicted Densities')

    # Titles for each subplot
    titles = ['Mean Density', 'Median Density', 'Variance of Density']
    classes = ['Car', 'Pedestrian', 'Cyclist']

    true_data = [
        [density_mean_true_car, density_mean_true_Pedestrian, density_mean_true_Cyclist],
        [density_median_true_car, density_median_true_Pedestrian, density_median_true_Cyclist],
        [density_variance_true_car, density_variance_true_Pedestrian, density_variance_true_Cyclist]
    ]

    predicted_data = [
        [density_mean_predicted_car, density_mean_predicted_Pedestrian, density_mean_predicted_Cyclist],
        [density_median_predicted_car, density_median_predicted_Pedestrian, density_median_predicted_Cyclist],
        [density_variance_predicted_car, density_variance_predicted_Pedestrian, density_variance_predicted_Cyclist]
    ]

    # Plotting
    for i, metric in enumerate(true_data):
        for j, cls in enumerate(metric):
            index = np.arange(n_groups)
            bar_width = 0.35

            axs[i, j].bar(index, cls, bar_width, label='True')
            axs[i, j].bar(index + bar_width, predicted_data[i][j], bar_width, label='Predicted')

            axs[i, j].set_title(f'{titles[i]} for {classes[j]}')
            axs[i, j].set_xticks(index + bar_width / 2)
            axs[i, j].set_xticklabels([str(x) for x in range(1, n_groups + 1)])
            axs[i, j].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plot.png', dpi=300)
    # Show plot
    plt.show()

    n_groups = len(true_entropy)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the positions and width for the bars
    index = np.arange(n_groups)
    bar_width = 0.35

    # Plotting the true and predicted entropy
    ax.bar(index, true_entropy, bar_width, label='True Entropy')
    ax.bar(index + bar_width, predected_entropy, bar_width, label='Predicted Entropy')

    # Adding labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Sample')
    ax.set_ylabel('Entropy')
    ax.set_title('Comparison of True and Predicted Entropy')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([str(x) for x in range(1, n_groups + 1)])
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('plot_2.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()