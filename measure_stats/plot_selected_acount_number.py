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

# def collect_info_individual(infomation, true_entropy, pred_entropy, density_mean_true,
# density_median_true, density_variance_true, density_mean_pred, density_median_pred, density_variance_pred
#                             ):
def collect_info_individual(element):
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

    total_num_car_pred = torch.zeros(1).to('cuda:0')
    total_num_cyclist_pred = torch.zeros(1).to('cuda:0')
    total_num_pedestrain_pred = torch.zeros(1).to('cuda:0')

    for infomation in element:
        for i_index in range(len(infomation['pred_labels'])):
            if infomation['pred_labels'][i_index] == 1:
                total_num_car_pred += 1
            elif infomation['pred_labels'][i_index] == 2:
                total_num_pedestrain_pred += 1
            elif infomation['pred_labels'][i_index] == 3:
                total_num_cyclist_pred += 1

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

    print("Actual counts:")
    print(f"Total number of cars: {total_num_car.cpu().item()}")
    print(f"Total number of cyclists: {total_num_cyclist.cpu().item()}")
    print(f"Total number of pedestrians: {total_num_pedestrain.cpu().item()}")

    print("\nPredicted counts:")
    print(f"Total number of predicted cars: {total_num_car_pred.cpu().item()}")
    print(f"Total number of predicted cyclists: {total_num_cyclist_pred.cpu().item()}")
    print(f"Total number of predicted pedestrians: {total_num_pedestrain_pred.cpu().item()}")
    return [total_num_car.cpu().item(), total_num_pedestrain.cpu().item(), total_num_cyclist.cpu().item(), total_num_car_pred.cpu().item(), total_num_pedestrain_pred.cpu().item(), total_num_cyclist_pred.cpu().item()]

def calculate_difference_score(car, pedestrian, cyclist):
    total_num = car + pedestrian + cyclist
    difference_score = abs(car/total_num - pedestrian/total_num) + abs(car/total_num - cyclist/total_num) + abs(pedestrian/total_num - cyclist/total_num)

    return difference_score

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
    for element in data:
        for infomation in element:

            total_num_car += infomation['num_bbox']['Car']
            total_num_cyclist += infomation['num_bbox']['Pedestrian']
            total_num_pedestrain += infomation['num_bbox']['Cyclist']
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
    base_path = "/people/cs/r/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/output/active-kitti_models/pv_rcnn_active_crb_record_static/select-100/active_label/"
    all_template = 'selected_all_info_epoch_{}_rank_0.pkl'
    file_template = 'selected_frames_epoch_{}_rank_0.pkl'

    all_cycylist_pedetrian_path = '/people/cs/r/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/output/active-kitti_models/pv_rcnn_active_crb_stage_2_stage_1_with_confidence_refinment/select-100/active_label/'

    # Initialize a dictionary to hold the data from each epoch
    path_elements = all_cycylist_pedetrian_path.split('/')
    save_file_name = path_elements[10]
    # Loop through the epochs 40, 80, 120, ..., 240
      # Start at 40, end at 240, step by 40
    total_num_car = 0
    total_num_pedestrain = 0
    total_num_cyclist = 0
    total_num_car_pred = 0
    total_num_pedestrain_pred = 0
    total_num_cyclist_pred = 0
    true_ratio = []
    predicted_ratio = []
    total_true_ratio = 0
    total_predicted_ratio = 0
    all_file_path = base_path + all_template.format(40)
    all_data = read_pkl_file(all_file_path)
    for i in range(1, 7):
        # file_path = base_path + file_template.format(240)


        data = read_pkl_file(all_cycylist_pedetrian_path + file_template.format(i * 40))

        # new_data = collect_info(data)
        current_all_info = []

        for j in range(len(data['frame_id'])):
            current_all_info.append(all_data[int(data['frame_id'][j])])
        # del all_data
        num_car, num_pedestrain, num_cyclist, car_pred, pedestrain_pred, cyclist_pred = collect_info_individual(current_all_info)
        total_num_car += num_car
        total_num_pedestrain += num_pedestrain
        total_num_cyclist += num_cyclist

        total_num_car_pred += car_pred
        total_num_pedestrain_pred += pedestrain_pred
        total_num_cyclist_pred += cyclist_pred
        true_ratio.append(calculate_difference_score(num_car, num_pedestrain, num_cyclist))
        predicted_ratio.append(calculate_difference_score(car_pred, pedestrain_pred, cyclist_pred))

    total_true_ratio = calculate_difference_score(total_num_car, total_num_pedestrain, total_num_cyclist)
    total_predicted_ratio = calculate_difference_score(total_num_car_pred, total_num_pedestrain_pred, total_num_cyclist_pred)
    statistics_dictionary = {}
    statistics_dictionary['total_num_car'] = total_num_car
    statistics_dictionary['total_num_pedestrain'] = total_num_pedestrain
    statistics_dictionary['total_num_cyclist'] = total_num_cyclist
    statistics_dictionary['total_num_car_pred'] = total_num_car_pred
    statistics_dictionary['total_num_pedestrain_pred'] = total_num_pedestrain_pred
    statistics_dictionary['total_num_cyclist_pred'] = total_num_cyclist_pred
    statistics_dictionary['true_ratio'] = true_ratio
    statistics_dictionary['predicted_ratio'] = predicted_ratio
    statistics_dictionary['total_true_ratio'] = total_true_ratio
    statistics_dictionary['total_predicted_ratio'] = total_predicted_ratio
    with open(save_file_name, 'wb') as file:
        pickle.dump(statistics_dictionary, file)
    print("finished")

if __name__ == '__main__':
    main()