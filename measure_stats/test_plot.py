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


def collect_info_individual(element):
    selected_frames = []
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

    for key, infomation in element.items():
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
        # prediceted_en


    print("Actual counts:")
    print(f"Total number of cars: {total_num_car.cpu().item()}")
    print(f"Total number of cyclists: {total_num_cyclist.cpu().item()}")
    print(f"Total number of pedestrians: {total_num_pedestrain.cpu().item()}")

    print("\nPredicted counts:")
    print(f"Total number of predicted cars: {total_num_car_pred.cpu().item()}")
    print(f"Total number of predicted cyclists: {total_num_cyclist_pred.cpu().item()}")
    print(f"Total number of predicted pedestrians: {total_num_pedestrain_pred.cpu().item()}")

    return selected_frames

# path = "/home/012/r/rx/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/output/active-kitti_models/pv_rcnn_select_all_cyclist_pedestrian/select-100/active_label/selected_all_info_epoch_40_rank_0.pkl"
# data = read_pkl_file(path)

def main():
    base_path = '/home/012/r/rx/rxm210041/Desktop/test_3d_active/CRB-active-3Ddet/output/active-kitti_models/pv_rcnn_active_crb_record_static/select-100/active_label/'
    all_template = 'selected_all_info_epoch_{}_rank_0.pkl'
    file_template = 'selected_frames_epoch_{}_rank_0.pkl'

    # Initialize a dictionary to hold the data from each epoch


    # Loop through the epochs 40, 80, 120, ..., 240
      # Start at 40, end at 240, step by 40
    overall_data = []
    for i in range(1, 7):
        file_path = base_path + file_template.format(i * 40)
        all_file_path = base_path + all_template.format(1 * 40)
        all_data = read_pkl_file(all_file_path)
        # data = read_pkl_file(file_path)

        # # new_data = collect_info(data)
        # current_all_info = []
        #
        # for j in range(len(data['frame_id'])):
        #     current_all_info.append(all_data[int(data['frame_id'][j])])

        extracted = collect_info_individual(all_data)
        # overall_data.append(extracted)

    print("finished")

if __name__ == '__main__':
    main()