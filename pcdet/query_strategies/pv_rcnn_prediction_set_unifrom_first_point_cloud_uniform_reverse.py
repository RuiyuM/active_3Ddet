import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
from pcdet.datasets import build_active_dataloader
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import numpy as np
import wandb
import time
import scipy
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from sklearn.neighbors import KernelDensity
from scipy.cluster.vq import vq
from typing import Dict, List
import os
import pickle
import copy

class Set_Unifrom_invidual_uniform_reverse(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Set_Unifrom_invidual_uniform_reverse, self).__init__(model, labelled_loader, unlabelled_loader, rank,
                                                         active_label_dir, cfg)

        # coefficients controls the ratio of selected subset
        self.k1 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K1', 5)
        self.k2 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K2', 3)

        # bandwidth for the KDE in the GPDB module
        self.bandwidth = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'BANDWDITH', 5)
        # ablation study for prototype selection
        self.prototype = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'CLUSTERING', 'kmeans++')
        # controls the boundary of the uniform prior distribution
        self.alpha = 0.95

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None):
        select_dic = {}

        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)

        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        self.enable_dropout(self.model)
        num_class = len(self.labelled_loader.dataset.class_names)
        check_value = {}
        cls_results = {}
        reg_results = {}
        density_list = {}
        label_list = {}
        predicted_label_dict = {}
        # experiment:

        '''
        -------------  Stage 1: Consise Label Sampling ----------------------
        '''
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)

                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])

                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)

                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        cls_weight = self.adding_weight_to_unique_proportions(pred_dicts, batch_inx)
                        reg_weight = self.adding_weight_regression(pred_dicts, batch_inx)
                        unique_proportions = (unique_proportions / sum(counts) * cls_weight * reg_weight)
                        entropy = Categorical(probs=unique_proportions).entropy()
                        check_value[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    total_num_car_pred = torch.zeros(1).to('cuda:0')
                    total_num_cyclist_pred = torch.zeros(1).to('cuda:0')
                    total_num_pedestrain_pred = torch.zeros(1).to('cuda:0')

                    for i_index in range(len(pred_dicts[batch_inx]['pred_labels'])):
                        if pred_dicts[batch_inx]['pred_labels'][i_index] == 1:
                            total_num_car_pred += 1
                        elif pred_dicts[batch_inx]['pred_labels'][i_index] == 2:
                            total_num_pedestrain_pred += 1
                        elif pred_dicts[batch_inx]['pred_labels'][i_index] == 3:
                            total_num_cyclist_pred += 1

                    predicted_label_dict[unlabelled_batch['frame_id'][batch_inx]] = {
                        "Car": total_num_car_pred,
                        "Cyclist": total_num_cyclist_pred,
                        "Pedestrian": total_num_pedestrain_pred  # Corrected spelling
}
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        total_num_car = torch.zeros(1).to('cuda:0')
        total_num_pedestrain = torch.zeros(1).to('cuda:0')
        total_num_cyclist = torch.zeros(1).to('cuda:0')
        selected_frames = []
        index_of_greedy = 0
        # 试试500



        # Assuming predicted_label_dict is your original dictionary
        predicted_label_dict_deep_copy = copy.deepcopy(predicted_label_dict)

        # Now, changes to predicted_label_dict_deep_copy will not affect predicted_label_dict and vice versa.

        while total_num_pedestrain <= 2000 and index_of_greedy <= 10000:
            if index_of_greedy == 0:  # initially, we randomly select a frame.

                for key, value in predicted_label_dict.items():
                    Cyclist_value = value['Cyclist']
                    if Cyclist_value != 0:
                        total_num_car += value['Car']
                        total_num_pedestrain += value['Pedestrian']
                        total_num_cyclist += value['Cyclist']
                        del predicted_label_dict[key]
                        selected_frames.append(key)
                        break
                index_of_greedy += 1
            else:
                difference_init = float('inf')
                best_frame = None
                for key, value in predicted_label_dict.items():
                    current_num_car = total_num_car + value['Car']
                    current_num_Pedestrian = total_num_pedestrain + value['Pedestrian']
                    current_num_Cyclist = total_num_cyclist + value['Cyclist']

                    current_difference = abs(current_num_car - current_num_Pedestrian) + abs(
                        current_num_car - current_num_Cyclist) + abs(current_num_Pedestrian - current_num_Cyclist)

                    if current_difference < difference_init:
                        difference_init = current_difference
                        best_frame = key

                total_num_car += predicted_label_dict[best_frame]['Car']
                total_num_pedestrain += predicted_label_dict[best_frame]['Pedestrian']
                total_num_cyclist += predicted_label_dict[best_frame]['Cyclist']

                if best_frame is None:
                    index_of_greedy += 10000
                    break

                if best_frame is not None:
                    total_num_car += predicted_label_dict[best_frame]['Car'] if predicted_label_dict[best_frame]['Car'] is not None else 0
                    total_num_pedestrain += predicted_label_dict[best_frame]['Pedestrian'] if predicted_label_dict[best_frame]['Pedestrian'] is not None else 0
                    total_num_cyclist += predicted_label_dict[best_frame]['Cyclist'] if predicted_label_dict[best_frame]['Cyclist'] is not None else 0

                del predicted_label_dict[best_frame]
                selected_frames.append(best_frame)
                index_of_greedy += 1



        selected_frames = self.sorting_out_excessive_car_samples(selected_frames, predicted_label_dict_deep_copy)

        # save to local
        print('successfully saved selected_all_info for epoch {} for rank {}'.format(cur_epoch, self.rank))

        return selected_frames


    def adding_weight_to_unique_proportions(self, pred_dicts, batch_inx):
        pred_scores = pred_dicts[batch_inx]['pred_logits']
        pred_labels = pred_dicts[batch_inx]['pred_labels']
        device = pred_scores.device

        # Assuming labels are 1, 2, 3, adjust size if you have more labels
        scores_list = [torch.tensor([], device=device) for _ in range(3)]  # List to hold scores for each label

        # Group scores by label
        for label, score in zip(pred_labels, pred_scores):
            index = label.item() - 1  # Adjusting for 0-based indexing
            scores_list[index] = torch.cat((scores_list[index], score.unsqueeze(0)), dim=0)

        # Calculate entropy for each label
        entropy_list = torch.zeros(3, device=device)  # Initialize entropy list
        for i, scores in enumerate(scores_list):
            if scores.nelement() == 0:
                continue  # Skip if no scores for this label
            # Normalize scores to probabilities
            probs = F.softmax(scores, dim=1)
            # Calculate entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1)
            entropy_list[i] = entropy.mean()

        return entropy_list

    def adding_weight_regression(self, pred_dicts, batch_inx):
        pred_logits = pred_dicts[batch_inx]['pred_scores']
        device = pred_logits.device
        pred_labels = pred_dicts[batch_inx]['pred_labels']
        scores_sum = torch.zeros(3)  # For labels 1, 2, 3
        # Initialize a tensor to count the occurrences of each label
        label_counts = torch.zeros(3)

        # Iterate over each label and score
        for label, score in zip(pred_labels, pred_logits):
            # Subtract 1 from label to use it as an index (since labels are 1-based and indices are 0-based)
            index = label.item() - 1
            scores_sum[index] += score.item()
            label_counts[index] += 1

        # Compute the mean score for each label
        mean_confidence_box = scores_sum / label_counts
        mean_confidence_box = torch.nan_to_num(mean_confidence_box)
        return mean_confidence_box.to(device)

    def sorting_out_excessive_car_samples(self, selected_frames, label_dictionary):
        car_number_dict = {}
        for i in range(len(selected_frames)):
            frame_key = selected_frames[i]
            # Check if the key exists in label_dictionary
            if frame_key in label_dictionary:
                car_number_dict[frame_key] = label_dictionary[frame_key]
            else:
                # Key is not in label_dictionary, skip this iteration and move forward
                continue
        new_car_number_dict = {}
        for key, value in car_number_dict.items():
            new_car_number_dict[key] = self.calculate_closeness(value)

        sorted_keys = sorted(new_car_number_dict, key=lambda x: new_car_number_dict[x], reverse=True)

        return sorted_keys[:100]

    def calculate_closeness(self, sample):
        # Assuming 'Car', 'Cyclist', 'Pedestrian' are keys with tensor values
        total = sample['Car'] + sample['Cyclist'] + sample['Pedestrian']

        # Check if the total is 0 to avoid division by zero
        # Since total is a tensor, we use torch.all() to check if all elements are 0
        if torch.all(total == 0):
            return float('inf')  # This will effectively exclude samples with no detections

        ideal_count = total / 3.0
        car_diff = torch.abs(sample['Car'] - ideal_count)
        cyclist_diff = torch.abs(sample['Cyclist'] - ideal_count)
        pedestrian_diff = torch.abs(sample['Pedestrian'] - ideal_count)

        # Sum of absolute differences. Since these are tensors, the result is also a tensor.
        # Use .item() to extract a single value from the tensor if it contains only one element.
        return car_diff + cyclist_diff + pedestrian_diff