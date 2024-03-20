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


class Uniform_test_400(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Uniform_test_400, self).__init__(model, labelled_loader, unlabelled_loader, rank,
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
        check_value = []
        cls_results = {}
        reg_results = {}
        density_list = {}
        label_list = {}

        # experiment:
        all_pred_dicts = []
        predicted_label_dict = {}
        density_label_1 = torch.zeros(1).to('cuda:0')
        density_label_2 = torch.zeros(1).to('cuda:0')
        density_label_3 = torch.zeros(1).to('cuda:0')


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
                all_pred_dicts.append(pred_dicts)
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
                        check_value.append(entropy)
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
                        "Pedestrian": total_num_pedestrain_pred,
                        "Cyclist": total_num_cyclist_pred,
                        # Corrected spelling
                    }
                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    # save the density records for the Stage 3
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx][
                        'pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

                    for label, density in zip(pred_dicts[batch_inx]['pred_labels'], pred_dicts[batch_inx][
                        'pred_box_unique_density']):
                        if label == 1:
                            density_label_1 += density
                        elif label == 2:
                            density_label_2 += density
                        elif label == 3:
                            density_label_3 += density

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        density_label_1 = density_label_1 / len(label_list)
        density_label_2 = density_label_2 / len(label_list)
        density_label_3 = density_label_3 / len(label_list)

        check_value.sort()
        log_data = [[idx, value] for idx, value in enumerate(check_value)]
        table = wandb.Table(data=log_data, columns=['idx', 'selection_value'])
        wandb.log({'value_dist_epoch_{}'.format(cur_epoch): wandb.plot.line(table, 'idx', 'selection_value',
                                                                            title='value_dist_epoch_{}'.format(
                                                                                cur_epoch))})

        # sort and get selected_frames
        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        # narrow down the scope 300 500
        selected_frames = list(select_dic.keys())[::-1][:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS) * 2]
        new_predicted_label_dict = {}
        new_label_list = {}
        new_density_list = {}
        for element in selected_frames:
            new_predicted_label_dict[element] = predicted_label_dict[element]
            new_label_list[element] = label_list[element]
            new_density_list[element] = density_list[element]
        total_num_car = torch.zeros(1).to('cuda:0')
        total_num_pedestrain = torch.zeros(1).to('cuda:0')
        total_num_cyclist = torch.zeros(1).to('cuda:0')
        selected_frames_label = []
        index_of_greedy = 0
        # 试试200, 300 如何
        while index_of_greedy <= 100:
            if index_of_greedy == 0:  # initially, we randomly select a frame.

                for key, value in new_predicted_label_dict.items():
                    Cyclist_value = value['Cyclist']
                    if Cyclist_value != 0:
                        total_num_car += value['Car']
                        total_num_pedestrain += value['Pedestrian']
                        total_num_cyclist += value['Cyclist']
                        del new_predicted_label_dict[key]
                        selected_frames_label.append(key)
                        break
                index_of_greedy += 1
            else:
                difference_init = float('inf')
                best_frame = None
                for key, value in new_predicted_label_dict.items():
                    current_num_car = total_num_car + value['Car']
                    current_num_Pedestrian = total_num_pedestrain + value['Pedestrian']
                    current_num_Cyclist = total_num_cyclist + value['Cyclist']

                    current_difference = abs(current_num_car - current_num_Pedestrian) + abs(
                        current_num_car - current_num_Cyclist) + abs(current_num_Pedestrian - current_num_Cyclist)

                    if current_difference < difference_init:
                        difference_init = current_difference
                        best_frame = key

                total_num_car += new_predicted_label_dict[best_frame]['Car']
                total_num_pedestrain += new_predicted_label_dict[best_frame]['Pedestrian']
                total_num_cyclist += new_predicted_label_dict[best_frame]['Cyclist']

                if best_frame is None:
                    index_of_greedy += 10000
                    break

                if best_frame is not None:
                    total_num_car += new_predicted_label_dict[best_frame]['Car'] if new_predicted_label_dict[best_frame][
                                                                                    'Car'] is not None else 0
                    total_num_pedestrain += new_predicted_label_dict[best_frame]['Pedestrian'] if \
                    new_predicted_label_dict[best_frame]['Pedestrian'] is not None else 0
                    total_num_cyclist += new_predicted_label_dict[best_frame]['Cyclist'] if \
                    new_predicted_label_dict[best_frame]['Cyclist'] is not None else 0

                del new_predicted_label_dict[best_frame]
                selected_frames_label.append(best_frame)
                index_of_greedy += 1

        density_car = torch.zeros(1).to('cuda:0')
        density_pedestrain = torch.zeros(1).to('cuda:0')
        density_cyclist = torch.zeros(1).to('cuda:0')
        selected_frames_density = []
        index_of_greedy = 0
        while index_of_greedy <= 100:
            if index_of_greedy == 0:
                for frame in selected_frames:
                    for label, density in zip(new_label_list[frame], new_density_list[frame]):
                        if label == 1:
                            density_car += density
                        elif label == 2:
                            density_pedestrain += density
                        elif label == 3:
                            density_cyclist += density
                    if density_car > 0 and density_pedestrain > 0 and density_cyclist > 0:

                        del new_label_list[frame]
                        del new_density_list[frame]
                        selected_frames_density.append(frame)
                        break
                index_of_greedy += 1
            else:
                difference_init = float('inf')
                best_frame = None
                for key in new_label_list:
                    cur_density_car = torch.zeros(1).to('cuda:0')
                    cur_density_pedestrain = torch.zeros(1).to('cuda:0')
                    cur_density_cyclist = torch.zeros(1).to('cuda:0')
                    for label, density in zip(new_label_list[key], new_density_list[key]):
                        if label == 1:
                            cur_density_car += density
                        elif label == 2:
                            cur_density_pedestrain += density
                        elif label == 3:
                            cur_density_cyclist += density

                    cur_density_car += density_car
                    cur_density_pedestrain += density_pedestrain
                    cur_density_cyclist += density_cyclist
                    current_difference = abs(cur_density_car / (index_of_greedy + 1) - density_label_1) + abs(
                        cur_density_pedestrain / (index_of_greedy + 1) - density_label_2) + abs(
                        cur_density_cyclist / (index_of_greedy + 1) - density_label_3)

                    if current_difference < difference_init:
                        difference_init = current_difference
                        best_frame = key

                for label, density in zip(new_label_list[best_frame], new_density_list[best_frame]):
                    if label == 1:
                        density_car += density
                    elif label == 2:
                        density_pedestrain += density
                    elif label == 3:
                        density_cyclist += density
                del new_label_list[best_frame]
                del new_density_list[best_frame]
                selected_frames_density.append(best_frame)
                index_of_greedy += 1

        common_frames = set(selected_frames_label) & set(selected_frames_density)
        current_frames = list(common_frames)
        if len(current_frames) < 100:
            for frame in selected_frames:
                if frame not in current_frames:
                    current_frames.append(frame)
                if len(current_frames) == 100:
                    break
        # save to local
        print('successfully saved selected_all_info for epoch {} for rank {}'.format(cur_epoch, self.rank))

        return current_frames[:100]


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
            if selected_frames[i] in label_dictionary:
                car_number_dict[selected_frames[i]] = label_dictionary[selected_frames[i]]


        sorted_keys = sorted(car_number_dict, key=lambda x: car_number_dict[x].item(), reverse=True)

        return sorted_keys[:100]