
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
from scipy.stats import gaussian_kde

class Double_greedy_not_mean_2(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Double_greedy_not_mean_2, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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


        true_label = {}
        gt_points = {}
        gt_points_label = {}
        category_labels = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
        #experiment:
        all_pred_dicts = {}

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
                    all_pred_dicts[int(unlabelled_batch['frame_id'][batch_inx])] = pred_dicts[batch_inx]
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs = unique_proportions / sum(counts)).entropy()
                        check_value.append(entropy)

                    # save the density records for the Stage 3

                    current_gt_points = pred_dicts[batch_inx]['gt_points']

                    labels = []
                    values = []

                    # Iterate through each category in the data
                    for category, tensor in current_gt_points.items():
                        if isinstance(tensor,
                                      torch.Tensor) and tensor.numel() > 0:  # Check if it's a tensor with elements
                            # Append the label for each element in the tensor
                            labels.extend([category_labels[category]] * tensor.size(0))
                            # Append the values
                            values.extend(tensor.tolist())

                    # Convert lists to tensors
                    label_tensor = torch.tensor(labels, dtype=torch.int)
                    value_tensor = torch.tensor(values, dtype=torch.float)
                    gt_points[unlabelled_batch['frame_id'][batch_inx]] = value_tensor.to('cuda:0')
                    gt_points_label[unlabelled_batch['frame_id'][batch_inx]] = label_tensor.to('cuda:0')


                    true_label[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['num_bbox']


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
        # 试试300如何
        while total_num_cyclist <= 350 and index_of_greedy <= 10000:
            if index_of_greedy == 0:  # initially, we randomly select a frame.

                for key, value in true_label.items():
                    Cyclist_value = value['Cyclist']
                    if Cyclist_value != 0:
                        total_num_car += value['Car']
                        total_num_pedestrain += value['Pedestrian']
                        total_num_cyclist += value['Cyclist']
                        del true_label[key]
                        selected_frames.append(key)
                        break
                index_of_greedy += 1
            else:
                difference_init = float('inf')
                best_frame = None
                for key, value in true_label.items():
                    current_num_car = total_num_car + value['Car']
                    current_num_Pedestrian = total_num_pedestrain + value['Pedestrian']
                    current_num_Cyclist = total_num_cyclist + value['Cyclist']

                    current_difference = abs(current_num_car - current_num_Pedestrian) + abs(
                        current_num_car - current_num_Cyclist) + abs(current_num_Pedestrian - current_num_Cyclist)

                    if current_difference < difference_init:
                        difference_init = current_difference
                        best_frame = key
                if best_frame is None:
                    index_of_greedy += 10000
                    break

                if best_frame is not None:
                    total_num_car += true_label[best_frame]['Car'] if true_label[best_frame]['Car'] is not None else 0
                    total_num_pedestrain += true_label[best_frame]['Pedestrian'] if true_label[best_frame]['Pedestrian'] is not None else 0
                    total_num_cyclist += true_label[best_frame]['Cyclist'] if true_label[best_frame]['Cyclist'] is not None else 0

                del true_label[best_frame]
                selected_frames.append(best_frame)
                index_of_greedy += 1

        '''
        -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
        '''

        sampled_density_list = [gt_points[i] for i in selected_frames]
        sampled_label_list = [gt_points_label[i] for i in selected_frames]

        """ Build the uniform distribution for each class """
        start_time = time.time()
        density_all = torch.cat(list(gt_points.values()), 0)
        label_all = torch.cat(list(gt_points_label.values()), 0)
        unique_labels, label_counts = torch.unique(label_all, return_counts=True)
        sorted_density = [torch.sort(density_all[label_all == unique_label])[0] for unique_label in unique_labels]
        global_density_max = [int(sorted_density[unique_label][-1]) for unique_label in range(len(unique_labels))]
        global_density_min = [int(sorted_density[unique_label][0]) for unique_label in range(len(unique_labels))]

        x_eval = []
        for i in range(num_class):
            x_min = global_density_min[i]
            x_max = global_density_max[i]
            x_eval.append(np.linspace(x_min, x_max, 1000))  # or 2000, based on need and computational feasibility
        uniform_dist_per_cls = []
        for j in range(num_class):
            kde = gaussian_kde(sorted_density[j].cpu())

            # Evaluate the KDE over a range of values
            density_values = kde.evaluate(x_eval[j])
            uniform_dist_per_cls.append(density_values)

        print("--- Build the uniform distribution running time: %s seconds ---" % (time.time() - start_time))

        density_list, label_list, frame_id_list = sampled_density_list, sampled_label_list, selected_frames

        selected_frames: List[str] = []
        selected_box_densities: torch.tensor = torch.tensor([]).cuda()
        selected_box_labels: torch.tensor = torch.tensor([]).cuda()

        # looping over N_r samples
        if self.rank == 0:
            pbar = tqdm.tqdm(total=self.cfg.ACTIVE_TRAIN.SELECT_NUMS, leave=leave_pbar,
                             desc='global_density_div_for_epoch_%d' % cur_epoch, dynamic_ncols=True)

        for j in range(self.cfg.ACTIVE_TRAIN.SELECT_NUMS):
            if j == 0:  # initially, we randomly select a frame.

                selected_frames.append(frame_id_list[j])
                selected_box_densities = torch.cat((selected_box_densities, density_list[j]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[j]))

                # remove selected frame
                del density_list[0]
                del label_list[0]
                del frame_id_list[0]

            else:  # go through all the samples and choose the frame that can most reduce the KL divergence
                best_frame_id = None
                best_frame_index = None
                best_inverse_coff = -1

                for i in range(len(density_list)):
                    unique_proportions = np.zeros(num_class)
                    KL_scores_per_cls = np.zeros(num_class)

                    for cls in range(num_class):
                        if (label_list[i] == cls + 1).sum() == 0:
                            unique_proportions[cls] = 1
                            KL_scores_per_cls[cls] = np.inf
                        else:
                            # get existing selected box densities
                            selected_box_densities_cls = selected_box_densities[selected_box_labels == (cls + 1)]
                            # append new frame's box densities to existing one
                            selected_box_densities_cls = torch.cat((selected_box_densities_cls,
                                                                    density_list[i][label_list[i] == (cls + 1)]))
                            # initialize kde
                            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(
                                selected_box_densities_cls.cpu().numpy()[:, None])

                            logprob = kde.score_samples(x_eval[cls][:, None])
                            KL_score_per_cls = scipy.stats.entropy(uniform_dist_per_cls[cls], np.exp(logprob))
                            KL_scores_per_cls[cls] = KL_score_per_cls
                            # ranging from 0 to 1 this thing is measure how different the current distribution is from the unifrom one
                            unique_proportions[cls] = 2 / np.pi * np.arctan(np.pi / 2 * KL_score_per_cls)

                    inverse_coff = np.mean(1 - unique_proportions)
                    # KL_save_list.append(inverse_coff)
                    if inverse_coff > best_inverse_coff:
                        best_inverse_coff = inverse_coff
                        best_frame_index = i
                        best_frame_id = frame_id_list[i]

                # remove selected frame
                selected_box_densities = torch.cat((selected_box_densities, density_list[best_frame_index]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[best_frame_index]))
                del density_list[best_frame_index]
                del label_list[best_frame_index]
                del frame_id_list[best_frame_index]

                selected_frames.append(best_frame_id)

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        self.model.eval()
        # returned the index of acquired bounding boxes
        return selected_frames
