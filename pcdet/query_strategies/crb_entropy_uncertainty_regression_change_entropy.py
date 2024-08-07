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


class Uncertainty_Regression_CRBSampling_change_entropy(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Uncertainty_Regression_CRBSampling_change_entropy, self).__init__(model, labelled_loader, unlabelled_loader, rank,
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
                # all_pred_dicts.append(pred_dicts)
                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    cls_weight = self.adding_weight_to_unique_proportions(pred_dicts, batch_inx)
                    reg_weight = self.adding_weight_regression(pred_dicts, batch_inx)
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    elif len(value) == 1:
                        # Only one class present
                        unique_proportions = torch.zeros(num_class).cuda()
                        existing_class_index = value[0] - 1
                        existing_count = counts.float().item()
                        unique_proportions[existing_class_index] = existing_count

                        # Calculate missing classes' counts based on your requirements
                        missing_classes_indices = [i for i in range(num_class) if i != existing_class_index]
                        unique_proportions[missing_classes_indices[0]] = 3 * existing_count
                        if len(missing_classes_indices) > 1:
                            unique_proportions[missing_classes_indices[1]] = 6 * existing_count

                        # Calculate entropy
                        unique_proportions = (unique_proportions / unique_proportions.sum() * cls_weight * reg_weight)
                        entropy = Categorical(probs=unique_proportions).entropy()
                    elif len(value) == 2:
                        # Two classes present, adjust the count for the missing class
                        unique_proportions = torch.zeros(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()

                        # Find the missing class and adjust its count
                        missing_class_index = [i for i in range(1, num_class + 1) if i not in value.cpu().numpy()][
                                                  0] - 1
                        missing_class_count = 6 * counts.float().mean().item()
                        unique_proportions[
                            missing_class_index] = missing_class_count  # Adjusted as per the updated requirement

                        # Calculate entropy
                        unique_proportions = (unique_proportions / unique_proportions.sum() * cls_weight * reg_weight)
                        entropy = Categorical(probs=unique_proportions).entropy()
                    else:
                        # Three or more classes present
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()

                        # Calculate entropy
                        # unique_proportions = (unique_proportions / sum(counts) * cls_weight * reg_weight)
                        unique_proportions = (unique_proportions / unique_proportions.sum() * cls_weight * reg_weight)
                        entropy = Categorical(probs=unique_proportions).entropy()
                    check_value.append(entropy)

                    # save the hypothetical labels for the regression heads at Stage 2
                    # cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    # reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    # save the density records for the Stage 3
                    # density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx][
                    #     'pred_box_unique_density']


            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()



        # sort and get selected_frames
        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        # narrow down the scope
        selected_frames = list(select_dic.keys())[::-1][:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS)]

        # save to local
        with open(os.path.join(self.active_label_dir,
                               'selected_all_info_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(all_pred_dicts, f)
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