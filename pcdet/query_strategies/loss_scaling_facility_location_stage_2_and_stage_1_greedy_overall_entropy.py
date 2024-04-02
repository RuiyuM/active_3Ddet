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
from apricot import FacilityLocationSelection


class Loss_scaling_Facility_Location_stage_21_greedy_overall_entrpy(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Loss_scaling_Facility_Location_stage_21_greedy_overall_entrpy, self).__init__(model, labelled_loader,
                                                                               unlabelled_loader, rank,
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
        # counting label
        val_dataloader_iter_labelled = iter(self.labelled_loader)
        val_loader_labelled = self.labelled_loader
        total_it_each_epoch_labelled = len(self.labelled_loader)

        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch_labelled, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        self.enable_dropout(self.model)
        num_class = len(self.labelled_loader.dataset.class_names)
        '''
        -------------  Stage 1: Consise Label Sampling ----------------------
        '''
        total_num_car = torch.zeros(1).to('cuda:0')
        total_num_cyclist = torch.zeros(1).to('cuda:0')
        total_num_pedestrain = torch.zeros(1).to('cuda:0')
        for cur_it in range(total_it_each_epoch_labelled):
            try:
                unlabelled_batch = next(val_dataloader_iter_labelled)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader_labelled)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)
                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                    total_num_car += pred_dicts[batch_inx]['num_bbox']['Car']
                    total_num_pedestrain += pred_dicts[batch_inx]['num_bbox']['Pedestrian']
                    total_num_cyclist += pred_dicts[batch_inx]['num_bbox']['Cyclist']

        unique_proportions = torch.ones(num_class).cuda()
        value = torch.tensor([0, 1, 2], device='cuda')
        counts = torch.tensor([total_num_car, total_num_pedestrain, total_num_cyclist])
        unique_proportions[value] = counts.cuda()
        xx = Categorical(probs=unique_proportions / sum(counts)).probs
        entropy_selected = -(xx * xx.log2()).sum()
        # next_entropy = -(xx * xx.log()).sum()
        # entropy_selected = Categorical(probs=unique_proportions / sum(counts)).entropy()

        # total_num_car -= 1282
        # total_num_pedestrain -= 759
        # total_num_cyclist -= 480

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
        predicted_label_dict = {}
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
                        entropy = Categorical(probs=unique_proportions / sum(counts)).entropy()
                        check_value.append(entropy)

                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    # save the density records for the Stage 3
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx][
                        'pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

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
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # check_value.sort()
        # log_data = [[idx, value] for idx, value in enumerate(check_value)]
        # table = wandb.Table(data=log_data, columns=['idx', 'selection_value'])
        # wandb.log({'value_dist_epoch_{}'.format(cur_epoch) : wandb.plot.line(table, 'idx', 'selection_value',
        #     title='value_dist_epoch_{}'.format(cur_epoch))})
        #
        # # sort and get selected_frames
        # select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        # # narrow down the scope
        selected_frames = list(label_list.keys())

        selected_id_list, selected_infos = [], []
        unselected_id_list, unselected_infos = [], []

        '''
        -------------  Stage 2: Representative Prototype Selection ----------------------
        '''

        # rebuild a dataloader for K1 samples
        for i in range(len(self.pairs)):
            if self.pairs[i][0] in selected_frames:
                selected_id_list.append(self.pairs[i][0])
                selected_infos.append(self.pairs[i][1])
            else:
                # no need for unselected part
                if len(unselected_id_list) == 0:
                    unselected_id_list.append(self.pairs[i][0])
                    unselected_infos.append(self.pairs[i][1])

        selected_id_list, selected_infos, \
            unselected_id_list, unselected_infos = \
            tuple(selected_id_list), tuple(selected_infos), \
                tuple(unselected_id_list), tuple(unselected_infos)
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        labelled_set, _, \
            grad_loader, _, \
            _, _ = build_active_dataloader(
            self.cfg.DATA_CONFIG,
            self.cfg.CLASS_NAMES,
            1,
            False,
            workers=self.labelled_loader.num_workers,
            logger=None,
            training=True,
            active_training=active_training
        )
        grad_dataloader_iter = iter(grad_loader)
        total_it_each_epoch = len(grad_loader)

        self.model.train()
        fc_grad_embedding_list = []
        index_list = []

        # start looping over the K1 samples
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='inf_grads_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(grad_dataloader_iter)

            except StopIteration:
                unlabelled_dataloader_iter = iter(grad_loader)
                unlabelled_batch = next(grad_dataloader_iter)

            load_data_to_gpu(unlabelled_batch)

            pred_dicts, _, _ = self.model(unlabelled_batch)

            roi_labels = pred_dicts['roi_labels']
            #
            rcnn_cls_preds_per_sample = pred_dicts['rcnn_cls']  # 128， 1
            rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]

            rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg']
            rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]

            cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss(
                {'rcnn_cls': rcnn_cls_preds_per_sample, 'rcnn_cls_labels': rcnn_cls_gt_per_sample})

            reg_loss = self.model.roi_head.get_box_reg_layer_loss(
                {'rcnn_reg': rcnn_reg_preds_per_sample, 'reg_sample_targets': rcnn_reg_gt_per_sample})
            # summed lost
            summed_lost = torch.sum(reg_loss, dim=2)
            label_sums = []
            for label in torch.unique(roi_labels):
                mask = (roi_labels == label)
                label_sum = torch.sum(summed_lost[mask])
                label_sums.append(label_sum)

            label_counts = []
            for label in torch.unique(roi_labels):
                mask = (roi_labels == label)
                count = torch.sum(mask)
                label_counts.append(count)

            result = [x / y for x, y in zip(label_sums, label_counts)]# 3,
            result = torch.tensor(result)
            result = torch.tensor(result)
            label_counts_tensor = torch.tensor(
                [label_count.item() for label_count in label_counts])  # Convert list of tensors to a tensor

            # Calculate weights as the inverse of the label counts
            weights = 1.0 / label_counts_tensor

            # # Normalize the weights
            # weights_normalized = weights / weights.sum()
            max_weight = weights.max()
            # Adjust the loss by multiplying with the normalized weights
            scaling_factor = 1.00 / max_weight
            adjusted_weights = weights * scaling_factor

            adjusted_loss = result * adjusted_weights
            result = torch.mean(adjusted_loss)
            # clean cache
            del rcnn_cls_preds_per_sample, rcnn_cls_gt_per_sample
            del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample
            torch.cuda.empty_cache()

            loss = cls_loss + result
            self.model.zero_grad()
            loss.backward()

            fc_grads = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
            fc_grad_embedding_list.append(fc_grads)
            index_list.append(unlabelled_batch['frame_id'][0])

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # stacking gradients for K1 candiates
        fc_grad_embeddings = torch.stack(fc_grad_embedding_list, 0)
        num_sample = fc_grad_embeddings.shape[0]
        fc_grad_embeddings = fc_grad_embeddings.view(num_sample, -1)
        start_time = time.time()
        fc_grad_embeddings = F.softmax(fc_grad_embeddings, dim=1)
        # choose the prefered prototype selection method and select the K2 with facility location
        all_features = fc_grad_embeddings
        # N, M, K = all_features.shape
        # all_features = all_features.reshape(N, -1)
        # all_features = csr_matrix(all_features)
        # first time euclidian, second time manhattan
        # than cosine
        # 第一次500
        selector = FacilityLocationSelection(n_samples=500, metric='manhattan')

        # Fit the selector to your data
        selector.fit(all_features)

        # Transform the data (optional here since you're more likely interested in the indices of the selected samples)
        selected_indices = selector.ranking
        selected_frame_ids = [index_list[i] for i in selected_indices]
        print("--- {%s} running time: %s seconds for fc grads---" % (self.prototype, time.time() - start_time))

        new_select_dic = {}
        for element in selected_frame_ids:
            new_select_dic[element] = select_dic[element]

        # sort and get selected_frames
        new_select_dic = dict(sorted(new_select_dic.items(), key=lambda item: item[1]))
        # narrow down the scope
        # 第一次是了 300 然后试了200不行，然后试了400
        selected_frames = list(new_select_dic.keys())[::-1][:int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS) * 3]

        self.model.eval()

        ################## greedy algo to select frame which increaes entropy#########
        selected_frames_final = []
        index_of_greedy = 0
        best_entropy = 0
        while len(selected_frames_final) < 100 and index_of_greedy <= 110:
            if_selected = False
            if index_of_greedy == 0:
                current_frame = selected_frames[0]
                total_num_car = total_num_car + predicted_label_dict[current_frame]['Car']
                total_num_pedestrain = total_num_pedestrain + predicted_label_dict[current_frame]['Pedestrian']
                total_num_cyclist = total_num_cyclist + predicted_label_dict[current_frame]['Cyclist']
                unique_proportions = torch.ones(num_class).cuda()
                value = torch.tensor([0, 1, 2], device='cuda')
                counts = torch.tensor([total_num_car, total_num_pedestrain, total_num_cyclist])
                unique_proportions[value] = counts.cuda()
                xx = Categorical(probs=unique_proportions / sum(counts)).probs
                best_entropy = -(xx * xx.log2()).sum()
                selected_frames_final.append(current_frame)
                del selected_frames[0]
            else:
                initial_entropy = best_entropy
                best_frame = None
                for e_index in selected_frames:
                    cur_car = total_num_car + predicted_label_dict[e_index]['Car']
                    cur_ped = total_num_pedestrain + predicted_label_dict[e_index]['Pedestrian']
                    cur_cyclist = total_num_cyclist + predicted_label_dict[e_index]['Cyclist']
                    unique_proportions = torch.ones(num_class).cuda()
                    value = torch.tensor([0, 1, 2], device='cuda')
                    counts = torch.tensor([cur_car, cur_ped, cur_cyclist])
                    unique_proportions[value] = counts.cuda()
                    xx = Categorical(probs=unique_proportions / sum(counts)).probs
                    current_entropy = -(xx * xx.log2()).sum()
                    if current_entropy > initial_entropy:
                        initial_entropy = current_entropy
                        best_frame = e_index
                        if_selected = True
                if best_frame:
                    selected_frames_final.append(best_frame)
                if not if_selected:
                    best_frame = selected_frames[0]
                    selected_frames_final.append(best_frame)
                total_num_car += predicted_label_dict[best_frame]['Car']
                total_num_pedestrain += predicted_label_dict[best_frame]['Pedestrian']
                total_num_cyclist += predicted_label_dict[best_frame]['Cyclist']
                unique_proportions = torch.ones(num_class).cuda()
                value = torch.tensor([0, 1, 2], device='cuda')
                counts = torch.tensor([total_num_car, total_num_pedestrain, total_num_cyclist])
                unique_proportions[value] = counts.cuda()
                xx = Categorical(probs=unique_proportions / sum(counts)).probs
                best_entropy = -(xx * xx.log2()).sum()
                selected_frames.remove(best_frame)
            index_of_greedy += 1
        # returned the index of acquired bounding boxes
        if len(selected_frames_final) < 100:
            # Calculate how many elements to extract from selected_frames
            elements_needed = 100 - len(selected_frames_final)

            # Extract the top elements_needed from selected_frames
            top_elements = selected_frames[:elements_needed]

            # Append extracted elements to selected_frames_final
            selected_frames_final.extend(top_elements)

        return selected_frames_final

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