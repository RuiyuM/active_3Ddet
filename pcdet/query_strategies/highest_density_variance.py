
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

class Highest_density(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(Highest_density, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

        # coefficients controls the ratio of selected subset
        self.k1 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K1', 5)
        self.k2 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K2', 3)
        
        # bandwidth for the KDE in the GPDB module
        self.bandwidth = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'BANDWDITH', 5)
        # ablation study for prototype selection
        self.prototype = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'CLUSTERING', 'kmeans++')
        # controls the boundary of the uniform prior distribution
        self.alpha = 0.95

    def select_top_keys_with_replacement(self, check_value_Car, check_value_Pedestrian, check_value_Cyclist):
        # Helper function to get top N keys with replacements for duplicates
        def get_top_keys_with_replacement(check_value_dict, n, existing_keys):
            sorted_keys = sorted(check_value_dict, key=check_value_dict.get, reverse=True)
            top_keys = []
            for key in sorted_keys:
                if len(top_keys) >= n:
                    break
                if key not in existing_keys:
                    top_keys.append(key)
                existing_keys.add(key)  # Add to existing to prevent future duplicates
            return top_keys, existing_keys

        # Start with Car keys since they don't need replacement for duplicates initially
        top_cars, existing_keys = get_top_keys_with_replacement(check_value_Car, 80, set())

        # Get top Pedestrian keys, replacing duplicates with subsequent top keys
        top_pedestrians, existing_keys = get_top_keys_with_replacement(check_value_Pedestrian, 10, existing_keys)

        # Get top Cyclist keys, replacing duplicates with subsequent top keys
        top_cyclists, existing_keys = get_top_keys_with_replacement(check_value_Cyclist, 10, existing_keys)

        # Combine all top keys
        final_keys = top_cars + top_pedestrians + top_cyclists

        return final_keys

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None):


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
        check_value_Car = {}
        check_value_Pedestrian = {}
        check_value_Cyclist = {}
        cls_results = {}
        reg_results = {}
        density_list = {}
        label_list = {}

        #experiment:
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
                all_pred_dicts.append(pred_dicts)
                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    check_value_Car[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['variance_points']['Car']
                    check_value_Pedestrian[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['variance_points']['Pedestrian']
                    check_value_Cyclist[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['variance_points']['Cyclist']


                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting

                    # save the density records for the Stage 3
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()




        # sort and get selected_frames

        # narrow down the scope
        selected_frames = self.select_top_keys_with_replacement(check_value_Car, check_value_Pedestrian, check_value_Cyclist)



        # save to local
        with open(os.path.join(self.active_label_dir,
                               'selected_all_info_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(all_pred_dicts, f)
        print('successfully saved selected_all_info for epoch {} for rank {}'.format(cur_epoch, self.rank))

        # returned the index of acquired bounding boxes
        return selected_frames
