from __future__ import absolute_import

from .random_sampling import RandomSampling
from .entropy_sampling import EntropySampling
from .badge_sampling import BadgeSampling
from .coreset_sampling import CoresetSampling
from .llal_sampling import LLALSampling
from .montecarlo_sampling import MonteCarloSampling
from .confidence_sampling import ConfidenceSampling
from .crb_sampling import CRBSampling
from .test_crb import Test_CRBSampling
from .original_crb_only_has_entropy import Entropy_only_CRBSampling
from .crb_entropy_cls_confi import Cls_confi_CRBSampling
from .crb_entropy_reg_confi import Regression_confi_CRBSampling
from .crb_entropy_cls_reg_confi import Cls_Regression_CRBSampling
from .crb_entropy_uncertainty_regression import Uncertainty_Regression_CRBSampling
from .crb_entropy_uncertainty_only import Uncertainty_only_CRBSampling
from .best_entropy_possible import Entropy_only_Best
from .highest_density_variance import Highest_density
from .greedy_label_and_original_density import Double_greedy
from .greedy_label_and_not_mean_density import Double_greedy_not_mean
from .greedy_label_and_not_mean_density_2 import Double_greedy_not_mean_2
from .greedy_label_and_not_mean_density_5_1_1 import Double_greedy_5_1_1
from .select_all_cyclist_pedestrian import Select_all_cyclist_pedestrian
from .pv_rcnn_prediction_confidence_select_as_less_car_as_possible import Uncertainty_Regression_as_less_car

__factory = {
    'random': RandomSampling,
    'entropy': EntropySampling,
    'badge': BadgeSampling,
    'coreset': CoresetSampling,
    'llal': LLALSampling,
    'montecarlo': MonteCarloSampling,
    'confidence': ConfidenceSampling,
    'crb': CRBSampling,
    'test_crb': Test_CRBSampling,
    'entropy_only_crb': Entropy_only_CRBSampling,
    'Cls_entropy_crb': Cls_confi_CRBSampling,
    'Regression_entropy_crb': Regression_confi_CRBSampling,
    'Cls_Regression_entropy_crb': Cls_Regression_CRBSampling,
    'Uncertainty_Regression_crb': Uncertainty_Regression_CRBSampling,
    'Uncertainty_only_crb': Uncertainty_only_CRBSampling,
    'best_entropy': Entropy_only_Best,
    'highest_density': Highest_density,
    'Double_greedy': Double_greedy,
    'Double_greedy_no_mean': Double_greedy_not_mean,
    'double_greedy_no_mean_2': Double_greedy_not_mean_2,
    'double_greedy_5_1_1': Double_greedy_5_1_1,
    'select_all_cyclist_pedestrian': Select_all_cyclist_pedestrian,
    'Uncertainty_Regression_crb_less_car': Uncertainty_Regression_as_less_car,
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)