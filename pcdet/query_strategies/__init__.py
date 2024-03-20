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
from .pv_rcnn_use_prediction_only_select_leaset_car_samples import Prediction_leaset_car_samples
from .pv_rcnn_prediction_set_unifrom_first_point_cloud_uniform import Set_Unifrom_invidual_uniform
from .pv_rcnn_prediction_set_unifrom_first_point_cloud_uniform_reverse import Set_Unifrom_invidual_uniform_reverse
from .pv_rcnn_prediction_set_unifrom_first_point_cloud_uniform_with_confidence_score import Set_Unifrom_invidual_uniform_with_confidence
from .pv_rcnn_prediction_set_unifrom_first_point_cloud_uniform_with_confi_half_score import Set_Unifrom_invidual_uniform_with_half_confidence
from .uniform_test import Uniform_test
from .uniform_test_400 import Uniform_test_400
from .facility_700_unifrom_300 import Facility_location_700_uniform_300
from .facility_only import Facility_location_only
from .crb_entropy_uncertainty_regression_with_facility_location import Uncertainty_Regression_CRBSampling_with_facility
from .crb_sampling_stage_2_and_stage_1 import CRBSampling_stage_2_stage_1
from .crb_entropy_uncertainty_regression_with_max_coverage_selection import Uncertainty_Regression_CRBSampling_with_Max_coverage
from .facility_location_stage_2_and_stage_1 import Facility_Location_stage_2_stage_1
from .max_coverage_stage_2_and_stage_1 import Max_Coverage_stage_2_stage_1
from .crb_sampling_stage_2_and_stage_1_and_overall_uniformity import CRBSampling_stage_2_stage_1_Uniformity


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
    'prediction_leaset_car_samples': Prediction_leaset_car_samples,
    'set_unifrom_individual_uniform': Set_Unifrom_invidual_uniform,
    'Set_Unifrom_invidual_uniform_reverse': Set_Unifrom_invidual_uniform_reverse,
    'Set_Unifrom_invidual_uniform_with_confidence': Set_Unifrom_invidual_uniform_with_confidence,
    'Set_Unifrom_invidual_uniform_with_half_confidence': Set_Unifrom_invidual_uniform_with_half_confidence,
    'uniform_test': Uniform_test,
    'uniform_test_400': Uniform_test_400,
    'facility_location_700_unifrom_300': Facility_location_700_uniform_300,
    'facility_only': Facility_location_only,
    'Uncertainty_Regression_crb_with_facility_location_first': Uncertainty_Regression_CRBSampling_with_facility,
    'crb_stage_2_stage_1': CRBSampling_stage_2_stage_1,
    'Uncertainty_Regression_crb_with_max_coverage': Uncertainty_Regression_CRBSampling_with_Max_coverage,
    'facility_selection_crb_stage_2_stage_1': Facility_Location_stage_2_stage_1,
    'max_coverage_crb_stage_2_stage_1': Max_Coverage_stage_2_stage_1,
    "crb_stage_2_stage_1_and_uniformity": CRBSampling_stage_2_stage_1_Uniformity
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)