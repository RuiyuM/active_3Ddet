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
from .crb_sampling_stage_2_and_overall_uniformity_stage_1 import stage_2_Uniformity_stage_1
from .pv_rcnn_prediction_confidence_select_as_much_cyclist_as_possible import Uncertainty_Regression_as_much_cyclist
from .dynamic_sampling_represet_least_car import Dynamic_sampling_stage_2_stage_1_with_min_car
from .dynamic_sampling_represet_least_car_inside_balancing import Dynamic_sampling_stage_2_stage_1_inside_balancing
from .dynamic_sampling_represet_least_car_facility_manhaten import Dynamic_sampling_stage_2_stage_1_with_min_car_manhattan
from .facility_location_stage_2_and_stage_1_with_initial_set import Facility_Location_stage_2_stage_1_with_initial_set
from .dynamic_sampling_represet_least_car_consider_initial_selection import Dynamic_sampling_stage_2_stage_1_consider_initial_selection
from .dynamic_sampling_represet_least_car_inside_balancing_with_initial_selection_set import Dynamic_sampling_stage_2_stage_1_inside_balancing_with_initial_select
from .facility_location_stage_2_and_stage_1_greedy_overall_entropy import Facility_Location_stage_21_greedy_overall_entrpy
from .crb_entropy_uncertainty_regression_change_entropy import Uncertainty_Regression_CRBSampling_change_entropy
from .facility_location_stage_2_and_stage_1_changed_entropy import Facility_Location_stage_2_stage_1_changed_entropy
from .facility_location_stage_2_and_stage_1_change_training import Facility_Location_stage_21_change_training
from .facility_location_stage_2_and_stage_1_greedy_each_query_entropy import Facility_Location_stage_21_greedy_each_query_entrpy
from .facility_location_stage_2_and_stage_1_with_GMM import Facility_Location_stage_2_stage_1_with_GMM
from .greedy_overall_entropy_facility_location_stage_2_and_stage_1 import greedy_overall_entrpy_Facility_Location_stage_21
from .greedy_changed_entropy_facility_location_stage_2_and_stage_1 import Greedy_changed_entropy_Facility_Location_stage_2_stage_1
from .greedy_overall_entropy_facility_location_only_stage_1 import greedy_overall_entrpy_Facility_Location_only_stage_1
from .max_coverage_stage_2_and_stage_1_changed_entropy import Max_coverage_stage_2_stage_1_changed_entropy
from .max_coverage_stage_2_and_stage_1_greedy_overall_entropy import Max_coverage_stage_21_greedy_overall_entrpy
from .facility_location_stage_2_changed_entropy_stage_1_ import Facility_Location_stage_2_greedy_changed_entropy_stage_1
from .facility_location_stage_2_greedy_overall_entropy_stage_1 import Facility_Location_stage_2greedy_overall_entrpy_stage1
from .loss_scaling_facility_location_stage_2_and_stage_1_greedy_overall_entropy import Loss_scaling_Facility_Location_stage_21_greedy_overall_entrpy
from .loss_scaleing_facility_location_stage_2_and_stage_1_changed_entropy import LOSS_SCALING_Facility_Location_stage_2_stage_1_changed_entropy
from .loss_scaling_both_cls_and_reg_facility_location_stage_2_and_stage_1_greedy_overall_entropy import Loss_scaling_both_cls_reg
from .facility_location_stage_2_and_stage_1_with_loss_weighting import Loss_Weighting_Facility_Location_stage_2_stage_1
from .loss_inverse_weight_abligation_1_3 import Loss_scaling_Facility_Location_stage_21_abligation_1_3
from .loss_inverse_weight_abligation_1_only import Loss_scaling_Facility_Location_stage_21_abligation_1_only
from .loss_inverse_weight_abligation_3_only import Loss_scaling_Facility_Location_stage_21_abligation_3_only
from .final_method_keep_computational_graph import Final_method_Keep_Graph_for_Gradient_Computation
from .final_method_max_coverage import Final_method_max_coverage_1
from .final_method_max_coverage_test import Final_method_max_coverage_Test
from .final_method_max_coverage_test_without_scaling import Final_method_max_coverage_Test_Without_scaling
from .final_method_max_coverage_test_without_first_stage import Final_method_max_coverage_Test_Without_first_stage
from .final_method_max_coverage_test_without_first_stage_small_big import Final_method_max_coverage_Test_without_first_stage_small_big
from .final_method_max_coverage_test_combine import Final_method_max_coverage_Test_combine
from .final_method_max_coverage_reverse_23 import Final_method_max_coverage_Reverse_23
from .final_method_max_coverage_reverse_23_revised import Final_method_max_coverage_Reverse_23_revised
from .final_method_max_coverage_test_change_scaling import Final_method_max_coverage_Test_change_scaling
from .final_method_max_coverage_test_scaling_entropy import Final_method_max_coverage_Test_scaling_entropy
from .final_method_max_coverage_test_LDAM import Final_method_max_coverage_Test_LDAM
from .final_method_max_coverage_test_LDAM_both import Final_method_max_coverage_Test_LDAM_BOTH
from .final_method_max_coverage_test_LDAM_entorpy_scalling import Final_method_max_coverage_Test_LDAM_Entropy_Scalling
from .final_method_max_coverage_test_LDAM_entorpy_scalling_both import Final_method_max_coverage_Test_LDAM_Entropy_Scalling_BOTH
from .final_method_max_coverage_test_LDAM_with_parameter import Final_method_max_coverage_Test_LDAM_With_Parameter
from .final_method_max_coverage_test_LDAM_with_parameter_both import Final_method_max_coverage_Test_LDAM_With_Parameter_both
from .final_method_max_coverage_test_LDAM_use_first_128_label import Final_method_max_coverage_Test_LDAM_Use_128_Label
from .final_method_max_coverage_test_LDAM_use_both_128_label import Final_method_max_coverage_Test_LDAM_Use_both_128_Label
from .final_method_max_coverage_test_LDAM_waymo import Final_method_max_coverage_Test_LDAM_waymo
from .final_method_max_coverage_test_LDAM_change_greedy import Final_method_max_coverage_Test_LDAM_change_greedy
from .final_method_max_coverage_test_LDAM_ablation_stage_2 import Final_method_max_coverage_Test_LDAM_Ablation_Stage_2
from .final_method_max_coverage_test_LDAM_ablation_stage_1 import Final_method_max_coverage_Test_LDAM_Ablation_Stage_1
from .final_method_max_coverage_test_LDAM_ablation_stage1_no_both_reweighting import Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Both
from .final_method_max_coverage_test_LDAM_ablation_stage1_no_classification_reweighting import Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Classification
from .final_method_max_coverage_test_LDAM_ablation_stage1_no_regression_reweighting import Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Regression
from .final_method_max_coverage_test_LDAM_ablation_only_cls import Final_method_max_coverage_Test_LDAM_ablation_only_cls
from .final_method_max_coverage_test_LDAM_ablation_only_reg import Final_method_max_coverage_Test_LDAM_ablation_only_reg
from .final_method_max_coverage_test_LDAM_ablation_stage2_step_1_only import Final_method_max_coverage_Test_LDAM_ablation_stage2_step1
from .final_method_max_coverage_test_LDAM_ablation_stage2_step_2_only import Final_method_max_coverage_Test_LDAM_ablation_stage2_step_2
from .final_method_max_coverage_test_LDAM_feature_based_function import Final_method_max_coverage_Test_LDAM_feature_based_function

from .final_method_max_coverage_test_LDAM_stage_1_step_1 import Final_method_max_coverage_Test_LDAM_ablation_Stage_1_Step_1
from .final_method_max_coverage_test_LDAM_stage_1_step_2 import Final_method_max_coverage_Test_LDAM_ablation_Stage_1_Step_2


from .final_method_max_coverage_test_LDAM_feature_based_function_4_2_1 import Final_method_max_coverage_Test_LDAM_feature_based_function_4_2_1
from .final_method_max_coverage_test_LDAM_feature_based_function_5_2_1 import Final_method_max_coverage_Test_LDAM_feature_based_function_5_2_1
from .final_method_max_coverage_test_LDAM_feature_based_function_6_3_1 import Final_method_max_coverage_Test_LDAM_feature_based_function_6_3_1
from .final_method_max_coverage_test_LDAM_feature_based_function_6_4_1 import Final_method_max_coverage_Test_LDAM_feature_based_function_6_4_1

from .final_method_max_coverage_test_LDAM_C_entropy_based import Final_method_max_coverage_Test_LDAM_feature_based_function_entropy_based
from .STONE_sampling import STONE_Sampling_active


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
    'Uncertainty_Regression_crb_much_cyclist': Uncertainty_Regression_as_much_cyclist,
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
    "crb_stage_2_stage_1_and_uniformity": CRBSampling_stage_2_stage_1_Uniformity,
    "crb_stage_2_overall_uniformity_stage_1": stage_2_Uniformity_stage_1,
    "Dynamic_sampling_representative_min_car": Dynamic_sampling_stage_2_stage_1_with_min_car,
    "Dynamic_sampling_representative_min_car_inside_balancing": Dynamic_sampling_stage_2_stage_1_inside_balancing,
    "Dynamic_sampling_representative_min_car_manhattan": Dynamic_sampling_stage_2_stage_1_with_min_car_manhattan,
    "facility_selection_crb_stage_2_stage_1_with_initial_set": Facility_Location_stage_2_stage_1_with_initial_set,
    "Dynamic_sampling_representative_min_car_consider_initial_selection": Dynamic_sampling_stage_2_stage_1_consider_initial_selection,
    "Dynamic_sampling_representative_min_car_inside_balancing_with_initial": Dynamic_sampling_stage_2_stage_1_inside_balancing_with_initial_select,
    "facility_selection_crb_stage_21_overall_entropy": Facility_Location_stage_21_greedy_overall_entrpy,
    "Uncertainty_Regression_crb_change_entropy": Uncertainty_Regression_CRBSampling_change_entropy,
    "facility_selection_crb_stage_21_changed_entropy": Facility_Location_stage_2_stage_1_changed_entropy,
    "facility_selection_crb_stage_21_changed_training": Facility_Location_stage_21_change_training,
    "facility_selection_crb_stage_21_each_query_entropy": Facility_Location_stage_21_greedy_each_query_entrpy,
    "facility_selection_crb_stage_21_with_GMM": Facility_Location_stage_2_stage_1_with_GMM,
    "greedy_overall_entropy_facility_21": greedy_overall_entrpy_Facility_Location_stage_21,
    "Greedy_changed_entropy_Facility_stage21": Greedy_changed_entropy_Facility_Location_stage_2_stage_1,
    "greedy_overall_entropy_facility_only_stage_1": greedy_overall_entrpy_Facility_Location_only_stage_1,
    "max_coverage_crb_stage_21_changed_entropy": Max_coverage_stage_2_stage_1_changed_entropy,
    "max_coverage_crb_stage21_overall_entropy": Max_coverage_stage_21_greedy_overall_entrpy,
    "facility_selection_stage2_greedy_changed_entropy_stage1": Facility_Location_stage_2_greedy_changed_entropy_stage_1,
    "facility_selection_stage2_greedy_overall_entropy_stage1": Facility_Location_stage_2greedy_overall_entrpy_stage1,
    "Loss_scale_facility_2_1_overall_entropy": Loss_scaling_Facility_Location_stage_21_greedy_overall_entrpy,
    "Loss_SCALING_facility_21_change_entropy": LOSS_SCALING_Facility_Location_stage_2_stage_1_changed_entropy,
    "LOSS_scaling_both_cls_reg": Loss_scaling_both_cls_reg,
    "Loss_weighting_facility_selection_crb_stage_2_stage_1": Loss_Weighting_Facility_Location_stage_2_stage_1,
    "abligation_study_1_3": Loss_scaling_Facility_Location_stage_21_abligation_1_3,
    "abligation_study_1_only": Loss_scaling_Facility_Location_stage_21_abligation_1_only,
    "abligation_study_3_only": Loss_scaling_Facility_Location_stage_21_abligation_3_only,
    "Final_Method_keep_gradient_computation": Final_method_Keep_Graph_for_Gradient_Computation,
    "Final_Method_Max_Coverage": Final_method_max_coverage_1,
    "Final_Method_Max_Coverage_Test_case": Final_method_max_coverage_Test,
    "Final_Method_Max_Coverage_Test_case_without_scaling": Final_method_max_coverage_Test_Without_scaling,
    "final_method_max_coverage_test_without_first_stage1": Final_method_max_coverage_Test_Without_first_stage,
    "final_method_max_coverage_test_without_first_stage_small_big1": Final_method_max_coverage_Test_without_first_stage_small_big,
    "final_method_max_coverage_combine": Final_method_max_coverage_Test_combine,
    "final_method_max_coverage_reverse23": Final_method_max_coverage_Reverse_23,
    "final_method_max_coverage_reverse23_revised1": Final_method_max_coverage_Reverse_23_revised,
    "final_method_max_modify_loss_scaling1": Final_method_max_coverage_Test_change_scaling,
    "final_method_max_modify_entropy_both": Final_method_max_coverage_Test_scaling_entropy,
    "final_method_waymo": Final_method_max_coverage_Test_LDAM_waymo,
    "final_method_max_Coverage_test_LDAM": Final_method_max_coverage_Test_LDAM,
    "final_method_max_Coverage_test_LDAM_BOTH1": Final_method_max_coverage_Test_LDAM_BOTH,
    "final_method_max_Coverage_test_LDAM_Entropy_scalling1": Final_method_max_coverage_Test_LDAM_Entropy_Scalling,
    "final_method_max_Coverage_test_LDAM_Entropy_scalling1_BOth": Final_method_max_coverage_Test_LDAM_Entropy_Scalling_BOTH,
    "final_method_max_Coverage_test_LDAM_with_parameter1": Final_method_max_coverage_Test_LDAM_With_Parameter,
    "final_method_max_Coverage_test_LDAM_with_parameter1_Both2": Final_method_max_coverage_Test_LDAM_With_Parameter_both,
    "final_method_max_coverage_test_LDAM_first_128_label": Final_method_max_coverage_Test_LDAM_Use_128_Label,
    "final_method_max_coverage_test_LDAM_BOTH_128_label": Final_method_max_coverage_Test_LDAM_Use_both_128_Label,
    "final_method_change_greedy": Final_method_max_coverage_Test_LDAM_change_greedy,
    "ablation_stage_1_only": Final_method_max_coverage_Test_LDAM_Ablation_Stage_1,
    "ablation_stage_2_only": Final_method_max_coverage_Test_LDAM_Ablation_Stage_2,
    "ablation_stage_1_both_no_re_weighting": Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Both,
    "ablation_stage_1_regression_no_re_weighting": Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Regression,
    "ablation_stage_1_classification_no_re_weighting": Final_method_max_coverage_Test_LDAM_Stage_1_No_reweight_Classification,
    "ablation_only_cls_loss": Final_method_max_coverage_Test_LDAM_ablation_only_cls,
    "ablation_only_regression_loss": Final_method_max_coverage_Test_LDAM_ablation_only_reg,
    "ablation_stage2_step_1": Final_method_max_coverage_Test_LDAM_ablation_stage2_step1,
    "ablation_stage2_step_2": Final_method_max_coverage_Test_LDAM_ablation_stage2_step_2,
    "ablation_stage1_feature_based_f": Final_method_max_coverage_Test_LDAM_feature_based_function,
    "ablation_stage1_step_1": Final_method_max_coverage_Test_LDAM_ablation_Stage_1_Step_1,
    "ablation_stage1_step_2": Final_method_max_coverage_Test_LDAM_ablation_Stage_1_Step_2,
    "ablation_4_2_1": Final_method_max_coverage_Test_LDAM_feature_based_function_4_2_1,
    "ablation_5_2_1": Final_method_max_coverage_Test_LDAM_feature_based_function_5_2_1,
    "ablation_6_3_1": Final_method_max_coverage_Test_LDAM_feature_based_function_6_3_1,
    "ablation_6_4_1": Final_method_max_coverage_Test_LDAM_feature_based_function_6_4_1,
    "C_entropy_based": Final_method_max_coverage_Test_LDAM_feature_based_function_entropy_based,
    'STONE': STONE_Sampling_active,



}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)