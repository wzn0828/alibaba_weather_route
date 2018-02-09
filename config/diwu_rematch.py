savepath                    = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/rematch_downloaded_data/'

wind_save_path              = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/wind_numpy_2018_02'
rainfall_save_path          = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/rainfall_numpy_2018_02'

fig_wind_train_path         = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/train_wind'
fig_wind_test_path          = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/test_wind'
fig_rainfall_train_path     = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/train_rainfall'
fig_rainfall_test_path      = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/test_rainfall'
fig_wind_with_rainfall_train_path = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/train_wind_with_rainfall'
fig_wind_with_rainfall_test_path = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/Figures/test_wind_with_rainfall'


TrainRealFile               = 'In_situMeasurementforTraining_201802.csv'
TrainForecastFile           = 'ForecastDataforTraining_201802.csv'
TestForecastFile            = 'ForecastDataforTesting_201802.csv'

########################################################################################################################
plot_real_wind              = False
plt_forecast_wind_train     = False
plt_forecast_wind_test      = False
plot_real_wind_multiprocessing          = False
plt_forecast_wind_train_multiprocessing = False
plt_forecast_wind_test_multiprocessing = False
plot_all_wind               = False
plot_all_wind_new           = False
plot_all_rainfall           = False
plot_wind_with_rainfall     = False
plot_train_model            = False
plot_test_model             = False

model_unique                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hour_unique                 = (3, 20)
# submission
submission_dummy            = False
add_day                     = 1   #[1| 6]
submission_path             = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions'
num_threads                 = 30

########################################################################################################################
# A star search
search_method               = 'a_star_search_3D'
A_star_search_2D            = False
A_star_search_3D            = False
A_star_search_3D_multiprocessing = False
A_star_search_3D_multiprocessing_multicost = False
A_star_search_3D_multiprocessing_rainfall_wind = False
model_number                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_world_shape            = (548, 421)
hourly_travel_distance      = 30
total_hours                 = 20-3+1  # 18 hours to travel
time_length                 = hourly_travel_distance * total_hours  # total number of unit time (2 min is a unit time). We can fly maximum 18 hours which is 18 *30 unit time
model_description           = 'A_star_search_3D'  #['A_star_search_3D_risky', 'A_star_search_3D_conservative']
A_star_fix_missing          = False

risky                       = False   # this flag will set the path weight to 1 to let A star choose the most efficient(risky) path
wall_wind                   = 15    # Set this lower will also reduce the risk!
wall_rainbfall              = 4
risky_coeff                 = 15.  # This will only take effect is risky is set to False
wind_exp                    = False
wind_exp_mean               = 10
wind_exp_std                = 5
low_wind_pass               = 10
conservative                = False
costs_linear                = False
costs_exponential           = False  #costs
costs_sigmoid               = True  # sigmoid Costs
costs_exponential_upper     = 16
costs_exponential_lower     = 13
costs_exp_basenumber        = 100
cost_sigmoid                = True
#  c1 * (1 / (1 + np.exp(-c2 * (wind_speed - c3)))) + c4
c1                          = 10   #-1
c2                          = 1
c3                          = 14.5
c_baseline_a_star           = 0
# c_baseline_start            = c1 * (-1) /2
c_baseline_start            = 0
c_baseline_end              = 0

# wzn nomenclature
inter_speed                 = 15
costs_sig_speed_time        = 5

use_real_weather            = False
real_hour                   = 3

colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'black', 'yellow', 'magenta']
markers                     = [">", (5, 0), (5, 1), '+', (5, 2)]
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = time_length  # this ensure that the wind hard threshold, we will not trespass the wind wall unless not viable route was found.


# visualisation
evaluation_plot_multi        = False
# csvs_for_evaluation          = ['/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Precomputed_A_star/Train_A_star_search_3D_conservative_wall_wind_15_model_number_[10].csv',
#                                '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Train_reinforcement_learning_solution_multiprocessing_ExpectedSarsa_Double_____2018-01-31-22-16-40/Train_reinforcement_learning_solution_multiprocessing_ExpectedSarsa_Double.csv',
#                                '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Train_reinforcement_learning_solution_multiprocessing_qLearning_Double_____2018-01-31-22-13-08/Train_reinforcement_learning_solution_multiprocessing_qLearning_Double.csv',
#                                 '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Precomputed_A_star/Train_costsSigmoid_speedTime_5_interSpeed_14.5_model_mean_[1-10].csv']
# csv_names = ['Model 10', 'Double Expected Sarsa', 'Double QLearning', 'WZN best Sigmoid Cost']

csvs_for_evaluation          = ['/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Precomputed_A_star/Test_A_star_search_3D_conservative_wall_wind_15_model_number_[3].csv',
                               '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments/Test_a_star_search_3D_costsSigmoid_costTime10_speedTime5_interSpeed_15_model_number_[3]_____2018-01-31-11-59-29/Test_a_star_search_3D_costsSigmoid_model_number_[3].csv',
                                '/home/wzn/PycharmProjects/alibaba_weather_route/Submissions/Test_a_star_search_3D_costsSigmoid_speedTime5_interspeed14.5_model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].csv']
csv_names = ['Model 3', 'sig: mean c1: 10, c2: 5, c3: 15', 'sig: mean c1: 10000, c2: 5, c3: 14.5']
collect_csv_for_submission_fraction = False
evaluation                  = False

########################################################################################################################
# reinforcement_learning solution
# important parameters
# assignment algorithm.
day_list                    = [1, 2, 3, 4, 5]  # train [1, 2, 3, 4, 5]  # test [6, 7, 8, 9, 10]
goal_city_list              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
reinforcement_learning_solution = False
reinforcement_learning_solution_new = False
reinforcement_learning_solution_multiprocessing = False
reinforcement_learning_solution_wind_and_rainfall = False
reinforcement_learning_solution_multiprocessing_wind_and_rainfall = True
# For train
A_star_precompute_path          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions/Precomputed_A_star/train_diwu_assignment_dict_allA_star_search_3D_____2018-02-08-01-57-18'
A_star_precompute_mean_path     = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions/Precomputed_A_star/train_diwu_assignment_dict_allA_star_search_3D_____2018-02-08-01-57-18/Train_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-07-01-19-40'
# For test
#A_star_precompute_path          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions/Precomputed_A_star/assignment_dict_all_2018_02_07_diwuA_star_search_3D_____2018-02-07-08-53-11'
# A_star_precompute_mean_path     ='/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions/Precomputed_A_star/assignment_dict_all_2018_02_07_diwuA_star_search_3D_____2018-02-07-08-53-11/Test_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-06-09-56-56'
A_star_csv_patterns             = 'Test_costsSigmoid_speedTime_4_interSpeed_14.5_model_number_*_day: %d, city: %d, start_hour: *, start_min: *.csv'

loop_switch_to_linear_cost      = 50
return_to_start                 = False
strong_wind_return              = False     # will go back to the previous state
include_all                     = False     # A flag indicating include all other A star heuristics
reward_goal                     = time_length * 1.0
reward_move                     = 0.0

# Dyna model hyper-parameters
maxSteps                        = time_length  # Maze maximum steps
random_state                    = 1
planningSteps                   = time_length    # planning steps for Dyna model
alpha                           = 1.0      # Learning step size
gamma                           = 0.999
theta                           = 1e-3
epsilon                         = 0.01
# the following are the parameters for second round update
epsilon_start                   = 0.1
epsilon_end                     = 0.01
alpha_start                     = 0.5
alpha_end                       = 0.01
polynomial_alpha                = False
polynomial_alpha_coefficient    = 0.8

qLearning                       = False  # flag for qLearning
double                          = False  # flag for double qLearning
expected                        = True  # flag for expected Sarsa
priority                        = True   # flag for prioritized sweeping
plus                            = False   # Dyna Plus algorithm
optimal_length_relax            = 1.5
heuristic                       = False
increase_epsilon                = 1.5  # for every maxSteps fail to reach the goal, we increase the epilson

################################f########################################################################################
## FCN
wp_predict_weather              = False

fully_convolutional_wind_pred   = False
go_to_all_dir                   = None
#go_to_all_dir                   = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/go_to_all.npy'
train_model                     = True
valid_model                     = True
collect_train_valid_mse_iou     = False
model_name                      = 'segnet_basic'  # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']
train_model_path                = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/FCN_segnet_basic_____2018-01-01-14-25-58/epoch_470_mMSE:.12.057328mIOU:.0.499915_net.pth'
learning_rate                   = 1e-5
n_epochs                        = 1000
valid_epoch                     = 10
optimizer                       = 'sgd'  #' 'adam', 'sgd'
momentum                        = 0.9
weight_decay                    = 1e-4
train_days                      = [1, 2, 3, 4]
valid_days                      = [5]
random_crop                     = 416   # We choose 64 here because 30*2+1=61 is the expansion and 64 is the closest to resnet input requirement to integer
random_crop_valid               = 416
batch_size                      = 18

#####################################################################################


assignment_for_A_star_route     = False
assignment_for_A_star_route_10min = False
#cost_num_steps_dir              = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Train_a_star_search_3D_costsSigmoid_5_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-06-12-15-51'
cost_num_steps_dir           = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Train_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-07-01-19-40'
#file_patterns                   = 'costs_num_steps_day_%d_goalcity_%d_starthour_*.json'
file_patterns                   = 'costs_num_steps_day_%d_goalcity_%d_starthour_*.json'
file_patterns_min                   = 'costs_num_steps_day_%d_goalcity_%d_starthour_*_startmin_*.json'
csv_patterns                    = 'Train_a_star_search_3D_risky_use_real_weather_day: %d, city: %d, start_hour: %d'
csv_patterns                    = 'Train_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_day: %d, city: %d, start_hour: %d, start_min: %d.csv'
#csv_patterns                    = 'Test_a_star_search_3D_costsSigmoid_5_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_day: %d, city: %d, start_hour: %d, start_min: %d'
#csv_patterns  ='Test_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_day: %d, city: %d, start_hour: %d, start_min: %d.csv'
combined_csv_name               = 'A_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].csv'
threshold_manhattan_distance    = 3.0  # if cost is larger than this distance, we will not consider this route first

# assignment mins algorithm.
assignment_for_A_star_route_min = False
#file_patterns_min               = 'costs_num_steps_day_%d_goalcity_%d_starthour_*.json'
#cost_num_steps_dir_min          = '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments/Test_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-06-09-56-56'

########################################################################################################################
# evaluation
debug_draw                  = False
evaluation_plot             = False  # a flag for visualising predicted route
evaluation_days             = [2]  # [1, 2, 3, 4, 5]
evaluation_goal_cities      = [4]  #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evaluation_with_rainfall    = False
csv_for_evaluation          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Train_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-07-01-19-40/Train_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].csv'
evaluation_plot_real_with_mean = False
