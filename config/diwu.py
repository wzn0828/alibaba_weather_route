savepath                    = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/downloaded_data/'
fig_save_path               = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/Figure_post_12_05'
fig_save_train_path         = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/train_models_post_12_05'
fig_save_test_path          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/test_models_post_12_05'
wind_save_path              = '/media/samsumg_1tb/Alibaba_tianchi_RL/wind_numpy_12_05_multiprocessing_float32'
A_star_precompute_path      = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Precomputed_A_star'
TrainRealFile               = 'In_situMeasurementforTraining_201712.csv'
TrainForecastFile           = 'ForecastDataforTraining_201712.csv'
TestForecastFile            = 'ForecastDataforTesting_201712.csv'

########################################################################################################################
plot_real_wind              = False
plt_forecast_wind_train     = False
plt_forecast_wind_test      = False
plt_forecast_wind_train_multiprocessing = False
plt_forecast_wind_test_multiprocessing = False
plot_all_wind               = False
plot_train_model            = False
plot_test_model             = False
model_unique                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hour_unique                 = (3, 20)
# submission
submission_dummy            = False
add_day                     = 1   #[1| 6]
submission_path             = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions'
num_threads                 = 3

########################################################################################################################
# A star search
search_method               = 'a_star_search_3D'
A_star_search_2D            = False
A_star_search_3D            = False
A_star_search_3D_multiprocessing = False
A_star_search_3D_multiprocessing_multicost = False
model_number                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_world_shape            = (548, 421)
hourly_travel_distance      = 30
total_hours                 = 20-3+1  # 18 hours to travel
time_length                 = hourly_travel_distance * total_hours  # total number of unit time (2 min is a unit time). We can fly maximum 18 hours which is 18 *30 unit time
model_description           = 'A_star_search_3D'  #['A_star_search_3D_risky', 'A_star_search_3D_conservative']
A_star_fix_missing          = False

risky                       = False   # this flag will set the path weight to 1 to let A star choose the most efficient(risky) path
wall_wind                   = 15    # Set this lower will also reduce the risk!
risky_coeff                 = 15.  # This will only take effect is risky is set to False
wind_exp                    = False
wind_exp_mean               = 10
wind_exp_std                = 5
low_wind_pass               = 10
conservative                = False
costs_linear                = False
costs_exponential           = False  #costs
costs_sigmoid               = False  # sigmoid Costs
costs_exponential_upper     = 16
costs_exponential_lower     = 13
costs_exp_basenumber        = 100
cost_sigmoid                = True
#  c1 * (1 / (1 + np.exp(-c2 * (wind_speed - c3)))) + c4
c1                          = -10   #-1
c2                          = 1
c3                          = 15
c_baseline_a_star           = 0
# c_baseline_start            = c1 * (-1) /2
c_baseline_start            = 0
c_baseline_end              = 0


use_real_weather            = False
real_hour                   = 3

colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'black', 'yellow', 'magenta']
markers                     = [">", (5, 0), (5, 1), '+', (5, 2)]
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = time_length  # this ensure that the wind hard threshold, we will not trespass the wind wall unless not viable route was found.

########################################################################################################################
# evaluation
debug_draw                  = False
evaluation_plot             = False  # a flag for visualising predicted route
evaluation_days             = [6, 7, 8, 9, 10]  # [1, 2, 3, 4, 5]
evaluation_goal_cities      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evaluation                  = False
collect_csv_for_submission_fraction = False
csv_for_evaluation          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/Precomputed_A_star/Train_A_star_search_3D_conservative_wall_wind_15_model_number_[10].csv'


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
########################################################################################################################
# reinforcement_learning solution
# important parameters
day_list                    = [1, 2, 3]  # train [1, 2, 3, 4, 5]  # test [6, 7, 8, 9, 10]
goal_city_list              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
reinforcement_learning_solution = False
reinforcement_learning_solution_new = False
reinforcement_learning_solution_multiprocessing = True
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
polynomial_alpha                = True
polynomial_alpha_coefficient    = 0.8

qLearning                       = False  # flag for qLearning
double                          = True  # flag for double qLearning
expected                        = True  # flag for expected Sarsa
priority                        = True   # flag for prioritized sweeping
plus                            = False   # Dyna Plus algorithm
optimal_length_relax            = 540
heuristic                       = False
increase_epsilon                = 1.5  # for every maxSteps fail to reach the goal, we increase the epilson

################################f########################################################################################
## FCN
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

