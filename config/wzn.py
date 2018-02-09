savepath                    = '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/rematch_downloaded_data/'
fig_save_path               = '/home/wzn/PycharmProjects/alibaba_weather_route/Figures/Figure_post_12_05'
fig_wind_save_train_path         = '/home/wzn/PycharmProjects/alibaba_weather_route/Figures/Wind/Train_models_wind_2018_02'
fig_wind_save_test_path          = '/home/wzn/PycharmProjects/alibaba_weather_route/Figures/Wind/Test_models_wind_2018_02'
fig_rainfall_save_train_path         = '/home/wzn/PycharmProjects/alibaba_weather_route/Figures/Rainfall/Train_models_rainfall_2018_02'
fig_rainfall_save_test_path          = '/home/wzn/PycharmProjects/alibaba_weather_route/Figures/Rainfall/Test_models_rainfall_2018_02'

wind_save_path              = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/wind_numpy_2018_02'
rainfall_save_path          = '/media/samsumg_1tb/Alibaba_tianchi_RL/Rematch/rainfall_numpy_2018_02'

TrainRealFile               = 'In_situMeasurementforTraining_201802.csv'
TrainForecastFile           = 'ForecastDataforTraining_201802.csv'
TestForecastFile            = 'ForecastDataforTesting_201802.csv'

########################################################################################################################
plot_real_wind              = False
plot_real_wind_multiprocessing = False
plt_forecast_wind_train     = False
plt_forecast_wind_test      = False
plt_forecast_wind_train_multiprocessing = False
plt_forecast_wind_test_multiprocessing = False
plot_all_wind               = False
plot_train_model            = False
plot_test_model             = False
plot_all_wind_new           = False
plot_all_rainfall           = False
model_unique                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hour_unique                 = (3, 20)
# submission
submission_dummy            = False
add_day                     = 1 #[1| 6]
submission_path             = '/home/wzn/PycharmProjects/alibaba_weather_route/Submissions'
num_threads                 = 25

########################################################################################################################
# A star search
A_star_search_2D            = False
A_star_search_3D            = False
A_star_search_3D_multiprocessing = False
A_star_search_3D_multiprocessing_multicost = False
search_method               = 'a_star_search_3D'  #search methods ['a_star_search_3D','dijkstra']
model_number                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_world_shape            = (548, 421)
total_hours                 = 20-3+1  # 18 hours to travel
hourly_travel_distance      = 30
time_length                 = 30 * 18  # total number of unit time (2 min is a unit time). We can fly maximum 18 hours which is 18 *30 unit time
model_description           = search_method  #['A_star_search_3D_risky', 'A_star_search_3D_conservative']

A_star_fix_missing          = False

# important parameters
day_list                    = [1, 2, 3, 4, 5]  # train [1, 2, 3, 4, 5]  # test [6, 7, 8, 9, 10]
goal_city_list              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
risky                       = False   # this flag will set the path weight to 1 to let A star choose the most efficient(risky) path
wall_wind                   = 15    # Set this lower will also reduce the risk!
wall_rainfall               = 4
risky_coeff                 = 15.  # This will only take effect is risky is set to False
risky_coeff_rainfall        = 4.  # This will only take effect is risky is set to False
wind_exp                    = False
wind_exp_mean               = 5
wind_exp_std                = 5
rainfall_exp_mean           = 1.3
rainfall_exp_std            = 1.3
use_real_weather            = False
conservative                = False
real_hour                   = 3

costs_exponential           = False  #costs
costs_exp_basenumber        = 10**4
costs_sigmoid               = True  # sigmoid Costs
costs_sig_speed_time        = 4
rainfall_costs_sig_speed_time = 1.5
costs_sig_inter_speed       = 14.5
rainfall_costs_sig_inter_speed = 3.8
costs_sig_path              = '/home/wzn/PycharmProjects/alibaba_weather_route/config/costs_sigmoid'

colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'white']
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = 1440  # this ensure that the wind hard threshold, we will not trespass the wind wall unless not viable route was found.
strong_rainfall_penalty_coeff   = 1440

# evaluation
debug_draw                  = False
evaluation_days             = [5]  # [1, 2, 3, 4, 5]
evaluation_goal_cities      = [7]  #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evaluation_start_hour       = [5]
evaluation                  = False
collect_csv_for_submission_fraction = False
csv_for_evaluation          = '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments/Train_a_star_search_3D_costsSigmoid_5_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-06-12-15-51/Train_a_star_search_3D_costsSigmoid_5_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_day: 5, city: 7, start_hour: 5.csv'

# evalutation plot
evaluation_plot             = False  # a flag for visualising predicted route
eval_day                    = [3]
eval_city                   = [9]

########################################################################################################################
# reinforcement_learning solution
reinforcement_learning_solution = False
reinforcement_learning_solution_new = False
reinforcement_learning_solution_multiprocessing = False
a_star_loop                     = 1000
return_to_start                 = False
strong_wind_return              = False     # will go back to the previous state
include_all                     = False     # A flag indicating include all other A star heuristics
reward_goal                     = time_length * 1.0
reward_move                     = 0.0
reward_obstacle                 = time_length * -1.0

# Dyna model hyper
maxSteps                        = time_length  # Maze maximum steps
random_state                    = 1
planningSteps                   = time_length     # planning steps for Dyna model
alpha                           = 1      # Learning step size
gamma                           = 0.99
theta                           = 1e-1
epsilon                         = 0.01
epsilon_start                   = 0.1
epsilon_end                     = 0.001
alpha_start                     = 0.1
alpha_end                       = 0.01

qLearning                       = True  # flag for qLearning
expected                        = False  # flag for expected Sarsa
priority                        = True   # flag for prioritized sweeping
plus                            = True   # Dyna Plus algorithm
optimal_length_relax            = 1.5
heuristic                       = False
increase_epsilon                = 1.5  # for every maxSteps fail to reach the goal, we increase the epilson

########################################################################################################################
## FCN
fully_convolutional_wind_pred   = False
go_to_all_dir                   = None
#go_to_all_dir                   = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments/go_to_all.npy'
a_star_loop                     = time_length
train_model                     = True
valid_model                     = True
collect_train_valid_mse_iou     = False
model_name                      = 'segnet_basic'  # Model to use ['fcn8' | 'lenet' | 'alexNet' | 'vgg16' |  'vgg19' | 'resnet50' | 'InceptionV3']
train_model_path                = '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments/FCN_segnet_basic_____2018-01-01-14-25-58/epoch_470_mMSE:.12.057328mIOU:.0.499915_net.pth'
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


########################################################################################################################
## Weather Prediction
wp_generate_weather_data_multiprocessing = False
wp_wind_indexes                 = 'weather_prediction_data/wind_13_17_index.pickle'
wp_generate_weather_indexes     = False
wp_used_model_number            = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
wp_onelayer_data_path           = 'weather_prediction_data/3x3-sample_label_weather'
wp_twolayer_data_path           = 'weather_prediction_data/5x5-sample_label_weather'

wp_predict_weather              = False
cuda                            = True
loss                            = 'MSELoss'     # ['L1Loss', 'MSELoss', 'SmoothL1Loss']
wp_model_name                   = 'fully_connected_model'
wp_load_trained_model           = False
wp_trained_model_path           = ''
wp_optimizer                    = 'LBFGS'       # ['LBFGS','adam','rmsprop','sgd']
wp_learning_rate                = 0.01          # Training learning rate
wp_momentum                     = 0.9
wp_weight_decay                 = 0.            # Weight decay or L2 parameter norm penalty
wp_lr_decay_epoch               = 10
# wp_fc
wp_fc_input_dim                 = 90            # input dim, [90, 250]
wp_fc_nonlinear                 = 'ReLU'        # ['Tanh', 'ReLU']


########################################################################################################################
## A_star_search_3D_multiprocessing_rainfall_wind
A_star_search_3D_multiprocessing_rainfall_wind = True