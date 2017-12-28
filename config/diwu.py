savepath                    = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/downloaded_data/'
fig_save_path               = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/Figure_post_12_05'
fig_save_train_path         = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/train_models_post_12_05'
fig_save_test_path          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Figures/test_models_post_12_05'
wind_save_path              = '/media/samsumg_1tb/Alibaba_tianchi_RL/wind_numpy_12_05_multiprocessing'

TrainRealFile               = 'In_situMeasurementforTraining_201712.csv'
TrainForecastFile           = 'ForecastDataforTraining_201712.csv'
TestForecastFile            = 'ForecastDataforTesting_201712.csv'

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
add_day                     = 1 #[1| 6]
submission_path             = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions'
num_threads                 = 12

# A star search
A_star_search_2D            = False
A_star_search_3D            = False
A_star_search_3D_multiprocessing = False
risky                       = True   # this flag will set the path weight to 1 to let A star choose the most efficient(risky) path
wall_wind                   = 15    # Set this lower will also reduce the risk!
risky_coeff                 = 15.  # This will only take effect is risky is set to False
wind_exp                    = False
wind_exp_mean               = 5
wind_exp_std                = 5
use_real_weather            = False

grid_world_shape            = (548, 421)
time_length                 = 30 * 18  # total number of unit time (2 min is a unit time). We can fly maximum 18 hours which is 18 *30 unit time
model_number                = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
day_list                    = [3]  # [1, 2, 3, 4, 5]  # train [1, 2, 3, 4, 5]  # test [6, 7, 8, 9, 10]
goal_city_list              = [9]  #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
model_description           = 'A_star_search_3D'  #['A_star_search_3D_risky', 'A_star_search_3D_conservative']

real_hour                   = 3
hourly_travel_distance      = 30

colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'white']
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = time_length  # this ensure that the wind hard threshold, we will not trespass the wind wall unless not viable route was found.

# evaluation
debug_draw                  = False
evaluation_days             = [1, 2, 3, 4, 5]  # [1, 2, 3, 4, 5]
evaluation_goal_cities      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evaluation                  = False
csv_for_evaluation          = '/home/stevenwudi/PycharmProjects/alibaba_weather_route/Submissions/Train_A_star_search_3D_conservative_wall_wind_15.csv'

# reinforcement_learning solution
reinforcement_learning_solution = True
return_to_start                 = True
strong_wind_return              = True    # will go back to the previous state
include_all                     = False     # A flag indicating include all other A star heuristics
reward_goal                     = 1.0
reward_move                     = 0.0
reward_obstacle                 = -10.0

# Dyna model hyper
maxSteps                        = 1e5  # Maze maximum steps
random_state                    = 0
planningSteps                   = 5     # planning steps for Dyna model
alpha                           = 0.5      # Learning step size
gamma                           = 0.95
theta                           = 1e-4
epsilon                         = 0.1
qLearning                       = True  # flag for qLearning
expected                        = False  # flag for expected Sarsa
priority                        = True   # flag for prioritized sweeping
plus                            = True   # Dyna Plus algorithm
temp_model                      = 1  # a flag saying the temporary model
optimal_length_relax            = 1.5
heuristic                       = False
increase_epsilon                = 1.5  # for every maxSteps fail to reach the goal, we increase the epilson