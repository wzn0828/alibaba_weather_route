savepath                    = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/downloaded_data/'
fig_save_path               = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Figures'
wind_save_path              = '/media/samsumg_1tb/Alibaba_tianchi_RL/wind_numpy'

plot_real_wind              = False
plt_forecast_wind_train     = False
plt_forecast_wind_test      = False
draw_weather                = False

# submission
submission_dummy            = False
add_day                     = 1 #[1| 6]
submission_path             = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Submissions'

# A star search
submission                  = True
A_star_search_2D            = False
A_star_search_3D            = False
risky                       = False   # this flag will set the path weight to 1 to let A star choose the most efficient(risky) path
wall_wind                   = 15    # Set this lower will also reduce the risk!
risky_coeff                 = 15.  # This will only take effect is risky is set to False
grid_world_shape            = (548, 421)
time_length                 = 30 * 18
model_number                = 1
day_list                    = [6, 7, 8, 9, 10]  # train [6, 7, 8, 9, 10]  # test [6, 7, 8, 9, 10]
goal_city_list              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
model_description           = 'A_star_search_3D'  #['A_star_search_3D_risky', 'A_star_search_3D_conservative']

#day_list                    = [3]  # test [6, 7, 8, 9, 10] or [1,2,3,4,5]
real_hour                   = 3
hourly_travel_distance      = 30

colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'white']
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = 24 * 30

# evaluation
debug_draw                  = True
evaluation                  = True
csv_for_evaluation          = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Submissions/A_star_search_3D_conservative_wall_wind_15.csv'