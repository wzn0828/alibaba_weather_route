savepath                    = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Experiments'
dataroot_dir                = '/media/samsumg_1tb/Alibaba_tianchi_RL/downloaded_data/'
fig_save_path               = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Figures'

plot_real_wind              = False
plt_forecast_wind_train     = False
plt_forecast_wind_test      = False
draw_weather                = False

# submission
submission_dummy                  = False
submission_path             = '/home/stevenwudi/PycharmProjects/gitlab-u4188/Submissions'

# A star search
submission                  = True
A_star_search               = True
model_number                = 1
day_list                    = [6, 7, 8, 9, 10]  # test [6, 7, 8, 9, 10] or [1,2,3,4,5]
real_hour                   = 3
goal_city_list              = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
wall_wind                   = 15
hourly_travel_distance      = 30
debug_draw                  = False
colormap                    = 'jet'  #['hot', 'jet']
#colors                     = ['red', 'magenta', 'cyan', 'yellow', 'green', 'blue']
colors                      = ['red', 'white']
wind_penalty_coeff          = 1
strong_wind_penalty_coeff   = 24 * 30