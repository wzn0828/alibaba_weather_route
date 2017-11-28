import os
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tools.Astar import GridWithWeights, a_star_search, draw_grid


def draw_path(wind_real_day_hour, city_data_df, weather_name, came_from, start_loc, goal_loc):
    # some plot here
    plt.figure(1)
    plt.clf()
    plt.imshow(wind_real_day_hour, cmap='jet')
    plt.colorbar()
    # we also draw some contours
    x = np.arange(0, wind_real_day_hour.shape[1], 1)
    y = np.arange(0, wind_real_day_hour.shape[0], 1)
    X, Y = np.meshgrid(x, y)
    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(weather_name[:-7])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    # we also plot the city location
    #for idx in range(city_data_df.index.__len__()):
    for idx in range(2):
        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
        cid = int(city_data_df.iloc[idx]['cid'])
        plt.scatter(y_loc, x_loc, c='r', s=40)
        plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

    id = goal_loc
    while came_from.get(id, None):
        (x2, y2) = came_from.get(id, None)
        plt.scatter(y2, x2, c='r', s=10)
        id = (x2, y2)


def A_star_serach(cf):
    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    weather_list = sorted(os.listdir(cf.savepath), reverse=True)
    weather_name = 'real_wind_day_%d_hour_%d.np.npy' % (cf.real_day, cf.real_hour)
    wind_real_day_hour = np.load(os.path.join(cf.savepath, weather_name))

    # A star algorithm here:
    diagram = GridWithWeights(wind_real_day_hour.shape[0], wind_real_day_hour.shape[1])
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[1]['xid']) - 1, int(city_data_df.iloc[1]['yid']) - 1)
    came_from, cost_so_far = a_star_search(diagram, start_loc, goal_loc)
    draw_grid(diagram, width=3, point_to=came_from, start=start_loc, goal=goal_loc)


    for weather_name in weather_list:
        wind_real_day_hour = np.load(os.path.join(cf.savepath, weather_name))
        # plot figures here
        #plt.close()
        plt.figure(1)
        plt.clf()
        plt.imshow(wind_real_day_hour, cmap='jet')
        plt.colorbar()
        # we also plot the city location
        for idx in range(city_data_df.index.__len__()):
            x_loc = int(city_data_df.iloc[idx]['xid']) - 1
            y_loc = int(city_data_df.iloc[idx]['yid']) - 1
            cid = int(city_data_df.iloc[idx]['cid'])
            plt.scatter(y_loc, x_loc, c='r', s=40)
            plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

        # we also draw some contours
        x = np.arange(0, wind_real_day_hour.shape[1], 1)
        y = np.arange(0, wind_real_day_hour.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        CS = plt.contour(X, Y, wind_real_day_hour,  (15,), colors='k')

        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(weather_name[:-7])

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.waitforbuttonpress(0.0001)
        plt.savefig(os.path.join(cf.fig_save_path, '%s.png'%(weather_name[:-7])), dpi=74, bbox_inches='tight')

