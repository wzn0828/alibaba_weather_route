import os
import pandas as pd
from timeit import default_timer as timer

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import *
from tools.Astar import GridWithWeights, a_star_search, draw_grid
from tools.simpleSub import a_star_submission


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


def A_star_2d_hourly_update_route(cf):
    """
    This is a naive 2d A star algorithm:
    we find hourly path by 2d A star according to the current hour wind(real),
    if the goal is not reached under current hour, we update the route using new wind info...
    until we reach the goal (or not)
    :param cf:
    :return:
    """

    # get the city locations
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
    if cf.submission:
        sub_csv = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])

    start_time = timer()
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            start_loc_hour = start_loc
            total_path = []
            total_undoable_path = []
            for hour in range(3, 21):
                if day < 6:  # meaning this is a training day
                    weather_name = 'real_wind_day_%d_hour_%d.np.npy' % (day, hour)
                else:
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.np.npy' % (cf.model_number, day, hour)
                print('Day: %d, city: %d, hour: %d' % (day, goal_city, hour))
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))

                # construct the 2d diagram
                diagram = GridWithWeights(wind_real_day_hour.shape[0], wind_real_day_hour.shape[1])
                # diagram.weights = wind_real_day_hour.copy()
                # we firstly naive set the wall to be cf.wall_wind (default is 15)
                wind_idx = np.argwhere(wind_real_day_hour > 0)
                diagram.weights = {tuple(loc): wind_real_day_hour[loc] * cf.wind_penalty_coeff for loc in wind_idx}

                wall_idx = np.argwhere(wind_real_day_hour > cf.wall_wind)
                diagram.weights = {tuple(loc): cf.strong_wind_penalty_coeff for loc in wall_idx}

                # initiate a star 2d search
                came_from, cost_so_far = a_star_search(diagram, start_loc_hour, goal_loc)
                current_loc = goal_loc
                route_list = []
                if current_loc not in came_from.keys():
                    print('We cannot reach the goal in hour: %d, continue!' % hour)
                    continue

                while came_from[current_loc] is not None:
                    prev_loc = came_from[current_loc]
                    route_list.append(prev_loc)
                    current_loc = prev_loc

                doable_path = route_list[-31:-1]
                undoable_path = route_list[:-31]

                total_path.append(doable_path)
                total_undoable_path.append(undoable_path)
                start_loc_hour = doable_path[0]

                if cf.debug_draw:
                    plt.clf()
                    plt.imshow(wind_real_day_hour, cmap=cf.colormap)
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
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title(weather_name[:-7])
                    # plt.savefig(os.path.join(cf.fig_save_path, '%s.png'%(weather_name[:-7])), dpi=74, bbox_inches='tight')

                    for c, undoable_path in enumerate(total_undoable_path):
                        for r in undoable_path:
                            plt.scatter(r[1], r[0], c=cf.colors[np.mod(c, len(cf.colors))], s=1)

                    for c, doable_path in enumerate(total_path):
                        for r in doable_path:
                            plt.scatter(r[1], r[0], c=cf.colors[np.mod(c, len(cf.colors))], s=10)

                    plt.waitforbuttonpress(1)

                if not len(undoable_path):
                    print('We reach the goal in hour: %d, using %.2f sec!' % (hour, timer() - city_start_time))
                    if cf.submission:
                        sub_df = a_star_submission(day, goal_city, start_loc, goal_loc, total_path)
                        sub_csv = pd.concat([sub_csv, sub_df], axis=0)
                    break

    if cf.submission:
        sub_csv.to_csv(cf.csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
        print('Finish writing submission, using %.2f sec!' % (timer() - start_time))

    return sub_csv


