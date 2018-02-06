import os
import pandas as pd
from timeit import default_timer as timer
import multiprocessing
import logging
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from tools.Astar import GridWithWeights, a_star_search
from tools.simpleSub import a_star_submission, a_star_submission_3d, collect_csv_for_submission
from tools.Astar_3D import a_star_search_3D, GridWithWeights_3D, dijkstra
from tools.evaluation import evaluation, evaluation_plot
from decimal import Decimal



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
                if len(route_list) <= 31:
                    print('We reach the goal in hour: %d, using %.2f sec!' % (hour, timer() - city_start_time))
                    sub_df = a_star_submission(day, goal_city, start_loc, goal_loc, total_path)
                    sub_csv = pd.concat([sub_csv, sub_df], axis=0)
                    break

            if len(route_list) > 31:
                print('Sadly, we never reached the goal')
                print('#' * 20 + '5' * 20 + '#' * 20)
    sub_csv.to_csv(cf.csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def A_star_search_3D(cf):
    """
    This is a 3D A star algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
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

    # sub_csv is for submission
    sub_csv = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])

    start_time = timer()
    for day in cf.day_list:
        wind_real_day_hour_total = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.total_hours)))
        for hour in range(3, 21):
            if cf.use_real_weather:
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    if day < 6:
                        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    else:
                        weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)

                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

            # we replicate the weather for the whole hour
            wind_real_day_hour[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
            if cf.risky:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
            elif cf.wind_exp:
                wind_real_day_hour[
                    wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[
                    wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = np.exp(
                    wind_real_day_hour[wind_real_day_hour < cf.wall_wind]).astype(
                    'int')  # with int op. if will greatly enhance the computatinal speed
            else:
                wind_real_day_hour[
                    wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] += 1

            wind_real_day_hour_total[:, :, hour - 3] = wind_real_day_hour[:, :]  # we replicate the hourly data

        # construct the 3d diagram
        diagram = GridWithWeights_3D(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length), cf.wall_wind, cf.hourly_travel_distance)
        diagram.weights = wind_real_day_hour_total

        for goal_city in cf.goal_city_list:
            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            # initiate a star 3d search
            # start location is the first time
            start_loc_3D = (start_loc[0], start_loc[1], 0)
            # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
            # we say we have reached the goal
            goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length + 1)]
            came_from, cost_so_far, current, wind_costs, rainfall_costs = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)

            route_list = []
            current_loc = list(set(goal_loc_3D) & set(current))  # instead of came_from.keys() by current

            if not len(current_loc):
                print('We cannot reach the goal city: %d, continue!' % goal_city)
                continue

            find_loc = current_loc[0]
            while came_from[find_loc] is not None:
                prev_loc = came_from[find_loc]
                route_list.append(prev_loc)
                find_loc = prev_loc

            # we reverse the route for plotting and saving
            route_list.reverse()
            if cf.debug_draw:
                for hour in range(3, 21):
                    if day < 6:  # meaning this is a training day
                        weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                        wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                    else:
                        wind_real_day_hour_temp = []
                        for model_number in cf.model_number:
                            # we average the result
                            weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                            wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                            wind_real_day_hour_temp.append(wind_real_day_hour_model)
                        wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                        wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
                        weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)

                    plt.clf()
                    plt.imshow(wind_real_day_hour, cmap=cf.colormap)
                    plt.colorbar()
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
                        else:
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
                    # we also draw some contours
                    x = np.arange(0, wind_real_day_hour.shape[1], 1)
                    y = np.arange(0, wind_real_day_hour.shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title(weather_name[:-4])
                    for h in range(3, hour+1):
                        for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                            plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

                    for p in route_list:
                        plt.scatter(p[1], p[0],  c='yellow', s=1)

                    plt.waitforbuttonpress(1)
                    if 30*(hour-2) > len(route_list):
                        break

            print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                  (day, goal_city, len(route_list), timer() - city_start_time))
            sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
            sub_csv = pd.concat([sub_csv, sub_df], axis=0)

    sub_csv.to_csv(cf.csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def A_star_3D_worker(cf, day, goal_city):
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_day_hour_total = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.total_hours)))
    for hour in range(3, 21):
        if cf.use_real_weather:
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
        else:
            wind_real_day_hour_temp = []
            for model_number in cf.model_number:
                # we average the result
                if day < 6:  # meaning this is a training day
                    weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                else:
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)

                wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_real_day_hour_temp.append(wind_real_day_hour_model)
            wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
            wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

        # we replicate the weather for the whole hour
        wind_real_day_hour[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
        if cf.risky:
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
        elif cf.wind_exp:
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = np.exp(wind_real_day_hour[wind_real_day_hour < cf.wall_wind]).astype('int')  # with int op. if will greatly enhance the computatinal speed
        else:
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            wind_real_day_hour[wind_real_day_hour < cf.wall_wind] += 1

        wind_real_day_hour_total[:, :, hour-3] = wind_real_day_hour[:, :]  # we replicate the hourly data

    # construct the 3d diagram
    diagram = GridWithWeights_3D(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length), cf.wall_wind, cf.hourly_travel_distance)
    diagram.weights = wind_real_day_hour_total

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # initiate a star 3d search
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
    # we say we have reached the goal
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]
    if cf.search_method == 'a_star_search_3D':
        came_from, cost_so_far, current, wind_costs, rainfall_costs = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)
    elif cf.search_method == 'dijkstra':
        came_from, cost_so_far, current = dijkstra(diagram, start_loc_3D, goal_loc_3D)

    route_list = []
    current_loc = list(set(goal_loc_3D) & set(current))

    if not len(current_loc):
        print('We cannot reach the goal city: %d, continue!' % goal_city)
        return

    find_loc = current_loc[0]
    while came_from[find_loc] is not None:
        prev_loc = came_from[find_loc]
        route_list.append(prev_loc)
        find_loc = prev_loc

    # we reverse the route for plotting and saving
    route_list.reverse()
    sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('Using model: %s, we reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!'
          % (str(cf.model_number), day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return


def A_star_search_3D_multiprocessing(cf):
    """
    This is a 3D A star algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    start_time = timer()
    jobs = []
    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by
    # multiprocessing.
    #multiprocessing.log_to_stderr(logging.DEBUG)
    multiprocessing.log_to_stderr()
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            p = multiprocessing.Process(target=A_star_3D_worker, args=(cf, day, goal_city))
            jobs.append(p)
            p.start()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()
    # sub_csv is for submission
    collect_csv_for_submission(cf)
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def A_star_3D_worker_multicost(cf, day, goal_city):
    # get the city locations
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_day_hour_total = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length)))
    for hour in range(3, 21):
        if day < 6:  # meaning this is a training day
            if cf.use_real_weather:
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_real_day_hour = np.round(wind_real_day_hour, 2)
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
        else:
            wind_real_day_hour_temp = []
            for model_number in cf.model_number:
                # we average the result
                weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
                wind_real_day_hour_temp.append(wind_real_day_hour_model)
            wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
            wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

        # print(wind_real_day_hour)
        # delta = 0.1
        # min = wind_real_day_hour.min()
        # max = wind_real_day_hour.max()
        # data = min+delta
        # counts = []
        # datas = []
        # while data < max:
        #     counts.append(int(((data-delta <= wind_real_day_hour) & (wind_real_day_hour < data)).sum()))
        #     datas.append(data)
        #     data += delta
        # print(np.asarray(counts).sum())
        # plt.plot(datas, counts)
        # plt.show()

        # --------set cost ---------#
        # we replicate the weather for the whole hour
        costs = wind_real_day_hour.copy()
        if cf.risky or cf.wind_exp or cf.conservative:
            costs[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
        if cf.risky:
            costs[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
        elif cf.wind_exp:
            costs[wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            costs[wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            costs[wind_real_day_hour < cf.wall_wind] = np.exp(costs[wind_real_day_hour < cf.wall_wind]).astype('int')  # with int op. if will greatly enhance the computatinal speed
        elif cf.conservative:
            costs[wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
            costs[wind_real_day_hour < cf.wall_wind] += 1
        elif cf.costs_exponential:
            costs.dtype = 'float64'
            costs[wind_real_day_hour <= 13] = np.float64(cf.costs_exp_basenumber ** (0))
            costs[wind_real_day_hour >= 16] = np.float64(cf.costs_exp_basenumber ** (3))
            costs[np.logical_and(13 < wind_real_day_hour, wind_real_day_hour < 16)] = np.float64(
                cf.costs_exp_basenumber ** (wind_real_day_hour[np.logical_and(13 < wind_real_day_hour, wind_real_day_hour < 16)] - 13))
        elif cf.costs_sigmoid:
            # variant of sigmoid function: y = cost_time*[1/(1+exp(-speed_time*(x-inter_speed)))]
            costs.dtype = 'float64'
            costs = sigmoid(costs, 10, cf.costs_sig_speed_time, cf.costs_sig_inter_speed)



        # print(costs[wind_real_day_hour <= 14])
        # print(np.asarray(costs[np.logical_and(14 < wind_real_day_hour, wind_real_day_hour < 15.5)]))
        # print(costs[wind_real_day_hour >= 15.5])

        wind_real_day_hour_total[:, :, hour - 3] = costs[:, :]  # we replicate the hourly data

    # construct the 3d diagram
    diagram = GridWithWeights_3D(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length), cf.wall_wind, cf.hourly_travel_distance)
    diagram.weights = wind_real_day_hour_total

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # initiate a star 3d search
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
    # we say we have reached the goal
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length + 1)]
    if cf.search_method == 'a_star_search_3D':
        came_from, cost_so_far, current, wind_costs, rainfall_costs = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)
    elif cf.search_method == 'dijkstra':
        came_from, cost_so_far, current = dijkstra(diagram, start_loc_3D, goal_loc_3D)

    route_list = []
    current_loc = list(set(goal_loc_3D) & set(current))

    if not len(current_loc):
        # print('We cannot reach the goal city: %d, continue!' % goal_city)
        return

    find_loc = current_loc[0]
    while came_from[find_loc] is not None:
        prev_loc = came_from[find_loc]
        route_list.append(prev_loc)
        find_loc = prev_loc

    # we reverse the route for plotting and saving
    route_list.reverse()
    if cf.debug_draw:
        for hour in range(3, 21):
            if day < 6:  # meaning this is a training day
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
                weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)

            plt.clf()
            plt.imshow(wind_real_day_hour, cmap=cf.colormap)
            plt.colorbar()
            # we also plot the city location
            for idx in range(city_data_df.index.__len__()):
                x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                cid = int(city_data_df.iloc[idx]['cid'])
                plt.scatter(y_loc, x_loc, c='r', s=40)
                if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
                else:
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
            # we also draw some contours
            x = np.arange(0, wind_real_day_hour.shape[1], 1)
            y = np.arange(0, wind_real_day_hour.shape[0], 1)
            X, Y = np.meshgrid(x, y)
            CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

            plt.clabel(CS, inline=1, fontsize=10)
            plt.title(weather_name[:-4])
            for h in range(3, hour+1):
                for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                    plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

            for p in route_list:
                plt.scatter(p[1], p[0],  c='yellow', s=1)

            plt.waitforbuttonpress(1)
            if 30*(hour-2) > len(route_list):
                break

    sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    # print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' % (day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return


def A_star_3D_worker_rainfall_wind(cf, day, goal_city, start_hour):
    # get the city locations
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_day_hour_total = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], cf.hour_unique[1] - cf.hour_unique[0] + 1))
    rainfall_real_day_hour_total = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], cf.hour_unique[1] - cf.hour_unique[0] + 1))
    # deal weather data
    for hour in range(3, 21):
        # --------extract weather data ---------#
        wind_real_day_hour, rainfall_real_day_hour = extract_weather_data(cf, day, hour)
        # --------set cost ---------#
        wind_cost, rainfall_cost = set_costs(cf, wind_real_day_hour, rainfall_real_day_hour)
        # we replicate the weather for the whole hour
        wind_real_day_hour_total[:, :, hour - 3] = wind_cost[:, :]  # we replicate the hourly data
        rainfall_real_day_hour_total[:, :, hour - 3] = rainfall_cost[:, :]  # we replicate the hourly data

    max_cost = np.maximum(wind_real_day_hour_total, rainfall_real_day_hour_total)

    # construct the 3d diagram
    diagram = GridWithWeights_3D(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length), cf.wall_wind, cf.hourly_travel_distance, wind_real_day_hour_total, rainfall_real_day_hour_total)
    diagram.weights = max_cost

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # initiate a star 3d search
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
    # we say we have reached the goal
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length + 1)]
    if cf.search_method == 'a_star_search_3D':
        came_from, cost_so_far, current, wind_costs, rainfall_costs = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)
    elif cf.search_method == 'dijkstra':
        came_from, cost_so_far, current = dijkstra(diagram, start_loc_3D, goal_loc_3D)

    route_list = []
    current_loc = list(set(goal_loc_3D) & set(current))

    if not len(current_loc):
        # print('We cannot reach the goal city: %d, continue!' % goal_city)
        return

    find_loc = current_loc[0]
    wind_cost_sum = 0
    rainfall_cost_sum = 0
    max_cost_sum = cost_so_far[find_loc]
    while came_from[find_loc] is not None:
        prev_loc = came_from[find_loc]
        wind_cost_sum += wind_costs[find_loc]
        rainfall_cost_sum += rainfall_costs[find_loc]
        route_list.append(prev_loc)
        find_loc = prev_loc
    num_steps = len(route_list)

    # save costs and num_steps


    # we reverse the route for plotting and saving
    route_list.reverse()
    if cf.debug_draw:
        for hour in range(3, 21):
            if day < 6:  # meaning this is a training day
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
                weather_name = 'Test_forecast_wind_model_count_%d_day_%d_hour_%d.npy' % (len(cf.model_number), day, hour)

            plt.clf()
            plt.imshow(wind_real_day_hour, cmap=cf.colormap)
            plt.colorbar()
            # we also plot the city location
            for idx in range(city_data_df.index.__len__()):
                x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                cid = int(city_data_df.iloc[idx]['cid'])
                plt.scatter(y_loc, x_loc, c='r', s=40)
                if wind_real_day_hour[x_loc, y_loc] >= cf.wall_wind:
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='black', fontsize=20)
                else:
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)
            # we also draw some contours
            x = np.arange(0, wind_real_day_hour.shape[1], 1)
            y = np.arange(0, wind_real_day_hour.shape[0], 1)
            X, Y = np.meshgrid(x, y)
            CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

            plt.clabel(CS, inline=1, fontsize=10)
            plt.title(weather_name[:-4])
            for h in range(3, hour+1):
                for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                    plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

            for p in route_list:
                plt.scatter(p[1], p[0],  c='yellow', s=1)

            plt.waitforbuttonpress(1)
            if 30*(hour-2) > len(route_list):
                break

    sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    # print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' % (day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return


def set_costs(cf, wind_real_day_hour, rainfall_real_day_hour):

    # wind cost
    wind_cost = wind_real_day_hour.copy()
    if cf.risky or cf.wind_exp or cf.conservative:
        wind_cost[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
    if cf.risky:
        wind_cost[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
    elif cf.wind_exp:
        wind_cost[wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
        wind_cost[wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
        wind_cost[wind_real_day_hour < cf.wall_wind] = np.exp(wind_cost[wind_real_day_hour < cf.wall_wind])
    elif cf.conservative:
        wind_cost[wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
        wind_cost[wind_real_day_hour < cf.wall_wind] += 1
    elif cf.costs_exponential:
        wind_cost.dtype = 'float64'
        wind_cost[wind_real_day_hour <= 13] = np.float64(cf.costs_exp_basenumber ** (0))
        wind_cost[wind_real_day_hour >= 16] = np.float64(cf.costs_exp_basenumber ** (3))
        wind_cost[np.logical_and(13 < wind_real_day_hour, wind_real_day_hour < 16)] = np.float64(cf.costs_exp_basenumber ** (wind_real_day_hour[np.logical_and(13 < wind_real_day_hour, wind_real_day_hour < 16)] - 13))
    elif cf.costs_sigmoid:
        # variant of sigmoid function: y = cost_time*[1/(1+exp(-speed_time*(x-inter_speed)))]
        wind_cost.dtype = 'float64'
        wind_cost = sigmoid(wind_cost, 100, cf.costs_sig_speed_time, cf.costs_sig_inter_speed)

    # rainfall cost
    rainfall_cost = rainfall_real_day_hour.copy()
    if cf.risky or cf.wind_exp or cf.conservative:
        rainfall_cost[rainfall_real_day_hour >= cf.wall_rainfall] = cf.strong_wind_penalty_coeff
    if cf.risky:
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] = 1  # Every movement will have a unit cost
    elif cf.wind_exp:
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] -= cf.rainfall_exp_mean  # Movement will have a cost proportional to the speed of rainfall. Here we used linear relationship
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] /= cf.rainfall_exp_mean  # Movement will have a cost proportional to the speed of rainfall. Here we used linear relationship
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] = np.exp(rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall])
    elif cf.conservative:
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] /= cf.risky_coeff_rainfall  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
        rainfall_cost[rainfall_real_day_hour < cf.wall_rainfall] += 1
    elif cf.costs_exponential:
        rainfall_cost.dtype = 'float64'
        rainfall_cost[rainfall_real_day_hour <= 3.5] = np.float64(cf.costs_exp_basenumber ** (0))
        rainfall_cost[rainfall_real_day_hour >= 4.5] = np.float64(cf.costs_exp_basenumber ** (3))
        rainfall_cost[np.logical_and(3.5 < rainfall_real_day_hour, rainfall_real_day_hour < 4.5)] = np.float64(cf.costs_exp_basenumber ** (rainfall_real_day_hour[np.logical_and(3.5 < rainfall_real_day_hour, rainfall_real_day_hour < 4.5)] - 3.5))
    elif cf.costs_sigmoid:
        # variant of sigmoid function: y = cost_time*[1/(1+exp(-speed_time*(x-inter_speed)))]
        rainfall_cost.dtype = 'float64'
        rainfall_cost = sigmoid(rainfall_cost*15.0/4, 100, cf.costs_sig_speed_time, cf.costs_sig_inter_speed)

    return wind_cost, rainfall_cost


def extract_weather_data(cf, day, hour):
    if day < 6:  # meaning this is a training day
        if cf.use_real_weather:
            # wind
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
            wind_real_day_hour = np.round(wind_real_day_hour, 2)
            # rainfall
            rainfall_name = 'real_rainfall_day_%d_hour_%d.npy' % (day, hour)
            rainfall_real_day_hour = np.load(os.path.join(cf.rainfall_save_path, rainfall_name))
            rainfall_real_day_hour = np.round(rainfall_real_day_hour, 2)
        else:
            wind_real_day_hour_temp = []
            rainfall_real_day_hour_temp = []
            for model_number in cf.model_number:
                # we average the result
                # wind
                weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
                wind_real_day_hour_temp.append(wind_real_day_hour_model)
                # rainfall
                rainfall_name = 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                rainfall_real_day_hour_model = np.load(os.path.join(cf.rainfall_save_path, rainfall_name))
                rainfall_real_day_hour_model = np.round(rainfall_real_day_hour_model, 2)
                rainfall_real_day_hour_temp.append(rainfall_real_day_hour_model)
            # wind
            wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
            wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
            # rainfall
            rainfall_real_day_hour_temp = np.asarray(rainfall_real_day_hour_temp)
            rainfall_real_day_hour = np.mean(rainfall_real_day_hour_temp, axis=0)
    else:
        wind_real_day_hour_temp = []
        rainfall_real_day_hour_temp = []
        for model_number in cf.model_number:
            # we average the result
            # wind
            weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
            wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
            wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
            wind_real_day_hour_temp.append(wind_real_day_hour_model)
            # rainfall
            rainfall_name = 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
            rainfall_real_day_hour_model = np.load(os.path.join(cf.rainfall_save_path, rainfall_name))
            rainfall_real_day_hour_model = np.round(rainfall_real_day_hour_model, 2)
            rainfall_real_day_hour_temp.append(rainfall_real_day_hour_model)
        # wind
        wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
        wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)
        # rainfall
        rainfall_real_day_hour_temp = np.asarray(rainfall_real_day_hour_temp)
        rainfall_real_day_hour = np.mean(rainfall_real_day_hour_temp, axis=0)

    return wind_real_day_hour, rainfall_real_day_hour


def A_star_search_3D_multiprocessing_multicost(cf):
    """
    This is a 3D A star algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    start_time = timer()
    jobs = []
    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by
    # multiprocessing.
    #multiprocessing.log_to_stderr(logging.DEBUG)

    multiprocessing.log_to_stderr()
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            #
            # if day == 3 and goal_city == 6:
            #     continue
            # if day == 3 and goal_city == 8:
            #     continue

            p = multiprocessing.Process(target=A_star_3D_worker_multicost, args=(cf, day, goal_city))
            jobs.append(p)
            p.start()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()

    # sub_csv is for submission
    collect_csv_for_submission(cf)
    # sub_csv = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])
    # sub_csv.to_csv(cf.csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))

    # print('evaluation')
    # print(cf.csv_file_name)
    # total_penalty = evaluation(cf, cf.csv_file_name)
    # print(int(np.sum(np.sum(total_penalty[0]))))
    # print(total_penalty[0].astype('int'))
    # print(np.sum(total_penalty[0].astype('int') == 1440))
    # print(total_penalty[1].astype('int'))


def A_star_search_3D_multiprocessing_rainfall_wind(cf):
    """
    This is a 3D A star algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    start_time = timer()
    jobs = []
    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by
    # multiprocessing.
    multiprocessing.log_to_stderr()
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))

    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            start_hours, dist_manhattan = extract_start_hours(cf, city_data_df, goal_city)
            for start_hour in start_hours:
                p = multiprocessing.Process(target=A_star_3D_worker_rainfall_wind, args=(cf, day, goal_city, start_hour))
                jobs.append(p)
                p.start()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()

    # sub_csv is for submission
    collect_csv_for_submission(cf)
    # sub_csv = pd.DataFrame(columns=['target', 'date', 'time', 'xid', 'yid'])
    # sub_csv.to_csv(cf.csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))

    # print('evaluation')
    # print(cf.csv_file_name)
    # total_penalty = evaluation(cf, cf.csv_file_name)
    # print(int(np.sum(np.sum(total_penalty[0]))))
    # print(total_penalty[0].astype('int'))
    # print(np.sum(total_penalty[0].astype('int') == 1440))
    # print(total_penalty[1].astype('int'))


def extract_start_hours(cf, city_data_df, goal_city):
    """
    This script is used to extract start hours
    :param cf:
    :param goal_city:
    :return:
    """
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    hours_total = np.array((range(cf.hour_unique[0], cf.hour_unique[1]+1)))

    dist_manhattan = abs(start_loc[0] - goal_loc[0]) + abs(start_loc[1] - goal_loc[1])
    hours_needed = int(np.ceil(dist_manhattan/30)) - 1
    start_hours = hours_total[:-hours_needed]
    return start_hours, dist_manhattan


def A_star_fix_missing(cf):
    """
    This is a 3D A star algorithm for fixing missing model, day, city because of the memory issue
    :param cf:
    :return:
    """
    start_time = timer()
    jobs_all = []
    jobs = []
    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by
    multiprocessing.log_to_stderr()
    if cf.day_list[-1] < 6:
        name_len = 62
    else:
        name_len = 61

    for model_number in range(10):
        cf.model_number = [model_number+1]
        name_prefix = cf.exp_dir.split('/')[-1][:name_len] + '['+str(model_number+1) +']'
        dir_name = [x for x in os.listdir(cf.savepath) if len(x) >= len(name_prefix) and x[:len(name_prefix)] == name_prefix]
        for day in cf.day_list:
            for goal_city in cf.goal_city_list:
                csv_file_name = os.path.join(cf.savepath, dir_name[0], name_prefix+'_day: %d, city: %d' % (day, goal_city) + '.csv')
                cf.exp_dir = os.path.join(cf.savepath, dir_name[0])
                cf.csv_file_name = os.path.join(cf.exp_dir, name_prefix + '.csv')
                if not os.path.isfile(csv_file_name):
                    p = multiprocessing.Process(target=A_star_3D_worker, args=(cf, day, goal_city))
                    jobs.append(p)
                    print("starting %s" % csv_file_name.split('/')[-1])
                    p.start()
        jobs_all.append(jobs)

    for model_number in range(10):
        cf.model_number = [model_number+1]
        name_prefix = cf.exp_dir.split('/')[-1][:name_len] + '['+str(model_number+1) +']'
        dir_name = [x for x in os.listdir(cf.savepath) if len(x) >= len(name_prefix) and x[:len(name_prefix)] == name_prefix]
        cf.exp_dir = os.path.join(cf.savepath, dir_name[0])
        cf.csv_file_name = os.path.join(cf.exp_dir, name_prefix + '.csv')
        for j in jobs_all[model_number-1]:
            j.join()
        # for submission
        collect_csv_for_submission(cf)
    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))

def sigmoid(speeds, cost_time, speed_time, inter_speed):
    # variant of sigmoid function: y = cost_time*[1/(1+exp(-speed_time*(x-inter_speed)))]
    return cost_time * (1 / (1 + np.exp(-speed_time * (speeds - inter_speed)))) + 1
