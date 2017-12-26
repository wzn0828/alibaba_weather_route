import os
import pandas as pd
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import numpy as np

from tools.Dyna_3D import Dyna_3D
from tools.Maze3D import Maze_3D
from tools.simpleSub import a_star_submission_3d
from tools.evaluation import a_star_length
from tools.Astar_3D import a_star_search_3D, GridWithWeights_3D, walk_final_grid_go_to


def reinforcement_learning_solution(cf):
    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    #total_penalty = a_star_length(cf, cf.csv_for_evaluation)
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
        wind_real_day_hour_total = np.zeros(shape=(len(cf.model_number), cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length)))
        for hour in range(3, 21):
            if day < 6:  # meaning this is a training day
                if cf.use_real_weather:
                    weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                else:
                    wind_real_day_hour_temp = []
                    for model_number in cf.model_number:
                        # we average the result
                        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                        wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                        wind_real_day_hour_temp.append(wind_real_day_hour_model)
                    wind_real_day_hour = np.asarray(wind_real_day_hour_temp)
            else:
                wind_real_day_hour_temp = []
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_real_day_hour_temp.append(wind_real_day_hour_model)
                wind_real_day_hour = np.asarray(wind_real_day_hour_temp)

            wind_real_day_hour_total[:, :, :, (hour-3)*30:(hour-2)*30] = wind_real_day_hour[:, :, :, np.newaxis]  # we replicate the hourly data
        # construct the 3d diagram
        diagram = GridWithWeights_3D(cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.time_length), cf.wall_wind)
        diagram.weights = wind_real_day_hour_total.mean(axis=0)

        for goal_city in cf.goal_city_list:
            city_start_time = timer()
            start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
            # initiate a star 3d search
            # start location is the first time
            start_loc_3D = (start_loc[0], start_loc[1], 0)
            # the goal location spans from all the time stamps--> as long as we reach the goal in any time stamp,
            # we say we have reached the goal
            goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]
            came_from_a_star, cost_so_far = a_star_search_3D(diagram, start_loc_3D, goal_loc_3D)
            go_to_all, steps = walk_final_grid_go_to(start_loc_3D, goal_loc_3D, came_from_a_star)

            # construct the 3d maze
            maze = Maze_3D(height=cf.grid_world_shape[0],
                           width=cf.grid_world_shape[1],
                           time_length=int(cf.time_length),
                           start_state=start_loc_3D,
                           goal_states=goal_loc_3D,
                           return_to_start=cf.return_to_start,
                           reward_goal=cf.reward_goal,
                           reward_move=cf.reward_move,
                           reward_obstacle=cf.reward_obstacle,
                           maxSteps=cf.maxSteps,
                           wind_real_day_hour_total=wind_real_day_hour_total,
                           cf=cf)

            model = Dyna_3D(rand=np.random.RandomState(cf.random_state),
                            maze=maze,
                            epsilon=cf.epsilon,
                            gamma=cf.gamma,
                            planningSteps=cf.planningSteps,
                            qLearning=cf.qLearning,
                            expected=cf.expected,
                            alpha=cf.alpha,
                            priority=cf.priority,
                            theta=cf.theta,
                            policy_init=go_to_all,
                            optimal_length_relax=cf.optimal_length_relax,
                            heuristic=cf.heuristic)

            start_time = timer()
            # track steps / backups for each episode
            steps = []
            # play for an episode
            while True:
                steps.append(model.play())
                # print best action w.r.t. current state-action values
                # printActions(currentStateActionValues, maze)
                # check whether the (relaxed) optimal path is found
                came_from = model.checkPath(len(go_to_all.keys()))
                if came_from:
                    break
            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set(came_from.keys()))
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
                        weather_name = 'real_wind_day_%d_hour_%d.np.npy' % (day, hour)
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

