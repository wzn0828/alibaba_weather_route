import os
import sys
import pandas as pd
import random
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import numpy as np
from tools.Dyna_3D import Dyna_3D
from tools.Maze3D import Maze_3D
import multiprocessing
from tools.simpleSub import a_star_submission_3d, collect_csv_for_submission


def load_a_star_precompute(cf):
    A_star_model_precompute_csv = []
    if cf.day_list[0] <= 5:
        for m in range(10):
            csv_file = os.path.join(cf.A_star_precompute_path, 'Train_A_star_search_3D_conservative_wall_wind_15_model_number_[%d].csv'%(m+1))
            A_star_model_precompute_csv.append(pd.read_csv(csv_file, names=['target', 'date', 'time', 'xid', 'yid']))
    else:
        for m in range(10):
            csv_file = os.path.join(cf.A_star_precompute_path, 'Test_A_star_search_3D_conservative_wall_wind_15_model_number_[%d].csv'%(m+1))
            A_star_model_precompute_csv.append(pd.read_csv(csv_file, names=['target', 'date', 'time', 'xid', 'yid']))
    return A_star_model_precompute_csv


def process_wind(cf, day):
    wind_real_day_hour_total = np.zeros(shape=(len(cf.model_number), cf.grid_world_shape[0], cf.grid_world_shape[1], int(cf.total_hours)))
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

        # we replicate the weather for the whole hour
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

        wind_real_day_hour_total[:, :, :, hour - 3] = wind_real_day_hour[:, :, :]  # we replicate the hourly data

    return wind_real_day_hour_total


def initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total):
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
                    optimal_length_relax=cf.optimal_length_relax,
                    heuristic=cf.heuristic)
    return model


def extract_a_star(A_star_model_precompute_csv, day, goal_city):
    go_to_all = []
    for csv in A_star_model_precompute_csv:
        go_to = {}
        predicted_df_now = csv.loc[(csv['date'] == day) & (csv['target'] == goal_city)]
        for t in range(len(predicted_df_now)-1):
            go_to[(predicted_df_now.iloc[t]['xid']-1, predicted_df_now.iloc[t]['yid']-1, t)] = \
                (predicted_df_now.iloc[t+1]['xid']-1, predicted_df_now.iloc[t+1]['yid']-1, t)
        go_to_all.append(go_to)

    return go_to_all


def draw_predicted_route(cf, day, go_to_all, city_data_df, route_list):
    plt.close("all")
    # draw figure maximum
    plt.figure(1)
    plt.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    for hour in range(3, 21):
        if day < 6:  # meaning this is a training day
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            title = weather_name[:-4]
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
        ########## Plot all a-star
        for m, go_to in enumerate(go_to_all):
            for p in go_to.keys():
                plt.scatter(p[1], p[0], c='black', s=1)
            # anno_point = int(len(go_to.keys()) * (1+m) / 10)
            anno_point = int(len(go_to.keys()) / 2)
            go_to_sorted = sorted(go_to)
            p = go_to_sorted[anno_point]
            # plt.annotate(str(m+1), xy=(p[1], p[0]), color='indigo', fontsize=20)

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

        for h in range(3, hour + 1):
            for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)
                if h >= hour and wind_real_day_hour[p[0], p[1]] >= cf.wall_wind:
                    title = weather_name[:-4] + '.  Crash at: ' + str(p) + ' in hour: %d' % (
                        p[2] / 30 + 3)
                    print(title)
        for p in route_list:
            plt.scatter(p[1], p[0], c='yellow', s=1)

        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(title)
        plt.waitforbuttonpress(0.01)
        if 30 * (hour - 2) > len(route_list):
            break


def reinforcement_learning_solution(cf):
    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    cf.debug_draw = True
    cf.go_to_all_dir = True
    cf.day_list = [3]
    cf.goal_city_list = [9]
    cf.risky = False
    cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute(cf)
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

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
                wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
                wind_real_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

            # we replicate the weather for the whole hour
            wind_real_day_hour[wind_real_day_hour >= cf.wall_wind] = cf.strong_wind_penalty_coeff
            if cf.risky:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = 1  # Every movement will have a unit cost
            elif cf.wind_exp:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] -= cf.wind_exp_mean  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.wind_exp_std  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] = np.exp(wind_real_day_hour[wind_real_day_hour < cf.wall_wind]).astype(
                    'int')  # with int op. if will greatly enhance the computatinal speed
            else:
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] /= cf.risky_coeff  # Movement will have a cost proportional to the speed of wind. Here we used linear relationship
                wind_real_day_hour[wind_real_day_hour < cf.wall_wind] += 1

            wind_real_day_hour_total[:, :, :, (hour - 3) * 30:(hour - 2) * 30] = wind_real_day_hour[:, :, :, np.newaxis]  # we replicate the hourly data

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

            # We collect trajectories from all wind model
            go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
            steps_a_star_all = [len(x) for x in go_to_all]
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

            print('Play for using A star trajectory with with model wind')
            num_episode = 0
            success_flag = False
            while num_episode < cf.a_star_loop or not success_flag:
                # if num_episode >= cf.a_star_loop:
                #     model.qlearning = False
                #     model.expected = True
                #     model.policy_init = []
                #     model.epsilon = 0.05
                #     model.maze.reward_obstacle = -1
                # if num_episode > 1000:
                #     break
                num_episode += 1
                model.maze.wind_model = random.choice(range(10))
                model.a_star_model = random.choice(range(10))
                print("A star model: %d, wind model: %d" % (model.maze.wind_model, model.a_star_model))
                steps.append(model.play(environ_step=True))
                # print best action w.r.t. current state-action values
                # printActions(currentStateActionValues, maze)
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag = model.checkPath(np.mean(steps_a_star_all))
                if success_flag:
                    print('Success in ep: %d, with %d steps' % (num_episode, len(came_from)))
                else:
                    print('Fail in ep: %d, with %d steps' % (num_episode, len(came_from)))

            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set([currentState]))
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
                        title = weather_name[:-4]
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
                    ########## Plot all a-star
                    for m, go_to in enumerate(go_to_all):
                        for p in go_to.keys():
                            plt.scatter(p[1], p[0], c='black', s=1)
                        #anno_point = int(len(go_to.keys()) * (1+m) / 10)
                        anno_point = int(len(go_to.keys()) /2 )
                        go_to_sorted = sorted(go_to)
                        p = go_to_sorted[anno_point]
                        #plt.annotate(str(m+1), xy=(p[1], p[0]), color='indigo', fontsize=20)

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

                    for h in range(3, hour+1):
                        for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                            plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)
                            if h >= hour and wind_real_day_hour[p[0], p[1]] >= cf.wall_wind:
                                title = weather_name[:-4] + '.  Crash at: ' + str(p) + ' in hour: %d' % (p[2] / 30 + 3)
                                print(title)
                    for p in route_list:
                        plt.scatter(p[1], p[0],  c='yellow', s=1)

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title(title)
                    plt.waitforbuttonpress(0.01)
                    if 30*(hour-2) > len(route_list):
                        break

            print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                  (day, goal_city, len(route_list), timer() - city_start_time))

    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_new(cf):

    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    cf.debug_draw = True
    cf.day_list = [2]
    cf.goal_city_list = [8]
    cf.risky = False
    cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute(cf)

    for day in cf.day_list:
        print("Processing wind data...")
        wind_real_day_hour_total = process_wind(cf, day)

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

            # We collect trajectories from all wind model
            go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
            steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
            # construct the 3d maze
            cf.qlearning = True
            cf.expected = False
            model = initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total)

            start_time = timer()
            # track steps / backups for each episode
            steps = []
            print('Play for using A star trajectory with with model wind')

            total_Q = np.zeros(len(cf.model_number))
            total_reward = np.zeros(len(cf.model_number))
            action_value_function_sum = np.zeros(len(cf.model_number))
            model.epsilon = 0
            model.policy_init = go_to_all
            for m in range(len(cf.model_number)):
                model.maze.wind_model = m
                model.a_star_model = m
                print("A star model: %d, wind model: %d" % (model.maze.wind_model, model.a_star_model))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q[m], total_reward[m], action_value_function_sum[m]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                print('Success in ep: %d, with %d steps. Q: %.2f' % (m, len(came_from), action_value_function_sum[m]))

            success_flag = False
            save_length = int(cf.a_star_loop * cf.optimal_length_relax) +1
            total_Q_new = np.zeros(save_length)
            total_reward_new = np.zeros(save_length)
            action_value_function_sum_new = np.zeros(save_length)
            num_episode = 0
            model.policy_init = []
            alpha_step = (cf.alpha_start - cf.alpha_end) / cf.a_star_loop
            epsilon_step = (cf.epsilon_start - cf.epsilon_end) / cf.a_star_loop
            model.epsilon = cf.epsilon_start
            model.alpha = cf.alpha_start
            model.qlearning = True
            model.expected = False
            while num_episode < cf.a_star_loop or not success_flag:
                if num_episode >= cf.a_star_loop * cf.optimal_length_relax:
                    break

                model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
                print("Episode %d, wind model: %d" % (num_episode, model.maze.wind_model))
                steps.append(model.play(environ_step=True))
                # check whether the (relaxed) optimal path is found
                came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], action_value_function_sum_new[num_episode]\
                    = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
                if success_flag:
                    print('Success in ep: %d, with %d steps. sum(Q): %.2f, R: %.4f' %
                          (num_episode, len(came_from), action_value_function_sum_new[num_episode], total_reward_new[num_episode]))
                else:
                    print('Fail in ep: %d, with %d steps' % (num_episode, len(came_from)))
                num_episode += 1
                model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
                model.alpha = max(cf.alpha_end, model.alpha - alpha_step)  # we don't want our learning rate to be too large

            print('Finish, using %.2f sec!, updating %d steps.' % (timer() - start_time, np.sum(steps)))
            route_list = []
            current_loc = list(set(goal_loc_3D) & set([currentState]))
            find_loc = current_loc[0]
            while came_from[find_loc] is not None:
                prev_loc = came_from[find_loc]
                route_list.append(prev_loc)
                find_loc = prev_loc

            # we reverse the route for plotting and saving
            route_list.reverse()
            if cf.debug_draw:
                draw_predicted_route(cf, day, go_to_all, city_data_df, route_list)

            print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' %
                  (day, goal_city, len(route_list), timer() - city_start_time))

    print('Finish! using %.2f sec!' % (timer() - start_time))


def reinforcement_learning_solution_worker(cf, day, goal_city, A_star_model_precompute_csv):
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_day_hour_total = process_wind(cf, day)

    city_start_time = timer()
    start_loc = (int(city_data_df.iloc[0]['xid']) - 1, int(city_data_df.iloc[0]['yid']) - 1)
    goal_loc = (int(city_data_df.iloc[goal_city]['xid']) - 1, int(city_data_df.iloc[goal_city]['yid']) - 1)
    # start location is the first time
    start_loc_3D = (start_loc[0], start_loc[1], 0)
    goal_loc_3D = [(goal_loc[0], goal_loc[1], t) for t in range(cf.time_length)]

    # We collect trajectories from all wind model
    go_to_all = extract_a_star(A_star_model_precompute_csv, day, goal_city)
    steps_a_star_all_mean = np.array([len(x) for x in go_to_all]).mean()
    # construct the 3d maze, we need Q Learning to initiate the value
    cf.qlearning = True
    cf.expected = False
    model = initialise_maze_and_model(cf, start_loc_3D, goal_loc_3D, wind_real_day_hour_total)
    steps = []
    model.epsilon = 0
    model.policy_init = go_to_all
    # A star model initialisation
    for m in range(len(cf.model_number)):
        model.maze.wind_model = m
        model.a_star_model = m
        steps.append(model.play(environ_step=True))

    success_flag = False
    save_length = int(cf.a_star_loop * cf.optimal_length_relax) + 1
    total_Q_new = np.zeros(save_length)
    total_reward_new = np.zeros(save_length)
    action_value_function_sum_new = np.zeros(save_length)
    num_episode = 0
    model.policy_init = []
    model.alpha = cf.alpha_start
    epsilon_step = (cf.epsilon_start - cf.epsilon_end) / cf.a_star_loop
    alpha_step = (cf.alpha_start - cf.alpha_end) / cf.a_star_loop
    model.epsilon = cf.epsilon_start
    # using Expected sarsa to refine
    model.qlearning = cf.qLearning
    model.expected = cf.expected
    # weather information fusion
    while num_episode <= cf.a_star_loop or not success_flag:
        if num_episode >= cf.a_star_loop * cf.optimal_length_relax:
            break
        # check whether the (relaxed) optimal path is found
        model.maze.wind_model = model.rand.choice(range(len(cf.model_number)))
        steps.append(model.play(environ_step=True))

        came_from, currentState, success_flag, total_Q_new[num_episode], total_reward_new[num_episode], \
        action_value_function_sum_new[num_episode] \
            = model.checkPath(steps_a_star_all_mean, set_wind_to_zeros=False)
        num_episode += 1
        model.epsilon = max(cf.epsilon_end, model.epsilon - epsilon_step)
        model.alpha = max(cf.alpha_end, model.alpha - alpha_step)  # we don't want our learning rate to be too large

    # check whether the (relaxed) optimal path is found
    route_list = []
    current_loc = list(set(goal_loc_3D) & set([currentState]))
    if len(current_loc):
        find_loc = current_loc[0]
        while came_from[find_loc] is not None:
            prev_loc = came_from[find_loc]
            route_list.append(prev_loc)
            find_loc = prev_loc
        # we reverse the route for plotting and saving
        route_list.reverse()
    else:
        # if we didn't find a route, we choose the A* shortest path
        a_star_shortest_index = np.argmin(np.array([len(x) for x in go_to_all]))
        go_to_A_star_min = go_to_all[a_star_shortest_index]
        route_list.append(start_loc_3D)
        current_loc = start_loc_3D
        while current_loc in go_to_A_star_min.keys():
            next_loc = go_to_A_star_min[current_loc]
            current_loc = (next_loc[0], next_loc[1], next_loc[2]+1)
            route_list.append(current_loc)
        route_list.pop()  # for a_star_submission_3d, we don't attach goal

    sub_df = a_star_submission_3d(day, goal_city, goal_loc, route_list)
    csv_file_name = cf.csv_file_name[:-4] + '_day: %d, city: %d' % (day, goal_city) + '.csv'
    sub_df.to_csv(csv_file_name, header=False, index=False, columns=['target', 'date', 'time', 'xid', 'yid'])
    print('We reach the goal for day: %d, city: %d with: %d steps, using %.2f sec!' % (day, goal_city, len(route_list), timer() - city_start_time))
    sys.stdout.flush()
    return


def reinforcement_learning_solution_multiprocessing(cf):
    """
    This is a RL algorithm:
    The whole diagram is expanded with a third dimension T which has length 18*30 = 540
    :param cf:
    :return:
    """
    # we use A -star algorithm for deciding when to stop running the model
    # get the city locations
    start_time = timer()

    # when debugging concurrenty issues, it can be useful to have access to the internals of the objects provided by multiprocessing.
    multiprocessing.log_to_stderr()
    print("Read Precompute A star route...")
    A_star_model_precompute_csv = load_a_star_precompute(cf)

    jobs = []
    for day in cf.day_list:
        for goal_city in cf.goal_city_list:
            p = multiprocessing.Process(target=reinforcement_learning_solution_worker, args=(cf, day, goal_city, A_star_model_precompute_csv))
            jobs.append(p)
            p.start()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()
    # sub_csv is for submission
    collect_csv_for_submission(cf)

    print('Finish writing submission, using %.2f sec!' % (timer() - start_time))


