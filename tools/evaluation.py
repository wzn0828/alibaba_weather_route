import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def evaluation(cf, csv_for_evaluation):
    """
    This is a script for evaluating predicted route's length
    :param cf:
    :param csv_for_evaluation:
    :return:
    """
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    predicted_df = pd.read_csv(csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid'])
    predicted_df_idx = 0
    total_penalty = np.ones(shape=(5, 10)) * 24 * 60
    crash_time_stamp = np.zeros(shape=(5, 10)).astype(int)
    average_wind = np.zeros(shape=(5, 10))
    max_wind = np.zeros(shape=(5, 10))
    if cf.debug_draw:
        # draw figure maximum
        plt.figure(1)
        plt.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    for day in cf.evaluation_days:
        for goal_city in cf.evaluation_goal_cities:
            # print('Day: %d, city: %d' % (day, goal_city))
            crash_flag = False
            route_list = []
            # For evaluation, we don't need to substract 1 here because we have add 1 one submission
            start_loc = (int(city_data_df.iloc[0]['xid']), int(city_data_df.iloc[0]['yid']))
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']), int(city_data_df.iloc[goal_city]['yid']))

            if predicted_df_idx >= len(predicted_df) - 1:
                # meaining the rest of the goals are not reached (e.g., day 4, and 5)
                break

            start_loc_pred = (predicted_df.iloc[predicted_df_idx]['xid'], predicted_df.iloc[predicted_df_idx]['yid'])
            day_pred = predicted_df.iloc[predicted_df_idx]['date']
            target_pred = predicted_df.iloc[predicted_df_idx]['target']

            if goal_city != predicted_df.iloc[predicted_df_idx]['target']:
                # print('Sadly, we never reached the goal: %d' % goal_city)
                total_penalty[day - 1, goal_city - 1] = 24 * 60
                # print('#' * 20 + '5' * 20 + '#' * 20)
                continue

            assert start_loc == start_loc_pred, "Starting x, y not the same!"
            assert day_pred == day, "Starting day not the same!"
            assert target_pred == goal_city, "Starting city not the same!"
            min = 0
            acc_min = 0
            hour = 3
            weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
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
                plt.title(weather_name[:-7] + str(hour) + '_' + 'Goalcity' + str(goal_city))

            next_loc_pred = start_loc_pred

            while not next_loc_pred == goal_loc:
                route_list.append(next_loc_pred)
                min += 2
                acc_min += 2
                predicted_df_idx += 1
                next_loc_pred = (predicted_df.iloc[predicted_df_idx]['xid'], predicted_df.iloc[predicted_df_idx]['yid'])
                day_pred_next = predicted_df.iloc[predicted_df_idx]['date']
                target_pred_next = predicted_df.iloc[predicted_df_idx]['target']
                assert day_pred_next == day_pred, "Predict day not the same!"
                assert target_pred_next == target_pred, "Predict city not the same!"
                assert np.sum(np.abs(next_loc_pred[0] - start_loc_pred[0]) + np.abs(next_loc_pred[1] - start_loc_pred[1])) <=1, "Unlawful move!"

                if min >= 60:
                    min = 0
                    hour += 1
                    weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                    if cf.debug_draw:
                        for h in range(3, hour):
                            for p in route_list[(h-3)*30:(h-2)*30]:
                                plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)
                        plt.waitforbuttonpress(0.1)
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
                        plt.title(weather_name[:-7] + str(hour) + '_' + 'Goalcity' + str(goal_city))

                # Now we check whether the aircraft crash or not
                if wind_real_day_hour[next_loc_pred[0]-1, next_loc_pred[1]-1] >= 15.:
                    # print('Crash! Day: %d, city: %d, hour: %d, min: %d' % (day, goal_city, hour, min))
                    crash_flag = True
                    total_penalty[day-1, goal_city-1] = 24 * 60
                    crash_time_stamp[day-1, goal_city-1] = hour*30 + min
                    # we break the loop
                    break
                else:
                    start_loc_pred = next_loc_pred
                    average_wind[day-1, goal_city-1] += wind_real_day_hour[next_loc_pred[0]-1, next_loc_pred[1]-1]
                    max_wind[day-1, goal_city-1] = max(max_wind[day-1, goal_city-1], wind_real_day_hour[next_loc_pred[0]-1, next_loc_pred[1]-1])

            # plot the last bit route
            if cf.debug_draw:
                for h in range(3, hour):
                    for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                        plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

                for p in route_list[(hour-3)*30:]:
                    plt.scatter(p[1], p[0], c=cf.colors[np.mod(hour, 2)], s=10)
                plt.waitforbuttonpress(0.5)

            if predicted_df_idx < len(predicted_df):
                if not crash_flag:
                    if next_loc_pred == goal_loc:
                        total_penalty[day-1, goal_city-1] = acc_min
                        # print('Goal reached in %d mins' % acc_min)
                        predicted_df_idx += 1
                else:
                    # it is a crash, we need to iterate
                    if predicted_df_idx < len(predicted_df)-1:
                        while predicted_df.iloc[predicted_df_idx+1]['target'] == predicted_df.iloc[predicted_df_idx]['target']:
                            predicted_df_idx += 1
                            if predicted_df_idx >= len(predicted_df)-1:
                                break
                        predicted_df_idx += 1

    average_wind = np.divide(average_wind, total_penalty)
    return total_penalty, crash_time_stamp, average_wind, max_wind


def a_star_length(cf, csv_for_evaluation):
    """
    This is a script for evaluating predicted route's length
    :param cf:
    :param csv_for_evaluation:
    :return:
    """
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    predicted_df = pd.read_csv(csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid'])
    predicted_df_idx = 0
    total_penalty = np.ones(shape=(5, 10)) * 24 * 60

    for day in [1, 2, 3, 4, 5]:
        for goal_city in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print('Day: %d, city: %d' % (day, goal_city))
            route_list = []
            start_loc = (int(city_data_df.iloc[0]['xid']), int(city_data_df.iloc[0]['yid']))
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']), int(city_data_df.iloc[goal_city]['yid']))

            if predicted_df_idx >= len(predicted_df) - 1:
                # meaining the rest of the goals are not reached (e.g., day 4, and 5)
                break

            start_loc_pred = (predicted_df.iloc[predicted_df_idx]['xid'], predicted_df.iloc[predicted_df_idx]['yid'])
            day_pred = predicted_df.iloc[predicted_df_idx]['date']
            target_pred = predicted_df.iloc[predicted_df_idx]['target']

            if goal_city != predicted_df.iloc[predicted_df_idx]['target']:
                print('Sadly, we never reached the goal: %d' % goal_city)
                total_penalty[day - 1, goal_city - 1] = 24 * 60
                print('#' * 20 + '5' * 20 + '#' * 20)
                continue

            assert start_loc == start_loc_pred, "Starting x, y not the same!"
            assert day_pred == day, "Starting day not the same!"
            assert target_pred == goal_city, "Starting city not the same!"
            min = 0
            acc_min = 0
            hour = 3

            next_loc_pred = start_loc_pred

            while not next_loc_pred == goal_loc:
                route_list.append(next_loc_pred)
                min += 2
                acc_min += 2
                predicted_df_idx += 1
                next_loc_pred = (predicted_df.iloc[predicted_df_idx]['xid'], predicted_df.iloc[predicted_df_idx]['yid'])
                day_pred_next = predicted_df.iloc[predicted_df_idx]['date']
                target_pred_next = predicted_df.iloc[predicted_df_idx]['target']
                assert day_pred_next == day_pred, "Predict day not the same!"
                assert target_pred_next == target_pred, "Predict city not the same!"
                assert np.sum(np.abs(next_loc_pred[0] - start_loc_pred[0]) + np.abs(next_loc_pred[1] - start_loc_pred[1])) <=1, "Unlawful move!"

                start_loc_pred = next_loc_pred

                if min >= 60:
                    min = 0
                    hour += 1

            if predicted_df_idx < len(predicted_df):
                if next_loc_pred == goal_loc:
                    total_penalty[day-1, goal_city-1] = acc_min
                    print('Goal reached in %d mins' % acc_min)
                    predicted_df_idx += 1
                else:
                    # it is a crash, we need to iterate
                    if predicted_df_idx < len(predicted_df)-1:
                        while predicted_df.iloc[predicted_df_idx+1]['target'] == predicted_df.iloc[predicted_df_idx]['target']:
                            predicted_df_idx += 1
                            if predicted_df_idx >= len(predicted_df)-1:
                                break
                        predicted_df_idx += 1

    return total_penalty


def evaluation_plot(cf):
    """
    This is a script for visualising predicted route's length
    :param cf:
    :param csv_for_evaluation:
    :return:
    """
    csv_for_evaluation = cf.csv_for_evaluation
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    predicted_df = pd.read_csv(csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid'])

    # draw figure maximum
    plt.figure(1)
    plt.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    for day in cf.evaluation_days:
        for goal_city in cf.evaluation_goal_cities:
            print('Day: %d, city: %d' % (day, goal_city))
            route_list = []
            start_loc = (int(city_data_df.iloc[0]['xid']), int(city_data_df.iloc[0]['yid']))
            goal_loc = (int(city_data_df.iloc[goal_city]['xid']), int(city_data_df.iloc[goal_city]['yid']))
            predicted_df_idx = 0
            if predicted_df_idx >= len(predicted_df) - 1:
                # meaining the rest of the goals are not reached (e.g., day 4, and 5)
                break

            start_loc_pred = (predicted_df.iloc[predicted_df_idx]['xid'], predicted_df.iloc[predicted_df_idx]['yid'])
            predicted_df_now = predicted_df.loc[(predicted_df['date']==day) & (predicted_df['target']==goal_city)]

            day_pred = predicted_df_now.iloc[predicted_df_idx]['date']
            target_pred = predicted_df_now.iloc[predicted_df_idx]['target']

            if goal_city != predicted_df_now.iloc[predicted_df_idx]['target']:
                print('Sadly, we never reached the goal: %d' % goal_city)
                print('#' * 20 + '5' * 20 + '#' * 20)
                continue

            assert start_loc == start_loc_pred, "Starting x, y not the same!"
            assert day_pred == day, "Starting day not the same!"
            assert target_pred == goal_city, "Starting city not the same!"
            min = 0
            acc_min = 0
            hour = 3
            if day < 6:  # meaning this is a training day
                weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            else:
                weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (3, day, hour)

            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))

            # begin to draw
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
            plt.title(weather_name[:-6] + str(hour) + '_' + '_goal city' + str(goal_city))

            next_loc_pred = start_loc_pred

            while not next_loc_pred == goal_loc:
                route_list.append(next_loc_pred)
                min += 2
                acc_min += 2
                predicted_df_idx += 1
                next_loc_pred = (predicted_df_now.iloc[predicted_df_idx]['xid'], predicted_df_now.iloc[predicted_df_idx]['yid'])
                day_pred_next = predicted_df_now.iloc[predicted_df_idx]['date']
                target_pred_next = predicted_df_now.iloc[predicted_df_idx]['target']
                assert day_pred_next == day_pred, "Predict day not the same!"
                assert target_pred_next == target_pred, "Predict city not the same!"
                assert np.sum(np.abs(next_loc_pred[0] - start_loc_pred[0]) + np.abs(next_loc_pred[1] - start_loc_pred[1])) <=1, "Unlawful move!"

                if min >= 60:
                    #plt.waitforbuttonpress()
                    min = 0
                    hour += 1
                    if day < 6:  # meaning this is a training day
                        weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                    else:
                        weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (3, day, hour)
                    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                    plt.clf()
                    # we plot every hour
                    for h in range(3, hour):
                        for p in route_list[(h-3)*30:(h-2)*30]:
                            plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

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
                    plt.title(weather_name[:-6] + str(hour) + '_' + '_goal city' + str(goal_city))
                    plt.waitforbuttonpress(0.1)
                # Now we check whether the aircraft crash or not
                if wind_real_day_hour[next_loc_pred[0]-1, next_loc_pred[1]-1] >= 15.:
                    print('Crash! Day: %d, city: %d, hour: %d, min: %d' % (day, goal_city, hour, min))
                    # we break the loop
                    # break
                    plt.title('Crash! Day: %d, city: %d, hour: %d, min: %d' % (day, goal_city, hour, min))
                    plt.waitforbuttonpress(0.001)
                    start_loc_pred = next_loc_pred
                else:
                    start_loc_pred = next_loc_pred

            # plot the last bit route
            for h in range(3, hour):
                for p in route_list[(h - 3) * 30:(h - 2) * 30]:
                    plt.scatter(p[1], p[0], c=cf.colors[np.mod(h, 2)], s=10)

            for p in route_list[(hour-3)*30:]:
                plt.scatter(p[1], p[0], c=cf.colors[np.mod(hour, 2)], s=10)

            plt.waitforbuttonpress(3)

            if predicted_df_idx < len(predicted_df):
                if next_loc_pred == goal_loc:
                    print('Goal reached in %d mins' % acc_min)
                    predicted_df_idx += 1
                else:
                    # it is a crash, we need to iterate
                    if predicted_df_idx < len(predicted_df)-1:
                        while predicted_df.iloc[predicted_df_idx+1]['target'] == predicted_df.iloc[predicted_df_idx]['target']:
                            predicted_df_idx += 1
                            if predicted_df_idx >= len(predicted_df)-1:
                                break
                        predicted_df_idx += 1


