import os
import pandas as pd
import multiprocessing
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import matplotlib.gridspec as gridspec
from tools.extract_multation import extract_multation
from tools.A_star_alibaba import extract_weather_data


def plot_real_wind(cf):
    # Create the data generators
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TrainRealFile))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    date_unique = pd.unique(wind_real_df['date_id'])
    hour_unique = pd.unique(wind_real_df['hour'])

    for d_unique in date_unique:
        for h_unique in hour_unique:
            print('Processing real data for date: %d, hour: %d' % (d_unique, h_unique))
            wind_real_df_day = wind_real_df.loc[wind_real_df['date_id'] == d_unique]
            wind_real_df_day_hour = wind_real_df_day.loc[wind_real_df_day['hour'] == h_unique]

            if not len(x_unique) * len(y_unique) == wind_real_df_day_hour.index.__len__():
                print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                      (len(x_unique) * len(y_unique), wind_real_df_day_hour.index.__len__()))

            wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
            rainfall_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
            for idx in range(wind_real_df_day_hour.index.__len__()):
                x_loc = int(wind_real_df_day_hour.iloc[idx]['xid']) - 1
                y_loc = int(wind_real_df_day_hour.iloc[idx]['yid']) - 1
                wind = np.float32(wind_real_df_day_hour.iloc[idx]['wind'])
                wind_real_day_hour[x_loc, y_loc] = wind
                rainfall = np.float32(wind_real_df_day_hour.iloc[idx]['rainfall'])
                rainfall_real_day_hour[x_loc, y_loc] = rainfall

            np.save(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy'%(d_unique, h_unique)), wind_real_day_hour)
            np.save(os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)), rainfall_real_day_hour)


def plot_real_wind_worker(cf, wind_real_df_day_hour, d_unique, h_unique, x_unique, y_unique):
    start_time = timer()
    if not len(x_unique) * len(y_unique) == wind_real_df_day_hour.index.__len__():
        print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
              (len(x_unique) * len(y_unique), wind_real_df_day_hour.index.__len__()))

    wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    rainfall_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    for idx in range(wind_real_df_day_hour.index.__len__()):
        x_loc = int(wind_real_df_day_hour.iloc[idx]['xid']) - 1
        y_loc = int(wind_real_df_day_hour.iloc[idx]['yid']) - 1
        wind = np.float32(wind_real_df_day_hour.iloc[idx]['wind'])
        wind_real_day_hour[x_loc, y_loc] = wind
        rainfall = np.float32(wind_real_df_day_hour.iloc[idx]['rainfall'])
        rainfall_real_day_hour[x_loc, y_loc] = rainfall

    np.save(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy' % (d_unique, h_unique)), wind_real_day_hour)
    np.save(os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)),
            rainfall_real_day_hour)
    print('Finish writing Real weather, using %.2f sec!' % (timer() - start_time))


def plot_real_wind_multiprocessing(cf):
    # Create the data generators
    start_time = timer()
    # Create the data generators
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TrainRealFile))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    date_unique = pd.unique(wind_real_df['date_id'])
    hour_unique = pd.unique(wind_real_df['hour'])

    jobs = []
    multiprocessing.log_to_stderr()

    for d_unique in date_unique:
        for h_unique in hour_unique:
            if not os.path.exists(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy' % (d_unique, h_unique))):
                print('Processing real data for date: %d, hour: %d' % (d_unique, h_unique))
                wind_real_df_day = wind_real_df.loc[wind_real_df['date_id'] == d_unique]
                wind_real_df_day_hour = wind_real_df_day.loc[wind_real_df_day['hour'] == h_unique]

                p = multiprocessing.Process(target=plot_real_wind_worker, args=(cf, wind_real_df_day_hour, d_unique, h_unique, x_unique, y_unique))
                p.start()
                jobs.append(p)

            # because of the memory constraint, we need to wait for the previous to finish to finish in order
            # to initiate another function...
            if len(jobs) > cf.num_threads:
                jobs[-cf.num_threads].join()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()

    print('Finish writing Real weather, using %.2f sec!' % (timer() - start_time))


def plt_forecast_wind_train(cf):
    # Create the data generators
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TrainForecastFile))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    model_unique = pd.unique(wind_real_df['model'])

    for m_unique in model_unique:
        wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
        for d_unique in cf.day_list:
            wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
            hour_unique = sorted(pd.unique(wind_real_df_model_day['hour']))
            for h_unique in hour_unique:
                if not os.path.exists(os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))):
                    print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))
                    wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]

                    if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
                        print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                              (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

                    wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
                    for idx in range(wind_real_df_model_day_hour.index.__len__()):
                        x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
                        y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
                        wind = np.float32(wind_real_df_model_day_hour.iloc[idx]['wind'])
                        wind_real_day_hour[x_loc, y_loc] = wind

                    np.save(os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique)),
                            wind_real_day_hour)


def plt_forecast_wind_train_workers(cf, wind_real_df_model_day_hour, m_unique, d_unique, h_unique, x_unique, y_unique):
    start_time = timer()
    if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
        print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
              (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

    wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    rainfall_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    for idx in range(wind_real_df_model_day_hour.index.__len__()):
        x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
        y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
        wind = np.float32(wind_real_df_model_day_hour.iloc[idx]['wind'])
        wind_real_day_hour[x_loc, y_loc] = wind
        rainfall = np.float32(wind_real_df_model_day_hour.iloc[idx]['rainfall'])
        rainfall_real_day_hour[x_loc, y_loc] = rainfall

    np.save(os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique)), wind_real_day_hour)
    np.save(os.path.join(cf.rainfall_save_path, 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique)), rainfall_real_day_hour)
    print('Finish writing Train weather, using %.2f sec!' % (timer() - start_time))


def plt_forecast_wind_train_multiprocessing(cf):
    # Create the data generators
    start_time = timer()
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TrainForecastFile))
    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    #model_unique = pd.unique(wind_real_df['model'])
    # we use multiprocessing here
    jobs = []
    multiprocessing.log_to_stderr()

    for m_unique in cf.model_unique:
        for d_unique in [1, 2, 3, 4, 5]:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1]+1):
                if not os.path.exists(os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))):
                    print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))

                    wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
                    wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
                    wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]
                    p = multiprocessing.Process(target=plt_forecast_wind_train_workers, args=(cf, wind_real_df_model_day_hour, m_unique, d_unique, h_unique, x_unique, y_unique))
                    p.start()
                    jobs.append(p)

                # because of the memory constraint, we need to wait for the previous to finish to finish in order
                # to initiate another function...
                if len(jobs) > cf.num_threads:
                    jobs[-cf.num_threads].join()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()

    print('Finish writing Train weather, using %.2f sec!' % (timer() - start_time))


def plt_forecast_wind_test_workers(cf, wind_real_df_model_day_hour, m_unique, d_unique, h_unique, x_unique, y_unique):

    start_time = timer()
    if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
        print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
              (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

    wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    rainfall_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
    for idx in range(wind_real_df_model_day_hour.index.__len__()):
        x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
        y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
        wind = np.float32(wind_real_df_model_day_hour.iloc[idx]['wind'])
        rainfall = np.float32(wind_real_df_model_day_hour.iloc[idx]['rainfall'])
        wind_real_day_hour[x_loc, y_loc] = wind
        rainfall_real_day_hour[x_loc, y_loc] = rainfall

    np.save(os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' %
                         (m_unique, d_unique, h_unique)), wind_real_day_hour)
    np.save(os.path.join(cf.rainfall_save_path, 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' %
                         (m_unique, d_unique, h_unique)), rainfall_real_day_hour)
    print('Finish writing one hour wind & rainfall data saving, using %.2f sec!' % (timer() - start_time))


def plt_forecast_wind_test_multiprocessing(cf):
    # Create the data generators
    start_time = timer()
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TestForecastFile))
    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    # we use multiprocessing here
    jobs = []
    multiprocessing.log_to_stderr()

    for m_unique in cf.model_unique:
        for d_unique in [6, 7, 8, 9, 10]:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1]+1):
                if not os.path.exists(os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))):
                    print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))

                    wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
                    wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
                    wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]
                    p = multiprocessing.Process(target=plt_forecast_wind_test_workers, args=(cf, wind_real_df_model_day_hour, m_unique, d_unique, h_unique, x_unique, y_unique))
                    p.start()
                    jobs.append(p)
                # because of the memory constraint, we need to wait for the previous to finish to finish in order
                # to initiate another function...
                if len(jobs) > cf.num_threads:
                    jobs[-cf.num_threads].join()

    # waiting for the all the job to finish
    for j in jobs:
        j.join()

    print('Finish writing Test weather, using %.2f sec!' % (timer() - start_time))


def plt_forecast_wind_test(cf):
    # Create the data generators
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, cf.TestForecastFile))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    model_unique = pd.unique(wind_real_df['model'])

    for m_unique in model_unique:
        wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
        for d_unique in cf.day_list:
            wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
            hour_unique = sorted(pd.unique(wind_real_df_model_day['hour']))
            for h_unique in hour_unique:
                start_time = timer()
                print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))
                wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]

                if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
                    print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                          (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

                wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
                for idx in range(wind_real_df_model_day_hour.index.__len__()):
                    x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
                    y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
                    wind = np.float32(wind_real_df_model_day_hour.iloc[idx]['wind'])
                    wind_real_day_hour[x_loc, y_loc] = wind

                np.save(os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' %
                                     (m_unique, d_unique, h_unique)), wind_real_day_hour)
                print('Finish writing one hour wind data saving, using %.2f sec!' % (timer() - start_time))


def evaluation_plot_multi(cf):
    """
    This is a script for visualising predicted route's length from multiple output
    :param cf:
    :param csvd_for_evaluation:
    :return:
    """
    csvs_for_evaluation = cf.csvs_for_evaluation
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    predicted_dfs = []
    for csv_for_evaluation in csvs_for_evaluation:
        predicted_dfs.append(pd.read_csv(csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid']))

    # draw figure maximum
    plt.figure(1)
    plt.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    for day in cf.evaluation_days:
        for goal_city in cf.evaluation_goal_cities:
            # begin to draw
            plt.clf()
            reach_count = np.zeros(len(predicted_dfs))
            for hour in range(3, 20):
                if day < 6:  # meaning this is a training day
                    weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                else:
                    weather_name = 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (3, day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
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

                # we plot every hour
                for p_idx, predicted_df in enumerate(predicted_dfs):
                    predicted_df_now = predicted_df.loc[(predicted_df['date'] == day) & (predicted_df['target'] == goal_city)]
                    for h in range(3, hour+1):
                        for idx in list(range((h-3)*30, (h-2)*30)):
                            if len(predicted_df_now) > idx:
                                plt.scatter(predicted_df_now.iloc[idx]['yid'], predicted_df_now.iloc[idx]['xid'],
                                            c=cf.colors[np.mod(p_idx, len(cf.colors))], s=10, marker=cf.markers[p_idx])
                            else:
                                print('Reached!')
                                reach_count[p_idx] = 1
                # plot some legend
                for p_idx, predicted_df in enumerate(predicted_dfs):
                    plt.scatter(predicted_df_now.iloc[0]['yid'], predicted_df_now.iloc[0]['xid'],
                                c=cf.colors[np.mod(p_idx, len(cf.colors))], s=10, marker=cf.markers[p_idx],label=cf.csv_names[p_idx])
                plt.legend(loc='upper right', shadow=True)
                plt.waitforbuttonpress(1)
                if np.sum(reach_count) == len(predicted_dfs):
                    break


def plot_state_action_value(model, city_data_df, cf):
    plt.imshow(np.max(np.max(model.stateActionValues, axis=3), axis=2))
    plt.clim(0, 540)
    #cb = plt.colorbar(shrink=0.5, aspect=20, fraction=.12, pad=.02)
    cb = plt.colorbar()
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
    plt.axis('off')

    idx = 0
    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
    cid = int(city_data_df.iloc[idx]['cid'])
    plt.scatter(y_loc, x_loc, c='r', s=40)
    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=50)

    idx = cf.goal_city_list[0]
    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
    cid = int(city_data_df.iloc[idx]['cid'])
    plt.scatter(y_loc, x_loc, c='r', s=40)
    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=50)

    plt.show()


def plot_all_wind(cf):

    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))

    if cf.plot_train_model:
        # plot figures here
        fig = plt.figure(num=1)
        fig.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        left = gridspec.GridSpec(1, 1)
        left.update(left=0.05, right=0.2, hspace=0.05)
        right = gridspec.GridSpec(2, 5)
        right.update(left=0.25, right=0.98, hspace=0.05)

        total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)+1))
        for d_unique in cf.day_list:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1]+1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy'%(d_unique, h_unique)))
                total_weather[:, :, 0] = wind_real_day_hour
                for m_unique in cf.model_number:
                    np_file = os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique] = np.load(np_file)

                # We need to normalise the map to standardize visualisation
                total_weather_max = np.max(total_weather)
                total_weather = (total_weather-15)/total_weather_max
                # plot all the real and forecast model here:
                fig.clf()
                plt.subplot(left[:, :])
                plt.imshow(total_weather[:, :, 0],  cmap='jet')
                plt.colorbar()
                plt.title('real_wind_day_%d_hour_%d' % (d_unique, h_unique))
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
                CS = plt.contour(X, Y, total_weather[:, :, 0], (0,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                for m_unique in cf.model_unique:
                    # plot all the real and forecast model here:
                    plt.subplot(right[(m_unique-1)//5, np.mod(m_unique, 5)-1])
                    plt.imshow(total_weather[:, :, m_unique],  cmap='jet')
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
                    CS = plt.contour(X, Y, total_weather[:, :, m_unique], (0,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title('Model %d' % m_unique)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.waitforbuttonpress(0.0001)
                # plt.show()
                save_fig_name = os.path.join(cf.fig_wind_train_path, '%s.png' % ('Train_real_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')

    elif cf.plot_test_model:
        # plot figures here
        fig = plt.figure(num=1)
        fig.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        right = gridspec.GridSpec(2, 5)
        right.update(left=0.02, right=0.98, hspace=0.05)

        total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)))
        for d_unique in cf.day_list:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (
                    m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique-1] = np.load(np_file)

                # We need to normalise the map to standardize visualisation
                fig.clf()
                # total_weather_max = np.max(total_weather)
                # total_weather = (total_weather - 15) / total_weather_max

                for m_unique in cf.model_unique:
                    # plot all the real and forecast model here:
                    plt.subplot(right[(m_unique - 1) // 5, np.mod(m_unique, 5) - 1])
                    plt.imshow(total_weather[:, :, m_unique-1], cmap='jet')
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                    # we also draw some contours
                    x = np.arange(0, cf.grid_world_shape[1], 1)
                    y = np.arange(0, cf.grid_world_shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, total_weather[:, :, m_unique-1], (cf.wall_wind,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)

                #plt.colorbar()
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                fig.suptitle('Test_forecast_wind_day_%d_hour_%d.npy' % (d_unique, h_unique), size=20)
                plt.waitforbuttonpress(0.01)
                save_fig_name = os.path.join(cf.fig_wind_test_path, '%s.png' % ('Test_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')

    else:
        plt.figure(1)
        plt.clf()

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        # we plot everything
        weather_list = sorted(os.listdir(cf.wind_save_path), reverse=True)
        for weather_name in weather_list:
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))

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
            plt.title(weather_name[:-4])

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.waitforbuttonpress(0.0001)
            #plt.show()
            save_fig_name = os.path.join(cf.fig_save_path, '%s.png'% (weather_name[:-4]))
            print('Saving figure %s' % save_fig_name)
            plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')


def plot_all_wind_new(cf):

    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))  # plot figures here
    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    left = gridspec.GridSpec(1, 1)
    left.update(left=0.05, right=0.2, hspace=0.05)
    right = gridspec.GridSpec(2, 5)
    right.update(left=0.25, right=0.98, hspace=0.05)

    # real & train data
    total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))
    for d_unique in [1, 2, 3, 4, 5]:
        for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
            print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
            wind_real_day_hour = np.load(
                os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy' % (d_unique, h_unique)))
            total_weather[:, :, 0] = wind_real_day_hour
            for m_unique in cf.model_unique:
                np_file = os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (
                m_unique, d_unique, h_unique))
                total_weather[:, :, m_unique] = np.load(np_file)

            # We need to normalise the map to standardize visualisation
            total_weather_max = np.max(total_weather)
            total_weather = (total_weather - 15) / total_weather_max
            # plot all the real and forecast model here:
            fig.clf()
            plt.subplot(left[:, :])
            plt.imshow(total_weather[:, :, 0], cmap='jet')
            plt.colorbar()
            plt.title('real_wind_day_%d_hour_%d' % (d_unique, h_unique))
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
            CS = plt.contour(X, Y, total_weather[:, :, 0], (0,), colors='k')
            plt.clabel(CS, inline=1, fontsize=10)
            for m_unique in cf.model_unique:
                # plot all the real and forecast model here:
                plt.subplot(right[(m_unique - 1) // 5, np.mod(m_unique, 5) - 1])
                plt.imshow(total_weather[:, :, m_unique], cmap='jet')
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
                CS = plt.contour(X, Y, total_weather[:, :, m_unique], (0,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                plt.title('Model %d' % m_unique)

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.waitforbuttonpress(0.0001)
            # plt.show()
            save_fig_name = os.path.join(cf.fig_wind_save_train_path,
                                         '%s.png' % ('Train_real_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
            plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')

    # plot figures here
    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    right = gridspec.GridSpec(2, 5)
    right.update(left=0.02, right=0.98, hspace=0.05)

    # test data
    total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)))
    for d_unique in [6, 7, 8, 9, 10]:
        for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
            print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
            for m_unique in cf.model_unique:
                np_file = os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (
                    m_unique, d_unique, h_unique))
                total_weather[:, :, m_unique - 1] = np.load(np_file)

            # We need to normalise the map to standardize visualisation
            fig.clf()
            total_weather_max = np.max(total_weather)
            total_weather = (total_weather - 15) / total_weather_max

            for m_unique in cf.model_unique:
                # plot all the real and forecast model here:
                plt.subplot(right[(m_unique - 1) // 5, np.mod(m_unique, 5) - 1])
                plt.imshow(total_weather[:, :, m_unique - 1], cmap='jet')
                # we also plot the city location
                for idx in range(city_data_df.index.__len__()):
                    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                    cid = int(city_data_df.iloc[idx]['cid'])
                    plt.scatter(y_loc, x_loc, c='r', s=40)
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                # we also draw some contours
                x = np.arange(0, cf.grid_world_shape[1], 1)
                y = np.arange(0, cf.grid_world_shape[0], 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, total_weather[:, :, m_unique - 1], (0,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            fig.suptitle('Test_forecast_wind_day_%d_hour_%d.npy' % (d_unique, h_unique), size=20)
            plt.waitforbuttonpress(0.0001)
            save_fig_name = os.path.join(cf.fig_wind_save_test_path,
                                         '%s.png' % ('Test_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
            plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')


def plot_all_rainfall(cf):

    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))  # plot figures here
    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    left = gridspec.GridSpec(1, 1)
    left.update(left=0.05, right=0.2, hspace=0.05)
    right = gridspec.GridSpec(2, 5)
    right.update(left=0.25, right=0.98, hspace=0.05)

    # real & train rainfall data
    if cf.plot_train_model:
        total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))
        for d_unique in cf.day_list:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                rainfall_real_day_hour = np.load(os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)))
                total_weather[:, :, 0] = rainfall_real_day_hour
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.rainfall_save_path, 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique] = np.load(np_file)

                # We need to normalise the map to standardize visualisation
                # total_weather_max = np.max(total_weather)
                # total_weather = (total_weather - 4) / total_weather_max
                # plot all the real and forecast model here:
                fig.clf()
                plt.subplot(left[:, :])
                plt.imshow(total_weather[:, :, 0], cmap='jet')
                plt.colorbar()
                plt.title('real_rainfall_day_%d_hour_%d' % (d_unique, h_unique))
                # we also plot the city location
                for idx in range(city_data_df.index.__len__()):
                    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                    cid = int(city_data_df.iloc[idx]['cid'])
                    plt.scatter(y_loc, x_loc, c='r', s=40)
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                # we also draw some contours
                x = np.arange(0, rainfall_real_day_hour.shape[1], 1)
                y = np.arange(0, rainfall_real_day_hour.shape[0], 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, total_weather[:, :, 0], (cf.wall_rainbfall,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                for m_unique in cf.model_unique:
                    # plot all the real and forecast model here:
                    plt.subplot(right[(m_unique - 1) // 5, np.mod(m_unique, 5) - 1])
                    plt.imshow(total_weather[:, :, m_unique], cmap='jet')
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                    # we also draw some contours
                    x = np.arange(0, rainfall_real_day_hour.shape[1], 1)
                    y = np.arange(0, rainfall_real_day_hour.shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, total_weather[:, :, m_unique], (cf.wall_rainbfall,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title('Model %d' % m_unique)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.waitforbuttonpress(0.0001)
                # plt.show()
                save_fig_name = os.path.join(cf.fig_rainfall_train_path, '%s.png' % ('Train_real_models_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))
                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')

    elif cf.plot_test_model:
        # plot figures here
        fig = plt.figure(num=1)
        fig.clf()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        right = gridspec.GridSpec(2, 5)
        right.update(left=0.02, right=0.98, hspace=0.05)

        # test data
        total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)))
        for d_unique in [6, 7, 8, 9, 10]:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.rainfall_save_path, 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (
                        m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique - 1] = np.load(np_file)

                # We need to normalise the map to standardize visualisation
                fig.clf()
                # total_weather_max = np.max(total_weather)
                # total_weather = (total_weather - 4) / total_weather_max

                for m_unique in cf.model_unique:
                    # plot all the real and forecast model here:
                    plt.subplot(right[(m_unique - 1) // 5, np.mod(m_unique, 5) - 1])
                    plt.imshow(total_weather[:, :, m_unique - 1], cmap='jet')
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                    # we also draw some contours
                    x = np.arange(0, cf.grid_world_shape[1], 1)
                    y = np.arange(0, cf.grid_world_shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, total_weather[:, :, m_unique - 1], (cf.wall_rainbfall,), colors='orange')
                    plt.clabel(CS, inline=1, fontsize=10)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                fig.suptitle('Test_forecast_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique), size=20)
                plt.waitforbuttonpress(0.01)
                save_fig_name = os.path.join(cf.fig_rainfall_test_path, '%s.png' % ('Test_models_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))
                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')


def plot_wind_with_rainfall(cf):
    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))

    # plot figures here
    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    left = gridspec.GridSpec(1, 1)
    left.update(left=0.05, right=0.45, hspace=0.05)
    right = gridspec.GridSpec(1, 1)
    right.update(left=0.5, right=0.98, hspace=0.05)

    if cf.plot_train_model:
        total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)+1))
        for d_unique in cf.day_list:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1]+1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy'%(d_unique, h_unique)))
                total_weather[:, :, 0] = wind_real_day_hour
                for m_unique in cf.model_number:
                    np_file = os.path.join(cf.wind_save_path, 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique] = np.load(np_file)

                mean_wind = np.mean(total_weather[:, :, 1:], axis=2)
                fig.clf()
                plt.subplot(left[0, 0])
                plt.imshow(total_weather[:, :, 0], cmap='jet')
                plt.title('real wind')
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
                CS = plt.contour(X, Y, total_weather[:, :, 0], (cf.wall_wind,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                # we plot the average prediction

                plt.subplot(left[1, 0])
                plt.imshow(mean_wind, cmap='jet')
                plt.colorbar()
                plt.title('mean wind')

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
                CS = plt.contour(X, Y, mean_wind, (cf.wall_wind,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)

                # we draw rainfall
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                rainfall_real_day_hour = np.load(os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)))
                total_weather[:, :, 0] = rainfall_real_day_hour
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.rainfall_save_path, 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique] = np.load(np_file)

                mean_rainfall = np.mean(total_weather, axis=2)
                plt.subplot(right[0, 0])
                plt.imshow(total_weather[:, :, 0], cmap='jet')
                plt.title('real rainfall')
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
                CS = plt.contour(X, Y, total_weather[:, :, 0], (cf.wall_rainbfall,), colors='orange')
                plt.clabel(CS, inline=1, fontsize=10)

                plt.subplot(right[1, 0])
                plt.imshow(mean_rainfall, cmap='jet')
                plt.colorbar()
                plt.title('mean rainfall')

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
                CS = plt.contour(X, Y, mean_rainfall, (cf.wall_rainbfall,), colors='orange')
                plt.clabel(CS, inline=1, fontsize=10)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                fig.suptitle('Train_wind_with_rainfall_day_%d_hour_%d' % (d_unique, h_unique), size=20)
                plt.waitforbuttonpress(0.01)
                save_fig_name = os.path.join(cf.fig_wind_with_rainfall_train_path, '%s.png' % ('Train_wind_with_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))
                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')

    elif cf.plot_test_model:
        for d_unique in cf.day_list:
            for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
                print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)))
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.wind_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.npy' % (
                    m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique-1] = np.load(np_file)

                mean_wind = np.mean(total_weather[:, :, 1:], axis=2)
                fig.clf()
                plt.subplot(left[0, 0])
                plt.imshow(mean_wind, cmap='jet')
                plt.colorbar()
                plt.title('mean wind')

                # we also plot the city location
                for idx in range(city_data_df.index.__len__()):
                    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                    cid = int(city_data_df.iloc[idx]['cid'])
                    plt.scatter(y_loc, x_loc, c='r', s=40)
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                # we also draw some contours
                x = np.arange(0, total_weather.shape[1], 1)
                y = np.arange(0, total_weather.shape[0], 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, mean_wind, (cf.wall_wind,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)

                # we draw rainfall
                # print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
                total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique)))
                for m_unique in cf.model_unique:
                    np_file = os.path.join(cf.rainfall_save_path, 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (
                        m_unique, d_unique, h_unique))
                    total_weather[:, :, m_unique - 1] = np.load(np_file)

                mean_rainfall = np.mean(total_weather, axis=2)
                plt.subplot(right[0, 0])
                plt.imshow(mean_rainfall, cmap='jet')
                plt.colorbar()
                plt.title('mean rainfall')

                # we also plot the city location
                for idx in range(city_data_df.index.__len__()):
                    x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                    y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                    cid = int(city_data_df.iloc[idx]['cid'])
                    plt.scatter(y_loc, x_loc, c='r', s=40)
                    plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=20)

                # we also draw some contours
                x = np.arange(0, total_weather.shape[1], 1)
                y = np.arange(0, total_weather.shape[0], 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, mean_rainfall, (cf.wall_rainbfall,), colors='orange')
                plt.clabel(CS, inline=1, fontsize=10)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                fig.suptitle('Test_wind_with_rainfall_day_%d_hour_%d' % (d_unique, h_unique), size=20)
                plt.waitforbuttonpress(0.01)
                save_fig_name = os.path.join(cf.fig_wind_with_rainfall_test_path, '%s.png' % ('Test_wind_with_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))

                plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')



def plot_multation(cf):

    # # Create the data generators
    # fig = plt.figure(num=1)
    # fig.clf()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    # left = gridspec.GridSpec(1, 1)
    # left.update(left=0.05, right=0.2, hspace=0.05)
    # right = gridspec.GridSpec(2, 5)
    # right.update(left=0.25, right=0.98, hspace=0.05)

    # real & train data
    total_weather_multation = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))

    for d_unique in [1, 2, 3, 4, 5]:
        for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
            print('Processing multation data for  date: %d, hour: %d' % (d_unique, h_unique))

            use_real_weather_state = cf.use_real_weather
            cf.use_real_weather = True

            # real_weather_multation = extract_multation(cf, d_unique, h_unique)
            # total_weather_multation[:, :, 0] = real_weather_multation

            models = cf.model_number
            cf.use_real_weather = False
            # for model_num in cf.model_number:
            #     cf.model_number = list([model_num])
            #     total_weather_multation[:, :, model_num] = extract_multation(cf, d_unique, h_unique)

            cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            wind_real_day_hour, rainfall_real_day_hour = extract_weather_data(cf, d_unique, h_unique)
            weather_day_hour = np.maximum(wind_real_day_hour, rainfall_real_day_hour * 15.0 / 4)
            mean_weather_multation = extract_multation(weather_day_hour, cf.radius)



            cf.model_number = models
            cf.use_real_weather = use_real_weather_state

            # We need to normalise the map to standardize visualisation
            # total_weather_max = np.max(total_weather_multation)
            # mean_motation = total_weather_multation.mean()
            # total_weather_multation = (total_weather_multation - mean_motation) / total_weather_max


            # mean_weather_multation = (mean_weather_multation - mean_motation) / total_weather_max

            # plot all the real and forecast model here:
            # fig.clf()
            # plt.subplot(left[:, :])
            # plt.imshow(total_weather_multation[:, :, 0], cmap='jet')
            # plt.colorbar()
            # plt.title('real_weather_multation_day_%d_hour_%d' % (d_unique, h_unique))
            #
            # total_diff = np.zeros_like(real_weather_multation)
            # for model_num in cf.model_number:
            #     # # plot all the real and forecast model here:
            #     # plt.subplot(right[(model_num - 1) // 5, np.mod(model_num, 5) - 1])
            #     # # plt.imshow(total_weather_multation[:, :, model_num], cmap='jet')
            #     # plt.imshow(mean_weather_multation, cmap='jet')
            #     total_diff += total_weather_multation[:, :, model_num]
            #     diff = np.abs(total_weather_multation[:, :, model_num] - real_weather_multation).sum()
            #     print('Model_num_'+str(model_num)+':'+str(diff))
            #     # plt.title('Model %d' % model_num)
            #
            #
            # diff = np.abs(total_diff/10 - real_weather_multation).sum()
            # print('Mean_diff' + str(model_num) + ':' + str(diff))


            # diff =  np.abs(mean_weather_multation - real_weather_multation).sum()
            # print('Mean_Model_' + str(model_num) + ':' + str(diff))

            print('Mean_Model_max:'+str(mean_weather_multation.max()))
            print('Mean_Model_min:' + str(mean_weather_multation.min()))
            print('Mean_Model_mean:' + str(mean_weather_multation.mean()))


            # fig.clf()
            # plt.subplot(right[0, 0])
            # plt.imshow(mean_weather_multation, cmap='jet')
            # plt.title('Model mean' )
            #
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            # plt.waitforbuttonpress(100)
            # plt.show()
            # save_fig_name = os.path.join(cf.fig_wind_save_train_path,
            #                              '%s.png' % ('Train_real_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
            # plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')


def evaluation_plot_real_with_mean(cf):
    """
    This is a script for visualising predicted route's length with real weather juxtaposing
    with mean prediction
    :param cf:
    :param csvd_for_evaluation:
    :return:
    """

    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    predicted_df = pd.read_csv(cf.csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid'])
    # for csv_for_evaluation in cf.csv_for_evaluation:
    #     predicted_dfs.append(pd.read_csv(csv_for_evaluation, names=['target', 'date', 'time', 'xid', 'yid']))

    # plot figures here

    fig = plt.figure(num=1)
    fig.clf()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    for day in cf.evaluation_days:
        for goal_city in cf.evaluation_goal_cities:
            # begin to draw
            plt.clf()
            df = predicted_df.loc[(predicted_df['date'] == day) & (predicted_df['target'] == goal_city)]
            crash_wind = False
            crash_rainfall = False
            starting_hour = int(df.iloc[0]['time'][:2])

            for step in range(1, len(df)):
                hour = int(df.iloc[step]['time'][:2])
                hour_prev_step = int(df.iloc[step-1]['time'][:2])
                if hour != hour_prev_step:
                    real_weather, mean_weather, real_wind, real_rainfall = get_wind_rainfall(cf, day, hour_prev_step)
                    # so we finish prevous hour, draw it!
                    clim_max = real_weather.max()
                    plt.clf()
                    weather_names = ['date: %d, city: %d, hour: %d' % (day, goal_city, hour_prev_step) + '_real-weather',
                                     'mean weather', 'real wind', 'real rainfall']

                    for (plot_number, im) in zip(range(4), [real_weather, mean_weather, real_wind, real_rainfall]):
                        plt.subplot(2, 2, plot_number+1)
                        plt.imshow(im, cmap=cf.colormap)
                        if plot_number < 2:
                            plt.clim(0, clim_max)
                        plt.colorbar()
                        # we also plot the city location
                        for idx in range(city_data_df.index.__len__()):
                            x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                            y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                            cid = int(city_data_df.iloc[idx]['cid'])
                            plt.scatter(y_loc, x_loc, c='r', s=40)
                            plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=10)
                        # we also draw some contours
                        x = np.arange(0, im.shape[1], 1)
                        y = np.arange(0, im.shape[0], 1)
                        X, Y = np.meshgrid(x, y)
                        if plot_number==3:
                            CS = plt.contour(X, Y, im, (4,), colors='white')
                            plt.clabel(CS, inline=1, fontsize=10)
                        else:
                            CS = plt.contour(X, Y, im, (15,), colors='k')
                            plt.clabel(CS, inline=1, fontsize=10)
                        plt.title(weather_names[plot_number])

                        # we plot every hour
                        crash_wind = False
                        crash_rainfall = False
                        for i in range(step):
                            if i <= step-30:
                                plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                            else:
                                # check whether we have crashed!
                                if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                                    crash_wind = True
                                elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                                    crash_rainfall = True
                                if not crash_wind or crash_rainfall:
                                    plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                                else:
                                    plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='red', s=1)

                    plt.waitforbuttonpress(0.1)
                    crash_wind = False
                    crash_rainfall = False
                    for i in range(max(0, step-30), step):
                        # check whether we have crashed!
                        min = int(df.iloc[i]['time'][3:])
                        if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                            crash_wind = True
                            plt.suptitle('Crash on wind:  Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                                         % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1], mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                            plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)
                        elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                            crash_rainfall = True
                            plt.suptitle('Crash on rainfall: Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                                         % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1], mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                            plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)

                        if crash_wind or crash_rainfall:
                            break

                if crash_wind or crash_rainfall:
                    plt.waitforbuttonpress(0)
                    break

            # The following code it the copy of previous code apart from the loop
            # We still need to plot the last bit
            if not (crash_wind or crash_rainfall):
                hour_prev_step = int(df.iloc[-1]['time'][:2])
                # so we finish prevous hour, draw it!
                clim_max = real_weather.max()
                plt.clf()
                weather_names = ['date: %d, city: %d, hour: %d' % (day, goal_city, hour_prev_step) + '_real-weather', 'mean weather', 'real wind', 'real rainfall']
                real_weather, mean_weather, real_wind, real_rainfall = get_wind_rainfall(cf, day, hour_prev_step)

                for (plot_number, im) in zip(range(4), [real_weather, mean_weather, real_wind, real_rainfall]):
                    plt.subplot(2, 2, plot_number + 1)
                    plt.imshow(im, cmap=cf.colormap)
                    if plot_number < 2:
                        plt.clim(0, clim_max)
                    plt.colorbar()
                    # we also plot the city location
                    for idx in range(city_data_df.index.__len__()):
                        x_loc = int(city_data_df.iloc[idx]['xid']) - 1
                        y_loc = int(city_data_df.iloc[idx]['yid']) - 1
                        cid = int(city_data_df.iloc[idx]['cid'])
                        plt.scatter(y_loc, x_loc, c='r', s=40)
                        plt.annotate(str(cid), xy=(y_loc, x_loc), color='white', fontsize=10)
                    # we also draw some contours
                    x = np.arange(0, im.shape[1], 1)
                    y = np.arange(0, im.shape[0], 1)
                    X, Y = np.meshgrid(x, y)
                    if plot_number == 3:
                        CS = plt.contour(X, Y, im, (4,), colors='white')
                        plt.clabel(CS, inline=1, fontsize=10)
                    else:
                        CS = plt.contour(X, Y, im, (15,), colors='k')
                        plt.clabel(CS, inline=1, fontsize=10)
                    plt.title(weather_names[plot_number])

                    # we plot every hour
                    crash_wind = False
                    crash_rainfall = False
                    for i in range(len(df)):
                        if i <= len(df) - 30:
                            plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                        else:
                            # check whether we have crashed!
                            if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                                crash_wind = True
                            elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                                crash_rainfall = True
                            if not crash_wind or crash_rainfall:
                                plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='orange', s=1)
                            else:
                                plt.scatter(df.iloc[i]['yid'], df.iloc[i]['xid'], c='red', s=1)

                plt.waitforbuttonpress(1)
                crash_wind = False
                crash_rainfall = False
                for i in range(max(0, len(df) - 30), len(df)):
                    # check whether we have crashed!
                    min = int(df.iloc[i]['time'][3:])
                    if real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 15:
                        crash_wind = True
                        plt.suptitle('Crash on wind:  Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f' %
                                     (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, real_wind[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1],mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                        plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)
                    elif real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1] >= 4.0:
                        crash_rainfall = True
                        plt.suptitle('Crash on rainfall: Day: %d, city: %d, starting hour: %d, hour: %d, min: %d, xid: %d, yid : %d with cost: %.2f, mean cost: %2.f'
                            % (day, goal_city, starting_hour, hour_prev_step, min, df.iloc[i]['yid'] - 1,
                               df.iloc[i]['xid'] - 1, real_rainfall[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1],
                               mean_weather[df.iloc[i]['xid'] - 1, df.iloc[i]['yid'] - 1]))
                        plt.scatter(df.iloc[i]['yid'] - 1, df.iloc[i]['xid'] - 1, c='r', s=10)

                    if crash_wind or crash_rainfall:
                        plt.waitforbuttonpress(0)
                        break


def get_wind_rainfall(cf, d_unique, h_unique):
    # left is the real
    # we plot wind
    print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
    total_wind = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))
    total_rainfall = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))

    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, 'real_wind_day_%d_hour_%d.npy' % (d_unique, h_unique)))
    total_wind[:, :, 0] = wind_real_day_hour
    for m_unique in cf.model_number:
        np_file = os.path.join(cf.wind_save_path,'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
        total_wind[:, :, m_unique] = np.load(np_file)

    # We plot rainfall
    rainfall_real_day_hour = np.load(os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)))
    total_rainfall[:, :, 0] = rainfall_real_day_hour
    for m_unique in cf.model_unique:
        np_file = os.path.join(cf.rainfall_save_path, 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (m_unique, d_unique, h_unique))
        total_rainfall[:, :, m_unique] = np.load(np_file)

    # we have maximum of 30
    # total_weather = np.minimum(total_weather, 30)

    real_weather = np.minimum(np.maximum(wind_real_day_hour, rainfall_real_day_hour * 15. / 4), 30)
    mean_weather = np.maximum(np.mean(total_wind[:, :, 1:], axis=2), np.mean(total_rainfall[:, :, 1:], axis=2) * 15. / 4)

    real_wind = wind_real_day_hour
    real_rainfall = rainfall_real_day_hour

    return real_weather, mean_weather, real_wind, real_rainfall



