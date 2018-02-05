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
                for m_unique in cf.model_unique:
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
                save_fig_name = os.path.join(cf.fig_wind_save_train_path, '%s.png' % ('Train_real_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
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
                total_weather_max = np.max(total_weather)
                total_weather = (total_weather - 15) / total_weather_max

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
                    CS = plt.contour(X, Y, total_weather[:, :, m_unique-1], (0,), colors='k')
                    plt.clabel(CS, inline=1, fontsize=10)

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                fig.suptitle('Test_forecast_wind_day_%d_hour_%d.npy' % (d_unique, h_unique), size=20)
                plt.waitforbuttonpress(0.0001)
                save_fig_name = os.path.join(cf.fig_wind_save_test_path, '%s.png' % ('Test_models_wind_day_%d_hour_%d' % (d_unique, h_unique)))
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
    total_weather = np.zeros(shape=(cf.grid_world_shape[0], cf.grid_world_shape[1], len(cf.model_unique) + 1))
    for d_unique in [1, 2, 3, 4, 5]:
        for h_unique in range(cf.hour_unique[0], cf.hour_unique[1] + 1):
            print('Processing forecast data for  date: %d, hour: %d' % (d_unique, h_unique))
            rainfall_real_day_hour = np.load(
                os.path.join(cf.rainfall_save_path, 'real_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique)))
            total_weather[:, :, 0] = rainfall_real_day_hour
            for m_unique in cf.model_unique:
                np_file = os.path.join(cf.rainfall_save_path, 'Train_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (
                m_unique, d_unique, h_unique))
                total_weather[:, :, m_unique] = np.load(np_file)

            # We need to normalise the map to standardize visualisation
            total_weather_max = np.max(total_weather)
            total_weather = (total_weather - 4) / total_weather_max
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
                x = np.arange(0, rainfall_real_day_hour.shape[1], 1)
                y = np.arange(0, rainfall_real_day_hour.shape[0], 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, total_weather[:, :, m_unique], (0,), colors='k')
                plt.clabel(CS, inline=1, fontsize=10)
                plt.title('Model %d' % m_unique)

            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.waitforbuttonpress(0.0001)
            # plt.show()
            save_fig_name = os.path.join(cf.fig_rainfall_save_train_path,
                                         '%s.png' % ('Train_real_models_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))
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
                np_file = os.path.join(cf.rainfall_save_path, 'Test_forecast_rainfall_model_%d_day_%d_hour_%d.npy' % (
                    m_unique, d_unique, h_unique))
                total_weather[:, :, m_unique - 1] = np.load(np_file)

            # We need to normalise the map to standardize visualisation
            fig.clf()
            total_weather_max = np.max(total_weather)
            total_weather = (total_weather - 4) / total_weather_max

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
            fig.suptitle('Test_forecast_rainfall_day_%d_hour_%d.npy' % (d_unique, h_unique), size=20)
            plt.waitforbuttonpress(0.0001)
            save_fig_name = os.path.join(cf.fig_rainfall_save_test_path,
                                         '%s.png' % ('Test_models_rainfall_day_%d_hour_%d' % (d_unique, h_unique)))
            plt.savefig(save_fig_name, dpi=74, bbox_inches='tight')