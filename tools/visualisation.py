import os
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_real_wind(cf, draw=False):

    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'In-situMeasurementforTraining_20171124_2.csv'))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    date_unique = pd.unique(wind_real_df['date_id'])
    hour_unique = pd.unique(wind_real_df['hour'])
    wind_unique = pd.unique(wind_real_df['wind'])

    for d_unique in date_unique:
        for h_unique in hour_unique:
            print('Processing real data for date: %d, hour: %d' % (d_unique, h_unique))
            wind_real_df_day = wind_real_df.loc[wind_real_df['date_id'] == d_unique]
            wind_real_df_day_hour = wind_real_df_day.loc[wind_real_df_day['hour'] == h_unique]

            if not len(x_unique) * len(y_unique) == wind_real_df_day_hour.index.__len__():
                print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                      (len(x_unique) * len(y_unique), wind_real_df_day_hour.index.__len__()))

            wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
            for idx in range(wind_real_df_day_hour.index.__len__()):
                x_loc = int(wind_real_df_day_hour.iloc[idx]['xid']) - 1
                y_loc = int(wind_real_df_day_hour.iloc[idx]['yid']) - 1
                wind = int(wind_real_df_day_hour.iloc[idx]['wind'])
                wind_real_day_hour[x_loc, y_loc] = wind

            np.save(os.path.join(cf.savepath, 'real_wind_day_%d_hour_%d.np'%(d_unique, h_unique)),wind_real_day_hour)

            if draw:
                # plot figures here
                plt.close()
                plt.figure()
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
                x = np.arange(0, len(y_unique), 1)
                y = np.arange(0, len(x_unique), 1)
                X, Y = np.meshgrid(x, y)
                CS = plt.contour(X, Y, wind_real_day_hour,  (15,), colors='k')

                plt.clabel(CS, inline=1, fontsize=10)
                plt.title('real_wind_day_%d_hour_%d'%(d_unique, h_unique))

                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.waitforbuttonpress(0.01)
                plt.savefig(os.path.join(cf.fig_save_path, 'real_wind_day_%d_hour_%d.png'%(d_unique, h_unique)), dpi=74, bbox_inches='tight')


def plt_forecast_wind_train(cf, draw=False):
    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'ForecastDataforTraining_20171124.csv'))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    model_unique = pd.unique(wind_real_df['model'])
    wind_unique = pd.unique(wind_real_df['wind'])

    for m_unique in model_unique:
        wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
        date_unique = sorted(pd.unique(wind_real_df_model['date_id']))
        for d_unique in date_unique:
            wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
            hour_unique = sorted(pd.unique(wind_real_df_model_day['hour']))
            for h_unique in hour_unique:
                print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))
                wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]

                if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
                    print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                          (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

                wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
                for idx in range(wind_real_df_model_day_hour.index.__len__()):
                    x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
                    y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
                    wind = int(wind_real_df_model_day_hour.iloc[idx]['wind'])
                    wind_real_day_hour[x_loc, y_loc] = wind

                np.save(os.path.join(cf.savepath, 'forecast_wind_model_%d_day_%d_hour_%d.np' % (m_unique, d_unique, h_unique)),
                        wind_real_day_hour)

                if draw:
                    # plot figures here
                    plt.close()
                    plt.figure()
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
                    x = np.arange(0, len(y_unique), 1)
                    y = np.arange(0, len(x_unique), 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title('forecast_wind_model_%d_day_%d_hour_%d' % (m_unique, d_unique, h_unique))

                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.waitforbuttonpress(0.01)
                    plt.savefig(os.path.join(cf.fig_save_path, 'forecast_wind_model_%d_day_%d_hour_%d.png' %
                                             (m_unique, d_unique, h_unique)), dpi=74, bbox_inches='tight')


def plt_forecast_wind_test(cf, draw=False):
    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))
    wind_real_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'ForecastDataforTesting_20171124.csv'))

    x_unique = pd.unique(wind_real_df['xid'])
    y_unique = pd.unique(wind_real_df['yid'])
    model_unique = pd.unique(wind_real_df['model'])

    for m_unique in model_unique:
        wind_real_df_model = wind_real_df.loc[wind_real_df['model'] == m_unique]
        date_unique = sorted(pd.unique(wind_real_df_model['date_id']))
        for d_unique in date_unique:
            wind_real_df_model_day = wind_real_df_model.loc[wind_real_df_model['date_id'] == d_unique]
            hour_unique = sorted(pd.unique(wind_real_df_model_day['hour']))
            for h_unique in hour_unique:
                print('Processing forecast data for model: %d,  date: %d, hour: %d' % (m_unique, d_unique, h_unique))
                wind_real_df_model_day_hour = wind_real_df_model_day.loc[wind_real_df_model_day['hour'] == h_unique]

                if not len(x_unique) * len(y_unique) == wind_real_df_model_day_hour.index.__len__():
                    print('There are some missing data or redudant data: pixel range: %d, given wind pixel range: %d.' %
                          (len(x_unique) * len(y_unique), wind_real_df_model_day_hour.index.__len__()))

                wind_real_day_hour = np.zeros(shape=(len(x_unique), len(y_unique)))
                for idx in range(wind_real_df_model_day_hour.index.__len__()):
                    x_loc = int(wind_real_df_model_day_hour.iloc[idx]['xid']) - 1
                    y_loc = int(wind_real_df_model_day_hour.iloc[idx]['yid']) - 1
                    wind = int(wind_real_df_model_day_hour.iloc[idx]['wind'])
                    wind_real_day_hour[x_loc, y_loc] = wind

                np.save(os.path.join(cf.savepath, 'Test_forecast_wind_model_%d_day_%d_hour_%d.np' %
                                     (m_unique, d_unique, h_unique)), wind_real_day_hour)
                if draw:
                    # plot figures here
                    plt.close()
                    plt.figure()
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
                    x = np.arange(0, len(y_unique), 1)
                    y = np.arange(0, len(x_unique), 1)
                    X, Y = np.meshgrid(x, y)
                    CS = plt.contour(X, Y, wind_real_day_hour, (15,), colors='k')

                    plt.clabel(CS, inline=1, fontsize=10)
                    plt.title('Test_forecast_wind_model_%d_day_%d_hour_%d' % (m_unique, d_unique, h_unique))

                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.waitforbuttonpress(0.01)
                    plt.savefig(os.path.join(cf.fig_save_path, 'Test_forecast_wind_model_%d_day_%d_hour_%d.png' %
                                             (m_unique, d_unique, h_unique)), dpi=74, bbox_inches='tight')


def plot_all_wind(cf):

    # Create the data generators
    city_data_df = pd.read_csv(os.path.join(cf.dataroot_dir, 'CityData.csv'))

    weather_list = sorted(os.listdir(cf.savepath), reverse=True)
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
