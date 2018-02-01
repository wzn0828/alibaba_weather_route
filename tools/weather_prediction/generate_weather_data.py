import numpy as np
import os
import pickle
from timeit import default_timer as timer
import multiprocessing


def wp_generate_weather_data_multiprocessing(cf):

    # # get idnexes
    indexes = generate_weather_indexes(cf)

    # generate data
    start_time = timer()
    multiprocessing.log_to_stderr()
    jobs = []
    for day_hour, locations in indexes.items():
        p = multiprocessing.Process(target=wp_generate_weather_data_worker, args=(cf, day_hour, locations))
        jobs.append(p)
        p.start()


    # waiting for the all the job to finish
    for j in jobs:
        j.join()


def wp_generate_weather_data_worker(cf, day_hour, locations):
    # get estimation value
    sampledata_onelayer = [np.zeros((len(cf.wp_used_model_number), 3, 3)) for n in range(len(locations))]
    sampledata_twolayer = [np.zeros((len(cf.wp_used_model_number), 5, 5)) for n in range(len(locations))]
    for model_idx, model_number in enumerate(cf.wp_used_model_number):
        # get modle weather
        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day_hour[0], day_hour[1])
        wind_model_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
        wind_model_day_hour = np.round(wind_model_day_hour, 2)
        wind_model_day_hour_expand_two_labs = expand_array(cf, wind_model_day_hour)
        # get location where wind speed are used
        for loc_idx, location in enumerate(locations):
            location = tuple(location)
            row_num = location[0] + 2
            column_num = location[1] + 2
            sampledata_onelayer[loc_idx][model_idx, :, :] = wind_model_day_hour_expand_two_labs[
                                                            row_num - 1:row_num + 2, column_num - 1:column_num + 2]
            sampledata_twolayer[loc_idx][model_idx, :, :] = wind_model_day_hour_expand_two_labs[
                                                            row_num - 2:row_num + 3, column_num - 2:column_num + 3]
    # get real weather data
    real_weather_name = 'real_wind_day_%d_hour_%d.npy' % (day_hour[0], day_hour[1])
    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, real_weather_name))
    wind_real_day_hour = np.round(wind_real_day_hour, 2)
    # save data
    for loc_idx, location in enumerate(locations):
        location = tuple(location)
        onelayerdatafilename = os.path.join(cf.wp_onelayer_data_path,
                                            '3x3_' + str(day_hour[0]) + '_' + str(day_hour[1]) + '_' + str(
                                                location) + '.npy')
        twolayerdatafilename = os.path.join(cf.wp_twolayer_data_path,
                                            '5x5_' + str(day_hour[0]) + '_' + str(day_hour[1]) + '_' + str(
                                                location) + '.npy')
        np.save(onelayerdatafilename, (sampledata_onelayer[loc_idx], wind_real_day_hour[location]))
        np.save(twolayerdatafilename, (sampledata_twolayer[loc_idx], wind_real_day_hour[location]))


def expand_array(cf, wind_model_day_hour):
    wind_model_day_hour_expand_two_labs = np.zeros((cf.grid_world_shape[0] + 4, cf.grid_world_shape[1] + 4))
    wind_model_day_hour_expand_two_labs[2:-2, 2:-2] = wind_model_day_hour
    wind_model_day_hour_expand_two_labs[0:2, 2:-2] = wind_model_day_hour[0, :]
    wind_model_day_hour_expand_two_labs[-2:, 2:-2] = wind_model_day_hour[-1, :]
    wind_model_day_hour_expand_two_labs[2:-2, 0:2] = wind_model_day_hour[:, 0][np.newaxis].T
    wind_model_day_hour_expand_two_labs[2:-2, -2:] = wind_model_day_hour[:, -1][np.newaxis].T
    wind_model_day_hour_expand_two_labs[0:2, 0:2] = wind_model_day_hour[0, 0]
    wind_model_day_hour_expand_two_labs[0:2, -2:] = wind_model_day_hour[0, -1]
    wind_model_day_hour_expand_two_labs[-2:, 0:2] = wind_model_day_hour[-1, 0]
    wind_model_day_hour_expand_two_labs[-2:, -2:] = wind_model_day_hour[-1, -1]
    return wind_model_day_hour_expand_two_labs


# generate indexes where wind speed if more than a low bound and lower than a high bound
def generate_weather_indexes(cf):
    index_day_hour = {}
    if cf.wp_generate_weather_indexes:
        for day in range(1, 6):
            for hour in range(3, 21):
                model_index = np.zeros(cf.grid_world_shape)
                # wind model data
                for model_number in cf.model_number:
                    # we average the result
                    weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                    wind_model_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
                    wind_model_day_hour = np.round(wind_model_day_hour, 2)
                    model_index += np.logical_and(13 <= wind_model_day_hour, wind_model_day_hour <= 17)
                # wind real data
                real_weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, real_weather_name))
                wind_real_day_hour = np.round(wind_real_day_hour, 2)
                # union set
                model_index += np.logical_and(13 <= wind_real_day_hour, wind_real_day_hour <= 17)

                # record
                index_day_hour[(day, hour)] = np.argwhere(model_index > 0)
        # save
        with open(cf.wp_wind_indexes, 'wb') as handle:
            pickle.dump(index_day_hour, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cf.wp_wind_indexes, 'rb') as handle:
            index_day_hour = pickle.load(handle)

    return index_day_hour


def test_error_ratio(cf, day, hour):
    cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    wind_real_day_hour_temp = []
    for model_number in cf.model_number:
        # we average the result
        weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
        wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
        wind_real_day_hour_model = np.round(wind_real_day_hour_model, 2)
        wind_real_day_hour_temp.append(wind_real_day_hour_model)
    wind_real_day_hour_temp = np.asarray(wind_real_day_hour_temp)
    wind_model_day_hour = np.mean(wind_real_day_hour_temp, axis=0)

    weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
    wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, weather_name))
    wind_real_day_hour = np.round(wind_real_day_hour, 2)
    # wind_real_day_hour[wind_real_day_hour==0] = 1e+8

    return (wind_real_day_hour-wind_model_day_hour)/wind_real_day_hour



