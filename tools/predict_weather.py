import numpy as np
import os

def generate_weather_data(cf):
    for day in range(1, 6):
        for hour in range(3, 21):
            wind_predict_day_hour_accumulative = []
            for model_number in cf.model_number:
                # we average the result
                weather_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number, day, hour)
                wind_real_day_hour_model = np.load(os.path.join(cf.wind_save_path, weather_name))
                wind_predict_day_hour_accumulative.append(wind_real_day_hour_model)
                wind_predict_day_hour_accumulative = np.asarray(wind_predict_day_hour_accumulative)

            real_weather_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
            wind_real_day_hour = np.load(os.path.join(cf.wind_save_path, real_weather_name))
