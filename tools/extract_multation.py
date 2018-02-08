import numpy as np
# from tools.A_star_alibaba import extract_weather_data


def extract_multation(weather, radius):
    shape = weather.shape
    multation = np.zeros(shape)
    expanded_weather = expand_array(weather, radius)

    for row in range(shape[0]):
        row_num = row + radius
        for column in range(shape[1]):
            column_num = column + radius
            local_weather = expanded_weather[row_num-radius: row_num+radius+1, column_num-radius: column_num+radius+1]
            multation[row, column] = local_weather.max() - weather[row, column]

    return multation

def expand_array(weather, radius):

    weather_expanded = np.zeros((weather.shape[0] + 2*radius, weather.shape[1] + 2*radius))

    weather_expanded[radius:-radius, radius:-radius] = weather
    weather_expanded[0:radius, radius:-radius] = weather[0, :]
    weather_expanded[-radius:, radius:-radius] = weather[-1, :]
    weather_expanded[radius:-radius, 0:radius] = weather[:, 0][np.newaxis].T
    weather_expanded[radius:-radius, -radius:] = weather[:, -1][np.newaxis].T
    weather_expanded[0:radius, 0:radius] = weather[0, 0]
    weather_expanded[0:radius, -radius:] = weather[0, -1]
    weather_expanded[-radius:, 0:radius] = weather[-1, 0]
    weather_expanded[-radius:, -radius:] = weather[-1, -1]
    
    return weather_expanded