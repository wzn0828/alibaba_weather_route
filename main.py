import imp
import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from config.configuration import Configuration
from tools.utils import HMS, configurationPATH

from tools.visualisation import plot_real_wind, plt_forecast_wind_train, plt_forecast_wind_test, plot_all_wind, plt_forecast_wind_test_multiprocessing,plt_forecast_wind_train_multiprocessing, plot_real_wind_multiprocessing, plot_all_wind_new, plot_all_rainfall, evaluation_plot_multi, plot_wind_with_rainfall
from tools.A_star_alibaba import A_star_2d_hourly_update_route, A_star_search_3D, A_star_search_3D_multiprocessing, A_star_search_3D_multiprocessing_multicost, A_star_fix_missing, A_star_search_3D_multiprocessing_rainfall_wind

from tools.simpleSub import submit_phase, collect_csv_for_submission_fraction
from tools.evaluation import evaluation, evaluation_plot, evaluation_with_rainfall
from tools.RL_alibaba import reinforcement_learning_solution, reinforcement_learning_solution_multiprocessing, reinforcement_learning_solution_new
from weather_prediction.wp_predict_weather import wp_predict_weather
from tools.Assignment_for_A_star_route import assignment_for_A_star_route, assignment_for_A_star_route_10min
from tools.Assignment_for_A_star_route_min import assignment_for_A_star_route_min


def process(cf):
    ### Following is the plotting alogrithm #############
    if cf.plot_real_wind:
        print('plot_real_wind')
        plot_real_wind(cf)
    if cf.plot_real_wind_multiprocessing:
        print('plot_real_wind_multiprocessing')
        plot_real_wind_multiprocessing(cf)
    if cf.plt_forecast_wind_train:
        print('plot_forecast_wind_train')
        plt_forecast_wind_train(cf)
    if cf.plt_forecast_wind_train_multiprocessing:
        print('plt_forecast_wind_train_multiprocessing')
        plt_forecast_wind_train_multiprocessing(cf)
    if cf.plt_forecast_wind_test:
        print('plt_forecast_wind_test')
        plt_forecast_wind_test(cf)
    if cf.plt_forecast_wind_test_multiprocessing:
        print('plt_forecast_wind_test_multiprocessing')
        plt_forecast_wind_test_multiprocessing(cf)
    if cf.plot_all_wind:
        print('Draw weather')
        plot_all_wind(cf)

    if cf.plot_all_wind_new:
        print('Draw weather: wind')
        plot_all_wind_new(cf)
    if cf.plot_all_rainfall:
        print('Draw weather: rainfall')
        plot_all_rainfall(cf)

    if cf.plot_wind_with_rainfall:
        print('plot_wind_with_rainfall')
        plot_wind_with_rainfall(cf)

    ### Following is the A Star alogrithm #############
    if cf.A_star_search_2D:
        print('A_star_search_2D')
        A_star_2d_hourly_update_route(cf)

    # This is one of the core algorithm
    if cf.A_star_search_3D:
        print('A_star_search_3D')
        A_star_search_3D(cf)

    if cf.A_star_search_3D_multiprocessing:
        print('A_star_search_3D_multiprocessing')
        A_star_search_3D_multiprocessing(cf)

    if cf.A_star_search_3D_multiprocessing_multicost:
        print('A_star_search_3D_multiprocessing')
        A_star_search_3D_multiprocessing_multicost(cf)

    if cf.A_star_search_3D_multiprocessing_rainfall_wind:
        print('A_star_search_3D_multiprocessing')
        A_star_search_3D_multiprocessing_rainfall_wind(cf)

    if cf.A_star_fix_missing:
        print('A_star_fix_missing')
        A_star_fix_missing(cf)

    ### Following is the RL alogrithm #############
    if cf.reinforcement_learning_solution:
        print('reinforcement_learning_solution')
        reinforcement_learning_solution(cf)

    if cf.reinforcement_learning_solution_new:
        print('reinforcement_learning_solution_new')
        reinforcement_learning_solution_new(cf)

    if cf.reinforcement_learning_solution_multiprocessing:
        print("reinforcement_learning_solution_multiprocessing")
        reinforcement_learning_solution_multiprocessing(cf)

    ### Following is the submission script #############
    if cf.submission_dummy:
        print("submission")
        submit_phase(cf)

    if cf.collect_csv_for_submission_fraction:
        print('collect_csv_for_submission_fraction')
        collect_csv_for_submission_fraction(cf)

    ### Following is the evaluation script #############
    if cf.evaluation:
        print('evaluation')
        total_penalty, crash_time_stamp, average_wind, max_wind = evaluation(cf, cf.csv_for_evaluation)

        print(int(np.sum(np.sum(total_penalty))))
        print(total_penalty.astype('int'))
        print(crash_time_stamp.astype('int'))
        np.set_printoptions(precision=2)
        print(average_wind)
        print(max_wind)
        print(np.sum(total_penalty.astype('int') == 1440))

    if cf.evaluation_plot:
        print('evaluation_plot')
        evaluation_plot(cf)
    if cf.evaluation_plot_multi:
        print('evaluation_plot_multi')
        evaluation_plot_multi(cf)
    if cf.evaluation_with_rainfall:
        print('evaluation_with_rainfall')
        total_penalty, crash_time_stamp, average_wind, max_wind, average_rain, max_rain = evaluation_with_rainfall(cf)
        print(int(np.sum(np.sum(total_penalty))))
        print(total_penalty.astype('int'))
        print(crash_time_stamp.astype('int'))
        np.set_printoptions(precision=2)
        print(average_wind)
        print(max_wind)
        print(average_rain)
        print(max_rain)
        print(np.sum(total_penalty.astype('int') == 1440))


    ### weather prediction
    if cf.wp_predict_weather:
        print('weather: predict weather data')
        wp_predict_weather(cf)

    #### assignment algorithm #############
    if cf.assignment_for_A_star_route:
        print('assignment_for_A_star_route')
        assignment_for_A_star_route(cf)

    if cf.assignment_for_A_star_route_10min:
        print('assignment_for_A_star_route_10min')
        assignment_for_A_star_route_10min(cf)

    if cf.assignment_for_A_star_route_min:
        print('assignment_for_A_star_route')
        assignment_for_A_star_route_min(cf)


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='./config/diwu_rematch.py', help='Configuration file')

    arguments = parser.parse_args()
    assert arguments.config_path is not None, 'Please provide a path using -c config/pathname in the command line'
    print('\n > Start Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    start_time = time.time()
    # Define the user paths

    # Load configuration files
    configuration = Configuration(arguments.config_path)
    cf = configuration.load()
    configurationPATH(cf)

    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))


if __name__ == "__main__":
    main()
