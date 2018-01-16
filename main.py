import imp
import os, sys
import argparse
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from config.configuration import Configuration, Logger
from tools.utils import HMS, configurationPATH
from tools.visualisation import plot_real_wind, plt_forecast_wind_train, plt_forecast_wind_test, plot_all_wind, plt_forecast_wind_test_multiprocessing,plt_forecast_wind_train_multiprocessing
from tools.A_star_alibaba import A_star_2d_hourly_update_route, A_star_search_3D, A_star_search_3D_multiprocessing, A_star_search_3D_multiprocessing_multicost, A_star_fix_missing
from tools.simpleSub import submit_phase, collect_csv_for_submission_fraction
from tools.evaluation import evaluation, evaluation_plot
from tools.RL_alibaba import reinforcement_learning_solution, reinforcement_learning_solution_multiprocessing, reinforcement_learning_solution_new


def process(cf):
    ### Following is the plotting alogrithm #############
    if cf.plot_real_wind:
        print('plot_real_wind')
        plot_real_wind(cf)
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
        total_penalty = evaluation(cf, cf.csv_for_evaluation)
        print(int(np.sum(np.sum(total_penalty))))
        print(total_penalty.astype('int'))
        print(np.sum(total_penalty.astype('int') == 1440))

    if cf.evaluation_plot:
        print('evaluation_plot')
        evaluation_plot(cf)


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/wzn/PycharmProjects/alibaba_weather_route/config/wzn.py', help='Configuration file')

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

    for powertime in range(1, 6):
        cf.costs_exp_basenumber = 10**powertime

        cf.model_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        costs_method = "costsExponential_" + "baseNumber_" + str(cf.costs_exp_basenumber)
        cf.model_description = costs_method + "_model_mean_[1-10]"
        cf.exp_dir = os.path.join(cf.savepath, 'Train_' + cf.model_description + '_' * 5 + datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"))
        cf.csv_file_name = os.path.join(cf.exp_dir, 'Train_' + cf.model_description + '.csv')

        if not cf.evaluation and not cf.plot_real_wind and not cf.plt_forecast_wind_train and not cf.plt_forecast_wind_test \
                and not cf.plt_forecast_wind_train_multiprocessing and not cf.plt_forecast_wind_test_multiprocessing \
                and not cf.reinforcement_learning_solution and not cf.evaluation_plot and \
                not cf.collect_csv_for_submission_fraction and not cf.A_star_fix_missing and not cf.reinforcement_learning_solution_new:
            # Enable log file
            os.mkdir(cf.exp_dir)
            cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
            sys.stdout = Logger(cf.log_file)
            # we print the configuration file here so that the configuration is traceable
            print(help(cf))

        # mean
        process(cf)

        # Train /test/predict with the network, depending on the configuration
        for i in range(1, 11):
            cf.model_number = list([i])
            cf.model_description = costs_method + 'model_number_' + str(cf.model_number)
            cf.exp_dir = os.path.join(cf.savepath, 'Train_' + cf.model_description + '_' * 5 + datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S"))
            cf.csv_file_name = os.path.join(cf.exp_dir, 'Train_' + cf.model_description + '.csv')

            if not cf.evaluation and not cf.plot_real_wind and not cf.plt_forecast_wind_train and not cf.plt_forecast_wind_test \
                    and not cf.plt_forecast_wind_train_multiprocessing and not cf.plt_forecast_wind_test_multiprocessing \
                    and not cf.reinforcement_learning_solution and not cf.evaluation_plot and \
                    not cf.collect_csv_for_submission_fraction and not cf.A_star_fix_missing and not cf.reinforcement_learning_solution_new:
                # Enable log file
                os.mkdir(cf.exp_dir)
                cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
                sys.stdout = Logger(cf.log_file)
                # we print the configuration file here so that the configuration is traceable
                print(help(cf))

            process(cf)


    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))


if __name__ == "__main__":
    main()
