import argparse
import time
from datetime import datetime

from config.configuration import Configuration
from tools.utils import HMS, configurationPATH
from tools.visualisation import plot_real_wind, plt_forecast_wind_train, plt_forecast_wind_test, plot_all_wind
from tools.A_star_alibaba import A_star_2d_hourly_update_route
from tools.simpleSub import submit_phase


def process(cf):
    if cf.plot_real_wind:
        print('plot_real_wind')
        plot_real_wind(cf)
    if cf.plt_forecast_wind_train:
        print('plot_forecast_wind_train')
        plt_forecast_wind_train(cf)
    if cf.plt_forecast_wind_test:
        print('plt_forecast_wind_test')
        plt_forecast_wind_test(cf)
    if cf.draw_weather:
        print('Draw weather')
        plot_all_wind(cf)

    if cf.A_star_search:
        print('A_star_serach')
        A_star_2d_hourly_update_route(cf)

    if cf.submission_dummy:
        print("submission")
        submit_phase(cf)


def main():
    # Get parameters from arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_path', type=str, default='/home/stevenwudi/PycharmProjects/gitlab-u4188/config/diwu.py', help='Configuration file')

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

    # Train /test/predict with the network, depending on the configuration
    process(cf)

    # End Time
    end_time = time.time()
    print('\n > End Time:')
    print('   ' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S'))
    print('\n   ET: ' + HMS(end_time - start_time))


if __name__ == "__main__":
    main()
