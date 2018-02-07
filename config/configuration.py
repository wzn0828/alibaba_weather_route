import imp
import os, sys
from datetime import datetime


# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")  # , 0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Configuration():
    def __init__(self, config_path):

        self.config_path = config_path

    def load(self):
        # Load configuration file...
        print(self.config_path)
        cf = imp.load_source('config', self.config_path)

        if cf.A_star_search_3D_multiprocessing_multicost or cf.A_star_search_3D_multiprocessing_rainfall_wind:
            if cf.risky:
                cf.model_description += '_risky'
            elif cf.wind_exp:
                cf.model_description += '_wind_exp_mean_' + str(cf.wind_exp_mean) + '_std_' + str(cf.wind_exp_std)
            elif cf.costs_exponential:
                cf.model_description += '_costsExponential_' + str(cf.costs_exp_basenumber) + '_'
            elif cf.costs_sigmoid:
                cf.model_description += '_costsSigmoid_' + str(cf.costs_sig_speed_time) + '_' + str(cf.costs_sig_inter_speed) + '_'
            elif cf.conservative:
                cf.model_description += '_conservative'

            # if cf.wall_wind:
            #     cf.model_description += '_wall_wind_'+str(cf.wall_wind)
            if cf.use_real_weather:
                cf.model_description += '_use_real_weather'
            elif cf.model_number:
                cf.model_description += '_model_number_' + str(cf.model_number)

        if cf.reinforcement_learning_solution_multiprocessing:
            cf.model_description = 'reinforcement_learning_solution_multiprocessing'
            if cf.qLearning:
                cf.model_description += '_qLearning'
            elif cf.expected:
                cf.model_description += '_ExpectedSarsa'
            if cf.double:
                cf.model_description += '_Double'

            if cf.polynomial_alpha:
                cf.model_description += '_polynomial_alpha'

            if cf.costs_exponential:
                cf.model_description += '_costsExponential'
            elif cf.costs_sigmoid:
                cf.model_description += '_costsSigmoid'
            elif cf.conservative:
                cf.model_description += '_conservative'

        if cf.wp_predict_weather:
            cf.model_description = 'weather_prediction_' + cf.wp_model_name + '_' + cf.loss + '_'
            if cf.wp_model_name == 'fully_connected_model':
                cf.model_description += str(cf.wp_fc_input_dim) + '_' + cf.wp_fc_nonlinear


        if cf.day_list[0] > 5:  # This is for submitting test file
            cf.exp_dir = os.path.join(cf.savepath, 'Test_' + cf.model_description + '_' * 5 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            # ad_hoc for testing
            # cf.exp_dir = '/home/wzn/PycharmProjects/alibaba_weather_route/Experiments/Test_a_star_search_3D_costsSigmoid_4_14.5__model_number_[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]_____2018-02-06-09-56-56'
            cf.csv_file_name = os.path.join(cf.exp_dir, 'Test_' + cf.model_description + '.csv')
        elif cf.fully_convolutional_wind_pred:
            cf.exp_dir = os.path.join(cf.savepath, 'FCN_' + cf.model_name + '_' * 5 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        elif cf.wp_predict_weather:
            cf.exp_dir = os.path.join(cf.savepath, cf.model_description + '_' * 5 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        else:
            cf.exp_dir = os.path.join(cf.savepath, 'Train_' + cf.model_description + '_' * 5 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            cf.csv_file_name = os.path.join(cf.exp_dir, 'Train_' + cf.model_description + '.csv')

        if not cf.evaluation and not cf.plot_real_wind and not cf.plt_forecast_wind_train and not cf.plt_forecast_wind_test \
                and not cf.plt_forecast_wind_train_multiprocessing and not cf.plt_forecast_wind_test_multiprocessing \
                and not cf.reinforcement_learning_solution and not cf.evaluation_plot and \
                not cf.collect_csv_for_submission_fraction and not cf.A_star_fix_missing and not cf.reinforcement_learning_solution_new\
                and not cf.assignment_for_A_star_route and not cf.evaluation_with_rainfall and not cf.assignment_for_A_star_route_10min:
            # Enable log file
            os.mkdir(cf.exp_dir)
            cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
            sys.stdout = Logger(cf.log_file)
            # we print the configuration file here so that the configuration is traceable
            print(help(cf))

        return cf


