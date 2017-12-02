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

        if cf.day_list[0] > 5:  # This is for submitting test file
            cf.exp_dir = os.path.join(cf.savepath, 'Test_' + cf.model_description + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        else:
            cf.exp_dir = os.path.join(cf.savepath, 'Train_' + cf.model_description + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        if not cf.evaluation:
            # Enable log file
            os.mkdir(cf.exp_dir)
            cf.log_file = os.path.join(cf.exp_dir, "logfile.log")
            sys.stdout = Logger(cf.log_file)
            # we print the configuration file here so that the configuration is traceable
            print(help(cf))

            cf.csv_file_name = os.path.join(cf.exp_dir, cf.model_description + '.csv')

        return cf


