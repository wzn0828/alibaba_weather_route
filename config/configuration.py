import imp
import time
import os
from distutils.dir_util import copy_tree
import shutil


class Configuration():
    def __init__(self, config_path):

        self.config_path = config_path

    def load(self):
        # Load configuration file...
        print(self.config_path)
        cf = imp.load_source('config', self.config_path)

        return cf


