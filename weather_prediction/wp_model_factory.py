
from weather_prediction.wp_pytorch_predictModels import *
import torch
from torchvision import models
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle
import json



class Model_Factory_Predict_Weather():
    def __init__(self, cf):
        # init a weather prediction model
        self.model_name = cf.wp_model_name
        if cf.wp_model_name == 'fully_connected_model':
            self.net = fully_connected_model(wp_fc_input_dim=cf.wp_fc_input_dim,
                                             wp_fc_nonlinear=cf.wp_fc_nonlinear,
                                             cuda=cf.cuda)

        # Set the loss criterion
        if cf.loss == 'L1Loss':
            self.crit = nn.L1Loss()
        elif cf.loss == 'MSELoss':
            self.crit = nn.MSELoss()
        elif cf.loss == 'SmoothL1Loss':
            self.crit = nn.SmoothL1Loss()

        # set data type
        self.net.float()
        if cf.cuda and torch.cuda.is_available():
            print('Using cuda')
            self.net = self.net.cuda()
            self.crit = self.crit.cuda()

        # set path and logfile
        self.exp_dir = cf.savepath + '_' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S') + '_' + cf.model_name
        os.mkdir(self.exp_dir)
        # Enable log file
        self.log_file = os.path.join(self.exp_dir, "logfile.log")
        sys.stdout = Logger(self.log_file)