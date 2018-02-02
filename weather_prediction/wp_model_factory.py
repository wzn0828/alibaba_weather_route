
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
        if cf.loss == 'MSELoss':
            self.crit = nn.MSELoss()
        elif cf.loss == 'SmoothL1Loss':
            self.crit = nn.SmoothL1Loss()
