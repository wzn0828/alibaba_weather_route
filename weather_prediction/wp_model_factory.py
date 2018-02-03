
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

def adjust_learning_rate(lr, optimizer, epoch, decrease_epoch=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch > 0 and epoch%decrease_epoch == 0:
        lr = lr * (0.1 ** (epoch // decrease_epoch))
        if lr >= 1.0e-6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[-1]['lr']
    else:
        lr = optimizer.param_groups[-1]['lr']

    return lr


class Model_Factory_Predict_Weather():

    def __init__(self, cf):
        self.cf = cf
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

        # load from a pretrained model
        if cf.load_trained_model:
            print("Load from pretrained_model weight: " + cf.wp_trained_model_path)
            self.net.load_state_dict(torch.load(cf.wp_trained_model_path))

        # construct optimiser
        if cf.wp_optimizer == 'LBFGS':
            self.optimizer = optim.LBFGS(self.net.parameters(), lr=cf.wp_learning_rate)
        elif cf.wp_optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=cf.wp_learning_rate, weight_decay=cf.wp_weight_decay)
        elif cf.wp_optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=cf.wp_learning_rate, momentum=cf.wp_momentum,
                                           weight_decay=cf.wp_weight_decay)
        elif cf.wp_optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=cf.wp_learning_rate, momentum=cf.wp_momentum,
                                       weight_decay=cf.wp_weight_decay, nesterov=True)

    def train(self, cf, train_loader, epoch):
        # begin to train
        self.net.train()
        lr = adjust_learning_rate(self.cf.learning_rate, self.optimizer, epoch, decrease_epoch=cf.lr_decay_epoch)
        print('learning rate:', lr)

        if cf.optimizer == 'LBFGS':
            def closure():
                self.optimizer.zero_grad()
                out = self.net(*input)[0]
                loss = self.crit(out, train_target)
                if cf.cuda:
                    print('loss: ', loss.data.cpu().numpy()[0])
                else:
                    print('loss: ', loss.data.numpy()[0])
                loss.backward()
                return loss
            self.optimiser.step(closure)
        else:
        train_losses = []
        for i, (sementic, input_trajectory, target_trajectory) in enumerate(train_loader):
            self.optimiser.zero_grad()
            sementic, input_trajectory, target_trajectory = Variable(sementic.cuda(async=True), requires_grad=False), \
                                                            Variable(input_trajectory.cuda(async=True),
                                                                     requires_grad=False), \
                                                            Variable(target_trajectory.cuda(async=True),
                                                                     requires_grad=False)
            if cf.model_name == 'CNN_LSTM_To_FC' or cf.model_name == 'DropoutCNN_LSTM_To_FC':
                input = tuple([sementic, input_trajectory])
            else:
                input = tuple([input_trajectory])
            output = self.net(*input)[0]
            self.loss = self.crit(output, target_trajectory)
            train_losses.append(self.loss.data[0])
            self.loss.backward()
            self.optimiser.step()

        train_loss = np.array(train_losses).mean()
        print('Train Loss', epoch, train_loss)

        return train_loss, lastupdate_epoch

