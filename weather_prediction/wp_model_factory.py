
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

        # if cf.optimizer == 'LBFGS':
        #     def closure():
        #         self.optimizer.zero_grad()
        #         out = self.net(*input)[0]
        #         loss = self.crit(out, train_target)
        #         if cf.cuda:
        #             print('loss: ', loss.data.cpu().numpy()[0])
        #         else:
        #             print('loss: ', loss.data.numpy()[0])
        #         loss.backward()
        #         return loss
        #     self.optimiser.step(closure)
        # else:
        train_losses = []
        for i, (input_samples, output_labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            input_samples, output_labels = Variable(input_samples.cuda(async=True), requires_grad=False), \
                                           Variable(output_labels.cuda(async=True), requires_grad=False)

            output_prediction = self.net(input_samples)
            self.loss = self.crit(output_prediction, output_labels)
            train_losses.append(self.loss.data[0])
            self.loss.backward()
            self.optimiser.step()

        train_loss = np.array(train_losses).mean()
        print('Train Loss', epoch, train_loss)

        return train_loss

    def test(self, cf, valid_loader, data_mean, data_std, epoch=None):
        self.net.eval()
        # for experiment: output predicted trajectory
        predicted_weather = []
        valid_losses = []
        MSEs = []
        for i, (input_samples, output_labels) in enumerate(valid_loader):
            # print(i)
            input_samples, output_labels = Variable(input_samples.cuda(async=True)), \
                                           Variable(output_labels.cuda(async=True))

            output_prediction = self.net(input_samples)
            # output predicted weather
            predicted_weather.append((output_prediction.data.cpu().numpy())*data_std + data_mean)

            # cal loss
            loss = self.crit(output_prediction, output_labels)
            valid_losses.append(loss.data[0])

            # eavluation
            MSEs.append((data_std**2) * nn.MSELoss(output_prediction, output_labels))

        # output predicted weather
        cf.predicted_weather_file_name = os.path.join(cf.exp_dir, 'Test_' + cf.model_description + '_predicted_weather.npy')
        np.save(cf.predicted_weather_file_name, np.array(predicted_weather))

        # loss mean
        self.loss = np.array(valid_losses).mean()

        # evaluation mean
        self.MSE = np.array(MSEs).mean()

        # Save weights and scores
        if epoch:
            print('############### VALID #############################################')
            print('Valid Loss:', epoch, self.loss)
            print('MSE: %.4f' % self.MSE)
            model_checkpoint = 'Epoch:%2d_MSE:%.4f.PTH' % (epoch, self.MSE)
            torch.save(self.net.state_dict(), os.path.join(self.exp_dir, model_checkpoint))
        else:
            print('############### TEST #############################################')
            print('Test Loss:', epoch, self.loss)
            print('MSE: %.4f' % self.MSE)

        return self.loss