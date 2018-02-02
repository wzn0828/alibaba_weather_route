from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class fully_connected_model(nn.Module):
    """
    This model use fully connected layers to predicted models.
    """

    def __init__(self, wp_fc_input_dim, wp_fc_nonlinear, cuda=True):
        '''
        :param wp_fc_input_dim: the dim of input
        :param wp_fc_nonlinear: the nonlinear layer type: ReLU or Tanh or else
        :param cuda: whether or not to use GPU
        '''
        super(fully_connected_model, self).__init__()
        # fully connected part
        self.linear1 = nn.Linear(wp_fc_input_dim, 90)
        self.linear2 = nn.Linear(90, 40)
        self.linear3 = nn.Linear(40, 1)
        if wp_fc_nonlinear == 'Tanh':
            self.nonlinear1 = self.nonlinear2 = nn.Tanh()
        else:
            self.nonlinear1 = self.nonlinear2 = nn.ReLU()

        # cpu or gpu
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, wp_model_weather_datas):
        '''
        :param wp_model_weather_datas: Train data or test data, type is Variable, size is (batchSize, *, wp_fc_input_dim).
        :return: predicted weather data, type is Variable, Size is (batchSize, *, 1)
        '''

        # fully connected part:
        output_linear1 = self.linear1(wp_model_weather_datas)
        output_nonlinear1 = self.nonlinear1(output_linear1)
        output_linear2 = self.linear2(output_nonlinear1)
        output_nonlinear2 = self.nonlinear2(output_linear2)
        output_linear3 = self.linear3(output_nonlinear2)

        return output_linear3