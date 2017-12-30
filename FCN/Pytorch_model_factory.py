import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
from FCN.Pytorch_fcn import FeatureResNet, SegResNet
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


# Build the model
class Model_Factory_semantic_seg():
    def __init__(self, cf):

        self.model_name = cf.model_name
        self.cf = cf
        if cf.model_name == 'segnet_basic':
            forward_net = FeatureResNet()
            self.net = SegResNet(forward_net).cuda()
        elif cf.model_name == 'drn_c_26':
            print('Not implemented')
            pass

        self.crit = nn.MSELoss().cuda()

        # Construct optimiser
        if cf.train_model_path:
            print('Not implemented')
            pass

        # self.net = DRNSegF(self.net, 20)
        params_dict = dict(self.net.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'bn' in key:
                # No weight decay on batch norm
                params += [{'params': [value], 'weight_decay': 0}]
            elif '.bias' in key:
                # No weight decay plus double learning rate on biases
                params += [{'params': [value], 'lr': 2 * cf.learning_rate, 'weight_decay': 0}]
            else:
                params += [{'params': [value]}]
        if cf.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(params, lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)
        elif cf.optimizer == 'sgd':
            self.optimizer = optim.SGD(params, lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)
        elif cf.optimizer == 'adam':
            self.optimizer = optim.Adam(params, lr=cf.learning_rate, weight_decay=cf.weight_decay)

        self.scores, self.mean_scores = [], []

        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def train(self, train_loader, epoch):
        # TODO: write adjust learning rate
        #lr = self.adjust_learning_rate(self.cf.learning_rate, self.optimiser, epoch)
        lr = self.cf.learning_rate
        print('learning rate:', lr)
        self.net.train()
        for i, sample in enumerate(train_loader):
            self.optimizer.zero_grad()
            input, target = Variable(sample['image'].cuda(async=True), requires_grad=False), Variable(sample['target'].cuda(async=True), requires_grad=False)
            output = self.net(input)
            self.loss = self.crit(output, target)
            print(epoch, i, self.loss.data[0])
            self.loss.backward()
            self.optimizer.step()

    def test(self, val_loader, epoch, cf):
        self.net.eval()
        total_mse = []
        for i, (input, target) in enumerate(val_loader):
            input, target = Variable(input.cuda(async=True), volatile=True), Variable(target.cuda(async=True), volatile=True)
            output = self.net(input)
            # TODO: write error of predicting wind >= 15
            total_mse.append(self.crit(output, target))

        if False:
            image = np.squeeze(input.data.cpu().numpy())
            image[0, :, :] = image[0, :, :] * cf.rgb_std[0] + cf.rgb_mean[0]
            image[1, :, :] = image[1, :, :] * cf.rgb_std[1] + cf.rgb_mean[1]
            image[2, :, :] = image[2, :, :] * cf.rgb_std[2] + cf.rgb_mean[2]
            pred_image = np.squeeze(pred.data.cpu().numpy())
            class_image = np.squeeze(target.data.cpu().numpy())
            plt.figure()
            plt.subplot(1,3,1);plt.imshow(image.transpose(1, 2, 0));plt.title('RGB')
            plt.subplot(1,3,2);plt.imshow(pred_image);plt.title('Prediction')
            plt.subplot(1,3,3);plt.imshow(class_image);plt.title('GT')
            plt.waitforbuttonpress(1)
            print('Training testing')

        # Calculate average IoU
        print(total_mse.mean())
        self.scores.append(total_mse)

        # Save weights and scores
        torch.save(self.net.state_dict(), os.path.join(cf.exp_dir, 'epoch_' + str(epoch) + '_' + 'mIOU:.%4f' % total_mse.mean() + '_net.pth'))

        # Plot scores
        self.mean_scores.append(total_mse.mean())
        es = list(range(len(self.mean_scores)))
        plt.switch_backend('agg')  # Allow plotting when running remotely
        plt.plot(es, self.mean_scores, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.savefig(os.path.join(self.exp_dir, 'ious.png'))
