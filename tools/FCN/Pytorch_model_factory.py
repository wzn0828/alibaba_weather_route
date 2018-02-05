import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
from matplotlib import pyplot as plt

from tools.FCN.Pytorch_fcn import FeatureResNet, SegResNet
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
            print('Loading model from: ' + cf.train_model_path)
            self.net.load_state_dict(torch.load(cf.train_model_path))

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

        self.mean_ious, self.mean_scores = [], []

        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def train(self, train_loader, epoch):
        # TODO: write adjust learning rate
        #lr = self.adjust_learning_rate(self.cf.learning_rate, self.optimiser, epoch)
        lr = self.cf.learning_rate
        # This has any effect only on modules such as Dropout or BatchNorm.
        self.net.train()
        for i, sample in enumerate(train_loader):
            #  zero the parameter gradients
            self.optimizer.zero_grad()
            input, target = Variable(sample['image'].cuda(async=True), requires_grad=False), Variable(sample['target'].cuda(async=True), requires_grad=False)
            output = self.net(input)
            self.loss = self.crit(output, target)
            self.loss.backward()
            self.optimizer.step()
        return lr, self.loss.data[0]

    def test(self, val_loader, epoch, cf):
        # This has any effect only on modules such as Dropout or BatchNorm.
        self.net.eval()
        total_mse = []
        total_strong_wind_iou = []
        for i, sample in enumerate(val_loader):
            input, target = Variable(sample['image'].cuda(async=True), requires_grad=False), Variable(sample['target'].cuda(async=True), requires_grad=False)
            output = self.net(input)
            total_mse.append(self.crit(output, target))
            total_strong_wind_iou.append(self.iou_strong_wind(output, target))

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
        total_mse = np.array(total_mse)
        total_mse_mean = total_mse.mean().cpu().data.numpy()
        total_strong_wind_iou_mean = np.array(total_strong_wind_iou).mean()
        print('MSE: %.4f, Strong wind IOU: %.4f' %(total_mse_mean, total_strong_wind_iou_mean))

        # Save weights and scores
        torch.save(self.net.state_dict(), os.path.join(cf.exp_dir, 'epoch_' + str(epoch) + '_' +
                                                       'mMSE:.%4f' % total_mse_mean + 'mIOU:.%4f' % total_strong_wind_iou_mean + '_net.pth'))

        # Plot scores
        self.mean_scores.append(total_mse_mean)
        self.mean_ious.append(total_strong_wind_iou_mean)

        es = list(range(len(self.mean_scores)))

        plt.close("all")
        plt.switch_backend('agg')
        fig, ax1 = plt.subplots()
        ax1.plot(es, self.mean_scores, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Square Error')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(es, self.mean_ious, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        plt.savefig(os.path.join(cf.exp_dir, 'MSE_IOU.png'))

    def iou_strong_wind(self, output, target):
        """
        Calcluate class intersection over unions
        :param output:
        :param target:
        :return:
        """

        pred_inds = output >= self.cf.wall_wind
        target_inds = target >= self.cf.wall_wind
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Ca  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            return 0  # If there is no ground truth, do not include in evaluation
        else:
            return intersection / max(union, 1)

    def collect_train_valid_mse_iou(self, DG):
        """
        Collect the MSE and IOU of the data
        :param DG:
        :return:
        """

        mse_all = []
        iou_all = []
        for i, sample in enumerate(DG.dataloader['train']):
            #  zero the parameter gradients
            images, target = sample['image'], sample['target']
            mse, iou = self.calculate_mse_iou(images, target)
            mse_all.append(mse)
            iou_all.append(iou)

        for i, sample in enumerate(DG.dataloader['valid']):
            #  zero the parameter gradients
            images, target = sample['image'], sample['target']
            mse, iou = self.calculate_mse_iou(images, target)
            mse_all.append(mse)
            iou_all.append(iou)

        return np.array(mse_all).mean(), np.array(iou_all).mean()

    def calculate_mse_iou(self, images, target):
        model_num = images.shape[1]
        mse_all = []
        iou_all = []
        for m in range(model_num):
            mse = (images[:, m] - target) **2
            mse_all.append(mse.mean())

            pred_inds = images[:, m] >= self.cf.wall_wind
            target_inds = target >= self.cf.wall_wind
            intersection = (pred_inds[target_inds]).long().sum()
            union = pred_inds.long().sum() + target_inds.long().sum() - intersection
            iou_all.append(intersection / max(union, 1))

        return np.array(mse_all).mean(), np.array(iou_all).mean()
