import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from skimage import transform
from torchvision import transforms
import torch


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        target = target[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'target': torch.from_numpy(landmarks).float()}


class Dataset_Generators():
    """ Initially we use dataset"""

    def __init__(self, cf):
        self.cf = cf
        self.dataloader = {}
        # Load training set

        print('\n > Loading training, valid, test set')
        train_dataset = ImageDataGenerator(dataset_split='train', cf=cf,
                                           transform=transforms.Compose([
                                               RandomCrop(cf.random_crop),
                                               ToTensor()
                                           ])
                                           )
        val_dataset = ImageDataGenerator(dataset_split='valid', cf=cf,
                                         transform=transforms.Compose([
                                             RandomCrop(cf.random_crop_valid),
                                             ToTensor()]))
        self.dataloader['train'] = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, pin_memory=True)
        self.dataloader['valid'] = DataLoader(val_dataset, batch_size=1, pin_memory=True)


class ImageDataGenerator(Dataset):
    def __init__(self, dataset_split, cf, transform=None):
        # We use the first four days weather as training and the last one day as validation
        self.transform = transform
        self.image_files, self.target_files = [], []
        if dataset_split == 'train':
            days = cf.train_days
        else:
            days = cf.valid_days

        for day in days:
            for hour in range(cf.hour_unique[0], cf.hour_unique[1]+1):
                for model_number in range(0, 10):
                    # Now we are not reading images, we are reading from NP file, hence, the range(1, 11)
                    img_name = 'Train_forecast_wind_model_%d_day_%d_hour_%d.npy' % (model_number+1, day, hour)
                    self.image_files.append(os.path.join(cf.wind_save_path, img_name))
                target_name = 'real_wind_day_%d_hour_%d.npy' % (day, hour)
                self.target_files.append(os.path.join(cf.wind_save_path, target_name))

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self, i):
        # Load images and perform augmentations with PIL
        wind_real_day_hour_temp = []
        for model_number in range(0, 10):
            wind_real_day_hour_model = np.load(self.image_files[i*10 + model_number])
            wind_real_day_hour_temp.append(wind_real_day_hour_model)
            # PyTorch require the last channel right> w * h * 10
        image = np.asarray(wind_real_day_hour_temp).transpose(1, 2, 0)
        target = np.load(self.target_files[i])
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)
        return sample


class Dataset_Generators_no_crop():
    """ Initially we use dataset"""

    def __init__(self, cf):
        self.cf = cf
        self.dataloader = {}
        # Load training set

        print('\n > Loading training, valid, test set')
        train_dataset = ImageDataGenerator(dataset_split='train', cf=cf,
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ])
                                           )
        val_dataset = ImageDataGenerator(dataset_split='valid', cf=cf,
                                         transform=transforms.Compose([
                                             ToTensor()]))
        self.dataloader['train'] = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, pin_memory=True)
        self.dataloader['valid'] = DataLoader(val_dataset, batch_size=1, pin_memory=True)


