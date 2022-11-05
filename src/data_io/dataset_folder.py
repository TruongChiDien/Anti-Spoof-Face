import cv2
import torch
from torchvision import datasets
import numpy as np
import os


def opencv_loader(path):
    img = cv2.imread(path)
    return img


class DatasetFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=opencv_loader):
        super(DatasetFolder, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            print('image is None --> ', path)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, org_path, ft_path, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(org_path, transform, target_transform, loader)
        if not os.path.exists(ft_path):
            raise Exception('Not found Fourier Transformed data')
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.transform = transform
        self.ft_path = ft_path

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # generate the FT picture of the sample
        sample_name = os.path.splitext(path)[0]
        ft_sample_path = os.path.join(self.ft_root, target, sample_name + '.npy')
        ft_sample = np.load(ft_sample_path)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, ft_sample, target