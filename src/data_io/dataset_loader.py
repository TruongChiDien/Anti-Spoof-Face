from torch.utils.data import DataLoader
from src.data_io.dataset_folder import DatasetFolderFT, DatasetFolder
from src.data_io import transform as trans
import os


def get_ft_loader(conf, set='train'):
    transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple([conf.h_input, conf.w_input]),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    
    org_path = os.path.join(conf.org_root, set)
    ft_path = os.path.join(conf.ft_root, set)
    dataset_ft = DatasetFolderFT(org_path, ft_path, transform,
                            None, conf.ft_width, conf.ft_height)
    
    loader_ft = DataLoader(
        dataset_ft,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)

    return loader_ft


def get_normal_loader(conf, set='train'):
    transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple([conf.h_input, conf.w_input]),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    

    org_path = os.path.join(conf.org_root, set)
    dataset = DatasetFolder(org_path, transform)
    
    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)

    return loader
