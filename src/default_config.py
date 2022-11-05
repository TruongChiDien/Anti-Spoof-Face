import torch
from easydict import EasyDict
from src.utility import make_if_not_exist, get_scale_heigh_width, get_kernel
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config_js = yaml.safe_load(f)

    config = EasyDict(config_js)
    return config

def update_config(args):
    config_path = args.config

    conf = load_config(config_path)

    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.kernel_size = get_kernel(conf.h_input, conf.w_input)
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]

    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, conf.patch_info)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(conf.log_path)

    conf.model_path = "{}/{}_{}.pth".format(snapshot_dir, conf.patch_info, conf.model_type)

    return conf

def save_config(conf, save_path):
    with open(save_path, 'w') as f:
        for key in conf.keys():
            if type(conf[key]) is list:
                f.write(f'{key}:\n')
                for line in conf[key]:
                    f.write(f'- {line}\n')
            else:
                f.write(f'{key}: {conf[key]}\n')
