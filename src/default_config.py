import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_scale_width_height, get_kernel
import yaml

def update_config(args):
    config_path = args.config

    with open(config_path) as f:
        config_js = yaml.safe_load(f)

    conf = EasyDict(config_js)

    scale, w_input, h_input = get_scale_width_height(conf.patch_info)
    conf.input_size = [h_input, w_input]
    conf.scale = scale
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    job_name = conf.patch_info
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, conf.patch_info)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(conf.log_path)

    conf.model_path = snapshot_dir
    conf.job_name = job_name

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
