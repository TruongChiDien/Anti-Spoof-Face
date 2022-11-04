from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_scale_heigh_width(patch_info):
    pieces = patch_info.split('_')
    scale, HxW = int(pieces[-2]), pieces[-1]
    h_input, w_input = map(int, HxW.split('x'))
    return scale,h_input,w_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
