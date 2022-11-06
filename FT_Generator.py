import cv2
import numpy as np
import os
from glob import glob
import pathlib
import argparse
from tqdm import tqdm

def generate(args):
    cls_paths = glob(os.path.join(args.data_path, '*'))
    for cls_path in cls_paths:
        cls = cls_path.split(os.sep)[-1]
        src = os.path.join(args.data_path, cls)
        dst = os.path.join(args.ft_path, cls)
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(glob(os.path.join(src, '*'))):
            img_name = img_path.split(os.sep)[-1].split('.')[0]
            img = cv2.imread(img_path)
            fimg = generate_FT(img)
            np.save(os.path.join(dst, img_name + '.npy'), fimg)


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)    # Normalize
    return fimg

def parse_args():
    """parsing and configuration"""
    desc = "Generate Fourier Transform for dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--data_path", type=str, default="raw_data/org", help="path to dataset")
    parser.add_argument("--ft_path", type=str, default="raw_data/ft", help="path to save trasformed images")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate(args)