import cv2
import os
from pathlib import Path
from tqdm import tqdm
from glob import glob
import argparse
import pandas as pd

def make_data(args):
    label = pd.read_csv(os.path.join(args.root, 'label.csv'))
    list_video = glob(os.path.join(args.root, 'videos', '*'))

    idx_0, idx_1 = 0, 0
    for video in tqdm(list_video):
        video_name = os.path.basename(video)
        cls = label[label['fname'] == video_name]['liveness_score'].values[0]
        if cls == 0:
            idx_0 = extract_frame(video, cls, args.dest, idx_0, args.skip)
        else:
            idx_1 = extract_frame(video, cls, args.dest, idx_1, args.skip)



def extract_frame(video_path, cls, dest, idx, skip=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            file_path = os.path.join(dest, str(cls), str(idx) + '.jpg')
            cv2.imwrite(file_path, frame)
            idx += 1

        count += 1

    return idx

if __name__ == '__main__()':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='train')
    parser.add_argument('--dest', type=str, default='data')
    parser.add_argument('--skip', type=int, default=10)

    args = parser.parse_args()
    make_data(args)






    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % skip == 0:
            dst = os.path.join('train_frame', str(cls), str(index) + '.jpg')
            cv2.imwrite(dst, frame)
            index += 1
        
        count += 1
    
    if cls == 0:
        index_0 = index
    else:
        index_1 = index