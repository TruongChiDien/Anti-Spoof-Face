import pandas as pd
import numpy as np
from glob import glob
import os
import argparse
import cv2
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.default_config import load_config


def load_model(conf):
    model = AntiSpoofPredict(conf)
    return model

def make_prediction(args):
    if not os.path.exists(args.video_dirs):
        raise Exception(f'Not found directory {args.video_dirs}')
    videos = glob(os.path.join(args.video_dirs, '*'))
    video_names = []
    predictions = []
    conf = load_config(args.config)
    model = load_model(conf)
    for video in videos:
        video_name = os.path.basename(video)
        video_names.append(video_name)
        prediction = []
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_FPS, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prediction.append(predict(model, frame, conf))

        confidence_score = round(sum(prediction)/len(prediction), 5)
        predictions.append(confidence_score)

    df = pd.DataFrame(columns=['fname', 'liveness_score'])
    df['fname'] = video_names
    df['liveness_score'] = predictions
    df.to_csv(args.output)
        


def predict(model, image, conf):
    image_cropper = CropImage()
    param = {
        "org_img": image,
        "scale": conf.scale,
        "out_w": conf.w_input,
        "out_h": conf.h_input,
        "crop": True,
    }
    if conf.scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    prediction = model.predict(img, conf)
    return prediction[1]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dirs', type=str, default='public')
    parser.add_argument('--output', type=str, default='predict.csv')
    parser.add_argument('--config', type=str, default='configs/test_config.yaml')

    args = parser.parse_args()

    make_prediction(args)