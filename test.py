# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.default_config import load_config
warnings.filterwarnings('ignore')



def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_path, conf):
    model_test = AntiSpoofPredict(conf.devices[0])
    image_cropper = CropImage()
    image = cv2.imread(image_path)
    # result = check_image(image)
    # if result is False:
    #     return
    image_bbox = model_test.get_bbox(image)
    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": conf.scale,
        "out_w": conf.w_input,
        "out_h": conf.h_input,
        "crop": True,
    }
    if conf.scale is None:
        param["crop"] = False
    img = image_cropper.crop(**param)
    start = time.time()
    prediction = model_test.predict(img, conf)
    test_speed = time.time() - start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]

    if label == 1:
        print("This image is Real Face. Score: {:.2f}.".format(value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("This image is Fake Face. Score: {:.2f}.".format(value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_path)[-1]
    result_image_name = image_path.replace(format_, "_result" + format_)
    cv2.imwrite(result_image_name, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse arguments for testing')
    parser.add_argument("--img_path", type=str, default="./images/sample/image_F1.jpg", help="image used to test")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file")  

    args = parser.parse_args()
    conf = load_config(args.config)
    test(args.img_path, conf)
