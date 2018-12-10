#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 11/8/2018 4:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import numpy as np
import cv2
from PIL import Image

def read_image_bgr(path):
    '''
    :param path:
    :return: (h, w, 3)
    '''
    try:
        image = np.asarray(Image.open(path).convert('RGB'))
    except Exception as ex:
        print(path)

    return image[:, :, ::-1].copy()

def preprocess_image(x):
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def resize_image(image, min_side=448, max_side=448):
    '''
    resize image to dsize
    :param img: input (h, w, 3) = (rows, cols, 3)
    :param size:
    :return: out (h, w, 3)
    '''
    (h, w, _) = image.shape

    scale = np.asarray((min_side, max_side),dtype=float) / np.asarray((h, w),dtype=float)

    # resize the image with the computed scale
    # cv2.resize(image, (w, h))
    img = cv2.resize(image, (min_side, max_side))

    return img, scale
