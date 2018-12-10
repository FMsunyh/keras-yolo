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
# @Time    : 11/9/2018 3:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os

import keras
import keras.preprocessing.image

from core.models.model import create_yolo
from core.preprocessing import PascalVocGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def create_model():
    image = keras.layers.Input((448, 448, 3))
    # image = keras.layers.Input((None, None, 3))
    gt_boxes = keras.layers.Input((7, 7, 25))
    return create_yolo([image, gt_boxes], num_classes=20, weights=None)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).', default='/home/syh/train_data/VOCdevkit/VOC2007')
    parser.add_argument('--root_path', help='Size of the batches.', default= os.path.join(os.path.expanduser('~'), 'keras_frcnn'), type=str)

    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # create the model
    print('Creating model, this may take a second...')
    model = create_model()

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    # print model summary
    print(model.summary(line_length=180))

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
    )

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args.pascal_path,
        'trainval',
        transform_generator = train_image_data_generator
    )

    # create a generator for testing data
    test_generator = PascalVocGenerator(
        args.pascal_path,
        'test',
        transform_generator = test_image_data_generator
    )

    # start training
    batch_size = 2
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // batch_size,
        epochs=100,
        verbose=1,
        validation_data=test_generator,
        validation_steps=500,  # len(test_generator.image_names) // batch_size,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(args.root_path, 'snapshots/yolo(v1)_voc_best.h5'), monitor='val_loss', verbose=1, mode='min', save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
        ],
    )

    # store final result too
    model.save('snapshots/yolo(v1)_voc_best.h5')


    '''
    cd tools
    python train_yolo.py pascal /home/syh/train_data/VOCdevkit/VOC2007
    '''