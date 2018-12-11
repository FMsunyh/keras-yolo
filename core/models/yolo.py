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
# @Time    : 11/23/2018 10:38 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import core
import keras
from functools import wraps
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import tensorflow as tf

from core.models.utils import compose


class Yolo(object):
    def __init__(self, training=True, num_classes=21, weights=None):
        '''
        Fast RCNN introduced in Faster R-CNN.
        '''
        super(Yolo, self).__init__()

        self.training = training
        self.num_classes = num_classes

        self.network()



    @wraps(Conv2D)
    def _Conv2D(self,*args, **kwargs):
        """Wrapper to set Darknet parameters for Convolution2D."""
        _kwargs = {'kernel_regularizer': l2(5e-4)}
        _kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
        _kwargs.update(kwargs)
        return Conv2D(*args, **_kwargs)

    def _Conv2D_BN_Leaky(self, *args, **kwargs):
        """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
        no_bias_kwargs = {'use_bias': False}
        no_bias_kwargs.update(kwargs)
        return compose(
            self._Conv2D(*args, **no_bias_kwargs),
            BatchNormalization(),
            # ReLU())
            LeakyReLU(alpha=0.1))

    def network(self):
        # Convolution block 0
        self.out_0      = self._Conv2D_BN_Leaky(64, kernel_size=(7,7), strides=(1,1), name='conv_0')
        self.pooling_0  = MaxPooling2D(pool_size=(2,2), strides=2, name='pooling_0')

        # Convolution block 1
        self.out_1      = self._Conv2D_BN_Leaky(192, kernel_size=(3, 3), strides=(1, 1), name='conv_1')
        self.pooling_1  = MaxPooling2D(pool_size=(2,2), strides=2, name='pooling_1')

        # Convolution block 2
        self.out_2      = self._Conv2D_BN_Leaky(128, kernel_size=(1, 1), strides=(1, 1), name='conv_2')
        self.out_3      = self._Conv2D_BN_Leaky(256, kernel_size=(3, 3), strides=(1, 1), name='conv_3')
        self.out_4      = self._Conv2D_BN_Leaky(256, kernel_size=(1, 1), strides=(1, 1), name='conv_4')
        self.out_5      = self._Conv2D_BN_Leaky(512, kernel_size=(3, 3), strides=(1, 1), name='conv_5')
        self.pooling_2  = MaxPooling2D(pool_size=(2,2), strides=2, name='pooling_2')

        # Convolution block 3
        self.out_6      = self._Conv2D_BN_Leaky(256, kernel_size=(1, 1), strides=(1, 1), name='conv_6')
        self.out_7      = self._Conv2D_BN_Leaky(512, kernel_size=(3, 3), strides=(1, 1), name='conv_7')
        self.out_8      = self._Conv2D_BN_Leaky(256, kernel_size=(1, 1), strides=(1, 1), name='conv_8')
        self.out_9      = self._Conv2D_BN_Leaky(512, kernel_size=(3, 3), strides=(1, 1), name='conv_9')
        self.out_10     = self._Conv2D_BN_Leaky(256, kernel_size=(1, 1), strides=(1, 1), name='conv_10')
        self.out_11     = self._Conv2D_BN_Leaky(512, kernel_size=(3, 3), strides=(1, 1), name='conv_11')
        self.out_12     = self._Conv2D_BN_Leaky(256, kernel_size=(1, 1), strides=(1, 1), name='conv_12')
        self.out_13     = self._Conv2D_BN_Leaky(512, kernel_size=(3, 3), strides=(1, 1), name='conv_13')
        self.out_14     = self._Conv2D_BN_Leaky(512, kernel_size=(1, 1), strides=(1, 1), name='conv_14')
        self.out_15     = self._Conv2D_BN_Leaky(1024,kernel_size=(3, 3), strides=(1, 1), name='conv_15')
        self.pooling_3  = MaxPooling2D(pool_size=(2,2), strides=2, name='pooling_3')

        # Convolution block 4
        self.out_16     = self._Conv2D_BN_Leaky(512, kernel_size=(1, 1), strides=(1, 1), name='conv_16')
        self.out_17     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(1, 1), name='conv_17')
        self.out_18     = self._Conv2D_BN_Leaky(512, kernel_size=(1, 1), strides=(1, 1), name='conv_18')
        self.out_19     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(1, 1), name='conv_19')
        self.out_20     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(1, 1), name='conv_20')
        self.out_21     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(2, 2), name='conv_21')

        # Convolution block 5
        self.out_22     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(1, 1), name='conv_22')
        self.out_23     = self._Conv2D_BN_Leaky(1024, kernel_size=(3, 3), strides=(1, 1), name='conv_23')

        self.flatten = keras.layers.Flatten(name='Flatten')
        self.fc_0 = keras.layers.Dense(units=512,name='fc_0')
        self.fc_1 = keras.layers.Dense(units=4096,name='fc_1')
        self.fc_2 = keras.layers.Dense(units=7*7*30, name='fc_2')

        if self.training:
            self.loss = core.layers.Loss(num_classes=self.num_classes)

    def __call__(self, inputs, mask=None):
        if self.training:
            image, gt_boxes = inputs
            image_shape = core.layers.Dimensions()(image)
        else:
            image = inputs

        classification = None
        regression = None
        loss = None

        x = self.out_0(image)
        x = self.pooling_0(x)

        x = self.out_1(x)
        x = self.pooling_1(x)

        x = self.out_2(x)
        x = self.out_3(x)
        x = self.out_4(x)
        x = self.out_5(x)
        x = self.pooling_2(x)

        x = self.out_6(x)
        x = self.out_7(x)
        x = self.out_9(x)
        x = self.out_10(x)
        x = self.out_11(x)
        x = self.out_12(x)
        x = self.out_13(x)
        x = self.out_14(x)
        x = self.out_15(x)
        x = self.pooling_3(x)

        x = self.out_16(x)
        x = self.out_17(x)
        x = self.out_18(x)
        x = self.out_19(x)
        x = self.out_20(x)
        x = self.out_21(x)
        x = self.out_22(x)
        x = self.out_23(x)
        x = self.flatten(x)
        x = self.fc_0(x)
        x = self.fc_1(x)
        x = self.fc_2(x)

        if self.training:
            coord_loss, object_loss, noobject_loss, cls_loss = self.loss([x, gt_boxes, image_shape])

        return coord_loss, object_loss, noobject_loss, cls_loss, x
