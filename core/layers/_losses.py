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
# @Time    : 11/20/2018 3:22 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import tensorflow as tf

class Loss(keras.layers.Layer):
    def __init__(self, num_classes=21, cell_size=7, boxes_per_cell=2, *args, **kwargs):
        self.num_classes = num_classes
        self.cell_size   = cell_size
        self.boxes_per_cell   = boxes_per_cell

        super(Loss, self).__init__(*args, **kwargs)

    def classification_loss(self,labels,  classification):

        cls_loss = 0
        return cls_loss

    def regression_loss(self, regression_target, regression):
        reg_loss = 0
        return reg_loss

    def call(self, inputs):
        predicts, labels = inputs

        classification_labels = 0
        classification = 0
        cls_loss = self.classification_loss(classification_labels, classification)

        regression_target = 0
        regression = 0
        reg_loss = self.regression_loss(regression_target, regression)

        loss = cls_loss + reg_loss
        self.add_loss(reg_loss)

        loss = tf.constant(0,dtype=tf.float32)
        return loss

    def compute_output_shape(self, input_shape):
        return [(1,)]

    def compute_mask(self, inputs, mask=None):
        return [None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'sigma'       : self.sigma,
        }

