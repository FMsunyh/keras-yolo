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
import numpy as np


# class Loss(keras.layers.Layer):
#     def __init__(self, num_classes=21, cell_size=7, boxes_per_cell=2, *args, **kwargs):
#         self.num_classes = num_classes
#         self.cell_size   = cell_size
#         self.boxes_per_cell   = boxes_per_cell
#
#         super(Loss, self).__init__(*args, **kwargs)
#
#     def classification_loss(self,labels,  classification):
#         labels = tf.constant(1,dtype=tf.float32)
#         classification = tf.constant([0.1,0.9],dtype=tf.float32)
#         cls_loss = keras.backend.sparse_categorical_crossentropy(labels, classification)
#         cls_loss = keras.backend.sum(cls_loss)
#
#         return cls_loss
#
#     def regression_loss(self, regression_target, regression):
#
#         reg_loss = tf.constant(0,dtype=tf.float32)
#         return reg_loss
#
#     def confidence_loss(self,confidence_labels, confidence):
#
#         cond_loss = tf.constant(0,dtype=tf.float32)
#         return cond_loss
#
#
#     def call(self, inputs):
#         predicts, labels = inputs
#
#         classification_labels = 0
#         index_classification = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes)
#         classification = tf.reshape(predicts[:, :index_classification], [-1, self.cell_size, self.cell_size, self.num_classes])
#
#         confidence_labels = 0
#         index_confidence = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes + self.boxes_per_cell)
#         confidence = tf.reshape(predicts[:, index_classification:index_confidence],[-1, self.cell_size, self.cell_size, self.boxes_per_cell])
#
#         response = tf.reshape(labels[:, :, :, 0], [None, self.cell_size, self.cell_size, 1])
#         boxes = tf.reshape(labels[:, :, :, 1:5], [None, self.cell_size, self.cell_size, 1, 4])
#         boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
#         classes = labels[:, :, :, 5:]
#
#         cls_loss = self.classification_loss(classification_labels, classification)
#         self.add_loss(cls_loss)
#
#
#
#         cond_loss = self.confidence_loss(confidence_labels, confidence)
#         self.add_loss(cond_loss)
#
#         regression_target = 0
#         regression = tf.reshape(predicts[:, index_confidence:], [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
#         reg_loss = self.regression_loss(regression_target, regression)
#         self.add_loss(reg_loss)
#
#         return [cls_loss, cls_loss, cls_loss]
#
#     def compute_output_shape(self, input_shape):
#         return [(1,), (1,), (1,)]
#
#     def compute_mask(self, inputs, mask=None):
#         return [None, None, None]
#
#     def get_config(self):
#         return {
#             'num_classes' : self.num_classes,
#             'cell_size'   : self.cell_size,
#         }

class Loss(keras.layers.Layer):
    def __init__(self, num_classes=20, cell_size=7, boxes_per_cell=2, *args, **kwargs):
        self.num_classes = num_classes
        self.cell_size   = cell_size
        self.boxes_per_cell   = boxes_per_cell
        self.image_size = 448
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = 1.0
        self.noobject_scale = 1.0
        self.class_scale = 2.0
        self.coord_scale = 5.0


        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        super(Loss, self).__init__(*args, **kwargs)

    def classification_loss(self,labels,  classification):
        labels = tf.constant(1,dtype=tf.float32)
        classification = tf.constant([0.1,0.9],dtype=tf.float32)
        cls_loss = keras.backend.sparse_categorical_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        return cls_loss

    def regression_loss(self, regression_target, regression):

        reg_loss = tf.constant(0,dtype=tf.float32)
        return reg_loss

    def confidence_loss(self,confidence_labels, confidence):

        cond_loss = tf.constant(0,dtype=tf.float32)
        return cond_loss


    def call(self, inputs):

        #
        predicts, labels = inputs

        index_classification = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes)

        predict_classes = tf.reshape(predicts[:, :index_classification],
                                     [-1, self.cell_size, self.cell_size, self.num_classes])
        index_confidence = tf.multiply(tf.pow(self.cell_size, 2), self.num_classes + self.boxes_per_cell)
        predict_scales = tf.reshape(predicts[:, index_classification:index_confidence],
                                    [-1, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, index_confidence:],
                                   [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [-1, self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [-1, self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [1, 1, 1, 1])
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset,
                                                                                    (0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                               boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # cls_loss
        class_delta = response * (predict_classes - classes)
        cls_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                    name='cls_loss') * self.class_scale

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                     name='object_loss') * self.object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                       name='noobject_loss') * self.noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * self.coord_scale

        self.add_loss(cls_loss)
        self.add_loss(object_loss)
        self.add_loss(noobject_loss)
        self.add_loss(coord_loss)

        return [cls_loss, object_loss, noobject_loss, coord_loss]

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def compute_output_shape(self, input_shape):
        return [(1,), (1,), (1,), (1,)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'cell_size'   : self.cell_size,
        }
