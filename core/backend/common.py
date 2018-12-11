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

import keras.backend
import core.backend
import tensorflow

def overlapping(anchors, gt_boxes):
    """
    overlaps between the anchors and the gt boxes
    :param anchors: Generated anchors
    :param gt_boxes: Ground truth bounding boxes
    :return:
    """

    # assert keras.backend.ndim(anchors) == 2
    # assert keras.backend.ndim(gt_boxes) == 2
    reference = core.backend.overlap(anchors, gt_boxes)

    gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

    argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

    indices = keras.backend.stack([
        tensorflow.range(keras.backend.shape(anchors)[0]),
        keras.backend.cast(argmax_overlaps_inds, "int32")
    ], axis=0)

    indices = keras.backend.transpose(indices)

    max_overlaps = tensorflow.gather_nd(reference, indices)

    return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds

def overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], 1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], 1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], 1), b[:, 1]) + 1

    iw = keras.backend.maximum(iw, 0)
    ih = keras.backend.maximum(ih, 0)

    ua = keras.backend.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1) + area - iw * ih

    ua = keras.backend.maximum(ua, 0.0001)

    intersection = iw * ih

    return intersection / ua
