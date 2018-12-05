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
# @Time    : 11/28/2018 1:36 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import keras
import numpy as np

class TensorReshape(keras.layers.Layer):
    """ Nearly identical to keras.layers.Reshape, but allows reshaping tensors of unknown shape.

    # Arguments
        target_shape: Target shape of the input.
    """
    def __init__(self, target_shape, *args, **kwargs):
        self.target_shape = tuple(target_shape)
        super(TensorReshape, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return keras.backend.reshape(inputs, (keras.backend.shape(inputs)[0],) + self.target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.

        # Returns
            The new output shape with a `-1` replaced with its computed value.

        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            # compute target shape when possible
            return (input_shape[0],) + self._fix_unknown_dimension(input_shape[1:], self.target_shape)
        else:
            return (input_shape[0],) + tuple(s if s != -1 else None for s in self.target_shape)

    def get_config(self):
        return {'target_shape': self.target_shape}


class Dimensions(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        return keras.backend.shape(inputs)[1:3]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2,)


# class NonMaximumSuppression(keras.layers.Layer):
#     def __init__(self, num_classes, nms_threshold=0.4, max_boxes=300, *args, **kwargs):
#         self.num_classes   = num_classes
#         self.nms_threshold = nms_threshold
#         self.max_boxes     = max_boxes
#         super(NonMaximumSuppression, self).__init__(*args, **kwargs)
#
#     def call(self, inputs, **kwargs):
#         boxes, classification = inputs
#
#         boxes          = keras.backend.reshape(boxes, (-1, 4))
#         classification = keras.backend.reshape(classification, (-1, self.num_classes))
#
#         scores = keras.backend.max(classification, axis=1)
#         labels = keras.backend.argmax(classification, axis=1)
#         indices = core.backend.where(keras.backend.greater(labels, 0))
#
#         boxes          = core.backend.gather_nd(boxes, indices)
#         scores         = core.backend.gather_nd(scores, indices)
#         classification = core.backend.gather_nd(classification, indices)
#
#         nms_indices = core.backend.non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)
#
#         boxes          = keras.backend.gather(boxes, nms_indices)
#         classification = keras.backend.gather(classification, nms_indices)
#
#         boxes          = keras.backend.expand_dims(boxes, axis=0)
#         classification = keras.backend.expand_dims(classification, axis=0)
#
#         return [boxes, classification]
#
#     def compute_output_shape(self, input_shape):
#         return [(input_shape[0][0], None, 4), (input_shape[1][0], None, self.num_classes)]
#
#     def compute_mask(self, inputs, mask=None):
#         return [None, None]
#
#     def get_config(self):
#         return {
#             'num_classes'   : self.num_classes,
#             'nms_threshold' : self.nms_threshold,
#             'max_boxes'     : self.max_boxes,
#         }

# class RegressBoxes(keras.layers.Layer):
#     def call(self, inputs, **kwargs):
#         anchors, regression = inputs
#         return core.backend.bbox_transform_inv(anchors, regression)
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0]
