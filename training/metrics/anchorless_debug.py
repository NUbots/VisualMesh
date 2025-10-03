# Copyright (C) 2017-2020 Trent Houliston <trent@houliston.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf

from .anchorless_confusion_base import AnchorlessConfusionBase


class AnchorlessDebug(AnchorlessConfusionBase):
    def __init__(self, name, threshold, use_offsets=True, **kwargs):
        super(AnchorlessDebug, self).__init__(name, threshold, use_offsets, **kwargs)
        self.gt_centers_count = self.add_weight(name="gt_centers", initializer="zeros", dtype=tf.int32)
        self.pred_centers_count = self.add_weight(name="pred_centers", initializer="zeros", dtype=tf.int32)
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Call parent to update confusion matrix
        super().update_state(y_true, y_pred, sample_weight)

        # Additional debug tracking
        if len(tf.shape(y_true)) >= 2 and tf.shape(y_true)[1] >= 1:
            hm_true = y_true[:, 0]
            hm_pred_logits = y_pred[:, 0]
            hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)

            gt_centers = tf.reduce_sum(tf.cast(tf.greater_equal(hm_true, 0.99), tf.int32))
            pred_centers = tf.reduce_sum(tf.cast(tf.greater_equal(hm_pred_prob, self.threshold), tf.int32))

            self.gt_centers_count.assign_add(gt_centers)
            self.pred_centers_count.assign_add(pred_centers)
            self.batch_count.assign_add(1)

    def result(self):
        # Return the total number of ground truth centers seen for debugging
        return tf.cast(self.gt_centers_count, self.dtype)

    def reset_state(self):
        super().reset_state()
        self.gt_centers_count.assign(0)
        self.pred_centers_count.assign(0)
        self.batch_count.assign(0)
