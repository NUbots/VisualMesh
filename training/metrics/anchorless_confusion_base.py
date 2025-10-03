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

from .confusion_base import ConfusionBase


class AnchorlessConfusionBase(ConfusionBase):
    def __init__(self, name, threshold, use_offsets=True, **kwargs):
        super(AnchorlessConfusionBase, self).__init__(name, size=2, **kwargs)
        self.threshold = threshold
        self.use_offsets = use_offsets
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update confusion matrix based on heatmap peaks vs ground truth centers.

        Args:
            y_true: [batch*N, 3] with channels [heatmap, offset_x, offset_y]
            y_pred: [batch*N, 3] with channels [heatmap_logits, offset_x, offset_y]
        """
        # Check if inputs are valid (have at least 2 dimensions and at least 1 channel)
        valid_true = tf.logical_and(tf.greater_equal(tf.rank(y_true), 2), tf.greater(tf.shape(y_true)[1], 0))
        valid_pred = tf.logical_and(tf.greater_equal(tf.rank(y_pred), 2), tf.greater(tf.shape(y_pred)[1], 0))
        valid_input = tf.logical_and(valid_true, valid_pred)

        def process_valid_inputs():
            # Extract heatmap channels
            hm_true = y_true[:, 0]  # [batch*N] - ground truth heatmap (1.0 at centers, gaussian elsewhere)
            hm_pred_logits = y_pred[:, 0]  # [batch*N] - predicted heatmap logits

            # Convert logits to probabilities with numerical stability
            hm_pred_logits = tf.clip_by_value(hm_pred_logits, -20.0, 20.0)  # Prevent extreme logits
            hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)  # [batch*N]

            # Find ground truth centers (nodes with value 1.0)
            # Use a small tolerance for floating point comparison
            gt_centers = tf.greater_equal(hm_true, 0.99)  # [batch*N] bool

            # Find predicted centers above threshold
            pred_centers = tf.greater_equal(hm_pred_prob, self.threshold)  # [batch*N] bool

            # Convert boolean to int32 for confusion matrix indices
            gt_labels = tf.cast(gt_centers, tf.int32)  # [batch*N] - 0 or 1
            pred_labels = tf.cast(pred_centers, tf.int32)  # [batch*N] - 0 or 1

            # Create indices for confusion matrix [gt_label, pred_label]
            indices = tf.stack([gt_labels, pred_labels], axis=1)  # [batch*N, 2]

            # All indices should be valid (0 or 1), so add them all
            self.confusion.scatter_nd_add(indices, tf.ones(tf.shape(indices)[0], dtype=self.confusion.dtype))
            self.total_samples.assign_add(tf.shape(indices)[0])

        def skip_invalid_inputs():
            # Do nothing for invalid inputs
            pass

        # Use tf.cond to handle the branching properly
        tf.cond(valid_input, process_valid_inputs, skip_invalid_inputs)

    def reset_state(self):
        """Reset confusion matrix state (renamed from reset_states for TF compatibility)."""
        self.confusion.assign(tf.zeros_like(self.confusion))
        self.total_samples.assign(0)
