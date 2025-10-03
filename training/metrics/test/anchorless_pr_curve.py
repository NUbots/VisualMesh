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
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR DEALINGS IN THE SOFTWARE.

import tensorflow as tf

from .curve import Curve


class AnchorlessPRCurve(Curve):
    def __init__(self, use_offsets=True, **kwargs):
        super(AnchorlessPRCurve, self).__init__(**kwargs)
        self.use_offsets = use_offsets

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update curve with heatmap predictions vs ground truth centers.

        Args:
            y_true: [N, 3] with channels [heatmap, offset_x, offset_y]
            y_pred: [N, 3] with channels [heatmap_logits, offset_x, offset_y]
        """
        # Extract heatmap channels
        hm_true = y_true[:, 0]  # [N] - ground truth heatmap (1.0 at centers, gaussian elsewhere)
        hm_pred_logits = y_pred[:, 0]  # [N] - predicted heatmap logits

        # Convert logits to probabilities for thresholding
        hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)  # [N]

        # Find ground truth centers (nodes with value 1.0)
        gt_centers = tf.cast(tf.greater_equal(hm_true, 1.0), tf.int64)  # [N] - 1 for centers, 0 otherwise

        # Prepare data for curve: we'll threshold on prediction probabilities
        # X = prediction probabilities, c = [positive_count, negative_count] for each sample
        X = tf.expand_dims(hm_pred_prob, axis=-1)  # [N, 1]
        c = tf.stack([gt_centers, 1 - gt_centers], axis=-1)  # [N, 2] - [pos, neg] counts

        self.update(X, c)
