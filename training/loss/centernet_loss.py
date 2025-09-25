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


class CenterNetLoss:
    """
    CenterNet-style focal loss for anchorless object detection.

    This loss function handles the extreme class imbalance between object centers
    and background pixels by:
    1. Reducing loss for easy negatives (background far from objects)
    2. Focusing training on hard examples near object boundaries
    3. Using smooth L1 loss for offset regression (if applicable)
    """

    def __init__(self, alpha=2.0, beta=4.0, **kwargs):
        """
        Initialize the anchorless loss function.

        Args:
            alpha: Focal loss exponent for hard negative mining (default: 2.0)
            beta: Focal loss exponent for positive examples (default: 4.0)
        """
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_true, y_pred):
        """
        Compute the CenterNet focal loss.

        Args:
            y_true: Ground truth heatmap [batch_size, num_points, 1]
            y_pred: Predicted heatmap [batch_size, num_points, 1]

        Returns:
            Scalar loss value
        """
        # Ensure predictions are in valid range [0, 1]
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Separate positive and negative examples
        # Positive: where ground truth > 0 (near object centers)
        # Negative: where ground truth = 0 (background)
        pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)  # peaks only
        neg_mask = 1.0 - pos_mask
        neg_weight = tf.pow(1.0 - y_true, self.beta)

        pos_loss = - tf.pow(1.0 - y_pred, self.alpha) * tf.math.log(tf.clip_by_value(y_pred, 1e-6, 1.0)) * pos_mask
        neg_loss = - tf.pow(y_pred, self.alpha) * tf.math.log(tf.clip_by_value(1.0 - y_pred, 1e-6, 1.0)) * neg_mask * neg_weight

        num_objs = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
        total = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / num_objs

        return total
