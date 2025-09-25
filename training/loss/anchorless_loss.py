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


class AnchorlessLoss:
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
        Compute the anchorless focal loss.

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
        pos_mask = tf.cast(tf.greater(y_true, 0.0), tf.float32)
        neg_mask = tf.cast(tf.equal(y_true, 0.0), tf.float32)

        # Count positive examples for normalization
        num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)

        # Focal loss for positive examples
        # When prediction is wrong (y_pred low, y_true high), loss is high
        pos_loss = -tf.pow(1.0 - y_pred, self.alpha) * tf.math.log(y_pred) * y_true
        pos_loss = tf.reduce_sum(pos_loss * pos_mask)

        # Focal loss for negative examples
        # Reduce loss for easy negatives (y_pred already low)
        # Focus on hard negatives (y_pred high but y_true = 0)
        neg_loss = -tf.pow(y_pred, self.alpha) * tf.pow(1.0 - y_true, self.beta) * tf.math.log(1.0 - y_pred)
        neg_loss = tf.reduce_sum(neg_loss * neg_mask)

        # Normalize by number of positive examples
        total_loss = (pos_loss + neg_loss) / num_pos

        return total_loss

class CenterNetLossLogits:
    """
    CenterNet focal loss (logits version).
    Use this if your network head outputs raw logits (no sigmoid in the model).
    """

    def __init__(self, alpha=2.0, beta=4.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_true, logits):
        # Convert logits to probs only for weighting, but use stable cross-entropy
        p = tf.sigmoid(logits)

        pos_mask = tf.cast(tf.equal(y_true, 1.0), tf.float32)
        neg_mask = 1.0 - pos_mask
        neg_weight = tf.pow(1.0 - y_true, self.beta)

        # Positive focal loss
        pos_loss = -tf.pow(1.0 - p, self.alpha) * tf.nn.log_sigmoid(logits) * pos_mask
        # Negative focal loss
        neg_loss = -tf.pow(p, self.alpha) * tf.nn.log_sigmoid(-logits) * neg_mask * neg_weight

        num_objs = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
        total = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)) / num_objs

        return total


class AnchorlessLossWithOffset:
    """
    Extended anchorless loss that handles both heatmap and offset regression.
    This is for the full CenterNet approach with both detection and localization.
    """

    def __init__(self, alpha=2.0, beta=4.0, offset_weight=1.0, **kwargs):
        """
        Initialize the extended anchorless loss function.

        Args:
            alpha: Focal loss exponent for hard negative mining
            beta: Focal loss exponent for positive examples
            offset_weight: Weight for offset regression loss
        """
        self.alpha = alpha
        self.beta = beta
        self.offset_weight = offset_weight

    def heatmap_loss(self, y_true_hm, y_pred_hm):
        """Compute focal loss for heatmap prediction."""
        y_pred_hm = tf.clip_by_value(y_pred_hm, 1e-7, 1.0 - 1e-7)

        pos_mask = tf.cast(tf.greater(y_true_hm, 0.0), tf.float32)
        neg_mask = tf.cast(tf.equal(y_true_hm, 0.0), tf.float32)
        num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)

        # Positive loss
        pos_loss = -tf.pow(1.0 - y_pred_hm, self.alpha) * tf.math.log(y_pred_hm) * y_true_hm
        pos_loss = tf.reduce_sum(pos_loss * pos_mask)

        # Negative loss
        neg_loss = -tf.pow(y_pred_hm, self.alpha) * tf.pow(1.0 - y_true_hm, self.beta) * tf.math.log(1.0 - y_pred_hm)
        neg_loss = tf.reduce_sum(neg_loss * neg_mask)

        return (pos_loss + neg_loss) / num_pos

    def offset_loss(self, y_true_hm, y_true_offset, y_pred_offset):
        """Compute smooth L1 loss for offset regression, only at positive locations."""
        # Only compute offset loss where we have positive examples
        pos_mask = tf.cast(tf.greater(y_true_hm, 0.0), tf.float32)
        num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)

        # Smooth L1 loss (Huber loss with delta=1.0)
        diff = y_true_offset - y_pred_offset
        abs_diff = tf.abs(diff)

        smooth_l1 = tf.where(
            abs_diff < 1.0,
            0.5 * tf.square(diff),  # L2 loss for small errors
            abs_diff - 0.5          # L1 loss for large errors
        )

        # Apply only to positive locations and average over 2D offsets
        smooth_l1 = tf.reduce_sum(smooth_l1, axis=-1, keepdims=True)  # Sum over x,y dimensions
        offset_loss = tf.reduce_sum(smooth_l1 * pos_mask) / num_pos

        return offset_loss

    def __call__(self, y_true, y_pred):
        """
        Compute combined heatmap + offset loss.

        Expected format:
        y_true: {"heatmap": [batch, points, 1], "offset": [batch, points, 2]}
        y_pred: {"heatmap": [batch, points, 1], "offset": [batch, points, 2]}

        Or if using single output:
        y_true: [batch, points, 3] where [:,:,0] = heatmap, [:,:,1:3] = offset
        y_pred: [batch, points, 3] where [:,:,0] = heatmap, [:,:,1:3] = offset
        """

        # Handle dictionary format
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            y_true_hm = y_true["heatmap"]
            y_true_offset = y_true["offset"]
            y_pred_hm = y_pred["heatmap"]
            y_pred_offset = y_pred["offset"]
        else:
            # Handle concatenated format [heatmap, offset_x, offset_y]
            y_true_hm = y_true[..., 0:1]  # [batch, points, 1]
            y_true_offset = y_true[..., 1:3]  # [batch, points, 2]
            y_pred_hm = y_pred[..., 0:1]  # [batch, points, 1]
            y_pred_offset = y_pred[..., 1:3]  # [batch, points, 2]

        # Compute individual losses
        hm_loss = self.heatmap_loss(y_true_hm, y_pred_hm)
        off_loss = self.offset_loss(y_true_hm, y_true_offset, y_pred_offset)

        # Combine losses
        total_loss = hm_loss + self.offset_weight * off_loss

        return total_loss


class AdaptiveAnchorlessLoss:
    """
    Adaptive version that automatically balances heatmap and offset losses
    based on their relative magnitudes during training.
    """

    def __init__(self, alpha=2.0, beta=4.0, **kwargs):
        self.alpha = alpha
        self.beta = beta

        # Learnable loss weights (initialized to equal importance)
        self.heatmap_weight = tf.Variable(1.0, trainable=True, name="heatmap_weight")
        self.offset_weight = tf.Variable(1.0, trainable=True, name="offset_weight")

    def __call__(self, y_true, y_pred):
        """Compute adaptive weighted loss."""
        # Use the same loss computation as AnchorlessLossWithOffset
        loss_fn = AnchorlessLossWithOffset(self.alpha, self.beta, 1.0)

        # Handle format conversion
        if isinstance(y_true, dict):
            y_true_hm = y_true["heatmap"]
            y_true_offset = y_true["offset"]
            y_pred_hm = y_pred["heatmap"]
            y_pred_offset = y_pred["offset"]
        else:
            y_true_hm = y_true[..., 0:1]
            y_true_offset = y_true[..., 1:3]
            y_pred_hm = y_pred[..., 0:1]
            y_pred_offset = y_pred[..., 1:3]

        # Compute individual losses
        hm_loss = loss_fn.heatmap_loss(y_true_hm, y_pred_hm)
        off_loss = loss_fn.offset_loss(y_true_hm, y_true_offset, y_pred_offset)

        # Apply learnable weights with softmax normalization
        weights = tf.nn.softmax([self.heatmap_weight, self.offset_weight])

        total_loss = weights[0] * hm_loss + weights[1] * off_loss

        return total_loss
