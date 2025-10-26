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
# COPYRIGHT WARRANTIES BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import tensorflow as tf


class AnchorlessPeakNodeDistance(tf.keras.metrics.Metric):
    """
    Node-level peak distance metric for anchorless object detection.

    Measures the accuracy of predicted object center nodes by computing the
    node index distance between the highest activation node and the nearest
    ground truth center node. Provides a discrete distance metric that's
    independent of mesh geometry but still meaningful for evaluation.
    """

    def __init__(self, name, distance_threshold=5, **kwargs):
        """
        Initialize peak node distance metric.

        Args:
            name: Metric name for logging
            distance_threshold: Node distance threshold for P@θ calculation
        """
        super().__init__(name=name, **kwargs)
        self.distance_threshold = distance_threshold

        # Track various statistics
        self.total_correct_nodes = self.add_weight(name="correct_nodes", initializer="zeros", dtype=tf.int32)
        self.total_predictions = self.add_weight(name="total_predictions", initializer="zeros", dtype=tf.int32)
        self.total_ground_truths = self.add_weight(name="total_ground_truths", initializer="zeros", dtype=tf.int32)
        self.within_threshold = self.add_weight(name="within_threshold", initializer="zeros", dtype=tf.int32)

        # Simple distance sum for mean calculation
        self.distance_sum = self.add_weight(name="distance_sum", initializer="zeros", dtype=tf.float32)
        self.num_distances = self.add_weight(name="num_distances", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update with node distance measurements."""
        # Build boolean guard with tensor ops
        ok_rank = tf.logical_and(tf.rank(y_true) >= 2, tf.rank(y_pred) >= 2)
        ok_width = tf.logical_and(tf.shape(y_true)[1] >= 1, tf.shape(y_pred)[1] >= 1)
        do_update = tf.logical_and(ok_rank, ok_width)

        def body():
            # Extract heatmaps
            hm_true = y_true[:, 0]  # Ground truth heatmap
            hm_pred_logits = y_pred[:, 0]  # Predicted heatmap logits

            # Convert logits to probabilities
            hm_pred_logits = tf.clip_by_value(hm_pred_logits, -20.0, 20.0)
            hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)

            # Find ground truth center nodes (value >= 1.0, matching label generation)
            gt_center_indices = tf.where(tf.greater_equal(hm_true, 1.0))[:, 0]
            num_gt = tf.shape(gt_center_indices)[0]
            self.total_ground_truths.assign_add(num_gt)

            # Check for meaningful predictions and update accordingly
            has_prediction = tf.reduce_max(hm_pred_prob) > 0.01

            def update_with_prediction():
                peak_idx = tf.argmax(hm_pred_prob)
                self.total_predictions.assign_add(1)

                # Check if we have ground truth to compare against
                def update_distances():
                    # Calculate node index distances
                    gt_indices = tf.cast(gt_center_indices, tf.int64)
                    peak_idx_int64 = tf.cast(peak_idx, tf.int64)

                    # Distances from peak to all GT centers
                    distances = tf.abs(gt_indices - peak_idx_int64)
                    min_distance = tf.reduce_min(distances)

                    # Perfect node match
                    self.total_correct_nodes.assign_add(tf.cast(tf.equal(min_distance, 0), tf.int32))

                    # Within threshold
                    self.within_threshold.assign_add(tf.cast(min_distance <= self.distance_threshold, tf.int32))

                    # Accumulate distance sum for mean calculation
                    self.distance_sum.assign_add(tf.cast(min_distance, tf.float32))
                    self.num_distances.assign_add(1)
                    return tf.constant(0)

                return tf.cond(num_gt > 0, update_distances, lambda: tf.constant(0))

            tf.cond(has_prediction, update_with_prediction, lambda: tf.constant(0))
            return tf.constant(0)

        # Both branches must return same-shaped tensor
        return tf.cond(do_update, body, lambda: tf.constant(0))

    def result(self):
        """Return proportion of predictions within threshold."""
        return tf.cond(
            self.total_predictions > 0,
            lambda: tf.cast(self.within_threshold, tf.float32) / tf.cast(self.total_predictions, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def reset_state(self):
        """Reset all accumulated state."""
        self.total_correct_nodes.assign(0)
        self.total_predictions.assign(0)
        self.total_ground_truths.assign(0)
        self.within_threshold.assign(0)
        self.distance_sum.assign(0.0)
        self.num_distances.assign(0)

    def save(self, output_path):
        """Save metric results to file."""
        import os
        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        with open("{}.txt".format(base_path), "w") as f:
            accuracy = self.result().numpy()
            avg_distance = (self.distance_sum.numpy() / max(1, self.num_distances.numpy()))

            f.write("Peak Node Distance Metric (threshold={})\n".format(self.distance_threshold))
            f.write("Accuracy within threshold: {:.4f}\n".format(accuracy))
            f.write("Average node distance: {:.2f}\n".format(avg_distance))
            f.write("Total predictions: {}\n".format(self.total_predictions.numpy()))
            f.write("Predictions within threshold: {}\n".format(self.within_threshold.numpy()))


class AnchorlessPeakAccuracy(tf.keras.metrics.Metric):
    """
    Exact peak node accuracy metric for anchorless object detection.

    Measures the proportion of predictions where the peak node exactly
    matches a ground truth center node (node distance = 0).
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_predictions = self.add_weight(name="correct", initializer="zeros", dtype=tf.int32)
        self.total_predictions = self.add_weight(name="total", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update with exact node matches."""
        # Build boolean guard with tensor ops
        ok_rank = tf.logical_and(tf.rank(y_true) >= 2, tf.rank(y_pred) >= 2)
        ok_width = tf.logical_and(tf.shape(y_true)[1] >= 1, tf.shape(y_pred)[1] >= 1)
        do_update = tf.logical_and(ok_rank, ok_width)

        def body():
            hm_true = y_true[:, 0]
            hm_pred_logits = y_pred[:, 0]

            hm_pred_logits = tf.clip_by_value(hm_pred_logits, -20.0, 20.0)
            hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)

            # Ground truth centers (hard peaks >= 1.0, matching label generation)
            gt_center_indices = tf.where(tf.greater_equal(hm_true, 1.0))[:, 0]

            # Check for meaningful predictions
            has_prediction = tf.reduce_max(hm_pred_prob) > 0.01

            def update_with_prediction():
                peak_idx = tf.argmax(hm_pred_prob)
                self.total_predictions.assign_add(1)

                # Check if peak matches any GT center
                num_gt = tf.shape(gt_center_indices)[0]

                def check_matches():
                    matches = tf.equal(tf.cast(gt_center_indices, tf.int64), tf.cast(peak_idx, tf.int64))
                    self.correct_predictions.assign_add(tf.cast(tf.reduce_any(matches), tf.int32))
                    return tf.constant(0)

                return tf.cond(num_gt > 0, check_matches, lambda: tf.constant(0))

            tf.cond(has_prediction, update_with_prediction, lambda: tf.constant(0))
            return tf.constant(0)

        # Both branches must return same-shaped tensor
        return tf.cond(do_update, body, lambda: tf.constant(0))

    def result(self):
        """Return accuracy (proportion of exact matches)."""
        return tf.cond(
            self.total_predictions > 0,
            lambda: tf.cast(self.correct_predictions, tf.float32) / tf.cast(self.total_predictions, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def reset_state(self):
        """Reset state."""
        self.correct_predictions.assign(0)
        self.total_predictions.assign(0)

    def save(self, output_path):
        """Save metric results to file."""
        import os
        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        with open("{}.txt".format(base_path), "w") as f:
            accuracy = self.result().numpy()
            f.write("Peak Exact Accuracy Metric\n")
            f.write("Exact match accuracy: {:.4f}\n".format(accuracy))
            f.write("Total predictions: {}\n".format(self.total_predictions.numpy()))
            f.write("Exact matches: {}\n".format(self.correct_predictions.numpy()))


class AnchorlessSpatialAccuracy(tf.keras.metrics.Metric):
    """
    Spatial distance accuracy for anchorless object detection.

    Measures accuracy based on actual spatial distance in nm coordinates
    (like what you see in TensorBoard), rather than discrete node indices.
    This includes offset vector contributions for sub-node precision.
    """

    def __init__(self, name, distance_threshold=0.1, **kwargs):
        """
        Args:
            name: Metric name
            distance_threshold: Spatial distance threshold in nm coordinates
        """
        super().__init__(name=name, **kwargs)
        self.distance_threshold = distance_threshold

        self.within_threshold = self.add_weight(name="within_threshold", initializer="zeros", dtype=tf.int32)
        self.total_predictions = self.add_weight(name="total_predictions", initializer="zeros", dtype=tf.int32)
        self.distance_sum = self.add_weight(name="distance_sum", initializer="zeros", dtype=tf.float32)
        self.num_distances = self.add_weight(name="num_distances", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update with spatial distance measurements."""
        # This would need mesh vertices (V) and offset scale to work properly
        # For now, just count as a placeholder - we'll implement this properly
        return tf.constant(0)

    def result(self):
        """Return proportion of predictions within spatial threshold."""
        return tf.cond(
            self.total_predictions > 0,
            lambda: tf.cast(self.within_threshold, tf.float32) / tf.cast(self.total_predictions, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def reset_state(self):
        """Reset state."""
        self.within_threshold.assign(0)
        self.total_predictions.assign(0)
        self.distance_sum.assign(0.0)
        self.num_distances.assign(0)

    def save(self, output_path):
        """Save metric results to file."""
        import os
        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        with open("{}.txt".format(base_path), "w") as f:
            accuracy = self.result().numpy()
            avg_distance = (self.distance_sum.numpy() / max(1, self.num_distances.numpy()))

            f.write("Spatial Distance Accuracy (threshold={:.3f})\n".format(self.distance_threshold))
            f.write("Spatial accuracy: {:.4f}\n".format(accuracy))
            f.write("Average spatial distance: {:.4f}\n".format(avg_distance))
            f.write("Total predictions: {}\n".format(self.total_predictions.numpy()))
            f.write("Predictions within threshold: {}\n".format(self.within_threshold.numpy()))


class AnchorlessOffsetAccuracy(tf.keras.metrics.Metric):
    """
    Offset prediction accuracy for anchorless object detection.

    When a prediction peak matches a ground truth center node exactly,
    measures the accuracy of the predicted offset vector. This provides
    sub-node precision evaluation for cases where the discrete node
    selection is correct.

    Computes error in Visual Mesh nm coordinate units by unscaling both
    predicted and ground truth offset vectors.
    """

    def __init__(self, name, error_threshold=0.1, intersections=6, **kwargs):
        """
        Args:
            name: Metric name
            error_threshold: Offset error threshold for accuracy calculation (in nm units)
            intersections: Number of intersections used in offset scaling (default 6)
        """
        super().__init__(name=name, **kwargs)
        self.error_threshold = error_threshold
        # Offset scale used in labeller: 0.5 / intersections
        self.offset_scale = 0.5 / float(intersections)

        self.offset_error_sum = self.add_weight(name="offset_error_sum", initializer="zeros", dtype=tf.float32)
        self.num_errors = self.add_weight(name="num_errors", initializer="zeros", dtype=tf.int32)
        self.accurate_offsets = self.add_weight(name="accurate", initializer="zeros", dtype=tf.int32)
        self.total_offsets = self.add_weight(name="total", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update with offset accuracy measurements."""
        # Build boolean guard with tensor ops
        ok_rank = tf.logical_and(tf.rank(y_true) >= 2, tf.rank(y_pred) >= 2)
        ok_width = tf.logical_and(tf.shape(y_true)[1] >= 3, tf.shape(y_pred)[1] >= 3)
        do_update = tf.logical_and(ok_rank, ok_width)

        def body():
            hm_true = y_true[:, 0]
            offset_true = y_true[:, 1:3]

            hm_pred_logits = y_pred[:, 0]
            offset_pred = y_pred[:, 1:3]

            hm_pred_logits = tf.clip_by_value(hm_pred_logits, -20.0, 20.0)
            hm_pred_prob = tf.nn.sigmoid(hm_pred_logits)

            # Ground truth centers (hard peaks >= 1.0, matching label generation)
            gt_center_indices = tf.where(tf.greater_equal(hm_true, 1.0))[:, 0]

            # Check for meaningful predictions
            has_prediction = tf.reduce_max(hm_pred_prob) > 0.01

            def update_with_prediction():
                peak_idx = tf.argmax(hm_pred_prob)
                num_gt = tf.shape(gt_center_indices)[0]

                # Check if peak exactly matches a GT center
                def check_exact_match():
                    peak_idx_int64 = tf.cast(peak_idx, tf.int64)
                    matches = tf.equal(tf.cast(gt_center_indices, tf.int64), peak_idx_int64)
                    has_exact_match = tf.reduce_any(matches)

                    def evaluate_offset():
                        # We have an exact node match, evaluate offset accuracy
                        self.total_offsets.assign_add(1)

                        # Get true and predicted offsets at peak location (both in scaled space)
                        offset_true_scaled = offset_true[peak_idx]
                        offset_pred_scaled = offset_pred[peak_idx]

                        # Unscale both offsets to get actual nm coordinate error
                        offset_true_nm = offset_true_scaled * self.offset_scale
                        offset_pred_nm = offset_pred_scaled * self.offset_scale

                        # Compute offset error in nm units
                        offset_error_nm = tf.norm(offset_pred_nm - offset_true_nm)

                        # Accumulate error for mean calculation
                        self.offset_error_sum.assign_add(offset_error_nm)
                        self.num_errors.assign_add(1)

                        # Check if within threshold (threshold is in nm units)
                        self.accurate_offsets.assign_add(tf.cast(offset_error_nm <= self.error_threshold, tf.int32))
                        return tf.constant(0)

                    return tf.cond(has_exact_match, evaluate_offset, lambda: tf.constant(0))

                return tf.cond(num_gt > 0, check_exact_match, lambda: tf.constant(0))

            tf.cond(has_prediction, update_with_prediction, lambda: tf.constant(0))
            return tf.constant(0)

        # Both branches must return same-shaped tensor
        return tf.cond(do_update, body, lambda: tf.constant(0))

    def result(self):
        """Return proportion of offsets within error threshold."""
        return tf.cond(
            self.total_offsets > 0,
            lambda: tf.cast(self.accurate_offsets, tf.float32) / tf.cast(self.total_offsets, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def reset_state(self):
        """Reset state."""
        self.offset_error_sum.assign(0.0)
        self.num_errors.assign(0)
        self.accurate_offsets.assign(0)
        self.total_offsets.assign(0)

    def save(self, output_path):
        """Save metric results to file."""
        import os
        base_path = os.path.join(output_path, "test", self.name)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        with open("{}.txt".format(base_path), "w") as f:
            accuracy = self.result().numpy()
            avg_error = (self.offset_error_sum.numpy() / max(1, self.num_errors.numpy()))

            f.write("Offset Accuracy Metric (threshold={:.3f} nm)\n".format(self.error_threshold))
            f.write("Offset accuracy: {:.4f}\n".format(accuracy))
            f.write("Average offset error: {:.4f} nm\n".format(avg_error))
            f.write("Total offset evaluations: {}\n".format(self.total_offsets.numpy()))
            f.write("Accurate offsets: {}\n".format(self.accurate_offsets.numpy()))
            f.write("Offset scale used: {:.6f}\n".format(self.offset_scale))
