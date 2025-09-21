# ...existing code...

import hashlib
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from training.op import map_visual_mesh, unmap_visual_mesh
from training.projection import project


class AnchorlessImages(tf.keras.callbacks.Callback):
    def __init__(self, output_path, dataset, model, max_distance, geometry, radius, sigma):
        super(AnchorlessImages, self).__init__()

        self.max_distance = max_distance
        self.radius = radius
        self.sigma = sigma
        self.map_args = {"model": model, "geometry": geometry, "radius": radius}
        self.writer = tf.summary.create_file_writer(os.path.join(output_path, "images"))

        # Load the dataset and extract a single record from it
        for d in dataset:
            data = d

        self.X = data["X"]
        self.Y = data["Y"]
        self.G = data["G"]
        self.Hoc = tf.reshape(data["Hoc"], (-1, 4, 4))
        self.img = tf.reshape(data["jpg"], (-1,))
        self.lens = {
            "projection": tf.reshape(data["lens"]["projection"], (-1,)),
            "focal_length": tf.reshape(data["lens"]["focal_length"], (-1,)),
            "centre": tf.reshape(data["lens"]["centre"], (-1, 2)),
            "k": tf.reshape(data["lens"]["k"], (-1, 2)),
            "fov": tf.reshape(data["lens"]["fov"], (-1,)),
        }

        # Work out the data ranges
        cs = [0] + np.cumsum(data["n"]).tolist()
        self.ranges = list(zip(cs, cs[1:]))
        self.nm = tf.concat(
            [
                unmap_visual_mesh(data["V"][r[0] : r[1]], height=self.Hoc[i][2, 3], **self.map_args)
                for i, r in enumerate(self.ranges)
            ],
            axis=0,
        )

    def _heatmap_overlay(self, heatmap_pred, heatmap_true, nm, Hoc, lens, dims):
        """Create heatmap visualization overlay with custom color scheme."""

        # Project mesh points to pixel coordinates
        uPCo = map_visual_mesh(nm, height=Hoc[2, 3], **self.map_args)
        uPCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uPCo)
        px = tf.cast(tf.round(project(uPCc, dims, lens["projection"], lens["focal_length"], lens["centre"], lens["k"])), tf.int32)

        # Filter to on-screen points
        on_screen = tf.reduce_all(tf.logical_and(px >= 0, px < tf.expand_dims(dims, 0)), axis=-1)
        px_filtered = tf.gather(px, tf.squeeze(tf.where(on_screen), axis=-1))
        pred_filtered = tf.gather(tf.squeeze(heatmap_pred, axis=-1), tf.squeeze(tf.where(on_screen), axis=-1))
        true_filtered = tf.gather(tf.squeeze(heatmap_true, axis=-1), tf.squeeze(tf.where(on_screen), axis=-1))

        if tf.size(px_filtered) == 0:
            return tf.zeros((*dims, 3), dtype=tf.float32)

        # Amplify heatmap values for better visibility
        pred_amplified = tf.clip_by_value(pred_filtered * 3.0, 0.0, 1.0)
        true_amplified = tf.clip_by_value(true_filtered * 3.0, 0.0, 1.0)

        # Create 3x3 dot positions for each point
        dot_size = 3
        offset_range = tf.range(-(dot_size//2), dot_size//2 + 1)
        dx, dy = tf.meshgrid(offset_range, offset_range)
        offsets = tf.stack([tf.reshape(dy, [-1]), tf.reshape(dx, [-1])], axis=1)  # [9, 2]

        # Expand points to 3x3 squares
        px_expanded = tf.expand_dims(px_filtered, 1) + tf.expand_dims(offsets, 0)  # [N, 9, 2]
        px_expanded = tf.reshape(px_expanded, [-1, 2])  # [N*9, 2]

        # Filter out points that go outside image bounds
        valid_mask = tf.reduce_all(tf.logical_and(px_expanded >= 0, px_expanded < tf.expand_dims(dims, 0)), axis=-1)
        px_valid = tf.gather(px_expanded, tf.squeeze(tf.where(valid_mask), axis=-1))

        # Replicate values for each 3x3 position
        pred_expanded = tf.repeat(pred_amplified, dot_size * dot_size)
        true_expanded = tf.repeat(true_amplified, dot_size * dot_size)
        pred_valid = tf.gather(pred_expanded, tf.squeeze(tf.where(valid_mask), axis=-1))
        true_valid = tf.gather(true_expanded, tf.squeeze(tf.where(valid_mask), axis=-1))

        if tf.size(px_valid) == 0:
            return tf.zeros((*dims, 3), dtype=tf.float32)

        # Ground Truth: White (peak) → Black (edges)
        # Create grayscale overlay where high values are white, low values are black
        gt_overlay = tf.scatter_nd(px_valid, true_valid, dims)
        gt_overlay = tf.stack([gt_overlay, gt_overlay, gt_overlay], axis=-1)  # RGB all same = grayscale

        # Prediction: Red (edges) → Yellow (center)
        # For red→yellow gradient: Red (1,0,0) at low values → Yellow (1,1,0) at high values
        # Red channel: always full intensity where there's any prediction
        # Green channel: scales with prediction value (creates the red→yellow gradient)
        pred_red = tf.where(
            tf.scatter_nd(px_valid, pred_valid, dims) > 0,
            1.0,  # Full red intensity where any prediction exists
            0.0
        )
        pred_green = tf.scatter_nd(px_valid, pred_valid, dims)  # Green scales with prediction value
        pred_blue = tf.zeros_like(pred_red)  # No blue for red→yellow gradient

        pred_overlay = tf.stack([pred_red, pred_green, pred_blue], axis=-1)

        # Prediction overrides ground truth
        # Where prediction exists (> 0), use prediction colors
        # Where prediction is zero, show ground truth
        pred_mask = tf.reduce_max(pred_overlay, axis=-1, keepdims=True)  # Mask where prediction exists

        # Final overlay: prediction overrides ground truth
        overlay = gt_overlay * (1.0 - tf.cast(tf.greater(pred_mask, 0.0), tf.float32)) + pred_overlay

        return tf.clip_by_value(overlay, 0.0, 1.0)

    def _blend(self, a, b):
        """Blend two images."""
        return a * (1.0 - tf.reduce_max(b, axis=-1, keepdims=True)) + b

    def image(self, img, heatmap_pred, heatmap_true, Hoc, lens, nm):
        """Generate visualization image."""

        # Hash of the image file for sorting later
        img_hash = hashlib.md5()
        img_hash.update(img)
        img_hash = img_hash.digest()

        # Decode the image and convert it to float32
        img = tf.image.convert_image_dtype(tf.image.decode_image(img, channels=3, expand_animations=False), tf.float32)
        dims = img.shape[:2]

        # Create heatmap overlay
        heatmap_overlay = self._heatmap_overlay(heatmap_pred, heatmap_true, nm, Hoc, lens, dims)

        # Draw detection ring (same as SeekerImages)
        f = lens["focal_length"]
        uRCo = self._ring(angle=tf.atan(self.max_distance / Hoc[2, 3]), width=2, f=f)
        uRCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uRCo)
        ring_px = tf.cast(tf.round(project(uRCc, dims, lens["projection"], f, lens["centre"], lens["k"])), tf.int32)

        # Filter ring to on-screen pixels
        ring_on_screen = tf.reduce_all(tf.logical_and(ring_px >= 0, ring_px < tf.expand_dims(dims, 0)), axis=-1)
        ring_px_filtered = tf.gather(ring_px, tf.squeeze(tf.where(ring_on_screen), axis=-1))

        if tf.size(ring_px_filtered) > 0:
            ring_overlay = tf.scatter_nd(ring_px_filtered, tf.ones_like(ring_px_filtered[:, 0], dtype=tf.float32), dims)
            ring_overlay = tf.clip_by_value(tf.einsum("ij,k->ijk", ring_overlay, tf.constant([1.0, 1.0, 1.0])), 0.0, 1.0)
        else:
            ring_overlay = tf.zeros((*dims, 3), dtype=tf.float32)

        # Blend all overlays
        output = self._blend(self._blend(img, ring_overlay), heatmap_overlay)

        return (img_hash, output)

    def _ring(self, angle, width, f):
        """Generate ring points (same as SeekerImages)."""
        px_angle = tf.math.reciprocal(f)
        pts = 2 * tf.cast(2.0 * math.pi * f * tf.math.tan(angle), dtype=tf.int32)

        theta = tf.linspace(angle - 0.5 * width * px_angle, angle + 0.5 * width * px_angle, 2 * width)
        z = -tf.cos(theta)
        w = tf.expand_dims(tf.sin(theta), 1)

        psi = tf.linspace(0.0, 2.0 * math.pi, pts)
        x = tf.expand_dims(tf.cos(psi), 0) * w
        y = tf.expand_dims(tf.sin(psi), 0) * w

        return tf.reshape(tf.stack([x, y, tf.broadcast_to(tf.expand_dims(z, axis=1), x.shape)], axis=-1), [-1, 3])

    def on_epoch_end(self, epoch, logs=None):
        """Generate and save visualization images."""

        # Make predictions
        predictions = self.model((self.X, self.G))

        # Generate images for each sample
        images = []
        for i, r in enumerate(self.ranges):
            images.append(
                self.image(
                    img=self.img[i].numpy(),
                    heatmap_pred=predictions[r[0] : r[1]],
                    heatmap_true=self.Y[r[0] : r[1]],
                    Hoc=self.Hoc[i],
                    lens={k: v[i] for k, v in self.lens.items()},
                    nm=self.nm[r[0] : r[1]],
                )
            )

        # Write out the images to tensorboard
        with self.writer.as_default():
            for i, img in enumerate(sorted(images, key=lambda image: image[0])):
                tf.summary.image("Anchorless Heatmaps/{}".format(i), tf.expand_dims(img[1], axis=0), step=epoch, max_outputs=1)
