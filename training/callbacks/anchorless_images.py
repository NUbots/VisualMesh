
import hashlib
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from training.op import map_visual_mesh, unmap_visual_mesh
from training.projection import project


class AnchorlessImages(tf.keras.callbacks.Callback):
    def __init__(self, output_path, dataset, model, max_distance, geometry, radius, sigma, offset_scale=1.0, use_offsets=True):
        super(AnchorlessImages, self).__init__()
        self.offset_scale = float(offset_scale)
        self.use_offsets = use_offsets

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
        px_filtered = tf.gather(px, tf.squeeze(tf.where(on_screen), axis=1))
        pred_filtered = tf.gather(tf.squeeze(heatmap_pred, axis=-1), tf.squeeze(tf.where(on_screen), axis=1))
        true_filtered = tf.gather(tf.squeeze(heatmap_true, axis=-1), tf.squeeze(tf.where(on_screen), axis=1))

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
        px_valid = tf.gather(px_expanded, tf.squeeze(tf.where(valid_mask), axis=1))

        # Replicate values for each 3x3 position
        pred_expanded = tf.repeat(pred_amplified, dot_size * dot_size)
        true_expanded = tf.repeat(true_amplified, dot_size * dot_size)
        pred_valid = tf.gather(pred_expanded, tf.squeeze(tf.where(valid_mask), axis=1))
        true_valid = tf.gather(true_expanded, tf.squeeze(tf.where(valid_mask), axis=1))

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

    def _nm_to_px(self, pts_nm, Hoc, lens, dims):
        # nm -> observation plane -> camera -> pixels
        uPCo = map_visual_mesh(pts_nm, height=Hoc[2, 3], **self.map_args)    # [M,3]
        uPCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uPCo)                     # [M,3]
        px_rc = tf.cast(
            tf.round(project(uPCc, dims, lens["projection"], lens["focal_length"], lens["centre"], lens["k"])),
            tf.int32
        )  # [M,2] = [row(y), col(x)]

        H, W = int(dims[0]), int(dims[1])

        # Clip in row/col space first
        r = tf.clip_by_value(px_rc[:, 0], 0, H - 1)
        c = tf.clip_by_value(px_rc[:, 1], 0, W - 1)

        # Convert to (x,y) for OpenCV
        px_xy = tf.stack([c, r], axis=-1)  # [M,2] = (x, y)
        return px_xy.numpy().astype(np.int32)


    def _draw_circles_cv(self, rgb_np, centers_px, color_bgr, radius_px=6, thickness=2):
        """
        Draw circles with OpenCV at integer pixel centers. centers_px: np.array [M,2] (x,y).
        """
        if centers_px.size == 0:
            return rgb_np
        for (x, y) in centers_px:
            cv2.circle(rgb_np, (int(x), int(y)), int(radius_px), color_bgr, int(thickness), lineType=cv2.LINE_AA)
        return rgb_np

    def _split_heatmap_channels(self, heatmap_true, heatmap_pred):
        """Split heatmap tensors into heatmap and offset channels."""
        # Accept both [N,1] (heat only) and [N,3] (heat + offsets)
        def _split_gt(Y):
            if Y.shape[-1] == 1:
                return Y, tf.zeros([Y.shape[0], 2], dtype=Y.dtype)  # no offsets provided
            return Y[:, 0:1], Y[:, 1:3]

        def _split_pred(P):
            if P.shape[-1] == 1:
                # Already a prob or a sigmoid head; treat as prob for viz and no offsets
                return P, tf.zeros([P.shape[0], 2], dtype=P.dtype), True  # (hm_prob, off, is_prob=True)
            # 3 channels: assume channel-0 is logit, 1:2 offsets linear
            return P[:, 0:1], P[:, 1:3], False  # (hm_*, off, is_prob=False (i.e., logits))

        hm_true, off_true = _split_gt(heatmap_true)
        hm_pred_raw, off_pred, pred_is_prob = _split_pred(heatmap_pred)

        # Convert predictions to probabilities for visualization
        hm_pred_prob = tf.sigmoid(hm_pred_raw) if not pred_is_prob else tf.clip_by_value(hm_pred_raw, 0.0, 1.0)

        return hm_true, off_true, hm_pred_prob, off_pred

    def _create_detection_ring_overlay(self, Hoc, lens, dims):
        """Create detection range ring overlay."""
        f = lens["focal_length"]
        uRCo = self._ring(angle=tf.atan(self.max_distance / Hoc[2, 3]), width=2, f=f)
        uRCc = tf.einsum("ij,ki->kj", Hoc[:3, :3], uRCo)
        ring_px = tf.cast(tf.round(project(uRCc, dims, lens["projection"], f, lens["centre"], lens["k"])), tf.int32)

        ring_on_screen = tf.reduce_all(tf.logical_and(ring_px >= 0, ring_px < tf.expand_dims(dims, 0)), axis=-1)
        ring_px_filtered = tf.gather(ring_px, tf.squeeze(tf.where(ring_on_screen), axis=1))

        if tf.size(ring_px_filtered) > 0:
            ring_overlay = tf.scatter_nd(
                ring_px_filtered,
                tf.ones_like(ring_px_filtered[:, 0], dtype=tf.float32),
                dims
            )
            ring_overlay = tf.clip_by_value(
                tf.einsum("ij,k->ijk", ring_overlay, tf.constant([1.0, 1.0, 1.0])),
                0.0, 1.0
            )
        else:
            ring_overlay = tf.zeros((*dims, 3), dtype=tf.float32)

        return ring_overlay

    def _detect_and_draw_centers(self, hm_true, off_true, hm_pred_prob, off_pred, nm, Hoc, lens, dims, output_img):
        # Flatten
        hm_true_flat = tf.reshape(hm_true, [-1])     # [N]
        hm_pred_flat = tf.reshape(hm_pred_prob, [-1])# [N]

        # 1) Pick GT centers with a threshold, not equality
        true_where = tf.where(hm_true_flat >= 1.0)  # [K,1]
        true_idx = tf.cast(tf.reshape(true_where, [-1]), tf.int32)  # [K]

        # 2) Compute GT center pixels (node + optional offset)
        if tf.size(true_idx) > 0:
            nm_gt  = tf.gather(nm, true_idx)         # [K,2]
            if self.use_offsets:
                off_gt = tf.gather(off_true, true_idx)   # [K,2]
                center_nm_true = nm_gt + off_gt * self.offset_scale
            else:
                center_nm_true = nm_gt  # Just use the node position
            true_px = self._nm_to_px(center_nm_true, Hoc, lens, dims)  # np.int32 [K,2]
        else:
            true_px = np.zeros((0, 2), dtype=np.int32)

        # 3) Pred centers: take top-K probs (K=GT count; fallback to 1)
        K = int(tf.size(true_idx).numpy())
        if K <= 0:
            K = 1

        topk = tf.math.top_k(hm_pred_flat, k=K, sorted=True)
        pred_idx = tf.cast(topk.indices, tf.int32)  # [K]

        nm_pr  = tf.gather(nm, pred_idx)        # [K,2]
        if self.use_offsets:
            off_pr = tf.gather(off_pred, pred_idx)  # [K,2]
            center_nm_pred = nm_pr + off_pr * self.offset_scale
        else:
            center_nm_pred = nm_pr  # Just use the node position
        pred_px = self._nm_to_px(center_nm_pred, Hoc, lens, dims)

        # 4) Draw
        output_np = (output_img.numpy() * 255.0).astype(np.uint8)
        output_np = self._draw_circles_cv(output_np, true_px, (255, 255, 255), radius_px=6, thickness=2)  # white GT
        output_np = self._draw_circles_cv(output_np, pred_px, (0, 0, 0), radius_px=6, thickness=2)        # black predictions
        return tf.convert_to_tensor(output_np / 255.0, dtype=tf.float32)


    def image(self, img, heatmap_pred, heatmap_true, Hoc, lens, nm):
        """Generate visualization image with GT (white) and Pred (black) centers overlaid."""
        # Image preprocessing
        img_hash = hashlib.md5()
        img_hash.update(img)
        img_hash = img_hash.digest()

        img = tf.image.convert_image_dtype(
            tf.image.decode_image(img, channels=3, expand_animations=False),
            tf.float32,
        )
        dims = img.shape[:2]   # (H, W)

        # Split heatmap channels
        hm_true, off_true, hm_pred_prob, off_pred = self._split_heatmap_channels(heatmap_true, heatmap_pred)

        # Create heatmap overlay
        heatmap_overlay = self._heatmap_overlay(
            heatmap_pred=hm_pred_prob,  # [N,1] probabilities
            heatmap_true=hm_true,       # [N,1] GT heat
            nm=nm, Hoc=Hoc, lens=lens, dims=dims
        )

        # Create detection ring overlay
        ring_overlay = self._create_detection_ring_overlay(Hoc, lens, dims)

        # Blend base image with overlays
        output = self._blend(self._blend(img, ring_overlay), heatmap_overlay)

        # Detect and draw center points
        output = self._detect_and_draw_centers(hm_true, off_true, hm_pred_prob, off_pred, nm, Hoc, lens, dims, output)

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
