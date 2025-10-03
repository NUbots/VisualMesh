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


import math
import tensorflow as tf
from training.op import difference_visual_mesh, unmap_visual_mesh
from training.projection import project

class Anchorless:
    def __init__(self, sigma, mesh, geometry, offset_scale=1.0, **config):
        """
        CenterNet-style ground-truth label generator in Visual Mesh space.
        Produces:
          - Y: [num_nodes, 1] heatmap with hard 1.0 peaks at nearest node(s)
          - center_indices: [num_targets] int vector of peak node ids
        """
        self.sigma = sigma
        self.offset_scale = offset_scale
        self.mesh_model = mesh["model"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]

    def features(self):
        return {
            "seeker/targets": tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
        }

    def __call__(self, image, Hoc, V, valid, **features):
        targets = features["seeker/targets"]
        projection = features["lens/projection"]
        focal_length = features["lens/focal_length"]
        centre = features["lens/centre"]
        k = features["lens/k"]
        dims = tf.shape(image)[:2]

        # No targets → pure background heatmap
        if tf.size(targets) == 0:
            return {
                "Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32),
                "center_indices": tf.zeros((0,), dtype=tf.int64),
            }

        # Filter to on-screen targets
        target_dirs, _ = tf.linalg.normalize(targets, axis=-1)
        px = tf.cast(tf.round(project(target_dirs, dims, projection, focal_length, centre, k)), tf.int32)
        on_screen = tf.reduce_all(tf.logical_and(px >= 0, px < tf.expand_dims(dims, 0)), axis=-1)
        targets = tf.gather(targets, tf.squeeze(tf.where(on_screen), axis=-1))

        if tf.size(targets) == 0:
            return {
                "Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32),
                "center_indices": tf.zeros((0,), dtype=tf.int64),
            }

        # Transform targets into observation-plane camera space and keep only below the camera
        uOCo, _ = tf.linalg.normalize(tf.einsum("ij,kj->ki", Hoc[:3, :3], targets), axis=-1)
        uOCo = tf.gather(uOCo, tf.squeeze(tf.where(uOCo[:, 2] < 0), axis=-1))

        if tf.size(uOCo) == 0:
            return {
                "Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32),
                "center_indices": tf.zeros((0,), dtype=tf.int64),
            }

        # Visual Mesh args
        height = Hoc[2, 3]
        args = {"model": self.mesh_model, "height": height, "geometry": self.geometry, "radius": self.radius}

        # Get our vectors in nm coordinates
        mesh_nm = unmap_visual_mesh(V, **args)          # [N_nodes, 2]
        target_nm = unmap_visual_mesh(uOCo, **args)     # [N_tgts, 2]

        # Replicate out the points so they are the same size
        n_nodes = tf.shape(mesh_nm)[0]
        n_targets = tf.shape(target_nm)[0]

        # Pairwise nm differences node↔target using Visual Mesh geometry
        m = tf.reshape(tf.tile(mesh_nm, (1, n_targets)), (-1, 2))  # [N_nodes * N_tgts, 2]
        t = tf.tile(target_nm, (n_nodes, 1))                       # [N_nodes * N_tgts, 2]
        diff = tf.reshape(difference_visual_mesh(t, m, **args), (n_nodes, n_targets, 2))

        # Distances in mesh "cells"
        d_cells = tf.norm(diff, axis=-1)                            # [N_nodes, N_tgts]

        # Gaussian splat per target, then max over targets
        gauss = tf.exp(-0.5 * tf.square(d_cells / self.sigma))      # [N_nodes, N_tgts]
        H_gt = tf.reduce_max(gauss, axis=-1)                         # [N_nodes]

        # Hard 1.0 peaks at nearest node(s) to each target
        nearest_idx = tf.argmin(d_cells, axis=0)                     # [N_tgts]
        H_gt = tf.tensor_scatter_nd_update(
            H_gt,
            indices=tf.reshape(nearest_idx, (-1, 1)),
            updates=tf.ones_like(nearest_idx, dtype=H_gt.dtype),
        )

        # Final shapes
        H_gt = tf.expand_dims(H_gt, -1)                              # [N_nodes, 1]

        # Offsets: only at center nodes
        center_nm  = tf.gather(mesh_nm, nearest_idx)             # [T,2]
        off_centers = difference_visual_mesh(target_nm, center_nm, **args)  # [T,2]
        off_centers = off_centers / self.offset_scale            # optional scaling to ~[-1,1]

        # Scatter into a node-wise offset map (zeros elsewhere).
        # If multiple targets share a node, average their offsets.
        idx = tf.cast(nearest_idx, tf.int32)
        sums = tf.math.unsorted_segment_sum(off_centers, idx, n_nodes)                 # [N,2]
        counts = tf.math.unsorted_segment_sum(tf.ones_like(off_centers[:, :1]), idx, n_nodes)  # [N,1]
        offsets = tf.math.divide_no_nan(sums, counts)                            # [N,2]

        # Pack into single Y: [N, 3]  (channel 0=heatmap, 1:3=offsets)
        Y = tf.concat([H_gt, offsets], axis=-1)
        return {
            "Y": Y,
            "center_indices": nearest_idx,
        }
