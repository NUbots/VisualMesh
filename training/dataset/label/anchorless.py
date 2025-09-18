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

# ...existing code...

import math
import tensorflow as tf
from training.op import difference_visual_mesh, map_visual_mesh, unmap_visual_mesh
from training.projection import project


class Anchorless:
    def __init__(self, sigma, mesh, geometry, **config):
        """
        Anchorless label generator for CenterNet-style heatmap detection.
        Uses Visual Mesh coordinate system properly.
        """
        self.sigma = sigma

        # Grab our relevant fields
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

        # If no targets, return zeros
        if tf.size(targets) == 0:
            return {"Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32)}

        # Use same filtering as Seeker (on-screen and ratio checks)
        target_dirs, _ = tf.linalg.normalize(targets, axis=-1)
        px = tf.cast(tf.round(project(target_dirs, dims, projection, focal_length, centre, k)), tf.int32)
        on_screen = tf.reduce_all(tf.logical_and(px >= 0, px < tf.expand_dims(dims, 0)), axis=-1)
        targets = tf.gather(targets, tf.squeeze(tf.where(on_screen), axis=-1))

        if tf.size(targets) == 0:
            return {"Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32)}

        # Transform targets to observation plane space (same as Seeker)
        uOCo, l = tf.linalg.normalize(tf.einsum("ij,kj->ki", Hoc[:3, :3], targets), axis=-1)
        uOCo = tf.gather(uOCo, tf.squeeze(tf.where(uOCo[:, 2] < 0), axis=-1))

        if tf.size(uOCo) == 0:
            return {"Y": tf.zeros((tf.shape(V)[0], 1), dtype=tf.float32)}

        # Get mesh arguments
        height = Hoc[2, 3]
        args = {"model": self.mesh_model, "height": height, "geometry": self.geometry, "radius": self.radius}

        # Convert to Visual Mesh coordinates (same as Seeker)
        mesh_nm = unmap_visual_mesh(V, **args)
        target_nm = unmap_visual_mesh(uOCo, **args)

        # Use difference_visual_mesh to get proper distances in Visual Mesh space
        n_nodes = tf.shape(mesh_nm)[0]
        n_targets = tf.shape(target_nm)[0]

        # Replicate points (same as Seeker)
        m = tf.reshape(tf.tile(mesh_nm, (1, n_targets)), (-1, 2))
        t = tf.tile(target_nm, (n_nodes, 1))

        # Calculate differences using Visual Mesh geometry
        diff = tf.reshape(difference_visual_mesh(t, m, **args), (n_nodes, n_targets, 2))

        # Calculate distances in Visual Mesh space
        distances = tf.norm(diff, axis=-1)  # [N, M]

        # Generate Gaussian heatmap using Visual Mesh distances
        gaussian_values = tf.exp(-(distances ** 2) / (2 * self.sigma ** 2))
        heatmap = tf.reduce_max(gaussian_values, axis=-1, keepdims=True)  # [N, 1]

        return {"Y": heatmap}
