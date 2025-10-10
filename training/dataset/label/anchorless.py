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
from training.projection import unproject

class Anchorless:
    def __init__(self, sigma, mesh, geometry, offset_scale=1.0, use_offsets=True, **config):
        """
        CenterNet-style ground-truth label generator in Visual Mesh space.
        Produces:
          - Y: [num_nodes, 3] with [heatmap, offset_x, offset_y] channels
          - center_indices: [num_targets] int vector of peak node ids
        """
        self.sigma = sigma
        self.offset_scale = offset_scale
        self.use_offsets = use_offsets
        self.mesh_model = mesh["model"]
        self.geometry = tf.constant(geometry["shape"], dtype=tf.string, name="GeometryType")
        self.radius = geometry["radius"]

    @staticmethod
    def compute_per_node_scales(mesh_nm, valid, graph_indices):
        """
        Compute per-node local scale factors using mesh topology.
        
        This uses the Visual Mesh neighbor structure to efficiently compute mean distances
        to neighbors for each node, avoiding O(N²) complexity.

        Args:
            mesh_nm: [N, 2] - mesh node positions in nm coordinates  
            valid: [N] - boolean mask for valid nodes
            graph_indices: [N, K] - neighbor indices for each node

        Returns:
            [N] - per-node scale factors (mean distance to neighbors)
        """
        n_nodes = tf.shape(mesh_nm)[0]
        
        # Handle edge case where no nodes exist
        def empty_case():
            return tf.constant([], dtype=tf.float32)
        
        def normal_case():
            # Handle both scalar and vector valid masks
            valid_shape = tf.shape(valid)
            is_scalar = tf.equal(tf.size(valid_shape), 0)
            
            def scalar_case():
                # If valid is a scalar, broadcast it to all nodes
                return tf.fill([n_nodes], tf.cast(valid, tf.float32))
            
            def vector_case():
                # If valid is already a vector, just cast it  
                return tf.cast(valid, tf.float32)
            
            valid_mask = tf.cond(is_scalar, scalar_case, vector_case)  # [N]
            
            n_neighbors = tf.shape(graph_indices)[1]  # K
            
            # Get positions for each node and its neighbors
            # mesh_nm: [N, 2], graph_indices: [N, K]
            neighbor_positions = tf.gather(mesh_nm, graph_indices)  # [N, K, 2]
            
            # Compute distances to each neighbor
            node_positions = tf.expand_dims(mesh_nm, 1)  # [N, 1, 2]
            distances = tf.norm(neighbor_positions - node_positions, axis=-1)  # [N, K]
            
            # Handle invalid neighbors (negative indices in graph mean no neighbor)
            valid_neighbors = tf.cast(graph_indices >= 0, tf.float32)  # [N, K]
            
            # Compute mean distance to valid neighbors for each node
            sum_distances = tf.reduce_sum(distances * valid_neighbors, axis=1)  # [N]
            count_neighbors = tf.maximum(tf.reduce_sum(valid_neighbors, axis=1), 1.0)  # [N]
            per_node_scales = sum_distances / count_neighbors  # [N]
            
            # For nodes with no valid neighbors, use global average
            has_neighbors = tf.cast(count_neighbors > 1.0, tf.float32)
            global_scale = tf.reduce_mean(per_node_scales * has_neighbors) / tf.maximum(tf.reduce_mean(has_neighbors), 1e-6)
            per_node_scales = tf.where(has_neighbors > 0.5, per_node_scales, global_scale)
            
            # Ensure minimum scale to avoid division by very small numbers
            per_node_scales = tf.maximum(per_node_scales, 1e-4)
            
            # For invalid nodes, use the global scale
            scales = tf.where(valid_mask > 0.5, per_node_scales, global_scale)
            return scales
        
        return tf.cond(tf.equal(n_nodes, 0), empty_case, normal_case)

    def features(self):
        return {
            "anchorless/target_pixels": tf.io.FixedLenSequenceFeature([2], tf.float32, allow_missing=True),
        }

    def __call__(self, image, Hoc, V, G, valid, **features):
        target_pixels = features["anchorless/target_pixels"]
        projection = features["lens/projection"]
        focal_length = features["lens/focal_length"]
        centre = features["lens/centre"]
        k = features["lens/k"]
        dims = tf.shape(image)[:2]
        n_nodes = tf.shape(V)[0]

        Y = None
        center_indices = None

        # No targets → pure background heatmap
        if tf.size(target_pixels) == 0:
            Y = tf.zeros((n_nodes, 3), dtype=tf.float32)
            center_indices = tf.zeros((0,), dtype=tf.int64)
        else:
            # Filter pixels to on-screen region (if not already done)
            px_int = tf.cast(tf.round(target_pixels), tf.int32)
            on_screen = tf.reduce_all(tf.logical_and(px_int >= 0, px_int < tf.expand_dims(dims, 0)), axis=-1)
            target_pixels = tf.gather(target_pixels, tf.squeeze(tf.where(on_screen), axis=-1))

            if tf.size(target_pixels) == 0:
                Y = tf.zeros((n_nodes, 3), dtype=tf.float32)
                center_indices = tf.zeros((0,), dtype=tf.int64)
            else:
                # Unproject pixels to unit vectors in camera space
                camera_rays = unproject(target_pixels, dims, projection, focal_length, centre, k)
                
                # Transform to observation-plane camera space and filter below camera
                uOCo, _ = tf.linalg.normalize(tf.einsum("ij,kj->ki", Hoc[:3, :3], camera_rays), axis=-1)
                uOCo = tf.gather(uOCo, tf.squeeze(tf.where(uOCo[:, 2] < 0), axis=-1))

                if tf.size(uOCo) == 0:
                    Y = tf.zeros((n_nodes, 3), dtype=tf.float32)
                    center_indices = tf.zeros((0,), dtype=tf.int64)
                else:
                    # We have valid targets to process
                    # Visual Mesh args
                    height = Hoc[2, 3]
                    args = {"model": self.mesh_model, "height": height, "geometry": self.geometry, "radius": self.radius}

                    # Get our vectors in nm coordinates
                    mesh_nm = unmap_visual_mesh(V, **args)          # [N_nodes, 2]
                    target_nm = unmap_visual_mesh(uOCo, **args)     # [N_tgts, 2]

                    # Replicate out the points so they are the same size
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

                    if self.use_offsets:
                        # Compute per-node local scales based on mesh topology
                        per_node_scales = self.compute_per_node_scales(mesh_nm, valid, G)  # [N]

                        # Offsets: only at the single peak node per target
                        # nearest_idx: [T] (node id for each target)
                        center_nm   = tf.gather(mesh_nm, nearest_idx)                         # [T,2]
                        off_centers = difference_visual_mesh(target_nm, center_nm, **args)    # [T,2]

                        # Normalize by per-node local scale to make offsets "fraction of a local cell"
                        # Then apply global offset_scale to control overall magnitude for visualization/training
                        local_scales = tf.gather(per_node_scales, nearest_idx)               # [T] - scale for each target's node
                        local_scales_2d = tf.expand_dims(local_scales, -1)                   # [T, 1] for broadcasting  
                        off_centers = off_centers / (local_scales_2d * self.offset_scale)    # [T,2] - normalized by both scales
                        
                        # Clip to [-1, 1] range to match tanh activation
                        off_centers = tf.clip_by_value(off_centers, -1.0, 1.0)              # [T,2] - bounded offsets

                        # Scatter the per-target offsets straight into a dense [N,2] map.
                        # Any rare duplicate node ids will be "last write wins".
                        offsets = tf.tensor_scatter_nd_update(
                            tf.zeros((n_nodes, 2), dtype=H_gt.dtype),
                            tf.expand_dims(tf.cast(nearest_idx, tf.int32), axis=1),  # indices: [[j0],[j1],...]
                            off_centers,
                        )  # [N_nodes, 2]

                        # Pack labels: [heatmap, offset_x, offset_y]
                        Y = tf.concat([H_gt, offsets], axis=-1)                    # [N_nodes, 3]

                    else:
                        zeros = tf.zeros((n_nodes, 2), dtype=H_gt.dtype)
                        Y = tf.concat([H_gt, zeros], axis=-1)

                        # No need to compute scales when not using offsets

                    center_indices = tf.cast(nearest_idx, tf.int64)

        # For now, keep the same output format as before
        # Per-node scales can be recomputed at inference time using the same mesh
        return {
            "Y": Y,
            "center_indices": center_indices,
        }
