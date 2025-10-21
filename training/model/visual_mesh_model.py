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
from training.layer import GraphConvolution, DepthwiseSeparableGraphConvolution
from tensorflow.keras.layers import Concatenate

class VisualMeshModel(tf.keras.Model):
    def _apply_variables(self, option):
        if isinstance(option, str):
            return self.output_dims if option == "$output_dims" else option
        elif isinstance(option, dict):
            return {k: self._apply_variables(v) for k, v in option.items()}
        elif isinstance(option, list):
            return [self._apply_variables(v) for v in option]   # <-- list, not set
        else:
            return option

    def _make_op(self, op, options):
        # If options are none, make an empty argument list
        options = {} if options is None else options

        # There are some variables you can put into the options, so lets apply them
        options = self._apply_variables(options)

        if op == "GraphConvolution":
            return GraphConvolution(**options)
        elif op == "DepthwiseSeparableGraphConvolution":
            return DepthwiseSeparableGraphConvolution(**options)
        elif hasattr(tf.keras.layers, op):
            return getattr(tf.keras.layers, op)(**options)
        else:
            raise RuntimeError("Model attempted to make an unknown operation type: {}".format(op))

    def _topological_sort(self, v, graph, visited=None, stack=None):
        visited = set() if visited is None else visited
        stack = list() if stack is None else stack

        visited.add(v[0])
        for k in graph[v[0]]:
            if k not in visited:
                self._topological_sort((k, graph[k]), graph, visited, stack)

        if v[0] not in ["X", "G"]:
            stack.append(v[0])

        return stack

    def __init__(self, structure, output_dims):
        super(VisualMeshModel, self).__init__()

        # Build up the graph of operations
        self.output_dims = output_dims
        self.ops = {k: (self._make_op(v["op"], v.get("options")), v["inputs"]) for k, v in structure.items()}

        # Perform a topological sort to ensure that the results are done in order
        self.stages = self._topological_sort(
            ("output", structure["output"]["inputs"]),
            {**{k: v["inputs"] for k, v in structure.items()}, "X": [], "G": []},
        )

    def call(self, X, training=False):
        # Split inputs
        logits, G = X
        results = {"X": logits, "G": G}

        # Execute graph in topological order
        for s in self.stages:
            op, inputs = self.ops[s]
            in_tensors = [results[i] for i in inputs]

            # Keras Concatenate expects a single list arg
            if isinstance(op, tf.keras.layers.Layer) and isinstance(op, Concatenate):
                results[s] = op(in_tensors)           # pass list
            else:
                results[s] = op(*in_tensors)          # pass positional

        out = results["output"]   # shape [N, C]
        out = out[:-1, :]         # drop the last node, keep rank-2
        return out
