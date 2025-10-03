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

from .anchorless_confusion_base import AnchorlessConfusionBase


class AnchorlessPrecision(AnchorlessConfusionBase):
    def __init__(self, name, threshold, use_offsets=True, **kwargs):
        super(AnchorlessPrecision, self).__init__(name, threshold, use_offsets, **kwargs)

    def result(self):
        # True positives (predicted and labelled true)
        tp = tf.cast(self.confusion[1, 1], self.dtype)
        # For all labels where positive was predicted (all positives)
        p = tf.cast(tf.reduce_sum(self.confusion[:, 1]), self.dtype)

        # Return 0 if no predictions were made, otherwise compute precision
        return tf.cond(
            tf.greater(p, 0),
            lambda: tp / p,
            lambda: tf.constant(0.0, dtype=self.dtype)
        )
