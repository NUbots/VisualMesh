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

from ..metrics import *
from ..metrics.test import SeekerHourglass


def Metrics(config):

    if config["label"]["type"] == "Classification":

        classes = config["label"]["config"]["classes"]
        metrics = [
            AveragePrecision("metrics/average_precision", len(classes)),
            AverageRecall("metrics/average_recall", len(classes)),
        ]
        for i, c in enumerate(classes):
            metrics.append(ClassPrecision("metrics/{}_precision".format(c["name"]), i, len(classes)))
            metrics.append(ClassRecall("metrics/{}_recall".format(c["name"]), i, len(classes)))

        return metrics

    elif config["label"]["type"] == "Seeker":
        return [
            SeekerPrecision("metrics/precision75", 0.75),
            SeekerRecall("metrics/recall75", 0.75),
            SeekerStdDev("metrics/stddev75", 0.75),
            SeekerPrecision("metrics/precision50", 0.5),
            SeekerRecall("metrics/recall50", 0.5),
            SeekerStdDev("metrics/stddev50", 0.5),
            SeekerHourglass("metrics/hourglass"),
        ]

    elif config["label"]["type"] == "Anchorless":
        # Anchorless detection metrics based on heatmap peak detection
        use_offsets = config["label"]["config"].get("use_offsets", True)

        metrics = [
            # Traditional PR curves at different thresholds
            AnchorlessPrecision("metrics/precision95", 0.95, use_offsets),
            AnchorlessRecall("metrics/recall95", 0.95, use_offsets),
            AnchorlessPrecision("metrics/precision75", 0.75, use_offsets),
            AnchorlessRecall("metrics/recall75", 0.75, use_offsets),
            AnchorlessPrecision("metrics/precision50", 0.50, use_offsets),
            AnchorlessRecall("metrics/recall50", 0.50, use_offsets),
            AnchorlessPrecision("metrics/precision25", 0.25, use_offsets),
            AnchorlessRecall("metrics/recall25", 0.25, use_offsets),
            AnchorlessPrecision("metrics/precision10", 0.10, use_offsets),
            AnchorlessRecall("metrics/recall10", 0.10, use_offsets),

            # Peak-to-peak distance evaluation metrics
            AnchorlessPeakAccuracy("metrics/peak_accuracy"),
            AnchorlessPeakNodeDistance("metrics/peak_distance_1", distance_threshold=1),
            AnchorlessPeakNodeDistance("metrics/peak_distance_3", distance_threshold=3),
            AnchorlessPeakNodeDistance("metrics/peak_distance_5", distance_threshold=5),
            AnchorlessPeakNodeDistance("metrics/peak_distance_10", distance_threshold=10),
        ]

        # Add offset accuracy metrics if offsets are enabled
        if use_offsets:
            metrics.extend([
                AnchorlessOffsetAccuracy("metrics/offset_accuracy_01", error_threshold=0.1),
                AnchorlessOffsetAccuracy("metrics/offset_accuracy_05", error_threshold=0.5),
                AnchorlessOffsetAccuracy("metrics/offset_accuracy_10", error_threshold=1.0),
            ])

        return metrics

    else:
        raise RuntimeError("Cannot create metrics, {} is not a supported  type".format(config["label"]["type"]))
