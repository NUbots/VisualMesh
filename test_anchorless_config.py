#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/angy-mei/VisualMesh')

# Test basic imports
print("Testing basic imports...")
try:
    from training.dataset.label import Classification, Anchorless, Seeker
    print("✓ Label imports successful")
except Exception as e:
    print(f"✗ Label imports failed: {e}")

try:
    from training.callbacks import ClassificationImages, AnchorlessImages, SeekerImages
    print("✓ Callback imports successful")
except Exception as e:
    print(f"✗ Callback imports failed: {e}")

try:
    from training.loss import FocalLoss, AnchorlessLoss, SeekerLoss, ClassificationLoss
    print("✓ Loss imports successful")
except Exception as e:
    print(f"✗ Loss imports failed: {e}")

# Test configuration functions
print("\nTesting configuration functions...")

# Mock minimal config for testing
test_config = {
    "label": {
        "type": "Anchorless",
        "config": {
            "sigma": 0.1
        }
    },
    "training": {
        "validation": {
            "progress_images": 4
        }
    },
    "dataset": {
        "config": {}
    },
    "projection": {
        "config": {
            "mesh": {
                "model": "XYGRID8",
                "max_distance": 5.0
            },
            "geometry": {
                "shape": "SPHERE",
                "radius": 1.0
            }
        }
    }
}

try:
    from training.flavour.loss import Loss
    loss_fn = Loss(test_config)
    print("✓ Loss configuration successful")
except Exception as e:
    print(f"✗ Loss configuration failed: {e}")

try:
    from training.flavour.test_metrics import TestMetrics
    metrics = TestMetrics(test_config)
    print("✓ Test metrics configuration successful")
except Exception as e:
    print(f"✗ Test metrics configuration failed: {e}")

print("\nAll basic tests completed!")
