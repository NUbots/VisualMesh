#!/usr/bin/env python3

import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tfrecord(output_file, input_files):

    with tf.io.TFRecordWriter(output_file) as writer:

        for image_file, mask_file, lens_file in tqdm(
            input_files,
            desc="Creating {}".format(os.path.basename(output_file)),
            leave=True,
            unit="files",
            dynamic_ncols=True,
        ):

            with open(lens_file, "r") as f:
                lens = yaml.safe_load(f)
            with open(image_file, "rb") as f:
                image = f.read()
            with open(mask_file, "rb") as f:
                mask = f.read()

            # Create the record
            writer.write(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": bytes_feature(image),
                            "mask": bytes_feature(mask),
                            "lens/projection": bytes_feature(lens["projection"].encode("utf-8")),
                            "lens/fov": float_feature(lens["fov"]),
                            "lens/focal_length": float_feature(lens["focal_length"]),
                            "lens/centre": float_list_feature(lens["centre"]),
                            "lens/k": float_list_feature(lens["k"]),
                            "Hoc": float_list_feature(np.array(lens["Hoc"]).flatten().tolist()),
                        }
                    )
                ).SerializeToString()
            )


def process_folder(input_path):
    """Process a single folder and return the files split into training, validation, and test sets."""

    image_files = glob(os.path.join(input_path, "image*.jpg"))
    mask_files = glob(os.path.join(input_path, "mask*.png"))
    lens_files = glob(os.path.join(input_path, "lens*.yaml"))

    # Extract which numbers are in each of the folders
    image_re = re.compile(r"image([^.]+)\.jpg$")
    mask_re = re.compile(r"mask([^.]+)\.png$")
    lens_re = re.compile(r"lens([^.]+)\.yaml$")
    image_nums = set([image_re.search(os.path.basename(f)).group(1) for f in image_files])
    mask_nums = set([mask_re.search(os.path.basename(f)).group(1) for f in mask_files])
    lens_nums = set([lens_re.search(os.path.basename(f)).group(1) for f in lens_files])
    common_nums = image_nums & mask_nums & lens_nums

    files = [
        (
            os.path.join(input_path, "image{}.jpg".format(n)),
            os.path.join(input_path, "mask{}.png".format(n)),
            os.path.join(input_path, "lens{}.yaml".format(n)),
        )
        for n in common_nums
    ]

    nf = len(files)

    # Define split ratios
    training_ratio = 0.45
    validation_ratio = 0.10

    # Calculate indices for splits
    training_end = round(nf * training_ratio)
    validation_end = round(nf * (training_ratio + validation_ratio))

    # Split the files
    training_files = files[0:training_end]
    validation_files = files[training_end:validation_end]
    test_files = files[validation_end:nf]

    return training_files, validation_files, test_files


if __name__ == "__main__":

    # Parse our command line arguments
    command = argparse.ArgumentParser(description="Utility for training a Visual Mesh network")
    command.add_argument("input_paths", nargs='+', help="Path(s) to the input folders")
    command.add_argument("output_path", action="store", help="Path to place the output tfrecord files")
    args = command.parse_args()
    input_paths = args.input_paths
    output_path = args.output_path

    # Process each input folder and collect files
    all_training_files = []
    all_validation_files = []
    all_test_files = []

    for input_path in input_paths:
        print(f"Processing folder: {input_path}")
        training_files, validation_files, test_files = process_folder(input_path)

        all_training_files.extend(training_files)
        all_validation_files.extend(validation_files)
        all_test_files.extend(test_files)

        print(f"  Training: {len(training_files)} files")
        print(f"  Validation: {len(validation_files)} files")
        print(f"  Test: {len(test_files)} files")

    print(f"\nTotal files:")
    print(f"  Training: {len(all_training_files)} files")
    print(f"  Validation: {len(all_validation_files)} files")
    print(f"  Test: {len(all_test_files)} files")

    # Create the output folder
    os.makedirs(output_path, exist_ok=True)

    # Create the three datasets
    make_tfrecord(os.path.join(output_path, "training.tfrecord"), all_training_files)
    make_tfrecord(os.path.join(output_path, "validation.tfrecord"), all_validation_files)
    make_tfrecord(os.path.join(output_path, "testing.tfrecord"), all_test_files)
