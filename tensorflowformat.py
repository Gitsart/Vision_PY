import tensorflow as tf
from object_detection.utils import dataset_util
import pandas as pd
import os
import cv2
import numpy as np

def create_tf_example(image_path, bboxes, labels, label_map):
    # Read image data
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Create the feature dictionary
    image_format = b'jpg'  # or png if that's your image type
    encoded_image_data = tf.io.gfile.GFile(image_path, 'rb').read()

    # List of bboxes
    xmins = [bbox[0] / width for bbox in bboxes]
    ymin = [bbox[1] / height for bbox in bboxes]
    xmax = [bbox[2] / width for bbox in bboxes]
    ymax = [bbox[3] / height for bbox in bboxes]
    
    classes_text = [label_map[label] for label in labels]
    classes = [label_map[label] for label in labels]

    # Create tf.train.Example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_csv_to_tfrecord(csv_file, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    annotations = pd.read_csv(csv_file)

    for idx, row in annotations.iterrows():
        image_path = os.path.join('images/', row['image'])
        bboxes = [(row['xmin'], row['ymin'], row['xmax'], row['ymax'])]
        labels = [row['label']]
        tf_example = create_tf_example(image_path, bboxes, labels, label_map)
        writer.write(tf_example.SerializeToString())
    writer.close()

# Define a label map (mapping screw to an integer ID)
label_map = {'Screw': 1}  # Example label map (you can extend this if needed)

# Convert annotations to TFRecord
convert_csv_to_tfrecord('annotations.csv', 'output.tfrecord', label_map)
