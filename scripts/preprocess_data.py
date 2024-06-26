import os
import cv2
import numpy as np
import logging

IMAGE_SIZE = (256, 256)
NUM_CHANNELS = 3


def preprocess_data(image_folder, label_folder):
    """
    Preprocess the data by loading images and labels, resizing them, and normalizing pixel values.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing labels.

    Returns:
        np.array: Processed images.
        np.array: Processed labels.
    """
    try:
        image_paths = sorted([os.path.join(image_folder, filename)
                             for filename in os.listdir(image_folder)])
        label_paths = sorted([os.path.join(label_folder, filename)
                             for filename in os.listdir(label_folder)])

        images = [cv2.imread(img_path) for img_path in image_paths]
        labels = [cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                  for label_path in label_paths]

        images = [cv2.resize(img, IMAGE_SIZE) for img in images]
        labels = [cv2.resize(label, IMAGE_SIZE) for label in labels]

        images = [img / 255.0 for img in images]
        labels = [label / 255.0 for label in labels]

        return np.array(images), np.array(labels)
    except Exception as e:
        logging.error("Error in data preprocessing: %s", e)
        return None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_folder = 'data/images'
    label_folder = 'data/labels'
    images, labels = preprocess_data(image_folder, label_folder)
    logging.info("Data preprocessing completed.")
