import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import tensorflow as tf
from preprocess_data import preprocess_data
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


def evaluate_model():
    """
    Evaluate the trained model on test data and visualize predictions.
    """
    IMAGE_FOLDER = 'data/images'
    LABEL_FOLDER = 'data/labels'
    PREDICTION_FOLDER = 'predictions'

    # Create the predictions directory if it does not exist
    if not os.path.exists(PREDICTION_FOLDER):
        os.makedirs(PREDICTION_FOLDER)

    model = tf.keras.models.load_model('models/roof_segmentation_model.keras')

    X, y = preprocess_data(IMAGE_FOLDER, LABEL_FOLDER)
    if X is None or y is None:
        logging.error("Failed to preprocess data.")
        return

    # Ensure we have enough data for testing
    if len(X) < 5:
        logging.error(
            "Not enough data for evaluation. Need at least 5 images.")
        return

    # Use the last 5 images as test data
    X_test, y_test = X[-5:], y[-5:]

    # Check for NaN or infinite values in the test data
    if np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
        logging.error("Test data contains NaN values.")
        return
    if np.any(np.isinf(X_test)) or np.any(np.isinf(y_test)):
        logging.error("Test data contains infinite values.")
        return

    # Print shapes and ranges for debugging
    logging.info("X_test shape: %s", X_test.shape)
    logging.info("y_test shape: %s", y_test.shape)
    logging.info("X_test min: %f, max: %f", np.min(X_test), np.max(X_test))
    logging.info("y_test min: %f, max: %f", np.min(y_test), np.max(y_test))

    try:
        # Perform evaluation
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info("Test Loss: %f", loss)
        logging.info("Test Accuracy: %f", accuracy)
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        return

    try:
        predicted_masks = model.predict(X_test)

        # Check for NaN or infinite values in the predictions
        if np.any(np.isnan(predicted_masks)) or np.any(np.isinf(predicted_masks)):
            logging.error("Predictions contain NaN or infinite values.")
            return

        # Print shapes and ranges of predictions
        logging.info("predicted_masks shape: %s", predicted_masks.shape)
        logging.info("predicted_masks min: %f, max: %f", np.min(
            predicted_masks), np.max(predicted_masks))

        # Save predicted masks as images in the predictions directory
        for i, predicted_mask in enumerate(predicted_masks):
            cv2.imwrite(os.path.join(
                PREDICTION_FOLDER, f'predicted_mask_{i}.png'), (predicted_mask * 255).astype(np.uint8))

        # Visualize some predictions and save the plot
        num_images = min(X_test.shape[0], 5)
        fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(10, 10))
        for i in range(num_images):
            axes[i, 0].imshow(X_test[i])
            axes[i, 0].axis("off")
            axes[i, 0].set_title("Input Image")

            axes[i, 1].imshow(y_test[i].squeeze(), cmap="gray")
            axes[i, 1].axis("off")
            axes[i, 1].set_title("Ground Truth Mask")

            axes[i, 2].imshow(predicted_masks[i].squeeze(), cmap="gray")
            axes[i, 2].axis("off")
            axes[i, 2].set_title("Predicted Mask")

        plt.tight_layout()
        # Save the plot as an image file
        plt.savefig(os.path.join(PREDICTION_FOLDER, 'predictions.png'))
        logging.info("Saved visualization to predictions/predictions.png")
    except Exception as e:
        logging.error("Error during predictions or visualization: %s", e)
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_model()
