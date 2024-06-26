import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split
from preprocess_data import preprocess_data
import logging


def build_unet_model(input_size=(256, 256, 3)):
    """
    Build a U-Net model for image segmentation.

    Args:
        input_size (tuple): Size of the input images.

    Returns:
        keras.Model: U-Net model.
    """
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Additional convolutional and pooling layers can be added here

    # Decoder
    up2 = UpSampling2D(size=(2, 2))(pool1)
    concat2 = concatenate([conv1, up2], axis=-1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(concat2)

    # Additional convolutional and upsampling layers can be added here

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv2)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    IMAGE_FOLDER = 'data/images'
    LABEL_FOLDER = 'data/labels'
    EPOCHS = 20
    BATCH_SIZE = 4
    VALIDATION_SPLIT = 0.2

    X, y = preprocess_data(IMAGE_FOLDER, LABEL_FOLDER)
    if X is None or y is None:
        logging.error("Failed to preprocess data.")
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X[:25], y[:25], test_size=0.2, random_state=42)

    model = build_unet_model(input_size=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3)

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                        callbacks=[early_stopping, reduce_lr])

    model.save('models/roof_segmentation_model.keras')
    logging.info("Model training completed and saved.")
