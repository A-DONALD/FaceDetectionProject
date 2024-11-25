import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train_data(train_dir, img_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Charge et prépare les données d'entraînement et de validation.
    """
    data_gen = ImageDataGenerator(rescale=1. / 255, validation_split=validation_split)

    train_data = data_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_data = data_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_data, val_data


def load_test_data(test_dir, img_size=(128, 128), batch_size=32):
    """
    Charge les données de test.
    """
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data = test_data_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    return test_data
