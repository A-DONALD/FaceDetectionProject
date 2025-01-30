import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


def load_dataset(dataset_path, img_size=(600, 600), batch_size=32):
    # charge the dataset en divided it in en train/test set
    train_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    test_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Preprocessing : Pixel normalization between 0 and 1
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset