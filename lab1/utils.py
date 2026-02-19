import tensorflow as tf

IMG_SIZE = (64, 64)
BATCH_SIZE = 32


def load_datasets(dataset_path):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    normalization = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), tf.one_hot(y, len(class_names))))
    val_ds = val_ds.map(lambda x, y: (normalization(x), tf.one_hot(y, len(class_names))))

    return train_ds, val_ds, class_names
