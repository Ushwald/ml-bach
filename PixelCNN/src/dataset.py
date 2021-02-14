import tensorflow as tf

from logger import setup_logger

log = setup_logger(__name__)


def create_dataset(config):
    log.info("Loading dataset...")

    image_size, simplified_image_size = ((128, 128, 1), (128, 128))

    log.info(f"Input images have dimension {image_size}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="training",
        image_size=simplified_image_size,
    )

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        config.input_dir,
        seed=config.seed,
        color_mode="grayscale",
        batch_size=config.batch_size,
        validation_split=0.2,
        subset="validation",
        image_size=simplified_image_size,
    )

    if config.class_conditional:
        log.info(f"Class names: {train_ds.class_names}")

    # Normalize the data between 0 and 1
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 255
    )
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    validation_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    if config.class_conditional:
        # Concatenate the labels into one tensor
        train_ds = train_ds.map(lambda x, y: ((x, y), y))
        validation_ds = validation_ds.map(lambda x, y: ((x, y), y))

    # Add a prefetch mechanic so producers and consumers can work at the same time
    train_ds = train_ds.cache().prefetch(2)
    validation_ds = validation_ds.cache().prefetch(2)

    log.info("Loading dataset done")

    return (train_ds, validation_ds, image_size)
