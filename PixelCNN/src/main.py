import tensorflow as tf

import logger
from argument_parser import setup_argument_parser
from dataset import create_dataset
from model import create_conditional_model, create_model, predict, train

# Setup global logger
log = logger.setup_logger(__name__)


def main():
    # Extract config from arguments
    config = setup_argument_parser()

    log.info("Starting...")

    log.info("Program will run with following parameters:")
    log.info(config)

    # Create the image dataset from the data/input folder
    ds, val_ds, image_size = create_dataset(config)

    # Whether we are testing or training
    if config.training:
        model, dist = train(ds, val_ds, config, image_shape=image_size)
    else:
        log.info("Loading model...")
        # Load model
        latest = tf.train.latest_checkpoint(config.checkpoints)
        log.info(latest)

        # Create a new model instance
        if config.class_conditional:
            model, dist = create_conditional_model(config, image_size)
        else:
            model, dist = create_model(config, image_size)

        # Load the params back into the model
        model.load_weights(latest).expect_partial()

        log.info("Loading done")

    log.info("Predicting...")

    predict(dist, config)

    log.info("Prediction done...")

    log.info("  Done  ")


if __name__ == "__main__":
    main()
