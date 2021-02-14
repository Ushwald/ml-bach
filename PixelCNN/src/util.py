import os

import tf.keras

from logger import setup_logger

log = setup_logger(__name__)


def save_image(image, image_counter, config):
    log.info(f"Saving image {image_counter} to disk...")
    if not config.output_dir.endswith("/"):
        config.output_dir += "/"
    if not os.path.isdir(config.output_dir):
        try:
            os.makedirs(config.output_dir)
        except Exception as e:
            log.warn(f"makedir exception: {e}")

    tf.keras.preprocessing.image.save_img(
        config.output_dir + f"output_{image_counter}.png", image
    )

    log.info("Saving done")
