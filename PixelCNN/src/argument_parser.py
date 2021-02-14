import argparse


def setup_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", help="input directory where all images are found")
    parser.add_argument(
        "output_dir",
        help="output directory where all newly generated images will be placed",
    )
    parser.add_argument(
        "--output_number",
        help="Depends on how many images should be sampled from the resulting model distribution",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--class_conditional",
        help="flag: Whether to use the class conditional model or not",
        action="store_true",
    )
    parser.add_argument(
        "--training", help="flag: Set to true when training", action="store_true"
    )
    parser.add_argument(
        "--continue_from_checkpoint",
        help="flag: Set to true when training",
        action="store_true",
    )
    parser.add_argument(
        "--epochs", help="Set the number of epochs", default=10, type=int
    )
    parser.add_argument(
        "--batch_size",
        help="Sets the size of the batch, ideally this divides nicely through the number of images",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--checkpoints", help="Set the path to save checkpoints", default="."
    )
    parser.add_argument(
        "--image_size_file",
        help="Specification file for the highest and lowest note of the input images, default size is (128,128,1)",
        default=None,
    )
    parser.add_argument(
        "--seed",
        help="value: Set a seed for the determined randomness",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--log_level",
        help="Specifies the level of precision for the logger",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--patience",
        help="patience level of early stopping",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--kfold_splits",
        help="Number of splits in the kfold cross-validation",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate of the optimizer, range is 0-1",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--name",
        help="Name of the program right now",
        type=str,
        default="default_name_for_file",
    )
    parser.add_argument(
        "--dropout_rate",
        help="dropout rate of the pixelCNN distribution",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--optimizer",
        help="Determines the optimizer to use for gradient descent",
        type=str,
        default="adam",
        choices=["adam", "nadam", "adamax"],
    )

    return parser.parse_args()
