# https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
# https://arxiv.org/pdf/2004.08476.pdf

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

import os
import shutil
import matplotlib.pyplot as plt
import argparse

from training_data import load_training_data_folder

tf.get_logger().setLevel('ERROR')


def create_arg_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=""" 
            Train a classifier
        """,
        epilog=""
    )
    parser.add_argument(
        '-t',
        '--training-data-folder',
        help="The training data folder",
        default="/Users/gt/Projects/rasa-nlu-benchmark/data/AskUbuntuCorpus"
    )
    parser.add_argument(
        '-p',
        '--pretraining-data-folders',
        nargs='+',
        help="The augmentation data folders",
        default=[]
    )
    return parser


def main():

    parser = create_arg_parse()

    args = parser.parse_args()

    # for now let's skip pre-training
    # https://www.tensorflow.org/text/guide/bert_preprocessing_guide

    train_data, test_data = load_training_data_folder(args.training_data_folder)


if __name__ == '__main__':
    main()