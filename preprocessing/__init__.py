print("ok")

import yaml
import tensorflow as tf
import argparse

# let's use T5 https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/#train-steps
# http://peterbloem.nl/blog/transformers
# https://aclanthology.org/2020.findings-emnlp.63.pdf


def create_arg_parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=""" 
            Train tokenizer
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
        '-v',
        '--vocab-config-file',
        help="The vocab config file",
        default="configs/vocab_config.yml"
    )
    return parser


def main():
    from training_data import load_training_data_folder

    parser = create_arg_parse()

    args = parser.parse_args()

    train_data, test_data = load_training_data_folder(args.training_data_folder)



if __name__ == '__main__':
    main()