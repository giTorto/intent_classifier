import re
from torch.utils.data import Dataset
import os
import glob
from training_data.md_reader import load_training_data_folder
from transformers import T5Tokenizer
import argparse
from argparse import Namespace

# more here https://github.com/RasaHQ/NLU-training-data


class MDRasaDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build(data_dir)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self, data_dir):
        self._buil_examples_from_files(data_dir)

    def _buil_examples_from_files(self, data_dir):
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

        training_data, test_data = load_training_data_folder(data_dir)
        for example in training_data:
            text = example.text
            label = example.intent.get("intent")  # an alternative is to try numbers
            line = text.strip()
            line = REPLACE_NO_SPACE.sub("", line)
            line = REPLACE_WITH_SPACE.sub("", line)
            line = line + ' </s>'

            target = label + " </s>"
            print(text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


def get_dataset(tokenizer, args):
    return MDRasaDataset(tokenizer=tokenizer, data_dir=args.data_dir, max_len=args.max_seq_length)


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

    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    args_ns = Namespace(**{
        "data_dir": "/Users/gt/Projects/rasa-nlu-benchmark/converted_data/ask_ubuntu_y",
        "max_seq_length": 512})

    data_obj = get_dataset(t5_tokenizer, args_ns)
    print(data_obj.inputs[0])
    print(data_obj.targets[0])

if __name__ == '__main__':
    main()
