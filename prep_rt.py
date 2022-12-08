from bert_ordinal.datasets import load_data, save_to_disk_with_labels
from mt_ord_bert_utils import tokenize
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("out")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, num_labels, _ = load_data(args.dataset, num_dataset_proc=8)
    dataset = dataset.map(
        tokenize,
        input_columns="text",
        batched=True,
        desc="Tokenizing",
        num_proc=8,
    )
    save_to_disk_with_labels(args.out, dataset, num_labels)


if __name__ == "__main__":
    main()