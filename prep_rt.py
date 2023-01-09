from bert_ordinal.datasets import load_data, save_to_disk_with_labels
from mt_ord_bert_utils import get_tokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("out")
    parser.add_argument("--sample-pct", type=float, default=None)
    return parser.parse_args()


def tokenize(text):
    tokenizer = get_tokenizer()
    return tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=None,
        truncation=True,
        padding=False,
        return_length=True
    )


def main():
    args = parse_args()
    dataset, num_labels, _ = load_data(args.dataset, num_dataset_proc=8)
    if args.sample_pct is not None:
        for label in ("train", "test"):
            num_samples = int(len(dataset[label]) * args.sample_pct / 100.0 + 0.5)
            print(f"Sampling {num_samples} out of {len(dataset[label])} samples from {label} set")
            dataset[label] = (
                dataset[label].shuffle(seed=42).select(range(num_samples))
            )
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