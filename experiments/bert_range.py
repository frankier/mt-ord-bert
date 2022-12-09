"""
The idea of this is just to get the initial output distribution of BERT so we
can initialise linear layers well.
"""

from bert_ordinal.baseline_models.regression import BertForMultiScaleSequenceRegression
from bert_ordinal.datasets import load_from_disk_with_labels
from asciihist import asciihist
from transformers import logging
import torch
import sys
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_examples", type=int)
    parser.add_argument("dataset")
    parser.add_argument("--zero-bias", action="store_true")
    parser.add_argument("--range-init", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, num_labels = load_from_disk_with_labels(args.dataset)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(args.num_examples))

    base_model = "bert-base-cased"
    logging.set_verbosity_error()
    model = BertForMultiScaleSequenceRegression.from_pretrained(
        base_model, num_labels=num_labels
    )
    if args.zero_bias:
        model._zero_bias()
    if args.range_init:
        model.init_scales_range()
    logging.set_verbosity_warning()
    input_ids = torch.tensor(dataset["train"]["input_ids"])
    task_ids = torch.LongTensor(dataset["train"]["task_ids"]).unsqueeze(-1)
    all_outs = []
    with torch.inference_mode():
        for chunk in range(0, len(input_ids), 32):
            print("chunk", chunk, chunk+32)
            model_out = model.forward(input_ids=input_ids[chunk:chunk+32], task_ids=task_ids[chunk:chunk+32], labels=None)
            if args.range_init:
                val = model_out.logits
            else:
                val = model_out.hidden_linear
            all_outs.append(val)
    all_outs = torch.vstack(all_outs).detach().numpy().flatten()
    print(np.mean(all_outs))
    print(np.std(all_outs))
    asciihist(all_outs, bins=40)


if __name__ == "__main__":
    main()