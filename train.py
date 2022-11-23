import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import evaluate
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
)

from bert_ordinal import BertForOrdinalRegression, Trainer, ordinal_decode_labels_pt
from bert_ordinal.datasets import load_data
from bert_ordinal.eval import qwk, qwk_multi_norm
from bert_ordinal.element_link import link_registry

metric_accuracy = evaluate.load("accuracy")
metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")

# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExtraArguments:
    dataset: str
    model: str = None
    threads: Optional[int] = None
    trace_labels_predictions: bool = False
    num_dataset_proc: Optional[int] = None
    warm_dataset_cache: bool = False


_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return _tokenizer


def tokenize(text):
    tokenizer = get_tokenizer()
    return tokenizer(text, padding="max_length", truncation=True, return_tensors="np")


def main():
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        training_args, args = parser.parse_args_into_dataclasses()

    # args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dataset, num_labels, _ = load_data(
        args.dataset, num_dataset_proc=args.num_dataset_proc
    )
    dataset = dataset.map(
        tokenize,
        input_columns="text",
        batched=True,
        desc="Tokenizing",
        num_proc=args.num_dataset_proc,
    )
    if args.warm_dataset_cache:
        print("Dataset cache warmed")
        return

    import packaging.version

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.13"):
        print(
            f"Warning: multi-scale datasets such as {args.dataset} are not support with torch < 1.13",
            file=sys.stderr,
        )

    if args.model == "class":
        from bert_ordinal.baseline_models.classification import BertForMultiScaleSequenceClassification
        model = BertForMultiScaleSequenceClassification.from_pretrained(
            "bert-base-cased", num_labels=num_labels
        )

        def proc_logits(logits):
            return logits.argmax(dim=-1)
    elif args.model in link_registry:
        from bert_ordinal import BertForMultiScaleOrdinalRegression
        model = BertForMultiScaleOrdinalRegression.from_pretrained(
            "bert-base-cased", num_labels=num_labels, link=args.model
        )
        link = model.link

        def proc_logits(logits):
            return torch.hstack([link.top_from_logits(li) for li in logits[1].unbind()])
    else:
        print(f"Unknown model type {args.model}", file=sys.stderr)
        sys.exit(-1)

    label_names = ["labels", "task_ids"]

    training_args.label_names = label_names
    training_args.optim = "adamw_torch"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels, task_ids = labels
        batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
        for idx, task_id in enumerate(task_ids):
            batch_num_labels[idx] = num_labels[task_id]

        if args.trace_labels_predictions:
            print()
            print("Computing metrics based upon")
            print("labels", labels)
            print("predictions", predictions)

        mse = metric_mse.compute(predictions=predictions, references=labels)
        res = {
            **metric_accuracy.compute(predictions=predictions, references=labels),
            **metric_mae.compute(predictions=predictions, references=labels),
            **mse,
            "rmse": (mse["mse"]) ** 0.5,
        }
        res["qwk"] = qwk_multi_norm(predictions, labels, batch_num_labels)
        return res

    print("")
    print(
        f" ** Training model {args.model} on dataset {args.dataset} ** "
    )
    print("")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=lambda logits, _labels: proc_logits(logits),
    )
    trainer.train()


if __name__ == "__main__":
    main()
