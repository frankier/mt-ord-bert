import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional
import evaluate
import numpy as np
import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from pprint import pprint

from bert_ordinal import Trainer
from bert_ordinal.datasets import load_from_disk_with_labels
from bert_ordinal.eval import qwk_multi_norm, eval_preds
from bert_ordinal.element_link import link_registry
from bert_ordinal.label_dist import summarize_label_dist, PRED_AVGS


metric_accuracy = evaluate.load("accuracy")
metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")

# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Begin W&B
import wandb
wandb.init(project="mt-ord-bert", entity="frobertson")
# WEnd &B


@dataclass
class ExtraArguments:
    dataset: str
    model: str = None
    discrimination_mode: str = "per_task"
    threads: Optional[int] = None
    trace_labels_predictions: bool = False
    num_dataset_proc: Optional[int] = None
    smoke: bool = False


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

    dataset, num_labels = load_from_disk_with_labels(args.dataset)

    import packaging.version

    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.13"):
        print(
            f"Warning: multi-scale datasets such as {args.dataset} are not support with torch < 1.13",
            file=sys.stderr,
        )
    
    if args.smoke:
        base_model = "prajjwal1/bert-tiny"
        torch.set_num_threads(1)
    else:
        base_model = "bert-base-cased"

    if args.model == "class":
        from bert_ordinal.baseline_models.classification import BertForMultiScaleSequenceClassification
        model = BertForMultiScaleSequenceClassification.from_pretrained(
            base_model, num_labels=num_labels
        )

        def proc_logits(logits):
            label_dists = logits.softmax(dim=-1)
            return {
                "label_dists": label_dists,
                **summarize_label_dist(label_dists),
            }
    elif args.model in link_registry:
        from bert_ordinal import BertForMultiScaleOrdinalRegression
        model = BertForMultiScaleOrdinalRegression.from_pretrained(
            base_model, num_labels=num_labels, link=args.model
        )
        link = model.link

        def proc_logits(logits):
            label_dists = torch.hstack([link.label_dist_from_logits(li) for li in logits[1].unbind()])
            return {
                "label_dists": label_dists,
                **summarize_label_dist(label_dists),
            }
    else:
        print(f"Unknown model type {args.model}", file=sys.stderr)
        sys.exit(-1)

    label_names = ["labels", "task_ids"]

    training_args.label_names = label_names
    training_args.optim = "adamw_torch"

    def compute_metrics(eval_pred):
        pred_label_dists, labels = eval_pred
        labels, task_ids = labels
        batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
        for idx, task_id in enumerate(task_ids):
            batch_num_labels[idx] = num_labels[task_id]

        if args.trace_labels_predictions:
            print()
            print("Computing metrics based upon")
            print("labels", labels)
            print("predictions")
            pprint(pred_label_dists)

        res = {}
        for avg in PRED_AVGS:
            for k, v in eval_preds(pred_label_dists[avg], labels, batch_num_labels).items():
                res[f"{avg}_{k}"] = v
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
