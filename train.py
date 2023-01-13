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
from os.path import join as pjoin

from bert_ordinal import Trainer
from bert_ordinal.datasets import load_from_disk_with_labels
from bert_ordinal.dump import DumpWriterCallback
from bert_ordinal.eval import evaluate_pred_dist_avgs, evaluate_predictions
from bert_ordinal.element_link import link_registry
from bert_ordinal.label_dist import PRED_AVGS, summarize_label_dists, summarize_label_dist
from bert_ordinal.transformers_utils import silence_warnings
from mt_ord_bert_utils import get_tokenizer


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
    num_samples: Optional[int] = None
    model: str = None
    discrimination_mode: str = "per_task"
    threads: Optional[int] = None
    trace_labels_predictions: bool = False
    num_dataset_proc: Optional[int] = None
    smoke: bool = False
    pilot_quantiles: bool = False
    pilot_sample_size: int = 256
    peak_class_prob: float = 0.5
    dump_initial_model: Optional[str] = None
    fitted_ordinal: Optional[str] = None
    dump_results: Optional[str] = None
    sampler: str = "default"
    num_vgam_workers: int = 8
    initial_probe: bool = False
    initial_probe_lr: Optional[float] = None
    initial_probe_steps: Optional[float] = None


def prepare_dataset_for_fast_inference(dataset, label_names):
    wanted_columns = {"input_ids", "attention_mask", "token_type_ids", "label", "label_ids", "scale_points", "length", *label_names}
    dataset = dataset.remove_columns(list(set(dataset.column_names) - wanted_columns))
    return dataset.sort("length")


def init_weights(training_args, args, model, tokenizer, train_dataset, eval_dataset):
    if args.pilot_quantiles:
        # We wait until after Trainer is initialised to make sure the model is on the GPU
        if args.model == "threshold":
            model.pilot_quantile_init(train_dataset, tokenizer, args.pilot_sample_size, training_args.per_device_train_batch_size)
        elif args.model in ("regress", "regress_l1", "regress_adjust_l1", "mono_regress", "mono_regress_l1", "mono_regress_adjust_l1"):
            model.pilot_quantile_init(train_dataset, tokenizer, args.pilot_sample_size, training_args.per_device_train_batch_size, np.asarray(train_dataset["task_ids"]), np.asarray(train_dataset["label"]))
        else:
            model.pilot_quantile_init(train_dataset, tokenizer, args.pilot_sample_size, training_args.per_device_train_batch_size, peak_class_prob=args.peak_class_prob)
    if args.model == "fixed_threshold":
        model.quantile_init(train_dataset)
    if args.fitted_ordinal:
        model.init_std_hidden_pilot(train_dataset, tokenizer, args.pilot_sample_size, training_args.per_device_train_batch_size)
        model.set_ordinal_heads(torch.load(args.fitted_ordinal))
    if args.initial_probe:
        linear_probe_args = TrainingArguments(
            learning_rate=args.initial_probe_lr if args.initial_probe_lr is not None else training_args.learning_rate,
            max_steps=args.initial_probe_steps or -1,
            num_train_epochs=1.0,
            prediction_loss_only=True,
            output_dir=pjoin(training_args.output_dir, "linear_probe"),
            save_strategy="no",
            lr_scheduler_type="constant",
        )
        linear_probe_trainer = Trainer(
            model=model,
            args=linear_probe_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        for param in model.bert.parameters():
            param.requires_grad = False
        linear_probe_trainer.train()
        for param in model.bert.parameters():
            param.requires_grad = True


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        training_args, args = parser.parse_args_into_dataclasses()
    wandb.config.update(args)

    # args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dataset, num_labels = load_from_disk_with_labels(args.dataset)

    if args.num_samples is not None:
        for label in ("train", "test"):
            dataset[label] = (
                dataset[label].shuffle(seed=42).select(range(args.num_samples))
            )

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

    if isinstance(num_labels, int):
        label_names = ["labels"]
        if args.model == "class":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                base_model, num_labels=num_labels
            )

            def proc_logits(logits):
                label_dist = logits.softmax(dim=-1)
                return (
                    label_dist,
                    *summarize_label_dist(label_dist).values()
                )
        elif args.model == "regress":
            from transformers import BertForSequenceClassification
            with silence_warnings():
                model = BertForSequenceClassification.from_pretrained(
                    base_model, num_labels=num_labels, problem_type="regression"
                )
            proc_logits = None
        elif args.model in link_registry:
            from bert_ordinal import BertForOrdinalRegression
            model = BertForOrdinalRegression.from_pretrained(
                base_model, num_labels=num_labels, link=args.model, discrimination_mode=args.discrimination_mode
            )
            link = model.link

            def proc_logits(logits):
                label_dist = link.label_dist_from_logits(logits[0])
                return (
                    logits[1],
                    label_dist,
                    *summarize_label_dist(label_dist).values()
                )
        else:
            print(f"Unknown model type {args.model}", file=sys.stderr)
            sys.exit(-1)
    else:
        label_names = ["labels", "task_ids"]
        if args.model == "class":
            from bert_ordinal.baseline_models.classification import BertForMultiScaleSequenceClassification
            model = BertForMultiScaleSequenceClassification.from_pretrained(
                base_model, num_labels=num_labels
            )

            def proc_logits(logits):
                label_dists = logits.softmax(dim=-1)
                return (
                    label_dists,
                    *summarize_label_dists(label_dists).values(),
                )
        elif args.model in ("regress", "regress_l1", "regress_adjust_l1"):
            from bert_ordinal.baseline_models.regression import BertForMultiScaleSequenceRegression
            with silence_warnings():
                model = BertForMultiScaleSequenceRegression.from_pretrained(
                    base_model,
                    num_labels=num_labels,
                    loss=(
                        {
                            "regress": "mse",
                            "regress_l1": "mae",
                            "regress_adjust_l1": "adjust_l1"
                        }[args.model]
                    ),
                )
            #model.init_scales_empirical(np.asarray(dataset["train"]["task_ids"]), np.asarray(dataset["train"]["label"]))
            #model.init_scales_range()
            def proc_logits(logits):
                return logits
        elif args.model in ("mono_regress", "mono_regress_l1", "mono_regress_adjust_l1"):
            from bert_ordinal.experimental_regression import BertForMultiMonotonicTransformSequenceRegression
            with silence_warnings():
                model = BertForMultiMonotonicTransformSequenceRegression.from_pretrained(
                    base_model,
                    num_labels=num_labels,
                    loss=(
                        {
                            "mono_regress": "mse",
                            "mono_regress_l1": "mae",
                            "mono_regress_adjust_l1": "adjust_l1"
                        }[args.model]
                    ),
                )
            #model.init_scales_empirical(np.asarray(dataset["train"]["task_ids"]), np.asarray(dataset["train"]["label"]))
            #model.init_scales_range()
            def proc_logits(logits):
                return logits
        elif args.model == "latent_softmax":
            from bert_ordinal.ordinal_models.experimental import BertForWithLatentAndSoftMax
            with silence_warnings():
                model = BertForWithLatentAndSoftMax.from_pretrained(
                    base_model, num_labels=num_labels
                )
            def proc_logits(logits):
                label_dists = logits[0].softmax(dim=-1)
                return (
                    label_dists,
                    *summarize_label_dists(label_dists).values(),
                )
        elif args.model == "threshold":
            from bert_ordinal.ordinal_models.experimental import BertForMultiScaleThresholdRegression
            with silence_warnings():
                model = BertForMultiScaleThresholdRegression.from_pretrained(
                    base_model, num_labels=num_labels
                )
            def proc_logits(logits):
                return logits
        elif args.model == "fixed_threshold":
            from bert_ordinal.ordinal_models.experimental import BertForMultiScaleFixedThresholdRegression
            with silence_warnings():
                model = BertForMultiScaleFixedThresholdRegression.from_pretrained(
                    base_model, num_labels=num_labels
                )
            def proc_logits(logits):
                return logits
        elif args.model == "metric":
            from bert_ordinal.ordinal_models.experimental import BertForLatentScaleMetricLearning
            with silence_warnings():
                model = BertForLatentScaleMetricLearning.from_pretrained(
                    base_model, num_labels=num_labels
                )
            def proc_logits(logits):
                return logits
        elif args.model in link_registry:
            from bert_ordinal import BertForMultiScaleOrdinalRegression
            model = BertForMultiScaleOrdinalRegression.from_pretrained(
                base_model, num_labels=num_labels, link=args.model, discrimination_mode=args.discrimination_mode
            )
            link = model.link

            def proc_logits(logits):
                label_dists = [link.label_dist_from_logits(li) for li in logits[0].unbind()]
                return (
                    logits[1],
                    label_dists,
                    *summarize_label_dists(label_dists).values(),
                )
        else:
            print(f"Unknown model type {args.model}", file=sys.stderr)
            sys.exit(-1)

    tokenizer = get_tokenizer()

    eval_train_dataset = prepare_dataset_for_fast_inference(dataset["train"], label_names)
    eval_test_dataset = prepare_dataset_for_fast_inference(dataset["test"], label_names)

    training_args.label_names = label_names
    training_args.optim = "adamw_torch"

    def compute_metrics(eval_pred):
        step = trainer.state.global_step
        dump_writer.start_step_dump(step)
        pred_label_dists, labels = eval_pred
        if len(label_names) == 2:
            labels, task_ids = labels
            batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
            for idx, task_id in enumerate(task_ids):
                batch_num_labels[idx] = num_labels[task_id]
        else:
            batch_num_labels = torch.tensor([num_labels]).repeat(len(labels))

        if args.trace_labels_predictions:
            print()
            print("Computing metrics based upon")
            print("labels")
            print(labels)
            print("predictions")
            pprint(pred_label_dists)

        def refit(test_hiddens):
            from bert_ordinal.eval import refit_eval
            print()
            print(" * Refit * ")
            print()
            print()
            return refit_eval(
                model,
                tokenizer,
                eval_train_dataset,
                training_args.train_batch_size,
                task_ids,
                test_hiddens,
                batch_num_labels,
                labels,
                dump_writer=dump_writer,
                num_workers=args.num_vgam_workers,
                mask_vglm_errors=True,
                suppress_vglm_output=True
            )

        if args.model == "metric":
            if args.dump_results:
                dump_writer.add_info_full("test", hidden=pred_label_dists.squeeze(-1))
            res = refit(pred_label_dists)
        elif args.model in ("threshold", "fixed_threshold"):
            predictions, hiddens  = pred_label_dists
            if args.dump_results:
                dump_writer.add_info_full("test", hidden=hiddens.squeeze(-1), predictions=predictions)
            res = {
                **evaluate_predictions(predictions, labels, batch_num_labels, task_ids),
                **refit(hiddens)
            }
        elif args.model in ("regress", "regress_l1", "regress_adjust_l1", "mono_regress", "mono_regress_l1", "mono_regress_adjust_l1"):
            raw_predictions, hiddens = pred_label_dists
            predictions = np.clip(raw_predictions.squeeze(-1) + 0.5, 0, batch_num_labels - 1).astype(int)
            if args.dump_results:
                dump_writer.add_info_full("test", hidden=hiddens.squeeze(-1), predictions=predictions)
            res =  {
                **evaluate_predictions(predictions, labels, batch_num_labels, task_ids),
                **refit(hiddens)
            }
        else:
            hiddens = pred_label_dists[0]
            label_dists = pred_label_dists[1]
            summarized_label_dists = dict(zip(PRED_AVGS, pred_label_dists[2:]))
            if args.dump_results:
                dump_writer.add_info_full("test", hidden=hiddens.squeeze(-1), label_dists=label_dists, **summarized_label_dists)
            res =  evaluate_pred_dist_avgs(summarized_label_dists, labels, batch_num_labels, task_ids)
        dump_writer.finish_step_dump(model)
        return res

    init_weights(training_args, args, model, tokenizer, dataset["train"], eval_test_dataset)

    print("")
    print(
        f" ** Training model {args.model} on dataset {args.dataset} ** "
    )
    print("")

    if proc_logits is None:
        preprocess_logits_for_metrics = None
    else:
        preprocess_logits_for_metrics = lambda logits, _labels: proc_logits(logits)

    if args.model == "metric":
        from bert_ordinal.ordinal_models.experimental import MetricLearningTrainer
        if args.sampler != "default":
            raise ValueError("Custom samplers not supported for metric learning")

        trainer = MetricLearningTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_test_dataset,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            tokenizer=tokenizer
        )
    else:
        if args.sampler == "default":
            training_args.group_by_length = True
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=eval_test_dataset,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                tokenizer=tokenizer
            )
        else:
            from bert_ordinal.ordinal_models.experimental import CustomSamplerTrainer
            trainer = CustomSamplerTrainer(
                sampler=args.sampler,
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=eval_test_dataset,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                tokenizer=tokenizer
            )
    if args.dump_results:
        dump_writer_cb = DumpWriterCallback(args.dump_results)
        dump_writer = dump_writer_cb.dump_writer
        trainer.add_callback(dump_writer_cb)
    if args.dump_initial_model is not None:
        trainer.save_model(training_args.output_dir + "/" + args.dump_initial_model)
    trainer.train()


if __name__ == "__main__":
    main()
