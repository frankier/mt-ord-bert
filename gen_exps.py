import argparse
from dataclasses import dataclass
from os import makedirs, listdir
from os.path import join as pjoin, isdir
from typing import Optional
import json


@dataclass
class ModelConfig:
    link: str
    discrimination_mode: Optional[str] = None

    def name(self):
        if self.discrimination_mode:
            return f"{self.link}_{self.discrimination_mode}"
        else:
            return self.link


MODELS = [
    ModelConfig("class"),
    *(
        ModelConfig(link, discrimination_mode)
        for link in ("fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat")
        for discrimination_mode in ("per_task", "multi")
    )
]


JOB_TMPL = {
    "warmup_ratio": 0.33,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "linear",
    "max_steps": 3000,
    "evaluation_strategy": "steps",
    "logging_strategy": "steps",
    "eval_steps": 100,
    "logging_steps": 100,
    "save_steps": 100,
    "report_to": "wandb",
    "dataloader_num_workers": 8,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_dataset_proc": 8
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="The input directory for the datasets")
    parser.add_argument("--jsons-out", help="The output directory for the JSONs")
    parser.add_argument("--log-root", help="The output directory for training logs and checkpoints")
    return parser.parse_args()


def dump_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    args = parse_args()
    makedirs(args.jsons_out, exist_ok=True)
    for dataset in listdir(args.data_root):
        dataset_dir = pjoin(args.data_root, dataset)
        if not isdir(dataset_dir):
            continue
        for model_config in MODELS:
            comb_name = f"{dataset}_{model_config.name()}"
            log_dir = pjoin(args.log_root, comb_name)
            makedirs(log_dir, exist_ok=True)
            dump_json(
                {
                    **JOB_TMPL,
                    "dataset": dataset_dir,
                    "model": model_config.link,
                    "output_dir": log_dir,
                },
                pjoin(args.jsons_out, f"{comb_name}.json")
            )

if __name__ == "__main__":
    main()
