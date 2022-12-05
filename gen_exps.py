import argparse
from dataclasses import dataclass
import os
from os import makedirs, listdir
from os.path import join as pjoin, isdir
from typing import Optional
import json
from string import Template


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclass
class ModelConfig:
    link: str
    discrimination_mode: Optional[str] = None

    def name(self):
        if self.discrimination_mode:
            return f"{self.link}_{self.discrimination_mode}"
        else:
            return self.link


MODELS = {
    "rt": [
        ModelConfig("class"),
        *(
            ModelConfig(link, discrimination_mode)
            for link in ("fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat")
            for discrimination_mode in ("per_task", "multi")
        )
    ],
    "rt_one": [
        ModelConfig("class"),
        *(
            ModelConfig(link)
            for link in ("fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat")
            for discrimination_mode in ("none", "multi")
        )
    ]
}

JOB_TMPL = {
    "learning_rate": 1e-5,
    "evaluation_strategy": "steps",
    "logging_strategy": "steps",
    "report_to": "wandb",
    "dataloader_num_workers": 8,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_dataset_proc": 8
}

DATA_JOB_TMPL = {
    "rt": {
        "lr_scheduler_type": "linear",
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_steps": 3000,
        "eval_steps": 100,
        "logging_steps": 100,
        "save_steps": 100,
    },
    "rt_one": {
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_ratio": 0.1,
        "max_steps": 1000,
        "eval_steps": 50,
        "logging_steps": 50,
        "save_steps": 100,
    },
}

SLURM_TMPL = Template("""#!/bin/bash
#SBATCH -t $time
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1,nvme:500
#SBATCH --partition=gpusmall
#SBATCH --account=project_2004993
#SBATCH --cpus-per-task=16
#SBATCH --output=$logdir/$expcomb-%j.out

set -euo pipefail
IFS=$$'\\n\\t'

rm -rf $$LOCAL_SCRATCH/frankier__hf_datasets
time cp -r /scratch/project_2004993/frankier/huggingface_cache/datasets $$LOCAL_SCRATCH/hf_datasets

module load cuda/11.5.0
module load tykky

HF_DATASETS_CACHE=$$LOCAL_SCRATCH/hf_datasets \
	./cpre/bin/python $script_dir/train.py \
	$conf_json_path
""")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", help="The input directory for the datasets")
    parser.add_argument("--out", help="The output directory")
    return parser.parse_args()


def dump_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    args = parse_args()
    makedirs(args.out, exist_ok=True)
    dirs = {}
    for out_dir in ["jsons", "outs", "results", "jobs"]:
        dirs[out_dir] = pjoin(args.out, out_dir)
        makedirs(pjoin(args.out, out_dir), exist_ok=True)
    for dataset in listdir(args.data_root):
        dataset_dir = pjoin(args.data_root, dataset)
        if not isdir(dataset_dir):
            continue
        for model_config in MODELS[dataset]:
            comb_name = f"{dataset}_{model_config.name()}"
            log_dir = pjoin(dirs["results"], comb_name)
            makedirs(log_dir, exist_ok=True)
            conf_json_path = pjoin(dirs["jsons"], f"{comb_name}.json")
            config = {
                **JOB_TMPL,
                **DATA_JOB_TMPL[dataset],
                "dataset": dataset_dir,
                "model": model_config.link,
                "output_dir": log_dir,
            }
            config["discrimination_mode"] = model_config.discrimination_mode
            dump_json(config, conf_json_path)
            if dataset == "rt_one":
                time = "1:00:00"
            else:
                time = "2:00:00"
            slurm_content = SLURM_TMPL.substitute(
                time=time,
                logdir=dirs["outs"],
                expcomb=comb_name,
                script_dir=SCRIPT_DIR,
                conf_json_path=conf_json_path
            )
            with open(pjoin(dirs["jobs"], f"{comb_name}.slurm"), "w") as f:
                f.write(slurm_content)


if __name__ == "__main__":
    main()
