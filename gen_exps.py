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
    config: Optional[dict] = None

    def name(self):
        if self.discrimination_mode:
            return f"{self.link}_{self.discrimination_mode}"
        else:
            return self.link

    def get_config(self):
        if self.config is not None:
            res = self.config
        else:
            res = {}
        if self.discrimination_mode is not None:
            res["discrimination_mode"] = self.discrimination_mode
        return res


MONO_CONFIG = {
    "scale_lr_multiplier": 2,
    "pilot_train_init": True
}

REGRESS_CONFIG = {
    "pilot_quantiles": True,
}


ALL_MUTISCALE_MODELS = [
    ModelConfig("class"),
    ModelConfig("regress", config=REGRESS_CONFIG),
    ModelConfig("regress_l1", config=REGRESS_CONFIG),
    ModelConfig("regress_adjust_l1", config=REGRESS_CONFIG),
    ModelConfig("mono", config=MONO_CONFIG),
    ModelConfig("mono_l1", config=MONO_CONFIG),
    ModelConfig("mono_adjust_l1", config=MONO_CONFIG),
    ModelConfig("latent_softmax"),
    ModelConfig("threshold"),
    ModelConfig("fixed_threshold"),
    ModelConfig("metric"),
    *(
        ModelConfig(link, discrimination_mode)
        for link in ("fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat")
        for discrimination_mode in ("per_task", "multi")
    )
]
ALL_SINGLE_SCALE_MODELS = [
    ModelConfig("class"),
    ModelConfig("regress"),
    *(
        ModelConfig(link, discrimination_mode)
        for link in ("fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat")
        for discrimination_mode in ("none", "multi")
    )
]

MODELS = {
    "rt": ALL_MUTISCALE_MODELS,
    "rt_cxc_1k": ALL_MUTISCALE_MODELS,
    "rt_one": ALL_SINGLE_SCALE_MODELS,
    "rt_irr5": ALL_MUTISCALE_MODELS,
}

JOB_TMPL = {
    "learning_rate": 1e-5,
    "evaluation_strategy": "steps",
    "logging_strategy": "steps",
    "report_to": "wandb",
    "dataloader_num_workers": 16,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "logging_first_step": True,
    "num_vgam_workers": 32,
}

DATA_JOB_TMPL = {
    "rt": {
        "lr_scheduler_type": "linear",
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_steps": 3000,
        "eval_steps": 100,
        "logging_steps": 100,
        "save_steps": 3000,
    },
    "rt_cxc_1k": {
        "lr_scheduler_type": "linear",
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "eval_steps": 1000,
        "logging_steps": 1000,
        "save_steps": 1000,
        "use_bert_large_wholeword": True,
    },
    "rt_irr5": {
        "lr_scheduler_type": "linear",
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_steps": 1500,
        "eval_steps": 100,
        "logging_steps": 100,
        "save_steps": 100,
    },
    "rt_irr5_5pct": {
        "lr_scheduler_type": "linear",
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_steps": 207,
        "eval_steps": 23,
        "logging_steps": 23,
        "save_steps": 207,
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
#SBATCH --cpus-per-task=32
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
    parser.add_argument("--dataset", help="Only generate for one or more datasets", nargs='+')
    parser.add_argument("--model", help="Only generate for one or more models", nargs='+')
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
    for out_dir in ["jsons", "outs", "results", "jobs", "dumps"]:
        dirs[out_dir] = pjoin(args.out, out_dir)
        makedirs(pjoin(args.out, out_dir), exist_ok=True)
    for dataset in listdir(args.data_root):
        if args.dataset and dataset not in args.dataset:
            continue
        dataset_dir = pjoin(args.data_root, dataset)
        if not isdir(dataset_dir):
            continue
        last_bit = dataset.rsplit("_", 1)[-1]
        if last_bit.endswith("pct"):
            orig_dataset = dataset.rsplit("_", 1)[0]
        else:
            orig_dataset = dataset
        for model_config in MODELS[orig_dataset]:
            if args.model and model_config.name() not in args.model:
                continue
            comb_name = f"{dataset}_{model_config.name()}"
            log_dir = pjoin(dirs["results"], comb_name)
            makedirs(log_dir, exist_ok=True)
            dump_dir = pjoin(dirs["dumps"], comb_name)
            makedirs(dump_dir, exist_ok=True)
            conf_json_path = pjoin(dirs["jsons"], f"{comb_name}.json")
            config = {
                **JOB_TMPL,
                **(DATA_JOB_TMPL[dataset] if dataset in DATA_JOB_TMPL else DATA_JOB_TMPL[orig_dataset]),
                "dataset": dataset_dir,
                "model": model_config.link,
                "output_dir": log_dir,
                "dump_results": dump_dir,
            }
            config.update(model_config.get_config())
            dump_json(config, conf_json_path)
            if orig_dataset == "rt_cxc_1k":
                time = "16:00:00"
            elif orig_dataset in ("rt_one", "rt_irr5"):
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
