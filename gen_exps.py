import argparse
from os import makedirs
from os.path import join as pjoin
import json


MODELS = ["class", "fwd_cumulative", "fwd_sratio", "bwd_cratio", "fwd_acat"]
DATASETS = ["multiscale_rt_critics"]


WARM_TMPL = {
    "output_dir": "/dev/null",
    "warm_dataset_cache": True,
    "num_dataset_proc": 1
}


JOB_TMPL = {
    #"output_dir": "multiscale_rt_critics_output",
    #"dataset": "multiscale_rt_critics",
    "logging_strategy": "epoch",
    "warmup_ratio": 0.1,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "linear",
    "num_train_epochs": 1,
    "evaluation_strategy": "steps",
    "logging_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 500,
    "report_to": "tensorboard",
    "dataloader_num_workers": 8,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_dataset_proc": 8
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons-out", help="The output directory for the JSONs")
    parser.add_argument("--out-root", help="The output directory for training")
    return parser.parse_args()


def dump_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


def main():
    args = parse_args()
    makedirs(args.jsons_out, exist_ok=True)
    for dataset in DATASETS:
        dump_json(
            {
                **WARM_TMPL,
                "dataset": dataset,
            },
            pjoin(args.jsons_out, f"_warm_{dataset}.json")
        )
        for model in MODELS:
            comb = f"{dataset}_{model}"
            output_dir = pjoin(args.out_root, comb)
            makedirs(output_dir, exist_ok=True)
            dump_json(
                {
                    **JOB_TMPL,
                    "dataset": dataset,
                    "model": model,
                    "output_dir": output_dir,
                },
                pjoin(args.jsons_out, f"{comb}.json")
            )

if __name__ == "__main__":
    main()
