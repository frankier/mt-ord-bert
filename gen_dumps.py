# Usage: gen_dumps.py job.json
import sys
import json
import subprocess
import os


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


sbatch_script = """
#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1,nvme:500
#SBATCH --partition=gpusmall
#SBATCH --account=project_2004993
#SBATCH --cpus-per-task=16

set -euo pipefail

rm -rf $LOCAL_SCRATCH/frankier__hf_datasets
time cp -r /scratch/project_2004993/frankier/huggingface_cache/datasets $LOCAL_SCRATCH/hf_datasets

module load cuda/11.5.0
module load tykky

export HF_DATASETS_CACHE=$LOCAL_SCRATCH/hf_datasets

"""

for arg in sys.argv[1:]:
    json_path, checkpoint = arg.split(":", 1)
    with open(json_path) as f:
        var_name = json_path.split("/")[-1].split(".")[0]
        obj = json.load(f)
        dataset = obj["dataset"]
        output_dir = obj["output_dir"]
        sbatch_script += f"""
        ./cpre/bin/python {SCRIPT_DIR}/../dump.py \
        --dataset {dataset} \
        --model {output_dir}/{checkpoint} \
        --results {output_dir}/{var_name}-{checkpoint}.jsonl \
        --task-thresholds {output_dir}/{var_name}-{checkpoint}.thresh.pkl
        """

subprocess.run("sbatch", input=sbatch_script.strip().encode("utf-8"))