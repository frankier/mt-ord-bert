#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for conf in $1/*.json; do

if  [[ $(basename $conf) == _* ]] ;
then
    continue
fi

echo "Submitting $conf"

sbatch <<EOT
#!/bin/bash
#SBATCH -t 16:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1,nvme:500
#SBATCH --partition=gpusmall
#SBATCH --account=project_2004993
#SBATCH --cpus-per-task=16

set -euo pipefail
IFS=$'\n\t'

rm -rf \$LOCAL_SCRATCH/frankier__hf_datasets
time cp -r /scratch/project_2004993/frankier/huggingface_cache/datasets \$LOCAL_SCRATCH/hf_datasets

module load cuda/11.5.0
module load tykky

HF_DATASETS_CACHE=\$LOCAL_SCRATCH/hf_datasets \
	./cpre/bin/python $SCRIPT_DIR/train.py \
	$conf

EOT

done
