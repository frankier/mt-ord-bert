# mt-ord-bert

This is the dataset/experiment specific code for running training and evaluation
of the ordinal regression NLP models.

## Installation

A tyykyy/mahti Conda yaml is in `env.yml`.

## How to run it

### Step 1) Prepare the datasets

    $ mkdir datasets
    $ python prep_rt.py rt_critics_big_irregular_5 datasets/rt_irr5
    $ python prep_rt.py rt_critics_one datasets/rt_one
    $ python prep_rt.py multiscale_rt_critics datasets/rt

### Step 2) Generate the jsons

    $ python gen_exps.py --data-root datasets --out rungroup__my_experiments

### Step 3) Train

    $ sbatch rungroup__my_experiments/jobs/blah_blah.slurm
