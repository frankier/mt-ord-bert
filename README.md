# mt-ord-bert

This is the dataset/experiment specific code for running training and evaluation
of the ordinal regression NLP models.

## How to run it

### Step 1) Prepare the datasets

    $ mkdir datasets
    $ python pre_rt.py datasets/rt

### Step 2) Generate the jsons

    $ python gen_exps.py --data-dir datasets --jsons-out link_rt_exp_jsons --log-root link_rt_exp_results

### Step 3) Train

    $ ./train.sh
