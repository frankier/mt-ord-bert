import sys
import pickle
from os.path import join as pjoin
from bert_ordinal.datasets import load_data
from .mt_ord_bert_utils import tokenize


dataset, num_labels, _ = load_data(
    "multiscale_rt_critics", num_dataset_proc=8,
)
dataset = dataset.map(
    tokenize,
    input_columns="text",
    batched=True,
    desc="Tokenizing",
    num_proc=8,
)
dataset.save_to_disk(sys.argv[1])
pickle.dump(num_labels, open(pjoin(sys.argv[1], "num_labels.pkl"), "wb"))