import sys
from bert_ordinal.datasets import load_data, save_to_disk_with_labels
from mt_ord_bert_utils import tokenize


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
save_to_disk_with_labels(sys.argv[1], dataset, num_labels)