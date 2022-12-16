import argparse
from transformers import AutoTokenizer
import pandas

from bert_ordinal.datasets import auto_dataset
from bert_ordinal.transformers_utils import auto_load, auto_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Input model")
    parser.add_argument("dataset", help="Input dataset")
    parser.add_argument("outf", help="Dump output file")
    parser.add_argument("--normalize", help="Normalize hiddens", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, num_labels = auto_dataset(args.dataset)
    model = auto_load(args.model)
    for scale in model.scales:
        assert scale.weight > 0
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    pipeline = auto_pipeline(model=model, tokenizer=tokenizer)
    cols = {k: [] for k in ["task_ids", "label", "scale_points", "hidden"]}
    for idx, row in enumerate(dataset["train"]):
        cols["task_ids"].append(row["task_ids"])
        cols["label"].append(row["label"])
        cols["scale_points"].append(row["scale_points"])
        row_out = pipeline(row)
        cols["hidden"].append(row_out["hidden"])
    df = pandas.DataFrame(cols)
    if args.normalize:
        hidden_np = df["hidden"].to_numpy()
        df["hidden"] = (df["hidden"] - hidden_np.mean()) / hidden_np.std()
    df.to_parquet(args.outf)


if __name__ == "__main__":
    main()