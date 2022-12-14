import torch
import argparse
from bert_ordinal.transformers_utils import auto_load
import streamlit as st
import pandas
import altair as alt


def softmax_to_adjcat(softmax_weights):
    return softmax_weights[1:] - softmax_weights[:-1]


def all_pairs_cosine_similarity(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0,1))


def softmax_adjcat_cossim_score(softmax_weights):
    adjcat_weights = softmax_to_adjcat(softmax_weights)
    #print("Adjcat weights")
    #print(adjcat_weights)
    return all_pairs_cosine_similarity(adjcat_weights, adjcat_weights)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Input directory of model dump", required=True)
    return parser.parse_args()


@st.cache()
def get_model(model_path):
    print("Loading model")
    return auto_load(model_path)


@st.cache()
def get_weight_data(model):
    all_adjcat_weights = []
    for cls in model.classifiers:
        full_weight = torch.cat((cls.weight, cls.bias.unsqueeze(1)), dim=1)
        adjcat_weights = softmax_to_adjcat(full_weight)
        #print(softmax_adjcat_cossim_score(full_weight))
        all_adjcat_weights.append(adjcat_weights.detach().numpy())
    return all_adjcat_weights


def main():
    args = parse_args()
    model = get_model(args.model)
    all_adjcat_weights = get_weight_data(model)
    selected = st.selectbox("Task", range(len(all_adjcat_weights)))
    print("selected", selected)
    print(all_adjcat_weights[selected].T)
    colnames = [f"{i}=>{i+1}" for i in range(len(all_adjcat_weights[selected]))]
    df = pandas.DataFrame(all_adjcat_weights[selected].T, columns=colnames)
    print(df)
    st.altair_chart(
        alt.Chart(df).mark_point().encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=colnames,
            column=colnames
        ).interactive()
    )



if __name__ == "__main__":
    main()