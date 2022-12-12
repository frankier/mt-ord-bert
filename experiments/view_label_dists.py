import datasets
import streamlit as st
import polars as pl
import altair as alt


st.set_page_config(
    page_title="View label dists",
    page_icon="📊",
)


@st.cache
def load_data():
    d = datasets.load_dataset("frankier/processed_multiscale_rt_critics")
    assert isinstance(d, datasets.DatasetDict)
    return datasets.concatenate_datasets(list(d.values()))


def main():
    dataset = load_data()
    df = dataset.data.table.to_pandas()
    print(df)
    groups = df.groupby(["critic_name", "group_id", "publisher_name"]).size().to_frame("count").reset_index()
    print(groups)
    groups.sort_values("count", ascending=False, inplace=True)
    group_names = groups["critic_name"] + " @ " + groups["publisher_name"]  + "; " + groups["group_id"].astype(str) + " (" + groups["count"].astype(str) + ")"
    selection = st.selectbox("Critic name", group_names)
    critic_name, group_id = selection.split("; ")
    critic_name, _publisher_name = critic_name.split(" @ ")
    group_id, _cnt = group_id.split(" (")
    df_sel = df[(df["critic_name"] == critic_name) & (df["group_id"] == int(group_id))]
    print("df_sel", df_sel)

    scale_points = df_sel.iloc[0]["scale_points"]
    st.altair_chart(alt.Chart(df_sel).mark_bar().encode(
        alt.X("label:O", scale=alt.Scale(domain=list(range(scale_points)))),
        alt.Y("count()"),
        tooltip="review_score"
    ))


if __name__ == "__main__":
    main()
