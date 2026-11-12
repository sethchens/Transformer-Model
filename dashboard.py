import pandas as pd
import streamlit as st

RESULTS_PATH = "evaluation/results.jsonl"

# label shown in the UI -> (column in results.jsonl, unit, is lower better?)
MEASURES = {
    "Perplexity":         ("ppl",          "",        True),
    "Parameters":         ("param",        "",        True),
    "Train memory (MB)":  ("train_mem_mb", "MB",      True),
    "Infer memory (MB)":  ("inf_mem_mb",   "MB",      True),
    "Speed (tokens/s)":   ("tps",          "tok/s",   False),
}

# runs that aren't comparable with the tiktoken baseline (hidden by default)
HIDDEN_BY_DEFAULT = ["baseline_char", "stalled_lr1e-3"]


def load_results(path):
    """Read one JSON object per line into a table (re-read on every rerun)."""
    return pd.read_json(path, lines=True)


# ------------------ PAGE ------------------
st.set_page_config(page_title="Transformer Experiments", layout="wide")
st.title("Transformer Experiments")
st.caption(f"Reads {RESULTS_PATH} · every run compared to the baseline")

if st.button("Refresh"):
    st.rerun()

df = load_results(RESULTS_PATH)

# ------------------ CONTROLS ------------------
all_runs = df["name"].tolist()
default_runs = [r for r in all_runs if r not in HIDDEN_BY_DEFAULT]
runs = st.multiselect("Runs", all_runs, default=default_runs)

label = st.radio("Compare by", list(MEASURES), horizontal=True)
col, unit, lower_is_better = MEASURES[label]

view = df[df["name"].isin(runs)]

# ------------------ LAYOUT ------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("All runs")
    st.dataframe(
        view[["name", "ppl", "param", "train_mem_mb", "inf_mem_mb", "tps", "timestamp"]],
        hide_index=True,
        column_config={
            "name": "Run",
            "ppl": st.column_config.NumberColumn("Perplexity", format="%.3f"),
            "param": st.column_config.NumberColumn("Params", format="%d"),
            "train_mem_mb": st.column_config.NumberColumn("Train mem (MB)", format="%.0f"),
            "inf_mem_mb": st.column_config.NumberColumn("Infer mem (MB)", format="%.0f"),
            "tps": st.column_config.NumberColumn("Tokens/s", format="%.0f"),
            "timestamp": "When",
        },
    )

with right:
    better = "lower is better" if lower_is_better else "higher is better"
    st.subheader(f"{label} ({better})")
    if view.empty:
        st.info("Pick at least one run.")
    else:
        st.bar_chart(view.set_index("name")[col], horizontal=True)
