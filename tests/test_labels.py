import json
from pathlib import Path

import pytest

from evaluation.labels import method_label

TIKTOKEN = {"vocab_size": 50257, "block_size": 256, "n_embd": 384, "n_layer": 6, "n_head": 6}


def rec(config=None, train=None):
    return {"config": {**TIKTOKEN, **(config or {})}, "train": train}


# ---------- every kind of run so far ----------

@pytest.mark.parametrize("record, expected", [
    # old records: no norm / warmup / clip keys at all -> defaults
    (rec(train={"learning_rate": 3e-4}), "Baseline"),
    (rec(train={"learning_rate": 1e-3}), "lr 1e-3"),
    (rec({"norm": "rmsnorm"}, {"learning_rate": 3e-4}), "RMSNorm"),
    # newer records: keys present, set to their defaults
    (rec({"norm": "layernorm"}, {"learning_rate": 1e-3, "grad_clip": None, "warmup_iters": 0}), "lr 1e-3"),
    (rec({"norm": "layernorm"}, {"learning_rate": 1e-3, "grad_clip": None, "warmup_iters": 200}), "lr 1e-3 + warmup 200"),
    (rec({"norm": "layernorm"}, {"learning_rate": 1e-3, "grad_clip": 1.0, "warmup_iters": 0}), "lr 1e-3 + clip 1.0"),
    (rec({"norm": "layernorm"}, {"learning_rate": 1e-3, "grad_clip": 1.0, "warmup_iters": 200}), "lr 1e-3 + warmup 200 + clip 1.0"),
])
def test_labels(record, expected):
    assert method_label(record) == expected


def test_char_run_is_flagged_even_without_train_section():
    # baseline_char has a tiny config and "train": null
    char = {"config": {"vocab_size": 65, "block_size": 8, "n_embd": 32, "n_layer": 3, "n_head": 4}, "train": None}
    assert method_label(char) == "char tokenizer (not comparable)"


def test_missing_sections_mean_baseline():
    assert method_label({}) == "Baseline"


def test_run_name_is_ignored():
    # the label comes from the config, never from what the run was called
    r = rec(train={"run_name": "rmsnorm", "learning_rate": 3e-4})
    assert method_label(r) == "Baseline"


def test_float_noise_from_pandas_still_counts_as_baseline_lr():
    # pd.read_json can read 0.0003 as 0.00030000000000000003
    assert method_label(rec(train={"learning_rate": 0.00030000000000000003})) == "Baseline"


# ---------- the real results file ----------

def test_every_record_in_results_gets_a_label():
    path = Path("evaluation/results.jsonl")
    if not path.exists():
        pytest.skip("no results file")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            label = method_label(json.loads(line))
            assert isinstance(label, str) and label
