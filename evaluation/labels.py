"""Turn a results.jsonl record into a readable method label.

The label is worked out from the record's saved config when results are read,
not stored, so old records get labels too and wording can change freely.
"""

import math

# what the baseline uses; a missing key in an old record means this default
BASELINE = {
    "vocab_size": 50257,
    "norm": "layernorm",
    "learning_rate": 3e-4,
    "warmup_iters": 0,
    "grad_clip": None,
}


def _get(d, key):
    """Read a key from a record section that may be missing or None."""
    if not isinstance(d, dict):
        return BASELINE[key]
    value = d.get(key, BASELINE[key])
    return BASELINE[key] if value is None and key != "grad_clip" else value


def _fmt_lr(lr):
    """3e-4, 1e-3 style (matches how the runs are talked about)."""
    mantissa, exp = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exp)}"


def method_label(record):
    """'Baseline', or the differences from it joined with ' + '.

    record: a dict (or pandas row) with optional 'config' and 'train' sections.
    """
    cfg = record.get("config")
    train = record.get("train")

    if _get(cfg, "vocab_size") != BASELINE["vocab_size"]:
        return "char tokenizer (not comparable)"

    parts = []
    if _get(cfg, "norm") != BASELINE["norm"]:
        parts.append("RMSNorm" if _get(cfg, "norm") == "rmsnorm" else _get(cfg, "norm"))
    lr = _get(train, "learning_rate")
    if not math.isclose(lr, BASELINE["learning_rate"]):  # pandas may read 3e-4 as 0.00030000000000000003
        parts.append(f"lr {_fmt_lr(lr)}")
    warmup = _get(train, "warmup_iters")
    if warmup != BASELINE["warmup_iters"]:
        parts.append(f"warmup {warmup}")
    clip = _get(train, "grad_clip")
    if clip is not None:
        parts.append(f"clip {clip}")

    return " + ".join(parts) if parts else "Baseline"
