"""Shared, fail-closed helpers for the FTS v6 retrospective reanalysis."""
from __future__ import annotations
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
import numpy as np

VERSION = "6.0.0"
STATUS = "retrospective secondary reanalysis; confirmation cohort previously inspected"
NPZ_FILES = ("wesad_features_meta.npz", "physionet_stress_meta.npz",
             "physionet_exercise_active_meta.npz")

def runtime_versions():
    versions={"python":sys.version}
    for name in ["numpy","scipy","pandas","scikit-learn","xgboost-cpu","xgboost","shap","neurokit2","joblib"]:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    return versions

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False),
                          encoding="utf-8")

def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    needed = {"X", "feature_names", "groups", "conditions", "block_ids",
              "window_ids", "window_start_s", "window_end_s", "tier", "cost_cycles"}
    if needed - a.keys(): raise ValueError(f"{path}: missing {needed - a.keys()}")
    n, d = a["X"].shape
    if len(a["feature_names"]) != d: raise ValueError("Invalid feature schema")
    for k in ("groups", "conditions", "block_ids", "window_ids", "window_start_s", "window_end_s"):
        if len(a[k]) != n: raise ValueError(f"Misaligned metadata: {k}")
    if len(set(a["window_ids"].astype(str))) != n: raise ValueError("Duplicate window IDs")
    if not np.allclose(a["window_end_s"]-a["window_start_s"], 60):
        raise ValueError("Every analysis window must be 60 seconds")
    return a
