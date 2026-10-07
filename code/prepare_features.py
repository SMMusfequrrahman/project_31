"""Raw 60-second extraction for the complete retrospective workflow.
Task mappings are checked against embedded official evidence, without claiming human review.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import pickle
import re
from pathlib import Path
from common import VERSION, sha256, write_json

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, find_peaks, welch

try:
    import neurokit2 as nk
except ImportError:
    nk = None


# ---------- CONFIGURATION (environment variables are easiest in Colab) ----------
MODE = os.environ.get("FTS_PREP_MODE", "template")
WESAD_DIR = Path(os.environ.get("FTS_WESAD_DIR", "/content/drive/MyDrive/WESAD"))
PHYS_ROOT = Path(os.environ.get("FTS_PHYS_ROOT", "/content/drive/MyDrive/PhysioNet_Dataset"))
OUT_DIR = Path(os.environ.get(
    "FTS_DATA_DIR", "/content/drive/MyDrive/FTS_publication_reanalysis/publication_data"
))
FS_TARGET = 64
WINDOW_SECONDS = 60
STRIDE_SECONDS = 60  # nonoverlapping windows for the publication reanalysis
MIN_LABEL_PURITY = 0.90
EPS = 1e-8
INCLUDE_EXERCISE_PROTOCOLS = {"AEROBIC"}
WESAD_EXPECTED_SUBJECTS = {
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
    "S11", "S13", "S14", "S15", "S16", "S17",
}
WESAD_LABEL_PROTOCOL = {
    "stress": 2,
    "rest": 1,
    "excluded": [0, 3, 4, 5, 6, 7],
}
# Anaerobic active bouts are 30/45 s in the source protocol and therefore do
# not support the paper's 60-s window definition. Analyze them separately only
# after changing the window definition for every dataset.
# -------------------------------------------------------------------------------

MANIFEST_PATH = OUT_DIR / "physionet_interval_manifest.csv"
RAW_INPUTS = {}

def record_input(path):
    path = Path(path)
    if str(path) not in RAW_INPUTS:
        RAW_INPUTS[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

FEATURE_ORDER = [
    "bvp_mean_hr", "bvp_sdnn", "bvp_rmssd", "bvp_pnn50",
    "eda_mean", "eda_std", "eda_slope", "eda_min", "eda_max", "eda_range",
    "temp_mean", "temp_std", "temp_slope", "acc_mean", "acc_std", "acc_max",
    "bvp_lf", "bvp_hf", "bvp_lf_hf", "eda_scl_mean", "eda_scr_count",
    "eda_scr_mean_amp", "eda_scr_auc",
]

REGISTRY = {
    "bvp_mean_hr": ("EFS", 13 * FS_TARGET * WINDOW_SECONDS + 100),
    "bvp_sdnn": ("EFS", 350), "bvp_rmssd": ("EFS", 400),
    "bvp_pnn50": ("EFS", 300), "eda_mean": ("EFS", 3854),
    "eda_std": ("EFS", 15388), "eda_slope": ("EFS", 15390),
    "eda_min": ("EFS", 3840), "eda_max": ("EFS", 3840),
    "eda_range": ("EFS", 7681), "temp_mean": ("EFS", 3854),
    "temp_std": ("EFS", 15388), "temp_slope": ("EFS", 15390),
    "acc_mean": ("EFS", 3854), "acc_std": ("EFS", 15388),
    "acc_max": ("EFS", 3840), "bvp_lf": ("COMP", 300000),
    "bvp_hf": ("COMP", 300000), "bvp_lf_hf": ("COMP", 300000),
    "eda_scl_mean": ("COMP", np.inf), "eda_scr_count": ("COMP", np.inf),
    "eda_scr_mean_amp": ("COMP", np.inf), "eda_scr_auc": ("COMP", np.inf),
}


def canonical_subject(folder_name: str) -> str:
    """Map split folders such as f14_a/f14_b to one participant ID."""
    return re.sub(r"_[ab]$", "", folder_name)


def parse_timestamp(value: object) -> dt.datetime:
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"Unrecognized timestamp: {text}") from exc


def e4_start_and_fs(path: Path) -> tuple[dt.datetime, float]:
    record_input(path)
    df = pd.read_csv(path, header=None)
    return parse_timestamp(df.iloc[0, 0]), float(df.iloc[1, 0])


def read_e4(path: Path, n_columns: int) -> tuple[np.ndarray, float, dt.datetime]:
    record_input(path)
    df = pd.read_csv(path, header=None)
    start = parse_timestamp(df.iloc[0, 0])
    fs = float(df.iloc[1, 0])
    values = df.iloc[2:, :n_columns].astype(float).to_numpy()
    if n_columns == 3:
        values = np.sqrt(np.sum(values ** 2, axis=1))
    else:
        values = values[:, 0]
    return values.astype(float), fs, start


def tag_seconds(folder: Path) -> np.ndarray:
    start, _ = e4_start_and_fs(folder / "EDA.csv")
    record_input(folder / "tags.csv")
    raw = pd.read_csv(folder / "tags.csv", header=None).iloc[:, 0].tolist()
    tags = [0.0]
    for value in raw:
        tags.append((parse_timestamp(value) - start).total_seconds())
    tags = np.asarray(tags, float)
    if np.any(np.diff(tags) <= 0):
        raise ValueError(f"Tags are not strictly increasing in {folder}")
    return tags


def resample_folder(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    channels = {}
    starts = []
    for name, ncol in (("BVP", 1), ("EDA", 1), ("TEMP", 1), ("ACC", 3)):
        values, fs, start = read_e4(folder / f"{name}.csv", ncol)
        channels[name] = (values, fs)
        starts.append(start)
    if len(set(starts)) != 1:
        raise ValueError(f"Signal start times differ in {folder}")
    duration = min(len(v) / fs for v, fs in channels.values())
    time_s = np.arange(0.0, duration, 1.0 / FS_TARGET)
    signal = np.column_stack([
        np.interp(time_s, np.arange(len(channels[name][0])) / channels[name][1], channels[name][0])
        for name in ("BVP", "EDA", "TEMP", "ACC")
    ])
    return signal, time_s


def suggested_stress_condition(folder_name: str, left_index: int) -> str:
    """Suggestion only; every included row still requires reviewed=1."""
    version = "v1" if folder_name.startswith("S") else "v2"
    stress_starts = {"v1": {5, 7, 9, 11}, "v2": {2, 4, 6, 8}}[version]
    if left_index == 0:
        return "baseline"
    if left_index in stress_starts:
        return "stress"
    return "rest_or_transition"


def generate_interval_manifest() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for protocol in ("STRESS", "AEROBIC", "ANAEROBIC"):
        for folder_text in sorted(glob.glob(str(PHYS_ROOT / protocol / "*"))):
            folder = Path(folder_text)
            if not folder.is_dir():
                continue
            try:
                tags = tag_seconds(folder)
            except Exception as exc:
                rows.append({
                    "protocol": protocol, "raw_folder": folder.name,
                    "subject_id": canonical_subject(folder.name),
                    "start_tag_index": -1, "end_tag_index": -1,
                    "start_s": np.nan, "end_s": np.nan, "duration_s": np.nan,
                    "suggested_condition": "READ_ERROR", "condition": "",
                    "include": 0, "reviewed": 0, "note": str(exc),
                })
                continue
            for i in range(len(tags) - 1):
                suggestion = (
                    suggested_stress_condition(folder.name, i)
                    if protocol == "STRESS" else "REVIEW_PROTOCOL_NOTEBOOK"
                )
                rows.append({
                    "protocol": protocol, "raw_folder": folder.name,
                    "subject_id": canonical_subject(folder.name),
                    "start_tag_index": i, "end_tag_index": i + 1,
                    "start_s": tags[i], "end_s": tags[i + 1],
                    "duration_s": tags[i + 1] - tags[i],
                    "suggested_condition": suggestion, "condition": "",
                    "include": 0, "reviewed": 0, "note": "",
                })
    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {MANIFEST_PATH}")
    print("Review every included interval against the official PhysioNet notebook.")
    print("Allowed conditions: baseline, rest, stress, exercise, cooldown, exclude")


def slope_per_second(signal: np.ndarray, fs: float) -> float:
    time_s = np.arange(len(signal), dtype=float) / fs
    return float(np.polyfit(time_s, signal, 1)[0])


def bandpass(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)


def ibi_ms(bvp: np.ndarray, fs: float) -> np.ndarray | None:
    clean = bandpass(bvp, 0.5, 8.0, fs)
    peaks, _ = find_peaks(
        clean, distance=int(0.33 * fs), prominence=max(np.std(clean) * 0.4, EPS)
    )
    return None if len(peaks) < 4 else np.diff(peaks) / fs * 1000.0


def extract_features(window: np.ndarray, fs: float) -> np.ndarray:
    bvp, eda, temp, acc = window.T
    ibi = ibi_ms(bvp, fs)
    values: dict[str, float] = {}
    if ibi is None:
        values.update({k: np.nan for k in ("bvp_mean_hr", "bvp_sdnn", "bvp_rmssd", "bvp_pnn50")})
    else:
        delta = np.diff(ibi)
        values.update({
            "bvp_mean_hr": 60000.0 / np.mean(ibi),
            "bvp_sdnn": np.std(ibi),
            "bvp_rmssd": np.sqrt(np.mean(delta ** 2)) if len(delta) else np.nan,
            "bvp_pnn50": np.mean(np.abs(delta) > 50) * 100 if len(delta) else np.nan,
        })
    if ibi is None or len(ibi) < 4:
        values.update({"bvp_lf": np.nan, "bvp_hf": np.nan, "bvp_lf_hf": np.nan})
    else:
        beat_t = np.cumsum(ibi) / 1000.0
        grid = np.arange(beat_t[0], beat_t[-1], 0.25)
        if len(grid) < 8:
            values.update({"bvp_lf": np.nan, "bvp_hf": np.nan, "bvp_lf_hf": np.nan})
        else:
            tach = np.interp(grid, beat_t, ibi)
            freq, power = welch(tach - tach.mean(), fs=4.0, nperseg=min(len(tach), 64))
            lf = np.trapezoid(power[(freq >= 0.04) & (freq < 0.15)], freq[(freq >= 0.04) & (freq < 0.15)])
            hf = np.trapezoid(power[(freq >= 0.15) & (freq < 0.40)], freq[(freq >= 0.15) & (freq < 0.40)])
            values.update({"bvp_lf": lf, "bvp_hf": hf, "bvp_lf_hf": lf / hf if hf > 0 else np.nan})
    values.update({
        "eda_mean": eda.mean(), "eda_std": eda.std(),
        "eda_slope": slope_per_second(eda, fs), "eda_min": eda.min(),
        "eda_max": eda.max(), "eda_range": eda.max() - eda.min(),
        "temp_mean": temp.mean(), "temp_std": temp.std(),
        "temp_slope": slope_per_second(temp, fs),
        "acc_mean": acc.mean(), "acc_std": acc.std(), "acc_max": acc.max(),
    })
    values.update({
        "eda_scl_mean": np.nan, "eda_scr_count": np.nan,
        "eda_scr_mean_amp": np.nan, "eda_scr_auc": np.nan,
    })
    if nk is not None and np.var(eda) >= 1e-4:
        try:
            factor = max(1, int(round(fs / 8)))
            eda_ds = decimate(eda, factor, ftype="fir", zero_phase=True) if factor > 1 else eda
            result = nk.eda_phasic(eda_ds, sampling_rate=fs / factor, method="cvxeda")
            tonic = result["EDA_Tonic"].to_numpy()
            phasic = result["EDA_Phasic"].to_numpy()
            _, info = nk.eda_peaks(phasic, sampling_rate=fs / factor)
            amps = np.asarray(info.get("SCR_Amplitude", []), float)
            amps = amps[np.isfinite(amps)]
            values.update({
                "eda_scl_mean": tonic.mean(), "eda_scr_count": len(amps),
                "eda_scr_mean_amp": amps.mean() if len(amps) else 0.0,
                "eda_scr_auc": np.trapezoid(np.clip(phasic, 0, None), dx=factor / fs),
            })
        except Exception:
            pass
    return np.asarray([values[name] for name in FEATURE_ORDER], np.float32)


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, ends))


def append_windows(
    signal: np.ndarray,
    start_s: float,
    end_s: float,
    subject_id: str,
    block_id: str,
    condition: str,
    dataset: str,
    protocol: str,
    rows: list[dict],
) -> None:
    if start_s < 0 or end_s > len(signal) / FS_TARGET + 1 / FS_TARGET:
        raise ValueError(f"Requested bounds exceed signal coverage: {block_id}")
    start = max(0, int(np.ceil(start_s * FS_TARGET)))
    end = min(len(signal), int(np.floor(end_s * FS_TARGET)))
    win = int(WINDOW_SECONDS * FS_TARGET)
    step = int(STRIDE_SECONDS * FS_TARGET)
    for left in range(start, end - win + 1, step):
        feat = extract_features(signal[left:left + win], FS_TARGET)
        row = {
            "dataset": dataset, "subject_id": subject_id, "block_id": block_id,
            "condition": condition, "protocol": protocol,
            "window_start_s": left / FS_TARGET,
            "window_end_s": (left + win) / FS_TARGET,
            "window_id": f"{dataset}:{block_id}:{left}:{left+win}",
        }
        row.update(dict(zip(FEATURE_ORDER, feat)))
        rows.append(row)


def extract_wesad() -> pd.DataFrame:
    if os.environ.get("FTS_TRUST_WESAD_PICKLES") != "1":
        raise RuntimeError("Confirm official trusted WESAD pickle files before loading them.")
    rows = []
    fs_native = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
    for path_text in sorted(glob.glob(str(WESAD_DIR / "S*" / "S*.pkl"))):
        path = Path(path_text)
        record_input(path)
        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        subject = path.parent.name
        wrist = data["signal"]["wrist"]
        acc = np.asarray(wrist["ACC"], float)
        channels = {
            "BVP": np.asarray(wrist["BVP"], float).ravel(),
            "EDA": np.asarray(wrist["EDA"], float).ravel(),
            "TEMP": np.asarray(wrist["TEMP"], float).ravel(),
            "ACC": np.sqrt(np.sum(acc ** 2, axis=1)),
        }
        labels = np.asarray(data["label"]).ravel()
        duration = min([len(channels[k]) / fs_native[k] for k in channels] + [len(labels) / 700])
        time_s = np.arange(0, duration, 1 / FS_TARGET)
        signal = np.column_stack([
            np.interp(time_s, np.arange(len(channels[k])) / fs_native[k], channels[k])
            for k in ("BVP", "EDA", "TEMP", "ACC")
        ])
        label_idx = np.clip(np.round(time_s * 700).astype(int), 0, len(labels) - 1)
        label_64 = labels[label_idx]
        # Preserve physical/raw feature units. Any imputation or scaling used by
        # a learner must be fitted within its source-training partition.
        for condition, code in (
            ("rest", WESAD_LABEL_PROTOCOL["rest"]),
            ("stress", WESAD_LABEL_PROTOCOL["stress"]),
        ):
            for run_no, (left, right) in enumerate(contiguous_runs(label_64 == code)):
                append_windows(
                    signal, left / FS_TARGET, right / FS_TARGET, subject,
                    f"{subject}_{condition}_{run_no}", condition, "wesad", "WESAD", rows,
                )
        print(f"WESAD {subject}: {sum(r['subject_id'] == subject for r in rows)} windows")
    result = pd.DataFrame(rows)
    if result.empty: raise ValueError("No WESAD windows: inspect raw paths and files")
    observed_subjects = set(result["subject_id"].astype(str))
    if observed_subjects != WESAD_EXPECTED_SUBJECTS:
        raise ValueError(
            "Unexpected WESAD participant set. Missing="
            f"{sorted(WESAD_EXPECTED_SUBJECTS - observed_subjects)}, extra="
            f"{sorted(observed_subjects - WESAD_EXPECTED_SUBJECTS)}"
        )
    if set(result["condition"].astype(str)) != {"rest", "stress"}:
        raise ValueError("WESAD must contain baseline/rest and TSST stress only")
    # Windows restart at condition boundaries and are nonoverlapping within a
    # block. This avoids the contradictory 1,735/2,031-window protocols and
    # reduces pseudo-replication from heavily overlapping windows.
    for _, block in result.groupby(["subject_id", "block_id"], sort=False):
        starts = np.sort(block["window_start_s"].to_numpy(float))
        if len(starts) > 1 and np.any(np.diff(starts) < WINDOW_SECONDS - 1e-9):
            raise ValueError("Overlapping WESAD publication windows detected")
    return result


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    from recovery import OFFICIAL
    required = {"protocol", "protocol_version", "source_block", "raw_folder", "subject_id",
                "start_tag_index", "end_tag_index", "condition", "include"}
    if not required.issubset(manifest.columns):
        raise ValueError("Manifest missing columns: " + str(sorted(required-set(manifest.columns))))
    chosen = manifest[manifest.include.astype(int)==1].copy()
    if set(chosen.condition) != {"rest", "stress", "exercise"}:
        raise ValueError("This workflow requires rest, TMCT stress and active aerobic exercise; no target baseline.")
    for r in chosen.itertuples():
        key=(r.protocol,r.protocol_version,r.source_block)
        if OFFICIAL.get(key)!=(int(r.start_tag_index),int(r.end_tag_index)):
            raise ValueError("Unsupported official protocol mapping: " + str(key))
        expected='exercise' if r.protocol=='AEROBIC' else 'stress' if r.source_block=='TMCT' else 'rest'
        if r.condition!=expected: raise ValueError("Task name/condition mismatch: " + str(key))
    if chosen.duplicated(['protocol','raw_folder','start_tag_index','end_tag_index']).any():
        raise ValueError("Duplicate included interval")
    return chosen


def interval_signal(row: pd.Series) -> tuple[np.ndarray, float, float]:
    folder = PHYS_ROOT / row["protocol"] / row["raw_folder"]
    signal, _ = resample_folder(folder)
    tags = tag_seconds(folder)
    i, j = int(row["start_tag_index"]), int(row["end_tag_index"])
    if not (0 <= i < j < len(tags)):
        raise ValueError(f"Bad tag indices for {folder}: {i}, {j}, n={len(tags)}")
    return signal, float(tags[i]), float(tags[j])


def extract_physionet(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Extract features in their physical/raw units. In particular, do not use a
    # stress-session baseline to normalize an aerobic recording acquired on a
    # different day. Model preprocessing is fitted from source-training data.
    signal_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def cached(protocol: str, raw_folder: str) -> tuple[np.ndarray, np.ndarray]:
        key = (protocol, raw_folder)
        if key not in signal_cache:
            # Bound memory usage: never retain every long raw recording.
            signal_cache.clear()
            folder = PHYS_ROOT / protocol / raw_folder
            signal_cache[key] = (resample_folder(folder)[0], tag_seconds(folder))
        return signal_cache[key]

    def reviewed_bounds(row: pd.Series, tags: np.ndarray) -> tuple[float, float]:
        """Use audited analysis bounds when supplied; otherwise tag boundaries."""
        i, j = int(row["start_tag_index"]), int(row["end_tag_index"])
        if not 0 <= i < j < len(tags): raise ValueError("Invalid live tag indices")
        if not np.allclose([tags[i], tags[j]], [row["start_s"], row["end_s"]], atol=0.05, rtol=0):
            raise ValueError(f"Raw tags disagree with saved manifest: {row['raw_folder']}")
        start_s, end_s = float(tags[i]), float(tags[j])
        if "analysis_start_s" in row.index and pd.notna(row["analysis_start_s"]):
            start_s = float(row["analysis_start_s"])
        if "analysis_end_s" in row.index and pd.notna(row["analysis_end_s"]):
            end_s = float(row["analysis_end_s"])
        if not (tags[i] <= start_s < end_s <= tags[j]):
            raise ValueError(
                f"Analysis bounds [{start_s}, {end_s}] fall outside tag interval "
                f"[{tags[i]}, {tags[j]}] for {row['protocol']}/{row['raw_folder']}"
            )
        return start_s, end_s

    rows = []
    flow = []
    for _, row in manifest.iterrows():
        condition = str(row["condition"])
        if condition not in {"baseline", "rest", "stress", "exercise"}:
            continue
        if condition == "exercise" and row["protocol"] not in INCLUDE_EXERCISE_PROTOCOLS:
            continue
        subject = str(row["subject_id"])
        signal, tags = cached(row["protocol"], row["raw_folder"])
        i, j = int(row["start_tag_index"]), int(row["end_tag_index"])
        start_s, end_s = reviewed_bounds(row, tags)
        block = f"{row['protocol']}_{row['raw_folder']}_tags{i}-{j}"
        before = len(rows)
        append_windows(
            signal, start_s, end_s, subject, block, condition,
            "physionet", str(row["protocol"]), rows,
        )
        flow.append({
            "dataset": "physionet", "subject_id": subject,
            "protocol": row["protocol"], "included": int(len(rows) > before),
            "exclusion_reason": "" if len(rows) > before else "interval shorter than window",
        })
    all_rows = pd.DataFrame(rows)
    stress = all_rows[all_rows["condition"].isin(["baseline", "rest", "stress"])].copy()
    exercise = all_rows[all_rows["condition"] == "exercise"].copy()
    return stress, exercise, pd.DataFrame(flow).drop_duplicates()


def save_npz(df: pd.DataFrame, path: Path, include_y: bool) -> None:
    arrays = {
        "X": df[FEATURE_ORDER].to_numpy(np.float32),
        "groups": df["subject_id"].to_numpy(dtype=str),
        "block_ids": df["block_id"].to_numpy(dtype=str),
        "conditions": df["condition"].to_numpy(dtype=str),
        "protocols": df["protocol"].to_numpy(dtype=str),
        "window_start_s": df["window_start_s"].to_numpy(float),
        "window_end_s": df["window_end_s"].to_numpy(float),
        "window_ids": df["window_id"].to_numpy(dtype=str),
        "format_version": np.asarray(VERSION),
        "feature_names": np.asarray(FEATURE_ORDER),
        "tier": np.asarray([REGISTRY[name][0] for name in FEATURE_ORDER]),
        "cost_cycles": np.asarray([REGISTRY[name][1] for name in FEATURE_ORDER], float),
        "window_seconds": np.asarray(WINDOW_SECONDS, dtype=np.int32),
        "stride_seconds": np.asarray(STRIDE_SECONDS, dtype=np.int32),
        "preprocessing_scope": np.asarray(
            "raw physical/device units; learner imputation fitted on training partitions"
        ),
    }
    if include_y:
        arrays["y"] = (df["condition"] == "stress").astype(np.int8).to_numpy()
    np.savez_compressed(path, **arrays)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MODE == "template":
        generate_interval_manifest()
        return
    if MODE != "extract":
        raise ValueError("MODE must be 'template' or 'extract'")
    if nk is None:
        raise RuntimeError("NeuroKit2 is required to preserve the original 23-feature scoring schema.")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Run MODE='template' first: {MANIFEST_PATH}")
    manifest = validate_manifest(pd.read_csv(MANIFEST_PATH))
    wesad = extract_wesad()
    phys_stress, phys_exercise, flow = extract_physionet(manifest)
    if phys_stress.empty or phys_exercise.empty:
        raise ValueError("PhysioNet extraction produced an empty arm")
    save_npz(wesad, OUT_DIR / "wesad_features_meta.npz", include_y=True)
    save_npz(phys_stress, OUT_DIR / "physionet_stress_meta.npz", include_y=True)
    save_npz(phys_exercise, OUT_DIR / "physionet_exercise_active_meta.npz", include_y=False)
    wesad_flow = (
        wesad.groupby(["dataset", "subject_id", "protocol", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
    )
    phys_flow = flow.copy()
    phys_flow["condition"] = "manifest_interval"
    phys_flow["n_windows"] = np.nan
    combined_flow = pd.concat(
        [
            wesad_flow,
            phys_flow[["dataset", "subject_id", "protocol", "condition", "n_windows"]],
        ],
        ignore_index=True,
    )
    combined_flow.to_csv(OUT_DIR / "subject_flow.csv", index=False)
    all_windows = pd.concat([wesad, phys_stress, phys_exercise], ignore_index=True)
    meta = ["dataset", "subject_id", "block_id", "condition", "protocol", "window_id", "window_start_s", "window_end_s"]
    all_windows[meta].to_csv(OUT_DIR / "window_index.csv", index=False)
    all_windows.groupby(["dataset", "subject_id", "protocol", "condition"]).size().rename("n_windows").reset_index().to_csv(OUT_DIR / "window_flow.csv", index=False)
    write_json(OUT_DIR / "raw_input_hashes.json", RAW_INPUTS)
    print("Saved unscreened reference features to", OUT_DIR)
    print("WESAD", wesad.shape, "PhysioNet stress", phys_stress.shape, "exercise", phys_exercise.shape)


if __name__ == "__main__":
    main()
