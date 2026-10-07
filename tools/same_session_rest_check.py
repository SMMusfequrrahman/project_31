
"""Score locked models on the post-exercise rest period of the aerobic session."""
import argparse, hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

FS = 64
WIN = 60 * FS
NOMINAL_REST_S = 120.0
FEATURE_SETS = ["fts6", "fts5", "all_efs", "all_efs_no_mean_hr", "target_importance6"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rest_bounds(subject, tags, signal_seconds):
    """Rest period after cool-down, from the official protocol figures.
    Version 1 (S..): 12 tags; rest starts at tag 12 and runs to the end of the recording.
    Version 2 (f..): 8 tags; rest is the interval between tags 7 and 8.
    tags[0] is the recording start (0 s), as in the archived extraction code."""
    n_marks = len(tags) - 1
    if subject.startswith("S"):
        if n_marks != 12:
            return None, f"version 1 recording has {n_marks} tags, not 12 (protocol not completed as drawn)"
        start = float(tags[12]); end = min(signal_seconds, start + NOMINAL_REST_S)
    else:
        if n_marks != 8:
            return None, f"version 2 recording has {n_marks} tags, not 8 (protocol not completed as drawn)"
        start = float(tags[7]); end = min(float(tags[8]), signal_seconds, start + NOMINAL_REST_S)
    if end - start < 60:
        return None, f"only {end - start:.0f} s of signal in the rest period"
    return (start, end), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--phys", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, phys, out = Path(a.run), Path(a.phys), Path(a.out)
    sys.path.insert(0, str(run / "code"))
    import prepare_features as PF
    import recovery as R

    cfg = json.loads((run / "design.json").read_text())
    names = list(PF.FEATURE_ORDER)
    partitions = {s: "discovery" for s in cfg["discovery_subjects"]}
    partitions.update({s: "evaluation" for s in cfg["confirmation_subjects"]})

    # ---- 1. extract the rest windows with the archived extraction code ----------
    rows, skipped = [], []
    for subject, partition in partitions.items():
        folder = phys / "AEROBIC" / subject
        if not folder.is_dir():
            skipped.append(dict(subject_id=subject, partition=partition, reason="no AEROBIC folder")); continue
        try:
            signal, _ = PF.resample_folder(folder)
            tags = PF.tag_seconds(folder)
        except Exception as exc:
            skipped.append(dict(subject_id=subject, partition=partition, reason=f"read error: {exc}")); continue
        bounds, reason = rest_bounds(subject, tags, len(signal) / FS)
        if bounds is None:
            skipped.append(dict(subject_id=subject, partition=partition, reason=reason)); continue
        start, end = bounds
        left = int(np.ceil(start * FS)); k = 0
        while left + WIN <= min(len(signal), int(np.floor(end * FS))):
            w = signal[left:left + WIN]
            feats = PF.extract_features(w, FS).astype(float)
            _, pulse_flags, _, _ = R.pulse(w[:, 0])
            flags = {"BVP": list(pulse_flags), "TEMP": R.temp_flags(w[:, 2]),
                     "EDA": [] if np.isfinite(w[:, 1]).all() else ["nonfinite_raw"],
                     "ACC": [] if np.isfinite(w[:, 3]).all() else ["nonfinite_raw"]}
            efs = {n for n in names if PF.REGISTRY[n][0] == "EFS"}
            for channel, members in R.CHANNEL_FEATURES.items():
                if any(not np.isfinite(feats[names.index(m)]) for m in members if m in efs):
                    flags[channel].append("nonfinite_extracted_EFS_feature")
            row = dict(subject_id=subject, partition=partition, window_no=k,
                       window_start_s=left / FS, window_end_s=(left + WIN) / FS,
                       rest_start_s=start, rest_end_s=end,
                       flagged_channels=";".join(c for c, f in flags.items() if f))
            row.update(dict(zip(names, feats)))
            rows.append(row)
            left += WIN; k += 1
        print(f"{subject}: {k} same-session rest window(s)", flush=True)
    windows = pd.DataFrame(rows)
    pd.DataFrame(skipped, columns=["subject_id", "partition", "reason"]).to_csv(out / "skipped_participants.csv", index=False)
    if windows.empty:
        raise SystemExit("No same-session rest windows could be extracted. See skipped_participants.csv.")
    windows.to_csv(out / "same_session_rest_windows.csv", index=False)

    X_ref = windows[names].to_numpy(float)
    X_scr = X_ref.copy()
    for i, flagged in enumerate(windows.flagged_channels):
        for channel in [c for c in flagged.split(";") if c]:
            X_scr[i, [names.index(m) for m in R.CHANNEL_FEATURES[channel]]] = np.nan

    # ---- 2. score the locked models; no fitting, no threshold change ------------
    branch = Path(json.loads((run / "quality_branch.json").read_text())["path"])
    analyses = [("unscreened_reference", run, X_ref), ("quality_screening", branch, X_scr)]
    preds = []
    for label, root, X in analyses:
        lock_path = root / "models" / "development_lock.json"
        if not lock_path.is_file():
            lock_path = run / "models" / "development_lock.json"
        lock = json.loads(lock_path.read_text())
        model_dir = lock_path.parent
        for name in FEATURE_SETS:
            spec = lock["models"][name]
            path = model_dir / spec["model_file"]
            if sha(path) != spec["sha256"]:
                raise SystemExit(f"Locked model file changed: {path}")
            model = joblib.load(path)
            prob = model.predict_proba(X[:, spec["columns"]])[:, 1]
            preds.append(pd.DataFrame(dict(analysis=label, feature_set=name,
                subject_id=windows.subject_id.to_numpy(), partition=windows.partition.to_numpy(),
                window_no=windows.window_no.to_numpy(), y_prob=prob, threshold=spec["threshold"],
                y_pred=(prob >= spec["threshold"]).astype(int))))
    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(out / "same_session_rest_predictions.csv", index=False)

    # ---- 3. participant-level rates and bootstrap intervals ---------------------
    summary, per_subject = [], []
    for label, root, _ in analyses:
        old = pd.read_csv(root / "results" / "participant_condition_rates.csv")
        for name in FEATURE_SETS:
            same = preds[(preds.analysis == label) & (preds.feature_set == name)].groupby("subject_id").y_pred.mean()
            o = old[old.feature_set == name].set_index("subject_id")
            for group in ["evaluation", "discovery"]:
                members = [s for s, p in partitions.items() if p == group and s in same.index]
                if group == "evaluation":
                    missing_old = [s for s in members if s not in o.index]
                    if missing_old:
                        raise SystemExit(f"No archived rates for {missing_old}")
                    t = pd.DataFrame(dict(same_session_rest=same.reindex(members),
                                          cross_session_rest=o.rest.reindex(members),
                                          exercise=o.exercise.reindex(members)))
                    t["excess_same_session"] = t.exercise - t.same_session_rest
                    t["excess_cross_session"] = t.exercise - t.cross_session_rest
                else:
                    t = pd.DataFrame(dict(same_session_rest=same.reindex(members)))
                per_subject.append(t.reset_index().assign(analysis=label, feature_set=name, partition=group))
                n = len(t)
                if n < 2:
                    continue
                counts = np.random.default_rng(cfg["bootstrap_seed"]).multinomial(n, np.full(n, 1 / n), size=cfg["n_bootstrap"])
                for metric in t.columns:
                    v = t[metric].to_numpy(float)
                    b = counts @ v / n
                    lo, hi = np.quantile(b, [0.025, 0.975])
                    summary.append(dict(analysis=label, feature_set=name, partition=group, metric=metric,
                                        estimate=float(v.mean()), ci_low=float(lo), ci_high=float(hi),
                                        n_participants=n))
    pd.concat(per_subject, ignore_index=True).to_csv(out / "same_session_participant_rates.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "same_session_summary.csv", index=False)
    (out / "README.txt").write_text(
        "Post hoc same-session rest check; not part of the archived run.\n"
        "Rest = period after cool-down in the aerobic recording (v1: after tag 12; v2: tags 7-8), first 120 s, "
        "non-overlapping 60-s windows.\nLocked models and source-derived thresholds were applied unchanged.\n"
        "This rest follows exercise and may still contain recovery effects.\n"
        "Exercise and cross-session rest rates are the archived participant rates, restricted to the same participants.\n")
    print("\nSAME-SESSION CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
