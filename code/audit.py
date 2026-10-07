"""Data review, raw-manifest validation, and derived-feature quality summaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from common import NPZ_FILES, read_npz, write_json

def manifest(path):
    m=pd.read_csv(path)
    required={"include","reviewed","protocol","protocol_version","raw_folder","subject_id",
        "start_tag_index","end_tag_index","start_s","end_s","analysis_start_s","analysis_end_s","condition","source_block"}
    if required-set(m):raise ValueError(f"Manifest missing columns: {required-set(m)}")
    for k in ["include","reviewed"]:
        if not m[k].fillna(0).isin([0,1]).all():raise ValueError(f"Nonbinary flag {k}")
    s=m[m.include==1].copy()
    if len(m)!=958 or len(s)!=226:raise ValueError("Expected the reviewed design: 958 source intervals, 226 selected")
    if s.condition.value_counts().to_dict()!={"exercise":104,"rest":62,"stress":31,"baseline":29}:raise ValueError("Condition totals changed")
    if s.subject_id.nunique()!=31:raise ValueError("Expected 31 selected target participants")
    if not set(s.protocol)<={"STRESS","AEROBIC"}:raise ValueError("Only stress and aerobic protocols are eligible")
    if (s[s.condition=="stress"].source_block!="TMCT").any():raise ValueError("Primary target stress must be TMCT")
    bounds=s[["start_s","end_s","analysis_start_s","analysis_end_s"]].to_numpy(float)
    if not np.isfinite(bounds).all():raise ValueError("Nonfinite interval bounds")
    if ((s.analysis_start_s<s.start_s)|(s.analysis_end_s>s.end_s)|(s.analysis_end_s-s.analysis_start_s<60-1e-8)).any():
        raise ValueError("Analysis bounds outside tags or shorter than 60 seconds")
    if s.duplicated(["protocol","raw_folder","start_tag_index","end_tag_index"]).any():raise ValueError("Duplicate selected intervals")
    if set(s.subject_id)&{"S02","S11","f07","f13","f14"}:raise ValueError("Historical exclusions changed; use a documented new protocol")
    for row in s.itertuples():
        if Path(row.raw_folder).name!=row.raw_folder or row.raw_folder in {".",".."}:raise ValueError("Unsafe raw folder name")
    print(s.groupby(["protocol","protocol_version","condition","source_block"]).size().to_string())
    print("Selected intervals:",len(s),"Selected participants:",s.subject_id.nunique())
    print("Structural checks passed. These checks do NOT validate semantic tag mapping.")
    return m,s

def units(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];flow=[];failures=[]
    for name in NPZ_FILES:
        a=read_npz(Path(data)/name)
        for condition in sorted(set(a["conditions"].astype(str))):
            mask=a["conditions"]==condition
            flow.append(dict(file=name,condition=condition,n_windows=int(mask.sum()),n_subjects=len(set(a["groups"][mask]))))
            for j,feature in enumerate(a["feature_names"].astype(str)):
                values=a["X"][mask,j].astype(float);finite=values[np.isfinite(values)]
                q=np.quantile(finite,[.05,.5,.95]) if len(finite) else [np.nan]*3
                rows.append(dict(file=name,condition=condition,feature=feature,tier=str(a["tier"][j]),
                    n_total=len(values),n_finite=len(finite),nonfinite_pct=100*(1-len(finite)/len(values)),
                    p05=q[0],median=q[1],p95=q[2]))
                if str(a["tier"][j])=="EFS" and not len(finite):failures.append(f"{name}/{condition}/{feature}: no finite values")
        if np.isinf(a["X"]).any():failures.append(name+": infinite feature values")
        for block in set(a["block_ids"].astype(str)):
            mask=a["block_ids"]==block
            starts=np.sort(a["window_start_s"][mask])
            if np.any(np.diff(starts)<60-1e-8):failures.append(name+": overlapping windows in "+block)
    table=pd.DataFrame(rows);table.to_csv(out/"unit_audit.csv",index=False)
    pd.DataFrame(flow).to_csv(out/"extracted_cohort.csv",index=False)
    core=table[table.feature.isin(["bvp_mean_hr","eda_mean","temp_mean","acc_mean"])]
    print(core.to_string(index=False))
    print("\nAll-feature missingness saved to unit_audit.csv. Acceleration remains native device units, not labeled g.")
    print("No mandatory exercise > stress > rest ordering. Finite values alone do not prove measurement validity.")
    write_json(out/"unit_gate.json",dict(hard_failures=failures,
        review_required=["units and raw acceleration scale","pulse-detection plausibility and motion artifacts",
            "missingness in all 23 features, including optional physiological decompositions","task labels and participant counts"]))
    if failures:raise ValueError("; ".join(failures))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["manifest","units"])
    p.add_argument("--path",required=True);p.add_argument("--output")
    x=p.parse_args()
    manifest(x.path) if x.action=="manifest" else units(x.path,x.output)
