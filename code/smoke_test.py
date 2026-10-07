"""Actual XGBoost/SHAP end-to-end test on SYNTHETIC data, run in Colab.

This checks software integration, not scientific validity or expected real F1.
Synthetic files are segregated from the real run and never enter paper outputs.
"""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from common import write_json,NPZ_FILES
from prepare_features import FEATURE_ORDER,REGISTRY,save_npz
from models import Engine
from statistics_report import run
from explain import run as explain_run,OUTPUT_FILES

def make_data(directory):
    rng=np.random.default_rng(403)
    def frame(subjects,dataset,conditions):
        rows=[]
        for s in subjects:
            offset=rng.normal(0,.15)
            for c in conditions:
                for j in range(4):
                    y=int(c=="stress");v=rng.normal(y*.5+offset,1,len(FEATURE_ORDER))
                    row=dict(zip(FEATURE_ORDER,v))
                    row.update(dataset=dataset,subject_id=s,block_id=f"{s}_{c}",condition=c,
                        protocol="AEROBIC" if c=="exercise" else "STRESS",window_start_s=j*60.,
                        window_end_s=(j+1)*60.,window_id=f"{dataset}:{s}:{c}:{j}")
                    rows.append(row)
        return pd.DataFrame(rows)
    target=["S01","S02","S03","S04","f01","f02","f03","f04"]
    for df,name,y in [(frame([f"W{i}" for i in range(6)],"wesad",["rest","stress"]),NPZ_FILES[0],True),
        (frame(target,"physionet",["baseline","rest","stress"]),NPZ_FILES[1],True),
        (frame(target,"physionet",["exercise"]),NPZ_FILES[2],False)]:save_npz(df,directory/name,y)
    return dict(seed=17,cv_repeats=1,cv_folds=3,xgb_trees=8,rf_trees=8,run_ablations=True,
        discovery_subjects=["S01","S02","f01","f02"],confirmation_subjects=["S03","S04","f03","f04"],
        n_bootstrap=30,bootstrap_seed=20260908,equivalence_margin=.02,
        cpu_hz=64000000,voltage_v=3.,active_current_a=.00333)

def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # A unique directory prevents test artifacts from ever being reused as real data.
    td=Path(tempfile.mkdtemp(prefix="synthetic_",dir=root))
    data=td/"data";data.mkdir();out=td/"models";results=td/"results"
    cfg=make_data(data)
    engine=Engine(cfg,data);engine.develop(out);engine.evaluate(out)
    run(data,out,results,cfg)
    explain_run(data,out,results,cfg)
    for f in OUTPUT_FILES:
        if not (results/f).is_file():raise AssertionError("Missing XAI output: "+f)
    audit=json.loads((results/"shap_audit.json").read_text())
    if audit["status"]!="passed":raise AssertionError("XAI audit failed")
    contrasts=pd.read_csv(results/"table_target_paired_differences.csv")
    if not ((contrasts.test=="all_efs_no_mean_hr")&(contrasts.reference=="all_efs")).any():
        raise AssertionError("Direct HR-removal contrast missing")
    # Reordering an identical membership must leave model predictions unchanged.
    ix_all=np.arange(len(engine.X));cols=list(range(6));reversed_cols=cols[::-1]
    a=engine.fit(ix_all,cols,119);b=engine.fit(ix_all,reversed_cols,119)
    if not np.array_equal(a.predict_proba(engine.X[:,cols]),b.predict_proba(engine.X[:,cols])):
        raise AssertionError("Column-order invariance failed")
    if engine.threshold(ix_all,cols,119)!=engine.threshold(ix_all,reversed_cols,119):
        raise AssertionError("Threshold column-order invariance failed")
    selected=json.loads((out/"fold_selections.json").read_text())
    for row in selected:
        if set(row["source_train_subjects"])&set(row["source_test_subjects"]):raise AssertionError("Source overlap")
        if row["feature_set"]=="fts5" and row["n_features"]!=5:raise AssertionError("FTS5 count")
    # Perturb excluded source/target rows: selection on the retained development data must not change.
    ix=np.flatnonzero(engine.g!="W5")
    first,_,_=engine.select(ix,101,"invariance_original")
    engine.X[engine.g=="W5"]+=9999
    pc=np.isin(engine.P["groups"],engine.conf)
    engine.P["X"][pc]+=9999;engine.P["y"][pc]=1-engine.P["y"][pc]
    engine.E["X"][np.isin(engine.E["groups"],engine.conf)]+=9999
    second,_,_=engine.select(ix,101,"invariance_perturbed")
    if first!=second:raise AssertionError("Excluded observations influenced development selection")
    write_json(root/"smoke_test_passed.json",dict(kind="synthetic XGBoost/SHAP integration test",status="passed",artifacts=str(td),
        tests=["complete model/statistics stages","exactly five FTS5 features","no source participant overlap",
            "selection invariant to held-out source and target-confirmation perturbation",
            "canonical feature-order prediction and threshold invariance","direct paired HR-removal contrast",
            "frozen-model SHAP additivity and prediction alignment","complete XAI tables and figures"],
        limitation="No raw-data scientific validation or publication guarantee"))
    print("SYNTHETIC SOFTWARE TEST PASSED. These are not publication results.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);main(p.parse_args().output)
