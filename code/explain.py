"""Explain locked source-trained XGBoost predictions without fitting or tuning.

Interventional TreeSHAP on the raw (log-odds) scale. A common source-only
background is used for all feature configurations and target conditions.
Intervals resample participants conditional on the frozen model/background.
Attributions describe the fitted model, not physiological causation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import read_npz,NPZ_FILES,sha256,write_json,runtime_versions,STATUS
from statistics_report import sample_counts,ci,validate_target

OUTPUT_FILES=["shap_values.csv.gz","shap_background_windows.csv","shap_subject_condition.csv",
    "shap_condition_summary.csv","shap_condition_contrasts.csv","shap_participant_consistency.csv",
    "shap_error_groups.csv","shap_local_examples.csv","shap_audit.json",
    "figure_shap_conditions.png","figure_shap_dependence.png","figure_shap_local_examples.png"]

def background_indices(groups,n,seed):
    """Round-robin subject sampling; IDs selected before looking at outcomes."""
    if not 2<=n<=100:raise ValueError("SHAP background must contain 2–100 windows")
    rng=np.random.default_rng(seed)
    pools=[list(rng.permutation(np.flatnonzero(groups==s))) for s in sorted(set(groups))]
    result=[]
    while len(result)<min(n,len(groups)):
        for pool in pools:
            if pool and len(result)<n:result.append(int(pool.pop()))
    return np.asarray(result,int)

def summarize(long,cfg,out):
    long=long.copy();long["abs_shap"]=long.shap_value.abs();long["positive_shap"]=(long.shap_value>0).astype(float)
    subject=long.groupby(["feature_set","subject_id","condition","feature"],as_index=False).agg(
        signed_mean=("shap_value","mean"),absolute_mean=("abs_shap","mean"),
        positive_fraction=("positive_shap","mean"),n_windows=("window_id","nunique"))
    subject.to_csv(out/"shap_subject_condition.csv",index=False)
    rows=[];contrasts=[];consistency=[]
    for model,block in subject.groupby("feature_set"):
        subjects=sorted(block.subject_id.unique());n=len(subjects)
        draws=np.vstack([np.ones(n,int),sample_counts(n,cfg["n_bootstrap"],cfg["bootstrap_seed"])])
        bycondition={}
        for condition,c in block.groupby("condition"):
            bycondition[condition]={}
            for measure in ["signed_mean","absolute_mean","positive_fraction"]:
                matrix=c.pivot(index="subject_id",columns="feature",values=measure).reindex(subjects)
                if matrix.isna().any().any():raise ValueError("Incomplete participant-condition SHAP matrix")
                bycondition[condition][measure]=matrix
                values=draws@matrix.to_numpy()/n
                for j,feature in enumerate(matrix.columns):
                    lo,hi=ci(values[1:,j])
                    rows.append(dict(feature_set=model,condition=condition,feature=feature,measure=measure,
                        estimate=values[0,j],ci_low=lo,ci_high=hi,n_subjects=n,
                        scale="raw log-odds" if measure!="positive_fraction" else "fraction",
                        estimand="equal-participant average; fixed model and background"))
            matrix=bycondition[condition]["absolute_mean"]
            for sid in subjects:
                own=matrix.loc[sid];other=matrix.drop(sid).mean()
                rho=float(spearmanr(own,other).statistic) if own.nunique()>1 and other.nunique()>1 else np.nan
                k=min(3,len(own));a=set(own.sort_values(ascending=False,kind="stable").index[:k]);b=set(other.sort_values(ascending=False,kind="stable").index[:k])
                consistency.append(dict(feature_set=model,condition=condition,subject_id=sid,
                    spearman_vs_other_subjects=rho,top3_jaccard=len(a&b)/len(a|b),
                    interpretation="descriptive within-condition consistency; undefined rho retained as missing"))
        for test,reference in [("stress","rest"),("exercise","rest"),("exercise","stress")]:
            for measure in ["signed_mean","absolute_mean"]:
                a=bycondition[test][measure];b=bycondition[reference][measure]
                values=draws@(a-b).to_numpy()/n
                for j,feature in enumerate(a.columns):
                    lo,hi=ci(values[1:,j]);contrasts.append(dict(feature_set=model,test=test,reference=reference,
                        feature=feature,measure=measure,estimate=values[0,j],ci_low=lo,ci_high=hi,
                        contrast="test minus reference",n_paired_subjects=n,analysis_status=STATUS))
    summary=pd.DataFrame(rows);summary.to_csv(out/"shap_condition_summary.csv",index=False)
    pd.DataFrame(contrasts).to_csv(out/"shap_condition_contrasts.csv",index=False)
    pd.DataFrame(consistency).to_csv(out/"shap_participant_consistency.csv",index=False)
    # Outcome-conditioned summaries are descriptive and never selection criteria.
    long["error_group"]=np.select([
        long.condition.eq("rest")&long.y_pred.eq(0),long.condition.eq("rest")&long.y_pred.eq(1),
        long.condition.eq("stress")&long.y_pred.eq(1),long.condition.eq("stress")&long.y_pred.eq(0),
        long.condition.eq("exercise")&long.y_pred.eq(1)],
        ["correct_rest","rest_false_positive","detected_stress","missed_stress","exercise_positive"],
        default="exercise_negative")
    ep=long.groupby(["feature_set","error_group","subject_id","feature"],as_index=False).agg(
        signed_mean=("shap_value","mean"),absolute_mean=("abs_shap","mean"),n_windows=("window_id","nunique"))
    ep.groupby(["feature_set","error_group","feature"],as_index=False).agg(
        signed_mean=("signed_mean","mean"),absolute_mean=("absolute_mean","mean"),
        n_subjects=("subject_id","nunique"),n_windows=("n_windows","sum")).to_csv(out/"shap_error_groups.csv",index=False)
    return long,summary

def figures(long,summary,out):
    focus=long[long.feature_set.eq("fts6")].copy()
    order=focus.groupby("feature").abs_shap.mean().sort_values().index.tolist()
    colors={"rest":"#2878a5","stress":"#c84a43","exercise":"#d99423"}
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,measure,title in zip(axes,["absolute_mean","signed_mean"],
        ["Mean absolute contribution","Mean signed contribution"]):
        for j,(condition,color) in enumerate(colors.items()):
            d=summary[(summary.feature_set=="fts6")&(summary.condition==condition)&(summary.measure==measure)].set_index("feature").loc[order]
            y=np.arange(len(order))+(j-1)*.23
            ax.hlines(y,d.ci_low,d.ci_high,color=color);ax.scatter(d.estimate,y,label=condition,color=color,s=22)
        ax.axvline(0,c="gray",lw=.5);ax.set(yticks=range(len(order)),yticklabels=order,xlabel="TreeSHAP contribution (log-odds)",title=title)
        ax.legend()
    fig.suptitle("Frozen FTS6: participant means and marginal 95% bootstrap intervals")
    fig.tight_layout();fig.savefig(out/"figure_shap_conditions.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,feature in zip(axes.flat,order[::-1]):
        for condition,color in colors.items():
            d=focus[(focus.feature==feature)&(focus.condition==condition)]
            ax.scatter(d.raw_value,d.shap_value,s=9,alpha=.45,color=color,label=condition)
        ax.axhline(0,c="gray",lw=.5);ax.set(xlabel=feature+" (extracted units)",ylabel="SHAP (log-odds)")
    axes.flat[0].legend();fig.suptitle("Target feature–attribution associations; missing raw values omitted")
    fig.tight_layout();fig.savefig(out/"figure_shap_dependence.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8));examples=[]
    for ax,group in zip(axes.flat,["detected_stress","missed_stress","rest_false_positive","exercise_positive"]):
        d=focus[focus.error_group==group]
        if d.empty:
            ax.text(.5,.5,"No eligible window",ha="center",transform=ax.transAxes);ax.set_title(group)
            examples.append(dict(error_group=group,status="unavailable",window_id=""));continue
        candidates=d.drop_duplicates("window_id").copy()
        candidates["distance"]=(candidates.y_prob-candidates.y_prob.median()).abs()
        chosen=candidates.sort_values(["distance","window_id"],kind="stable").iloc[0]
        local=d[d.window_id==chosen.window_id].sort_values("shap_value")
        ax.barh(local.feature,local.shap_value,color=np.where(local.shap_value>=0,"#c84a43","#2878a5"))
        ax.axvline(0,c="gray",lw=.5);ax.set_xlabel("Contribution relative to source background (log-odds)")
        ax.set_title(f"{group}: {chosen.subject_id}\np={chosen.y_prob:.3f}, threshold={chosen.threshold:.3f}; base={chosen.base_value:.3f}")
        examples.append(dict(error_group=group,status="available",window_id=chosen.window_id,
            subject_id=chosen.subject_id,probability=chosen.y_prob,threshold=chosen.threshold,
            rule="closest to within-group median probability; window-ID tie-break"))
    pd.DataFrame(examples).to_csv(out/"shap_local_examples.csv",index=False)
    fig.tight_layout();fig.savefig(out/"figure_shap_local_examples.png",dpi=180);plt.close(fig)

def run(data_dir,model_dir,out,cfg):
    import shap
    data_dir,model_dir,out=map(Path,[data_dir,model_dir,out]);out.mkdir(parents=True,exist_ok=True)
    lock=json.loads((model_dir/"development_lock.json").read_text())
    if lock["configuration"]!=cfg or lock["runtime"]!=runtime_versions():raise ValueError("Changed configuration/runtime")
    if lock["models_script_sha256"]!=sha256(Path(__file__).with_name("models.py")):raise ValueError("Changed development code")
    for file,digest in lock["inputs"].items():
        if sha256(data_dir/file)!=digest:raise ValueError("Changed model input")
    W,P,E=[read_npz(data_dir/f) for f in NPZ_FILES]
    predictions=pd.read_csv(model_dir/"target_predictions.csv");validate_target(predictions)
    idx=background_indices(W["groups"],cfg.get("shap_background_windows",90),cfg.get("shap_seed",318))
    pd.DataFrame(dict(dataset="wesad",subject_id=W["groups"][idx],window_id=W["window_ids"][idx])).to_csv(out/"shap_background_windows.csv",index=False)
    allrows=[];audits=[]
    for name in cfg.get("shap_models",["fts6","fts5","all_efs","target_importance6"]):
        spec=lock["models"][name]
        if spec["learner"]!="xgboost":raise ValueError("This explainer is restricted to XGBoost")
        path=model_dir/spec["model_file"]
        if sha256(path)!=spec["sha256"]:raise ValueError("Changed model artifact")
        pipe=joblib.load(path);imp,tree=pipe.steps[0][1],pipe.steps[-1][1]
        cols=spec["columns"];background=imp.transform(W["X"][idx][:,cols])
        explainer=shap.TreeExplainer(tree,data=background,feature_perturbation="interventional",model_output="raw")
        base=float(np.asarray(explainer.expected_value).item())
        for dataset in (P,E):
            mask=np.isin(dataset["groups"],cfg["confirmation_subjects"])&np.isin(dataset["conditions"],["rest","stress","exercise"])
            raw=dataset["X"][mask][:,cols];x=imp.transform(raw)
            phi=np.asarray(explainer.shap_values(x,check_additivity=True))
            if phi.shape!=x.shape:raise ValueError("Unexpected binary TreeSHAP shape")
            margin=tree.predict(x,output_margin=True);prob=pipe.predict_proba(raw)[:,1]
            error=float(np.max(np.abs(base+phi.sum(axis=1)-margin)))
            if not np.allclose(base+phi.sum(axis=1),margin,atol=2e-4,rtol=2e-5):raise ValueError("SHAP additivity failed")
            if not np.allclose(expit(base+phi.sum(axis=1)),prob,atol=2e-5):raise ValueError("Probability reconstruction failed")
            ids=dataset["window_ids"][mask]
            old=predictions[predictions.feature_set==name].set_index("window_id").loc[ids]
            if not np.allclose(old.y_prob.to_numpy(),prob,atol=1e-8):raise ValueError("SHAP/prediction rows differ")
            for j,feature in enumerate(spec["features"]):
                allrows.append(pd.DataFrame(dict(feature_set=name,subject_id=dataset["groups"][mask],
                    window_id=ids,condition=dataset["conditions"][mask],feature=feature,
                    raw_value=raw[:,j],imputed_value=x[:,j],shap_value=phi[:,j],base_value=base,
                    y_prob=prob,y_pred=(prob>=spec["threshold"]).astype(int),threshold=spec["threshold"])))
            audits.append(dict(feature_set=name,conditions=sorted(set(dataset["conditions"][mask])),
                n_windows=len(ids),max_additivity_error=error,model_sha256=spec["sha256"]))
        if sha256(path)!=spec["sha256"]:raise RuntimeError("Explaining mutated a model artifact")
    long=pd.concat(allrows,ignore_index=True)
    if long.duplicated(["feature_set","window_id","feature"]).any():raise ValueError("Duplicate SHAP rows")
    long.to_csv(out/"shap_values.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    long,summary=summarize(long,cfg,out);figures(long,summary,out)
    write_json(out/"shap_audit.json",dict(status="passed",analysis_status=STATUS,checks=audits,
        background_dataset="source WESAD only",n_background_windows=len(idx),
        method="interventional TreeSHAP",scale="raw log-odds",no_refitting=True,
        intervals="participant bootstrap; fixed model and background; marginal, exploratory",
        limitations=["Feature dependence and background choice influence SHAP; no causal interpretation.",
            "Participant consistency is descriptive; training-seed/background stability is not established.",
            "Exercise positives are challenge responses, not clinically verified false positives.",
            "XAI is post-evaluation analysis and must not feed model selection on this cohort."]))
    print("Frozen-model SHAP export complete; additivity and prediction alignment passed.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--models",required=True)
    p.add_argument("--output",required=True);p.add_argument("--config",required=True);a=p.parse_args()
    run(a.data,a.models,a.output,json.loads(Path(a.config).read_text()))
