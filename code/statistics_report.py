"""Complete secondary reporting. Fixed predictions; participant-paired resampling.

No significance-based success gate, automatic target threshold tuning, or
unsupported task-block permutation test. Bootstrap intervals are conditional
on the fitted models and sampled cohort; they do not capture all training or
population-selection uncertainty. New comparisons are retrospective.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (f1_score,roc_auc_score,average_precision_score,
    balanced_accuracy_score,confusion_matrix,roc_curve,precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import STATUS, write_json

def div(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b!=0)

def sample_counts(n,repetitions,seed):
    if n<2: raise ValueError("At least two participants are needed")
    # Each row is ONE paired bootstrap draw used for every feature configuration.
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=repetitions)

def metric_batch(frame, subjects, counts):
    """Exact weighted confusion, AUROC (half-credit ties), and stepwise AP."""
    index={s:i for i,s in enumerate(subjects)}
    gi=np.array([index[s] for s in frame.subject_id])
    w=counts[:,gi].astype(float)
    y=frame.y_true.to_numpy(int);p=frame.y_prob.to_numpy(float);h=frame.y_pred.to_numpy(int)
    tn=w@((y==0)&(h==0));fp=w@((y==0)&(h==1))
    fn=w@((y==1)&(h==0));tp=w@((y==1)&(h==1))
    sens=div(tp,tp+fn);spec=div(tn,tn+fp)
    out=dict(macro_f1=.5*(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn)),
        sensitivity=sens,specificity=spec,balanced_accuracy=(sens+spec)/2,
        precision=div(tp,tp+fp),npv=div(tn,tn+fn),accuracy=div(tp+tn,tp+tn+fp+fn),
        brier=div(w@((p-y)**2),w.sum(axis=1)),prevalence=div(tp+fn,w.sum(axis=1)))
    order=np.argsort(p,kind="stable");sp=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sp)!=0)+1]
    pos=np.add.reduceat(w[:,order]*y[order],starts,axis=1)
    neg=np.add.reduceat(w[:,order]*(1-y[order]),starts,axis=1)
    neg_before=np.cumsum(neg,axis=1)-neg
    out["auroc"]=div((pos*(neg_before+.5*neg)).sum(axis=1),pos.sum(axis=1)*neg.sum(axis=1))
    pos_desc=pos[:,::-1];neg_desc=neg[:,::-1]
    precision=div(np.cumsum(pos_desc,axis=1),np.cumsum(pos_desc+neg_desc,axis=1))
    out["average_precision"]=div(np.nansum(pos_desc*precision,axis=1),pos.sum(axis=1))
    return out,(tn,fp,fn,tp)

def ci(samples):
    a=np.asarray(samples); a=a[np.isfinite(a)]
    return tuple(np.quantile(a,[.025,.975])) if len(a) else (np.nan,np.nan)

def comparison_pairs(names):
    """Fixed retrospective contrasts, including the direct HR-removal test."""
    names=set(names)
    pairs=[(x,"fts6") for x in sorted(names) if x!="fts6"]
    pairs += [(x,"all_efs") for x in ["fts6","accuracy6","target_importance6","all_efs_no_mean_hr","logistic_all_efs"] if x in names]
    return list(dict.fromkeys((a,b) for a,b in pairs if a in names and b in names and a!=b))

def validate_target(p):
    needed={"subject_id","feature_set","window_id","block_id","condition","y_true","y_prob","y_pred","threshold"}
    if needed-set(p):raise ValueError(f"Missing prediction columns: {needed-set(p)}")
    if p.duplicated(["feature_set","window_id"]).any():raise ValueError("Duplicate target prediction keys")
    if not p.y_true.isin([0,1]).all() or not p.y_pred.isin([0,1]).all(): raise ValueError("Nonbinary outcome")
    if not np.isfinite(p.y_prob).all() or not p.y_prob.between(0,1).all():raise ValueError("Invalid probabilities")
    if not np.array_equal(p.y_pred.to_numpy(),(p.y_prob>=p.threshold).to_numpy(int)):
        raise ValueError("Predictions do not match their exported thresholds")
    if not np.array_equal(p.y_true.to_numpy(),p.condition.eq("stress").to_numpy(int)):
        raise ValueError("Condition/outcome mismatch")
    reference=p[p.feature_set=="fts6"].set_index("window_id").sort_index()
    if reference.empty:raise ValueError("FTS6 comparison reference missing")
    for name,d in p.groupby("feature_set"):
        d=d.set_index("window_id").sort_index()
        if not d.index.equals(reference.index):raise ValueError(f"Unmatched target windows: {name}")
        for key in ["subject_id","y_true","condition","block_id"]:
            if not d[key].equals(reference[key]):raise ValueError(f"Unmatched {key}: {name}")

def target_tables(p,cfg,out):
    subjects=sorted(p.subject_id.unique());n=len(subjects)
    counts=sample_counts(n,cfg["n_bootstrap"],cfg["bootstrap_seed"])
    draws=np.vstack([np.ones(n,dtype=int),counts])
    summaries=[];operating=[];differences=[];boot={};primary_frames={}
    for scope,conditions in [("primary_rest_stress",["rest","stress"])]:
        for name,d in p[p.condition.isin(conditions)].groupby("feature_set"):
            metrics,cm=metric_batch(d,subjects,draws)
            if scope=="primary_rest_stress":boot[name]=metrics;primary_frames[name]=d
            for metric,values in metrics.items():
                lo,hi=ci(values[1:])
                summaries.append(dict(scope=scope,feature_set=name,metric=metric,estimate=values[0],
                    ci_low=lo,ci_high=hi,n_windows=len(d),n_subjects=n,n_valid_bootstrap=int(np.isfinite(values[1:]).sum()),
                    analysis_status=STATUS))
            tn,fp,fn,tp=[int(a[0]) for a in cm]
            operating.append(dict(scope=scope,feature_set=name,tn=tn,fp=fp,fn=fn,tp=tp,
                threshold=float(d.threshold.iloc[0]),n_windows=len(d),n_subjects=n,
                **{k:float(v[0]) for k,v in metrics.items()}))
    for name,reference in comparison_pairs(boot):
        m=boot[name]
        for metric in m:
            delta=m[metric]-boot[reference][metric];lo,hi=ci(delta[1:])
            differences.append(dict(test=name,reference=reference,metric=metric,
                contrast="test minus reference",estimate=delta[0],ci_low=lo,ci_high=hi,
                n_subjects=n,analysis_status=STATUS))
    pd.DataFrame(summaries).to_csv(out/"table_target_metrics.csv",index=False)
    pd.DataFrame(operating).to_csv(out/"table_target_operating.csv",index=False)
    pd.DataFrame(differences).to_csv(out/"table_target_paired_differences.csv",index=False)
    # Equal-participant averaging is a separate estimand from pooled-window F1.
    sub=[]
    for name,d in primary_frames.items():
        for s,g in d.groupby("subject_id"):
            m,_=metric_batch(g,[s],np.ones((1,1),int))
            sub.append(dict(feature_set=name,subject_id=s,**{k:v[0] for k,v in m.items()}))
    pd.DataFrame(sub).to_csv(out/"target_subject_metrics.csv",index=False)
    participant_means=[]
    for name,g in pd.DataFrame(sub).groupby("feature_set"):
        g=g.set_index("subject_id").reindex(subjects)
        for metric in boot[name]:
            values=g[metric].to_numpy();b=draws@values/n
            lo,hi=ci(b[1:]);participant_means.append(dict(feature_set=name,metric=metric,
                estimate=b[0],ci_low=lo,ci_high=hi,estimand="mean of participant-specific metrics"))
    pd.DataFrame(participant_means).to_csv(out/"table_target_participant_average.csv",index=False)
    # Absolute condition rates and their exact excess decomposition.
    rates={};rate_rows=[];individual=[]
    for name,d in p[p.condition.isin(["rest","exercise"])].groupby("feature_set"):
        r=d.groupby(["subject_id","condition"]).y_pred.mean().unstack().reindex(subjects)
        if r[["rest","exercise"]].isna().any().any():raise ValueError("Incomplete paired rest/exercise cohort")
        b={c:draws@r[c].to_numpy()/n for c in ["rest","exercise"]}
        b["excess"]=b["exercise"]-b["rest"];rates[name]=b
        individual.append(r.reset_index().assign(feature_set=name,excess=r.exercise.to_numpy()-r.rest.to_numpy()))
        for metric,values in b.items():
            lo,hi=ci(values[1:]);rate_rows.append(dict(feature_set=name,metric=metric,
                estimate=values[0],ci_low=lo,ci_high=hi,n_paired_subjects=n,
                averaging="equal participant weights"))
    dec=[]
    for name,reference in comparison_pairs(rates):
        b=rates[name]
        components={"exercise_rate_decrease":rates[reference]["exercise"]-b["exercise"],
                    "rest_false_positive_increase":b["rest"]-rates[reference]["rest"],
                    "excess_reduction":rates[reference]["excess"]-b["excess"]}
        if not np.allclose(components["excess_reduction"],components["exercise_rate_decrease"]+components["rest_false_positive_increase"]):
            raise RuntimeError("Excess decomposition identity failed")
        for metric,values in components.items():
            lo,hi=ci(values[1:]);dec.append(dict(test=name,reference=reference,component=metric,
                estimate=values[0],ci_low=lo,ci_high=hi,n_paired_subjects=n,analysis_status=STATUS))
    pd.DataFrame(rate_rows).to_csv(out/"table_condition_rates.csv",index=False)
    pd.concat(individual).to_csv(out/"participant_condition_rates.csv",index=False)
    pd.DataFrame(dec).to_csv(out/"table_excess_decomposition.csv",index=False)
    # Calibration is descriptive; these bins are never used for target refitting.
    calibration=[]
    for name,d in primary_frames.items():
        bins=np.minimum((d.y_prob.to_numpy()*10).astype(int),9)
        for b in range(10):
            g=d[bins==b]
            if len(g):calibration.append(dict(feature_set=name,bin=b,n=len(g),mean_probability=g.y_prob.mean(),observed_stress_rate=g.y_true.mean()))
    pd.DataFrame(calibration).to_csv(out/"calibration_bins.csv",index=False)
    # No optimized target threshold is selected. Full grid is a labeled diagnostic.
    grid=[]
    for name,d in primary_frames.items():
        for t in np.linspace(.05,.95,19):
            h=d.y_prob>=t;tn,fp,fn,tp=confusion_matrix(d.y_true,h,labels=[0,1]).ravel()
            e=p[(p.feature_set==name)&p.condition.eq("exercise")].copy()
            e["positive"]=(e.y_prob>=t).astype(int)
            grid.append(dict(feature_set=name,threshold=t,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
                exercise_positive_rate_equal_participant=e.groupby("subject_id").positive.mean().mean(),
                macro_f1=f1_score(d.y_true,h,average="macro"),purpose="post hoc operating-curve diagnostic; do not tune on target"))
    pd.DataFrame(grid).to_csv(out/"threshold_sensitivity_descriptive.csv",index=False)
    return pd.DataFrame(operating),pd.DataFrame(rate_rows)

def source_tables(source,splits,out,margin):
    rows=[]
    for (model,fs,s,repeat),g in source.groupby(["model","feature_set","subject_id","repeat"]):
        for policy,column in [("source_developed","y_pred"),("fixed_0.5","y_pred_fixed05")]:
            rows.append(dict(model=model,feature_set=fs,subject_id=s,repeat=repeat,threshold_policy=policy,
                macro_f1=f1_score(g.y_true,g[column],average="macro",zero_division=0),
                balanced_accuracy=balanced_accuracy_score(g.y_true,g[column]),
                auroc=roc_auc_score(g.y_true,g.y_prob) if g.y_true.nunique()==2 else np.nan))
    m=pd.DataFrame(rows)
    m.to_csv(out/"source_subject_repeat_metrics.csv",index=False)
    # First average repeats within participant; repetitions are not extra people.
    s=m.groupby(["model","feature_set","subject_id","threshold_policy"],as_index=False)[["macro_f1","balanced_accuracy","auroc"]].mean()
    table=s.groupby(["model","feature_set","threshold_policy"]).agg(
        macro_f1_mean=("macro_f1","mean"),macro_f1_sd=("macro_f1","std"),
        balanced_accuracy_mean=("balanced_accuracy","mean"),auroc_mean=("auroc","mean"),
        n_subjects=("subject_id","nunique")).reset_index()
    table.to_csv(out/"table_source_feature_comparison.csv",index=False)
    board=table[(table.feature_set=="all_efs")&(table.threshold_policy=="fixed_0.5")].copy()
    board["validation_design"]="repeated participant-grouped CV; fixed-0.5 leaderboard"
    board.to_csv(out/"table_source_leaderboard.csv",index=False)
    a=s[(s.feature_set=="all_efs")&(s.threshold_policy=="fixed_0.5")].pivot(index="subject_id",columns="model",values="macro_f1").dropna()
    friedman=float(stats.friedmanchisquare(*[a[c].to_numpy() for c in a]).pvalue) if a.shape[1]>=3 else np.nan
    q=splits[splits.model=="xgboost"]
    f=q[q.feature_set=="fts6"].merge(q[q.feature_set=="all_efs"],on=["repeat","fold"],suffixes=("_test","_ref"),validate="one_to_one")
    d=(f.macro_f1_test-f.macro_f1_ref).to_numpy();n=len(d)
    ratio=float((f.n_test_subjects_test/f.n_train_subjects_test).mean())
    se=np.sqrt((1/n+ratio)*np.var(d,ddof=1)) if n>1 else np.nan
    mean=float(d.mean());qcrit=stats.t.ppf(.95,n-1)
    if se>0:
        p=max(stats.t.sf((mean+margin)/se,n-1),stats.t.cdf((mean-margin)/se,n-1))
    else:p=np.nan
    pd.DataFrame([dict(mean_difference=mean,corrected_se=se,ci90_low=mean-qcrit*se,
        ci90_high=mean+qcrit*se,equivalence_margin=margin,p_tost_exploratory=p,n_matched_splits=n,
        test_train_subject_ratio=ratio,friedman_p_descriptive=friedman,
        interpretation="Approximate resampling correction; margin not retrospectively preregistered. No automatic equivalence claim.")]).to_csv(out/"table_source_equivalence_exploratory.csv",index=False)

def costs_and_ranking(data_dir,model_dir,out,cfg):
    lock=json.loads((model_dir/"development_lock.json").read_text())
    with np.load(data_dir/"wesad_features_meta.npz",allow_pickle=False) as z:
        cost=dict(zip(z["feature_names"].astype(str),z["cost_cycles"].astype(float)))
    def cycles(features):
        bvp=[x for x in features if x.startswith("bvp_")]
        return sum(cost[x] for x in features if not x.startswith("bvp_"))+(cost["bvp_mean_hr"] if bvp else 0)+sum(cost[x] for x in bvp if x!="bvp_mean_hr")
    reference=cycles(lock["models"]["all_efs"]["features"])
    rows=[]
    for name,spec in lock["models"].items():
        c=cycles(spec["features"])
        rows.append(dict(feature_set=name,n_features=len(spec["features"]),modeled_extraction_cycles=c,
            modeled_extraction_energy_uJ=c/cfg["cpu_hz"]*cfg["voltage_v"]*cfg["active_current_a"]*1e6,
            modeled_cycle_reduction_pct=100*(1-c/reference),scope="analytical feature-extraction model only; not device measurements"))
    pd.DataFrame(rows).to_csv(out/"table_modeled_cost.csv",index=False)
    axes=pd.read_csv(model_dir/"fts_axes.csv")
    a=axes[axes.learner=="xgboost"][["feature","fts"]].rename(columns={"fts":"fts_xgb"})
    b=axes[axes.learner=="random_forest"][["feature","fts"]].rename(columns={"fts":"fts_rf"})
    r=a.merge(b,on="feature",validate="one_to_one")
    r=r[r.feature.isin(lock["models"]["all_efs"]["features"])].copy()
    r["rank_xgb"]=r.fts_xgb.rank(method="average",ascending=False)
    r["rank_rf"]=r.fts_rf.rank(method="average",ascending=False)
    r["spearman_rho_descriptive"]=stats.spearmanr(r.rank_xgb,r.rank_rf).statistic
    r.to_csv(out/"table_rank_concordance.csv",index=False)

def figures(source,p,rates,out):
    fig,ax=plt.subplots(2,3,figsize=(13,8))
    source=source[(source.model=="xgboost")&(source.feature_set=="fts6")&(source["repeat"]==0)]
    target=p[(p.feature_set=="fts6")&p.condition.isin(["rest","stress"])]
    for row,(d,label) in enumerate([(source,"Source grouped OOF: repeat 0"),(target,"Historical target cohort: secondary evaluation")]):
        cm=confusion_matrix(d.y_true,d.y_pred,labels=[0,1]);ax[row,0].imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2):ax[row,0].text(j,i,str(cm[i,j]),ha="center",va="center")
        ax[row,0].set(xticks=[0,1],xticklabels=["rest","stress"],yticks=[0,1],yticklabels=["rest","stress"],xlabel="Predicted",ylabel="Observed",title=label)
        fpr,tpr,_=roc_curve(d.y_true,d.y_prob)
        ax[row,1].plot(fpr,tpr,label=f"AUROC={roc_auc_score(d.y_true,d.y_prob):.3f}")
        ax[row,1].plot([0,1],[0,1],"--",color="gray");ax[row,1].set(xlabel="False-positive rate",ylabel="Sensitivity",xlim=(0,1),ylim=(0,1));ax[row,1].legend()
        precision,recall,_=precision_recall_curve(d.y_true,d.y_prob)
        ax[row,2].step(recall,precision,where="post",label=f"AP={average_precision_score(d.y_true,d.y_prob):.3f}")
        ax[row,2].axhline(d.y_true.mean(),ls="--",color="gray",label="Prevalence")
        ax[row,2].set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02));ax[row,2].legend()
    fig.tight_layout();fig.savefig(out/"figure_diagnostics.png",dpi=180);plt.close(fig)
    names=[x for x in ["all_efs","accuracy6","target_importance6","fts6","fts5","all_efs_no_mean_hr"] if x in set(rates.feature_set)]
    fig,ax=plt.subplots(1,3,figsize=(16,5))
    for a,metric,title in zip(ax,["rest","exercise","excess"],["Rest false-positive rate","Exercise predicted-stress rate","Exercise minus rest"]):
        d=rates[rates.metric==metric].set_index("feature_set").loc[names]
        for i,(_,r) in enumerate(d.iterrows()):
            a.vlines(i,r.ci_low,r.ci_high,color="black");a.plot(i,r.estimate,"o",color="tab:blue")
        a.set(xticks=range(len(names)),xticklabels=[x.replace("_", "\n") for x in names],title=title,ylabel="Participant-averaged rate")
        a.axhline(0,color="gray",lw=.6);a.tick_params(axis="x",rotation=0,labelsize=8)
    fig.suptitle("Report both components: a smaller difference alone is not better detection")
    fig.tight_layout();fig.savefig(out/"figure_tradeoffs.png",dpi=180);plt.close(fig)

def run(data_dir,model_dir,out,cfg):
    data_dir,model_dir,out=map(Path,[data_dir,model_dir,out]);out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(model_dir/"target_predictions.csv");validate_target(p)
    source=pd.read_csv(model_dir/"source_predictions.csv")
    splits=pd.read_csv(model_dir/"split_metrics.csv")
    operating,rates=target_tables(p,cfg,out)
    source_tables(source,splits,out,cfg["equivalence_margin"])
    costs_and_ranking(data_dir,model_dir,out,cfg);figures(source,p,rates,out)
    warnings=[STATUS,"Bootstrap intervals condition on fitted models and the observed participant cohort.",
        "Reported 95% bootstrap intervals are marginal and not multiplicity-adjusted; new contrasts are exploratory.",
        "Target outcomes are laboratory protocol labels, not clinical stress diagnoses.",
        "Source threshold-development OOF scores are not independent performance estimates.",
        "Source outer-fold results nest selection/removal; FTS5 has exactly five features.",
        "An exercise-excess decrease can partly reflect worse rest specificity; report both components.",
        "No new confirmatory p-values or publication-readiness score is generated.",
        "Rank correlation is descriptive; cost/physiology components are shared between learners.",
        "Analytical extraction cost excludes sensing, radio, memory, and model-inference energy.",
        "Runtime unit review cannot establish pulse-detection accuracy during motion or validate task labels."]
    write_json(out/"qa_report.json",dict(hard_failures=[],warnings=warnings,primary_conditions=["rest","stress"],
        target_participants=int(p.subject_id.nunique()),target_windows_per_configuration=int(p[p.feature_set=="fts6"].shape[0])))
    text="FTS v6 SECONDARY REANALYSIS\n\n"+"\n".join("- "+w for w in warnings)
    (out/"interpretation_limits.txt").write_text(text,encoding="utf-8")
    print(operating[operating.scope=="primary_rest_stress"].to_string(index=False))
    print("Tables and figures generated. Results are not an acceptance or deployment verdict.")

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--data",required=True);a.add_argument("--models",required=True)
    a.add_argument("--output",required=True);a.add_argument("--config",required=True)
    x=a.parse_args();run(x.data,x.models,x.output,json.loads(Path(x.config).read_text()))
