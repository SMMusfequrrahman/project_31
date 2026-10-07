"""FTS v6. Explicit source/target partitions; no confirmation-based tuning.

The historical cohort has already been inspected. All v6 target results are
retrospective secondary results, not a new independent confirmation experiment.
FTS axes retain v4's 23-feature scoring model and 16-feature eligible pool.
Thresholds use source-only LOGO predictions conditional on the selected columns;
those threshold-development predictions are NOT unbiased performance estimates.
Outer performance evaluation nests selection, removal, imputation and thresholds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import VERSION, STATUS, NPZ_FILES, read_npz, sha256, write_json, runtime_versions

DIRECTION = {"bvp_mean_hr":1, "bvp_sdnn":-1, "bvp_rmssd":-1, "bvp_pnn50":-1,
 "eda_mean":1,"eda_std":1,"eda_slope":1,"eda_min":1,"eda_max":1,"eda_range":1,
 "temp_mean":-1,"temp_std":0,"temp_slope":-1,"acc_mean":0,"acc_std":0,"acc_max":0,
 "bvp_lf":0,"bvp_hf":-1,"bvp_lf_hf":1,"eda_scl_mean":1,"eda_scr_count":1,
 "eda_scr_mean_amp":1,"eda_scr_auc":1}

def impute(train, *test):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    return (imp.fit_transform(train), *(imp.transform(x) for x in test))

def rank_top(score, eligible, n=6):
    # Stable, explicit index tie-break; never randomly break FTS zero ties.
    return sorted(map(int, eligible), key=lambda i: (-float(score[i]), i))[:n]

def canonical_columns(columns):
    """A feature set is unordered: preserve the registry order for every fit."""
    cols=list(map(int,columns))
    if len(cols)!=len(set(cols)):raise ValueError("Duplicate selected feature")
    return sorted(cols)

class Engine:
    def __init__(self, cfg, data_dir):
        self.cfg, self.data_dir = cfg, Path(data_dir)
        self.W, self.P, self.E = [read_npz(self.data_dir/f) for f in NPZ_FILES]
        self.names = self.W["feature_names"].astype(str).tolist()
        for a in (self.P,self.E):
            if a["feature_names"].astype(str).tolist()!=self.names: raise ValueError("Feature order differs")
        self.X = self.W["X"].astype(float)
        self.y = self.W["y"].astype(int)
        self.g = self.W["groups"].astype(str)
        self.efs = np.flatnonzero(self.W["tier"].astype(str)=="EFS")
        self.cost = self.W["cost_cycles"].astype(float)
        self.seed=int(cfg["seed"])
        self.disc=list(cfg["discovery_subjects"])
        self.conf=list(cfg["confirmation_subjects"])
        if set(self.disc)&set(self.conf): raise ValueError("Target participant overlap")
        matched=set(self.P["groups"].astype(str))&set(self.E["groups"].astype(str))
        if set(self.disc)|set(self.conf)!=matched: raise ValueError("Locked target split does not match raw matched cohort")
        self.pd = np.isin(self.P["groups"],self.disc)&np.isin(self.P["conditions"],["rest","stress"])
        self.ed = np.isin(self.E["groups"],self.disc)
        if set(self.P["y"][self.pd])!={0,1}: raise ValueError("Discovery requires rest and stress")
        self.threshold_cache={}
        self.audit=[]
        self.ablations=[]
        for a in (self.W,self.P,self.E):
            if np.isinf(a["X"]).any(): raise ValueError("Infinite features: audit extraction")
        if not np.isfinite(self.X[:,self.efs]).any(axis=0).all():
            raise ValueError("A source EFS feature has no finite value")

    def learner(self, kind, seed):
        if kind=="xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=self.cfg.get("xgb_trees",200), max_depth=4,
                learning_rate=.1,subsample=.9,colsample_bytree=1.,eval_metric="logloss",
                tree_method="hist",random_state=seed,n_jobs=1)
        if kind=="random_forest":
            return RandomForestClassifier(n_estimators=self.cfg.get("rf_trees",500),random_state=seed,n_jobs=1)
        raise ValueError(kind)

    def factory(self, kind, seed):
        if kind in ("xgboost","random_forest"): return self.learner(kind,seed)
        if kind=="logistic_regression": return make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=seed))
        if kind=="linear_svm": return make_pipeline(StandardScaler(),SVC(kernel="linear",probability=True,random_state=seed))
        if kind=="knn": return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3))
        if kind=="naive_bayes": return GaussianNB()
        raise ValueError(kind)

    def importance(self, model, x):
        import shap
        v=shap.TreeExplainer(model).shap_values(x)
        if isinstance(v,list): v=v[-1]
        v=np.asarray(v)
        if v.ndim==3: v=v[:,:,-1]
        if v.shape!=x.shape: raise ValueError(f"SHAP shape {v.shape}, expected {x.shape}")
        a=np.mean(np.abs(v),axis=0)
        return a/(a.sum()+1e-12),v

    def axes(self, ix, kind, seed):
        xs,ys,gs=self.X[ix],self.y[ix],self.g[ix]
        if len(np.unique(gs))<4: raise ValueError("Need at least four source training participants")
        k=min(8,len(self.names)); d=len(self.names)
        counts=[]; px=[]; sv=[]
        for f,(tr,te) in enumerate(LeaveOneGroupOut().split(xs,ys,gs)):
            a,b=impute(xs[tr],xs[te])
            model=self.learner(kind,seed+f).fit(a,ys[tr])
            importance,v=self.importance(model,b)
            hit=np.zeros(d)
            hit[rank_top(importance,range(d),k)]+=.5
            hit[rank_top(model.feature_importances_,range(d),k)]+=.5
            counts.append(hit);px.append(b);sv.append(v)
        xs_i,xt_i=impute(xs,self.P["X"][self.pd])
        sm=self.learner(kind,seed).fit(xs_i,ys)
        tm=self.learner(kind,seed+1000).fit(xt_i,self.P["y"][self.pd])
        si,_=self.importance(sm,xs_i);ti,_=self.importance(tm,xt_i)
        transfer=np.minimum(si,ti)/(np.maximum(si,ti)+1e-12)
        transfer*=np.where((si>=1/d)&(ti>=1/d),1.,.2)
        xx,vv=np.vstack(px),np.vstack(sv)
        physiology=np.full(d,.5)
        for i,name in enumerate(self.names):
            expected=DIRECTION.get(name,0)
            if expected and np.std(xx[:,i])>0 and np.std(vv[:,i])>0:
                correlation=np.corrcoef(xx[:,i],vv[:,i])[0,1]
                if np.isfinite(correlation):
                    physiology[i]=1. if np.sign(correlation)==expected else (.5 if correlation==0 else .1)
        cost_score=1-np.clip(self.cost/200000.,0,1)
        ax=pd.DataFrame(dict(feature=self.names,learner=kind,stability=np.mean(counts,axis=0),
            transfer=transfer,physiology=physiology,cost_score=cost_score,
            source_importance=si,target_discovery_importance=ti))
        ax["fts"]=(ax[["stability","transfer","physiology","cost_score"]].prod(axis=1))**.25
        ax["rank"]=ax.fts.rank(method="min",ascending=False).astype(int)
        return ax

    def threshold(self, ix, columns, seed, kind="xgboost"):
        columns=canonical_columns(columns)
        key=(tuple(map(int,ix)),tuple(columns),int(seed),kind)
        if key in self.threshold_cache: return self.threshold_cache[key]
        xs,ys,gs=self.X[ix][:,columns],self.y[ix],self.g[ix]
        probability=np.full(len(ix),np.nan)
        for f,(tr,te) in enumerate(LeaveOneGroupOut().split(xs,ys,gs)):
            a,b=impute(xs[tr],xs[te])
            m=self.factory(kind,seed+f).fit(a,ys[tr])
            probability[te]=m.predict_proba(b)[:,1]
        if not np.isfinite(probability).all(): raise ValueError("Incomplete source threshold OOF")
        grid=np.linspace(.05,.95,181)
        score=[f1_score(ys,probability>=t,average="macro",zero_division=0) for t in grid]
        t=float(grid[int(np.argmax(score))])
        self.threshold_cache[key]=t
        return t

    def fit(self, ix, columns, seed, model_name="xgboost"):
        columns=canonical_columns(columns)
        pipe=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),self.factory(model_name,seed))
        return pipe.fit(self.X[ix][:,columns],self.y[ix])

    def discovery_excess(self, ix, columns, seed):
        columns=canonical_columns(columns)
        # Identical negative condition (rest only) in discovery and evaluation.
        rest=self.pd&(self.P["conditions"]=="rest")
        model=self.fit(ix,columns,seed)
        threshold=self.threshold(ix,columns,seed)
        r=pd.DataFrame({"s":self.P["groups"][rest],"p":model.predict_proba(self.P["X"][rest][:,columns])[:,1]>=threshold}).groupby("s").p.mean()
        e=pd.DataFrame({"s":self.E["groups"][self.ed],"p":model.predict_proba(self.E["X"][self.ed][:,columns])[:,1]>=threshold}).groupby("s").p.mean()
        if set(r.index)!=set(self.disc) or set(e.index)!=set(self.disc): raise ValueError("Incomplete paired discovery subjects")
        return float((e-r).mean())

    def select(self, ix, seed, context):
        ax=self.axes(ix,"xgboost",seed)
        f6=rank_top(ax.fts.to_numpy(),self.efs)
        base=self.discovery_excess(ix,f6,seed)
        candidates=[]
        for removed in f6:
            cols=[i for i in f6 if i!=removed]
            excess=self.discovery_excess(ix,cols,seed)
            candidates.append((excess,removed,cols))
            self.ablations.append(dict(context=context,removed_feature=self.names[removed],
                baseline_excess=base,candidate_excess=excess,excess_reduction=base-excess,
                negative_condition="rest",target_partition="discovery"))
        _,removed,f5=min(candidates,key=lambda z:(z[0],z[1]))
        if len(f5)!=5: raise ValueError("FTS5 must have exactly five features")
        a=SimpleImputer(strategy="median",keep_empty_features=True).fit_transform(self.X[ix][:,self.efs])
        m=self.learner("xgboost",seed).fit(a,self.y[ix])
        score=np.zeros(len(self.names)); score[self.efs]=m.feature_importances_
        sets={"all_efs":list(map(int,self.efs)),"accuracy6":rank_top(score,self.efs),"fts6":f6,"fts5":f5}
        # Same labeled target-discovery access as the FTS transfer axis; final
        # classifier coefficients still come exclusively from source training.
        sets["target_importance6"]=rank_top(ax.target_discovery_importance.to_numpy(),self.efs)
        if self.cfg["run_ablations"]:
            # Simple heuristic comparator, NOT a reproduction of CAFS.
            sets["cost_accuracy6"]=rank_top(score/np.maximum(self.cost,1),self.efs)
            sets["all_efs_no_mean_hr"]=[i for i in self.efs if self.names[i]!="bvp_mean_hr"]
            components=["stability","transfer","physiology","cost_score"]
            for component in components:
                s=ax[[x for x in components if x!=component]].prod(axis=1).to_numpy()**(1/3)
                sets["fts_without_"+component+"6"]=rank_top(s,self.efs)
        return {k:canonical_columns(v) for k,v in sets.items()},ax,self.names[removed]

    def prediction_rows(self, a, mask, model, cols, threshold, name, context, repeat, fold, model_name="xgboost"):
        cols=canonical_columns(cols)
        ix=np.flatnonzero(mask) if np.asarray(mask).dtype==bool else np.asarray(mask)
        prob=model.predict_proba(a["X"][ix][:,cols])[:,1]
        truth=a["y"][ix] if "y" in a else np.zeros(len(ix),dtype=int)
        return pd.DataFrame({"dataset":"wesad" if a is self.W else "physionet",
            "split":context,"subject_id":a["groups"][ix],"window_id":a["window_ids"][ix],
            "block_id":a["block_ids"][ix],"condition":a["conditions"][ix],
            "window_start_s":a["window_start_s"][ix],"window_end_s":a["window_end_s"][ix],
            "y_true":truth,"y_prob":prob,"y_pred":(prob>=threshold).astype(int),
            "threshold":threshold,"model":model_name,"feature_set":name,"n_features":len(cols),
            "repeat":repeat,"fold":fold,"sampling_hz":64})

    def develop(self,out):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        predictions=[];scores=[]; selections=[]
        model_names=["xgboost","random_forest","logistic_regression","linear_svm","knn","naive_bayes"]
        # Content-addressed per-fold recovery after a Colab disconnect.
        import hashlib
        signature=hashlib.sha256(json.dumps(dict(cfg=self.cfg,runtime=runtime_versions(),code={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            inputs={f:sha256(self.data_dir/f) for f in NPZ_FILES}),sort_keys=True).encode()).hexdigest()
        cache=out/"fold_checkpoints";cache.mkdir(exist_ok=True)
        for repeat in range(self.cfg["cv_repeats"]):
            splitter=StratifiedGroupKFold(n_splits=self.cfg["cv_folds"],shuffle=True,random_state=self.seed+repeat)
            for fold,(tr,te) in enumerate(splitter.split(self.X,self.y,self.g)):
                seed=self.seed+10000*repeat+fold
                context=f"repeat{repeat}_fold{fold}"
                folder=cache/context;folder.mkdir(exist_ok=True)
                done=folder/"complete.json"
                if done.exists():
                    old=json.loads(done.read_text())
                    if old["signature"]!=signature:raise ValueError("Stale fold checkpoint: start a new v6 run")
                    for f,h in old["files"].items():
                        if not (folder/f).exists() or sha256(folder/f)!=h:raise ValueError("Changed/incomplete fold checkpoint")
                    predictions.append(pd.read_csv(folder/"predictions.csv"))
                    scores.extend(json.loads((folder/"scores.json").read_text()))
                    selections.extend(json.loads((folder/"selections.json").read_text()))
                    self.ablations.extend(json.loads((folder/"removals.json").read_text()))
                    print(context+": verified checkpoint reused",flush=True)
                    continue
                start_p,start_s,start_f,start_a=len(predictions),len(scores),len(selections),len(self.ablations)
                if set(self.g[tr])&set(self.g[te]):raise RuntimeError("Source participant leakage")
                print(f"{context}: nested selection/removal, {len(set(self.g[tr]))} training subjects",flush=True)
                sets,ax,removed=self.select(tr,seed,context)
                for kind in model_names:
                    for name,cols in (sets.items() if kind=="xgboost" else [("all_efs",sets["all_efs"])]):
                        threshold=self.threshold(tr,cols,seed,kind) if kind in ("xgboost","logistic_regression") else .5
                        # Also export fixed-0.5 predictions for a fair six-learner leaderboard.
                        model=self.fit(tr,cols,seed,kind)
                        pred=self.prediction_rows(self.W,te,model,cols,threshold,name,"grouped_oof",repeat,fold,kind)
                        pred["y_pred_fixed05"]=(pred.y_prob>=.5).astype(int)
                        predictions.append(pred)
                        scores.append(dict(model=kind,feature_set=name,repeat=repeat,fold=fold,
                            macro_f1=f1_score(pred.y_true,pred.y_pred,average="macro",zero_division=0),
                            macro_f1_fixed05=f1_score(pred.y_true,pred.y_pred_fixed05,average="macro",zero_division=0),
                            n_train_subjects=len(set(self.g[tr])),n_test_subjects=len(set(self.g[te]))))
                        selections.append(dict(context=context,model=kind,feature_set=name,n_features=len(cols),
                            features=[self.names[i] for i in cols],removed_feature=removed if name=="fts5" else None,
                            threshold=threshold,source_train_subjects=sorted(set(self.g[tr])),
                            source_test_subjects=sorted(set(self.g[te])),target_development_subjects=self.disc))
                pd.concat(predictions[start_p:],ignore_index=True).to_csv(folder/"predictions.csv",index=False)
                write_json(folder/"scores.json",scores[start_s:])
                write_json(folder/"selections.json",selections[start_f:])
                write_json(folder/"removals.json",self.ablations[start_a:])
                files=["predictions.csv","scores.json","selections.json","removals.json"]
                write_json(done,dict(signature=signature,files={f:sha256(folder/f) for f in files}))
        pd.concat(predictions,ignore_index=True).to_csv(out/"source_predictions.csv",index=False)
        pd.DataFrame(scores).to_csv(out/"split_metrics.csv",index=False)
        ix=np.arange(len(self.X))
        sets,ax,removed=self.select(ix,self.seed,"final_development")
        axrf=self.axes(ix,"random_forest",self.seed)
        pd.concat([ax,axrf],ignore_index=True).to_csv(out/"fts_axes.csv",index=False)
        final={}
        sets["logistic_all_efs"]=sets["all_efs"]
        for name,cols in sets.items():
            kind="logistic_regression" if name=="logistic_all_efs" else "xgboost"
            model=self.fit(ix,cols,self.seed,kind)
            threshold=self.threshold(ix,cols,self.seed,kind)
            path=out/f"model_{name}.joblib"
            joblib.dump(model,path)
            final[name]=dict(features=[self.names[i] for i in cols],columns=list(map(int,cols)),
                n_features=len(cols),threshold=threshold,model_file=path.name,sha256=sha256(path),learner=kind,
                target_label_access="discovery rest/stress" if (name.startswith("fts") and name!="fts_without_transfer6") or name=="target_importance6" else "none",
                exercise_selection_access="discovery exercise and rest" if name=="fts5" else "none")
        write_json(out/"fold_selections.json",selections)
        pd.DataFrame(self.ablations).to_csv(out/"discovery_removal_candidates.csv",index=False)
        write_json(out/"development_lock.json",dict(version=VERSION,analysis_status=STATUS,runtime=runtime_versions(),
            configuration=self.cfg,configuration_note="Thresholds are source-only development estimates, conditional on selected features.",
            removed_feature=removed,models=final,feature_schema=self.names,
            inputs={f:sha256(self.data_dir/f) for f in NPZ_FILES},
            models_script_sha256=sha256(Path(__file__)),
            source_subjects=sorted(set(self.g)),discovery_subjects=self.disc,confirmation_subjects=self.conf))
        print("Development complete. Selected models and thresholds locked before evaluation.",flush=True)

    def evaluate(self,out):
        out=Path(out)
        lock=json.loads((out/"development_lock.json").read_text())
        if lock["configuration"]!=self.cfg:raise ValueError("Configuration changed after development")
        if lock["runtime"]!=runtime_versions():raise ValueError("Runtime changed after development")
        if lock["models_script_sha256"]!=sha256(Path(__file__)):raise ValueError("Model code changed after development")
        for f,h in lock["inputs"].items():
            if sha256(self.data_dir/f)!=h: raise ValueError("Features changed after development")
        predictions=[]
        for name,spec in lock["models"].items():
            path=out/spec["model_file"]
            if sha256(path)!=spec["sha256"]: raise ValueError("Model artifact changed")
            # Only deserialize artifacts generated and hashed by this run.
            m=joblib.load(path)
            for a in (self.P,self.E):
                mask=np.isin(a["groups"],self.conf)
                predictions.append(self.prediction_rows(a,mask,m,spec["columns"],spec["threshold"],name,
                    "historical_confirmation_reanalysis",0,0,spec.get("learner","xgboost")))
        p=pd.concat(predictions,ignore_index=True)
        if p.duplicated(["feature_set","window_id"]).any():raise ValueError("Duplicate target predictions")
        # Equal memberships, seeds and estimators must give equal predictions.
        equivalences=[]
        names=list(lock["models"])
        for i,left in enumerate(names):
            for right in names[i+1:]:
                a,b=lock["models"][left],lock["models"][right]
                if a["columns"]==b["columns"] and a["learner"]==b["learner"]:
                    x=p[p.feature_set==left].sort_values("window_id")
                    y=p[p.feature_set==right].sort_values("window_id")
                    if a["threshold"]!=b["threshold"] or not np.array_equal(x.y_prob.to_numpy(),y.y_prob.to_numpy()):
                        raise RuntimeError("Equivalent feature sets produced unequal predictions")
                    equivalences.append([left,right])
        p.to_csv(out/"target_predictions.csv",index=False)
        write_json(out/"target_evaluation.json",dict(status=STATUS,n_participants=len(self.conf),
            no_target_threshold_tuning=True,primary_conditions=["rest","stress"],baseline="sensitivity only",
            equal_membership_checks=equivalences))
        print("Target evaluation complete; no target refitting or threshold optimization.",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["develop","evaluate"])
    parser.add_argument("--config",required=True);parser.add_argument("--data",required=True)
    parser.add_argument("--output",required=True)
    a=parser.parse_args();cfg=json.loads(Path(a.config).read_text())
    e=Engine(cfg,a.data)
    getattr(e,a.stage)(a.output)

if __name__=="__main__": main()
