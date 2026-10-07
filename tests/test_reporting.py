"""Synthetic reporting integration, independent of model-training dependencies."""
import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
import pandas as pd
from statistics_report import run
from prepare_features import FEATURE_ORDER,REGISTRY

class ReportingTest(unittest.TestCase):
    def test_all_tables_and_figures(self):
        rng=np.random.default_rng(31)
        names=FEATURE_ORDER
        sets={"all_efs":names[:16],"accuracy6":names[:6],"fts6":[names[i] for i in [0,4,5,6,10,12]],"fts5":[names[i] for i in [4,5,6,10,12]]}
        rows=[]
        for fs in sets:
            for s in range(8):
                for c in ["baseline","rest","stress","exercise"]:
                    for j in range(3):
                        y=int(c=="stress");p=float(np.clip(.15+.55*y+rng.normal(0,.25),.01,.99))
                        rows.append(dict(feature_set=fs,subject_id=f"T{s}",window_id=f"T{s}:{c}:{j}",
                            block_id=f"T{s}:{c}",condition=c,y_true=y,y_prob=p,y_pred=int(p>=.5),threshold=.5))
        target=pd.DataFrame(rows)
        source=[];splits=[]
        for repeat in range(2):
            for model in ["xgboost","random_forest","logistic_regression","linear_svm","knn","naive_bayes"]:
                for fs in (sets if model=="xgboost" else ["all_efs"]):
                    for s in range(6):
                        for c in ["rest","stress"]:
                            for j in range(3):
                                y=int(c=="stress");p=float(np.clip(.2+.5*y+rng.normal(0,.3),.01,.99))
                                source.append(dict(model=model,feature_set=fs,subject_id=f"W{s}",repeat=repeat,fold=s%3,
                                    y_true=y,y_prob=p,y_pred=int(p>=.5),y_pred_fixed05=int(p>=.5)))
                    for fold in range(3):splits.append(dict(model=model,feature_set=fs,repeat=repeat,fold=fold,
                        macro_f1=.8+rng.normal(0,.04),macro_f1_fixed05=.8,n_train_subjects=4,n_test_subjects=2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";models=root/"models";out=root/"results";data.mkdir();models.mkdir()
            target.to_csv(models/"target_predictions.csv",index=False)
            pd.DataFrame(source).to_csv(models/"source_predictions.csv",index=False)
            pd.DataFrame(splits).to_csv(models/"split_metrics.csv",index=False)
            axes=[]
            for learner in ["xgboost","random_forest"]:
                for f in names:axes.append(dict(feature=f,learner=learner,fts=rng.random()))
            pd.DataFrame(axes).to_csv(models/"fts_axes.csv",index=False)
            (models/"development_lock.json").write_text(json.dumps(dict(models={k:dict(features=v) for k,v in sets.items()})))
            np.savez(data/"wesad_features_meta.npz",feature_names=np.asarray(names),cost_cycles=np.asarray([REGISTRY[f][1] for f in names]))
            cfg=dict(n_bootstrap=100,bootstrap_seed=15,equivalence_margin=.02,cpu_hz=64000000,voltage_v=3.,active_current_a=.00333)
            run(data,models,out,cfg)
            self.assertEqual(len(list(out.iterdir())),20)
            self.assertGreater((out/"figure_tradeoffs.png").stat().st_size,10000)
            self.assertEqual(json.loads((out/"qa_report.json").read_text())["hard_failures"],[])
            table=pd.read_csv(out/"table_target_operating.csv")
            self.assertEqual(len(table),4)
            self.assertEqual(set(table.scope),{"primary_rest_stress"})
            pdiff=pd.read_csv(out/"table_target_paired_differences.csv")
            self.assertTrue(set(pdiff.reference)=={"fts6","all_efs"})

if __name__=="__main__":unittest.main()
