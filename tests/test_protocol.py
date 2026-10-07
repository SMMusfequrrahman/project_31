"""Partition/control-flow tests with an explicitly declared estimator test double.

These do NOT establish actual XGBoost or SHAP compatibility. That is the role
of code/smoke_test.py, which must pass in the actual Colab environment.
"""
import sys,tempfile,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models import Engine
from smoke_test import make_data

class RoutingEngine(Engine):
    def learner(self,kind,seed):
        return RandomForestClassifier(n_estimators=3,max_depth=3,random_state=seed,n_jobs=1)
    def importance(self,model,x):
        imp=model.feature_importances_.astype(float)
        return imp/(imp.sum()+1e-12),(x-x.mean(axis=0))*imp

class ProtocolTest(unittest.TestCase):
    def test_heldout_invariance_and_evaluation_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";data.mkdir();out=root/"models"
            cfg=make_data(data);cfg["run_ablations"]=False
            e=RoutingEngine(cfg,data);ix=np.flatnonzero(e.g!="W5")
            first,_,_=e.select(ix,77,"original")
            e.X[e.g=="W5"]+=1000
            mask=np.isin(e.P["groups"],e.conf)
            e.P["X"][mask]+=1000;e.P["y"][mask]=1-e.P["y"][mask]
            e.E["X"][np.isin(e.E["groups"],e.conf)]+=1000
            second,_,_=e.select(ix,77,"perturbed")
            self.assertEqual(first,second)
            self.assertEqual(len(first["fts5"]),5)
            e=RoutingEngine(cfg,data);e.develop(out);e.evaluate(out)
            for row in json.loads((out/"fold_selections.json").read_text()):
                self.assertFalse(set(row["source_train_subjects"])&set(row["source_test_subjects"]))
                self.assertEqual(set(row["target_development_subjects"]),set(cfg["discovery_subjects"]))
                if row["feature_set"]=="fts5":self.assertEqual(row["n_features"],5)
            # The completed development can resume verified fold checkpoints.
            RoutingEngine(cfg,data).develop(out)
            lock=json.loads((out/"development_lock.json").read_text())
            path=out/lock["models"]["fts6"]["model_file"]
            path.write_bytes(b"corrupted-test-model")
            with self.assertRaises(ValueError):RoutingEngine(cfg,data).evaluate(out)

if __name__=="__main__":unittest.main()
