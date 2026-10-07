"""Offline tests: no network, no raw dataset, no XGBoost/SHAP substitution claims."""
import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
from statistics_report import metric_batch,sample_counts,validate_target
from models import rank_top
from common import read_npz
from prepare_features import extract_features,save_npz,FEATURE_ORDER

class CoreTests(unittest.TestCase):
    def test_weighted_metrics_and_ties(self):
        d=pd.DataFrame(dict(subject_id=["a","a","b","b","c","c"],
            y_true=[0,1,0,1,0,1],y_prob=[.2,.7,.7,.7,.2,.9],y_pred=[0,1,1,1,0,1]))
        draws=np.vstack([np.ones(3,int),sample_counts(3,100,51)])
        m,_=metric_batch(d,["a","b","c"],draws)
        for k,c in enumerate(draws):
            w=np.repeat(c,2)
            self.assertAlmostEqual(m["auroc"][k],roc_auc_score(d.y_true,d.y_prob,sample_weight=w))
            self.assertAlmostEqual(m["average_precision"][k],average_precision_score(d.y_true,d.y_prob,sample_weight=w))
            self.assertAlmostEqual(m["macro_f1"][k],f1_score(d.y_true,d.y_pred,sample_weight=w,average="macro"))

    def test_identity_pairing(self):
        d=pd.DataFrame(dict(subject_id=["a","a"],feature_set=["fts6"]*2,window_id=["1","2"],
            block_id=["b1","b2"],condition=["rest","stress"],y_true=[0,1],y_prob=[.2,.8],y_pred=[0,1],threshold=[.5,.5]))
        p=pd.concat([d,d.assign(feature_set="fts5")],ignore_index=True)
        validate_target(p)
        p.loc[3,"subject_id"]="wrong"
        with self.assertRaises(ValueError):validate_target(p)

    def test_decomposition(self):
        er,rr,et,rt=.7857142857,.0842261905,.5535714286,.2276785714
        self.assertAlmostEqual((er-rr)-(et-rt),(er-et)+(rt-rr))
        self.assertAlmostEqual((rt-rr)/((er-rr)-(et-rt)),.38193343899,places=7)

    def test_top_set_size_ties(self):
        self.assertEqual(rank_top(np.ones(8),range(8),6),list(range(6)))
        f=rank_top(np.ones(8),range(8),6)
        self.assertEqual(len([i for i in f if i!=3]),5)

    def test_synthetic_pulse_and_unicode_npz(self):
        t=np.arange(60*64)/64
        w=np.column_stack([np.sin(2*np.pi*1.25*t),2+.002*t,33-.001*t,np.full(len(t),64.)])
        f=extract_features(w,64)
        self.assertAlmostEqual(float(f[0]),75,delta=1)
        row=dict(zip(FEATURE_ORDER,f));row.update(subject_id="S2",block_id="b",condition="stress",protocol="WESAD",window_id="w",window_start_s=0.,window_end_s=60.)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"sample.npz";save_npz(pd.DataFrame([row]),path,True)
            a=read_npz(path)
            self.assertTrue(all(v.dtype.kind!="O" for v in a.values()))

if __name__=="__main__":unittest.main()
