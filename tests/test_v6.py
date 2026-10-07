"""Regression checks for the added protocol controls."""
import sys,tempfile,json,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
import numpy as np
from models import canonical_columns
from explain import background_indices
from statistics_report import comparison_pairs
from common import sha256,write_json,NPZ_FILES
from smoke_test import make_data
from reuse_features import reuse,FILES

class V6Test(unittest.TestCase):
    def test_duplicate_features_rejected(self):
        self.assertEqual(canonical_columns([5,1,3]),[1,3,5])
        with self.assertRaises(ValueError):canonical_columns([1,1])

    def test_background_balance_and_reproducibility(self):
        groups=np.repeat(["A","B","C"],[30,50,100])
        a=background_indices(groups,90,3)
        self.assertTrue(np.array_equal(a,background_indices(groups,90,3)))
        self.assertEqual(len(set(a)),90)
        self.assertEqual(np.unique(groups[a],return_counts=True)[1].tolist(),[30,30,30])
        with self.assertRaises(ValueError):background_indices(groups,101,3)

    def test_direct_ablation_contrast(self):
        p=comparison_pairs(["all_efs","fts6","fts5","all_efs_no_mean_hr","target_importance6"])
        self.assertIn(("all_efs_no_mean_hr","all_efs"),p)
        self.assertIn(("target_importance6","fts6"),p)
        self.assertEqual(len(p),len(set(p)))

    def test_feature_reuse_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/"old";new=root/"new/data"
            (old/"code").mkdir(parents=True);(old/"data").mkdir();new.mkdir(parents=True)
            make_data(old/"data")
            for f in ["common.py","prepare_features.py"]:(old/"code"/f).write_text("# fixture")
            expected={p.name:sha256(p) for p in (old/"code").iterdir()}
            for p in [old/"data/physionet_interval_manifest.csv",new/"physionet_interval_manifest.csv"]:p.write_text("fixture manifest")
            for f in ["subject_flow.csv","window_flow.csv","window_index.csv"]:(old/"data"/f).write_text("fixture\n")
            raw=root/"raw.txt";raw.write_text("trusted fixture")
            write_json(old/"data/raw_input_hashes.json",{str(raw):dict(sha256=sha256(raw))})
            write_json(old/"provenance.json",dict(code_version="5.0.0",feature_hashes={f:sha256(old/"data"/f) for f in NPZ_FILES}))
            write_json(old/"stage_extract.json",dict(outputs={str(old/"data"/f):sha256(old/"data"/f) for f in FILES}))
            reuse(old,new,expected)
            self.assertEqual(sha256(old/"data"/NPZ_FILES[0]),sha256(new/NPZ_FILES[0]))
            self.assertFalse((new.parent/"models").exists())
            raw.write_text("modified")
            with self.assertRaises(ValueError):reuse(old,new,expected)

if __name__=="__main__":unittest.main()
