from pathlib import Path
import json
import tempfile
import unittest
import shutil
import pandas as pd
from prepare_features import validate_manifest
import complete_workflow as C

class CompleteTest(unittest.TestCase):
    def test_evidence_mapping_does_not_fabricate_review(self):
        assets=Path(__file__).resolve().parents[1]/'assets'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shutil.copytree(assets,root/'assets');(root/'data').mkdir();(root/'results').mkdir()
            C.manifest(root)
            m=pd.read_csv(root/'data/physionet_interval_manifest.csv');used=validate_manifest(m)
            self.assertEqual(len(used),197)
            self.assertEqual(set(used.condition),{'rest','stress','exercise'})
            self.assertTrue(used.reviewed.eq(0).all())
            proof=json.loads((root/'label_evidence.json').read_text())
            self.assertFalse(proof['human_review_attested'])
            bad=m.copy();i=bad.index[bad.include==1][0];bad.loc[i,'start_tag_index']+=1
            with self.assertRaises(ValueError):validate_manifest(bad)
            bad=m.copy();bad.loc[i,'condition']='stress'
            with self.assertRaises(ValueError):validate_manifest(bad)

    def test_archive_refuses_incomplete_workflow(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):C.archive(Path(d))

if __name__=='__main__':unittest.main()
