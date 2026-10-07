"""Colab-local regression and actual XGBoost/SHAP integration tests."""
from pathlib import Path
import argparse
import json
import sys
import unittest
from smoke_test import main as smoke

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    suite=unittest.defaultTestLoader.discover(str(r/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise SystemExit('Software tests failed or skipped; details above')
    smoke(r/'synthetic_software_tests')
    (r/'software_test_report.json').write_text(json.dumps(dict(status='passed',unit_tests=result.testsRun,
        skipped=len(result.skipped),xgboost_shap_integration='passed',data_kind='synthetic only',
        interpretation='Software behavior checked; real signal validity and publication suitability are not established'),indent=2))
