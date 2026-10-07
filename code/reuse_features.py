"""Copy verified v5 extraction artifacts into a separate v6 run.

No fitted model, target prediction, fold checkpoint or result is reused.
The original NPZ format_version is preserved and recorded in provenance.
"""
import argparse,json,shutil
from pathlib import Path
from common import NPZ_FILES,sha256,read_npz,write_json

FILES=list(NPZ_FILES)+["subject_flow.csv","window_flow.csv","window_index.csv","raw_input_hashes.json"]

def reuse(old,out,expected):
    old,out=Path(old),Path(out)
    if old.resolve()==out.parent.resolve():raise ValueError("Reuse requires a separate run")
    proof=json.loads((old/"provenance.json").read_text())
    if str(proof["code_version"])!="5.0.0":raise ValueError("Only the supplied v5 extraction is compatible")
    for name,digest in expected.items():
        if sha256(old/"code"/name)!=digest:raise ValueError("Prior extractor differs from supplied v5: "+name)
    if sha256(old/"data/physionet_interval_manifest.csv")!=sha256(out/"physionet_interval_manifest.csv"):
        raise ValueError("Reviewed manifests differ; perform full extraction")
    stage=json.loads((old/"stage_extract.json").read_text())
    for path,digest in stage["outputs"].items():
        if sha256(path)!=digest:raise ValueError("Changed prior extraction output: "+path)
    for name in FILES:
        path=old/"data"/name
        if str(path) not in stage["outputs"]:raise ValueError("Prior extraction did not record "+name)
        if name in NPZ_FILES:
            read_npz(path)
            if sha256(path)!=proof["feature_hashes"][name]:raise ValueError("Prior feature provenance mismatch")
    for path,info in json.loads((old/"data/raw_input_hashes.json").read_text()).items():
        if sha256(path)!=info["sha256"]:raise ValueError("Raw input differs from verified extraction: "+path)
    out.mkdir(parents=True,exist_ok=True)
    for name in FILES:
        source,target=old/"data"/name,out/name
        if target.exists() and sha256(source)!=sha256(target):raise ValueError("Refusing to overwrite different feature file")
        if not target.exists():shutil.copyfile(source,target)
    write_json(out/"feature_reuse_provenance.json",dict(source_run=str(old),
        prior_provenance_sha256=sha256(old/"provenance.json"),extractor_hashes=expected,
        copied_files={f:sha256(out/f) for f in FILES},
        note="Unmodified v5 feature schema and signal processing reused; all v6 model and result stages are new."))
    print("Verified extraction reused. All fitted models will be developed in the new run.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--old",required=True);p.add_argument("--output",required=True)
    p.add_argument("--expected",required=True);a=p.parse_args()
    reuse(a.old,a.output,json.loads(Path(a.expected).read_text()))
