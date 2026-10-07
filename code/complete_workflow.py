"""Complete Colab workflow: evidence-based labels, raw audit, fixed quality sensitivity.

Workflow 7.0.0 uses the v6.0.0 feature/model schema. No human review is fabricated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import zipfile
import numpy as np
import pandas as pd
import recovery as R

WORKFLOW_VERSION='7.0.0'
NPZS=R.NPZS
OFFICIAL_URL='https://physionet.org/files/wearable-device-dataset/1.0.1/Wearable_Dataset.ipynb'


def immutable(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()
    if p.exists() and p.read_bytes()!=b:raise ValueError('Recorded artifact changed: '+str(p))
    if not p.exists():p.write_bytes(b)


def manifest(root):
    from prepare_features import validate_manifest
    a=root/'assets/historical_manifest.csv'
    d=pd.read_csv(a)
    before=int(d.include.sum())
    d.loc[d.condition=='baseline','include']=0
    # Preserve the historical exclusion decisions; discard unused inferred task names.
    d.loc[d.include==0,['condition','source_block']]=['exclude','not analysed']
    d['reviewed']=0  # This field is NOT a human attestation.
    d['auto_label_status']=np.where(d.include==1,'official_mapping_checked_by_code','not_analysed')
    chosen=validate_manifest(d)
    for r in chosen.itertuples():
        if not np.isfinite([r.analysis_start_s,r.analysis_end_s,r.start_s,r.end_s]).all():
            raise ValueError('Nonfinite manifest times')
        if not r.start_s <= r.analysis_start_s < r.analysis_end_s <= r.end_s:
            raise ValueError('Analysis interval outside raw tag interval')
        if r.analysis_end_s-r.analysis_start_s<60:raise ValueError('Short included interval')
    target=root/'data/physionet_interval_manifest.csv';b=d.to_csv(index=False).encode()
    if target.exists() and target.read_bytes()!=b:raise ValueError('Manifest changed; use new workflow run')
    if not target.exists():target.write_bytes(b)
    mappings=chosen.groupby(['protocol','protocol_version','condition','source_block','start_tag_index','end_tag_index']).size().reset_index(name='n_intervals')
    mappings.to_csv(root/'results/official_mapping_checks.csv',index=False)
    immutable(root/'label_evidence.json',dict(workflow_version=WORKFLOW_VERSION,official_notebook=OFFICIAL_URL,
        human_review_attested=False,automated_official_mapping_check=True,
        numeric_raw_tag_check='performed during extraction and raw audit',
        original_selected_intervals=before,selected_intervals=len(chosen),selected_participants=int(chosen.subject_id.nunique()),
        baseline_policy='All target baseline intervals omitted; historical v1 baseline mapping unsupported',
        historical_manifest_sha256=R.sha(a),analysis_manifest_sha256=R.sha(target),
        evidence_files={str(p.relative_to(root)):R.sha(p) for p in sorted((root/'assets/protocol_evidence').glob('*')) if p.is_file()},
        exclusions={'S02':'documented duplicated stress signals','S11':'conservative matched-cohort exclusion: short baseline and split aerobic session; not an author exclusion',
                    'f07':'documented invalid signals/tags','f13':'documented bad wrist fit','f14':'documented split stress recording'},
        sampling_policy='Retain historical intervals: second half of rest; full TMCT; central 60 s of common 70/75/80/85 rpm stages',
        limitation='Protocol tasks are labels, not independently measured psychological states.'))
    print(mappings.to_string(index=False));print('Selected intervals:',len(chosen),'Participants:',chosen.subject_id.nunique())


def snapshot(root):
    # Immutable extraction provenance exists before either analysis is fitted.
    selected=[root/'design.json',root/'paths.json',root/'label_evidence.json',root/'runtime.json',root/'requirements_frozen.txt',
              *sorted((root/'code').glob('*.py')),*sorted((root/'data').glob('*.csv')),root/'data/raw_input_hashes.json']
    missing=[str(p) for p in selected if not p.is_file()]
    if missing:raise FileNotFoundError(str(missing))
    immutable(root/'provenance.json',dict(code_version='6.0.0',workflow_version=WORKFLOW_VERSION,
        status='Unscreened extraction reference for a retrospective quality sensitivity; not a scientific approval',
        files={str(p.relative_to(root)):dict(sha256=R.sha(p),bytes=p.stat().st_size) for p in selected},
        feature_hashes={n:R.sha(root/'data'/n) for n in NPZS}))


def prepare(root):
    b=R.prepare(root,root/'quality','screen_sensitivity')
    immutable(root/'quality_branch.json',dict(path=str(b),branch_sha256=R.sha(b/'branch.json')))
    flags=pd.read_csv(root/'quality/quality_queue_template.csv')
    if len(flags):print(flags.groupby(['partition','channel']).size().to_string())
    print('Uniform policy applied automatically; no names, approvals or review CSV editing required.')
    print('Heuristic masks are a retrospective sensitivity analysis, not validated signal repair.')


def branch(root):
    b=R.readj(root/'quality_branch.json');p=Path(b['path'])
    if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Branch outside workflow run')
    if R.sha(p/'branch.json')!=b['branch_sha256']:raise ValueError('Branch record changed')
    return p


def compare(root):
    b=branch(root);cfg=R.readj(root/'design.json');details=R.readj(b/'branch.json')
    done=R.readj(b/'analysis_complete.json')
    for n,h in done['outputs'].items():
        if R.sha(b/n)!=h:raise ValueError('Quality result changed: '+n)
    frames=[]
    for label,p in [('unscreened_reference',root),('automatic_quality_sensitivity',b)]:
        t=pd.read_csv(p/'results/table_target_metrics.csv');t.insert(0,'analysis',label);frames.append(t)
    table=pd.concat(frames,ignore_index=True)
    table.to_csv(root/'results/final_target_metrics_both_analyses.csv',index=False)
    frames=[]
    for label,p in [('unscreened_reference',root),('automatic_quality_sensitivity',b)]:
        t=pd.read_csv(p/'results/table_condition_rates.csv');t.insert(0,'analysis',label);frames.append(t)
    pd.concat(frames,ignore_index=True).to_csv(root/'results/final_exercise_tradeoffs_both_analyses.csv',index=False)
    # Same subject draws for screened-minus-reference contrasts, conditional on fitted models.
    from statistics_report import sample_counts,metric_batch,ci
    old=pd.read_csv(root/'models/target_predictions.csv');new=pd.read_csv(b/'models/target_predictions.csv')
    subjects=sorted(old.subject_id.unique());draws=np.vstack([np.ones(len(subjects),dtype=int),sample_counts(len(subjects),cfg['n_bootstrap'],cfg['bootstrap_seed'])])
    rows=[]
    for name in sorted(old.feature_set.unique()):
        a=old[(old.feature_set==name)&old.condition.isin(['rest','stress'])].set_index('window_id').sort_index()
        c=new[(new.feature_set==name)&new.condition.isin(['rest','stress'])].set_index('window_id').sort_index()
        if not a.index.equals(c.index) or not a[['subject_id','condition','y_true']].equals(c[['subject_id','condition','y_true']]):
            raise ValueError('Quality comparison changed endpoint/window identities')
        m,_=metric_batch(a,subjects,draws);n,_=metric_batch(c,subjects,draws)
        for metric in m:
            v=n[metric]-m[metric];lo,hi=ci(v[1:])
            rows.append(dict(feature_set=name,metric=metric,estimate=v[0],ci_low=lo,ci_high=hi,
                contrast='automatic quality sensitivity minus unscreened reference',n_subjects=len(subjects),
                inference='retrospective; marginal intervals conditional on fitted models'))
    pd.DataFrame(rows).to_csv(root/'results/quality_minus_reference_paired.csv',index=False)
    # Explicit count of imputed features is already in each SHAP row; aggregate for reporting.
    sh=pd.read_csv(b/'results/shap_values.csv.gz')
    sh['missing_input']=sh.raw_value.isna().astype(int)
    sh.groupby(['feature_set','condition','feature']).agg(n_rows=('missing_input','size'),n_imputed=('missing_input','sum')).reset_index().to_csv(root/'results/shap_imputation_counts.csv',index=False)
    selected=pd.read_csv(b/'results/table_target_operating.csv')
    f=selected[(selected.scope=='primary_rest_stress')&(selected.feature_set=='fts6')].iloc[0]
    text=f'''# Automatically assembled study notes — workflow {WORKFLOW_VERSION}

Working title: Cross-dataset generalization and explanation variability in wearable stress detection: a retrospective quality-sensitivity analysis.

Research questions
1. How does performance differ between participant-grouped WESAD validation and the previously inspected PhysioNet evaluation cohort?
2. How do frozen-model feature contributions vary across participants, rest, stress and exercise, including imputed inputs?
3. How do feature selection, mean-heart-rate removal and uniform signal-quality masking change sensitivity, specificity and exercise predicted-stress rates?

Design and provenance
Source: WESAD; target: PhysioNet wearable-device-dataset v1.0.1. Source-only and discovery-informed strategies are identified separately. Historical discovery/evaluation identities remain fixed. All target analyses are retrospective. Source validation uses {cfg['cv_repeats']} repeats of {cfg['cv_folds']}-fold participant-grouped CV, with selection, removal, preprocessing and thresholds developed within training partitions. Final thresholds use source data. Target outcomes do not tune thresholds or choose a preferred quality policy.

Task labels
TMCT, first/second rest and common aerobic stages use embedded official protocol evidence and actual raw-tag alignment. All target baseline intervals are omitted. No human review is automatically attested. Labels describe experimental tasks, not clinical diagnoses.

Quality handling
Each 60-second window is screened for raw nonfinite values, temperature outside 0–50 recorded degrees C, insufficient detected pulses, detected interbeat intervals or edge gaps exceeding 3 seconds, and estimated pulse rate outside 30–200 bpm. These bounds are heuristic sensitivity rules, not validated physiological standards. Nonfinite essential features are also flagged. The policy applies identically across participants and partitions. All features from each flagged channel-window are set to missing; imputers are fitted only on the corresponding training data. No temperature offset is invented and no missing beats are fabricated. Original windows and participant identities remain aligned across analyses. Optional EDA response summaries that are nonfinite alone do not invalidate all EDA features.

Routing
Changed channel-windows: {details['n_changed_channel_windows']}. Source/discovery redevelopment required: {details['redevelop']}. If false, the reference models, source validation and thresholds are reused; target predictions and SHAP are regenerated. If true, a separate complete nested development is run.

Generated numerical statement (quality sensitivity only)
FTS6: {int(f.n_subjects)} evaluation participants, {int(f.n_windows)} rest/stress windows; macro-F1 {f.macro_f1:.4f}, sensitivity {f.sensitivity:.4f}, specificity {f.specificity:.4f}, AUROC {f.auroc:.4f}. Use the matching intervals in final_target_metrics_both_analyses.csv. These values are generated at runtime, not promised improvements. Report the unscreened comparison and all quality-induced changes, including adverse results.

Interpretation limits
The unscreened analysis is a reference, not an automatically valid primary result. The quality-screened analysis is a retrospective sensitivity, not independent validation or certified measurement repair. Reporting both does not establish the best deployment policy. Participant bootstrap intervals are marginal, conditional on fitted models and observed participants, and not adjusted for multiple comparisons. Window counts are not independent participant counts. SHAP describes model behavior relative to a source background; imputed values are marked and contributions are not causal physiological effects. Exercise predicted-stress rate is an operational specificity challenge, not proof that all exercising participants are psychologically unstressed. Modeled feature-extraction energy excludes sensing and total device energy. No acceptance, journal quartile or employment outcome is inferred.

Sources
- Official dataset and associated paper: https://physionet.org/content/wearable-device-dataset/1.0.1/ ; https://doi.org/10.1038/s41597-025-04845-9
- Official protocol notebook: {OFFICIAL_URL}
'''
    (root/'results/STUDY_NOTES.md').write_text(text)
    immutable(root/'final_results.json',dict(workflow_version=WORKFLOW_VERSION,quality_branch=str(b),
        human_review_attested=False,analysis_status='retrospective automatic quality sensitivity with unscreened reference',
        source_discovery_redeveloped=details['redevelop'],n_changed_channel_windows=details['n_changed_channel_windows'],
        evidence_limits=['Heuristic signal screens','Previously inspected target cohort','Laboratory task labels','No publication verdict']))
    print(selected[selected.scope=='primary_rest_stress'].to_string(index=False))
    print('Complete comparison and study notes saved. No manually filled review file is required.')


def archive(root):
    # Verify every completed pipeline stage before selecting export files.
    required_stages=['manifest','extract','snapshot','audit','prepare','reference_develop','reference_evaluate','reference_statistics','reference_explain','quality_analysis','summary']
    for name in required_stages:
        marker=root/('stage_'+name+'.json')
        if not marker.exists():raise ValueError('Run the missing notebook stage first: '+name)
    for marker in sorted(root.glob('stage_*.json')):
        record=R.readj(marker)
        for p,h in {**record['inputs'],**record['outputs']}.items():
            if not Path(p).is_file() or R.sha(p)!=h:raise ValueError('Changed stage artifact: '+p)
    R.verify_origin(root)
    b=branch(root)
    for n,h in R.readj(b/'analysis_complete.json')['outputs'].items():
        if R.sha(b/n)!=h:raise ValueError('Changed quality artifact: '+n)
    # Full restart files stay on Drive. This review ZIP omits raw signals/NPZ/model binaries.
    paths=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix in ['.npz','.joblib','.zip','.pyc'] or '.tmp' in p.name:continue
        rel=p.relative_to(root)
        if any(x in rel.parts for x in ['fold_checkpoints','synthetic_software_tests','__pycache__']):continue
        if any(x.startswith('synthetic_') for x in rel.parts):continue
        paths.append(p)
    hashes={str(p.relative_to(root)):R.sha(p) for p in paths}
    content_key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    dest=root.parent/(root.name+'_'+content_key+'_review.zip')
    if not dest.exists():
        temp=dest.with_suffix('.zip.partial')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.relative_to(root))
            z.writestr('archive_hashes.json',json.dumps(hashes,indent=2,sort_keys=True))
        temp.replace(dest)
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError('ZIP corruption')
        if json.loads(z.read('archive_hashes.json'))!=hashes:raise ValueError('Existing ZIP has different contents')
        if set(z.namelist())!=set(hashes)|{'archive_hashes.json'}:raise ValueError('Unexpected ZIP contents')
        for n,h in hashes.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('ZIP hash mismatch: '+n)
    # Pointer outside root avoids changing the archived set on rerun.
    (root.parent/(root.name+'_archive_path.txt')).write_text(str(dest))
    print('REVIEW ZIP SAVED:',dest);print('SHA-256:',R.sha(dest))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['manifest','snapshot','audit','prepare','quality','summary','archive'])
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
    if a.action=='audit':R.audit(root,root/'quality')
    elif a.action=='quality':R.run_analysis(root,branch(root))
    else:globals()[a.action if a.action!='summary' else 'compare'](root)


if __name__=='__main__':main()
