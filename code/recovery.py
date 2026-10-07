"""FTS recovery: raw-signal audit and explicit retrospective quality analyses.

No target-performance optimization, silent label change or physiological repair.
Only hash-verified original study code/models/data are consumed. Review queues
start pending. Heuristic screens are not validated quality standards.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZS = ['wesad_features_meta.npz', 'physionet_stress_meta.npz', 'physionet_exercise_active_meta.npz']
CHANNEL_FEATURES = {
    'TEMP': ['temp_mean', 'temp_std', 'temp_slope'],
    'BVP': ['bvp_mean_hr', 'bvp_sdnn', 'bvp_rmssd', 'bvp_pnn50', 'bvp_lf', 'bvp_hf', 'bvp_lf_hf'],
    'EDA': ['eda_mean', 'eda_std', 'eda_slope', 'eda_min', 'eda_max', 'eda_range', 'eda_scl_mean', 'eda_scr_count', 'eda_scr_mean_amp', 'eda_scr_auc'],
    'ACC': ['acc_mean', 'acc_std', 'acc_max'],
}
SCREEN = {'temperature_review_bounds_C': [0, 50], 'pulse_review_hr_bounds_bpm': [30, 200],
          'pulse_review_max_interval_s': 3, 'pulse_review_max_edge_gap_s': 3,
          'status': 'Heuristic review screens; not validated physiological validity criteria. No flags use predictions or outcomes.'}
OFFICIAL = {('STRESS','v1','TMCT'):(5,6),('STRESS','v2','TMCT'):(2,3),
    ('STRESS','v1','First Rest'):(4,5),('STRESS','v1','Second Rest'):(6,7),
    ('STRESS','v2','First Rest'):(3,4),('STRESS','v2','Second Rest'):(7,8),
    **{('AEROBIC','v1',f'{rpm} rpm'):(i,i+1) for rpm,i in [(70,2),(75,3),(80,4),(85,5)]},
    **{('AEROBIC','v2',f'{rpm} rpm'):(i,i+1) for rpm,i in [(70,1),(75,2),(80,3),(85,4)]}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def readj(path): return json.loads(Path(path).read_text())


def writej(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def verify_origin(run):
    proof = readj(run/'provenance.json')
    if proof['code_version'] != '6.0.0': raise ValueError('Expected compatible 6.0.0 feature/model schema.')
    for n, item in proof['files'].items():
        p = (run/n).resolve()
        if not p.is_relative_to(run.resolve()): raise ValueError('Invalid provenance path')
        if not p.is_file() or sha(p) != item['sha256']: raise ValueError(f'Reference artifact changed: {p}')
    for n, h in proof['feature_hashes'].items():
        if sha(run/'data'/n) != h: raise ValueError(f'Original features changed: {n}')
    return proof


def pulse(window, fs=64):
    """The original peak extractor plus diagnostics; no repaired beats."""
    x = np.asarray(window, float)
    flags = []
    d = dict(n_peaks=0, hr_bpm=np.nan, ibi_min_ms=np.nan, ibi_max_ms=np.nan,
             sdnn_ms=np.nan, rmssd_ms=np.nan, start_gap_s=np.nan, end_gap_s=np.nan)
    if len(x) != int(60*fs): raise ValueError('Pulse window is not exactly 60 seconds')
    if not np.isfinite(x).all(): return d, ['nonfinite_raw'], np.array([], int), x
    b, a = butter(3, [.5/(fs/2), 8/(fs/2)], btype='band')
    clean = filtfilt(b, a, x)
    peaks, _ = find_peaks(clean, distance=int(.33*fs), prominence=max(.4*np.std(clean), 1e-8))
    d['n_peaks'] = len(peaks)
    if len(peaks) < 4: flags.append('fewer_than_four_peaks')
    if len(peaks):
        d.update(start_gap_s=float(peaks[0]/fs), end_gap_s=float((len(x)-1-peaks[-1])/fs))
        if max(d['start_gap_s'], d['end_gap_s']) > SCREEN['pulse_review_max_edge_gap_s']:
            flags.append('edge_gap_gt_3s_screen')
    if len(peaks) >= 2:
        ibi = np.diff(peaks)/fs*1000
        d.update(hr_bpm=float(60000/ibi.mean()), ibi_min_ms=float(ibi.min()), ibi_max_ms=float(ibi.max()),
                 sdnn_ms=float(ibi.std()), rmssd_ms=float(np.sqrt(np.mean(np.diff(ibi)**2))) if len(ibi)>1 else np.nan)
        if ibi.max() > 3000: flags.append('detected_interval_gt_3s_screen')
        if not 30 <= d['hr_bpm'] <= 200: flags.append('hr_outside_30_200_screen')
    return d, flags, peaks, clean


def temp_flags(values):
    x = np.asarray(values, float)
    flags = []
    if not np.isfinite(x).all(): flags.append('nonfinite_raw')
    f = x[np.isfinite(x)]
    if len(f) and ((f < 0).any() or (f > 50).any()): flags.append('temperature_outside_0_50_screen')
    return flags


def e4(path, cols):
    d = pd.read_csv(path, header=None)
    start = str(d.iloc[0, 0]); fs = float(d.iloc[1, 0])
    if not np.isfinite(fs) or fs <= 0: raise ValueError(f'Invalid sample rate {path}')
    x = d.iloc[2:, :cols].to_numpy(float)
    if cols == 3: x = np.sqrt(np.sum(x*x, axis=1))
    else: x = x[:, 0]
    return x, fs, start


def utc_seconds(value):
    try: return float(value)
    except (ValueError, TypeError): return pd.Timestamp(value, tz='UTC').timestamp()


def partition(dataset, subject, cfg):
    if dataset == 'wesad': return 'source'
    if subject in cfg['discovery_subjects']: return 'discovery'
    if subject in cfg['confirmation_subjects']: return 'evaluation'
    return 'unused_target'


def audit(run, out):
    proof = verify_origin(run)
    if (out/'audit_complete.json').exists():
        done = readj(out/'audit_complete.json')
        if done['origin_provenance_sha256'] != sha(run/'provenance.json') or done['code_sha256'] != sha(__file__):
            raise ValueError('Audit version changed. Use a new recovery folder.')
        for n,h in done['outputs'].items():
            if sha(out/n) != h: raise ValueError(f'Audit artifact changed: {n}')
        print('Verified completed raw audit reused.', flush=True); return
    out.mkdir(parents=True, exist_ok=True); (out/'plots').mkdir(exist_ok=True)
    writej(out/'screen_policy.json', SCREEN)
    cfg = readj(run/'design.json'); paths = readj(run/'paths.json'); rawhash = readj(run/'data/raw_input_hashes.json')
    index = pd.read_csv(run/'data/window_index.csv')
    arrays = {n: npz(run/'data'/n) for n in NPZS}
    feature_rows = {}
    for n, a in arrays.items():
        for i, wid in enumerate(a['window_ids'].astype(str)):
            if wid in feature_rows: raise ValueError('Duplicated window IDs')
            feature_rows[wid] = (n, dict(zip(a['feature_names'].astype(str), a['X'][i])),set(a['feature_names'][a['tier']=='EFS'].astype(str)))
    if set(index.window_id) != set(feature_rows): raise ValueError('Index and features differ')
    verified = {}
    def check(path):
        key = str(path)
        if key not in rawhash: raise ValueError(f'Raw input has no recorded hash: {path}')
        if key not in verified:
            h = sha(path)
            if h != rawhash[key]['sha256']: raise ValueError(f'Raw input changed: {path}')
            verified[key] = h
    manifest = pd.read_csv(run/'data/physionet_interval_manifest.csv')
    selected = manifest[manifest.include == 1].copy()
    selected['block_id'] = selected.apply(lambda r:f'{r.protocol}_{r.raw_folder}_tags{int(r.start_tag_index)}-{int(r.end_tag_index)}', axis=1)
    byblock = selected.set_index('block_id')
    rows, queue, errors, structural = [], [], [], []
    for (dataset, protocol, subject), group in index.groupby(['dataset','protocol','subject_id'], sort=True):
        print(f'Auditing {dataset}/{protocol}/{subject}: {len(group)} windows', flush=True)
        if dataset == 'wesad':
            path = Path(paths['wesad'])/subject/(subject+'.pkl'); check(path)
            # Official WESAD pickle already used in this run, hash checked before loading.
            with path.open('rb') as f: raw = pickle.load(f, encoding='latin1')
            wrist = raw['signal']['wrist']; native = {}
            for c, fs in [('BVP',64),('TEMP',4),('EDA',4),('ACC',32)]:
                a = np.asarray(wrist[c],float)
                native[c] = (np.sqrt(np.sum(a*a,axis=1)) if c=='ACC' else a.ravel(), fs)
            labels = np.asarray(raw['label']).ravel()
        else:
            folders = set(byblock.loc[group.block_id, 'raw_folder'])
            if len(folders) != 1: raise ValueError(f'Unexpected split folder: {subject}')
            folder = Path(paths['physionet'])/protocol/next(iter(folders))
            native = {}; starts = []
            for c in ['BVP','TEMP','EDA','ACC']:
                path = folder/(c+'.csv'); check(path)
                a,fs,stamp = e4(path,3 if c=='ACC' else 1)
                expected = {'BVP':64,'TEMP':4,'EDA':4,'ACC':32}[c]
                if fs != expected: raise ValueError(f'Unexpected native fs for {path}: {fs}')
                native[c] = (a,fs); starts.append(utc_seconds(stamp))
            if max(starts)-min(starts)>1e-6: raise ValueError(f'Channel start mismatch: {folder}')
            path = folder/'tags.csv'; check(path)
            tags = [0.] + [utc_seconds(v)-starts[0] for v in pd.read_csv(path,header=None).iloc[:,0]]
            for block in group.block_id.unique():
                m = byblock.loc[block]
                i,j = int(m.start_tag_index),int(m.end_tag_index)
                valid = 0 <= i < j < len(tags)
                valid = valid and np.allclose([tags[i],tags[j]],[m.start_s,m.end_s],atol=.05,rtol=0)
                windows = group[group.block_id == block]
                valid = valid and bool(((windows.window_start_s >= m.analysis_start_s-1e-8)&(windows.window_end_s <= m.analysis_end_s+1e-8)).all())
                structural.append(dict(block_id=block, raw_tag_match=bool(valid), condition=m.condition,
                    evidence_scope='Numeric tag alignment; task mapping checked separately against embedded official evidence'))
                if not valid: errors.append(f'Tag/bounds mismatch: {block}')
        for r in group.itertuples():
            grid = r.window_start_s + np.arange(3840)/64
            if not np.isclose(r.window_end_s-r.window_start_s,60): raise ValueError('Window duration changed')
            values = {}
            for c,(a,fs) in native.items():
                if grid[0]<0 or grid[-1]>len(a)/fs: raise ValueError(f'Window outside {c}: {r.window_id}')
                values[c] = np.interp(grid,np.arange(len(a))/fs,a)
            if dataset == 'wesad':
                y = labels[np.clip(np.round(grid*700).astype(int),0,len(labels)-1)]
                expected = 1 if r.condition=='rest' else 2
                if not np.all(y==expected): errors.append(f'WESAD label mismatch: {r.window_id}')
            d, pf, peaks, clean = pulse(values['BVP'])
            flags = {'BVP':pf, 'TEMP':temp_flags(values['TEMP']),
                     'EDA':[] if np.isfinite(values['EDA']).all() else ['nonfinite_raw'],
                     'ACC':[] if np.isfinite(values['ACC']).all() else ['nonfinite_raw']}
            file, feats, efs = feature_rows[r.window_id]
            for c, names in CHANNEL_FEATURES.items():
                if any(not np.isfinite(feats[n]) for n in names if n in efs): flags[c].append('nonfinite_extracted_EFS_feature')
            # Verify raw reconstruction corresponds to the saved features before acting.
            if np.isfinite(values['TEMP']).all() and not np.isclose(values['TEMP'].mean(),feats['temp_mean'],rtol=2e-6,atol=2e-5):
                errors.append(f'Temperature feature reconstruction mismatch: {r.window_id}')
            if len(peaks)>=4 and np.isfinite(d['hr_bpm']) and not np.isclose(d['hr_bpm'],feats['bvp_mean_hr'],rtol=2e-6,atol=2e-5):
                errors.append(f'Pulse feature reconstruction mismatch: {r.window_id}')
            base = dict(file=file,dataset=dataset,subject_id=subject,partition=partition(dataset,subject,cfg),
                        condition=r.condition,window_id=r.window_id,block_id=r.block_id,
                        window_start_s=r.window_start_s,window_end_s=r.window_end_s)
            v = dict(base, **d)
            for c,x in values.items():
                finite=x[np.isfinite(x)]
                v.update({c+'_min':float(finite.min()) if len(finite) else np.nan,
                          c+'_max':float(finite.max()) if len(finite) else np.nan,
                          c+'_nonfinite':int((~np.isfinite(x)).sum()),c+'_flags':';'.join(flags[c])})
            rows.append(v)
            if any(flags.values()):
                key = hashlib.sha256(r.window_id.encode()).hexdigest()[:16]
                fig, axes = plt.subplots(3,1,figsize=(11,7),sharex=True)
                axes[0].plot(grid,values['BVP'],lw=.6);axes[0].set_ylabel('Raw BVP')
                axes[1].plot(grid,clean,lw=.6);axes[1].scatter(grid[peaks],clean[peaks],s=10,c='red');axes[1].set_ylabel('Filtered + peaks')
                axes[2].plot(grid,values['TEMP'],lw=.8);axes[2].set_ylabel('TEMP (recorded C)');axes[2].set_xlabel('Recording time (s)')
                fig.suptitle(r.window_id+' | raw-signal review; no predictions shown')
                fig.tight_layout();fig.savefig(out/'plots'/f'{key}.png',dpi=110);plt.close(fig)
                for c,reasons in flags.items():
                    if reasons: queue.append(dict(base,channel=c,reasons=';'.join(reasons),plot=f'plots/{key}.png',
                        action='pending',reviewer='',evidence='',reviewed_utc=''))
        del native
        if dataset == 'wesad': del raw,wrist,labels
        gc.collect()
    pd.DataFrame(rows).to_csv(out/'all_window_quality.csv',index=False)
    qcols=['file','dataset','subject_id','partition','condition','window_id','block_id','window_start_s','window_end_s','channel','reasons','plot','action','reviewer','evidence','reviewed_utc']
    q = pd.DataFrame(queue,columns=qcols);q.to_csv(out/'quality_queue_template.csv',index=False)
    q.groupby(['partition','channel']).size().rename('flagged_channel_windows').reset_index().to_csv(out/'flag_counts.csv',index=False)
    pd.DataFrame(structural).to_csv(out/'numeric_label_checks.csv',index=False)
    keys=['protocol','protocol_version','condition','source_block','start_tag_index','end_tag_index']
    mappings=selected.groupby(keys).size().reset_index(name='n_intervals')
    mappings['mapping_id']=[f'mapping_{i:03d}' for i in range(len(mappings))]
    mappings['decision']='pending';mappings['reviewer']='';mappings['evidence']='';mappings['reviewed_utc']=''
    mappings.loc[mappings.condition=='baseline','decision']='not_used'
    mappings['analysis_scope']=np.where(mappings.condition=='baseline','omitted_from_recovery_endpoints','primary_or_exercise')
    mappings['official_mapping_supported']=[OFFICIAL.get((r.protocol,r.protocol_version,r.source_block))==(int(r.start_tag_index),int(r.end_tag_index)) for r in mappings.itertuples()]
    mappings['official_evidence']='Wearable_Dataset.ipynb: cells 9/11/14/16 protocol figures and cell 25 graph_multiple (zero-based cell indexing)'
    for r in mappings.itertuples():
        if r.condition!='baseline' and not r.official_mapping_supported:errors.append(f'Unsupported official mapping: {r.mapping_id}')
    mappings.to_csv(out/'label_mapping_review_template.csv',index=False)
    writej(out/'verified_raw_hashes.json',verified)
    outputs=['all_window_quality.csv','quality_queue_template.csv','flag_counts.csv','numeric_label_checks.csv',
             'label_mapping_review_template.csv','screen_policy.json','verified_raw_hashes.json']
    writej(out/'audit_complete.json',dict(origin_provenance_sha256=sha(run/'provenance.json'),
        code_sha256=sha(__file__),n_windows=len(rows),n_flagged_channel_windows=len(q),hard_errors=errors,
        outputs={n:sha(out/n) for n in outputs},status='automated audit complete; no human review attested'))
    print('Audit saved:',out,flush=True)
    print(q.groupby(['partition','channel']).size().to_string(),flush=True)
    if errors: print('Structural/reconstruction errors require resolution:',errors,flush=True)


def decisions(out, mode):
    done=readj(out/'audit_complete.json')
    if done['hard_errors']: raise ValueError('Resolve numeric label/reconstruction errors before analysis: '+str(done['hard_errors']))
    q=pd.read_csv(out/'quality_queue_template.csv',keep_default_na=False)
    label_ok=False
    if mode=='reviewed':
        d=pd.read_csv(out/'quality_decisions.csv',keep_default_na=False)
        keys=['window_id','channel']
        if d.duplicated(keys).any() or set(map(tuple,d[keys].to_numpy()))!=set(map(tuple,q[keys].to_numpy())):
            raise ValueError('Review must resolve each original flagged channel-window exactly once.')
        d=d.set_index(keys).loc[pd.MultiIndex.from_frame(q[keys])].reset_index()
        if not d.action.isin(['retain','mask_channel']).all(): raise ValueError('Complete quality_decisions.csv: action must be retain or mask_channel.')
        for c in ['reviewer','evidence','reviewed_utc']:
            if d[c].str.strip().eq('').any() or d[c].str.contains('YOUR_NAME|TODO|pending',case=False,regex=True).any():
                raise ValueError(f'Provide actual {c} for every quality decision.')
        # Use immutable queue identities; user edits cannot redirect a decision.
        q['action']=d.action.to_numpy();q['reviewer']=d.reviewer.to_numpy();q['evidence']=d.evidence.to_numpy();q['reviewed_utc']=d.reviewed_utc.to_numpy()
        l=pd.read_csv(out/'label_mapping_review.csv',keep_default_na=False)
        template=pd.read_csv(out/'label_mapping_review_template.csv',keep_default_na=False)
        if l.mapping_id.duplicated().any() or set(l.mapping_id)!=set(template.mapping_id): raise ValueError('Incomplete mapping review')
        l=l.set_index('mapping_id').loc[template.mapping_id].reset_index()
        for c in ['protocol','protocol_version','condition','source_block','start_tag_index','end_tag_index','n_intervals']:
            if not np.array_equal(l[c].to_numpy(),template[c].to_numpy()): raise ValueError('Mapping edited: correct labels through a new extraction, not this channel-masking recovery.')
        used=l.condition!='baseline'
        if not l.loc[used,'decision'].eq('confirmed').all(): raise ValueError('Resolve every primary/exercise task mapping using evidence before a reviewed analysis. Changed labels require new extraction.')
        for c in ['reviewer','evidence','reviewed_utc']:
            if l.loc[used,c].str.strip().eq('').any() or l.loc[used,c].str.contains('YOUR_NAME|TODO|pending',case=False,regex=True).any(): raise ValueError('Complete actual mapping review evidence.')
        label_ok=True
    elif mode=='screen_sensitivity':
        # Explicit stress test of the heuristic screen, not a scientific review verdict.
        q['action']='mask_channel';q['reviewer']='automatic heuristic sensitivity';q['evidence']='screen_policy.json';q['reviewed_utc']='not a human review'
    else: raise ValueError(mode)
    return q,label_ok


def changed_scope(changes,cfg):
    # Discovery baseline is unused; both discovery rest and exercise enter FTS development.
    source=(changes.partition=='source')
    disc=(changes.partition=='discovery') & changes.condition.isin(['rest','stress','exercise'])
    return bool((source|disc).any())


def prepare(run,out,mode):
    verify_origin(run)
    done=readj(out/'audit_complete.json')
    if done['code_sha256']!=sha(__file__):raise ValueError('Code changed after raw audit; use a new audit folder.')
    for n,h in done['outputs'].items():
        if sha(out/n)!=h:raise ValueError('Audit evidence changed: '+n)
    q,label_ok=decisions(out,mode);cfg=readj(run/'design.json')
    label_hash=sha(out/'label_mapping_review.csv') if label_ok else 'not_reviewed'
    signature=hashlib.sha256((q.to_csv(index=False)+mode+sha(out/'audit_complete.json')+label_hash).encode()).hexdigest()[:12]
    branch=out/f'{mode}_{signature}'
    if (branch/'branch.json').exists():
        b=readj(branch/'branch.json')
        for n,h in b['feature_hashes'].items():
            if sha(branch/'data'/n)!=h:raise ValueError('Prepared features changed')
        print('BRANCH:',branch);return branch
    branch.mkdir(exist_ok=True);(branch/'data').mkdir(exist_ok=True)
    changes=[]
    for file in NPZS:
        a=npz(run/'data'/file);names=a['feature_names'].astype(str).tolist();ids={s:i for i,s in enumerate(a['window_ids'].astype(str))}
        for r in q[(q.file==file)&(q.action=='mask_channel')].itertuples():
            i=ids[r.window_id]
            cols=[names.index(f) for f in CHANNEL_FEATURES[r.channel]]
            if not np.isnan(a['X'][i,cols]).all():
                a['X'][i,cols]=np.nan
                changes.append(dict(file=file,window_id=r.window_id,channel=r.channel,partition=r.partition,condition=r.condition,subject_id=r.subject_id))
        np.savez_compressed(branch/'data'/file,**a)
    c=pd.DataFrame(changes,columns=['file','window_id','channel','partition','condition','subject_id'])
    c.to_csv(branch/'changed_channel_windows.csv',index=False);q.to_csv(branch/'frozen_quality_decisions.csv',index=False)
    if label_ok:shutil.copy2(out/'label_mapping_review.csv',branch/'frozen_label_mapping_review.csv')
    writej(branch/'design.json',cfg)
    writej(branch/'branch.json',dict(origin_run=str(run),mode=mode,label_review_completed=label_ok,
        interpretation='Retrospective quality sensitivity; not independent confirmation or validated physiological repair.',
        redevelop=changed_scope(c,cfg),n_changed_channel_windows=len(c),
        feature_hashes={n:sha(branch/'data'/n) for n in NPZS},
        frozen_decisions_sha256=sha(branch/'frozen_quality_decisions.csv'),audit_sha256=sha(out/'audit_complete.json')))
    print('BRANCH:',branch,flush=True)
    print('Development required:',changed_scope(c,cfg),'Changed channel-windows:',len(c),flush=True)
    return branch


def evaluate_frozen(run,branch,cfg):
    import joblib
    from models import Engine
    from common import runtime_versions
    lock=readj(run/'models/development_lock.json')
    if lock['runtime']!=runtime_versions():raise ValueError('Model runtime differs from the original. Use the notebook isolated environment.')
    engine=Engine(cfg,branch/'data');predictions=[]
    old=pd.read_csv(run/'models/target_predictions.csv')
    for name,spec in lock['models'].items():
        p=run/'models'/spec['model_file']
        if sha(p)!=spec['sha256']:raise ValueError('Changed original model: '+name)
        model=joblib.load(p)
        for file,a in zip(NPZS[1:],[engine.P,engine.E]):
            mask=np.isin(a['groups'],cfg['confirmation_subjects'])
            original=npz(run/'data'/file);cols=spec['columns']
            before=model.predict_proba(original['X'][mask][:,cols])[:,1]
            ref=old[old.feature_set==name].set_index('window_id').loc[a['window_ids'][mask]]
            if not np.allclose(before,ref.y_prob,atol=1e-8):raise ValueError('Original probabilities not reproduced')
            pred=engine.prediction_rows(a,mask,model,cols,spec['threshold'],name,'retrospective_quality_sensitivity',0,0,spec['learner'])
            same=np.all(np.isclose(original['X'][mask][:,cols],a['X'][mask][:,cols],equal_nan=True,rtol=0,atol=0),axis=1)
            if not np.allclose(pred.y_prob.to_numpy()[same],before[same],atol=1e-8):raise ValueError('Unaffected predictions changed')
            predictions.append(pred)
        if sha(p)!=spec['sha256']:raise ValueError('Original model changed during evaluation')
    pd.concat(predictions,ignore_index=True).to_csv(branch/'models/target_predictions.csv',index=False)


def explain_frozen(run,branch,cfg):
    import joblib,shap
    from scipy.special import expit
    from explain import background_indices,summarize,figures
    lock=readj(run/'models/development_lock.json')
    W=npz(run/'data'/NPZS[0]);idx=background_indices(W['groups'],cfg.get('shap_background_windows',90),cfg.get('shap_seed',318))
    predictions=pd.read_csv(branch/'models/target_predictions.csv');rows=[];checks=[]
    for name in cfg.get('shap_models',['fts6','fts5','all_efs','target_importance6']):
        spec=lock['models'][name];path=run/'models'/spec['model_file']
        if sha(path)!=spec['sha256']:raise ValueError('Model hash mismatch')
        pipe=joblib.load(path);imp,tree=pipe.steps[0][1],pipe.steps[-1][1];cols=spec['columns']
        explainer=shap.TreeExplainer(tree,data=imp.transform(W['X'][idx][:,cols]),feature_perturbation='interventional',model_output='raw')
        base=float(np.asarray(explainer.expected_value).item())
        for file in NPZS[1:]:
            a=npz(branch/'data'/file);mask=np.isin(a['groups'],cfg['confirmation_subjects']) & np.isin(a['conditions'],['rest','stress','exercise'])
            raw=a['X'][mask][:,cols];x=imp.transform(raw);phi=np.asarray(explainer.shap_values(x,check_additivity=True))
            prob=pipe.predict_proba(raw)[:,1];margin=tree.predict(x,output_margin=True)
            if not np.allclose(base+phi.sum(axis=1),margin,atol=2e-4,rtol=2e-5):raise ValueError('SHAP additivity failed')
            if not np.allclose(expit(base+phi.sum(axis=1)),prob,atol=2e-5):raise ValueError('SHAP probability reconstruction failed')
            ref=predictions[predictions.feature_set==name].set_index('window_id').loc[a['window_ids'][mask]]
            if not np.allclose(ref.y_prob,prob,atol=1e-8):raise ValueError('SHAP prediction alignment failed')
            for j,feature in enumerate(spec['features']):
                rows.append(pd.DataFrame(dict(feature_set=name,subject_id=a['groups'][mask],window_id=a['window_ids'][mask],condition=a['conditions'][mask],feature=feature,raw_value=raw[:,j],imputed_value=x[:,j],shap_value=phi[:,j],base_value=base,y_prob=prob,y_pred=(prob>=spec['threshold']).astype(int),threshold=spec['threshold'])))
            checks.append(dict(feature_set=name,file=file,max_margin_error=float(np.abs(base+phi.sum(axis=1)-margin).max())))
    long=pd.concat(rows,ignore_index=True);res=branch/'results'
    long.to_csv(res/'shap_values.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    long,summary=summarize(long,cfg,res);figures(long,summary,res)
    writej(res/'shap_recovery_audit.json',dict(checks=checks,source_background='original source only',no_model_refitting=True))


def run_analysis(run,branch):
    verify_origin(run);b=readj(branch/'branch.json');cfg=readj(branch/'design.json')
    if str(run)!=b['origin_run']:raise ValueError('Wrong origin')
    if cfg!=readj(run/'design.json'):raise ValueError('Recovery configuration changed')
    if sha(branch/'frozen_quality_decisions.csv')!=b['frozen_decisions_sha256']:raise ValueError('Frozen decisions changed')
    for n,h in b['feature_hashes'].items():
        if sha(branch/'data'/n)!=h:raise ValueError('Recovery features changed')
    signature=dict(branch_sha256=sha(branch/'branch.json'),code_sha256=sha(__file__))
    if (branch/'analysis_complete.json').exists():
        done=readj(branch/'analysis_complete.json')
        if done['signature']!=signature:raise ValueError('Analysis changed after completion')
        for n,h in done['outputs'].items():
            if sha(branch/n)!=h:raise ValueError('Completed analysis artifact changed: '+n)
        print('Verified completed recovery analysis reused.');return
    (branch/'models').mkdir(exist_ok=True);(branch/'results').mkdir(exist_ok=True)
    # Exact workflow code is verified against the reference extraction provenance.
    sys.path.insert(0,str(run/'code'))
    from models import Engine
    import statistics_report
    if b['redevelop']:
        print('Source/discovery channel values changed: rerunning nested development in this branch.',flush=True)
        engine=Engine(cfg,branch/'data');engine.develop(branch/'models');engine.evaluate(branch/'models')
        statistics_report.run(branch/'data',branch/'models',branch/'results',cfg)
        import explain
        explain.run(branch/'data',branch/'models',branch/'results',cfg)
    else:
        print('Only unused/evaluation inputs changed: original fitted models and thresholds retained.',flush=True)
        evaluate_frozen(run,branch,cfg)
        p=pd.read_csv(branch/'models/target_predictions.csv');statistics_report.validate_target(p)
        _,rates=statistics_report.target_tables(p,cfg,branch/'results')
        source=pd.read_csv(run/'models/source_predictions.csv')
        splits=pd.read_csv(run/'models/split_metrics.csv')
        statistics_report.source_tables(source,splits,branch/'results',cfg['equivalence_margin'])
        statistics_report.costs_and_ranking(run/'data',run/'models',branch/'results',cfg)
        statistics_report.figures(source,p,rates,branch/'results')
        writej(branch/'results/source_reuse.json',dict(reason='No source/discovery changes; original source validation reused',
            source_predictions_sha256=sha(run/'models/source_predictions.csv'),development_lock_sha256=sha(run/'models/development_lock.json')))
        explain_frozen(run,branch,cfg)
    # The original v1 baseline mapping is not established by the official figure.
    # Baseline-inclusive sensitivity is outside this recovery's scope.
    for file in ['table_target_operating.csv','table_target_metrics.csv']:
        path=branch/'results'/file
        d=pd.read_csv(path);d=d[d.scope=='primary_rest_stress'];d.to_csv(path,index=False)
    old=pd.read_csv(run/'results/table_target_operating.csv');new=pd.read_csv(branch/'results/table_target_operating.csv')
    merged=new.merge(old,on=['scope','feature_set'],suffixes=('_recovery','_original'),validate='one_to_one')
    for k in ['macro_f1','sensitivity','specificity','auroc','brier']:merged[k+'_change']=merged[k+'_recovery']-merged[k+'_original']
    merged.to_csv(branch/'results/original_vs_recovery.csv',index=False)
    (branch/'results/INTERPRETATION.txt').write_text(
        'Retrospective quality sensitivity. Original scores are historical references, not automatically valid primary results.\n'
        +'Mode: '+b['mode']+'; full redevelopment: '+str(b['redevelop'])+'\n'
        +'Baseline-inclusive sensitivity omitted; version-1 baseline mapping was not supported by the official protocol figure. Primary rest/stress and exercise retained.\n'
        +'Heuristic masking does not certify signal validity. Official primary task mappings are automatically checked against embedded evidence. Screening thresholds are heuristic; independent pulse validity remains unestablished.\n'
        +'Participant bootstrap intervals are marginal and conditional on fitted models. No target threshold tuning.\n')
    outputs={str(p.relative_to(branch)):sha(p) for folder in ['results','models'] for p in sorted((branch/folder).rglob('*')) if p.is_file()}
    writej(branch/'analysis_complete.json',dict(signature=signature,outputs=outputs))
    print(new[new.scope=='primary_rest_stress'].to_string(index=False),flush=True)


def archive(out):
    # Review package includes diagnostic plots and exported results, not raw data,
    # NPZ arrays or serialized classifiers. CSVs include pseudonymous subject IDs.
    dest=out.parent/(out.name+'_review.zip')
    entries=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.suffix in ['.npz','.joblib','.zip'] or 'fold_checkpoints' in p.parts:continue
        entries.append(p)
    manifest={str(p.relative_to(out)):sha(p) for p in entries}
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in entries:z.write(p,p.relative_to(out))
        z.writestr('recovery_archive_hashes.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:
        if z.testzip() is not None:raise ValueError('ZIP corruption')
        for n,h in manifest.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archive hash mismatch')
    print('REVIEW_ZIP:',dest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','prepare','run','archive'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['reviewed','screen_sensitivity'],default='reviewed')
    p.add_argument('--branch',type=Path);a=p.parse_args()
    if a.action=='audit':audit(a.run,a.out)
    elif a.action=='prepare':prepare(a.run,a.out,a.mode)
    elif a.action=='run':
        if a.branch is None:raise ValueError('Specify prepared --branch')
        run_analysis(a.run,a.branch)
    else:archive(a.out)
