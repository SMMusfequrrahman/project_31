"""Bounded software checks; no synthetic number is a study result."""
from pathlib import Path
import json
import tempfile
import shutil
import sys
import unittest
import pickle
import numpy as np
import pandas as pd
import recovery as R


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from smoke_test import make_data
        from models import Engine
        import statistics_report
        cls.fixture_temp=tempfile.TemporaryDirectory()
        cls.fixture=Path(cls.fixture_temp.name)
        (cls.fixture/'data').mkdir()
        cfg=make_data(cls.fixture/'data')
        engine=Engine(cfg,cls.fixture/'data');engine.develop(cls.fixture/'models');engine.evaluate(cls.fixture/'models')
        statistics_report.run(cls.fixture/'data',cls.fixture/'models',cls.fixture/'results',cfg)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def test_clean_pulse_and_dropout(self):
        t=np.arange(3840)/64
        d,flags,_,_=R.pulse(np.sin(2*np.pi*1.2*t))
        self.assertFalse(flags);self.assertAlmostEqual(d['hr_bpm'],72,delta=1)
        _,flags,_,_=R.pulse(np.zeros(3840))
        self.assertIn('fewer_than_four_peaks',flags)
        x=np.sin(2*np.pi*1.2*t);x[(t>20)&(t<40)]=0
        _,flags,_,_=R.pulse(x)
        self.assertIn('detected_interval_gt_3s_screen',flags)

    def test_temperature_and_scope(self):
        self.assertFalse(R.temp_flags([32,33,34]))
        self.assertTrue(R.temp_flags([209,210]))
        self.assertIn('nonfinite_raw',R.temp_flags([32,np.nan]))
        for partition,condition,expected in [('source','rest',True),('discovery','stress',True),
            ('discovery','exercise',True),('discovery','baseline',False),('evaluation','stress',False),('unused_target','rest',False)]:
            self.assertEqual(R.changed_scope(pd.DataFrame([dict(partition=partition,condition=condition)]),{}),expected)

    def test_pending_review_is_not_attested(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);R.writej(out/'audit_complete.json',{'hard_errors':[]})
            q=pd.DataFrame([dict(window_id='w',channel='TEMP',action='pending',reviewer='',evidence='',reviewed_utc='')])
            q.to_csv(out/'quality_queue_template.csv',index=False);q.to_csv(out/'quality_decisions.csv',index=False)
            with self.assertRaises(ValueError):R.decisions(out,'reviewed')
            result,label_ok=R.decisions(out,'screen_sensitivity')
            self.assertFalse(label_ok);self.assertEqual(result.action.iloc[0],'mask_channel')

    def test_raw_audit_prepare_and_archive(self):
        # Minimal raw-signal fixture exercises identity alignment, all-window scan,
        # channel-only masking, decision routing and archive omission of NPZs.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run=root/'run';data=run/'data';data.mkdir(parents=True)
            rawroot=root/'raw';wesad=rawroot/'WESAD';phys=rawroot/'phys'
            names=[n for group in R.CHANNEL_FEATURES.values() for n in group]
            optional={'bvp_lf','bvp_hf','bvp_lf_hf','eda_scl_mean','eda_scr_count','eda_scr_mean_amp','eda_scr_auc'}
            tier=np.array(['COMP' if n in optional else 'EFS' for n in names])
            index=[];hashes={};man=[]
            for file,dataset,subject,protocol,condition,start,end,tagi,tagj in [
                (R.NPZS[0],'wesad','S2','WESAD','rest',0,60,0,1),
                (R.NPZS[1],'physionet','f01','STRESS','stress',20,80,2,3),
                (R.NPZS[2],'physionet','f01','AEROBIC','exercise',20,80,1,2)]:
                signals={c:np.sin(2*np.pi*1.2*np.arange(80*fs)/fs) if c=='BVP' else np.full(80*fs,210. if c=='TEMP' and condition=='exercise' else 32. if c=='TEMP' else 1.) for c,fs in [('BVP',64),('TEMP',4),('EDA',4),('ACC',32)]}
                if condition=='stress':signals['BVP'][int(62*64):]=0
                if dataset=='wesad':
                    folder=wesad/subject;folder.mkdir(parents=True)
                    wrist={c:(np.column_stack([x,np.zeros_like(x),np.zeros_like(x)]) if c=='ACC' else x[:,None]) for c,x in signals.items()}
                    path=folder/(subject+'.pkl')
                    with path.open('wb') as f:pickle.dump({'signal':{'wrist':wrist},'label':np.ones(80*700,int)},f)
                    hashes[str(path)]={'sha256':R.sha(path)}
                    block='S2_rest_0'
                else:
                    folder=phys/protocol/subject;folder.mkdir(parents=True)
                    for c,fs in [('BVP',64),('TEMP',4),('EDA',4),('ACC',32)]:
                        path=folder/(c+'.csv');x=signals[c]
                        if c=='ACC':table=np.vstack([[1000]*3,[fs]*3,np.column_stack([x,np.zeros_like(x),np.zeros_like(x)])])
                        else:table=np.r_[1000,fs,x]
                        pd.DataFrame(table).to_csv(path,index=False,header=False);hashes[str(path)]={'sha256':R.sha(path)}
                    path=folder/'tags.csv';pd.Series([1010,1020,1080] if protocol=='STRESS' else [1020,1080]).to_csv(path,index=False,header=False);hashes[str(path)]={'sha256':R.sha(path)}
                    block=f'{protocol}_{subject}_tags{tagi}-{tagj}'
                    man.append(dict(protocol=protocol,raw_folder=subject,subject_id=subject,protocol_version='v2',include=1,condition=condition,source_block='TMCT' if condition=='stress' else '70 rpm',start_tag_index=tagi,end_tag_index=tagj,start_s=start,end_s=end,analysis_start_s=start,analysis_end_s=end))
                wid=f'{dataset}:{block}:{int(start*64)}:{int(end*64)}';grid=start+np.arange(3840)/64
                v=np.zeros(len(names));v[[names.index(x) for x in optional]]=np.nan
                temp=np.interp(grid,np.arange(len(signals['TEMP']))/4,signals['TEMP']);v[names.index('temp_mean')]=temp.mean()
                pulse,_,_,_=R.pulse(signals['BVP'][int(start*64):int(end*64)]);v[names.index('bvp_mean_hr')]=pulse['hr_bpm']
                np.savez_compressed(data/file,X=v[None,:],feature_names=np.array(names),tier=tier,window_ids=np.array([wid]),groups=np.array([subject]),conditions=np.array([condition]))
                index.append(dict(dataset=dataset,protocol=protocol,subject_id=subject,condition=condition,window_id=wid,block_id=block,window_start_s=start,window_end_s=end))
            pd.DataFrame(index).to_csv(data/'window_index.csv',index=False);pd.DataFrame(man).to_csv(data/'physionet_interval_manifest.csv',index=False)
            R.writej(data/'raw_input_hashes.json',hashes);R.writej(run/'paths.json',dict(wesad=str(wesad),physionet=str(phys)))
            R.writej(run/'design.json',dict(discovery_subjects=[],confirmation_subjects=['f01']))
            R.writej(run/'provenance.json',dict(code_version='6.0.0',files={},feature_hashes={n:R.sha(data/n) for n in R.NPZS}))
            out=root/'audit';R.audit(run,out)
            self.assertEqual(R.readj(out/'audit_complete.json')['hard_errors'],[])
            q=pd.read_csv(out/'quality_queue_template.csv');self.assertEqual(set(q.channel),{'BVP','TEMP'})
            branch=R.prepare(run,out,'screen_sensitivity');self.assertFalse(R.readj(branch/'branch.json')['redevelop'])
            original=R.npz(data/R.NPZS[2]);changed=R.npz(branch/'data'/R.NPZS[2])
            self.assertTrue(np.isnan(changed['X'][0,names.index('temp_mean')]))
            self.assertEqual(original['X'][0,names.index('bvp_mean_hr')],changed['X'][0,names.index('bvp_mean_hr')])
            R.archive(out)
            import zipfile
            with zipfile.ZipFile(root/'audit_review.zip') as z:self.assertFalse(any(n.endswith('.npz') for n in z.namelist()))

    def test_frozen_integration(self):
        fixture=self.fixture
        if not fixture.exists():self.skipTest('Prior synthetic XGBoost fixture not available')
        cfg=R.readj(fixture/'models/development_lock.json')['configuration']
        with tempfile.TemporaryDirectory() as td:
            branch=Path(td);(branch/'data').mkdir();(branch/'models').mkdir();(branch/'results').mkdir()
            for n in R.NPZS:shutil.copy2(fixture/'data'/n,branch/'data'/n)
            a=R.npz(branch/'data'/R.NPZS[1]);i=np.flatnonzero(np.isin(a['groups'],cfg['confirmation_subjects']))[0]
            j=np.flatnonzero(a['feature_names']=='temp_mean')[0];a['X'][i,j]=np.nan
            np.savez_compressed(branch/'data'/R.NPZS[1],**a)
            R.evaluate_frozen(fixture,branch,cfg)
            R.explain_frozen(fixture,branch,cfg)
            p=pd.read_csv(branch/'models/target_predictions.csv')
            old=pd.read_csv(fixture/'models/target_predictions.csv')
            self.assertEqual(len(p),len(old));self.assertTrue(np.allclose(p.threshold,old.threshold))
            self.assertTrue((branch/'results/shap_recovery_audit.json').exists())

    def test_redevelopment_integration(self):
        fixture=self.fixture
        if not fixture.exists():self.skipTest('Prior synthetic XGBoost fixture not available')
        with tempfile.TemporaryDirectory() as td:
            origin=Path(td)/'origin';origin.mkdir()
            for folder in ['data','models','results']:shutil.copytree(fixture/folder,origin/folder)
            cfg=R.readj(origin/'models/development_lock.json')['configuration']
            R.writej(origin/'design.json',cfg)
            R.writej(origin/'provenance.json',dict(code_version='6.0.0',files={},feature_hashes={n:R.sha(origin/'data'/n) for n in R.NPZS}))
            branch=Path(td)/'branch';(branch/'data').mkdir(parents=True)
            for n in R.NPZS:shutil.copy2(origin/'data'/n,branch/'data'/n)
            a=R.npz(branch/'data'/R.NPZS[0]);j=np.flatnonzero(a['feature_names']=='temp_mean')[0];a['X'][0,j]=np.nan
            np.savez_compressed(branch/'data'/R.NPZS[0],**a)
            (branch/'frozen_quality_decisions.csv').write_text('synthetic_test_only\n')
            R.writej(branch/'design.json',cfg)
            R.writej(branch/'branch.json',dict(origin_run=str(origin),mode='screen_sensitivity',redevelop=True,
                feature_hashes={n:R.sha(branch/'data'/n) for n in R.NPZS},frozen_decisions_sha256=R.sha(branch/'frozen_quality_decisions.csv')))
            R.run_analysis(origin,branch)
            self.assertTrue((branch/'analysis_complete.json').exists())
            self.assertTrue((branch/'models/development_lock.json').exists())
            table=pd.read_csv(branch/'results/table_target_operating.csv')
            self.assertEqual(set(table.scope),{'primary_rest_stress'})


if __name__=='__main__':unittest.main(verbosity=2)
