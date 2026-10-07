"""Read-only numerical reconciliation of a complete-workflow review ZIP.
Standard library only. Never imports or executes code/models from the archive.
A pass means listed file/numerical checks passed, not scientific acceptance.
"""
from pathlib import Path, PurePosixPath
import argparse, collections, csv, hashlib, io, json, math, statistics, zipfile

VERSION='1.0.0'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def metrics(rows):
    if not rows:raise ValueError('Empty rest/stress comparison')
    tn=fp=fn=tp=0
    for r in rows:
        y=int(r['y_true']);h=int(r['y_pred']);p=float(r['y_prob']);t=float(r['threshold'])
        if y!=int(r['condition']=='stress'):raise ValueError('Outcome/condition mismatch')
        if not math.isfinite(p) or not 0<=p<=1 or not 0<=t<=1:raise ValueError('Invalid probability/threshold')
        if h!=int(p>=t):raise ValueError('Prediction does not match threshold')
        if y==0:tn+=int(h==0);fp+=int(h==1)
        else:fn+=int(h==0);tp+=int(h==1)
    if not (tp+fn and tn+fp):raise ValueError('Both rest and stress are required')
    def div(a,b):return a/b if b else math.nan
    # AUROC from average ranks: ties contribute one half.
    ordered=sorted(rows,key=lambda r:float(r['y_prob']));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and float(ordered[j]['y_prob'])==float(ordered[i]['y_prob']):j+=1
        rank_sum+=sum(int(r['y_true']) for r in ordered[i:j])*(i+1+j)/2
        i=j
    npos=tp+fn;nneg=tn+fp
    auc=(rank_sum-npos*(npos+1)/2)/(npos*nneg)
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,n_windows=len(rows),n_subjects=len({r['subject_id'] for r in rows}),
       macro_f1=(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2,
       sensitivity=tp/npos,specificity=tn/nneg,precision=div(tp,tp+fp),
       balanced_accuracy=(tp/npos+tn/nneg)/2,accuracy=(tp+tn)/len(rows),auroc=auc,
       brier=sum((float(r['y_prob'])-int(r['y_true']))**2 for r in rows)/len(rows))

class Review:
    def __init__(self,path):
        self.z=zipfile.ZipFile(path);infos=self.z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        for n in names:
            p=PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:raise ValueError('Unsafe archive member')
        if sum(i.file_size for i in infos)>2_000_000_000 or any(i.file_size>500_000_000 for i in infos):raise ValueError('Archive exceeds review-reader limits')
        self.names=set(names)
        if 'archive_hashes.json' not in self.names:
            raise ValueError('Expected the complete-workflow review ZIP from Cell 18, not an older v6/source-only ZIP')
        hashes=self.j('archive_hashes.json')
        if self.names!=set(hashes)|{'archive_hashes.json'}:raise ValueError('Archive member list differs from hash manifest')
        for n,h in hashes.items():
            actual=hashlib.sha256()
            with self.z.open(n) as f:
                for block in iter(lambda:f.read(1048576),b''):actual.update(block)
            if actual.hexdigest()!=h:raise ValueError('Archive hash mismatch: '+n)
    def text(self,n):
        if n not in self.names:raise ValueError('Required exported file missing: '+n)
        return self.z.read(n).decode('utf-8-sig')
    def j(self,n):return json.loads(self.text(n))
    def table(self,n):return list(csv.DictReader(io.StringIO(self.text(n))))
    def close(self):self.z.close()

def close(a,b,label,tol=1e-8,allow_undefined=False):
    if allow_undefined and math.isnan(float(a)) and (str(b).strip()=='' or math.isnan(float(b))):return
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol):
        raise ValueError('Numerical mismatch: '+label+f' ({a} versus {b})')

def run(path,out,allow_synthetic=False):
    path=Path(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review=Review(path)
    try:
        if 'SYNTHETIC_FIXTURE_ONLY.txt' in review.names and not allow_synthetic:
            raise ValueError('Synthetic fixture is not a real study review ZIP')
        cfg=review.j('design.json');labels=review.j('label_evidence.json');audit=review.j('quality/audit_complete.json')
        software=review.j('software_test_report.json');final=review.j('final_results.json')
        if audit['hard_errors']:raise ValueError('Raw audit records unresolved structural errors')
        if not labels.get('automated_official_mapping_check') or labels.get('human_review_attested'):
            raise ValueError('Unexpected label-check status; inspect the correct workflow output')
        if software.get('status')!='passed' or software.get('skipped')!=0:raise ValueError('Software gate did not pass completely')
        bname=PurePosixPath(review.j('quality_branch.json')['path']).name
        branch='quality/'+bname+'/'
        details=review.j(branch+'branch.json')
        if hashlib.sha256(review.z.read(branch+'branch.json')).hexdigest()!=review.j('quality_branch.json')['branch_sha256']:
            raise ValueError('Quality branch identity mismatch')
        if bool(details['redevelop'])!=bool(final['source_discovery_redeveloped']):raise ValueError('Redevelopment records disagree')
        lock=review.j('models/development_lock.json')
        qlock=review.j(branch+'models/development_lock.json') if details['redevelop'] else lock
        for n in [branch+'results/shap_condition_summary.csv',branch+'results/shap_condition_contrasts.csv',
                  branch+'results/shap_participant_consistency.csv',branch+'results/shap_local_examples.csv',
                  'results/quality_minus_reference_paired.csv','results/shap_imputation_counts.csv']:
            review.table(n)
        shapname=branch+'results/'+('shap_audit.json' if details['redevelop'] else 'shap_recovery_audit.json')
        shap=review.j(shapname)
        if details['redevelop'] and shap.get('status')!='passed':raise ValueError('SHAP audit failed')
        if not details['redevelop'] and not shap.get('no_model_refitting'):raise ValueError('Frozen SHAP status missing')
        summaries=[];feature_rows=[];decompositions=[];identities={}
        copies={};expected=set(cfg['confirmation_subjects']);all_metrics={}
        for analysis,prefix,this_lock in [('unscreened_reference','',lock),('quality_sensitivity',branch,qlock)]:
            pred=review.table(prefix+'models/target_predictions.csv')
            operating=review.table(prefix+'results/table_target_operating.csv')
            ci=review.table(prefix+'results/table_target_metrics.csv')
            rates=review.table(prefix+'results/table_condition_rates.csv')
            models=sorted({r['feature_set'] for r in pred})
            if len(pred)!=len({(r['feature_set'],r['window_id']) for r in pred}):raise ValueError('Duplicate prediction identity')
            required={'fts6','fts5','all_efs','accuracy6','target_importance6','logistic_all_efs','all_efs_no_mean_hr'}
            if not required.issubset(models):raise ValueError('Missing required comparator: '+str(required-set(models)))
            identities[analysis]={}
            by_rate={(r['feature_set'],r['metric']):r for r in rates}
            by_ci={(r['feature_set'],r['metric']):r for r in ci if r['scope']=='primary_rest_stress'}
            op={r['feature_set']:r for r in operating if r['scope']=='primary_rest_stress'}
            for name in models:
                d=[r for r in pred if r['feature_set']==name]
                if {r['subject_id'] for r in d}!=expected:raise ValueError('Evaluation participant membership differs from design')
                ids={r['window_id']:(r['subject_id'],r['condition'],r['y_true']) for r in d}
                identities[analysis][name]=ids
                for r in d:
                    close(r['threshold'],this_lock['models'][name]['threshold'],name+' threshold')
                    if int(r['y_pred'])!=int(float(r['y_prob'])>=float(r['threshold'])):raise ValueError('Exercise/rest/stress prediction mismatch')
                primary=[r for r in d if r['condition'] in ['rest','stress']]
                calc=metrics(primary)
                for metric,value in calc.items():close(value,op[name][metric],analysis+'/'+name+'/'+metric,allow_undefined=(metric=='precision'))
                for metric in ['macro_f1','sensitivity','specificity','auroc','brier']:
                    row=by_ci[(name,metric)];close(calc[metric],row['estimate'],'CI point estimate')
                    if not (math.isfinite(float(row['ci_low'])) and math.isfinite(float(row['ci_high'])) and float(row['ci_low'])<=float(row['ci_high'])):raise ValueError('Invalid exported CI bounds')
                    summaries.append(dict(analysis=analysis,feature_set=name,metric=metric,estimate=calc[metric],ci_low=row['ci_low'],ci_high=row['ci_high'],n_subjects=calc['n_subjects'],n_windows=calc['n_windows']))
                all_metrics[(analysis,name)]=calc
                per={}
                for condition in ['rest','exercise']:
                    groups=collections.defaultdict(list)
                    for r in d:
                        if r['condition']==condition:groups[r['subject_id']].append(int(r['y_pred']))
                    if set(groups)!=expected:raise ValueError('Missing paired rest/exercise participants')
                    per[condition]=statistics.mean(statistics.mean(v) for v in groups.values())
                    close(per[condition],by_rate[(name,condition)]['estimate'],'equal-participant '+condition)
                close(per['exercise']-per['rest'],by_rate[(name,'excess')]['estimate'],'exercise excess')
                feature_rows.append(dict(analysis=analysis,feature_set=name,features='; '.join(this_lock['models'][name]['features']),threshold=this_lock['models'][name]['threshold']))
            if any(v!=identities[analysis]['fts6'] for v in identities[analysis].values()):raise ValueError('Configurations have different evaluation windows')
            dcmp=review.table(prefix+'results/table_excess_decomposition.csv')
            grouped=collections.defaultdict(dict)
            for r in dcmp:grouped[(r['test'],r['reference'])][r['component']]=r
            for (test,reference),items in grouped.items():
                reduction=float(by_rate[(reference,'excess')]['estimate'])-float(by_rate[(test,'excess')]['estimate'])
                exercise=float(by_rate[(reference,'exercise')]['estimate'])-float(by_rate[(test,'exercise')]['estimate'])
                rest=float(by_rate[(test,'rest')]['estimate'])-float(by_rate[(reference,'rest')]['estimate'])
                for k,v in [('excess_reduction',reduction),('exercise_rate_decrease',exercise),('rest_false_positive_increase',rest)]:close(v,items[k]['estimate'],k)
                close(reduction,exercise+rest,'decomposition identity')
                decompositions.append(dict(analysis=analysis,test=test,reference=reference,exercise_rate_decrease=exercise,rest_positive_rate_increase=rest,excess_reduction=reduction,
                   rest_share_of_reduction=rest/reduction if reduction>0 else '',interpretation='equal participant weights; share is descriptive and may be negative or exceed one'))
            for name in ['table_target_operating.csv','table_target_metrics.csv','table_target_paired_differences.csv','table_condition_rates.csv',
                         'table_excess_decomposition.csv','table_source_feature_comparison.csv','shap_condition_summary.csv','shap_condition_contrasts.csv','shap_participant_consistency.csv']:
                copies[analysis+'__'+name]=review.z.read(prefix+'results/'+name)
        if identities['unscreened_reference']!=identities['quality_sensitivity']:raise ValueError('Original and quality analyses have different endpoint identities')
        f6=set(qlock['models']['fts6']['features']);f5=set(qlock['models']['fts5']['features'])
        removed=sorted(f6-f5)
        if len(f6)!=6 or len(f5)!=5 or not f5.issubset(f6):raise ValueError('Invalid FTS6/FTS5 membership')
        hr_only=(removed==['bvp_mean_hr'])
        def interval(analysis,name,metric):
            r=next(x for x in summaries if (x['analysis'],x['feature_set'],x['metric'])==(analysis,name,metric))
            return f"{r['estimate']:.3f} (95% marginal CI {float(r['ci_low']):.3f}–{float(r['ci_high']):.3f})"
        text=['# Results paragraphs generated from the supplied review ZIP','',
              'These are reconciled exported results. They do not constitute a complete manuscript or a publication decision.','',
              f"Archive SHA-256: `{digest(path)}`",'',
              '## Evaluation status','This is a retrospective, target-informed cross-dataset evaluation using previously inspected target participants. The unscreened reference and the automated quality sensitivity must remain identifiable throughout the manuscript.','',
              '## Performance']
        for analysis in ['unscreened_reference','quality_sensitivity']:
            m=all_metrics[(analysis,'fts6')]
            text.append(f"{analysis}: FTS6 was evaluated on {m['n_subjects']} target participants and {m['n_windows']} rest/stress windows. Macro-F1 was {interval(analysis,'fts6','macro_f1')}; AUROC was {interval(analysis,'fts6','auroc')}; sensitivity was {interval(analysis,'fts6','sensitivity')}; specificity was {interval(analysis,'fts6','specificity')}.\n")
        text+=['## Quality analysis',f"The automatic sensitivity changed {details['n_changed_channel_windows']} channel-windows. Full source/discovery redevelopment was required: {details['redevelop']}. Temperature and pulse-detection rules were heuristic; masking did not establish corrected physiology.",'',
               '## Feature removal',f"In the quality analysis, FTS5 removed: {', '.join(removed)}. FTS5 equals FTS6 without mean heart rate: {hr_only}. Use the explicitly named all_efs_no_mean_hr versus all_efs contrast for the direct full-set mean-heart-rate comparison.",'',
               '## Exercise and explanations','Use the accompanying equal-participant decomposition table; do not combine pooled specificity with participant-averaged excess. Exercise positive predictions are an operational challenge, not independent measurements proving absence of psychological stress. Use the copied SHAP condition contrasts and participant-consistency tables; retain raw-versus-imputed attribution distinctions. SHAP contribution patterns do not establish physiological causality.','',
               '## Required interpretation','Report numerical direction and uncertainty even if performance worsens. A failed equivalence test means equivalence was not demonstrated. Report no inherited permutation P values or claims of untouched confirmation unless their original protocol and chronology are independently established. This verifier checks exported point estimates and CI structure; it does not regenerate bootstrap intervals, validate raw physiological signals, or establish journal suitability.']
        def save(name,b):
            p=out/name
            if p.exists() and p.read_bytes()!=b:raise ValueError('Output already exists with different contents: '+str(p))
            if not p.exists():p.write_bytes(b)
        def savecsv(name,rows):
            s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);save(name,s.getvalue().encode())
        savecsv('RECONCILED_TARGET_METRICS.csv',summaries);savecsv('ACTUAL_FEATURE_SETS.csv',feature_rows);savecsv('EXERCISE_DECOMPOSITION_CHECKED.csv',decompositions)
        for name,b in copies.items():save(name,b)
        save('RESULTS_PARAGRAPHS.md','\n'.join(text).encode())
        report=dict(synthetic_test_fixture=('SYNTHETIC_FIXTURE_ONLY.txt' in review.names),status='listed_file_and_numerical_checks_passed',verifier_version=VERSION,archive_sha256=digest(path),
            target_participants=len(expected),quality_redevelopment=details['redevelop'],fts5_removed_features=removed,
            fts5_is_mean_hr_removal=hr_only,publication_ready_assessed=False,
            checked=['ZIP paths and hashes','recorded raw audit and software status','branch identity','locked thresholds','matching evaluation identities',
                     'confusion counts, macro-F1, sensitivity, specificity, precision, accuracy, balanced accuracy, AUROC and Brier score',
                     'point estimates and finite ordered CI bounds','equal-participant exercise/rest rates','excess decomposition','SHAP export presence and recorded audit status','actual feature membership'],
            not_checked=['raw signal validity','independent stress ground truth','recomputed bootstrap coverage','literature novelty','institutional ethics determination','coauthor approval','journal suitability'])
        save('VERIFICATION_REPORT.json',json.dumps(report,indent=2).encode())
        print(json.dumps(report,indent=2));print('FINISHING FILES:',out)
        return report
    finally:review.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    try:run(a.archive,a.output)
    except Exception as e:raise SystemExit('RECONCILIATION STOPPED: '+str(e))
