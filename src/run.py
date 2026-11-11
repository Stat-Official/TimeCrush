from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from prepare import prepare, ROOT, TABLES, REPORTS, write_csv, dump
from models import OUTCOMES, logistic, coda, wquant
from structure import clusters, input_error, trend_bias

MODELS = ROOT/'outputs/models'


def summary_stats(d, line):
    activity_fields=['sleep','personal','work','commute','study','jobsearch','housework','care','discretionary']
    activities={}
    for key in activity_fields:
        if key in d:
            activities[key]={'mean':float(np.average(d[key],weights=d.weight)),
                             'median':float(wquant(d[key],d.weight,.5))}
    return {'diaries':len(d),'people':int(d.person.nunique()),'mean':float(np.average(d.discretionary,weights=d.weight)),
            'median':int(wquant(d.discretionary,d.weight,.5)),'poor_rate':float(np.average(d.discretionary.lt(line),weights=d.weight)),
            'activities':activities}


def benchmarks(raw,d,main):
    official={'수면':484,'일':187,'가정관리':94,'학습':49,'이동':92,'가족 및 가구원 돌보기':18,'여가시간':308,'식사하기':85}
    codes={'수면':'^11','일':'^2','가정관리':'^4','학습':'^3','이동':'^9','가족 및 가구원 돌보기':'^5','여가시간':'^[678]','식사하기':'^121$'}
    rows=[]
    for name, pattern in codes.items():
        values=raw[main.loc[main.code.str.contains(pattern),'source_column']].sum(axis=1)
        all_avg=np.average(values,weights=d.weight)
        mask=d.main_total.eq(1440)
        clean=np.average(values[mask],weights=d.weight[mask])
        rows.append((name,official[name],all_avg,clean,all_avg-official[name],'2024 국가데이터처 공표표 1-1' if name!='식사하기' else '국가데이터처 본문 및 KOSIS 통계찾기'))
    write_csv(pd.DataFrame(rows,columns=['activity','official_min','all_diaries_reproduced_min','complete_only_min','all_minus_official_min','source']),TABLES/'kosis_crosscheck.csv')
    response=[]
    public={'lack':21.4,'fatigue':24.5,'life_dissatisfaction':13.8,'leisure_dissatisfaction':19.1}
    for key,p in public.items():
        estimate=np.average(d['time_poor_outcome_'+key],weights=d.person_weight)*100
        response.append((key,p,estimate,estimate-p))
    write_csv(pd.DataFrame(response,columns=['indicator','public_percent','reproduced_percent','difference_pp']),TABLES/'response_code_crosscheck.csv')


def descriptive(raw, all_d, main, d):
    median=int(wquant(d.discretionary,d.weight,.5))
    line=.6*median
    d['poor']=d.discretionary.lt(line).astype(int)
    rng=np.random.default_rng(20241109)
    _, people=np.unique(d.person,return_inverse=True)
    G=people.max()+1
    rows=[]
    for proportion in [.5,.6,.7]:
        cutoff=proportion*median
        boot=[]
        for _ in range(500):
            counts=rng.multinomial(G,np.full(G,1/G))
            weight=d.weight.to_numpy()*counts[people]
            threshold=proportion*wquant(d.discretionary[weight>0],weight[weight>0],.5)
            boot.append(np.average(d.discretionary.lt(threshold),weights=weight))
        for subgroup,mask in [('all',np.ones(len(d),bool)),('below',d.discretionary.lt(cutoff)),('at_or_above',d.discretionary.ge(cutoff))]:
            sub=d[mask]
            rate=np.average(sub.discretionary.lt(cutoff),weights=sub.weight)
            ci=np.quantile(boot,[.025,.975]) if subgroup=='all' else [np.nan,np.nan]
            rows.append((proportion,cutoff,subgroup,len(sub),sub.person.nunique(),rate,*ci,
                         *[np.average(sub['time_poor_outcome_'+out],weights=sub.weight) for out in OUTCOMES]))
    sens=pd.DataFrame(rows,columns=['median_share','cutoff_min','group','diaries','persons','poverty_rate','bootstrap_ci_low','bootstrap_ci_high',*OUTCOMES])
    write_csv(sens,TABLES/'poverty_sensitivity.csv')
    alternate=[]
    for label,values,w in [('base',d.discretionary,d.weight),('unweighted',d.discretionary,np.ones(len(d))),
                           ('exclude_noncommute_travel_from_obligations',d.discretionary+d.without_noncommute_travel,d.weight),
                           ('volunteer_as_obligation',d.discretionary-d.other,d.weight),
                           ('published_leisure_no_leisure_travel',raw.loc[all_d.age.between(19,34)&all_d.weekday_type.eq(1)&all_d.main_total.eq(1440),
                             main.loc[main.code.str.startswith(('6','7','8')),'source_column']].sum(axis=1),d.weight)]:
        med=wquant(values,w,.5)
        alt_line=.6*med
        alternate.append((label,med,alt_line,np.average(values<alt_line,weights=w),np.average(values<138,weights=w)))
    write_csv(pd.DataFrame(alternate,columns=['definition','median_min','cutoff_min','rate_at_own_cutoff','rate_at_138_min']),TABLES/'poverty_definition_sensitivity.csv')
    employed=d[d.status.eq(1)].copy()
    employed['band']=np.select([employed.discretionary.lt(line),employed.discretionary.lt(median)],['below_cutoff','cutoff_to_median'],default='median_or_above')
    cross=[]
    for (band,child),sub in employed.groupby(['band','child']):
        # Percentile cluster bootstrap within a cell. Small cells are clearly flagged.
        _,inv=np.unique(sub.person,return_inverse=True)
        n=inv.max()+1
        boot=[]
        w=sub.weight.to_numpy()
        y=sub.time_poor_outcome_lack.to_numpy()
        for _ in range(500):
            counts=rng.multinomial(n,np.full(n,1/n))
            boot.append(np.average(y,weights=w*counts[inv]))
        cross.append((band,child,len(sub),n,np.average(y,weights=w),*np.quantile(boot,[.025,.975]),n<30))
    write_csv(pd.DataFrame(cross,columns=['band','preschool_child','diaries','persons','always_lack_rate','ci95_low','ci95_high','small_cell']),TABLES/'same_time_crosstab.csv')
    groups=[]
    for fields in [['age_group'],['status'],['child'],['status','child'],['age_group','status'],['age_group','status','child']]:
        for keys,sub in d.groupby(fields,observed=True):
            keys=keys if isinstance(keys,tuple) else (keys,)
            if sub.person.nunique()>=30:
                keys=[int(k) if isinstance(k,(float,np.number)) else k for k in keys]
                groups.append(dict(zip(fields,keys))|summary_stats(sub,line))
    free=d.discretionary.lt(line)
    sleep_short=d.sleep.lt(420)
    overwork=d.weekly.gt(52).fillna(False)
    signal_count=free.astype(int)+sleep_short.astype(int)+overwork.astype(int)
    signal_type=np.select(
        [signal_count.ge(2),overwork,sleep_short,free],
        ['complex','overwork','sleep','discretionary'],
        default='none'
    )
    signal_type_benchmarks={
        key:float(d.loc[signal_type==key,'weight'].sum()/d.weight.sum()*100)
        for key in ['complex','overwork','sleep','discretionary','none']
    }
    return line, {'benchmark':summary_stats(d,line),'cutoff':line,'groups':groups,
                  'sensitivity':[dict(r) for r in sens[sens.group.eq('all')].to_dict('records')],
                  'signal_type_benchmarks':signal_type_benchmarks}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('raw_csv', type=Path)
    parser.add_argument('--skip-quantile', action='store_true')
    args=parser.parse_args()
    for folder in [TABLES,MODELS,REPORTS,ROOT/'data/processed',ROOT/'outputs/figures']:
        folder.mkdir(parents=True,exist_ok=True)
    raw,d_all,main_map,d=prepare(args.raw_csv)
    benchmarks(raw,d_all,main_map)
    line, summary=descriptive(raw,d_all,main_map,d)
    print('Descriptive:',summary['benchmark'],'cutoff:',line,flush=True)
    summary['validation']=logistic(d,TABLES,line)
    print('CV:',summary['validation'],flush=True)
    coda(d,MODELS)
    print('CoDA done',flush=True)
    cluster,profile=clusters(d,MODELS/'cluster.json')
    print('Cluster:',{k:cluster[k] for k in ['k','median_ari','cross_method_ari','ilr']},flush=True)
    input_error(d,line)
    trend_bias(d)
    dump(summary,REPORTS/'results_summary.json')
    if not args.skip_quantile:
        subprocess.run([sys.executable,str(ROOT/'src/run_quantile.py')],check=True)


if __name__=='__main__':
    main()
