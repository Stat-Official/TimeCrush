import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from models import SEED, ilr, replace_zero, wquant
from prepare import TABLES, dump, write_csv

CLUSTER_FIELDS = ['work','commute','study_job','housework','care','sleep']
TYPE_LABEL = dict(work='근로',commute='통근',study_job='학습·구직',housework='가사',care='돌봄',sleep='수면')


def clusters(d, model_path):
    d=d[d.poor.eq(1)]
    x=d[CLUSTER_FIELDS].to_numpy(float)
    w=d.weight.to_numpy(float)
    m=np.average(x,axis=0,weights=w)
    s=np.sqrt(np.average((x-m)**2,axis=0,weights=w))
    _, person=np.unique(d.person,return_inverse=True)
    rng=np.random.default_rng(SEED)
    npeople=person.max()+1
    draws=rng.multinomial(npeople,np.full(npeople,1/npeople),size=80)
    metrics=[]
    candidates={}
    for method in ['standardized','ilr']:
        z=(x-m)/s if method=='standardized' else ilr(replace_zero(x,3))
        for k in range(2,9):
            km=KMeans(k,n_init=25,random_state=SEED).fit(z,sample_weight=w)
            labels=km.labels_
            shares=np.bincount(labels,weights=w,minlength=k)/w.sum()
            aris=[]
            for i,draw in enumerate(draws):
                keep=draw[person]>0
                boot=KMeans(k,n_init=5,random_state=SEED+i).fit(z[keep],sample_weight=w[keep]*draw[person][keep])
                aris.append(adjusted_rand_score(labels,boot.predict(z)))
            sil=silhouette_score(z,labels)
            ari=np.median(aris)
            score=(sil+ari)/2
            counts=np.bincount(labels,minlength=k)
            metrics.append((method,k,sil,float(shares.min()),int(counts.min()),ari,np.quantile(aris,.1),len(aris),score))
            candidates[method,k]=(score,shares,km,counts)
    table=pd.DataFrame(metrics,columns=['method','k','silhouette','min_weight_share','min_diaries','median_bootstrap_ari','p10_bootstrap_ari','bootstraps','selection_score'])
    feasible=table[(table.min_weight_share>=.05)&(table.min_diaries>=40)]
    selected={}
    profiles=[]
    for method in ['standardized','ilr']:
        eligible=feasible[feasible.method.eq(method)]
        if eligible.empty:
            eligible=table[table.method.eq(method)]
        best=eligible.sort_values(['selection_score','k'],ascending=[False,True]).iloc[0]
        k=int(best.k)
        _,shares,km,counts=candidates[method,k]
        table.loc[best.name,'best_in_method']=True
        names=[]
        for j in range(k):
            mask=km.labels_==j
            p=np.average(x[mask],axis=0,weights=w[mask])
            strengths=(p-m)/s
            ordering=np.argsort(strengths)[::-1]
            name=TYPE_LABEL[CLUSTER_FIELDS[ordering[0]]]+' 시간 집중형'
            if name in names:
                name=TYPE_LABEL[CLUSTER_FIELDS[ordering[0]]]+'·'+TYPE_LABEL[CLUSTER_FIELDS[ordering[1]]]+' 시간 집중형'
            if name in names:
                name += f' ({j+1})'
            names.append(name)
            profiles.append((method,k,j,name,int(mask.sum()),int(d.person.iloc[np.flatnonzero(mask)].nunique()),shares[j],*p))
        selected[method]={'method':method,'k':k,'silhouette':float(best.silhouette),'median_ari':float(best.median_bootstrap_ari),
                          'centers':km.cluster_centers_.tolist(),'names':names}
    a=candidates['standardized',selected['standardized']['k']][2]
    b=candidates['ilr',selected['ilr']['k']][2]
    sensitivity=[]
    for delta in [1,3,6.5]:
        z=ilr(replace_zero(x,delta))
        zfit=KMeans(b.n_clusters,n_init=25,random_state=SEED).fit(z,sample_weight=w)
        sensitivity.append((delta,adjusted_rand_score(b.labels_,zfit.labels_)))
    write_csv(pd.DataFrame(sensitivity,columns=['zero_delta_minutes','ari_vs_3_minutes']),TABLES/'clusters_ilr_zero_sensitivity.csv')
    table['best_in_method']=table.best_in_method.eq(True)
    write_csv(table,TABLES/'clusters_validation.csv')
    profile=pd.DataFrame(profiles,columns=['method','k','cluster','name','diaries','persons','weighted_share',*CLUSTER_FIELDS])
    write_csv(profile,TABLES/'clusters_profiles.csv')
    model=dict(selected['standardized'])
    model.update({'fields':CLUSTER_FIELDS,'means':m.tolist(),'scales':s.tolist(),
                  'cross_method_ari':adjusted_rand_score(a.labels_,b.labels_),
                  'ilr':selected['ilr']})
    dump(model,model_path)
    return model,profile


def input_error(d, line):
    rng=np.random.default_rng(SEED+1)
    x=d[['discretionary','sleep','personal','work','commute','study_job','house_care']].to_numpy(int)
    w=d.weight.to_numpy(float)
    original=x[:,0]<line
    rows=[]
    for width in [10,20,30]:
        numerator=[]
        local=[]
        near=np.abs(x[:,0]-line)<=width
        for _ in range(300):
            eps=rng.integers(-width,width+1,size=len(d))
            source=rng.integers(1,x.shape[1],size=len(d))
            eps=np.clip(eps,-x[:,0],x[np.arange(len(d)),source])
            flip=((x[:,0]+eps)<line)!=original
            numerator.append(np.average(flip,weights=w))
            local.append(np.average(flip[near],weights=w[near]))
        rows.append((width,np.mean(numerator),np.mean(local),np.average(near,weights=w),300))
    write_csv(pd.DataFrame(rows,columns=['maximum_minutes','overall_flip','near_line_flip','near_line_share','repetitions']),TABLES/'input_error.csv')


def trend_bias(d):
    w=d.weight.to_numpy(float)
    y=d.poor.to_numpy(int)
    # The original weighted empirical support is reused with a controlled odds shift.
    scenarios={'weak_self_selection':(y,1.25),'medium_self_selection':(y,2.0),'strong_self_selection':(y,4.0),
               'women_overrepresented':(d.sex.eq(2).to_numpy(),2.5),'men_overrepresented':(d.sex.eq(1).to_numpy(),2.5),
               'employed_overrepresented':(d.status.eq(1).to_numpy(),2.5),'observed_jobsearch_overrepresented':((d.status.eq(3)&d.jobsearch.gt(0)).to_numpy(),8.0),
               'capital_overrepresented':(d.cap_region.to_numpy(),2.5)}
    rng=np.random.default_rng(SEED+2)
    trajectories=[]
    results=[]
    for name,(selector,ratio) in scenarios.items():
        for mode in ['stable_bias','drifting_bias']:
            for trend in [-1,1]:
                true,expected,draws=[],[],[]
                for t in range(8):
                    population=w*np.exp(y*trend*.045*t)
                    true.append(np.average(y,weights=population))
                    intensity=ratio if mode=='stable_bias' else ratio**(t/7 if trend<0 else 1-t/7)
                    prop=population*intensity**selector
                    expected.append(np.average(y,weights=prop))
                    draws.append(rng.binomial(800,expected[-1],size=400)/800)
                    trajectories.append((name,mode,trend,t,true[-1],expected[-1]))
                draws=np.array(draws).T
                slopes=(draws @ (np.arange(8)-3.5))/42
                results.append((name,mode,trend,100*(true[-1]-true[0]),100*(expected[-1]-expected[0]),
                                np.mean(np.sign(slopes)==trend),100*np.mean(np.array(expected)-true),400,800))
    write_csv(pd.DataFrame(results,columns=['scenario','bias','true_direction','true_delta_pp','expected_app_delta_pp','direction_accuracy','mean_level_bias_pp','repetitions','users_per_wave']),TABLES/'trend_bias.csv')
    write_csv(pd.DataFrame(trajectories,columns=['scenario','bias','true_direction','wave','true_rate','app_rate']),TABLES/'trend_trajectories.csv')
