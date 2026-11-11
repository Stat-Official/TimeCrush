import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import helmert
from scipy.optimize import linprog, minimize
from scipy.special import expit
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from prepare import TABLES, write_csv, dump

SEED = 20241109
OUTCOMES = ['lack', 'fatigue', 'leisure_dissatisfaction', 'life_dissatisfaction']
PARTS = ['discretionary','sleep','personal','work','commute','study_job','house_care']


def wquant(x, w, q):
    order = np.argsort(x, kind='stable')
    y, weights = np.asarray(x)[order], np.asarray(w)[order]
    return y[np.searchsorted(np.cumsum(weights), q * weights.sum(), side='left').clip(0, len(y)-1)]


def indicators(d, employed=False, minimal=False):
    frame = pd.DataFrame({'intercept': np.ones(len(d)), 'age_per_5': (d.age-27)/5,
                          'female': d.sex.eq(2).astype(int), 'married': d.marital.eq(2).astype(int),
                          'marital_other': d.marital.isin([3,4]).astype(int), 'preschool_child': d.child,
                          'capital_region': d.cap_region}, index=d.index)
    if not employed and not minimal:
        frame['employed'] = d.status.eq(1).astype(int)
    if employed:
        frame['weekly_hours_per_10'] = (d.weekly-40)/10
        for j in range(2,7):
            frame[f'position_code_{j}'] = d.position.eq(j).astype(int)
    if minimal:
        return frame.drop(columns='marital_other')
    return frame


def fit_logit(X, y, w, group=None):
    y, w = np.asarray(y), np.asarray(w, float)
    w = w / w.mean()
    def objective(b):
        eta = X @ b
        return np.dot(w, np.logaddexp(0, eta) - y*eta), X.T @ (w*(expit(eta)-y))
    fit = minimize(objective, np.zeros(X.shape[1]), method='BFGS', jac=True, options={'gtol':1e-6})
    b = fit.x
    if not np.all(np.isfinite(b)) or np.max(np.abs(fit.jac)) > .01:
        raise RuntimeError(f'Logit did not converge: {fit.message}, {fit.jac}')
    p = expit(X @ b)
    if group is None:
        return b, None
    info = (X.T * (w*p*(1-p))) @ X
    bread = np.linalg.pinv(info)
    _, inverse = np.unique(group, return_inverse=True)
    G, n, k = inverse.max()+1, len(X), X.shape[1]
    sums = np.zeros((G,k))
    np.add.at(sums, inverse, X*(w*(y-p))[:,None])
    cov = (G/(G-1))*((n-1)/(n-k)) * bread @ (sums.T @ sums) @ bread
    return b, cov


def logistic(d, table_folder, threshold):
    result = []
    diagnostics = []
    for label, subset, employed in [('all', d, False), ('employed', d[d.status.eq(1)].copy(), True),
                                      ('employed_weekly_positive',d[d.status.eq(1)&d.weekly.gt(0)].copy(),True)]:
        subset = subset.dropna(subset=['weekly','position']) if employed else subset
        frame = indicators(subset, employed)
        X, y, w = frame.to_numpy(float), subset.poor.to_numpy(int), subset.weight.to_numpy(float)
        assert np.linalg.matrix_rank(X)==X.shape[1]
        b, V = fit_logit(X, y, w, subset.person)
        se = np.sqrt(np.diag(V))
        lo, hi = b - 1.96*se, b + 1.96*se
        for i, name in enumerate(frame.columns):
            result.append((label, name, b[i], se[i], np.exp(b[i]), np.exp(lo[i]), np.exp(hi[i]), 2*norm.sf(abs(b[i]/se[i])), len(subset), subset.person.nunique()))
        b_hh, cov_hh = fit_logit(X, y, w, subset.household)
        for i,name in enumerate(frame.columns):
            diagnostics.append((label, name, np.sqrt(cov_hh[i,i]), b_hh[i]))
    write_csv(pd.DataFrame(result, columns=['model','term','beta','se_person','odds_ratio','ci95_low_or','ci95_high_or','p','n_diaries','n_persons']), table_folder/'logistic.csv')
    write_csv(pd.DataFrame(diagnostics, columns=['model','term','se_household','beta']), table_folder/'logistic_household_sensitivity.csv')

    def model_selection(sample):
        frame = indicators(sample)
        full = frame.to_numpy(float)
        y = sample.poor.to_numpy(int)
        w = sample.weight.to_numpy(dtype=float, copy=True)
        w /= w.mean()
        masks = [np.ones(len(frame.columns), dtype=bool), ~frame.columns.isin(['capital_region','marital_other'])]
        rows=[]
        for name,mask in zip(['full','reduced_sensitivity'], masks):
            X = full[:,mask]
            b,V = fit_logit(X,y,w,sample.person)
            eta = X@b
            ll = np.dot(w,y*eta-np.logaddexp(0,eta))
            omitted = ~masks[1]
            wald = float(b[omitted]@np.linalg.pinv(V[np.ix_(omitted,omitted)])@b[omitted]) if name=='full' else np.nan
            rows.append((name,len(b),ll,2*len(b)-2*ll,np.log(len(y))*len(b)-2*ll,wald,chi2.sf(wald,omitted.sum())))
        write_csv(pd.DataFrame(rows,columns=['model','parameters','normalized_weight_log_likelihood','pseudo_aic','pseudo_bic','wald_drop_block','wald_p']),table_folder/'logistic_selection.csv')

    model_selection(d)
    X = indicators(d).to_numpy(float)
    y = d.poor.to_numpy(int)
    w = d.weight.to_numpy(float)
    p = np.empty(len(d))
    held = np.empty(len(d), int)
    folds = []
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(X,y,d.person), 1):
        assert not set(d.person.iloc[train]) & set(d.person.iloc[test])
        b,_ = fit_logit(X[train],y[train],w[train])
        p[test] = expit(X[test] @ b)
        held[test] = fold
        # The fixed operational threshold is the main estimand. Fold-specific cutoff is a sensitivity.
        fold_cutoff = .6*wquant(d.discretionary.iloc[train],w[train],.5)
        fold_y = d.discretionary.lt(fold_cutoff).to_numpy(int)
        b_train,_ = fit_logit(X[train],fold_y[train],w[train])
        folds.append((fold,len(test),d.person.iloc[test].nunique(),roc_auc_score(y[test],p[test],sample_weight=w[test]),
                      fold_cutoff,roc_auc_score(fold_y[test],expit(X[test]@b_train),sample_weight=w[test])))
    write_csv(pd.DataFrame(folds,columns=['fold','test_diaries','test_people','fixed_cutoff_auc','training_only_cutoff','training_only_cutoff_auc']),table_folder/'cv_folds.csv')
    fpr,tpr,cutoffs=roc_curve(y,p,sample_weight=w)
    write_csv(pd.DataFrame({'fpr':fpr,'tpr':tpr,'probability_cutoff':np.where(np.isfinite(cutoffs),cutoffs,np.nan)}),table_folder/'cv_roc.csv')
    calibration=[]
    bins = pd.qcut(p,10,labels=False,duplicates='drop')
    for j in np.unique(bins):
        mask=bins==j
        calibration.append((int(j)+1,int(mask.sum()),np.average(p[mask],weights=w[mask]),np.average(y[mask],weights=w[mask])))
    write_csv(pd.DataFrame(calibration,columns=['bin','diaries','mean_prediction','observed_rate']),table_folder/'cv_calibration.csv')
    calibration_beta,_ = fit_logit(np.column_stack([np.ones(len(p)), np.log(p/(1-p))]),y,w)
    return {'auc':roc_auc_score(y,p,sample_weight=w),'brier':brier_score_loss(y,p,sample_weight=w),
            'null_brier':np.average((y-np.average(y,weights=w))**2,weights=w),
            'calibration_intercept':float(calibration_beta[0]),'calibration_slope':float(calibration_beta[1]),
            'fold_auc': [row[3] for row in folds], 'training_only_auc': [row[5] for row in folds]}


def quantile_fit(X, y, w, q):
    valid = w>0
    Xv, yv, wv = X[valid], y[valid], w[valid]
    fit = linprog(-yv, A_eq=sparse.csc_matrix(Xv.T), b_eq=np.zeros(X.shape[1]),
                  bounds=np.column_stack([wv*(q-1), wv*q]), method='highs-ds', options={'threads':1})
    if not fit.success:
        raise RuntimeError(fit.message)
    return -fit.eqlin.marginals


def quantile(d, repetitions=300):
    Xf = indicators(d)
    X, y, w = Xf.to_numpy(float),d.discretionary.to_numpy(float),d.weight.to_numpy(dtype=float, copy=True)
    w /= w.mean()
    groups, inverse = np.unique(d.person,return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(groups), np.full(len(groups),1/len(groups)), size=repetitions)
    rows=[]
    for q in [.10,.25,.50,.75]:
        b = quantile_fit(X,y,w,q)
        simulated = np.asarray([quantile_fit(X,y,w*count[inverse],q) for count in draws])
        lo, hi = np.quantile(simulated,[.025,.975],axis=0)
        rows += [(q,name,b[i],simulated[:,i].std(ddof=1),lo[i],hi[i],repetitions) for i,name in enumerate(Xf.columns)]
        print(f'Quantile {q}: {repetitions} cluster bootstrap fits', flush=True)
    write_csv(pd.DataFrame(rows,columns=['quantile','term','beta_min','bootstrap_se','ci95_low_min','ci95_high_min','bootstrap_reps']), TABLES/'quantile.csv')


def replace_zero(x, delta):
    zeros = x == 0
    total = x.sum(axis=1, keepdims=True)
    assert (total > zeros.sum(axis=1,keepdims=True)*delta).all()
    return np.where(zeros,delta,x*(1-zeros.sum(axis=1,keepdims=True)*delta/total))


def ilr(x):
    assert np.min(x)>0
    return np.log(x) @ helmert(x.shape[1]).T


def coda(d, model_dir):
    x = d[PARTS].to_numpy(float)
    assert (x.sum(axis=1)==1440).all()
    covars = indicators(d,minimal=True).to_numpy(float)
    y = d[['time_poor_outcome_'+o for o in OUTCOMES]].to_numpy(int)
    w = d.weight.to_numpy(float)
    _, person = np.unique(d.person,return_inverse=True)
    patterns = np.unique(x>0,axis=0)
    zeros = [{'part':name,'zero_diaries':int((x[:,i]==0).sum()),
              'weighted_zero_share':np.average(x[:,i]==0,weights=w)} for i,name in enumerate(PARTS)]
    write_csv(pd.DataFrame(zeros),TABLES/'coda_zeros.csv')
    pattern_rows=[]
    rows=[]
    models=[]
    sensitivity=[('positive_support',None),('replace_1',1),('replace_3',3),('replace_6_5',6.5)]
    for method,delta in sensitivity:
        effects = {(j,k,mins):[0.,0,0.,0.,np.zeros(person.max()+1)] for j in range(3,7) for k in range(len(OUTCOMES)) for mins in [10,20,30,45,60]}
        strata = [(np.ones(len(d),bool),np.ones(len(PARTS),bool))] if delta is not None else []
        if delta is None:
            strata = [(np.all((x>0)==p,axis=1),p) for p in patterns]
        for mask, partmask in strata:
            n=int(mask.sum())
            actual = x[mask][:,partmask]
            composition = actual if delta is None else replace_zero(actual,delta)
            parts = np.flatnonzero(partmask)
            key = '+'.join(PARTS[i] for i in parts)
            if delta is None:
                pattern_rows.append((key,n,int(np.unique(person[mask]).size),float(w[mask].sum()/w.sum()), n>=100 and len(parts)>=2))
            if n<100 or len(parts)<2 or 0 not in parts:
                continue
            Z=ilr(composition)
            candidates=np.column_stack([covars[mask], Z])
            selected=[]
            for j in range(candidates.shape[1]):
                cols=candidates[:,selected+[j]]
                if np.linalg.matrix_rank(cols)==len(selected)+1:
                    selected.append(j)
            X=candidates[:,selected]
            for k,outcome in enumerate(OUTCOMES):
                yy=y[mask,k]
                if min(yy.sum(),n-yy.sum())<15:
                    continue
                try:
                    b,V=fit_logit(X,yy,w[mask],person[mask])
                except RuntimeError:
                    continue
                if np.max(np.abs(b))>20 or not np.all(np.isfinite(V)):
                    continue
                if delta is None:
                    models.append({'support':key,'outcome':outcome,'n':n,'basis':helmert(len(parts)).tolist(),'parts':[PARTS[j] for j in parts],
                                   'kept_columns':selected,'beta':b.tolist(),'se':np.sqrt(np.diag(V).clip(0)).tolist()})
                p0=expit(X@b)
                local_weights=w[mask]/w[mask].mean()
                bread=np.linalg.pinv((X.T * (local_weights*p0*(1-p0))) @ X)
                score=np.zeros((person.max()+1,X.shape[1]))
                np.add.at(score,person[mask],X*(local_weights*(yy-p0))[:,None])
                n_people=np.unique(person[mask]).size
                correction=np.sqrt(n_people/(n_people-1)*(n-1)/(n-X.shape[1]))
                influence=correction*(score@bread)
                for donor in range(3,7):
                    if donor not in parts:
                        continue
                    loc=list(parts).index(donor)
                    to=list(parts).index(0)
                    for mins in [10,20,30,45,60]:
                        # Strictly positive post-donation donor keeps the same support.
                        feasible=(actual[:,loc]>mins)&(actual[:,to]>0)
                        if not feasible.any():
                            continue
                        moved=actual[feasible].copy()
                        moved[:,loc]-=mins
                        moved[:,to]+=mins
                        changed=ilr(moved if delta is None else replace_zero(moved,delta))
                        changedX=np.column_stack([covars[mask][feasible],changed])[:,selected]
                        p1=expit(changedX@b)
                        orig=p0[feasible]
                        ww=w[mask][feasible]
                        acc=effects[donor,k,mins]
                        acc[0]+=float(np.dot(ww,p1-orig))
                        acc[1]+=int(feasible.sum())
                        acc[2]+=float(ww.sum())
                        acc[3]+=float(np.dot(ww,orig))
                        derivative=(ww*p1*(1-p1)) @ changedX - (ww*orig*(1-orig)) @ X[feasible]
                        acc[4]+=influence@derivative
        for (donor,k,mins),(diff,n,weights,base,influence_sum) in effects.items():
            feasible_all=(x[:,donor]>mins)&(x[:,0]>0)
            weights_all=w[feasible_all].sum()
            change=100*diff/weights if weights else np.nan
            se=100*np.linalg.norm(influence_sum)/weights if weights else np.nan
            rows.append((method,PARTS[donor],OUTCOMES[k],mins,n,int(feasible_all.sum()),
                         weights/weights_all if weights_all else np.nan,
                         100*base/weights if weights else np.nan,change,se,change-1.96*se,change+1.96*se))
    data=pd.DataFrame(rows,columns=['method','donor','outcome','moved_min','modeled_n','feasible_same_support_n','weighted_coverage','base_percent','delta_percentage_points','conditional_cluster_se_pp','conditional_ci95_low_pp','conditional_ci95_high_pp'])
    write_csv(data,TABLES/'coda_reallocation.csv')
    write_csv(pd.DataFrame(pattern_rows,columns=['support','diaries','persons','weight_share','passes_size_rule']),TABLES/'coda_patterns.csv')
    dump({'parts':PARTS,'covariate_order':list(indicators(d,minimal=True).columns),'models':models},model_dir/'coda_models.json')
    return data
