import json
import re
import sqlite3
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from prepare import ROOT, TABLES, REPORTS, PRIVATE
from models import fit_logit, ilr, indicators, quantile_fit, replace_zero, wquant


def csv(name):
    return pd.read_csv(TABLES/(name+'.csv'))


def validate():
    sample=pd.read_csv(PRIVATE/'youth_weekdays.csv')
    d=json.loads((REPORTS/'results_summary.json').read_text(encoding='utf-8'))
    assert sample.age.between(19,34).all() and sample.weekday_type.eq(1).all()
    assert len(sample)==d['benchmark']['diaries'] and sample.person.nunique()==d['benchmark']['people']
    assert sample.groupby('person').size().le(2).all()
    assert sample.main_total.eq(1440).all()
    assert (sample.discretionary+sample.required+sample.obligated==1440).all()
    assert sample.loc[sample.raw_status.eq(2),'status'].eq(1).all()
    assert sample.loc[sample.raw_status.eq(3),'status'].eq(3).all()
    w=sample.weight.to_numpy(float)
    y=sample.discretionary.lt(d['cutoff']).to_numpy(int)
    assert np.isclose(np.dot(w,y)/w.sum(),d['benchmark']['poor_rate'])
    assert wquant(sample.discretionary,w,.5)*.6==d['cutoff']

    design=indicators(sample).to_numpy(float)
    b,_=fit_logit(design,y,w,sample.person)
    independent=LogisticRegression(C=np.inf,solver='lbfgs',fit_intercept=False,max_iter=2000,tol=1e-10).fit(design,y,sample_weight=w/w.mean())
    assert np.max(np.abs(expit(design@b)-independent.predict_proba(design)[:,1]))<1e-5
    assert np.allclose(b,csv('logistic').query("model=='all'").beta.to_numpy(),atol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',OptimizeWarning)
        for q in [.10,.25,.50,.75]:
            single=quantile_fit(np.ones((len(w),1)),sample.discretionary.to_numpy(float),w/w.mean(),q)
            assert np.isclose(single[0],wquant(sample.discretionary,w,q))
    q=csv('quantile')
    assert set(q['quantile'])=={.1,.25,.5,.75} and q.bootstrap_reps.eq(300).all()
    c=np.array([[0.,20.,1420.],[30.,0.,1410.]])
    for delta in [1,3,6.5]:
        replaced=replace_zero(c,delta)
        assert (replaced>0).all() and np.allclose(replaced.sum(axis=1),1440)
        assert np.allclose(ilr(replaced),ilr(replaced/1440))
    scenarios=csv('coda_reallocation')
    assert (scenarios.modeled_n<=scenarios.feasible_same_support_n).all()
    assert scenarios.weighted_coverage.dropna().between(0,1+1e-10).all()
    assert (scenarios.conditional_ci95_low_pp<=scenarios.delta_percentage_points).all()
    assert (scenarios.conditional_ci95_high_pp>=scenarios.delta_percentage_points).all()
    for name in ['standardized','ilr']:
        checks=csv('clusters_validation').query('method==@name')
        assert set(checks.k)==set(range(2,9)) and checks.best_in_method.sum()==1
        assert checks.bootstraps.eq(80).all()

    assert csv('kosis_crosscheck').all_minus_official_min.abs().max()<.5
    assert csv('response_code_crosscheck').difference_pp.abs().max()<.06
    app=(ROOT/'app/index.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',app,re.S)[1])
    assert payload['cutoff']==d['cutoff']
    assert len(payload['profiles'])==payload['cluster']['k']
    assert all(row['people']>=30 for row in payload['groups'])
    assert all(group.get('status',1) in [1,3] for group in payload['groups'])
    assert not any(key in payload for key in ['raw','person','household'])
    assert len(re.findall(r'<section class="panel"',app))==6
    html_ids=re.findall(r'\bid="([a-zA-Z0-9]+)"',app)
    assert len(html_ids)==len(set(html_ids))
    with sqlite3.connect(ROOT/'data/external/policies.sqlite') as con:
        count=con.execute('SELECT COUNT(*) FROM policies').fetchone()[0]
    assert count==len(payload['policies'])==10
    (REPORTS/'validation.md').write_text("""# 재현 검증

- 청년 평일 선택, 일지·개인 수, 행동 합계 1,440분, 경제활동 분기, 가중 기준선과 비율을 확인했다.
- 자체 로지스틱 적합을 scikit-learn 비벌점 적합의 개별 예측과 비교했다. 최대 허용 절대 차이는 0.00001이다.
- 네 분위의 절편 전용 해를 가중 경험분위와 비교했다. 모형 출력에 분위별 개인 재표집 300회가 포함된다.
- ilr 척도 불변성, 0 대체 총합 보존, CoDA 분모·포괄률·구간, 두 군집 방법의 K=2–8 출력을 점검했다.
- 전체 10세 이상 행동시간 8개와 주관적 응답 4개를 국가데이터처 공표값에 각각 0.5분과 0.06%p 이내로 재현했다.
- 앱의 기준선·프로필·비교 집계 및 SQLite 정책 수와 내장 정책 수가 일치한다.
- Chrome에서 합계 입력 차단, 예시 리포트와 비교, CoDA 표시, 유형, 정책, 저장 동의 및 삭제를 확인했다.
""",encoding='utf-8')
    print('Analysis and application data validations passed')


if __name__=='__main__':
    validate()
