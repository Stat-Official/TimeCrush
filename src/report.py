import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare import ROOT, REPORTS, TABLES, mdtable

FIGURES=ROOT/'outputs/figures'
GREEN='#155d50'
AMBER='#c67b33'
OUT={'lack':'항상 시간부족','fatigue':'매우 피곤','leisure_dissatisfaction':'여가 불만족','life_dissatisfaction':'삶 불만족'}
DONOR={'work':'근로','commute':'통근','study_job':'학습·구직','house_care':'가사·돌봄'}
BAND={'below_cutoff':'138분 미만','cutoff_to_median':'138~229분','median_or_above':'230분 이상'}


def table(name):
    return pd.read_csv(TABLES/(name+'.csv'))


def pct(v):
    return f'{100*v:.2f}%'


def interval(lo,hi,digits=2):
    return f'{lo:.{digits}f} ~ {hi:.{digits}f}'


def figure(fig,name):
    fig.savefig(FIGURES/(name+'.png'),dpi=175,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)


def charts(sens, effects, trend):
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11,
                         'axes.titleweight':'bold','axes.grid':False,'figure.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    p=sens.poverty_rate.to_numpy()*100
    e=np.array([p-sens.bootstrap_ci_low*100,sens.bootstrap_ci_high*100-p])
    ax.bar([0,1,2],p,yerr=e,color=['#a3c6ba',GREEN,'#a3c6ba'],capsize=5,width=.54)
    for i,(v,y) in enumerate(zip(p,e[1])):
        ax.text(i,v+y+1.2,f'{v:.1f}%',ha='center',weight='bold')
    ax.set(xticks=[0,1,2],xticklabels=[f'{r.median_share:.0%} / {r.cutoff_min:.0f} min' for r in sens.itertuples()],
           ylim=(0,39),ylabel='Weighted share of weekday diaries (%)',title='Time poverty depends on the cutoff')
    ax.yaxis.grid(alpha=.2)
    figure(fig,'poverty_sensitivity')

    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    roc=table('cv_roc')
    cal=table('cv_calibration')
    for ax in axes:
        ax.plot([0,1],[0,1],ls='--',lw=1,color='#9aa5a3')
        ax.set(xlim=(0,1),ylim=(0,1),aspect='equal')
    axes[0].plot(roc.fpr,roc.tpr,color=GREEN,lw=2)
    axes[0].set(title='Out-of-fold ROC',xlabel='False positive rate',ylabel='True positive rate')
    axes[1].plot(cal.mean_prediction,cal.observed_rate,color=GREEN,lw=2,marker='o')
    axes[1].set(xlim=(0,.6),ylim=(0,.6),title='Weighted calibration',xlabel='Mean predicted rate',ylabel='Observed rate')
    figure(fig,'model_validation')

    labels={'work':'Work','commute':'Commute','study_job':'Study / job search','house_care':'Housework / care'}
    fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
    pick=effects[effects.outcome.eq('fatigue')].set_index('donor').loc[list(labels)]
    y=np.arange(len(pick))
    v=pick.delta_percentage_points
    ax.errorbar(v,y,xerr=[v-pick.conditional_ci95_low_pp,pick.conditional_ci95_high_pp-v],fmt='o',color=GREEN,capsize=4)
    ax.axvline(0,color='#8c9593',ls='--',lw=1)
    ax.set(yticks=y,yticklabels=list(labels.values()),xlabel='Change in predicted probability (percentage points)',
           title='Severe fatigue after a 30-minute reallocation')
    ax.invert_yaxis()
    figure(fig,'coda_fatigue')

    fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained')
    pick=trend.query("scenario=='medium_self_selection' and true_direction==1")
    stable=pick[pick.bias.eq('stable_bias')]
    drift=pick[pick.bias.eq('drifting_bias')]
    ax.plot(stable.wave+1,100*stable.true_rate,color=GREEN,lw=2.2,label='Population')
    ax.plot(stable.wave+1,100*stable.app_rate,color='#527c94',lw=2,label='App · stable bias')
    ax.plot(drift.wave+1,100*drift.app_rate,color=AMBER,lw=2,label='App · drifting bias')
    ax.legend(frameon=False)
    ax.set(xlabel='Simulated wave',ylabel='Time poverty rate (%)',title='Changing selection bias can reverse the trend')
    figure(fig,'selection_bias')


def build():
    FIGURES.mkdir(exist_ok=True)
    summary=json.loads((REPORTS/'results_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((REPORTS/'data_audit.json').read_text(encoding='utf-8'))
    b=summary['benchmark']
    v=summary['validation']
    sens=table('poverty_sensitivity')
    sens_all=sens[sens.group.eq('all')]
    sens_below=sens[sens.group.eq('below')]
    baseline=sens_all[sens_all.median_share.eq(.6)].iloc[0]
    alt=table('poverty_definition_sensitivity')
    log=table('logistic')
    qr=table('quantile')
    cross=table('same_time_crosstab')
    coda=table('coda_reallocation')
    effects=coda.query("method=='positive_support' and moved_min==30")
    profiles=table('clusters_profiles')
    stability=table('clusters_validation').query('best_in_method')
    error=table('input_error')
    trend=table('trend_bias')
    trajectory=table('trend_trajectories')
    benchmark=table('kosis_crosscheck')
    policy=pd.read_csv(ROOT/'data/external/policies.csv')
    charts(sens_all,effects,trajectory)

    by_threshold = []
    for overall, below in zip(sens_all.itertuples(),sens_below.itertuples()):
        by_threshold.append([f'{overall.median_share:.0%}',f'{overall.cutoff_min:.0f}분',pct(overall.poverty_rate),
            f'{pct(overall.bootstrap_ci_low)} ~ {pct(overall.bootstrap_ci_high)}',f'{below.diaries:,}',
            *[pct(getattr(below,c)) for c in OUT]])
    lg=[]
    terms={'female':'여성','preschool_child':'미취학 자녀 있음','employed':'취업',
           'age_per_5':'만 나이 5세 증가','capital_region':'수도권','weekly_hours_per_10':'주업·부업 합계 10시간 증가'}
    for r in log.itertuples():
        if (r.model=='all' and r.term in terms) or (r.model=='employed' and r.term=='weekly_hours_per_10'):
            lg.append([('전체' if r.model=='all' else '취업자'),terms[r.term],f'{r.beta:.3f}',f'{r.se_person:.3f}',
                       f'{r.odds_ratio:.2f}',interval(r.ci95_low_or,r.ci95_high_or),'<0.001' if r.p<.001 else f'{r.p:.3f}'])
    qrows=[[f'{r.quantile:.2f}',f'{r.beta_min:.1f}',interval(r.ci95_low_min,r.ci95_high_min,1)] for r in qr.query("term=='preschool_child'").itertuples()]
    high_child_people=int(cross.query("band=='median_or_above' and preschool_child==1").iloc[0].persons)
    crows=[[BAND[r.band], '있음' if r.preschool_child else '없음',r.diaries,r.persons,pct(r.always_lack_rate),
            f'{pct(r.ci95_low)} ~ {pct(r.ci95_high)}'] for r in cross.itertuples()]
    corows=[[DONOR[r.donor],OUT[r.outcome],r.modeled_n,pct(r.weighted_coverage),f'{r.delta_percentage_points:+.2f}',
             interval(r.conditional_ci95_low_pp,r.conditional_ci95_high_pp)] for r in effects.itertuples()]
    pr=[[r.name,f'{r.diaries:,}',pct(r.weighted_share),*(round(getattr(r,k)) for k in ['work','commute','study_job','housework','care','sleep'])]
        for r in profiles.query("method=='standardized'").itertuples()]
    support=int((policy.discovery_route=='온통청년').sum())
    def md(rows,cols):
        return mdtable(pd.DataFrame(rows,columns=cols))

    sections=f"""# 청년 시간 진단 연구 결과

2024 생활시간조사 공공용 시간량 자료 · 만 19–34세 · 평일 일지 · 2026-11-09 정리

## 표본과 재현 결과

원자료 {audit['rows']:,}행·{audit['columns']}열에서 청년 평일 일지 {b['diaries']:,}건({b['people']:,}명)을 분석했다. 청년 평일 주행동 합계 미달 {audit['incomplete_youth_weekday']}건을 제외했다. 재량시간 가중 평균은 {b['mean']:.2f}분, 중앙값은 **{b['median']}분**이다. 중앙값의 60%인 **{summary['cutoff']:.0f}분 미만**은 **{pct(b['poor_rate'])}**였다. 기획서의 230분·138분·22.5%는 반올림 범위에서 재현됐다. 개인 단위 재표집 95% 구간은 {pct(baseline.bootstrap_ci_low)}~{pct(baseline.bootstrap_ci_high)}다. 추정 단위는 일지이며 가중치가 적용됐다.

## [0] 시간빈곤 기준의 민감도

오른쪽 네 응답률은 각 기준선 미만 일지에서 계산했다. 기준선 이상 및 전체 결과도 CSV에서 볼 수 있다.

{md(by_threshold,['중앙값 비율','기준선','가중 빈곤율','95% 구간','미만 일지','항상 부족','매우 피곤','여가 불만족','삶 불만족'])}

60% 기준선 이상 일지에서는 항상 부족 24.44%, 매우 피곤 25.63%, 여가 불만족 16.90%, 삶 불만족 12.31%였다. 비통근 목적 이동을 필수·의무에서 제외한 대안의 중앙값은 {alt.iloc[2].median_min:.0f}분이며 그 정의의 기준선으로는 {pct(alt.iloc[2].rate_at_own_cutoff)}가 미만이다. 공표 여가 분류만 쓴 대안은 중앙값 {alt.iloc[4].median_min:.0f}분과 빈곤율 {pct(alt.iloc[4].rate_at_own_cutoff)}를 보였다. 표준 가중치 없이 계산한 비율은 {pct(alt.iloc[1].rate_at_own_cutoff)}였다.

![기준선 민감도](../figures/poverty_sensitivity.png)

## [1] 어떤 개인조건이 관련되는가

당일 활동시간을 제외하고 개인 특성만 투입했다. 취업자 모형에는 통상 주업·부업 시간과 종사상지위 코드도 포함했다. 아래 표의 모형별 전체 항목, 계수 및 오차는 CSV에서 확인할 수 있다.

{md(lg,['모형','조건','β','개인 군집 SE','오즈비','95% OR 구간','p'])}

개인 분리 5겹 교차검증 AUC는 **{v['auc']:.3f}**로 기획서 잠정 0.650보다 {0.65-v['auc']:.3f} 낮다. 가중 Brier 점수는 {v['brier']:.3f}, 전체 비율만 사용한 비교값은 {v['null_brier']:.3f}다. 보정 절편 {v['calibration_intercept']:.3f}, 기울기 {v['calibration_slope']:.3f}다. 학습 폴드에서 기준선을 다시 계산한 결과도 5회 모두 138분이었다. 앱에서는 개인조건 모형 확률을 노출하지 않는다. 재현 데이터 안에서만 검증했다.

![개인 그룹 교차검증](../figures/model_validation.png)

### [1-B] 가중 분위회귀

미취학 자녀가 있는 집단에서 다른 조건이 같을 때의 재량시간 조건부 분위 차이다. 개인 단위로 분위별 300회 재표집했다. 차이는 분석한 네 분위에서 모두 음수였으며 이 자료에서는 상위 분위의 점추정 차이가 더 컸다. 분위 간 차이에 대한 별도 공동 검정은 수행하지 않았다.

{md(qrows,['분위','미취학 자녀 계수(분)','95% 구간(분)'])}

## [2] 재배분 모형

30분을 재량시간으로 옮기는 시나리오다. 양수 행동 조합이 유지되는 일지에 한해 비교했다. 95% 구간은 해당 일지 구성을 고정하고 계수 추정의 불확실성을 개인 단위로 반영했다.

{md(corows,['줄인 활동','응답','모형 일지','대상 가중 포괄률','변화(%p)','조건부 95% 구간(%p)'])}

학습·구직 0분은 가중 {pct(table('coda_zeros').query("part=='study_job'").iloc[0].weighted_zero_share)}다. 따라서 기본 분석은 0분을 그대로 둔 채 같은 양수 구성 안에서 비교했다. 1·3·6.5분 대체 민감도에서 통근 30분 재배분의 매우 피곤 예측 변화는 방향이 바뀌었다. 시나리오별 결과는 모델링 가능한 일지의 평균 예측 차이이며 관찰자료의 관련성을 요약한다.

![피곤함 예측값의 재배분 비교](../figures/coda_fatigue.png)

## [3] 같은 재량시간 구간의 체감 차이

취업 청년의 항상 시간부족 응답률이다. 구간 내 재량시간, 가구 여건 및 다른 조건은 동일하다고 볼 수 없다. 특히 재량시간 중앙값 이상에서 미취학 자녀가 있는 집단은 {high_child_people}명이고 구간이 넓다.

{md(crows,['재량시간','미취학 자녀','일지','사람','항상 부족','95% 구간'])}

## [4] 시간사용 유형

시간빈곤 일지 1,095건을 분석했다. 표준화 원시간과 ilr 변환 각각에서 K=2–8을 비교했다. 표준화 원시간은 K={int(stability.iloc[0].k)}, 실루엣 {stability.iloc[0].silhouette:.3f}, 개인 재표집 중앙 ARI {stability.iloc[0].median_bootstrap_ari:.3f}였다. ilr은 K={int(stability.iloc[1].k)}, 실루엣 {stability.iloc[1].silhouette:.3f}, 중앙 ARI {stability.iloc[1].median_bootstrap_ari:.3f}였다. 앱에는 0분 관측을 유지하며 돌봄 중심 집단을 별도로 보여주는 표준화 모형을 적용했다. 두 방법에서 돌봄 유형의 분리는 달랐다.

{md(pr,['유형','일지','가중 비중','근로(분)','통근(분)','학습·구직(분)','가사(분)','돌봄(분)','수면(분)'])}

## 공식 공표값 교차검산

2024년 전국 10세 이상 전체 요일·성별의 전체 평균과 같은 조건에서 주행동 시간을 재현했다. 8개 지표의 최대 절대 차이는 {benchmark.all_minus_official_min.abs().max():.3f}분이었다. 식사하기는 KOSIS 통계찾기 표기 및 국가데이터처 본문과 함께 확인했다. 나머지 수치는 국가데이터처 공표표 1-1에 직접 대조했다. 공표 그룹과 연구의 19–34세 평일 그룹은 범위가 다르다.

## 앱 검증과 정책 환류

±20분 입력 오차 시뮬레이션에서 전체 판정의 {pct(error.iloc[1].overall_flip)}가 바뀌었고, 기준선 ±20분의 일지에서는 {pct(error.iloc[1].near_line_flip)}가 바뀌었다. 앱은 이 범위를 경계선으로 표시한다. Chrome에서 예시 입력, 합계 차단, 리포트, 유사 그룹, 재배분, 유형, 정책 카드, 선택 저장과 삭제를 확인했다.

운영기관 자료에서 근거를 확인한 정책·서비스는 {len(policy)}개이고 그중 {support}개는 온통청년에서 후보를 찾았다. 지역구분과 적용대상은 정책 DB에서 확인할 수 있다. 개별 모집 일정은 회차·지역·단지에 따라 달라지는 항목을 구분했다.

가상의 8개 시점과 8가지 선택편향 유형에서 가입자 구성의 변화를 모의실험했다. 편향이 일정한 설정에서는 추세 방향을 자주 재현했지만, 중간 이상의 자기선택 편향이 시간에 따라 이동하는 설정에서는 방향이 뒤집혔다. 앱 내 수요·반복 기록을 해석할 때 가입자 구성과 접속 조건을 함께 추적할 필요가 있다.

![편향 변화와 추세](../figures/selection_bias.png)

## 재현 파일

- [데이터 감사](data_audit.md), [변수 정의](../../docs/variable_mapping.md), [행동분류표](../../docs/activity_classification.csv)
- [방법론](../../docs/methodology.md), [정책 활용 및 한계](../../docs/limitations.md), [직주근접 지도 검토](../../docs/map_feasibility.md)
- [분석표 폴더](../tables/), [정책 DB CSV](../../data/external/policies.csv), [실행 안내](../../README.md)
"""
    (REPORTS/'research_results.md').write_text(sections,encoding='utf-8')


if __name__=='__main__':
    build()
