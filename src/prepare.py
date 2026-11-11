from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs/tables'
REPORTS = ROOT / 'outputs/reports'
DOCS = ROOT / 'docs'
PRIVATE = ROOT / 'data/processed'
KEY = ['회차', '가구일련번호', '가구원일련번호']
FIELDS = {
    'age': ('만연령', '만 나이', '19–34세 필터'),
    'sex': ('성별코드', '1 남성, 2 여성', '성별 공표 소계와 대조'),
    'marital': ('혼인상태코드', '1 미혼, 2 배우자 있음, 3·4 기타', '3·4 통합. 1·2는 연령 및 가구주 관계와 대조'),
    'region': ('행정구역시도코드', '관측 17개 시도', '11·23·31 수도권, 그 외 비수도권'),
    'day': ('요일구분코드', '0 일요일, 1–5 월–금, 6 토요일', '개인 내 일지 식별'),
    'weekday_type': ('평토일구분코드', '1 평일, 2 토요일, 3 일요일', '요일 코드와 전수 교차 확인'),
    'children': ('18세미만자녀수', '0–4명', '1명 이상 여부'),
    'preschool': ('미취학자녀수', '0–3명', '1명 이상. 미취학가구원수와 구분'),
    'enrollment': ('교육재학코드', '관측 1–5 및 결측', '코드 라벨 확인 전 모형 제외'),
    'raw_status': ('경제활동상태코드', '1 근로, 2 직업 있음·일시 이탈 응답, 3 일하지 않음', '1·2 모두 산업·종사상지위·주업 응답. 3은 무직사유만 응답'),
    'unemployed_reason': ('무직사유코드', '관측 1–8 및 비해당', '코드 라벨 확인 전 모형 제외'),
    'position': ('종사상지위코드', '관측 1–6 및 비해당', '취업자 전용 추가 모형의 코드별 범주'),
    'main_hours': ('주업시간수', '취업자 주업 시간(시간)', '부업시간수와 합산. 0은 원값 유지'),
    'side_hours': ('부업시간수', '취업자 부업 시간(시간)', '주업시간수와 합산'),
    'lack': ('시가간부족정도코드', '1 항상 부족, 2 가끔 부족, 3 약간 여유, 4 항상 여유', '1을 항상 시간부족으로 이진화. 공표 92쪽 역검증'),
    'fatigue': ('피곤함정도코드', '1 매우, 2 약간, 3 별로, 4 전혀', '1을 매우 피곤으로 이진화. 공표 93쪽 역검증'),
    'life_dissatisfaction': ('삶만족도코드', '1 매우 만족 ~ 5 매우 불만족', '4·5 불만족. 공표 94쪽 역검증'),
    'leisure_dissatisfaction': ('여가만족코드', '1 매우 만족 ~ 5 매우 불만족', '4·5 불만족. 공표 95쪽 역검증'),
    'weight': ('연간_시간량가중값', '양수 실수', '일지 단위 평균·회귀에 사용'),
    'person_weight': ('연간_가구원가중값', '양수 실수', '개인 응답 공표 대조용. 연구 일지 가중치와 구분'),
    'household_weight': ('연간_가구가중값', '양수 실수', '가구용. 연구 일지 가중치와 구분'),
}
LABEL = dict(sleep='수면·잠 못 이룸', personal='식사·개인관리', work='근로', commute='통근', study='학습', jobsearch='구직·창업', housework='가사', care='돌봄', leisure='자유·여가', other='기타 재량')
ACTIVITY = re.compile(r'^\((주행동|동시행동)시간량_(\d+)\)\s*(.*)$')
PDF = 'https://kostat.go.kr/boardDownload.es?bid=220&list_no=437764&seq=9'


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def dump(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def activity_group(code):
    if code.startswith('9'):
        return {'910': 'personal', '921': 'commute', '922': 'commute', '923': 'work', '924': 'work',
                '930': 'study', '940': 'housework', '951': 'care', '952': 'care', '953': 'care',
                '954': 'care', '960': 'other', '970': 'leisure', '980': 'leisure'}[code], '이동 목적과 같은 활동으로 분류'
    if code.startswith('11'):
        return 'sleep', '공표 중분류 11. 잠 못 이룸 포함'
    if code == '25':
        return 'jobsearch', '공개 파일의 구직·창업 합산 항목'
    group = {'1': 'personal', '2': 'work', '3': 'study', '4': 'housework', '5': 'care',
             '6': 'other', '7': 'leisure', '8': 'leisure'}[code[0]]
    reason = '공식 대분류의 의미를 따름'
    if code.startswith('6'):
        reason = '공표표 1-1과 같이 재량에 포함. 의무로 옮기는 민감도 별도'
    return group, reason


def prepare(path):
    raw = pd.read_csv(path, encoding='cp949', low_memory=False)
    assert raw.shape[1] == 304 and raw.columns.is_unique
    mapped = {k: v[0] for k, v in FIELDS.items()}
    derived = raw[list(mapped.values())].rename(columns={v: k for k, v in mapped.items()}).copy()
    acts = []
    for column in raw:
        match = ACTIVITY.match(column)
        if match is None:
            continue
        channel, code, label = match.groups()
        group, reason = activity_group(code)
        kind = '필수' if group in ['sleep', 'personal'] else '재량' if group in ['leisure', 'other'] else '의무'
        acts.append((channel, code, label, column, group, LABEL[group], kind, reason, channel == '주행동'))
    classification = pd.DataFrame(acts, columns=['channel', 'code', 'label', 'source_column', 'group', 'group_ko', 'time_class', 'rationale', 'used_1440'])
    main = classification[classification.used_1440]
    assert len(main) == 116 and len(classification) == 232
    assert set(main.code) == set(classification[~classification.used_1440].code)
    write_csv(classification, DOCS / 'activity_classification.csv')
    for group, rows in main.groupby('group'):
        derived[group] = raw[rows.source_column].sum(axis=1, skipna=False)
    derived['main_total'] = derived[list(LABEL)].sum(axis=1, skipna=False)
    derived['required'] = derived.sleep + derived.personal
    derived['obligated'] = derived[['work', 'commute', 'study', 'jobsearch', 'housework', 'care']].sum(axis=1)
    derived['discretionary'] = 1440 - derived.required - derived.obligated
    derived['study_job'] = derived.study + derived.jobsearch
    derived['house_care'] = derived.housework + derived.care
    derived['cap_region'] = derived.region.isin([11, 23, 31]).astype(int)
    derived['child'] = derived.preschool.gt(0).astype(int)
    derived['weekly'] = derived.main_hours + derived.side_hours
    derived['status'] = derived.raw_status.replace({2: 1})
    assert derived.loc[derived.raw_status.isin([1,2]), ['position','main_hours']].notna().all().all()
    assert derived.loc[derived.raw_status.eq(3), 'unemployed_reason'].notna().all()
    derived['without_noncommute_travel'] = raw[main.loc[main.code.isin(['910','923','924','930','940','951','952','953','954']), 'source_column']].sum(axis=1)
    ids = raw.groupby(KEY, dropna=False, sort=False).ngroup()
    household_ids = raw.groupby(KEY[:2], dropna=False, sort=False).ngroup()
    derived['person'] = ids
    derived['household'] = household_ids
    derived['diary_order'] = raw.groupby(KEY, sort=False).cumcount() + 1
    derived['time_poor_outcome_lack'] = derived.lack.eq(1).astype(int)
    derived['time_poor_outcome_fatigue'] = derived.fatigue.eq(1).astype(int)
    for result in ['life_dissatisfaction', 'leisure_dissatisfaction']:
        derived['time_poor_outcome_' + result] = derived[result].isin([4,5]).astype(int)
    assert derived[list(FIELDS)].notna().all()[['age','sex','marital','lack','fatigue','life_dissatisfaction','leisure_dissatisfaction']].all()
    assert np.array_equal(derived.weekday_type, np.select([derived.day.eq(6), derived.day.eq(0)], [2, 3], default=1))
    assert raw.duplicated(KEY + ['요일구분코드']).sum() == 0
    assert (raw.groupby(KEY).size() == 2).all()
    assert (derived.groupby('person').day.agg(lambda s: (s.max()-s.min()) in [1,6])).all()
    assert raw[main.source_column].ge(0).all().all() and (raw[main.source_column] % 10 == 0).all().all()
    assert derived.weight.gt(0).all() and np.isfinite(derived.weight).all()
    assert derived.preschool.le(derived.children).all()
    youth = derived.age.between(19, 34)
    weekday = derived.weekday_type.eq(1)
    complete = derived.main_total.eq(1440)
    sample = derived.loc[youth & weekday & complete].copy().reset_index(drop=True)
    assert sample.status.isin([1,3]).all()
    assert (sample.discretionary == sample.leisure + sample.other).all()
    sample['age_group'] = pd.cut(sample.age, [18,24,29,34], labels=['19–24','25–29','30–34']).astype(str)
    field_rows = [(key, name, values, action) for key, (name, values, action) in FIELDS.items()]
    field_rows += [
        ('person', ' + '.join(KEY), '개인별 2일', '복합키를 내부 정수 ID로 변환'),
        ('household', ' + '.join(KEY[:2]), '가구 식별', '복합키를 내부 정수 ID로 변환'),
        ('diary_order', '요일구분코드 + 원자료 행 순서', '각 개인 2건의 파일 내 순번', '조사표 차수 필드 없음. 분석은 순번에 의존하지 않음'),
        ('status', '경제활동상태코드', '1 취업(원 코드 1·2), 3 미취업', '구직 상태의 별도 직접 변수는 확인되지 않음'),
        ('weekly', '주업시간수 + 부업시간수', '시간, 원값 0 포함', '취업자 모형에서만 사용. 0 포함/제외 비교'),
        ('discretionary', '주행동 116열', '1,440 − 필수 − 의무', '완결 일지만 사용'),
    ]
    mapping = pd.DataFrame(field_rows, columns=['analysis_field','source_column','observed_values','transformation'])
    write_csv(mapping, DOCS / 'variable_mapping.csv')
    (DOCS/'variable_mapping.md').write_text('# 실제 변수 정의서\n\nMDIS 별도 다운로드 코드북 미확보. 실제 헤더, 코드값, 공표 통계표와 교차 확인한 정의이며 교육재학·무직사유·종사상지위의 세부 라벨은 추정하지 않았다.\n\n' + mdtable(mapping) + '\n', encoding='utf-8')
    audit = {
        'file': path.name, 'format': 'CSV', 'encoding': 'CP949', 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
        'rows': len(raw), 'columns': raw.shape[1], 'persons': int(ids.nunique()), 'households': int(household_ids.nunique()),
        'duplicate_rows': int(raw.duplicated().sum()), 'duplicate_person_weekday_code': int(raw.duplicated(KEY + ['요일구분코드']).sum()),
        'weekday_rows': int(weekday.sum()), 'youth_all_days': int(youth.sum()), 'youth_weekday_before_completeness': int((youth & weekday).sum()),
        'incomplete_rows': int((~complete).sum()), 'incomplete_youth_weekday': int((youth & weekday & ~complete).sum()),
        'valid_youth_weekday': len(sample), 'valid_youth_persons': int(sample.person.nunique()),
        'valid_youth_persons_diaries_count': {str(k):int(v) for k,v in sample.person.value_counts().value_counts().items()},
        'weight_min': float(derived.weight.min()), 'weight_max': float(derived.weight.max()), 'weight_sum_all': float(derived.weight.sum()),
        'weight_sum_sample': float(sample.weight.sum()), 'main_codes': len(main), 'simultaneous_codes': len(main),
        'activity_totals': {str(k):int(v) for k,v in derived.main_total.value_counts().items()},
        'column_inventory': [{'column':c,'dtype':str(raw[c].dtype),'missing':int(raw[c].isna().sum()),
                              'blank_whitespace':int(raw[c].str.fullmatch(r'\s*',na=False).sum()) if raw[c].dtype == object else 0,
                              'distinct':int(raw[c].nunique())} for c in raw],
        'observed_code_counts': {key:{str(k):int(v) for k,v in derived[key].value_counts(dropna=False).items()}
                                 for key in ['day','weekday_type','sex','marital','region','children','preschool','enrollment','raw_status','status','unemployed_reason','position','lack','fatigue','life_dissatisfaction','leisure_dissatisfaction']},
    }
    write_csv(pd.DataFrame(audit['column_inventory']), TABLES / 'audit_columns.csv')
    dump(audit, REPORTS / 'data_audit.json')
    (REPORTS/'data_audit.md').write_text(f"""# 데이터 감사

- {path.name} · CP949 · {len(raw):,}행 × {raw.shape[1]}열. 행 전체 중복 0건, 개인·요일 중복 0건.
- {audit['households']:,}가구, {audit['persons']:,}명 모두 일지 2개. 개인키는 회차·가구일련번호·가구원일련번호. 일지 키는 요일 추가.
- 전체 주행동 합계 미달 {audit['incomplete_rows']}건. 청년 평일 {audit['youth_weekday_before_completeness']:,}건 중 {audit['incomplete_youth_weekday']}건 제외. 최종 {len(sample):,}건, {audit['valid_youth_persons']:,}명.
- 주행동 116개 및 동시행동 116개 전수 대응. 동시행동은 분 합산에서 제외. 시간량 가중값은 전 건 양수.
- 전체 열 목록·형·결측은 [감사 표](../tables/audit_columns.csv), 코드별 빈도와 가중치 요약은 [감사 JSON](data_audit.json)에 수록.

| 데이터 | 확인 범위 | 필요한 컬럼과 상태 | 처리 |
|---|---|---|---|
| 시간량_전체 | 업로드 원본 | 개인·가구·요일·연령·미취학 자녀·주행동·시간량 가중값 있음 | 주 분석 |
| 가구정보 | 시간량 파일 대체 가능 | 미취학 자녀 및 미성년 자녀 직접 열 있음 | 별도 결합 불필요 |
| 공표표 및 부록 | 국가데이터처 PDF | 행동분류, 조사 개요, 주관 문항 공표 분포 확인 | 코드와 시간 검산 |
| MDIS 별도 코드북 | 미확보 | 교육재학·무직사유 등 라벨 직접 확인 대기 | 상세 라벨 모델에서 미사용 |
| KOSIS | 공표값과 공식 통계포털 조회 범위는 별도 표 참조 | 동일 정의의 전체 10세 이상 요일평균 | 청년 자체 기준과 구분 |
| 온통청년·운영기관 | 정책 DB 참조 | 정책별 자격·일정 | 후보와 공식 근거 분리 |
| LH·Kakao | 공식 메타데이터 및 명세 확인 | 주소·단지 정보·경로 키 | 타당성 문서 참조 |
""", encoding='utf-8')
    write_csv(sample, PRIVATE / 'youth_weekdays.csv')
    return raw, derived, main, sample


def mdtable(frame):
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|') for x in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_csv', type=Path)
    args = parser.parse_args()
    prepare(args.raw_csv)
