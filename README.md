# 하루의 여백 · 청년 시간 진단

2024 생활시간조사 MDIS 공공용 시간량 자료에서 만 19–34세 청년 평일 일지의 재량시간을 계산하고, 비교·재배분·정책 탐색에 쓰는 연구 프로토타입이다.

## 바로 실행

1. 압축을 풀고 `app/index.html`을 브라우저에서 연다. **입력 예시 채우기**로 6개 화면을 체험할 수 있다.
2. 브라우저가 파일 주소의 저장을 제한하면 프로젝트 폴더에서 `python -m http.server 8000`을 실행하고 `http://localhost:8000/app/`을 연다.
3. [연구 결과](outputs/reports/research_results.md)에서 가중 결과, 민감도, 표본수와 그림을 확인한다.

## 원자료로 전체 재실행

Python 3.11 이상이 필요하다. 가중 분위회귀 및 군집 재표집에는 수십 초~수분이 걸릴 수 있다.

```bash
python -m pip install -r requirements.txt
python src/run.py "경로/2024_시간량_전체_20261103_44315.csv"
python src/build_app.py
python src/report.py
python src/validate.py
```

업로드한 원본에서 실제 열과 행동 매핑을 확인한 후 분석용 일지를 `data/processed/youth_weekdays.csv`로 생성한다. 원자료와 일지 단위 파생 파일은 압축 및 Git에서 제외했다. 집계 결과·정책 DB·모형·앱은 압축에 포함했다. 재표집 시드는 고정되어 있다.

## 결과와 파일

| 번호 | 분석 | 주요 결과표 |
|---|---|---|
| [0] | 가중 중앙값, 기준선, 50·60·70% 민감도 | poverty_sensitivity.csv |
| [1] | 개인조건 가중 로지스틱, ROC·보정 | logistic.csv, cv_*.csv |
| [1-B] | 4개 가중 분위, 개인 재표집 | quantile.csv |
| [2] | 시간구성, 0분 가정, 재배분 | coda_*.csv |
| [3] | 취업 청년 재량 구간별 시간부족 | same_time_crosstab.csv |
| [4] | 표준화·ilr 군집 및 안정성 | clusters_*.csv |

- `docs/`에는 [실제 변수 정의](docs/variable_mapping.md), [행동분류](docs/activity_classification.csv), [방법론](docs/methodology.md), [정책 활용 및 한계](docs/limitations.md), [지도 검토](docs/map_feasibility.md), [자료 출처](docs/data_sources.md)가 있다.
- `src/`의 `prepare.py`는 감사·전처리, `models.py`는 회귀·CoDA, `structure.py`는 군집·오차·편향을 담당한다. `run.py`가 분석을 실행하고 `build_app.py`, `report.py`, `validate.py`가 결과물과 검증표를 만든다.
- `outputs/`에는 [감사](outputs/reports/data_audit.md), [연구 결과](outputs/reports/research_results.md), 결과 CSV, 그림, 집계 기반 모형이 있다.
- `data/external/`에 정책 CSV·JSON·SQLite와 향후 이용기록 스키마가 있다. 정책 안내 확인일은 2026-11-09다.
- `app/`에 HTML 템플릿과 실행 가능한 단일 HTML이 있다. 내장 데이터는 실제 조사 집계와 모형 요약이며, 예시 버튼의 하루는 합성 입력이다.

## 주요 결과와 범위

청년 평일 4,920일지·3,255명에서 가중 재량시간 중앙값은 230분, 기준선은 138분이었다. 기준선 미만 일지 비율은 22.51%였다. 개인 그룹 분리 교차검증에서 조건 기반 모형의 분류력은 제한적이었다. 앱 진단은 입력한 활동시간으로 계산한다.

공식 교차검산은 KOSIS 식사하기 항목과 국가데이터처의 전체 요일 공표표에 대조했다. 2024 MDIS 별도 상세 코드북과 직접적인 구직 상태는 미확보다. 직주근접 지도는 설계 범위이며 Kakao 실조회에는 API 키가 필요하다. 앱 정책 안내는 제도 탐색용이며 이용기록은 동의 시 브라우저에 보관된다.
