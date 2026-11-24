# 이온 밀 식각 장비 예지보전

PHM Society 2018 Data Challenge 데이터로 이온 밀 식각 장비의 헬륨 냉각(flowcool) 계통 고장 3종을 미리 예측하는 팀 프로젝트다.

- 데이터 설명: [data/README.md](data/README.md)
- EDA 결과: [eda/eda_step.ipynb](eda/eda_step.ipynb) (레시피 스텝 기준, 최신), [eda/eda_v1.ipynb](eda/eda_v1.ipynb) (1분 요약 기준). 실행 결과 포함

## 장비별 전체 기간 EDA

```bash
python -m src.eda.run --tool 01M01
```

선택 장비의 원본 센서·TTF·고장 파일 전체를 읽는다(샘플링 없음). 전체 운전 상태의 품질 검사와 스텝 집계를 수행하고, 식각 중(셔터 1) 센서 분포·전체 기간 추세·고장 전 48시간 궤적을 만든다.

- 결과: [01M01 보고서](eda/reports/01M01/summary.md), `eda/reports/<장비>/tables/`, `figures/`, `manifest.json`
- 스텝 경계: 레시피·스텝·가동 번호·단계·웨이퍼·셔터 변경 또는 60초 초과 공백. 집계 데이터는 `data/interim/eda/<장비>/`에 별도 저장한다.
- 같은 장비·고장 종류의 연속 기록을 24시간 간격 기준으로 묶고, 6·12·24·48시간 민감도도 저장한다.
- `--gap-seconds`, `--merge-hours`, `--window-hours`로 공백 기준·사건 묶기·고장 전 관측률 창을 조절한다. 궤적 그림은 48시간 고정이다.
- 입력 파일 정보·설정·코드 해시가 같고 결과가 있으면 재사용한다. 강제 재실행은 `--force`.
- 전체 기간 조건별 z값은 설명용이다. 모델 입력으로 쓰려면 학습 구간 또는 과거 관측만으로 기준을 다시 계산해야 한다.

## GitHub 참고 모델 실험 (clone)

현재 실험 주제는 **냉각계통 고장 유형 분류 + 단기 잔여수명(RUL) 예측**이다.

- [설정·실행 방법](models/clone/github_rul/README.md), [실행 결과 노트북](models/clone/github_rul/github_rul.ipynb), [성능 보고서](models/clone/github_rul/results/REPORT.md)
- M01 장비 10대의 원시 데이터를 140초로 집계하고, GitHub 참고 저장소의 고장 사이클 분할·모델 구성을 적용했다.
- 고장까지 약 3.9시간 이내인 구간의 XGBoost 원인 분류: Macro F1 **0.9539**, 정확도 **96.1%**.
- RUL 앙상블의 고장 직전 구간 MAE: 압력 과다 약 **91분**, 누설 **69분**, 압력 저하 **95분**. 이번 설정에서는 세 고장 모두 R²가 음수였다.
- 원본 방식의 미래 TTF 기반 사이클 구분·backward fill·전체 사이클 isotonic을 포함한 **오프라인 조건부 평가**다. 실시간 조기경보나 처음 보는 장비의 성능으로 해석하지 않는다.
- 학습 코드·실행 결과·검증 기록을 공유한다. 원시 데이터·집계 캐시·학습 가중치는 git에서 제외하며, 위 실행 방법으로 다시 생성한다.
- 이전 [24시간 조기경보 변형 실험](models/clone/clone_24h.ipynb)은 별도 기록으로 보관한다.

## 프로토타입

정비팀용 홈페이지 화면 시안 3종. 브라우저에서 바로 열린다. **데모 데이터**이며, 고장 기록 건수(1,236건, 반복 기록 묶기 전)만 실제 값이다.

👉 [시안 선택 화면](https://madcoww.github.io/ion-mill-predictive-maintenance/app/prototype/)

| 시안 | 구성 | 링크 |
| --- | --- | --- |
| v1 분석형 | 밝은 화면·상단 메뉴. 장비·레시피·스텝 필터, 경보 → 근거 → 점검 → 기록 흐름, 장비 × 14일 위험 히트맵 | [열기](https://madcoww.github.io/ion-mill-predictive-maintenance/app/prototype/v1.html) |
| v2 운영 센터형 | 어두운 화면·왼쪽 메뉴. 장비 상태·식각 운전 비율, 경보 추이, 고장 기록 구성, 추세 보기 | [열기](https://madcoww.github.io/ion-mill-predictive-maintenance/app/prototype/v2.html) |
| v3 모니터링형 | 장비 20대 타일과 위험 점수 선, 선택 장비의 냉각 압력 편차·정상 범위, 최근 경보 | [열기](https://madcoww.github.io/ion-mill-predictive-maintenance/app/prototype/v3.html) |

- 경보 등급: P2 우선 점검 / P3 관찰 강화 / 판단 보류 (P1은 장비 연동이 필요해 시연에서 제외)
- 장비 타일·표 행·히트맵 칸을 누르면 해당 장비 상세로 바뀐다. 메뉴는 아직 화면 전환이 없다.
- 소스: [app/prototype/](app/prototype/) (공통 데모 데이터 `common.js`)

## 폴더 구조

```
data/        데이터 설명 (원본 raw/는 git에 올리지 않음)
eda/         EDA 노트북 (eda_step.ipynb, eda_v1.ipynb)
src/         공통 코드 (전처리·피처·라벨링)
models/      학습·평가 코드와 결과 (모델 파일은 git 제외)
app/         대시보드 (dashboard.py·api.py), 홈페이지 시안 (prototype/)
docs/        회의록·계획·발표 자료
```

## 시작하기

1. 저장소 받기

   ```bash
   git clone <저장소 주소>
   cd ion-mill-predictive-maintenance
   ```

2. 환경 만들기 (Python 3.14에서 확인)

   ```bash
   pip install -r requirements.txt
   ```

3. 원본 데이터 받기 (집계 파일을 공유받는 팀원은 이 단계 대신 받은 파일을 `data/processed/`에 두면 된다) — [Google Drive 원본](https://drive.google.com/open?id=15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb)에서 `phm_data_challenge_2018.tar.gz`(5.36 GB)를 받아 아래 위치에 둔다. 압축은 풀지 않아도 된다. 스크립트가 압축 파일을 직접 순차로 읽는다.

   ```
   data/raw/phm2018/phm_data_challenge_2018.tar.gz
   ```

   압축을 풀면 `data/raw/phm_data_challenge_2018/{train,test}/`가 생긴다(약 35 GB).

4. 작은 파일로 줄이기 (장비 20대, 약 5분). 결과는 `data/processed/`에 생기며 git에는 올리지 않는다.

   ```bash
   python -X utf8 src/make_processed.py
   ```

   - `minute/<장비>.parquet`: 전체 기간을 1분 단위로 요약(센서별 평균·최소·최대, TTF)
   - `prefault/<장비>.parquet`: 고장 24시간 전 이내 구간의 원본 행(4초 해상도)
   - `step/M01_01.parquet`: 레시피 스텝 구간 하나를 한 행으로 요약(센서별 평균·최소·최대, TTF, 지속시간). 기준은 `recipe_step` 하나로, 값이 바뀔 때마다 새 행. 원본의 약 3%(244만 행)
   - `faults.csv`: 고장 기록 합본

   1분 요약은 한 분 안에서 스텝이 바뀌면 두 스텝의 센서 값이 섞인다. 스텝별로 정상값이 다르므로, 스텝을 기준으로 분석할 때는 `step/`을 쓴다.

## 작업 규칙

- **원본 데이터는 커밋하지 않는다.** 라이선스가 명시돼 있지 않아 재배포하지 않는다. `.gitignore`가 `data/raw/`를 막는다.
- `main`에 직접 푸시하지 않고 `이름/작업내용` 브랜치(예: `kim/labeling`)에서 작업한 뒤 PR로 합친다.
- 커밋 메시지는 한 줄 요약으로 쓴다. 예: `라벨링: 고장 기록 1일 간격 묶기 추가`
- 노트북은 실행 결과를 지운 상태가 아니라 **실행이 끝난 상태로** 커밋해, 받은 사람이 바로 결과를 볼 수 있게 한다. 같은 노트북을 두 명이 동시에 고치지 않는다.
- 100 MB 넘는 파일은 커밋하지 않는다(GitHub 제한). 큰 산출물은 경로와 만드는 방법만 README에 적는다.
- 성능 평가는 **장비 단위로 나눠서** 한다. 같은 장비가 학습·검증에 섞이면 점수가 부풀려진다([data/README.md](data/README.md) 4절).
