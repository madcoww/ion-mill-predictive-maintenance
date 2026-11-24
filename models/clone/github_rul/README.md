# clone — GitHub와 같은 고장 원인 분류 + RUL 예측

**변경한 주제:** PHM2018 냉각계통의 고장 유형을 구분하고, 각 유형의 다음 고장까지 남은 시간을 예측한다.
이 폴더는 이전 `../clone_24h.ipynb`의 24시간 이진 분류와 다른 실험이다.

## 따라 한 GitHub 방식

원본: [ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance](https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance),
커밋 `6ba6b026340cd21a95f8846344c1996a02bcc97d`, MIT 라이선스.
원본 저장소는 `../upstream/`, 실행에 필요한 수정하지 않은 원본 코드와 MIT 라이선스는 `reference/`에 있다.

| 항목 | 이번 실행 |
|---|---|
| 데이터 | 공식 PHM2018 학습 원시 CSV + 같은 행의 공식 TTF |
| 장비 | 원본과 같은 M01 10대 (`01_M01` ~ `10_M01`) |
| 운전 상태 | 셔터 1만 사용 |
| 집계 | 140초, 원본과 같은 고장 사이클 안에서 집계 |
| 원인 분류 | 가장 가까운 고장이 14,000초 이내인 구간의 고장 유형 3종 |
| 원인 분류 모델 | XGBoost 깊이 6, 학습률 .1, 150개 + SMOTE |
| 원인 분류 분할 | 원본 V2처럼 `장비+Lot`을 Unit으로 두고 마지막 고장 사이클을 평가 |
| RUL 라벨 | 다음 고장까지 남은 초, 압력 과다·누설 14,000초 / 압력 저하 15,000초로 상한 적용 |
| RUL 분할 | 물리 장비별 마지막 고장 사이클 평가, 이전 사이클 학습 |
| RUL ML | HistGradientBoostingRegressor, 학습률 .05, 최대 250개, log1p 라벨 |
| RUL DL | 원본과 같은 TensorFlow/Keras, GRU 64→32(과다·저하) / LSTM 64→32(누설), 15행 입력 |
| DL 학습 | dropout .2, Dense 16, Adam .001, MSE, 최대 20 epoch, patience 4, 학습 안쪽 뒤 20% 검증 |
| 결합 | 원본의 RUL 기반 동적 가중 평균, 사이클 전체 isotonic 결과도 별도 보고 |

원본의 `add_reference_features` 함수를 AST로 추출해 직접 실행한다.
따라서 Drive 연결이나 원본 전체 학습 스크립트의 부수 효과 없이 같은 계산을 사용한다.

## 명시한 차이

- 저자의 `*_labeled_Final.csv`와 두 master CSV는 공개되지 않았다. 공식 원시 센서·TTF에서 입력을 재구성했다.
- 원본 `argmin`은 TTF 결측이 있으면 결측인 유형을 고를 수 있다. 결측을 무한대로 마스킹해
  **실제 알려진 다음 고장 유형**을 고르도록 수정했다. 3종 정답은 항상 입력에서 제외한다.
- 큰 중첩 시퀀스 배열 대신 배치마다 같은 15행을 읽는다. TensorFlow/Keras 구조·손실·주요 설정은 유지한다.
- seed 42, CPU 6스레드로 고정했다. 전처리기와 장비별 scaler도 가중치와 함께 저장한다.
- 전체 파이프라인의 정상/위험 게이트 성능은 주장하지 않는다. **위험 구간의 원인 분류**와 **조건부 RUL 회귀**를 평가한다.

## 결과 해석

원본의 backward fill과 고장 사이클 전체 isotonic을 포함한 수치는 **오프라인 재현용**이다.
이는 미래 사이클 경계와 이후 예측을 알고 계산하는 부분이 있어, 그대로 실시간 조기경보 성능으로 해석하면 안 된다.
isotonic 전의 HGBM·GRU/LSTM·동적 앙상블 결과도 모두 저장한다.

RUL 상한 때문에 먼 고장 시점의 정답은 모두 약 4시간이다. 따라서 항상 상한을 예측하는 기준선과
학습 평균 기준선을 함께 평가하고, **실제 남은 시간이 상한보다 짧은 구간만의 성능**도 따로 보고한다.
`source_score`는 참고 저장소 코드의 `sum(exp(-.001*y)*abs(y-pred))`이며, 공식 리더보드 점수와 동일하다고 주장하지 않는다.

## 실행

```powershell
& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 -u models/clone/github_rul/run_github_rul.py --task all --epochs 20
```

데이터 집계는 캐시에 보관한다. 중간부터 실행하려면 `--task classifier`, `--task F1`, `--task F2`,
`--task F3`를 사용하고, 마지막에 `--task finish`를 실행한다.

## 파일

- `github_rul.ipynb`: 실행 결과가 포함된 열람용 노트북.
- `results/REPORT.md`: 결과·해석·기준선 비교.
- `results/classifier.json`: 조건부 고장 원인 분류 성능, 유형별 성능, 혼동행렬.
- `results/rul_metrics.csv`: 고장 3종의 RUL 성능과 기준선.
- `results/preparation.csv`: 원시 데이터·140초 집계 행 수.
- `results/manifest.json`: 출처·커밋·설정·원본과의 차이.
- `artifacts/<고장>/`: 실제 모델·전처리·학습 이력·예측·분할 목록.
- `cache/`: 원시 데이터로부터 만든 140초 집계 데이터. git 제외.
