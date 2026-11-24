"""Build and execute a report notebook for the GitHub diagnosis/RUL task."""
from pathlib import Path
import json
import os

import nbformat
import pandas as pd
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def rul_table(metrics, scope):
    chosen = metrics.loc[metrics.scope.eq(scope)]
    lines = ["| 고장 | 모델 | MAE(초) | RMSE(초) | R² | 저장소 정의 점수 | 평가 행 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in chosen.itertuples():
        lines.append(f"| {row.fault_name} | {row.model} | {row.mae_seconds:,.1f} | {row.rmse_seconds:,.1f} | "
                     f"{row.r2:.3f} | {row.source_score:,.0f} | {row.rows:,} |")
    return "\n".join(lines)


def main():
    classifier = json.loads((RESULTS / "classifier.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "rul_metrics.csv")
    main_rows = metrics.loc[metrics.model.eq("offline_isotonic_ensemble") & metrics.scope.eq("all_capped")]
    metadata = {mode: json.loads((HERE / "artifacts" / mode / "metadata.json").read_text(encoding="utf-8")) for mode in ["F1", "F2", "F3"]}
    short_lines = ["| 고장 | 앙상블 전체 MAE | 앙상블 고장 직전 MAE | HGBM 전체 MAE | 항상 상한 예측 MAE | 평가 사이클 |",
                   "|---|---:|---:|---:|---:|---:|"]
    for row in main_rows.itertuples():
        fault_rows = metrics.loc[metrics.fault.eq(row.fault) & metrics.scope.eq("all_capped")].set_index("model")
        near_rows = metrics.loc[metrics.fault.eq(row.fault) & metrics.scope.eq("near_failure_only") & metrics.model.eq("offline_isotonic_ensemble")]
        near_mae = f"{near_rows.mae_seconds.iloc[0]:,.1f}초" if len(near_rows) else "평가 행 없음"
        short_lines.append(f"| {row.fault_name} | {row.mae_seconds:,.1f}초 | {near_mae} | {fault_rows.loc['HGBM', 'mae_seconds']:,.1f}초 | "
                           f"{fault_rows.loc['always_cap', 'mae_seconds']:,.1f}초 | {metadata[row.fault]['validation_episodes']} |")
    short_table = "\n".join(short_lines)
    negative_r2 = main_rows.loc[main_rows.r2.lt(0), "fault_name"].tolist()
    rul_conclusion = ("이번 고정 설정에서 " + ", ".join(negative_r2) + "의 R²는 0보다 낮다. 이 항목은 평균 예측 대비 회귀 설명력을 확보하지 못했다."
                      if negative_r2 else "R²와 두 기준선, 고장 직전 구간의 오차를 함께 판단한다.")
    classification_rows = []
    for label in ["압력 과다", "누설", "압력 저하"]:
        row = classifier["classification_report"][label]
        classification_rows.append(f"| {label} | {row['precision']:.1%} | {row['recall']:.1%} | {row['f1-score']:.1%} | {int(row['support']):,} |")
    classification_table = "| 고장 | 정밀도 | 재현율 | F1 | 평가 행 |\n|---|---:|---:|---:|---:|\n" + "\n".join(classification_rows)
    report = f"""# clone — 고장 원인 분류 + 남은 시간(RUL) 예측 실행 결과

주제를 GitHub와 같이 변경했다. **24시간 안 고장 여부 분류가 아니라, 고장 유형과 다음 고장까지 남은 시간을 예측한다.**
공식 PHM2018 원시 센서·TTF를 사용해 M01 장비 10대를 140초로 집계했다.
원본 TensorFlow/Keras 구조와 HGBM·XGBoost 설정을 사용했으며 저장 모델 재로딩 검증을 수행했다.

## 1. 고장 종류 분류

- 모델: XGBoost + SMOTE.
- **Macro F1 {classifier['macro_f1']:.4f}, 정확도 {classifier['accuracy']:.2%}.**
- 학습 {classifier['train_rows']:,}행, 평가 {classifier['validation_rows']:,}행.
- 고장까지 **14,000초(약 3.9시간) 이내인 구간만** 대상으로 가장 가까운 고장 3종을 구분한다.
- 원본 V2처럼 장비+Lot을 Unit으로 두고 마지막 고장 사이클을 평가했다.
  이는 정상/위험 분류나 처음 보는 물리 장비 검증을 의미하지 않는다.

{classification_table}

## 2. 고장별 RUL 핵심 결과

원본 방식 = HGBM + GRU/LSTM 동적 앙상블 + 사이클 전체 isotonic 보정.
표의 RUL 정답은 압력 과다·누설 최대 14,000초, 압력 저하 최대 15,000초로 제한한 값이다.

{short_table}

항상 상한을 예측하는 기준선이 더 좋은 항목도 있을 수 있으므로, MAE 수치만으로 성공을 판단하지 않는다.
고장에 가까운 구간의 오차와 저장소 정의 점수도 함께 확인한다.

{rul_conclusion}

## 3. 모든 평가 행 — 전체 모델·기준선

{rul_table(metrics, 'all_capped')}

## 4. 실제 남은 시간이 상한보다 짧은 구간만

먼 고장을 모두 상한으로 뭉개는 효과를 제외한 하위 평가다. 모델을 이 구간에서 다시 학습하거나 선택하지 않았다.

{rul_table(metrics, 'near_failure_only')}

## 5. 해석 범위와 원본과의 차이

- 저자의 labeled master CSV는 공개되지 않아 공식 센서·TTF에서 재구성했다. 원본 공개 수치와 완전히
  같은 입력이라고 확인할 수 없다. 데이터 재구성·분할 내역과 실제 평가 행 수를 저장했다.
- 원본은 TTF 결측 배열에 argmin을 적용한다. 이번에는 결측을 무한대로 바꾼 뒤 가장 가까운
  **알려진 고장**을 골랐다. 입력에는 어떤 TTF 정답도 넣지 않았다.
- 원본처럼 미래 TTF 리셋으로 사이클을 구분하고, backward fill과 사이클 전체 isotonic을 적용했다.
  따라서 **오프라인 조건부 벤치마크**다. 실시간 고장 사전경보 성능으로 해석하지 않는다.
- `dynamic_ensemble`은 isotonic 전 결과다. 다만 이 값도 같은 오프라인 입력·사이클 분할을 사용했으므로
  독립적인 온라인 평가라고 부르지 않는다.
- `source_score`는 저장소 코드의 `sum(exp(-.001*y_true)*abs(y_true-y_pred))`다. 공식 리더보드와 동일하다고
  주장하지 않으며, 평가 행 수가 다르면 합계도 직접 비교하기 어렵다.
- 정상/위험 Z-score 게이트와 원인별 라우팅의 통합 성능, 새 장비 일반화 성능은 이번 결과에 포함되지 않는다.
- 이전 24시간 분류 F1, 이전 RUL 실험의 R²와 문제·입력·상한·평가 사이클이 달라 직접 우열을 비교할 수 없다.

## 6. 출처·재실행

원본: [ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance](https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance),
커밋 `6ba6b026340cd21a95f8846344c1996a02bcc97d` (MIT).

`run_github_rul.py`가 원시 집계·학습·평가를 수행한다. 가중치는 `artifacts/`, 점수는 `results/`에 있다.
학습에 사용한 TensorFlow는 {metadata['F3']['tensorflow']}이며, 각 모델의 실제 epoch 수는 metadata.json에 기록했다.
설정과 데이터 행 수는 `manifest.json`, `preparation.csv`, 각 `episodes.csv`를 확인한다.

```powershell
& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 -u models/clone/github_rul/run_github_rul.py --task all --epochs 20
& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 models/clone/github_rul/build_report.py
```
"""
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# clone — GitHub 방식: 고장 종류 + RUL 예측\n\n"
            "**주제 변경:** 냉각계통의 고장 유형을 구분하고 고장까지 남은 시간을 예측합니다. "
            "M01 10대, 원시 데이터 140초 집계, 마지막 고장 사이클 평가입니다. "
            "24시간 조기경보와 다른 문제입니다.\n\n"
            "[설정과 차이](README.md) · [상세 보고서](results/REPORT.md) · "
            "[원본 GitHub](https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance)"),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
            "from IPython.display import display\n"
            "root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'models/clone/github_rul/results/rul_metrics.csv').exists())\n"
            "clone = root / 'models/clone/github_rul'\n"
            "classifier = json.loads((clone / 'results/classifier.json').read_text(encoding='utf-8'))\n"
            "metrics = pd.read_csv(clone / 'results/rul_metrics.csv')\n"
            "plt.rcParams['font.family'] = 'Malgun Gothic'\nplt.rcParams['axes.unicode_minus'] = False\n"
            "display(pd.read_csv(clone / 'results/preparation.csv'))"),
        nbformat.v4.new_markdown_cell("## 1. 고장 유형 분류\n\n"
            f"XGBoost + SMOTE, **Macro F1 {classifier['macro_f1']:.4f}**, 정확도 {classifier['accuracy']:.2%}. "
            "**고장까지 약 3.9시간 이내인 구간의 조건부 원인 분류**이며 정상/위험 탐지 성능은 아닙니다."),
        nbformat.v4.new_code_cell("display(pd.DataFrame(classifier['classification_report']).T.round(4))\n"
            "matrix = pd.DataFrame(classifier['confusion_matrix'], index=['실제 과다', '실제 누설', '실제 저하'], columns=['예측 과다', '예측 누설', '예측 저하'])\n"
            "display(matrix)"),
        nbformat.v4.new_markdown_cell("## 2. RUL 결과 — 모든 평가 행\n\n"
            "정답은 14,000~15,000초로 상한을 적용했습니다. 항상 상한 예측과 학습 평균 예측도 비교합니다. "
            "원본 방식 isotonic은 사이클 전체 예측을 보는 **오프라인 보정**입니다."),
        nbformat.v4.new_code_cell("columns = ['fault_name', 'model', 'mae_seconds', 'rmse_seconds', 'r2', 'source_score', 'rows']\n"
            "display(metrics.loc[metrics.scope.eq('all_capped'), columns].round(3))"),
        nbformat.v4.new_markdown_cell("## 3. 고장 가까이에서의 RUL 결과\n\n실제 남은 시간이 상한보다 짧은 평가 구간만 따로 봅니다."),
        nbformat.v4.new_code_cell("display(metrics.loc[metrics.scope.eq('near_failure_only'), columns].round(3))"),
        nbformat.v4.new_markdown_cell("## 4. 고장별 실제·예측 추세\n\n각 고장의 평가 사이클 중 고장 직전 행이 가장 많은 사이클 하나를 표시합니다."),
        nbformat.v4.new_code_cell("figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)\n"
            "for axis, mode in zip(axes, ['F1', 'F2', 'F3']):\n"
            "    metadata = json.loads((clone / 'artifacts' / mode / 'metadata.json').read_text(encoding='utf-8'))\n"
            "    predictions = pd.read_csv(clone / 'artifacts' / mode / 'predictions.csv')\n"
            "    near = predictions[predictions[metadata['target']] < metadata['cap_seconds']]\n"
            "    if near.empty:\n"
            "        axis.text(.5, .5, '고장 직전 평가 행 없음', ha='center')\n"
            "        continue\n"
            "    episode = near.episode_id.value_counts().index[0]\n"
            "    selected = near[near.episode_id.eq(episode)].sort_values('time')\n"
            "    relative_time = -selected[metadata['target']] / 3600\n"
            "    axis.plot(relative_time, selected.target_capped / 3600, color='black', label='실제')\n"
            "    axis.plot(relative_time, selected.HGBM / 3600, label='HGBM', alpha=.7)\n"
            "    axis.plot(relative_time, selected.offline_isotonic_ensemble / 3600, label='앙상블 + 오프라인 보정')\n"
            "    axis.set(title=f'{mode}: {selected.PhysicalTool.iloc[0]}', xlabel='실제 고장까지 남은 시간(h, 음수)', ylabel='RUL(h)')\n"
            "    axis.legend(fontsize=8)\n"
            "figure.savefig(clone / 'results/rul_examples.png', dpi=150)\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5. DL 학습 이력"),
        nbformat.v4.new_code_cell("figure, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)\n"
            "for axis, mode in zip(axes, ['F1', 'F2', 'F3']):\n"
            "    history = pd.read_csv(clone / 'artifacts' / mode / 'training_history.csv')\n"
            "    axis.plot(history.epoch, history.loss, label='train')\n"
            "    axis.plot(history.epoch, history.val_loss, label='validation')\n"
            "    axis.set(title=mode, xlabel='epoch', ylabel='log1p RUL MSE')\n"
            "    axis.legend()\n"
            "figure.savefig(clone / 'results/training_curves.png', dpi=150)\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 해석 범위\n\n"
            "원본과 같은 오프라인 문제로 바꾼 실행입니다. 고장 사이클 구분·backward fill·사이클 전체 isotonic에는 "
            "미래 정보가 있으므로 실시간 사전경보 성능으로 해석하지 않습니다. 원본의 비공개 labeled CSV 대신 공식 원시 "
            "센서·TTF를 재구성했고, 결측 TTF를 잘못된 고장 유형으로 고르는 argmin 동작은 수정했습니다. "
            "입력에는 TTF를 넣지 않았습니다. [상세 결과와 차이](results/REPORT.md)를 함께 확인하세요."),
    ]
    # Keep IPython/Jupyter runtime writes inside this experiment directory.
    runtime = HERE / "cache/notebook_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter")
    path = HERE / "github_rul.ipynb"
    nbformat.write(notebook, path)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(notebook, path)
    print("Saved report and executed notebook:", path)


if __name__ == "__main__":
    main()
