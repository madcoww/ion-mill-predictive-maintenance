"""Create a Korean report and execute a compact results notebook after training."""
from pathlib import Path
import json

import nbformat
import pandas as pd
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL_NAMES = {
    "baseline_lr13": "로지스틱 회귀 (13개, 재학습)",
    "clone_hgbm": "clone HGBM",
    "clone_xgb_smote": "clone XGBoost + SMOTE",
    "clone_gru": "clone GRU",
    "clone_hgbm_gru": "clone HGBM + GRU (1:1)",
}


def table_for(frame):
    lines = ["| 모델 | 정밀도 | 재현율 | F1 | 정확도 | ROC-AUC | PR-AUC | 경보일 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {MODEL_NAMES[row.model]} | {row.precision:.1%} | {row.recall:.1%} | {row.f1:.1%} | "
                     f"{row.accuracy:.1%} | {row.auc:.3f} | {row.pr_auc:.3f} | {row.alarm_days:,} |")
    return "\n".join(lines)


def main():
    summary = pd.read_csv(RESULTS / "summary.csv")
    selected = summary.loc[summary.lift_target.eq(4)]
    same = selected.loc[selected.validation.eq("same_tools")]
    unseen = selected.loc[selected.validation.eq("unseen_tools")]
    baseline = same.loc[same.model.eq("baseline_lr13")].iloc[0]
    clone = same.loc[~same.model.eq("baseline_lr13")].sort_values("f1", ascending=False).iloc[0]
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    report = f"""# clone — GitHub 모델의 24시간 조기경보 적용 결과

실제 실행 완료. 원본 ki-seoultech 모델의 구성과 특징 추출을 참고해 우리 목표로 바꾼 실험이다.
**원본 RUL 모델의 정확한 성능 재현이 아니다.** 원본/변경 사항은 [README](../README.md)에 기록했다.

## 핵심 결과

같은 장비의 미래 평가에서 재학습 LR의 F1은 **{baseline.f1:.1%}**, clone 중 최고 F1은
**{clone.f1:.1%} ({MODEL_NAMES[clone.model]})** 이다. 이번 적용에서는 LR이 우세하다.
처음 보는 장비 성능은 아래 합산 결과와 `fold_metrics.csv`를 함께 본다.

평가 **{int(baseline.days):,} 장비-일**, 양성 **{int(baseline.positive_days)}일 ({baseline.positive_days/baseline.days:.2%})**.
임계값은 보정 구간에서 정밀도 4배를 목표로 정했고, 평가 구간에 그대로 적용했다.
모든 모델에 같은 일별 정답·평가 행을 사용했다.

## 같은 장비의 미래 — 즉시 점검 기준

{table_for(same)}

## 처음 보는 장비 — 4묶음 합산, 즉시 점검 기준

{table_for(unseen)}

새 장비 5대씩 제외하고 15대로 학습·보정했다. 제외한 장비의 라벨을 전처리·임계값에 쓰지 않았다.
묶음별 임계값은 서로 다르다. 합산 ROC-AUC/PR-AUC는 서로 다른 모델 점수를 모은 참고 지표이며,
묶음별 점수는 `fold_metrics.csv`에 있다. 최종 혼동행렬은 각 묶음에서 고정한 임계값으로 합산했다.

## 해석

- LR은 센서 편차·시간 추세·과거 고장 로그를 사용한다. clone은 원본의 4센서 통계와 추가 센서를 사용한다.
  따라서 모델 알고리즘만의 통제 실험이 아니라 **입력 처리까지 포함한 두 방식의 비교**다.
- 원본의 짧은 RUL 문제에서 쓸 만한 복잡한 구조가, 넓은 24시간 경보에도 유리하다고 보장되지 않는다.
- GRU의 epoch별 학습·검증 손실은 `artifacts/<묶음>/gru_history.csv`에 보관했다.
  보정 기준을 만족하지 못해 임계값이 무한대이면 경보를 내지 않는다. 이때 F1=0은 모델 미실행이 아니다.
- 양성일 비율이 낮아 정확도는 모델 선택 지표로 쓰지 않는다. 항상 정상의 정확도는
  {1-baseline.positive_days/baseline.days:.2%}이며 재현율은 0이다.
- 사용자가 제시한 기존 실험(2,082일·136양성일)과 이번 표를 수치만으로 직접 비교하지 않는다.
  라벨을 구간 끝 이후의 고장으로 재계산했고, 분할 라벨 겹침·관측 마지막 하루를 제외했다.
- 장비-일 재현율을 고장 사건 재현율로 해석하면 안 된다. 하루 안에서 최대 점수와 양성 행이 다른
  시각일 수 있다. 실제 경보 선행시간·사건 탐지율은 별도의 시간 순서 평가가 필요하다.
- 이 결과는 1개 seed, 고정 하이퍼파라미터의 탐색 결과다. 최적화 가능성이나 모든 RUL 문제의 우열을 뜻하지 않는다.

## 실행 증거

- GPU: {manifest['device']}
- Python {manifest['python']}, scikit-learn {manifest['sklearn']}, PyTorch {manifest['torch']}
- 원본 커밋: `{manifest['source_commit']}` (MIT)
- 원본: {manifest['source']}
- 예측: `test_day_predictions.csv`, 분할: `split_boundaries.csv`, 환경·입력 해시: `manifest.json`
- 학습 코드: `../run_clone.py`, 저장 가중치: `../artifacts/`

## 재실행

```powershell
& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 -u models/clone/run_clone.py --folds all --epochs 20
& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 models/clone/build_report.py
```
"""
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# clone — GitHub 기반 24시간 조기경보 실험\n\n"
            "원본: [ki-seoultech](https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance). "
            "**RUL 원본의 정확한 재현이 아니라, 구조·특징 추출을 옮긴 24시간 이진 분류 실험**입니다.\n\n"
            "학습은 이미 실행해 가중치를 `artifacts/`에 저장했습니다. 이 노트북은 결과를 읽어 표·그림을 다시 만듭니다. "
            "전처리와 차이는 [README](README.md), 전체 결과는 [REPORT](results/REPORT.md)를 보세요."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\nfrom sklearn.metrics import precision_recall_curve\n"
            "from IPython.display import display\n"
            "root = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'models/clone/results/summary.csv').exists())\n"
            "clone = root / 'models/clone'\nsummary = pd.read_csv(clone / 'results/summary.csv')\n"
            "manifest = json.loads((clone / 'results/manifest.json').read_text(encoding='utf-8'))\n"
            "print('원본 커밋:', manifest['source_commit'])\nprint('실행 GPU:', manifest['device'])\n"
            "print('입력 구간:', f\"{manifest['rows']:,}\", '| 장비:', len(sum(manifest['unseen_tool_folds'], [])))"),
        nbformat.v4.new_markdown_cell("## 1. 같은 장비 / 처음 보는 장비 성능\n\n"
            "보정 구간에서 항상 경보 대비 정밀도 4배를 목표로 정한 '즉시 점검' 기준입니다. "
            "평가 구간에서 반드시 4배가 된다는 뜻은 아닙니다. Precision/Recall/F1은 장비-일 단위입니다."),
        nbformat.v4.new_code_cell("columns = ['validation', 'model', 'precision', 'recall', 'f1', 'accuracy', 'auc', 'pr_auc', 'days', 'positive_days', 'alarm_days']\n"
            "display(summary.loc[summary.lift_target.eq(4), columns].round(4))"),
        nbformat.v4.new_markdown_cell("## 2. 주의 / 즉시 점검 기준 비교"),
        nbformat.v4.new_code_cell("display(summary[['validation', 'model', 'lift_target', 'precision', 'recall', 'f1', 'fp', 'fn', 'tp', 'tn']].round(4))"),
        nbformat.v4.new_markdown_cell("## 3. Precision–recall 곡선\n\n"
            "같은 평가일의 곡선을 비교합니다. PR-AUC는 scikit-learn average precision 정의입니다. "
            "처음 보는 장비의 합산 곡선은 서로 다른 4개 모델 점수를 모은 참고치입니다."),
        nbformat.v4.new_code_cell("predictions = pd.read_csv(clone / 'results/test_day_predictions.csv')\n"
            "figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)\n"
            "for axis, name in zip(axes, ['same_tools', 'unseen_tools']):\n"
            "    selected = predictions[predictions.fold.eq('same_tools')] if name == 'same_tools' else predictions[predictions.fold.str.startswith('unseen_')]\n"
            "    for model, group in selected.groupby('model', sort=False):\n"
            "        precision, recall, thresholds = precision_recall_curve(group.label, group.score)\n"
            "        axis.plot(recall, precision, label=model)\n"
            "    base = selected[selected.model.eq('baseline_lr13')].label.mean()\n"
            "    axis.axhline(base, linestyle='--', color='grey', label=f'always alarm: {base:.3f}')\n"
            "    axis.set(title=name, xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))\n"
            "    axis.legend(fontsize=7)\n"
            "figure.savefig(clone / 'results/precision_recall.png', dpi=150)\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 4. GRU 학습 이력\n\n"
            "train loss가 줄어도 validation loss가 증가하면 과적합입니다. 선택은 학습 구간 내부의 검증 손실로만 했습니다."),
        nbformat.v4.new_code_cell("figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)\n"
            "for folder in sorted((clone / 'artifacts').iterdir()):\n"
            "    history = pd.read_csv(folder / 'gru_history.csv')\n"
            "    axis.plot(history.epoch, history.validation_loss, marker='o', label=folder.name)\n"
            "axis.set(xlabel='Epoch', ylabel='Validation weighted BCE', title='GRU early stopping')\n"
            "axis.legend()\nfigure.savefig(clone / 'results/gru_validation.png', dpi=150)\nplt.show()"),
        nbformat.v4.new_markdown_cell("## 5. 결론과 한계\n\n" + report.split("## 핵심 결과\n\n", 1)[1].split("## 같은 장비", 1)[0] +
            "\n\n이전 2,082일 실험과 달라진 라벨·분할 규칙 때문에, 이번에 다시 학습한 LR을 기준으로 비교합니다. "
            "고장 이력 입력을 사용하므로 신규 설치 직후처럼 이력이 전혀 없는 장비 성능은 아닙니다. "
            "고장 사건별 선행시간은 이 일별 점수와 별도로 검증해야 합니다."),
        nbformat.v4.new_markdown_cell("## 6. 전체 학습 재실행\n\n"
            "터미널에서 실행합니다. 이 명령은 현재 PC의 Python 경로입니다. 다른 환경에서는 경로를 바꾸세요.\n\n"
            "```powershell\n& 'C:/Users/rhkdd/anaconda3/python.exe' -X utf8 -u models/clone/run_clone.py --folds all --epochs 20\n```"),
    ]
    path = HERE / "clone_24h.ipynb"
    nbformat.write(notebook, path)
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(notebook, path)
    print("Saved", RESULTS / "REPORT.md", "and executed", path)


if __name__ == "__main__":
    main()
