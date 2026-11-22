"""대시보드 프로토타입용 경보 점수를 만든다.

모델: 로지스틱 회귀 + 클래스 가중치 (지도 학습)
정답(라벨): 구간 끝에서 24시간 안에 압력 저하(FlowCool Pressure Dropped Below Limit)가 오면 1
입력: 실습 노트북(models/practice_smote_ensemble.ipynb)과 같은 13개
  - 장비·레시피·스텝별 평소 대비 z값(냉각 압력, 냉각 유량)
  - 지난 6시간·24시간·7일 z 평균, 24시간 − 7일 변화
  - 마지막 고장·마지막 압력 저하 이후 시간, 지난 7일 고장 수

장비마다 기록 기간을 시간 순서로 나눈다.
  학습 0~60% | 보정 60~75% (경보 기준 정하기) | 하루 비움 | 평가 75~100% (대시보드에서 재생)

경보 등급 (장비-일 단위, 하루 점수 = 그날 구간 점수 평균, 기준은 보정 구간에서 정함)
  주의:     운영 기준 — 정밀도가 "항상 경보" 정밀도의 2배 이상이 되는 가장 낮은 기준
  즉시 점검: 같은 방식으로 4배 이상 (더 확실할 때)

결과 (data/processed/dashboard/, git에 올리지 않음)
  segments.parquet  평가 구간 행별 점수·입력
  daily.parquet     장비·평가 n일차별 요약 (냉각 압력·유량 z, 최고 점수, 등급)
  faults.parquet    중복 뺀 고장 기록 (장비, 평가 n일차, 종류)
  meta.json         모델 설명, 기준, 계수, 평가 지표

실행: python src/make_dashboard_data.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
OUT = PROC / "dashboard"

HOUR, DAY = 3600, 86400
PRESSURE, FLOW = "FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE"
DROP_FAULT = "FlowCool Pressure Dropped Below Limit"
DROP_TTF = "TTF_" + DROP_FAULT
FIT_END, CALIB_END = 0.60, 0.75       # 기록 기간 중 학습 끝, 보정 끝 비율
WATCH_MULTIPLIER = 2.0                # 주의 = 운영 기준: 정밀도가 항상 경보의 2배 이상
URGENT_MULTIPLIER = 4.0               # 즉시 점검: 정밀도가 항상 경보의 4배 이상 (더 확실할 때)
NO_FAULT_HOURS = 24 * 365             # 이전 고장이 없을 때 쓰는 값(1년)

FEATURE_NAMES = {                     # 화면에 보여 줄 입력 이름
    "p_r6h": "냉각 압력 최근 6시간", "p_r24h": "냉각 압력 최근 24시간", "p_r7D": "냉각 압력 최근 7일",
    "p_chg24": "냉각 압력 변화(24시간−7일)", "f_r6h": "냉각 유량 최근 6시간", "f_r24h": "냉각 유량 최근 24시간",
    "f_r7D": "냉각 유량 최근 7일", "f_chg24": "냉각 유량 변화(24시간−7일)", "since_any": "마지막 고장 이후 시간(로그)",
    "since_drop": "마지막 압력 저하 이후 시간(로그)", "n7d_any": "지난 7일 고장 수", "p_z": "냉각 압력 지금",
    "f_z": "냉각 유량 지금",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """스텝 구간(셔터 1)과 중복 뺀 고장 기록을 읽는다."""
    columns = ["Tool", "time_start", "time_end", "recipe", "recipe_step", DROP_TTF, PRESSURE + "_mean", FLOW + "_mean"]
    segments = pd.concat([pd.read_parquet(path, columns=columns) for path in sorted((PROC / "step_s1").glob("*.parquet"))],
                         ignore_index=True)
    segments = segments.sort_values(["Tool", "time_start"]).reset_index(drop=True)
    faults = pd.read_csv(PROC / "faults.csv")
    faults["Tool"] = faults.Tool.str[2:] + "_" + faults.Tool.str[:2]                       # 01M01 → M01_01
    faults["fault_type"] = faults.fault_name.str.replace(" [NoWaferID]", "", regex=False)
    faults = faults.drop_duplicates(["Tool", "time", "fault_type"]).sort_values(["Tool", "time"]).reset_index(drop=True)
    return segments, faults


def split_by_time(segments: pd.DataFrame) -> pd.DataFrame:
    """장비별로 학습/보정/비움/평가를 나누고, 정답과 평가 n일차를 붙인다."""
    tool_first = segments.groupby("Tool").time_start.min()
    tool_last = segments.groupby("Tool").time_end.max()
    span = tool_last - tool_first
    fit_end = segments.Tool.map(tool_first + span * FIT_END)
    test_start = segments.Tool.map(tool_first + span * CALIB_END)
    segments["part"] = np.select(
        [segments.time_end < fit_end, segments.time_end < test_start - DAY, segments.time_start >= test_start],
        ["fit", "calib", "test"], default="gap")
    segments["label"] = (segments[DROP_TTF] <= DAY).astype(int)
    # 기록 끝 24시간 안인데 TTF가 없으면 뒤에 고장이 났는지 알 수 없다 → 뺀다
    is_unknown = ((segments.Tool.map(tool_last) - segments.time_end) < DAY) & segments[DROP_TTF].isna()
    segments = segments[~is_unknown].copy()
    # 평가 n일차: 장비별 평가 시작부터 센 날짜 (장비마다 실제 날짜는 다르다)
    segments["test_day"] = np.floor((segments.time_start - segments.Tool.map(tool_first + span * CALIB_END)) / DAY).astype(int)
    return segments.reset_index(drop=True)


def nearest_fault_distance(segments: pd.DataFrame, faults: pd.DataFrame) -> np.ndarray:
    """구간마다 가장 가까운 고장 기록(앞 또는 뒤)까지의 시간(초)."""
    distance = np.full(len(segments), np.inf)
    for tool, row_index in segments.groupby("Tool").indices.items():
        fault_times = faults.loc[faults.Tool == tool, "time"].to_numpy()
        if len(fault_times) == 0:
            continue
        starts, ends = segments.time_start.to_numpy()[row_index], segments.time_end.to_numpy()[row_index]
        prev_pos = np.searchsorted(fault_times, starts)
        dist_prev = np.where(prev_pos > 0, starts - fault_times[np.maximum(prev_pos - 1, 0)], np.inf)
        next_pos = np.searchsorted(fault_times, ends)
        dist_next = np.where(next_pos < len(fault_times), fault_times[np.minimum(next_pos, len(fault_times) - 1)] - ends, np.inf)
        distance[row_index] = np.minimum(dist_prev, dist_next)
    return distance


def hours_since_last(event_times: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """starts 시각마다 직전 사건 이후 몇 시간 지났는지 (없으면 1년)."""
    if len(event_times) == 0:
        return np.full(len(starts), NO_FAULT_HOURS, dtype=float)
    prev_pos = np.searchsorted(event_times, starts)
    hours = np.where(prev_pos > 0, (starts - event_times[np.maximum(prev_pos - 1, 0)]) / HOUR, NO_FAULT_HOURS)
    return np.minimum(hours, NO_FAULT_HOURS)


def build_features(segments: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    """입력 13개를 만든다. 평소 기준(z값)은 학습 구간 평소 행에서만 구한다."""
    is_normal = nearest_fault_distance(segments, faults) > DAY
    context_key = segments.Tool + "|" + segments.recipe.astype(str) + "|" + segments.recipe_step.fillna(-1).astype(str)
    baseline_rows = (segments.part == "fit") & is_normal
    for sensor in [PRESSURE, FLOW]:
        stats = segments[baseline_rows].groupby(context_key[baseline_rows])[sensor + "_mean"].agg(["mean", "std", "size"])
        stats = stats[(stats["size"] >= 10) & (stats["std"] > 1e-6)]
        z_value = (segments[sensor + "_mean"] - context_key.map(stats["mean"])) / context_key.map(stats["std"])
        segments["z_" + sensor] = z_value.clip(-10, 10)

    feature_parts = []
    for tool, tool_frame in segments.groupby("Tool"):
        by_time = tool_frame.set_index(pd.to_datetime(tool_frame.time_start, unit="s"))
        part = pd.DataFrame(index=tool_frame.index)
        for sensor, short in [(PRESSURE, "p"), (FLOW, "f")]:
            z_values = by_time["z_" + sensor]
            for window in ["6h", "24h", "7D"]:
                part[f"{short}_r{window}"] = z_values.rolling(window, min_periods=1).mean().to_numpy()
            part[f"{short}_chg24"] = part[f"{short}_r24h"] - part[f"{short}_r7D"]
        tool_faults = faults[faults.Tool == tool]
        any_times = tool_faults.time.to_numpy()
        drop_times = tool_faults.loc[tool_faults.fault_type == DROP_FAULT, "time"].to_numpy()
        starts = tool_frame.time_start.to_numpy()
        part["since_any"] = np.log1p(hours_since_last(any_times, starts))
        part["since_drop"] = np.log1p(hours_since_last(drop_times, starts))
        part["n7d_any"] = np.searchsorted(any_times, starts) - np.searchsorted(any_times, starts - 7 * DAY)
        feature_parts.append(part)
    features = pd.concat(feature_parts).sort_index()
    features["p_z"] = segments["z_" + PRESSURE]
    features["f_z"] = segments["z_" + FLOW]
    return features[list(FEATURE_NAMES)].fillna(0)


def precision_recall_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """기준 이상을 경보로 볼 때 정밀도, 재현율, 경보 비율."""
    alarmed = scores >= threshold
    true_alarms = (alarmed & (labels == 1)).sum()
    precision = true_alarms / alarmed.sum() if alarmed.sum() else 0.0
    recall = true_alarms / labels.sum() if labels.sum() else 0.0
    return float(precision), float(recall), float(alarmed.mean())


def lowest_threshold_with_precision(scores: np.ndarray, labels: np.ndarray, multiplier: float) -> float:
    """낮은 기준부터 올리며, 정밀도가 (항상 경보 정밀도 × multiplier) 이상이 되는 첫 기준."""
    base_rate = labels.mean()
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 0.99, 300)))
    for threshold in candidates:
        precision, _, _ = precision_recall_at(scores, labels, threshold)
        if precision >= multiplier * base_rate:
            return float(threshold)
    return float(candidates[-1])                                  # 끝까지 못 넘으면 가장 높은 후보


def summarize_days(rows: pd.DataFrame) -> pd.DataFrame:
    """장비·n일차 단위 요약. 하루 점수 = 그날 구간 점수 평균, 하루 정답 = 그날 정답 1 구간이 하나라도 있음."""
    return rows.groupby(["Tool", "test_day"]).agg(
        day_score=("score", "mean"), max_score=("score", "max"), pressure_z=("p_z", "mean"), flow_z=("f_z", "mean"),
        segments=("score", "size"), label=("label", "max")).reset_index()


def level_of(scores, thresholds: dict) -> np.ndarray:
    """점수 → 등급 (0 정상, 1 주의, 2 즉시 점검)."""
    return np.where(scores >= thresholds["urgent"], 2, np.where(scores >= thresholds["watch"], 1, 0))


def day_metrics(days: pd.DataFrame, alarmed: np.ndarray) -> dict:
    """장비-일 단위 정밀도·재현율·하루 경보 수(20대 합)."""
    labels = days.label.to_numpy()
    true_alarms = (alarmed & (labels == 1)).sum()
    return {"precision": float(true_alarms / alarmed.sum()) if alarmed.sum() else 0.0,
            "recall": float(true_alarms / labels.sum()) if labels.sum() else 0.0,
            "alarms_per_day": float(alarmed.sum() / days.test_day.nunique())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    segments, faults = load_data()
    segments = split_by_time(segments)
    features = build_features(segments, faults)

    is_fit, is_calib, is_test = (segments.part == part for part in ("fit", "calib", "test"))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(features[is_fit], segments.label[is_fit])
    segments["score"] = model.predict_proba(features)[:, 1]
    segments["p_z"], segments["f_z"] = features.p_z, features.f_z

    # 경보 기준은 보정 구간의 장비-일 점수로 정한다 (평가 구간은 보지 않음)
    calib_days = summarize_days(segments[is_calib])
    calib_scores, calib_labels = calib_days.day_score.to_numpy(), calib_days.label.to_numpy()
    thresholds = {
        "watch": lowest_threshold_with_precision(calib_scores, calib_labels, WATCH_MULTIPLIER),
        "urgent": lowest_threshold_with_precision(calib_scores, calib_labels, URGENT_MULTIPLIER),
        "calib_day_base_rate": float(calib_labels.mean()),
    }

    # 평가 구간 행·장비-일 저장
    test_rows = segments.loc[is_test, ["Tool", "time_start", "time_end", "test_day", "recipe", "recipe_step", "label", "score"]]
    test_rows = pd.concat([test_rows, features[is_test]], axis=1)
    test_rows.to_parquet(OUT / "segments.parquet", index=False)
    test_days = summarize_days(test_rows)
    test_days["level"] = level_of(test_days.day_score, thresholds)
    test_days.to_parquet(OUT / "daily.parquet", index=False)

    # 평가 구간 장비-일 지표: 등급별, 단순 규칙(그날 냉각 압력 z 평균 ≤ −1), 항상 경보
    metrics = {
        "즉시 점검": day_metrics(test_days, test_days.level.to_numpy() >= 2),
        "주의 이상": day_metrics(test_days, test_days.level.to_numpy() >= 1),
        "단순 규칙(냉각 압력 z 평균 ≤ −1)": day_metrics(test_days, test_days.pressure_z.to_numpy() <= -1),
        "항상 경보": day_metrics(test_days, np.ones(len(test_days), dtype=bool)),
    }
    per_tool = []
    for tool, tool_days in test_days.groupby("Tool"):
        tool_result = day_metrics(tool_days, tool_days.level.to_numpy() >= 1)
        per_tool.append({"Tool": tool, "fault_days": int(tool_days.label.sum()), "days": len(tool_days), **tool_result})

    # 입력별 기여도 계산용: 표준화 평균·폭과 계수
    scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]

    # 고장 기록에 평가 n일차 붙이기 (평가 구간 앞은 음수)
    tool_first = segments.groupby("Tool").time_start.min()
    tool_last = segments.groupby("Tool").time_end.max()
    test_start_by_tool = tool_first + (tool_last - tool_first) * CALIB_END
    faults["test_day"] = np.floor((faults.time - faults.Tool.map(test_start_by_tool)) / DAY)
    faults[["Tool", "time", "test_day", "fault_type"]].dropna().to_parquet(OUT / "faults.parquet", index=False)

    meta = {
        "model": "로지스틱 회귀 + 클래스 가중치",
        "learning": "지도 학습",
        "label": "구간 끝에서 24시간 안에 압력 저하가 오면 1 (장비-일: 그런 구간이 하루에 하나라도 있으면 1)",
        "fault": DROP_FAULT,
        "split": "장비별 시간순: 학습 0~60%, 보정 60~75%, 하루 비움, 평가 75~100%",
        "day_score": "그날 구간 점수의 평균",
        "thresholds": thresholds,
        "threshold_rule": {
            "watch": f"주의 = 운영 기준: 보정 구간 장비-일 정밀도 ≥ 항상 경보의 {WATCH_MULTIPLIER:g}배가 되는 가장 낮은 기준",
            "urgent": f"즉시 점검 = 보정 구간 장비-일 정밀도 ≥ 항상 경보의 {URGENT_MULTIPLIER:g}배가 되는 가장 낮은 기준"},
        "metrics_test_days": metrics,
        "per_tool": per_tool,
        "feature_names": FEATURE_NAMES,
        "coefficients": dict(zip(FEATURE_NAMES, classifier.coef_[0].tolist())),
        "intercept": float(classifier.intercept_[0]),
        "scaler_mean": dict(zip(FEATURE_NAMES, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(FEATURE_NAMES, scaler.scale_.tolist())),
        "rows": {"fit": int(is_fit.sum()), "calib": int(is_calib.sum()), "test": int(is_test.sum())},
        "test_days": int(test_days.test_day.nunique()),
        "test_start_by_tool": {tool: float(start) for tool, start in test_start_by_tool.items()},   # API 재생 시계 기준
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"thresholds": thresholds, "metrics": metrics, "rows": meta["rows"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
