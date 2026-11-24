"""Run from the repository root: python -m src.eda.run --tool 01M01."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SENSORS = [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
]
CORE = ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE"]
FAULTS = {
    "FlowCool Pressure Dropped Below Limit": "pressure_low",
    "Flowcool Pressure Too High Check Flowcool Pump": "pressure_high",
    "Flowcool leak": "leak",
}
TTF = [f"TTF_{fault_name}" for fault_name in FAULTS]
CONDITIONS = ["stage", "Lot", "runnum", "recipe", "recipe_step", "FIXTURESHUTTERPOSITION"]


def save_table(frame: pd.DataFrame, folder: Path, name: str) -> None:
    frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def segment_ids(frame: pd.DataFrame, gap_seconds: float) -> pd.Series:
    """Use the original sorted stream, including shutter-off rows."""
    changed = pd.Series(False, index=frame.index)
    for column in CONDITIONS:
        previous = frame[column].shift()
        unchanged = frame[column].eq(previous) | (frame[column].isna() & previous.isna())
        changed |= ~unchanged
    changed |= frame["time"].diff().gt(gap_seconds)
    if len(changed):
        changed.iloc[0] = True
    return changed.cumsum().rename("segment_id")


def fault_events(faults: pd.DataFrame, merge_hours: float) -> pd.DataFrame:
    faults = faults.drop_duplicates(["Tool", "time", "fault_name"]).copy()
    faults["fault_type"] = faults["fault_name"].map(FAULTS)
    if faults["fault_type"].isna().any():
        raise ValueError("Unknown fault name; explicitly map it before analysis")
    faults = faults.sort_values(["Tool", "fault_type", "time"])
    gaps = faults.groupby(["Tool", "fault_type"])["time"].diff()
    faults["event_group"] = (gaps.isna() | gaps.gt(merge_hours * 3600)).cumsum()
    return faults.groupby("event_group", sort=False).agg(
        Tool=("Tool", "first"), fault_type=("fault_type", "first"),
        event_time=("time", "min"), last_record_time=("time", "max"),
        record_count=("time", "size"),
    ).reset_index(drop=True).sort_values("event_time").reset_index(drop=True)


def load_data(tool: str, raw_root: Path, tables: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[Path]]:
    stem = f"{tool[:2]}_{tool[2:]}"
    paths = [raw_root / "train" / f"{stem}_DC_train.csv",
             raw_root / "train/train_ttf" / f"{stem}_DC_train.csv",
             raw_root / "train/train_faults" / f"{stem}_train_fault_data.csv"]
    sensor = pd.read_csv(paths[0])
    target = pd.read_csv(paths[1])
    if not sensor["time"].equals(target["time"]):
        raise ValueError("Sensor and TTF row counts/timestamps differ")
    if sensor["time"].isna().any() or set(sensor["Tool"].unique()) != {tool}:
        raise ValueError("Missing timestamp or unexpected equipment ID")
    if not np.isfinite(sensor["time"].to_numpy()).all():
        raise ValueError("Non-finite timestamp")
    quality = {"raw_rows": len(sensor), "time_reversals_in_input": int(sensor["time"].diff().lt(0).sum()),
               "sensor_ttf_aligned": True}
    save_table(pd.DataFrame({"column": sensor.columns, "missing_rows": sensor.isna().sum().to_numpy(),
                            "missing_fraction": sensor.isna().mean().to_numpy()}), tables, "missingness")
    save_table(sensor[SENSORS].describe(percentiles=[.01, .05, .5, .95, .99]).T.reset_index(names="sensor"),
               tables, "sensor_summary_all_states")
    save_table(pd.DataFrame({"target": TTF, "missing_rows": target[TTF].isna().sum().to_numpy(),
                            "negative_rows": target[TTF].lt(0).sum().to_numpy()}), tables, "ttf_quality")
    frame = pd.concat([sensor, target.drop(columns="time")], axis=1)
    del sensor, target
    duplicate_mask = frame.duplicated()
    quality["exact_duplicates_removed"] = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].sort_values("time", kind="stable").reset_index(drop=True)
    del duplicate_mask
    timestamp_counts = frame.groupby("time").size()
    quality["conflicting_timestamp_count"] = int(timestamp_counts.gt(1).sum())
    save_table(timestamp_counts[timestamp_counts.gt(1)].rename("rows").reset_index(), tables, "conflicting_timestamps")
    del timestamp_counts
    # Conversion follows duplicate detection at the original parsed precision.
    for column in SENSORS:
        frame[column] = frame[column].astype("float32")
    faults = pd.read_csv(paths[2])
    if not faults.empty and set(faults["Tool"].unique()) != {tool}:
        raise ValueError("Fault file contains another equipment ID")
    if faults["time"].isna().any():
        raise ValueError("Fault record has no timestamp")
    return frame, faults, quality, paths


def summarize_steps(frame: pd.DataFrame, gap_seconds: float) -> pd.DataFrame:
    segments = segment_ids(frame, gap_seconds)
    grouped = frame.groupby(segments, sort=False)
    metadata = grouped.agg(time_start=("time", "min"), time_end=("time", "max"), n_rows=("time", "size"),
                           recipe=("recipe", "first"), recipe_step=("recipe_step", "first"),
                           runnum=("runnum", "first"), shutter=("FIXTURESHUTTERPOSITION", "first"))
    statistics = grouped[SENSORS].agg(["mean", "min", "max", "std"])
    statistics.columns = [f"{sensor}_{stat}" for sensor, stat in statistics.columns]
    metadata["duration_s"] = metadata["time_end"] - metadata["time_start"]
    return metadata.join(statistics).reset_index()


def trajectories(etching: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descriptive full-period condition baseline; never a forecasting feature."""
    condition_groups = etching.groupby(["recipe", "recipe_step"], dropna=False)
    baseline = condition_groups[CORE].agg(["count", "mean", "std"])
    baseline.columns = [f"{sensor}_{stat}" for sensor, stat in baseline.columns]
    means = condition_groups[CORE].transform("mean")
    scales = condition_groups[CORE].transform("std").replace(0, np.nan)
    standardized = (etching[CORE] - means) / scales
    times = etching["time"].to_numpy()
    rows = []
    for event_id, event in events.iterrows():
        for hour_before in range(1, 49):
            start = np.searchsorted(times, event.event_time - hour_before * 3600)
            stop = np.searchsorted(times, event.event_time - (hour_before - 1) * 3600)
            window = standardized.iloc[start:stop]
            for sensor in CORE:
                rows.append({"event_id": event_id, "fault_type": event.fault_type,
                             "hours_before": hour_before - .5, "sensor": sensor,
                             "observed_rows": stop - start, "valid_rows": int(window[sensor].count()),
                             "mean_z": window[sensor].mean()})
    per_event = pd.DataFrame(rows, columns=["event_id", "fault_type", "hours_before", "sensor", "observed_rows", "valid_rows", "mean_z"])
    aggregate = per_event.groupby(["fault_type", "sensor", "hours_before"]).agg(
        median_z=("mean_z", "median"), q25_z=("mean_z", lambda values: values.quantile(.25)),
        q75_z=("mean_z", lambda values: values.quantile(.75)), observed_events=("mean_z", "count"),
    ).reset_index()
    return baseline.reset_index(), per_event, aggregate


def coverage(frame: pd.DataFrame, etching: pd.DataFrame, events: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    times = np.unique(frame["time"].to_numpy())
    etching_times = etching["time"].to_numpy()
    rows = []
    for event_id, event in events.iterrows():
        start_time = event.event_time - window_hours * 3600
        start, stop = np.searchsorted(times, [start_time, event.event_time])
        observed = times[start:stop]
        etch_start, etch_stop = np.searchsorted(etching_times, [start_time, event.event_time])
        minute_bins = np.unique(np.floor((observed - start_time) / 60))
        rows.append({"event_id": event_id, "fault_type": event.fault_type, "event_time": event.event_time,
                     "unique_timestamps": len(observed), "observed_minutes": len(minute_bins),
                     "minute_coverage": len(minute_bins) / (window_hours * 60),
                     "etching_rows": etch_stop - etch_start,
                     "last_observation_gap_hours": (event.event_time - observed[-1]) / 3600 if len(observed) else np.nan,
                     "within_sensor_period": bool(times[0] <= event.event_time <= times[-1]),
                     "full_pre_window_within_period": bool(start_time >= times[0] and event.event_time <= times[-1])})
    return pd.DataFrame(rows, columns=["event_id", "fault_type", "event_time", "unique_timestamps",
                                      "observed_minutes", "minute_coverage", "etching_rows",
                                      "last_observation_gap_hours", "within_sensor_period",
                                      "full_pre_window_within_period"])


def render_figures(hourly: pd.DataFrame, events: pd.DataFrame, aggregate: pd.DataFrame,
                   etching: pd.DataFrame, output: Path, origin: float) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    selected = hourly.loc[hourly["scope"].eq("etching")]
    for axis, sensor in zip(axes, CORE):
        days = (selected["hour"] * 3600 - origin) / 86400
        axis.scatter(days, selected[sensor], s=3, color="#2563eb")
        for fault_type, color in zip(FAULTS.values(), ["#dc2626", "#d97706", "#7c3aed"]):
            for event_time in events.loc[events["fault_type"].eq(fault_type), "event_time"]:
                axis.axvline((event_time - origin) / 86400, color=color, alpha=.3, lw=.8)
        axis.set_ylabel(sensor, fontsize=8)
        axis.grid(alpha=.2)
    axes[0].set_title("Full-period etching sensor means (1 hour); fault events: red=low, orange=high, purple=leak")
    axes[-1].set_xlabel("Days since this tool's first observation")
    figure.tight_layout()
    figure.savefig(output / "full_period.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, sensor in zip(axes, CORE):
        quantiles = etching[sensor].quantile([.01, .25, .5, .75, .99])
        axis.bxp([dict(label="etching", whislo=quantiles.iloc[0], q1=quantiles.iloc[1],
                       med=quantiles.iloc[2], q3=quantiles.iloc[3], whishi=quantiles.iloc[4], fliers=[])],
                 showfliers=False)
        axis.set_title(sensor, fontsize=9)
        axis.grid(alpha=.2)
    figure.suptitle("All etching observations; whiskers = 1st / 99th percentiles")
    figure.tight_layout()
    figure.savefig(output / "sensor_distributions.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for row_index, fault_type in enumerate(FAULTS.values()):
        for column_index, sensor in enumerate(CORE):
            axis = axes[row_index, column_index]
            selected = aggregate.loc[aggregate["fault_type"].eq(fault_type) & aggregate["sensor"].eq(sensor)]
            axis.plot(-selected["hours_before"], selected["median_z"], color="#2563eb")
            axis.fill_between(-selected["hours_before"], selected["q25_z"], selected["q75_z"], alpha=.2)
            axis.axhline(0, color="gray", lw=.6)
            axis.set_title(f"{fault_type}: {sensor}", fontsize=8)
            axis.grid(alpha=.2)
            count_axis = axis.twinx()
            count_axis.plot(-selected["hours_before"], selected["observed_events"], color="gray", alpha=.4, ls=":")
            count_axis.set_ylim(bottom=0)
            count_axis.tick_params(labelsize=7, colors="gray")
            if column_index == 0:
                axis.set_ylabel("Condition z (descriptive)")
            if row_index == 2:
                axis.set_xlabel("Hours before event")
    figure.suptitle("Event median / IQR; dotted right axis = observed events; full-period reference")
    figure.tight_layout()
    figure.savefig(output / "prefault_trajectories.png", dpi=150)
    plt.close(figure)


def run(tool: str, args: argparse.Namespace) -> Path:
    output = args.output / tool
    manifest_path = output / "manifest.json"
    stem = f"{tool[:2]}_{tool[2:]}"
    source_paths = [args.raw_root / "train" / f"{stem}_DC_train.csv",
                    args.raw_root / "train/train_ttf" / f"{stem}_DC_train.csv",
                    args.raw_root / "train/train_faults" / f"{stem}_train_fault_data.csv"]
    config = {"tool": tool, "gap_seconds": args.gap_seconds, "merge_hours": args.merge_hours,
              "window_hours": args.window_hours, "sampling": "none", "time_unit_assumption": "seconds"}
    inputs = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
              for path in source_paths]
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps([config, inputs, code_hash], sort_keys=True).encode()).hexdigest()
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("status") == "complete" and previous.get("signature") == signature
                and previous.get("outputs") and Path(previous["step_data"]).exists()
                and all((output / path).exists() for path in previous["outputs"])):
            print(f"{tool}: unchanged, reusing {output}", flush=True)
            return output
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"status": "running", "signature": signature}), encoding="utf-8")
    print(f"{tool}: loading full raw period", flush=True)
    frame, faults, quality, _ = load_data(tool, args.raw_root, tables)
    quality.update(rows_after_dedup=len(frame), time_start=int(frame["time"].min()), time_end=int(frame["time"].max()))
    quality["span_days"] = (quality["time_end"] - quality["time_start"]) / 86400
    differences = frame["time"].diff()
    gaps = frame.loc[differences.gt(args.gap_seconds), ["time"]].copy()
    gaps["previous_time"] = frame["time"].shift().loc[gaps.index]
    gaps["gap_seconds"] = differences.loc[gaps.index]
    quality["gap_count"] = len(gaps)
    quality["median_positive_interval_seconds"] = float(differences[differences.gt(0)].median())
    save_table(gaps, tables, "observation_gaps")
    del differences
    print(f"{tool}: quality checked; segmenting {len(frame):,} rows", flush=True)
    steps = summarize_steps(frame, args.gap_seconds)
    step_path = ROOT / "data/interim/eda" / tool / f"steps_{signature[:12]}.parquet"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(step_path, index=False, compression="zstd")
    quality["all_state_segments"] = len(steps)
    quality["etching_segments"] = int(steps["shutter"].eq(1).sum())
    del steps
    # All original observations contribute. Hourly aggregation is only for overview plots.
    hourly_frames = []
    for scope, scope_frame in [("all_states", frame), ("etching", frame.loc[frame["FIXTURESHUTTERPOSITION"].eq(1)])]:
        grouped = scope_frame.groupby((scope_frame["time"] // 3600).rename("hour"))
        hourly = grouped[CORE].mean()
        hourly["n_rows"] = grouped.size()
        hourly["scope"] = scope
        hourly_frames.append(hourly.reset_index())
    del scope_frame, grouped
    hourly = pd.concat(hourly_frames, ignore_index=True)
    save_table(hourly, tables, "hourly_sensors")
    etching = frame.loc[frame["FIXTURESHUTTERPOSITION"].eq(1), ["time", "recipe", "recipe_step"] + SENSORS]
    if etching.empty:
        raise ValueError("No shutter=1 observations; use a different analysis scope")
    quality["etching_rows"] = len(etching)
    quality["etching_row_fraction"] = len(etching) / len(frame)
    save_table(etching[SENSORS].describe(percentiles=[.01, .05, .5, .95, .99]).T.reset_index(names="sensor"), tables, "sensor_summary_etching")
    save_table(etching.groupby(["recipe", "recipe_step"], dropna=False).size().rename("n_rows").reset_index(), tables, "condition_coverage")
    events = fault_events(faults, args.merge_hours)
    quality["fault_records_raw"] = len(faults)
    quality["fault_records_unique"] = len(faults.drop_duplicates(["Tool", "time", "fault_name"]))
    quality["fault_events"] = len(events)
    save_table(events, tables, "fault_events")
    sensitivity = [{"merge_hours": hours, "event_count": len(fault_events(faults, hours))} for hours in [6, 12, 24, 48]]
    save_table(pd.DataFrame(sensitivity), tables, "event_sensitivity")
    event_coverage = coverage(frame, etching, events, args.window_hours)
    save_table(event_coverage, tables, "prefault_coverage")
    print(f"{tool}: analyzing {len(events)} events and {len(etching):,} etching rows", flush=True)
    baseline, per_event, aggregate = trajectories(etching, events)
    save_table(baseline, tables, "condition_baseline_descriptive")
    save_table(per_event, tables, "prefault_by_event")
    save_table(aggregate, tables, "prefault_aggregate")
    render_figures(hourly, events, aggregate, etching, figures, quality["time_start"])
    (output / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    fault_lines = "\n".join(f"| {fault_type} | {count} |" for fault_type, count in events["fault_type"].value_counts().items())
    median_coverage = event_coverage["minute_coverage"].median() if len(event_coverage) else np.nan
    summary = f"""# {tool} 장비별 전체 기간 EDA

- 원본 **{quality['raw_rows']:,}행**, 기간 **{quality['span_days']:.1f}일**. 무작위 샘플링 없음.
- 완전 중복 **{quality['exact_duplicates_removed']:,}행** 제거 후 **{len(frame):,}행**. 센서·TTF 시각 일치 확인.
- 식각 중(셔터 1) **{len(etching):,}행 ({quality['etching_row_fraction']:.1%})**. 전체 상태 품질 검사는 별도 수행.
- 동일 시각의 서로 다른 기록 **{quality['conflicting_timestamp_count']:,}개 시각**은 보존. 행 가중 분석이므로 중복 시각 영향 가능.
- {args.gap_seconds:g}초 초과 관측 공백 **{len(gaps):,}개**. 공백은 보간하지 않음.
- 고장 기록 **{len(faults)}건 → 중복 제외 {quality['fault_records_unique']}건 → {args.merge_hours:g}시간 묶음 {len(events)}사건**.
- 고장 전 {args.window_hours:g}시간의 관측 분 점유율 중앙값 **{median_coverage:.1%}** (전체 운전 상태; 1분에 관측 1개 이상).

| 고장 종류 | 사건 수 |
|---|---:|
{fault_lines}

## 결과 그림

![전체 기간 센서 추이](figures/full_period.png)

![식각 중 센서 분포](figures/sensor_distributions.png)

![고장 전 48시간 궤적](figures/prefault_trajectories.png)

## 해석 기준

- 전체 기간 그림은 모든 식각 관측을 1시간 평균으로 요약. 빈 시간은 채우지 않음.
- 스텝은 레시피·가동 번호·단계·웨이퍼·셔터 변경과 {args.gap_seconds:g}초 초과 공백에서 분리.
- 고장 궤적은 사건별 시간당 평균의 중앙값과 사분위 범위. 오른쪽 점선은 실제 관측 사건 수.
- z값은 **전체 기간 식각 관측의 레시피×스텝 평균·표준편차** 기준. 정상 확정 기준이 아니며 미래 정보가 포함된 설명용 통계.
- 같은 장비·종류의 앞 기록과 간격이 {args.merge_hours:g}시간 이하면 같은 사건으로 연결. 실제 정비 사건의 정답은 아님.
- TTF 결측을 정상으로 채우지 않음. 예측 성능·사전 경고 시간은 이 EDA로 판단하지 않음.
- 고장 시각은 정비 정지 시각이며 실제 이상 시작 시각은 알 수 없음. 반복 사건도 독립 표본으로 볼 수 없음.
- 시간 단위는 초로 가정. 식각 비율은 **행 비율**이며 공백을 포함하는 달력 가동률이 아님.

## 재실행

```bash
python -m src.eda.run --tool {tool}
python -m src.eda.run --tool {tool} --force
```

입력 파일 크기·수정시각, 설정, 실행 코드 해시가 같고 산출물이 있으면 재사용.
전체 스텝 데이터: `{step_path.relative_to(ROOT).as_posix()}` (git 제외).
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    outputs = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and path != manifest_path]
    manifest = {"status": "complete", "signature": signature, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "inputs": inputs, "code_sha256": code_hash, "git_commit": commit,
                "versions": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                "outputs": outputs, "step_data": str(step_path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {output / 'summary.md'}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="One-tool full-period EDA (no sampling)")
    parser.add_argument("--tool", required=True, help="e.g. 01M01")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/phm_data_challenge_2018")
    parser.add_argument("--output", type=Path, default=ROOT / "eda/reports")
    parser.add_argument("--gap-seconds", type=float, default=60)
    parser.add_argument("--merge-hours", type=float, default=24)
    parser.add_argument("--window-hours", type=float, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"(?:0[1-9]|10)M0[12]", args.tool):
        parser.error("--tool must be 01M01 .. 10M02")
    if min(args.gap_seconds, args.merge_hours, args.window_hours) <= 0 or not all(
        np.isfinite(value) for value in [args.gap_seconds, args.merge_hours, args.window_hours]
    ):
        parser.error("Time settings must be finite and positive")
    run(args.tool, args)


if __name__ == "__main__":
    main()
