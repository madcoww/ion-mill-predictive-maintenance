"""원본 CSV(압축 해제본)를 장비별 parquet로 줄이는 스크립트.

만드는 것 (모두 data/processed/ 아래, git 제외):
  minute/<장비>.parquet   장비 전체 기간을 1분 단위로 요약 (센서별 평균·최소·최대 + TTF)
  prefault/<장비>.parquet 고장 24시간 전 이내 구간의 원본 행 (4초 해상도, 행 중복 없음)
  step/<M01_01>.parquet   레시피 스텝 구간 하나를 한 행으로 요약 (센서별 평균·최소·최대 + TTF)
  step_s1/<M01_01>.parquet 식각 중(셔터 1) 원본 행만 먼저 거른 뒤 스텝 구간으로 요약 (EDA 기본 데이터)
  faults.csv              20대 고장 기록 합본
  manifest.json           장비별 행 수·중복 제거 수·출력 행 수

규칙
  - 정답(TTF)은 센서 파일과 시각이 같아서 행 순서대로 붙인다. 어긋나면 중단한다.
  - 값까지 완전히 같은 중복 행만 제거한다. 같은 시각에 값이 다른 행은 그대로 둔다.
  - TTF 결측은 "그 기간 안에 고장 없음"이라는 뜻이라 채우지 않는다.
  - 시간은 원본 단위가 초라고 가정한다(1분 = 60, 24시간 = 86400).

사용법 (프로젝트 루트에서)
  python -X utf8 src/make_processed.py              # 20대 전부 (없는 출력만 만듦)
  python -X utf8 src/make_processed.py 05M02 01M01  # 지정한 장비만
  python -X utf8 src/make_processed.py --force      # 다시 만들기
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/phm_data_challenge_2018"
OUT = ROOT / "data/processed"

MINUTE = 60
PREFAULT_WINDOW = 24 * 3600

# 연속형 센서 16개 (표준화 값). FIXTURESHUTTERPOSITION은 코드값이라 따로 다룬다.
SENSORS = [
    "IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
    "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
]
TTF_COLS = [
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]


def label(tool: str) -> str:
    """01M01 → M01_01 (step/ 출력에 쓰는 장비 이름)"""
    return f"{tool[2:]}_{tool[:2]}"


def all_tools() -> list[str]:
    return sorted(p.name[:6].replace("_", "") for p in (RAW / "train").glob("*_DC_train.csv"))


def load_tool(tool: str) -> tuple[pd.DataFrame, int]:
    """센서+TTF를 합쳐 반환. (데이터프레임, 제거한 완전 중복 행 수)"""
    name = f"{tool[:2]}_{tool[2:]}_DC_train.csv"
    sensor = pd.read_csv(RAW / "train" / name, engine="pyarrow")
    ttf = pd.read_csv(RAW / "train/train_ttf" / name, engine="pyarrow")
    if len(sensor) != len(ttf) or not (sensor["time"].to_numpy() == ttf["time"].to_numpy()).all():
        raise ValueError(f"{tool}: 센서와 TTF의 시각이 어긋남")
    df = pd.concat([sensor, ttf.drop(columns="time")], axis=1)
    del sensor, ttf
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_raw - len(df)


def make_minute(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df["time"] // MINUTE, sort=True)
    stats = g[SENSORS].agg(["mean", "min", "max"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    info = g.agg(
        time_start=("time", "min"),
        n_rows=("time", "size"),
        stage=("stage", "first"),
        Lot=("Lot", "first"),
        runnum=("runnum", "first"),
        recipe=("recipe", "first"),
        recipe_step=("recipe_step", "first"),
        shutter=("FIXTURESHUTTERPOSITION", "first"),
        **{c: (c, "min") for c in TTF_COLS},
    )
    out = pd.concat([info, stats], axis=1).reset_index(drop=True)
    out.insert(0, "Tool", df["Tool"].iloc[0])
    return out.astype({c: "float32" for c in stats.columns})


def make_prefault(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df[TTF_COLS] <= PREFAULT_WINDOW).any(axis=1)
    out = df.loc[mask].reset_index(drop=True)
    return out.astype({c: "float32" for c in SENSORS})


def make_step(df: pd.DataFrame) -> pd.DataFrame:
    """레시피 스텝이 이어지는 구간 하나를 한 행으로 묶는다.

    기준은 recipe_step 하나다. 값이 바뀔 때만 새 구간을 시작한다.
    """
    df = df.sort_values("time", kind="stable")
    step = df["recipe_step"].fillna(-1)
    new = step.ne(step.shift())
    seg = new.cumsum().to_numpy()
    g = df.groupby(seg, sort=True)
    stats = g[SENSORS].agg(["mean", "min", "max"])
    stats.columns = [f"{c}_{s}" for c, s in stats.columns]
    info = g.agg(
        time_start=("time", "min"),
        time_end=("time", "max"),
        n_rows=("time", "size"),
        stage=("stage", "first"),
        Lot=("Lot", "first"),
        runnum=("runnum", "first"),
        recipe=("recipe", "first"),
        recipe_step=("recipe_step", "first"),
        shutter=("FIXTURESHUTTERPOSITION", "first"),
        **{c: (c, "min") for c in TTF_COLS},
    )
    info.insert(2, "duration_s", info["time_end"] - info["time_start"])
    out = pd.concat([info, stats], axis=1).reset_index(drop=True)
    out.insert(0, "Tool", label(df["Tool"].iloc[0]))
    return out.astype({c: "float32" for c in stats.columns})


def make_step_s1(df: pd.DataFrame) -> pd.DataFrame:
    """식각 중(셔터 1) 원본 행만 먼저 거른 뒤, 레시피 스텝 구간으로 묶는다.

    step/과 달리 한 구간 안에 셔터 1이 아닌 행이 섞이지 않는다.
    새 구간 시작: recipe_step이 바뀌거나, 바로 앞 원본 행이 셔터 1이 아니었을 때
    (셔터 1이 끊겼다가 같은 스텝에서 다시 시작해도 다른 구간으로 본다).
    """
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    is_etching = (df["FIXTURESHUTTERPOSITION"] == 1).to_numpy()
    etching_df = df.loc[is_etching]
    original_position = etching_df.index.to_series()
    step_value = etching_df["recipe_step"].fillna(-1)
    # 스텝이 바뀜 또는 원본에서 바로 앞 행이 셔터 1이 아님(위치가 1 넘게 벌어짐)
    starts_new_segment = step_value.ne(step_value.shift()) | original_position.diff().ne(1)
    segment_id = starts_new_segment.cumsum().to_numpy()
    grouped = etching_df.groupby(segment_id, sort=True)
    stats = grouped[SENSORS].agg(["mean", "min", "max"])
    stats.columns = [f"{sensor}_{stat_name}" for sensor, stat_name in stats.columns]
    info = grouped.agg(
        time_start=("time", "min"),
        time_end=("time", "max"),
        n_rows=("time", "size"),
        stage=("stage", "first"),
        Lot=("Lot", "first"),
        runnum=("runnum", "first"),
        recipe=("recipe", "first"),
        recipe_step=("recipe_step", "first"),
        shutter=("FIXTURESHUTTERPOSITION", "first"),
        **{ttf_col: (ttf_col, "min") for ttf_col in TTF_COLS},
    )
    info.insert(2, "duration_s", info["time_end"] - info["time_start"])
    out = pd.concat([info, stats], axis=1).reset_index(drop=True)
    out.insert(0, "Tool", label(df["Tool"].iloc[0]))
    return out.astype({col: "float32" for col in stats.columns})


def shutter_mix_stats(df: pd.DataFrame) -> dict:
    """step/(recipe_step만으로 나눈 구간)에서 구간 안에 셔터 값이 섞이는 정도."""
    df = df.sort_values("time", kind="stable")
    step_value = df["recipe_step"].fillna(-1)
    segment_id = step_value.ne(step_value.shift()).cumsum().to_numpy()
    shutter = df["FIXTURESHUTTERPOSITION"].fillna(-1)
    per_segment = pd.DataFrame({"segment": segment_id, "shutter": shutter.to_numpy()}).groupby("segment")["shutter"]
    n_unique = per_segment.nunique()
    first_value = per_segment.first()
    is_etching_row = (shutter == 1).to_numpy()
    row_is_mixed = n_unique.reindex(segment_id).to_numpy() > 1
    row_first_is_etching = (first_value.reindex(segment_id).to_numpy() == 1)
    return {
        "step_segments": int(len(n_unique)),
        "mixed_segments": int((n_unique > 1).sum()),
        "etching_rows": int(is_etching_row.sum()),
        "etching_rows_in_mixed": int((is_etching_row & row_is_mixed).sum()),
        # 옛 방식(첫 값이 1인 구간)이 잡은 행 중 실제 셔터 1 / 아닌 행
        "old_rows_etching": int((row_first_is_etching & is_etching_row).sum()),
        "old_rows_not_etching": int((row_first_is_etching & ~is_etching_row).sum()),
    }


MAKERS = {"minute": make_minute, "prefault": make_prefault, "step": make_step, "step_s1": make_step_s1}


def process(tool: str, force: bool) -> dict:
    paths = {k: OUT / k / f"{tool}.parquet" for k in MAKERS}
    paths["step"] = OUT / "step" / f"{label(tool)}.parquet"
    paths["step_s1"] = OUT / "step_s1" / f"{label(tool)}.parquet"
    todo = [k for k, p in paths.items() if force or not p.exists()]
    if not todo:
        return {"tool": tool, "skipped": True}
    t0 = time.time()
    df, n_dup = load_tool(tool)
    res = {"tool": tool, "rows_after_dedup": len(df), "exact_duplicates_dropped": n_dup}
    for k in todo:
        out = MAKERS[k](df)
        out.to_parquet(paths[k], engine="pyarrow", compression="zstd", index=False)
        res[f"{k}_rows"] = len(out)
    if "step_s1" in todo:
        # 옛 step/ 방식과 비교할 수 있게 셔터가 섞인 정도를 manifest에 남긴다
        res["shutter_mix"] = shutter_mix_stats(df)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def write_faults() -> int:
    paths = sorted((RAW / "train/train_faults").glob("*.csv"))
    faults = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    faults.to_csv(OUT / "faults.csv", index=False)
    return len(faults)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tools", nargs="*", help="예: 05M02 (생략하면 전부)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for sub in MAKERS:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    tools = args.tools or all_tools()

    n_faults = write_faults()
    print(f"faults.csv: {n_faults}행", flush=True)

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for i, tool in enumerate(tools, 1):
        res = process(tool, args.force)
        print(f"[{i}/{len(tools)}] {res}", flush=True)
        if not res.get("skipped"):
            manifest[tool] = {**manifest.get(tool, {}), **res}
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
