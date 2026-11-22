"""이온 밀 냉각 계통 경보 API (FastAPI) — 평가 구간을 실시간처럼 재생한다.

실제 장비 연결 대신, 평가 구간(장비별 기록 기간의 뒤 25%)의 스텝 구간을 시간 순서대로 흘려보낸다.
재생 시계가 지난 구간만 "도착한 데이터"로 보고, 그 시점까지의 정보로 상태를 계산한다.

  재생 시계: 평가 시작부터 지난 시간. 실제 1초 = speed초 (기본 600 → 1초에 10분)
  장비 상태: 최근 24시간에 도착한 구간 점수의 평균 → 주의/즉시 점검 기준과 비교
             (기준은 make_dashboard_data.py가 보정 구간 장비-일 평균 점수로 정한 값)
             최근 24시간에 식각 기록이 없으면 판정 불가

모델: 로지스틱 회귀 + 클래스 가중치 (지도 학습, 정답 = 24시간 안 압력 저하)
데이터: data/processed/dashboard/ (먼저 python src/make_dashboard_data.py 실행)

실행: uvicorn app.api:app --port 8000   (문서: http://localhost:8000/docs)
"""
from pathlib import Path
from datetime import datetime
import json
import sqlite3
import time
import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/dashboard"
INSPECTION_DB = DATA_DIR / "inspections.sqlite"
HOUR, DAY = 3600, 86400
STATUS_WINDOW = DAY                   # 상태 계산에 쓰는 최근 시간
DEFAULT_SPEED = 600.0                 # 실제 1초에 재생 600초(10분)
START_ELAPSED = 30 * DAY              # 처음 켤 때 재생 위치 (평가 30일차)
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "즉시 점검", -1: "판정 불가"}
FAULT_NAMES = {
    "FlowCool Pressure Dropped Below Limit": "압력 저하",
    "Flowcool Pressure Too High Check Flowcool Pump": "압력 과다",
    "Flowcool leak": "누설",
}

# ---------- 데이터 읽기 (서버 시작 때 한 번) ----------
meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
FEATURES = list(meta["feature_names"])
segments = pd.read_parquet(DATA_DIR / "segments.parquet")
faults = pd.read_parquet(DATA_DIR / "faults.parquet")
faults["fault_label"] = faults.fault_type.map(FAULT_NAMES).fillna("기타")

# 장비마다 평가 시작 시각이 다르므로 "평가 시작 후 지난 초"(elapsed)로 맞춘다
test_start_by_tool = pd.Series(meta["test_start_by_tool"])
segments["elapsed"] = segments.time_end - segments.Tool.map(test_start_by_tool)     # 구간이 끝나야 도착한 것으로 본다
faults["elapsed"] = faults.time - faults.Tool.map(test_start_by_tool)
segments = segments.sort_values(["Tool", "elapsed"]).reset_index(drop=True)
TOOLS = sorted(segments.Tool.unique())
MAX_ELAPSED = float(segments.elapsed.max())
segments_by_tool = {tool: frame.reset_index(drop=True) for tool, frame in segments.groupby("Tool")}
elapsed_by_tool = {tool: frame.elapsed.to_numpy() for tool, frame in segments_by_tool.items()}


class ReplayClock:
    """재생 시계. 멈춤·속도·위치를 바꿀 수 있다."""

    def __init__(self, speed: float, start_elapsed: float):
        self.lock = threading.Lock()
        self.speed = speed
        self.paused = False
        self.anchor_elapsed = start_elapsed          # 마지막으로 설정한 재생 위치
        self.anchor_wall = time.monotonic()          # 그때의 실제 시각

    def now(self) -> float:
        """지금 재생 위치(평가 시작 후 초)."""
        with self.lock:
            if self.paused:
                return self.anchor_elapsed
            moved = (time.monotonic() - self.anchor_wall) * self.speed
            return min(self.anchor_elapsed + moved, MAX_ELAPSED)

    def update(self, speed: float | None, elapsed: float | None, paused: bool | None) -> None:
        current = self.now()
        with self.lock:
            self.anchor_elapsed = current if elapsed is None else float(np.clip(elapsed, 0, MAX_ELAPSED))
            self.anchor_wall = time.monotonic()
            if speed is not None:
                self.speed = speed
            if paused is not None:
                self.paused = paused


clock = ReplayClock(DEFAULT_SPEED, START_ELAPSED)


def arrived_rows(tool: str, now: float, window: float) -> pd.DataFrame:
    """now까지 도착한 구간 중 최근 window초 것."""
    elapsed = elapsed_by_tool[tool]
    end = np.searchsorted(elapsed, now, side="right")
    start = np.searchsorted(elapsed, now - window, side="right")
    return segments_by_tool[tool].iloc[start:end]


def level_of(score: float) -> int:
    thresholds = meta["thresholds"]
    return 2 if score >= thresholds["urgent"] else (1 if score >= thresholds["watch"] else 0)


def tool_status(tool: str, now: float) -> dict:
    """장비 하나의 현재 상태."""
    recent = arrived_rows(tool, now, STATUS_WINDOW)
    past_faults = faults[(faults.Tool == tool) & (faults.elapsed <= now)]
    last_fault = None
    if len(past_faults):
        latest = past_faults.iloc[-1]
        last_fault = {"type": latest.fault_label, "hours_ago": round((now - latest.elapsed) / HOUR, 1)}
    if recent.empty:
        return {"tool": tool, "level": -1, "level_name": LEVEL_NAMES[-1], "score": None, "pressure_z": None,
                "flow_z": None, "segments_24h": 0, "last_fault": last_fault}
    score = float(recent.score.mean())
    level = level_of(score)
    return {"tool": tool, "level": level, "level_name": LEVEL_NAMES[level], "score": round(score, 4),
            "pressure_z": round(float(recent.p_z.mean()), 3), "flow_z": round(float(recent.f_z.mean()), 3),
            "segments_24h": int(len(recent)), "last_fault": last_fault}


def clock_info(now: float) -> dict:
    return {"elapsed_hours": round(now / HOUR, 2), "test_day": int(now // DAY), "hour_of_day": round((now % DAY) / HOUR, 1),
            "speed": clock.speed, "paused": clock.paused, "max_elapsed_hours": round(MAX_ELAPSED / HOUR, 1),
            "server_time": datetime.now().isoformat(timespec="seconds")}


def open_inspection_db() -> sqlite3.Connection:
    connection = sqlite3.connect(INSPECTION_DB)
    connection.execute("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, saved_at TEXT, tool TEXT, test_day INTEGER,
        alarm_level TEXT, day_score REAL, result TEXT, memo TEXT)""")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(inspections)")}
    if "elapsed_hours" not in columns:                    # 재생 위치 열 (이전 버전 표에 없으면 추가)
        connection.execute("ALTER TABLE inspections ADD COLUMN elapsed_hours REAL")
    return connection


# ---------- API ----------
app = FastAPI(title="이온 밀 냉각 계통 경보 API", version="0.1.0",
              description="평가 구간을 실시간처럼 재생하며 장비별 압력 저하 위험을 알려 준다 (프로토타입).")


class ReplayUpdate(BaseModel):
    speed: float | None = Field(None, gt=0, le=86400, description="실제 1초에 재생할 초")
    elapsed_hours: float | None = Field(None, ge=0, description="이 위치로 이동 (평가 시작 후 시간)")
    paused: bool | None = None


class InspectionIn(BaseModel):
    tool: str
    result: str = Field(..., description="이상 없음 / 부품 교체 / 조정·청소 / 기타")
    memo: str = ""


@app.get("/health")
def health():
    return {"ok": True, "tools": len(TOOLS), "segments": len(segments)}


@app.get("/meta")
def get_meta():
    """모델 설명, 경보 기준, 평가 성능."""
    return meta


@app.get("/clock")
def get_clock():
    return clock_info(clock.now())


@app.post("/replay")
def update_replay(update: ReplayUpdate):
    """재생 속도·위치·멈춤 바꾸기."""
    clock.update(update.speed, None if update.elapsed_hours is None else update.elapsed_hours * HOUR, update.paused)
    return clock_info(clock.now())


@app.get("/status")
def get_status():
    """지금 재생 위치 기준 20대 상태."""
    now = clock.now()
    return {"clock": clock_info(now), "tools": [tool_status(tool, now) for tool in TOOLS]}


@app.get("/alarms")
def get_alarms():
    """주의 이상 장비 (점수 높은 순)."""
    now = clock.now()
    alarms = [status for status in (tool_status(tool, now) for tool in TOOLS) if status["level"] >= 1]
    return {"clock": clock_info(now), "alarms": sorted(alarms, key=lambda status: status["score"], reverse=True)}


def check_tool(tool: str) -> None:
    if tool not in segments_by_tool:
        raise HTTPException(404, f"장비 {tool} 없음")


@app.get("/tools/{tool}/history")
def get_history(tool: str, hours: float = 168, bucket_minutes: int = 60):
    """최근 hours시간의 점수·냉각 센서 추이 (bucket_minutes 단위 평균)와 그 사이 고장."""
    check_tool(tool)
    now = clock.now()
    rows = arrived_rows(tool, now, hours * HOUR)
    history = []
    if len(rows):
        bucket = (rows.elapsed // (bucket_minutes * 60)).astype(int)
        grouped = rows.groupby(bucket).agg(elapsed=("elapsed", "max"), score=("score", "mean"),
                                           pressure_z=("p_z", "mean"), flow_z=("f_z", "mean"))
        grouped["elapsed_hours"] = (grouped.elapsed / HOUR).round(3)
        history = grouped.drop(columns="elapsed").round(4).to_dict("records")
    window_faults = faults[(faults.Tool == tool) & (faults.elapsed <= now) & (faults.elapsed > now - hours * HOUR)]
    return {"clock": clock_info(now), "tool": tool, "history": history,
            "faults": [{"elapsed_hours": round(row.elapsed / HOUR, 2), "type": row.fault_label} for row in window_faults.itertuples()]}


@app.get("/tools/{tool}/reasons")
def get_reasons(tool: str):
    """최근 24시간 입력 평균 기준, 점수를 올린(+)·내린(−) 입력 (계수 × 표준화 값)."""
    check_tool(tool)
    recent = arrived_rows(tool, clock.now(), STATUS_WINDOW)
    if recent.empty:
        return {"tool": tool, "reasons": []}
    reasons = []
    for feature in FEATURES:
        standardized = (recent[feature].mean() - meta["scaler_mean"][feature]) / meta["scaler_scale"][feature]
        contribution = float(meta["coefficients"][feature] * standardized)   # numpy 숫자는 JSON으로 못 보내므로 float로
        reasons.append({"feature": meta["feature_names"][feature], "contribution": round(contribution, 4)})
    return {"tool": tool, "reasons": sorted(reasons, key=lambda reason: reason["contribution"], reverse=True)}


@app.get("/inspections")
def list_inspections():
    connection = open_inspection_db()
    records = pd.read_sql("SELECT * FROM inspections ORDER BY id DESC", connection)
    connection.close()
    return records.replace({np.nan: None}).to_dict("records")


@app.post("/inspections")
def add_inspection(inspection: InspectionIn):
    """점검 결과 저장. 경보 등급·점수·재생 위치는 서버가 지금 상태로 채운다."""
    check_tool(inspection.tool)
    now = clock.now()
    status = tool_status(inspection.tool, now)
    connection = open_inspection_db()
    connection.execute(
        "INSERT INTO inspections (saved_at, tool, test_day, alarm_level, day_score, result, memo, elapsed_hours) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), inspection.tool, int(now // DAY), status["level_name"],
         status["score"], inspection.result, inspection.memo, round(now / HOUR, 2)))
    connection.commit()
    connection.close()
    return {"saved": True, "status": status}
