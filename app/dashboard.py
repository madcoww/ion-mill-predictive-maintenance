"""이온 밀 냉각 계통 정비 판단 지원 대시보드 (프로토타입, FastAPI 연동).

화면은 데이터를 직접 읽지 않고 FastAPI 서버(app/api.py)에 물어본다.
서버가 평가 구간을 실시간처럼 재생하고, 화면은 몇 초마다 새 상태를 받아 다시 그린다.

화면 4개 (기획안 정보구조도 기준)
  1. 전체 장비 현황   20대 상태(정상/주의/즉시 점검/판정 불가), 지금 경보 목록
  2. 장비 상세       위험 점수·냉각 센서 추이, 경보 근거 입력, 점검 후보
  3. 점검 기록       점검 결과 입력·조회 (서버가 SQLite에 저장)
  4. 모델·기준 비교   모델 설명, 경보 기준, 평가 구간 성능

모델: 로지스틱 회귀 + 클래스 가중치 (지도 학습, 정답 = 24시간 안 압력 저하)

실행 (터미널 2개)
  uvicorn app.api:app --port 8000
  streamlit run app/dashboard.py
API 주소를 바꾸려면 환경 변수 DASHBOARD_API_URL (기본 http://127.0.0.1:8000)
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("DASHBOARD_API_URL", "http://127.0.0.1:8000")   # Windows에서 localhost는 IPv6를 먼저 시도해 느리다
REFRESH_SECONDS = 2                   # 화면 자동 갱신 간격
LEVEL_COLORS = {0: "#4c9a52", 1: "#e08a1e", 2: "#b42318", -1: "#8a9199"}
FAULT_COLORS = {"압력 저하": "#b42318", "압력 과다": "#2b5c9e", "누설": "#7a3fb0"}
SPEED_OPTIONS = {"1초 = 1분": 60, "1초 = 10분": 600, "1초 = 1시간": 3600, "1초 = 6시간": 21600}
# 압력 저하 경보 때 볼 점검 후보 (일반적인 후보이며 장비 매뉴얼로 확인되지 않음)
DROP_CHECKLIST = [
    "헬륨 공급 압력과 레귤레이터 설정값",
    "냉각 유량 조절기(MFC) 동작",
    "척 표면 상태와 웨이퍼 밀착 (뒷면 헬륨이 새는지)",
    "냉각 배관 연결부 누설",
]

st.set_page_config(page_title="이온 밀 냉각 계통 대시보드", page_icon="⚙️", layout="wide")


# ---------- API 호출 ----------
def api_get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=5)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600)
def load_meta():
    """모델 설명·기준은 잘 안 바뀌므로 10분 캐시."""
    return api_get("/meta")


def level_badge(level: int, name: str) -> str:
    return (f"<span style='background:{LEVEL_COLORS[level]};color:white;padding:2px 10px;"
            f"border-radius:12px;font-size:0.85rem;font-weight:600'>{name}</span>")


def clock_text(clock: dict) -> str:
    state = "⏸ 멈춤" if clock["paused"] else "▶ 재생 중"
    return f"{state} · 평가 {clock['test_day']}일차 {clock['hour_of_day']:.1f}시 · 서버 시각 {clock['server_time'][11:]}"


try:
    meta = load_meta()
except requests.RequestException:
    st.error(f"API 서버({API_URL})에 연결할 수 없습니다. 먼저 `uvicorn app.api:app --port 8000`을 실행하세요.")
    st.stop()

tools = [status["tool"] for status in api_get("/status")["tools"]]

# ---------- 사이드바: 화면 선택, 재생 조절 ----------
st.sidebar.title("⚙️ 이온 밀 냉각 계통")
st.sidebar.caption(f"정비 판단 지원 · 프로토타입 · API {API_URL}")
page = st.sidebar.radio("화면", ["전체 장비 현황", "장비 상세", "점검 기록", "모델·기준 비교"], key="page")

st.sidebar.subheader("재생 (실시간 흉내)")
clock_now = api_get("/clock")
speed_label = st.sidebar.selectbox("속도", list(SPEED_OPTIONS),
                                   index=list(SPEED_OPTIONS.values()).index(clock_now["speed"]) if clock_now["speed"] in SPEED_OPTIONS.values() else 1)
if SPEED_OPTIONS[speed_label] != clock_now["speed"]:
    api_post("/replay", {"speed": SPEED_OPTIONS[speed_label]})
pause_column, play_column = st.sidebar.columns(2)
if pause_column.button("⏸ 멈춤", use_container_width=True):
    api_post("/replay", {"paused": True})
if play_column.button("▶ 재생", use_container_width=True):
    api_post("/replay", {"paused": False})
jump_day = st.sidebar.number_input("이동할 평가 일차", 0, int(clock_now["max_elapsed_hours"] // 24), value=int(clock_now["test_day"]))
if st.sidebar.button("이 날로 이동", use_container_width=True):
    api_post("/replay", {"elapsed_hours": jump_day * 24})
st.sidebar.info("서버가 평가 구간(장비별 기록 기간의 뒤 25%)을 시간 순서대로 흘려보냅니다. "
                "장비 상태는 최근 24시간에 도착한 기록으로 계산합니다.")
st.sidebar.warning("현재 모델은 **압력 저하**만 예측합니다. 압력 과다·누설은 아직 판정하지 않습니다.")


# ================= 1. 전체 장비 현황 =================
@st.fragment(run_every=REFRESH_SECONDS)
def overview_live():
    snapshot = api_get("/status")
    st.caption(clock_text(snapshot["clock"]))
    statuses = snapshot["tools"]
    counts = pd.Series([status["level"] for status in statuses]).value_counts()
    for column, (level, name) in zip(st.columns(4), [(2, "즉시 점검"), (1, "주의"), (0, "정상"), (-1, "판정 불가")]):
        column.metric(name, f"{counts.get(level, 0)}대")

    st.subheader("장비 20대")
    for row_start in range(0, len(statuses), 5):
        for column, status in zip(st.columns(5), statuses[row_start:row_start + 5]):
            with column.container(border=True):
                st.markdown(f"**{status['tool']}**  {level_badge(status['level'], status['level_name'])}", unsafe_allow_html=True)
                if status["level"] == -1:
                    st.caption("최근 24시간 식각 기록 없음")
                else:
                    st.caption(f"위험 점수 {status['score']:.2f} · 냉각 압력 z {status['pressure_z']:+.2f}")
                if st.button("상세 보기", key=f"open_{status['tool']}", use_container_width=True):
                    st.session_state.detail_tool = status["tool"]
                    st.session_state.page = "장비 상세"
                    st.rerun(scope="app")

    st.subheader("지금 경보 목록")
    alarms = [status for status in statuses if status["level"] >= 1]
    if not alarms:
        st.success("지금은 경보가 없습니다.")
        return
    alarm_rows = []
    for status in sorted(alarms, key=lambda item: item["score"], reverse=True):
        reasons = api_get(f"/tools/{status['tool']}/reasons")["reasons"]
        top_reasons = ", ".join([reason["feature"] for reason in reasons if reason["contribution"] > 0][:2])
        last_fault = status["last_fault"]
        alarm_rows.append({"장비": status["tool"], "등급": status["level_name"], "위험 점수": status["score"],
                           "주요 근거": top_reasons,
                           "마지막 고장": f"{last_fault['hours_ago']:.0f}시간 전 {last_fault['type']}" if last_fault else "평가 구간 없음"})
    st.dataframe(pd.DataFrame(alarm_rows), hide_index=True, use_container_width=True)


# ================= 2. 장비 상세 =================
@st.fragment(run_every=REFRESH_SECONDS)
def detail_live(tool: str, hours: int):
    snapshot = api_get("/status")
    status = next(item for item in snapshot["tools"] if item["tool"] == tool)
    st.caption(clock_text(snapshot["clock"]))

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### {tool}  {level_badge(status['level'], status['level_name'])}", unsafe_allow_html=True)
        if status["level"] == -1:
            st.caption("최근 24시간 식각 기록이 없어 판정하지 않습니다.")
        else:
            st.metric("위험 점수 (최근 24시간 평균)", f"{status['score']:.2f}")
            st.caption(f"냉각 압력 z {status['pressure_z']:+.2f} · 냉각 유량 z {status['flow_z']:+.2f} · 구간 {status['segments_24h']}개")
    with right:
        st.markdown("**경보 근거 — 점수를 올린 입력 / 내린 입력 (최근 24시간)**")
        reasons = api_get(f"/tools/{tool}/reasons")["reasons"]
        if not reasons:
            st.caption("근거를 계산할 기록이 없습니다.")
        else:
            shown = reasons[:4] + reasons[-2:]
            figure = go.Figure(go.Bar(x=[reason["contribution"] for reason in shown], y=[reason["feature"] for reason in shown],
                                      orientation="h",
                                      marker_color=["#b42318" if reason["contribution"] > 0 else "#2b5c9e" for reason in shown]))
            figure.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                                 xaxis_title="점수 기여 (로그 오즈)")
            st.plotly_chart(figure, use_container_width=True, key="reasons_chart")

    history = api_get(f"/tools/{tool}/history", hours=hours, bucket_minutes=60)
    rows = pd.DataFrame(history["history"])
    if rows.empty:
        st.caption("이 기간에 도착한 기록이 없습니다.")
        return
    rows["day"] = rows.elapsed_hours / 24
    thresholds = meta["thresholds"]
    figure = go.Figure()
    figure.add_hrect(y0=thresholds["urgent"], y1=1, fillcolor=LEVEL_COLORS[2], opacity=0.08, line_width=0)
    figure.add_hrect(y0=thresholds["watch"], y1=thresholds["urgent"], fillcolor=LEVEL_COLORS[1], opacity=0.08, line_width=0)
    figure.add_trace(go.Scatter(x=rows.day, y=rows.score, mode="lines", name="위험 점수 (1시간 평균)", line=dict(color="#4f8fe0", width=2)))   # 밝은·어두운 테마 모두 보이는 파랑
    fault_rows = pd.DataFrame(history["faults"])
    if not fault_rows.empty:
        for fault_label, group in fault_rows.groupby("type"):
            figure.add_trace(go.Scatter(x=group.elapsed_hours / 24, y=[1.02] * len(group), mode="markers", name=f"고장: {fault_label}",
                                        marker=dict(symbol="x", size=10, color=FAULT_COLORS.get(fault_label, "#555"))))
    figure.update_layout(title=f"위험 점수 — 최근 {hours}시간 (주황 띠 = 주의, 빨강 띠 = 즉시 점검, × = 실제 고장)", height=320,
                         margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(range=[0, 1.08], title="점수"),
                         xaxis_title="평가 일차", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(figure, use_container_width=True, key="score_chart")

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=rows.day, y=rows.pressure_z, name="냉각 압력 z", line=dict(color="#c2410c")))
    figure.add_trace(go.Scatter(x=rows.day, y=rows.flow_z, name="냉각 유량 z", line=dict(color="#2b5c9e")))
    figure.add_hline(y=0, line_color="#bbb")
    figure.update_layout(title="냉각 센서 1시간 평균 (같은 레시피·스텝 평소 대비 z값)", height=270,
                         margin=dict(l=10, r=10, t=50, b=10), xaxis_title="평가 일차", legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(figure, use_container_width=True, key="sensor_chart")


if page == "전체 장비 현황":
    st.title("전체 장비 현황")
    st.caption(f"압력 저하 위험 (지도 학습, 정답 = 24시간 안 압력 저하) · {REFRESH_SECONDS}초마다 자동 갱신")
    overview_live()
    st.caption(f"등급 기준 — 주의: 점수 ≥ {meta['thresholds']['watch']:.2f} (운영 기준, 정밀도 ≥ 항상 경보의 2배) · "
               f"즉시 점검: 점수 ≥ {meta['thresholds']['urgent']:.2f} (4배)")

elif page == "장비 상세":
    st.title("장비 상세")
    default_tool = st.session_state.get("detail_tool", tools[0])
    select_column, window_column = st.columns([2, 1])
    tool = select_column.selectbox("장비", tools, index=tools.index(default_tool))
    hours = window_column.select_slider("보는 기간", options=[24, 72, 168, 336, 720], value=168, format_func=lambda value: f"{value // 24}일")
    st.session_state.detail_tool = tool
    detail_live(tool, hours)
    st.markdown("**추천 점검 항목 (압력 저하)**")
    for item in DROP_CHECKLIST:
        st.checkbox(item, key=f"check_{tool}_{item}")
    st.caption("일반적인 점검 후보이며, 장비 매뉴얼로 확인된 정비 절차는 아닙니다.")

elif page == "점검 기록":
    st.title("점검 기록")
    statuses = api_get("/status")["tools"]
    alarm_tools = [status["tool"] for status in statuses if status["level"] >= 1]
    with st.form("inspection_form", clear_on_submit=True):
        st.markdown("지금 재생 위치 기준 점검 결과 입력 (경보 등급·점수는 서버가 저장 순간 상태로 채웁니다)")
        tool = st.selectbox("장비", alarm_tools + [tool for tool in tools if tool not in alarm_tools],
                            help="지금 경보가 난 장비가 위에 옵니다.")
        result = st.radio("점검 결과", ["이상 없음", "부품 교체", "조정·청소", "기타"], horizontal=True)
        memo = st.text_area("메모", placeholder="예: 헬륨 레귤레이터 압력 낮음, 재설정")
        if st.form_submit_button("저장", type="primary"):
            saved = api_post("/inspections", {"tool": tool, "result": result, "memo": memo})
            st.success(f"{tool} 점검 결과를 저장했습니다 (저장 순간 등급: {saved['status']['level_name']}).")

    records = pd.DataFrame(api_get("/inspections"))
    st.subheader("저장된 기록")
    if records.empty:
        st.caption("아직 기록이 없습니다.")
    else:
        st.dataframe(records.rename(columns={"saved_at": "저장 시각", "tool": "장비", "test_day": "평가 일차", "elapsed_hours": "평가 경과(시간)",
                                             "alarm_level": "경보 등급", "day_score": "위험 점수", "result": "점검 결과", "memo": "메모"})
                     .drop(columns="id"), hide_index=True, use_container_width=True)

else:
    st.title("모델·기준 비교")
    left, right = st.columns(2)
    with left:
        st.markdown("**모델**")
        st.markdown(f"""
- 모델: {meta['model']} · **{meta['learning']}**
- 정답: {meta['label']}
- 나누기: {meta['split'].replace('~', '–')}
- 점수: 구간마다 계산, 상태는 최근 24시간 평균 (기준은 {meta['day_score']}으로 정함)
- 학습 {meta['rows']['fit']:,}행 · 보정 {meta['rows']['calib']:,}행 · 평가 {meta['rows']['test']:,}행 ({meta['test_days']}일)
""")
    with right:
        st.markdown("**경보 기준 (보정 구간에서 정함)**")
        st.markdown(f"""
- {meta['threshold_rule']['watch']} → **{meta['thresholds']['watch']:.3f}**
- {meta['threshold_rule']['urgent']} → **{meta['thresholds']['urgent']:.3f}**
- 보정 구간 장비-일 고장 비율(항상 경보 정밀도): {meta['thresholds']['calib_day_base_rate']:.1%}
""")

    st.subheader("평가 구간 성능 (장비-일 단위)")
    metric_table = pd.DataFrame([{"방식": name, "재현율": values["recall"], "정밀도": values["precision"],
                                  "하루 경보 수 (20대 합)": values["alarms_per_day"]} for name, values in meta["metrics_test_days"].items()])
    st.dataframe(metric_table.style.format({"재현율": "{:.1%}", "정밀도": "{:.1%}", "하루 경보 수 (20대 합)": "{:.1f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("재현율 = 압력 저하 전날(24시간 안)인 장비-일 중 경보를 낸 비율. 정밀도 = 경보를 낸 장비-일 중 실제로 그런 날의 비율. "
               "'항상 경보'의 정밀도가 우연 수준이다.")

    left, right = st.columns(2)
    with left:
        st.markdown("**장비별 성능 (주의 이상)**")
        per_tool = pd.DataFrame(meta["per_tool"]).rename(columns={
            "Tool": "장비", "fault_days": "고장 전날 수", "days": "평가 일수", "recall": "재현율", "precision": "정밀도",
            "alarms_per_day": "하루 경보 비율"})
        st.dataframe(per_tool.style.format({"재현율": "{:.0%}", "정밀도": "{:.0%}", "하루 경보 비율": "{:.0%}"}),
                     hide_index=True, use_container_width=True, height=420)
    with right:
        st.markdown("**입력별 계수 (표준화 후)**")
        coefficients = pd.Series({meta["feature_names"][feature]: value for feature, value in meta["coefficients"].items()}).sort_values()
        figure = go.Figure(go.Bar(x=coefficients.values, y=coefficients.index, orientation="h",
                                  marker_color=["#b42318" if value > 0 else "#2b5c9e" for value in coefficients.values]))
        figure.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="계수 (+ = 점수를 올림)")
        st.plotly_chart(figure, use_container_width=True)

    st.info("한계: 압력 저하 1종만 예측합니다. 고장 이력 입력(마지막 고장 이후 시간 등)의 영향이 커서, "
            "고장이 잦은 장비에서 경보가 많아집니다. 성능은 장비-일 단위이며 사건 단위 채점과 다릅니다.")
