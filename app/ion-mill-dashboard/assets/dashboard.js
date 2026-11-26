
(() => {
  const root = document.querySelector("#equipment-heatmap");
  const find = (selector) => root.querySelector(selector);
  const statuses = { normal: "정상", watch: "관찰 강화", priority: "우선 점검", hold: "판단 보류" };
  const faultNames = { "low-pressure": "압력 저하", "high-pressure": "압력 과다", "helium-leak": "헬륨 누설" };
  const cases = {
    "02M01": { status: "hold", pressure: null },
    "03M01": { score: 0.07, pressure: -0.06 },
    "04M02": { status: "priority", fault: "low-pressure", score: 0.94, pressure: -2.42 },
    "07M01": { status: "watch", fault: "high-pressure", score: 0.57, pressure: 1.61 },
    "09M02": { status: "watch", fault: "helium-leak", score: 0.59, pressure: -0.28 },
  };
  const normalLimit = .60;
  const watchLimit = 1.20;
  const priorityLimit = 2;
  const chart = find("#ehp-trend");
  const detail = find(".ehp-detail");
  const dialog = find("#ehp-dialog");
  const model = find("#ehp-model");
  let selected = null;
  let frame = null;
  let opener = null;

  function format(value, decimals = 2) {
    return value.toFixed(decimals).replace("-", "−");
  }

  function readings(record, seed) {
    return Array.from({ length: 97 }, (_, index) => {
      const fraction = index / 96;
      if (index === 96) return record.pressure;
      const noise = Math.sin(index * 1.67 + seed * .83) * .19
        + Math.sin(index * 2.59 + seed) * .12 + Math.cos(index * .40) * .05;
      const drift = record.fault === "high-pressure" || record.fault === "low-pressure"
        ? fraction ** 2.45 : fraction ** 3;
      return noise * (1 - drift) + record.pressure * drift;
    });
  }

  function makeRecord(tool, module, index) {
    const id = tool + module;
    const record = {
      id, tool, module, status: "normal",
      score: (.06 + (index * 7 % 13) / 100),
      pressure: Number((Math.sin(index * 2.39) * .26).toFixed(2)),
      ...cases[id],
    };
    record.series = record.pressure === null ? [] : readings(record, index);
    return record;
  }

  function makeModule(record) {
    const button = document.createElement("button");
    const score = record.status === "hold" ? "보류" : record.score.toFixed(2);
    button.type = "button";
    button.className = "ehp-module";
    button.dataset.module = record.id;
    button.dataset.status = record.status;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `${record.id}, ${statuses[record.status]}${record.status === "hold" ? "" : `, 위험 점수 ${score}`}`);
    if (record.fault) button.setAttribute("aria-haspopup", "dialog");
    const name = document.createElement("span");
    name.textContent = record.module;
    const value = document.createElement("small");
    value.textContent = score;
    button.append(name, value);
    record.button = button;
    button.addEventListener("click", () => {
      select(record);
      if (record.fault) openModel(record);
    });
    return button;
  }

  function renderFloor() {
    let initial;
    for (let number = 1; number <= 10; number++) {
      const toolId = String(number).padStart(2, "0");
      const modules = ["M01", "M02"].map((module, column) => makeRecord(toolId, module, 2 * number + column));
      const tool = document.createElement("div");
      tool.className = "ehp-tool";
      const highlighted = modules.find((record) => record.fault);
      if (highlighted) tool.dataset.status = highlighted.status;

      const label = document.createElement("h3");
      label.className = "ehp-tool-name";
      label.textContent = `TOOL ${toolId}`;
      const core = document.createElement("div");
      core.className = "ehp-tool-core";
      core.setAttribute("aria-hidden", "true");
      const group = document.createElement("div");
      group.className = "ehp-modules";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", `TOOL ${toolId}`);
      for (const record of modules) {
        group.append(makeModule(record));
        if (record.id === "03M01") initial = record;
      }
      tool.append(label, core, group);
      find(number <= 5 ? "#ehp-row-top" : "#ehp-row-bottom").append(tool);
    }
    select(initial);
  }

  
  let detailMessages = {};

  const detailStatusKeys = {
    normal: "moduleStatusNormal",
    watch: "moduleStatusWatch",
    priority: "moduleStatusPriority",
    hold: "moduleStatusHold",
  };

  document.addEventListener("dashboardlanguagechange", (event) => {
    detailMessages = event.detail;
    detail.lang = document.documentElement.lang;
    if (selected) select(selected);
  });

  function select(record) {
    selected?.button.setAttribute("aria-pressed", "false");
    selected = record;
    record.button.setAttribute("aria-pressed", "true");
    detail.dataset.status = record.status;
    find("#ehp-detail-title").textContent =
      `${record.id} · ${detailMessages.moduleDetailPressure ?? "냉각 압력"}`;

    const badge = find("#ehp-detail-status");
    badge.textContent =
      detailMessages[detailStatusKeys[record.status]] ?? statuses[record.status];
    badge.dataset.status = record.status;
    const held = record.pressure === null;
    find("#ehp-pressure").textContent = held ? "—" : format(record.pressure);
    find("#ehp-pressure-unit").hidden = held;
    find("#ehp-chart-area").hidden = held;
    const descriptions = {
      hold: "판단에 필요한 데이터가 부족해 압력 수치와 추이를 표시하지 않습니다.",
      normal: "최근 24시간 냉각 압력은 데모 정상 범위 안에 있습니다.",
      "low-pressure": "냉각 압력이 하한 점검 임계값 아래로 내려갔습니다.",
      "high-pressure": "냉각 압력이 상한 관찰 임계값을 넘었습니다.",
      "helium-leak": "냉각 압력은 정상 범위입니다. 별도 헬륨 누설 신호에 따른 관찰 예시입니다.",
    };
    const noteKey = record.fault ?? record.status;
    find("#ehp-detail-note").textContent =
      detailMessages.moduleDetailNotes?.[noteKey] ?? descriptions[noteKey];
    renderTrend();
  }

  function renderTrend() {
    if (!selected?.series.length) return;
    const width = chart.clientWidth;
    const height = chart.clientHeight;
    if (width < 130 || height < 90) return;

    const high = selected.fault === "high-pressure";
    const direction = high ? 1 : -1;
    const [min, max] = high ? [-1, 3] : [-3, 1];
    const left = 25, right = width - 11, top = 10, bottom = height - 24;
    const y = (value) => top + (max - value) / (max - min) * (bottom - top);
    const x = (fraction) => left + (right - left) * fraction;

    chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
    chart.setAttribute("aria-label", `${selected.id} 최근 24시간 표준화 냉각 압력. 마지막 값 ${format(selected.pressure)} 시그마. 정상 범위 ±0.60 시그마.`);
    const grid = [];
    for (let value = min; value <= max; value++) {
      grid.push(`<line x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}" class="ehp-grid-line"/>
        <text x="${left - 5}" y="${y(value) + 3.5}" text-anchor="end">${format(value, 0)}</text>`);
    }

    const limits = [[watchLimit, "watch"], [priorityLimit, "priority"]];
    const markers = limits.map(([limit, name]) => {
      const level = direction * limit;
      find(`#ehp-${name}-label`).textContent = `${format(level)} σ`;
      return `<line x1="${left}" x2="${right}" y1="${y(level)}" y2="${y(level)}" class="ehp-limit ehp-limit-${name}"/>`;
    });
    const labels = ["−24h", "−18h", "−12h", "−6h", detailMessages.moduleChartNow ?? "지금"].map((label, index) => {
      const anchor = index === 0 ? "start" : index === 4 ? "end" : "middle";
      return `<text x="${x(index / 4)}" y="${height - 7}" text-anchor="${anchor}">${label}</text>`;
    });
    const points = selected.series.map((value, index) =>
      `${x(index / (selected.series.length - 1)).toFixed(2)},${y(value).toFixed(2)}`).join(" ");
    chart.innerHTML = `<rect class="ehp-range" x="${left}" y="${y(normalLimit)}" width="${right - left}" height="${y(-normalLimit) - y(normalLimit)}"/>
      ${grid.join("")}${markers.join("")}${labels.join("")}
      <polyline class="ehp-trace" points="${points}"/>
      <circle cx="${right}" cy="${y(selected.pressure)}" r="3.2" fill="var(--status)"/>`;
  }

  function openModel(record) {
    if (!Object.hasOwn(faultNames, record.fault)) return;
    opener = record.button;
    find("#ehp-dialog-title").textContent = `${record.id} · ${faultNames[record.fault]}`;
    frame = document.createElement("iframe");
    frame.title = `${record.id} 이온 밀 챔버 절개도`;
    frame.src = `./ion-mill.html?embed=1&fault=${record.fault}`;
    model.replaceChildren(frame);
    dialog.showModal();
  }

  dialog.addEventListener("close", () => {
    if (dialog.open) return;
    model.replaceChildren();
    frame = null;
    opener?.focus({ preventScroll: true });
    opener = null;
  });

  window.addEventListener("message", (event) => {
    if (!dialog.open || !frame || event.source !== frame.contentWindow) return;
    if (event.data?.type === "ion-mill:close") {
      dialog.close();
    } else if (event.data?.type === "ion-mill:resize") {
      const { height } = event.data;
      if (Number.isInteger(height) && height >= 300 && height <= 600) frame.style.height = `${height}px`;
    }
  });

  renderFloor();
  new ResizeObserver(renderTrend).observe(chart);
})();



(() => {
                const chart = document.querySelector("#fleet-status-chart");
                const series = chart.querySelector("#fleet-status-series");
                const description = chart.querySelector("#fleet-status-description");
                const segments = [
                  { key: "running", label: "Running", count: 104, color: "green", x: 8, y: 169, elbow: 94 },
                  { key: "standby", label: "Standby", count: 10, color: "blue", x: 8, y: 35, elbow: 112 },
                  { key: "warning", label: "Warning", count: 10, color: "amber", x: 322, y: 61, elbow: 246 },
                  { key: "critical", label: "Critical", count: 4, color: "red", x: 322, y: 119, elbow: 262, offset: 7 },
                ];
                const total = segments.reduce((sum, segment) => sum + segment.count, 0);
                document.querySelector("#fleet-status-total").textContent = total;
                const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
                const cx = 163, cy = 102, radius = 62;
                let startAngle = -12;

                function vector(angle, length) {
                  const radians = angle * Math.PI / 180;
                  return [Math.cos(radians) * length, Math.sin(radians) * length];
                }

                series.innerHTML = segments.map((segment) => {
                  const { key, label, count, color, x, y, elbow, offset = 0 } = segment;
                  const share = count / total;
                  const sweep = share * 360;
                  const mid = startAngle + sweep / 2;
                  const [dx, dy] = vector(mid, offset);
                  const [startX, startY] = vector(startAngle, radius);
                  startAngle += sweep;
                  const [endX, endY] = vector(startAngle, radius);
                  const [edgeX, edgeY] = vector(mid, radius + offset);
                  const [bendX, bendY] = vector(mid, radius + offset + 13);
                  const anchor = x < cx ? "start" : "end";
                  const ink = `rgb(var(--c-accent-${color}))`;

                  return `<g>
                    <path d="M ${cx + dx} ${cy + dy} l ${startX} ${startY} a ${radius} ${radius} 0 ${sweep > 180 ? 1 : 0} 1 ${endX - startX} ${endY - startY} Z"
                          fill="${ink}" stroke="rgb(var(--c-panel))" stroke-width="1.5"/>
                    <polyline points="${cx + edgeX},${cy + edgeY} ${cx + bendX},${cy + bendY} ${elbow},${y} ${x},${y}"
                              fill="none" stroke="${ink}" stroke-width="1.4" stroke-linejoin="round"/>
                    <circle cx="${x}" cy="${y}" r="2.5" fill="${ink}"/>
                    <text x="${x}" y="${y - 8}" text-anchor="${anchor}" fill="rgb(var(--c-on-surface))"
                          font-size="19" font-weight="700">${percent.format(share)}</text>
                    <text x="${x}" y="${y + 16}" text-anchor="${anchor}" fill="rgb(var(--c-on-surface-variant))"
                          font-size="12"><tspan data-i18n="${key}">${label}</tspan> · ${count}</text>
                  </g>`;
                }).join("");

                const labels = series.querySelectorAll("[data-i18n]");
                function describeFleet() {
                  description.textContent = segments.map(({ count }, index) =>
                    `${labels[index].textContent} · ${count} (${percent.format(count / total)})`
                  ).join(", ");
                }

                document.addEventListener("dashboardlanguagechange", describeFleet);
                describeFleet();
              })();


(function () {
      var KEY = 'Wafers-theme';                    // localStorage 저장 키
      var root = document.documentElement;           // <html> 요소
      var buttons = document.querySelectorAll('[data-theme-btn]'); // 테마 버튼들

      function applyTheme(theme) {
        root.setAttribute('data-theme', theme);      // CSS 변수 세트 교체 → 전체 색 변경
        try { localStorage.setItem(KEY, theme); } catch (e) {} // 선택 저장 (저장 불가 환경은 무시)
        buttons.forEach(function (btn) {
          btn.setAttribute('aria-pressed', String(btn.dataset.themeBtn === theme)); // 현재 테마 버튼만 활성 표시
        });
        document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } })); // 다른 스크립트(히트맵 등)용 알림
      }

      buttons.forEach(function (btn) {
        btn.addEventListener('click', function () { applyTheme(btn.dataset.themeBtn); }); // 클릭 시 해당 테마 적용
      });

      applyTheme(root.getAttribute('data-theme') || 'dark'); // 초기 상태를 버튼 표시에 동기화
    })();


(() => {
    const translations = {
      en: {
        enterprise: "Enterprise",
        operations: "Operation",
        overview: "Overview",
        equipment: "Equipment Layout",
        last24h: "Last 24 Hours",
        last7d: "Last 7 Days",
        last30d: "Last 30 Days",
        totalEquipment: "Total Equipment",
        equipmentMonitoring: "Equipment Monitoring",
        moduleStatusNormal: "Normal",
        moduleStatusWatch: "Increased Monitoring",
        moduleStatusPriority: "Priority Inspection",
        moduleStatusHold: "Assessment Pending",
        moduleRiskScoreNote: "Number on each module · Demo risk score (0–1)",
        moduleDetailPressure: "Cooling Pressure",
        moduleDetailDeviation: "Recent cooling pressure deviation (σ)",
        moduleChartNormalRange: "Normal Range",
        moduleChartWatch: "Watch",
        moduleChartInspect: "Inspection",
        moduleChartNow: "Now",
        moduleDetailNotes: {
          hold: "There is not enough data to show the pressure reading or trend.",
          normal: "Cooling pressure stayed within the demo normal range over the last 24 hours.",
          "low-pressure": "Cooling pressure fell below the lower inspection threshold.",
          "high-pressure": "Cooling pressure exceeded the upper watch threshold.",
          "helium-leak": "Cooling pressure is within the normal range. A separate helium leak signal triggered monitoring.",
        },
        demodata: "Demo Data",
        ionmillprocessroute: "Ion-Mill-Process-Route",
        multiSeriesTelemetryTrend: "Multi Series Telmetry Trend",
        telemetrySummary24h: "24-hour power load, thermal index, and harmonic vibration",
        telemetrySummary7d: "7-day power load, thermal index, and harmonic vibration",
        telemetrySummary30d: "30-day power load, thermal index, and harmonic vibration",
        equipmentStatus: "Equipment Status",
        livefleetdistribution: "Live-Fleet distribution",
        running: "Running",
        standby: "Stand-By",
        warning: "Warning",
        critical: "Critical",
        activeEquipment: "Active Equipment",
        fleetTotalLabel: "Total equipment:",
        fleetTotalUnit: " units",
        warningEvents: "Warning Events",
        systemAvailability: "System Availability",
        health: "Cluster Health",
        ingestion: "Ingestion-Rate",
        nominal: "Nominal",
        VP: "VP Operations",
        logout: "Log out",
        load: "Load (kW)",
        thermal: "Thermal (°C)",
        vibration: "Vibration",

      },
      ko: {
        enterprise: "운영 팀",
        operations: "운영 관리",
        overview: "운영 현황",
        equipment: "장비 배치도",
        last24h: "최근 24시간",
        last7d: "최근 7일",
        last30d: "최근 30일",
        totalEquipment: "전체 장비",
        equipmentMonitoring: "설비 모니터링",
        moduleStatusNormal: "정상",
        moduleStatusWatch: "관찰 강화",
        moduleStatusPriority: "우선 점검",
        moduleStatusHold: "판단 보류",
        moduleRiskScoreNote: "모듈 숫자 · 데모 위험 점수(0–1)",
        moduleDetailPressure: "냉각 압력",
        moduleDetailDeviation: "최근 냉각 압력 편차(σ)",
        moduleChartNormalRange: "정상 범위",
        moduleChartWatch: "관찰",
        moduleChartInspect: "점검",
        moduleChartNow: "지금",
        moduleDetailNotes: {
          hold: "판단에 필요한 데이터가 부족해 압력 수치와 추이를 표시하지 않습니다.",
          normal: "최근 24시간 냉각 압력은 데모 정상 범위 안에 있습니다.",
          "low-pressure": "냉각 압력이 하한 점검 임계값 아래로 내려갔습니다.",
          "high-pressure": "냉각 압력이 상한 관찰 임계값을 넘었습니다.",
          "helium-leak": "냉각 압력은 정상 범위입니다. 별도 헬륨 누설 신호에 따른 관찰 예시입니다.",
        },
        demodata: "데모 데이터",
        ionmillprocessroute: "이온 밀 공정 통로",
        multiSeriesTelemetryTrend: "시간별 설비 계측 데이터",
        telemetrySummary24h: "최근 24시간 전력 부하, 열 지수 및 고조파 진동",
        telemetrySummary7d: "최근 7일 전력 부하, 열 지수 및 고조파 진동",
        telemetrySummary30d: "최근 30일 전력 부하, 열 지수 및 고조파 진동",
        equipmentStatus: "설비 상태",
        livefleetdistribution: "실시간 설비 분포",
        running: "가동 중",
        standby: "대기 중",
        warning: "주의 요망",
        critical: "긴급 조치 필요",
        activeEquipment: "가동 중인 설비",
        fleetTotalLabel: "총 설비:",
        fleetTotalUnit: "대",
        warningEvents: "주의 발생 이력",
        systemAvailability: "시스템 가용률",
        health: "운영체제 상태",
        ingestion: "데이터 전송 속도",
        nominal: "정상",
        VP: "운영 책임자",
        logout: "로그아웃",
        load: "부하 (kW)",
        thermal: "온도 (°C)",
        vibration: "진동",

      },
    };

    const button = document.querySelector("#language-toggle");
    let language = "en";
      const telemetryRange = document.querySelector("#time-range");
const telemetrySummary = document.querySelector("#telemetry-summary");
const telemetryChart = document.querySelector("#telemetry-chart");

const telemetryPeriods = {
  "24h": { hours: 24, step: 1, ticks: [24, 18, 12, 6, 0], key: "telemetrySummary24h" },
  "7d": { hours: 168, step: 6, ticks: [7, 5, 3, 1, 0], key: "telemetrySummary7d" },
  "30d": { hours: 720, step: 24, ticks: [30, 20, 10, 0], key: "telemetrySummary30d" },
};

const telemetrySamples = Array.from({ length: 721 }, (_, hour) => [
  69 + 9 * Math.sin(hour / 6) + 8 * Math.sin(hour / 53),
  56 + 7 * Math.sin(hour / 9) + 6 * Math.cos(hour / 48),
  42 + 5 * Math.cos(hour / 4) + 5 * Math.sin(hour / 27),
]);

function renderTelemetry() {
  const { hours, step, ticks, key } = telemetryPeriods[telemetryRange.value];
  const messages = translations[language];
  const samples = telemetrySamples
    .slice(-(hours + 1))
    .filter((_, index) => index % step === 0);
  const unit = hours === 24
    ? (language === "ko" ? "시간" : "h")
    : (language === "ko" ? "일" : "d");

  telemetrySummary.textContent = messages[key];
  telemetryChart.setAttribute("aria-label", messages[key]);

  telemetryChart.querySelectorAll("polyline").forEach((line, series) => {
    const points = samples.map((sample, index) => {
      const x = 36 + 629 * index / (samples.length - 1);
      const y = 134 - 114 * (sample[series] - 25) / 75;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    line.setAttribute("points", points.join(" "));
  });

  telemetryChart.querySelector("#telemetry-time-axis").innerHTML = ticks.map((offset) => {
    const x = 36 + 629 * (1 - offset / ticks[0]);
    const text = offset === 0 ? messages.moduleChartNow : `−${offset}${unit}`;
    return `<text x="${x}" y="156">${text}</text>`;
  }).join("");
}

telemetryRange.addEventListener("change", renderTelemetry);
    function applyLanguage() {
      document.documentElement.lang = language;

      for (const element of document.querySelectorAll("[data-i18n]")) {
        const text = translations[language][element.dataset.i18n];
        if (text !== undefined) element.textContent = text;
      }

      button.textContent = language === "en" ? "한국어" : "English";
      renderTelemetry();
      document.dispatchEvent(new CustomEvent("dashboardlanguagechange", {
        detail: translations[language],
      }));
    }

    button.addEventListener("click", () => {
      language = language === "en" ? "ko" : "en";
      applyLanguage();
    });

    applyLanguage();
  })();

  window.addEventListener("pagehide", (event) => {
  if (event.persisted) {
    document.documentElement.style.visibility = "hidden";
  }
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    window.location.reload();
  }
});