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

  function select(record) {
    selected?.button.setAttribute("aria-pressed", "false");
    selected = record;
    record.button.setAttribute("aria-pressed", "true");
    detail.dataset.status = record.status;
    find("#ehp-detail-title").textContent = `${record.id} · 냉각 압력`;
    const badge = find("#ehp-detail-status");
    badge.textContent = statuses[record.status];
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
    find("#ehp-detail-note").textContent = descriptions[record.fault ?? record.status];
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
    const labels = ["−24h", "−18h", "−12h", "−6h", "지금"].map((label, index) => {
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
