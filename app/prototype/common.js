// 세 가지 홈페이지 버전(v1·v2·v3)이 함께 쓰는 데모 데이터와 그리기 도우미.
// 데이터는 전부 데모값이다. 단, FAULT_RECORD_COUNTS는 PHM 2018 학습 데이터의 실제 고장 기록 건수(반복 기록 묶기 전)다.

// ---------- 데모 데이터 (고정 시드라 새로고침해도 같은 화면) ----------
let randomSeed = 20181122;
function seededRandom() {
  randomSeed = (randomSeed * 1664525 + 1013904223) % 4294967296;
  return randomSeed / 4294967296;
}
function noise(amplitude) { return (seededRandom() - 0.5) * 2 * amplitude; }

const LEVEL_INFO = {
  0: { name: '정상', short: '정상', color: 'var(--normal)', badge: 'ok' },
  1: { name: 'P3 관찰 강화', short: 'P3', color: 'var(--watch)', badge: 'p3' },
  2: { name: 'P2 우선 점검', short: 'P2', color: 'var(--urgent)', badge: 'p2' },
  [-1]: { name: '판단 보류', short: '보류', color: 'var(--hold)', badge: 'hold' },
};
const WATCH_THRESHOLD = -1.2;    // 관찰 임계값 (압력 편차)
const URGENT_THRESHOLD = -2.0;   // 점검 임계값
const POINT_COUNT = 96;          // 24시간 × 15분

// 장비 20대: 01M01 ~ 10M02. 시나리오가 있는 장비만 지정, 나머지는 정상
const SCENARIOS = {
  '04M02': { level: 2, fault: '냉각 압력 저하', check: '유량 동반 하락 → 공급·유량 제어 이력', reason: '같은 스텝에서 냉각 압력이 정상 범위 아래로 3회 연속 운전 동안 머묾. 유량은 비슷해 <b>압력 계측·밀봉 상태</b>를 먼저 확인.' },
  '07M01': { level: 1, fault: '냉각 압력 과다', check: '유량 설정·단계 변경 대조 → 냉각 펌프 알람 이력', reason: '같은 유량에서 압력이 운전마다 조금씩 상승. 아직 관찰 임계값 수준이라 <b>다음 동일 단계에서 재비교</b>.' },
  '09M02': { level: 1, fault: '냉각 헬륨 누설', check: '아르곤 유량 변경 확인 → 웨이퍼 안착·밀봉 링', reason: '냉각 압력 하락과 함께 챔버 압력이 소폭 상승. 아르곤 유량 변경 구간과 겹치는지 <b>먼저 확인</b>.' },
  '02M01': { level: -1, fault: '-', check: '-', reason: '처음 보는 레시피라 정상 참고 운전이 2회뿐. 점수를 채우지 않고 <b>판단 보류</b>로 표시.' },
};

const tools = [];
for (let toolNumber = 1; toolNumber <= 10; toolNumber++) {
  for (const module of ['M01', 'M02']) {
    const toolName = String(toolNumber).padStart(2, '0') + module;
    const scenario = SCENARIOS[toolName] || { level: 0, fault: '-', check: '-', reason: '최근 24시간 냉각 압력이 같은 레시피·스텝의 정상 범위 안에 있음.' };
    const deviation = [];
    const riskScore = [];
    for (let pointIndex = 0; pointIndex < POINT_COUNT; pointIndex++) {
      const progress = pointIndex / (POINT_COUNT - 1);
      let drift = 0;
      if (scenario.level === 2) drift = -2.6 * Math.pow(Math.max(0, progress - 0.55) / 0.45, 1.6);
      if (scenario.level === 1) drift = -1.45 * Math.pow(Math.max(0, progress - 0.6) / 0.4, 1.3);
      deviation.push(drift + noise(0.35));
      riskScore.push(Math.min(1, Math.max(0, 0.12 + (-drift) * 0.33 + noise(0.05))));
    }
    tools.push({ name: toolName, ...scenario, deviation, riskScore });
  }
}

// 추가 데모값 (위 장비 곡선 뒤에 만들어 기존 값이 바뀌지 않게 한다)
for (const tool of tools) {
  tool.etchRatio = tool.level === -1 ? 0.42 : 0.7 + seededRandom() * 0.25;   // 최근 24시간 중 식각 운전 비율
  tool.lastRun = 1 + Math.floor(seededRandom() * 40);                       // 마지막 운전 후 지난 분
  // 최근 14일 일별 최고 위험 점수 (마지막 칸 = 오늘)
  tool.dailyRisk = Array.from({ length: 14 }, (unused, dayIndex) => {
    let base = 0.1 + seededRandom() * 0.15;
    if (tool.level === 2) base += Math.max(0, dayIndex - 9) * 0.17;
    if (tool.level === 1) base += Math.max(0, dayIndex - 10) * 0.11;
    if (tool.level === -1 && dayIndex >= 12) return null;
    return Math.min(1, base);
  });
}

const events = [
  { time: '14:32', tool: '04M02', level: 2 },
  { time: '13:05', tool: '07M01', level: 1 },
  { time: '11:48', tool: '09M02', level: 1 },
  { time: '09:20', tool: '04M02', level: 1 },
  { time: '02:15', tool: '05M01', level: 1, fault: '냉각 압력 저하', check: '해제됨 (다음 단계 정상 복귀)' },
];
const alarmCountsByHour = Array.from({ length: 24 }, () => ({ watch: 0, urgent: 0 }));
alarmCountsByHour[2].watch = 1; alarmCountsByHour[9].watch = 1; alarmCountsByHour[11].watch = 1;
alarmCountsByHour[13].watch = 1; alarmCountsByHour[14].urgent = 1;

// 실제 값: 학습 데이터 고장 기록 1,236건 (반복 기록 묶기 전)
const FAULT_RECORD_COUNTS = [
  { label: '냉각 압력 저하', count: 756, color: '#ff5d5d' },
  { label: '냉각 압력 과다', count: 340, color: '#4f8dff' },
  { label: '냉각 헬륨 누설', count: 139, color: '#b07cff' },
  { label: '기타', count: 1, color: '#8a9199' },
];

// 평가 방식별 성능 자리 (모델 평가 전이라 미측정)
const MODEL_ROWS = [
  { model: '규칙 기준선', learning: '학습 없음', label: '-' },
  { model: 'Isolation Forest', learning: '비지도', label: '라벨 미사용 (고장 사건표로 사후 검증)' },
  { model: 'Random Forest', learning: '지도', label: 'L ≤ T−t ≤ H 안 대상 정지 여부' },
];

// ---------- 그리기 도우미 ----------
const SVG_NS = 'http://www.w3.org/2000/svg';
function svgElement(tagName, attributes) {
  const element = document.createElementNS(SVG_NS, tagName);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}
function linePath(values, width, height, minValue, maxValue, padding = 0) {
  return values.map((value, index) => {
    const xPosition = padding + (index / (values.length - 1)) * (width - padding);
    const yPosition = height - ((value - minValue) / (maxValue - minValue)) * height;
    return (index === 0 ? 'M' : 'L') + xPosition.toFixed(1) + ' ' + yPosition.toFixed(1);
  }).join(' ');
}
function svgText(parent, x, y, text, attributes = {}) {
  const label = svgElement('text', { x, y, fill: 'var(--muted)', 'font-size': 11, ...attributes });
  label.textContent = text;
  parent.appendChild(label);
  return label;
}
function findTool(toolName) { return tools.find(candidate => candidate.name === toolName); }
function countByLevel(level) { return tools.filter(tool => tool.level === level).length; }

// 냉각 압력 편차 추세 + 정상 범위 + 임계값 (세 버전 공통 상세 차트)
function drawDeviationChart(chart, tool, options = {}) {
  chart.innerHTML = '';
  const chartWidth = 600, chartHeight = options.height || 190, leftPadding = 30, minValue = -3.5, maxValue = 1.5;
  const toY = value => chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;
  for (const gridValue of [-3, -2, -1, 0, 1]) {
    chart.appendChild(svgElement('line', { x1: leftPadding, x2: chartWidth, y1: toY(gridValue), y2: toY(gridValue), stroke: 'var(--grid)', 'vector-effect': 'non-scaling-stroke' }));
    svgText(chart, 0, toY(gridValue) + 4, gridValue);
  }
  chart.appendChild(svgElement('rect', { x: leftPadding, y: toY(0.6), width: chartWidth - leftPadding, height: toY(-0.6) - toY(0.6), fill: 'var(--band)' }));
  chart.appendChild(svgElement('line', { x1: leftPadding, x2: chartWidth, y1: toY(WATCH_THRESHOLD), y2: toY(WATCH_THRESHOLD), stroke: 'var(--watch)', 'stroke-dasharray': '6 5', 'vector-effect': 'non-scaling-stroke' }));
  chart.appendChild(svgElement('line', { x1: leftPadding, x2: chartWidth, y1: toY(URGENT_THRESHOLD), y2: toY(URGENT_THRESHOLD), stroke: 'var(--urgent)', 'stroke-dasharray': '6 5', 'vector-effect': 'non-scaling-stroke' }));
  if (tool.level !== -1) {
    const pathData = linePath(tool.deviation, chartWidth, chartHeight, minValue, maxValue, leftPadding);
    if (options.area) chart.appendChild(svgElement('path', { d: `${pathData} L${chartWidth} ${chartHeight} L${leftPadding} ${chartHeight} Z`, fill: 'var(--line)', opacity: 0.12 }));
    chart.appendChild(svgElement('path', { d: pathData, fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.6, 'vector-effect': 'non-scaling-stroke' }));
    chart.appendChild(svgElement('circle', { cx: chartWidth - 3, cy: toY(tool.deviation[POINT_COUNT - 1]), r: 4, fill: LEVEL_INFO[tool.level].color }));
  } else {
    svgText(chart, chartWidth / 2 - 60, chartHeight / 2, '정상 참고 운전 부족 · 판단 보류', { 'font-size': 13 });
  }
  ['−24h', '−18h', '−12h', '−6h', '지금'].forEach((labelText, labelIndex) => {
    svgText(chart, leftPadding + labelIndex * (chartWidth - leftPadding - 24) / 4, chartHeight + 17, labelText);
  });
}
