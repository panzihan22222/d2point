import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const FETCH_TIMEOUT_MS = 45000;
const MIN_POLYGON_VERTICES = 3;
const CLOSE_DISTANCE_PX = 10;
const GRID_CELL_MIN = 0.01;
const GRID_CELL_MAX = 5.0;
const GRID_CELL_EFFECTIVE_MIN = 0.109;

const ui = {
  gridCell: document.getElementById('gridCell'),
  basePercentile: document.getElementById('basePercentile'),
  modeView: document.getElementById('modeView'),
  modeDraw: document.getElementById('modeDraw'),
  finishPolygon: document.getElementById('finishPolygon'),
  undoPoint: document.getElementById('undoPoint'),
  clearSelection: document.getElementById('clearSelection'),
  estimateBtn: document.getElementById('estimateBtn'),
  exportBtn: document.getElementById('exportBtn'),
  resultBox: document.getElementById('resultBox'),
  hud: document.getElementById('hud'),
  canvas: document.getElementById('viewport'),
  overlayCanvas: document.getElementById('overlayCanvas')
};

let scene;
let camera;
let renderer;
let controls;
let overlayCtx;

let cloud;
let selectedCloud;

let cloudMeta;
let pointPositions = null;
let selectedIndices = [];

let drawMode = false;
let polygonDrawing = false;
let polygonPoints = [];
let hoverPoint = null;

init().catch((err) => {
  ui.hud.textContent = `初始化失败: ${err}`;
  ui.resultBox.textContent = String(err);
});

async function init() {
  setHud('加载元数据...');
  cloudMeta = await fetchJson('/api/meta');
  ui.gridCell.value = cloudMeta.default_grid_cell ?? 0.2;
  ui.basePercentile.value = cloudMeta.default_base_percentile ?? 0.0;
  normalizeParameterInputs();

  setupThree();

  setHud('加载点云二进制...');
  const buffer = await fetchArrayBuffer('/api/pointcloud.bin');
  const parsed = parsePointcloudBuffer(buffer);

  setHud('构建点云渲染对象...');
  createPointCloud(parsed.positions, parsed.colors, cloudMeta);

  hookUiEvents();
  setDrawMode(false);

  setHud(`已加载 ${cloudMeta.point_count.toLocaleString()} 点`);
  ui.resultBox.textContent = '切换到多边形模式，逐点点击绘制边界并计算体积。';

  animate();
}

function setupThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f7fb);

  const bboxMin = new THREE.Vector3(...cloudMeta.bbox_min);
  const bboxMax = new THREE.Vector3(...cloudMeta.bbox_max);
  const center = new THREE.Vector3(...cloudMeta.center);
  const size = new THREE.Vector3().subVectors(bboxMax, bboxMin);
  const diag = size.length();

  camera = new THREE.PerspectiveCamera(60, 1, 0.01, Math.max(1000, diag * 20));
  camera.position.set(center.x + diag * 0.7, center.y - diag * 0.8, center.z + diag * 0.5);
  camera.lookAt(center);

  renderer = new THREE.WebGLRenderer({ canvas: ui.canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.copy(center);

  overlayCtx = ui.overlayCanvas.getContext('2d');

  const ambient = new THREE.AmbientLight(0xffffff, 0.9);
  scene.add(ambient);

  const dir = new THREE.DirectionalLight(0xffffff, 0.5);
  dir.position.set(1, -1, 2);
  scene.add(dir);

  const box3 = new THREE.Box3(bboxMin, bboxMax);
  const helper = new THREE.Box3Helper(box3, 0x8aa4c8);
  scene.add(helper);

  onResize();
  window.addEventListener('resize', onResize);
}

function onResize() {
  const w = ui.canvas.clientWidth;
  const h = ui.canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();

  ui.overlayCanvas.width = Math.max(1, Math.floor(w));
  ui.overlayCanvas.height = Math.max(1, Math.floor(h));
  renderOverlay();
}

function parsePointcloudBuffer(buffer) {
  if (buffer.byteLength < 12) {
    throw new Error('点云数据头长度不足');
  }

  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'PCD1') {
    throw new Error('点云数据格式不匹配');
  }

  const count = view.getUint32(4, true);
  const hasColor = view.getUint32(8, true) === 1;

  const posOffset = 12;
  const posLength = count * 3;
  const needBytes = posOffset + posLength * 4 + (hasColor ? count * 3 : 0);
  if (buffer.byteLength < needBytes) {
    throw new Error(`点云数据不完整: 期望 ${needBytes} bytes, 实际 ${buffer.byteLength} bytes`);
  }

  const positions = new Float32Array(buffer, posOffset, posLength);
  let colors = null;
  if (hasColor) {
    const colorOffset = posOffset + posLength * 4;
    colors = new Uint8Array(buffer, colorOffset, count * 3);
  }
  return { count, positions, colors };
}

function createPointCloud(positions, colors, meta) {
  pointPositions = positions;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  if (colors) {
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
  } else {
    const count = positions.length / 3;
    const genColors = new Float32Array(count * 3);
    const zMin = meta.bbox_min[2];
    const zMax = meta.bbox_max[2];
    const dz = Math.max(zMax - zMin, 1e-6);
    for (let i = 0; i < count; i += 1) {
      const z = positions[i * 3 + 2];
      const t = (z - zMin) / dz;
      genColors[i * 3 + 0] = 0.1 + 0.6 * t;
      genColors[i * 3 + 1] = 0.3 + 0.5 * (1.0 - Math.abs(t - 0.5) * 2.0);
      genColors[i * 3 + 2] = 0.9 - 0.5 * t;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(genColors, 3));
  }

  const sizeDiag = new THREE.Vector3(
    meta.bbox_max[0] - meta.bbox_min[0],
    meta.bbox_max[1] - meta.bbox_min[1],
    meta.bbox_max[2] - meta.bbox_min[2]
  ).length();
  const pointSize = Math.max(0.03, sizeDiag * 0.0012);

  cloud = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      opacity: 0.9,
      transparent: true
    })
  );
  scene.add(cloud);

  selectedCloud = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.PointsMaterial({
      size: pointSize * 1.4,
      color: 0xff4040,
      opacity: 0.95,
      transparent: true
    })
  );
  scene.add(selectedCloud);
}

function hookUiEvents() {
  ui.modeView.addEventListener('click', () => {
    setDrawMode(false);
  });

  ui.modeDraw.addEventListener('click', () => {
    setDrawMode(true);
  });

  ui.finishPolygon.addEventListener('click', () => {
    finalizePolygonSelection();
  });

  ui.undoPoint.addEventListener('click', () => {
    undoPolygonPoint();
  });

  ui.clearSelection.addEventListener('click', () => {
    clearSelection();
  });

  ui.gridCell.addEventListener('blur', () => {
    normalizeParameterInputs();
  });

  ui.basePercentile.addEventListener('blur', () => {
    normalizeParameterInputs();
  });

  ui.estimateBtn.addEventListener('click', async () => {
    try {
      if (selectedIndices.length === 0) {
        throw new Error('请先在多边形模式下选中点云');
      }
      const params = getComputationParams();
      setHud('正在计算体积...');
      const payload = {
        indices: selectedIndices,
        grid_cell: params.gridCell,
        base_percentile: params.basePercentile
      };
      const result = await postJson('/api/estimate_volume', payload);
      renderResult(result);
      setHud(`完成: ${result.selected_point_count.toLocaleString()} 点`);
    } catch (err) {
      ui.resultBox.textContent = `计算失败: ${err}`;
      setHud('计算失败');
    }
  });

  ui.exportBtn.addEventListener('click', async () => {
    try {
      if (selectedIndices.length === 0) {
        throw new Error('请先在多边形模式下选中点云');
      }
      setHud('正在导出框内点云...');
      const result = await postJson('/api/export_selection', { indices: selectedIndices });
      ui.resultBox.textContent = [
        '导出成功',
        `选择模式: ${result.selection_type || 'manual'}`,
        `点数: ${result.selected_point_count.toLocaleString()}`,
        `文件: ${result.saved_ply}`
      ].join('\n');
      setHud('导出完成');
    } catch (err) {
      ui.resultBox.textContent = `导出失败: ${err}`;
      setHud('导出失败');
    }
  });

  ui.canvas.addEventListener('pointerdown', onCanvasPointerDown);
  ui.canvas.addEventListener('pointermove', onCanvasPointerMove);
  ui.canvas.addEventListener('dblclick', onCanvasDoubleClick);
  ui.canvas.addEventListener('contextmenu', onCanvasContextMenu);

  window.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase();
    if (key === 'v') {
      setDrawMode(false);
    } else if (key === 'b') {
      setDrawMode(true);
    } else if (key === 'enter') {
      finalizePolygonSelection();
    } else if (key === 'backspace') {
      if (drawMode) {
        event.preventDefault();
        undoPolygonPoint();
      }
    } else if (key === 'escape') {
      cancelPolygonDraft();
    }
  });
}

function setDrawMode(enabled) {
  drawMode = enabled;
  controls.enabled = !enabled;

  ui.modeDraw.classList.toggle('btn-active', enabled);
  ui.modeView.classList.toggle('btn-active', !enabled);

  if (enabled) {
    setHud('多边形模式: 左键逐点点击，双击/回车/右键完成');
  } else {
    cancelPolygonDraft(false);
    setHud('视角模式: 左键旋转，右键平移，滚轮缩放');
  }
}

function onCanvasPointerDown(event) {
  if (!drawMode || event.button !== 0) {
    return;
  }

  const pt = getCanvasPoint(event);
  if (!pt.inBounds) {
    return;
  }

  const point = { x: pt.x, y: pt.y };
  if (!polygonDrawing) {
    polygonDrawing = true;
    polygonPoints = [point];
    hoverPoint = point;
    setHud('已添加第 1 个顶点');
    renderOverlay();
    return;
  }

  if (
    polygonPoints.length >= MIN_POLYGON_VERTICES &&
    distance2D(point, polygonPoints[0]) <= CLOSE_DISTANCE_PX
  ) {
    finalizePolygonSelection();
    return;
  }

  polygonPoints.push(point);
  setHud(`已添加第 ${polygonPoints.length} 个顶点`);
  renderOverlay();
}

function onCanvasPointerMove(event) {
  if (!drawMode) {
    return;
  }

  const pt = getCanvasPoint(event);
  if (!pt.inBounds) {
    hoverPoint = null;
    renderOverlay();
    return;
  }

  hoverPoint = { x: pt.x, y: pt.y };
  if (polygonDrawing) {
    renderOverlay();
  }
}

function onCanvasDoubleClick(event) {
  if (!drawMode) {
    return;
  }
  event.preventDefault();
  finalizePolygonSelection();
}

function onCanvasContextMenu(event) {
  if (!drawMode) {
    return;
  }
  event.preventDefault();
  finalizePolygonSelection();
}

function undoPolygonPoint() {
  if (!polygonDrawing || polygonPoints.length === 0) {
    return;
  }

  polygonPoints.pop();
  if (polygonPoints.length === 0) {
    polygonDrawing = false;
    setHud('多边形已清空');
  } else {
    setHud(`已撤销，剩余 ${polygonPoints.length} 个顶点`);
  }
  renderOverlay();
}

function cancelPolygonDraft(showHint = true) {
  polygonDrawing = false;
  polygonPoints = [];
  hoverPoint = null;
  renderOverlay();
  if (showHint) {
    setHud('已取消当前多边形');
  }
}

function clearSelection() {
  selectedIndices = [];
  updateSelectedCloud();
  cancelPolygonDraft(false);
  ui.resultBox.textContent = '已清除选择，请切到多边形模式重新选取。';
  setHud('已清除选择');
}

async function finalizePolygonSelection() {
  if (!drawMode || !polygonDrawing) {
    return;
  }

  if (polygonPoints.length < MIN_POLYGON_VERTICES) {
    setHud('至少需要 3 个顶点才能闭合多边形');
    return;
  }

  const polygon = polygonPoints.slice();
  setHud('正在筛选多边形内点...');
  await new Promise((resolve) => {
    window.setTimeout(resolve, 0);
  });

  selectedIndices = pickPointsInPolygon(polygon);
  updateSelectedCloud();

  polygonDrawing = false;
  polygonPoints = [];
  hoverPoint = null;
  renderOverlay();

  ui.resultBox.textContent = [
    `当前选择点数: ${selectedIndices.length.toLocaleString()}`,
    '你可以切换视角后再次绘制多边形，或直接计算体积。'
  ].join('\n');
  setHud(`多边形选择完成: ${selectedIndices.length.toLocaleString()} 点`);
}

function getCanvasPoint(event) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x,
    y,
    inBounds: x >= 0 && x <= rect.width && y >= 0 && y <= rect.height
  };
}

function renderOverlay() {
  if (!overlayCtx) {
    return;
  }

  const w = ui.overlayCanvas.width;
  const h = ui.overlayCanvas.height;
  overlayCtx.clearRect(0, 0, w, h);

  if (!drawMode || polygonPoints.length === 0) {
    return;
  }

  if (polygonPoints.length >= 3) {
    overlayCtx.fillStyle = 'rgba(10, 102, 214, 0.16)';
    overlayCtx.beginPath();
    overlayCtx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
    for (let i = 1; i < polygonPoints.length; i += 1) {
      overlayCtx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
    }
    overlayCtx.closePath();
    overlayCtx.fill();
  }

  overlayCtx.strokeStyle = '#0a66d6';
  overlayCtx.lineWidth = 2;
  overlayCtx.beginPath();
  overlayCtx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
  for (let i = 1; i < polygonPoints.length; i += 1) {
    overlayCtx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
  }
  if (hoverPoint) {
    overlayCtx.lineTo(hoverPoint.x, hoverPoint.y);
  }
  overlayCtx.stroke();

  for (let i = 0; i < polygonPoints.length; i += 1) {
    const p = polygonPoints[i];
    overlayCtx.beginPath();
    overlayCtx.fillStyle = i === 0 ? '#ff8f00' : '#0a66d6';
    overlayCtx.arc(p.x, p.y, i === 0 ? 4.5 : 3.5, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function pickPointsInPolygon(polygon) {
  if (polygon.length < MIN_POLYGON_VERTICES) {
    return [];
  }

  const width = renderer.domElement.clientWidth;
  const height = renderer.domElement.clientHeight;
  const count = pointPositions.length / 3;

  const v = new THREE.Vector3();
  const picked = [];
  for (let i = 0; i < count; i += 1) {
    const base = i * 3;
    v.set(pointPositions[base], pointPositions[base + 1], pointPositions[base + 2]);
    v.project(camera);
    if (v.z < -1 || v.z > 1) {
      continue;
    }

    const sx = (v.x * 0.5 + 0.5) * width;
    const sy = (-v.y * 0.5 + 0.5) * height;
    if (pointInPolygon2D(sx, sy, polygon)) {
      picked.push(i);
    }
  }
  return picked;
}

function pointInPolygon2D(x, y, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    const dy = yj - yi;
    const safeDy = Math.abs(dy) < 1e-9 ? 1e-9 : dy;
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / safeDy + xi;
    if (intersect) {
      inside = !inside;
    }
  }
  return inside;
}

function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function updateSelectedCloud() {
  const oldGeometry = selectedCloud.geometry;

  if (selectedIndices.length === 0) {
    selectedCloud.geometry = new THREE.BufferGeometry();
    oldGeometry.dispose();
    return;
  }

  const arr = new Float32Array(selectedIndices.length * 3);
  for (let i = 0; i < selectedIndices.length; i += 1) {
    const src = selectedIndices[i] * 3;
    const dst = i * 3;
    arr[dst] = pointPositions[src];
    arr[dst + 1] = pointPositions[src + 1];
    arr[dst + 2] = pointPositions[src + 2];
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  selectedCloud.geometry = geo;
  oldGeometry.dispose();
}

function renderResult(result) {
  const v = result.volume;
  const requestedGrid = v.requested_grid_cell_m ?? Number(ui.gridCell.value);
  const effectiveGrid = v.grid_cell_m ?? requestedGrid;
  const adjusted = Boolean(v.grid_cell_adjusted);

  if (adjusted && Number.isFinite(effectiveGrid)) {
    ui.gridCell.value = Number(effectiveGrid).toFixed(3).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
  }

  const lines = [
    `方法: ${result.method}`,
    `选择模式: ${result.selection_type || 'manual'}`,
    `选中点数: ${result.selected_point_count.toLocaleString()}`,
    `请求网格分辨率: ${formatNum(requestedGrid)} m`,
    `实际积分网格: ${formatNum(effectiveGrid)} m`,
    `体积: ${formatNum(v.volume_m3)} m3`,
    `基准高程: ${formatNum(v.base_z)} m`,
    `基准百分位: ${formatNum(v.base_percentile)} %`,
    `投影面积: ${formatNum(v.footprint_area_m2)} m2`,
    `平均高度: ${formatNum(v.mean_height_m)} m`,
    `最大高度: ${formatNum(v.max_height_m)} m`,
    `网格数: ${v.cell_count}`
  ];

  if (Number.isFinite(v.recommended_min_grid_cell_m)) {
    lines.push(`建议最小网格: ${formatNum(v.recommended_min_grid_cell_m)} m`);
  }
  if (adjusted) {
    lines.push(`网格修正: 是`);
    if (v.grid_adjust_reason) {
      lines.push(`修正原因: ${v.grid_adjust_reason}`);
    }
  }
  lines.push(`报告文件: ${result.saved_report}`);

  ui.resultBox.textContent = lines.join('\n');
}

function formatNum(v) {
  return Number(v).toFixed(4);
}

function parseNumericInput(rawValue, fallbackValue) {
  const txt = String(rawValue ?? '').trim().replace(',', '.');
  if (txt.length === 0) {
    return fallbackValue;
  }
  const v = Number(txt);
  return Number.isFinite(v) ? v : fallbackValue;
}

function normalizeParameterInputs() {
  const gridRaw = String(ui.gridCell.value ?? '').trim();
  if (gridRaw.length > 0) {
    const grid = Math.min(
      GRID_CELL_MAX,
      Math.max(GRID_CELL_MIN, parseNumericInput(ui.gridCell.value, cloudMeta?.default_grid_cell ?? 0.2))
    );
    ui.gridCell.value = grid.toFixed(3).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
  }

  const pctRaw = String(ui.basePercentile.value ?? '').trim();
  if (pctRaw.length > 0) {
    const pct = Math.min(100.0, Math.max(0.0, parseNumericInput(ui.basePercentile.value, 0.0)));
    ui.basePercentile.value = pct.toFixed(2).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
  }
}

function getComputationParams() {
  const gridCellRaw = parseNumericInput(ui.gridCell.value, cloudMeta?.default_grid_cell ?? 0.2);
  const gridCell = Math.max(gridCellRaw, GRID_CELL_EFFECTIVE_MIN);
  const basePercentile = parseNumericInput(ui.basePercentile.value, 0.0);

  if (!Number.isFinite(gridCell) || gridCell < GRID_CELL_MIN || gridCell > GRID_CELL_MAX) {
    throw new Error(`网格分辨率需在 ${GRID_CELL_MIN} 到 ${GRID_CELL_MAX} 米之间`);
  }
  if (!Number.isFinite(basePercentile) || basePercentile < 0 || basePercentile > 100) {
    throw new Error('基准百分位需在 0 到 100 之间');
  }

  normalizeParameterInputs();
  return { gridCell, basePercentile };
}

function setHud(msg) {
  ui.hud.textContent = msg;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = FETCH_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new Error(`请求超时 (${Math.round(timeoutMs / 1000)}s): ${url}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function fetchJson(url) {
  const res = await fetchWithTimeout(url);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

async function fetchArrayBuffer(url) {
  const res = await fetchWithTimeout(url);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.arrayBuffer();
}

async function postJson(url, payload) {
  const res = await fetchWithTimeout(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
