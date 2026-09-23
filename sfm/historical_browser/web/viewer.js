"use strict";

const experimentSelect = document.querySelector("#experiment");
const subjectFieldset = document.querySelector("#subjects");
const loadButton = document.querySelector("#load");
const fitButton = document.querySelector("#fit");
const pointSizeInput = document.querySelector("#point-size");
const backgroundPresetInput = document.querySelector("#background-preset");
const backgroundCustomInput = document.querySelector("#background-custom");
const canvas = document.querySelector("#viewer");
const statusBox = document.querySelector("#status");
const titleBox = document.querySelector("#title");
const methodBox = document.querySelector("#method");
const cameraBox = document.querySelector("#camera");
const availabilityBox = document.querySelector("#availability");
const stageBadge = document.querySelector("#stage");
const pointsBadge = document.querySelector("#points");

let catalog = null;
let activeRequest = null;
let requestGeneration = 0;

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle("error", isError);
}

function selectedSubject() {
  return document.querySelector('input[name="subject"]:checked')?.value || null;
}

function currentExperiment() {
  return catalog?.experiments.find((item) => item.id === experimentSelect.value) || null;
}

function refreshDetails() {
  const experiment = currentExperiment();
  if (!experiment) return;
  titleBox.textContent = `${experiment.id} — ${experiment.title}`;
  methodBox.textContent = experiment.method;
  cameraBox.textContent = `${experiment.camera_policy}. Display: ${experiment.display_variant}.`;
  stageBadge.textContent = experiment.stage_type === "derived_cleanup" ? "Derived cleanup" :
    experiment.stage_type === "dense_reconstruction" ? "Dense MVS" :
    experiment.stage_type === "learned_reconstruction" ? "Learned SfM" : "Reconstruction";
  stageBadge.classList.toggle("cleanup", experiment.stage_type === "derived_cleanup");

  for (const radio of document.querySelectorAll('input[name="subject"]')) {
    const result = experiment.subjects[radio.value];
    radio.disabled = !result.available;
    radio.closest("label").title = result.reason || "";
  }
  let subject = selectedSubject();
  if (!subject || !experiment.subjects[subject].available) {
    const fallback = ["light", "dark"].find((name) => experiment.subjects[name].available);
    const radio = document.querySelector(`input[name="subject"][value="${fallback}"]`);
    if (radio) radio.checked = true;
    subject = fallback;
  }

  const result = experiment.subjects[subject];
  pointsBadge.textContent = result.point_count ? `${result.point_count.toLocaleString()} points` : "Not available";
  if (!result.available) {
    availabilityBox.textContent = result.reason;
  } else if (!result.on_disk) {
    availabilityBox.textContent = "The catalog entry exists, but its preserved PLY is missing from the data root.";
  } else if (experiment.id === "E10") {
    availabilityBox.textContent = "E10 is light-shirt only; no black-shirt E10 was run.";
  } else {
    availabilityBox.textContent = catalog.fresh_result ?
      "Recomputed result is ready to inspect." : "Preserved result is ready to inspect.";
  }
  loadButton.disabled = !(result.available && result.on_disk);
}

const gl = canvas.getContext("webgl", {antialias: true, alpha: false});
if (!gl) {
  setStatus("WebGL is unavailable in this browser.", true);
  throw new Error("WebGL unavailable");
}

const BACKGROUND_STORAGE_KEY = "cv802.saved-viewer.background.v1";
const BACKGROUND_PRESETS = Object.freeze({
  dark: "#090f1a",
  black: "#000000",
  white: "#ffffff",
  grey: "#808080",
});
let backgroundPreset = "dark";
let customBackgroundHex = "#315b78";
let backgroundHex = BACKGROUND_PRESETS.dark;

function validHexColour(value) {
  return typeof value === "string" && /^#[0-9a-fA-F]{6}$/.test(value);
}

function backgroundRgb(hex) {
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255);
}

function saveBackgroundPreference() {
  try {
    window.localStorage.setItem(BACKGROUND_STORAGE_KEY, JSON.stringify({
      preset: backgroundPreset,
      custom: customBackgroundHex,
    }));
  } catch (_error) {
    // The viewer still works when private-browser policy disables localStorage.
  }
}

function applyBackground(preset, customHex = customBackgroundHex, persist = true) {
  if (!Object.hasOwn(BACKGROUND_PRESETS, preset) && preset !== "custom") preset = "dark";
  if (validHexColour(customHex)) customBackgroundHex = customHex.toLowerCase();
  backgroundPreset = preset;
  backgroundHex = preset === "custom" ? customBackgroundHex : BACKGROUND_PRESETS[preset];
  backgroundPresetInput.value = preset;
  backgroundCustomInput.disabled = preset !== "custom";
  // The native picker always reflects the canvas even when a preset is active.
  backgroundCustomInput.value = backgroundHex;
  canvas.style.backgroundColor = backgroundHex;
  if (persist) saveBackgroundPreference();
  draw();
}

function restoreBackgroundPreference() {
  let saved = null;
  try {
    saved = JSON.parse(window.localStorage.getItem(BACKGROUND_STORAGE_KEY));
  } catch (_error) {
    saved = null;
  }
  const preset = saved && typeof saved === "object" ? saved.preset : "dark";
  const custom = saved && validHexColour(saved.custom) ? saved.custom : customBackgroundHex;
  applyBackground(preset, custom, false);
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compilation failed");
  }
  return shader;
}

const vertexShader = compileShader(gl.VERTEX_SHADER, `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uProjection;
  uniform float uYaw;
  uniform float uPitch;
  uniform float uDistance;
  uniform float uPointSize;
  uniform vec2 uPan;
  varying vec3 vColor;
  void main() {
    float cy = cos(uYaw), sy = sin(uYaw);
    float cp = cos(uPitch), sp = sin(uPitch);
    // The preserved COLMAP models use the opposite display-up convention.
    // Rotate 180 degrees around X for viewing only; PLY coordinates stay intact.
    vec3 source = vec3(aPosition.x, -aPosition.y, -aPosition.z);
    vec3 p = vec3(cy * source.x + sy * source.z,
                  source.y,
                  -sy * source.x + cy * source.z);
    p = vec3(p.x, cp * p.y - sp * p.z, sp * p.y + cp * p.z);
    p.xy += uPan;
    p.z -= uDistance;
    gl_Position = uProjection * vec4(p, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
  }
`);

const fragmentShader = compileShader(gl.FRAGMENT_SHADER, `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 delta = gl_PointCoord - vec2(0.5);
    if (dot(delta, delta) > 0.25) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`);

const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  throw new Error(gl.getProgramInfoLog(program) || "shader link failed");
}
gl.useProgram(program);

const locations = {
  position: gl.getAttribLocation(program, "aPosition"),
  color: gl.getAttribLocation(program, "aColor"),
  projection: gl.getUniformLocation(program, "uProjection"),
  yaw: gl.getUniformLocation(program, "uYaw"),
  pitch: gl.getUniformLocation(program, "uPitch"),
  distance: gl.getUniformLocation(program, "uDistance"),
  pointSize: gl.getUniformLocation(program, "uPointSize"),
  pan: gl.getUniformLocation(program, "uPan"),
};

const positionBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
let vertexCount = 0;
let yaw = 0.35;
let pitch = -0.12;
let distance = 3.1;
// Allow close inspection and wide overview without changing the saved cloud.
// The old [1.25, 12] camera-distance clamp prevented detailed shirt inspection.
const MIN_DISTANCE = 0.03;
const MAX_DISTANCE = 120;
let dragging = false;
let dragMode = "rotate";
let previousPointer = [0, 0];
let panX = 0;
let panY = 0;

function resetView() {
  yaw = 0.35;
  pitch = -0.12;
  distance = 3.1;
  panX = 0;
  panY = 0;
  draw();
}

function projectionMatrix(aspect) {
  const near = 0.002;
  const far = Math.max(100, distance + 10);
  const f = 1 / Math.tan(45 * Math.PI / 360);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
}

function draw() {
  resize();
  const [red, green, blue] = backgroundRgb(backgroundHex);
  gl.clearColor(red, green, blue, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!vertexCount) return;
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(program);
  gl.uniformMatrix4fv(locations.projection, false, projectionMatrix(canvas.width / canvas.height));
  gl.uniform1f(locations.yaw, yaw);
  gl.uniform1f(locations.pitch, pitch);
  gl.uniform1f(locations.distance, distance);
  gl.uniform1f(locations.pointSize, Number(pointSizeInput.value) * Math.min(window.devicePixelRatio || 1, 2));
  gl.uniform2f(locations.pan, panX, panY);

  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(locations.position);
  gl.vertexAttribPointer(locations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.enableVertexAttribArray(locations.color);
  gl.vertexAttribPointer(locations.color, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, vertexCount);
}

const TYPE_INFO = {
  char: [1, "getInt8"], int8: [1, "getInt8"],
  uchar: [1, "getUint8"], uint8: [1, "getUint8"],
  short: [2, "getInt16"], int16: [2, "getInt16"],
  ushort: [2, "getUint16"], uint16: [2, "getUint16"],
  int: [4, "getInt32"], int32: [4, "getInt32"],
  uint: [4, "getUint32"], uint32: [4, "getUint32"],
  float: [4, "getFloat32"], float32: [4, "getFloat32"],
  double: [8, "getFloat64"], float64: [8, "getFloat64"],
};

function locateHeader(bytes) {
  const marker = new TextEncoder().encode("end_header");
  outer: for (let start = 0; start <= Math.min(bytes.length - marker.length, 65536); start += 1) {
    for (let index = 0; index < marker.length; index += 1) {
      if (bytes[start + index] !== marker[index]) continue outer;
    }
    let body = start + marker.length;
    if (bytes[body] === 13) body += 1;
    if (bytes[body] === 10) body += 1;
    return body;
  }
  throw new Error("PLY end_header was not found");
}

function parseHeader(text) {
  const lines = text.split(/\r?\n/);
  if (lines[0] !== "ply") throw new Error("not a PLY file");
  const formatLine = lines.find((line) => line.startsWith("format "));
  if (!formatLine) throw new Error("PLY format is missing");
  const format = formatLine.split(/\s+/)[1];
  let inVertex = false;
  let count = null;
  const properties = [];
  for (const line of lines) {
    const fields = line.trim().split(/\s+/);
    if (fields[0] === "element") {
      inVertex = fields[1] === "vertex";
      if (inVertex) count = Number(fields[2]);
    } else if (inVertex && fields[0] === "property") {
      if (fields[1] === "list") throw new Error("list-valued vertex properties are unsupported");
      if (!TYPE_INFO[fields[1]]) throw new Error(`unsupported PLY type ${fields[1]}`);
      properties.push({type: fields[1], name: fields[2]});
    }
  }
  if (!Number.isSafeInteger(count) || count < 1) throw new Error("invalid PLY vertex count");
  for (const name of ["x", "y", "z", "red", "green", "blue"]) {
    if (!properties.some((item) => item.name === name)) throw new Error(`PLY is missing ${name}`);
  }
  return {format, count, properties};
}

function parsePly(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const bodyOffset = locateHeader(bytes);
  const header = parseHeader(new TextDecoder("ascii").decode(bytes.subarray(0, bodyOffset)));
  const positions = new Float32Array(header.count * 3);
  const colors = new Float32Array(header.count * 3);

  if (header.format === "binary_little_endian" || header.format === "binary_big_endian") {
    const little = header.format === "binary_little_endian";
    const stride = header.properties.reduce((total, property) => total + TYPE_INFO[property.type][0], 0);
    if (bodyOffset + stride * header.count > bytes.length) throw new Error("PLY vertex payload is truncated");
    const view = new DataView(arrayBuffer);
    for (let vertex = 0; vertex < header.count; vertex += 1) {
      let offset = bodyOffset + vertex * stride;
      const values = {};
      for (const property of header.properties) {
        const [size, getter] = TYPE_INFO[property.type];
        values[property.name] = size === 1 ? view[getter](offset) : view[getter](offset, little);
        offset += size;
      }
      const base = vertex * 3;
      positions[base] = values.x;
      positions[base + 1] = values.y;
      positions[base + 2] = values.z;
      colors[base] = values.red / 255;
      colors[base + 1] = values.green / 255;
      colors[base + 2] = values.blue / 255;
    }
  } else if (header.format === "ascii") {
    const rows = new TextDecoder("ascii").decode(bytes.subarray(bodyOffset)).trim().split(/\r?\n/);
    if (rows.length < header.count) throw new Error("PLY vertex payload is truncated");
    const indices = Object.fromEntries(header.properties.map((property, index) => [property.name, index]));
    for (let vertex = 0; vertex < header.count; vertex += 1) {
      const values = rows[vertex].trim().split(/\s+/).map(Number);
      const base = vertex * 3;
      positions[base] = values[indices.x];
      positions[base + 1] = values[indices.y];
      positions[base + 2] = values[indices.z];
      colors[base] = values[indices.red] / 255;
      colors[base + 1] = values[indices.green] / 255;
      colors[base + 2] = values[indices.blue] / 255;
    }
  } else {
    throw new Error(`unsupported PLY format ${header.format}`);
  }

  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = positions[index + axis];
      if (!Number.isFinite(value)) throw new Error("PLY contains a non-finite position");
      minimum[axis] = Math.min(minimum[axis], value);
      maximum[axis] = Math.max(maximum[axis], value);
    }
  }
  const center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
  const largestExtent = Math.max(...minimum.map((value, axis) => maximum[axis] - value));
  if (!(largestExtent > 0)) throw new Error("PLY has zero spatial extent");
  const scale = 2 / largestExtent;
  for (let index = 0; index < positions.length; index += 3) {
    positions[index] = (positions[index] - center[0]) * scale;
    positions[index + 1] = (positions[index + 1] - center[1]) * scale;
    positions[index + 2] = (positions[index + 2] - center[2]) * scale;
  }
  return {positions, colors, count: header.count};
}

async function loadCloud() {
  const experiment = currentExperiment();
  const subject = selectedSubject();
  if (!experiment || !subject) return;
  const result = experiment.subjects[subject];
  if (!result.available || !result.on_disk) return;
  if (activeRequest) activeRequest.abort();
  const generation = ++requestGeneration;
  const controller = new AbortController();
  activeRequest = controller;
  loadButton.disabled = true;
  setStatus(`Loading ${experiment.id} ${subject} (${result.point_count.toLocaleString()} points)…`);
  try {
    const response = await fetch(`/cloud/${experiment.id}/${subject}.ply`, {signal: controller.signal});
    if (!response.ok) {
      const error = await response.json().catch(() => ({error: response.statusText}));
      throw new Error(error.error || response.statusText);
    }
    const parsed = parsePly(await response.arrayBuffer());
    if (generation !== requestGeneration) return;
    if (parsed.count !== result.point_count) {
      throw new Error(`catalog expects ${result.point_count} vertices but PLY contains ${parsed.count}`);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, parsed.positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, parsed.colors, gl.STATIC_DRAW);
    vertexCount = parsed.count;
    resetView();
    fitButton.disabled = false;
    setStatus(`${experiment.id} · ${subject === "light" ? "light shirt" : "black shirt + crutches"} · ${parsed.count.toLocaleString()} coloured points`);
  } catch (error) {
    if (error.name !== "AbortError" && generation === requestGeneration) {
      setStatus(`Could not load cloud: ${error.message}`, true);
    }
  } finally {
    if (generation === requestGeneration) {
      activeRequest = null;
      refreshDetails();
    }
  }
}

function selectionChanged() {
  requestGeneration += 1;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  vertexCount = 0;
  fitButton.disabled = true;
  draw();
  refreshDetails();
  setStatus(catalog.fresh_result ? "Load the recomputed result to inspect it." :
    "Selection changed. Load its preserved result to inspect it.");
}

canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  dragMode = event.button === 2 || event.shiftKey ? "pan" : "rotate";
  previousPointer = [event.clientX, event.clientY];
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const dx = event.clientX - previousPointer[0];
  const dy = event.clientY - previousPointer[1];
  previousPointer = [event.clientX, event.clientY];
  if (dragMode === "pan") {
    const sensitivity = 0.0024 * distance;
    panX += dx * sensitivity;
    panY -= dy * sensitivity;
  } else {
    yaw += dx * 0.008;
    pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008));
  }
  draw();
});
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("pointercancel", () => { dragging = false; });
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const deltaPixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
  distance = Math.max(MIN_DISTANCE, Math.min(MAX_DISTANCE, distance * Math.exp(deltaPixels * 0.001)));
  draw();
}, {passive: false});
canvas.addEventListener("dblclick", resetView);
window.addEventListener("resize", draw);
pointSizeInput.addEventListener("input", draw);
backgroundPresetInput.addEventListener("change", () => {
  applyBackground(backgroundPresetInput.value, customBackgroundHex);
});
backgroundCustomInput.addEventListener("input", () => {
  if (backgroundPresetInput.value === "custom") {
    applyBackground("custom", backgroundCustomInput.value);
  }
});
fitButton.addEventListener("click", resetView);
loadButton.addEventListener("click", loadCloud);
experimentSelect.addEventListener("change", selectionChanged);
subjectFieldset.addEventListener("change", selectionChanged);

async function initialize() {
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error(response.statusText);
    catalog = await response.json();
    for (const experiment of catalog.experiments) {
      const option = document.createElement("option");
      const kind = experiment.stage_type === "derived_cleanup" ? "cleanup" :
        experiment.stage_type === "dense_reconstruction" ? "dense MVS" :
        experiment.stage_type === "learned_reconstruction" ? "learned SfM" : "reconstruction";
      option.value = experiment.id;
      option.textContent = `${experiment.id} · ${experiment.title} [${kind}]`;
      experimentSelect.append(option);
    }
    const requestedExperiment = new URLSearchParams(window.location.search).get("selected");
    if (requestedExperiment && catalog.experiments.some((item) => item.id === requestedExperiment)) {
      experimentSelect.value = requestedExperiment;
    }
    experimentSelect.disabled = false;
    subjectFieldset.disabled = false;
    document.querySelector('input[name="subject"][value="light"]').checked = true;
    refreshDetails();
    setStatus(catalog.fresh_result ? "The recomputed result is ready to load." :
      "Choose an experiment and subject, then load its preserved result.");
    if (new URLSearchParams(window.location.search).get("autoload") === "1") {
      await loadCloud();
    }
  } catch (error) {
    setStatus(`Could not read catalog: ${error.message}`, true);
  }
}

restoreBackgroundPreference();
initialize();
