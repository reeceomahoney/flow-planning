from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FILES = {
    "top_raw.jpg": "image/jpeg",
    "top_overlay.jpg": "image/jpeg",
    "top_cloud.jpg": "image/jpeg",
    "top_mask.png": "image/png",
    "left_raw.jpg": "image/jpeg",
    "left_overlay.jpg": "image/jpeg",
    "left_cloud.jpg": "image/jpeg",
    "scene_cloud.jpg": "image/jpeg",
    "left_mask.png": "image/png",
}
REFERENCE_FILES = {"top_reference.png", "left_reference.png"}
Trajectory3D = tuple[tuple[float, float, float], ...]
Trajectory2D = tuple[tuple[float, float], ...]


class LiveState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.images: dict[str, tuple[int, bytes]] = {}
        self.versions: dict[str, int] = {}
        self.samples: dict[str, tuple[int, dict[str, object]]] = {}
        self.sample_versions: dict[str, int] = {}
        self.gripper_markers: dict[str, tuple[float, float, float]] = {}
        self.camera_boxes: dict[str, tuple[tuple[float, float], ...]] = {}
        self.robot_state_data: tuple[float, ...] | None = None
        self.policy_trajectories_data: tuple[Trajectory3D, ...] = ()
        self.policy_pickup_steps_data: tuple[int, ...] = ()
        self.camera_trajectories_data: dict[str, tuple[Trajectory2D, ...]] = {}
        self.policy_status_data: dict[str, object] = {"state": "waiting"}
        self.policy_trajectories_visible = True
        self.status = b'{"running":false,"updated_at":0,"cameras":{}}'
        self.realign_requested = False
        self.trim_percent = 40.0

    def publish_image(self, name: str, payload: bytes) -> None:
        with self.condition:
            version = self.versions.get(name, 0) + 1
            self.versions[name] = version
            self.images[name] = (version, payload)
            self.condition.notify_all()

    def image(self, name: str) -> tuple[int, bytes] | None:
        with self.condition:
            return self.images.get(name)

    def wait_image(
        self, name: str, version: int, timeout: float
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.versions.get(name, 0) <= version:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return self.images.get(name)

    def publish_status(self, status: Mapping[str, object]) -> None:
        encoded = json.dumps(status).encode()
        with self.condition:
            self.status = encoded

    def publish_sample(self, name: str, sample: dict[str, object]) -> None:
        with self.condition:
            version = self.sample_versions.get(name, 0) + 1
            self.sample_versions[name] = version
            self.samples[name] = (version, sample)
            self.condition.notify_all()

    def sample(self, name: str) -> tuple[int, dict[str, object]] | None:
        with self.condition:
            return self.samples.get(name)

    def publish_gripper_marker(
        self,
        name: str,
        marker: tuple[float, float, float] | None,
    ) -> None:
        with self.condition:
            if marker is None:
                self.gripper_markers.pop(name, None)
            else:
                self.gripper_markers[name] = marker

    def gripper_marker(self, name: str) -> tuple[float, float, float] | None:
        with self.condition:
            return self.gripper_markers.get(name)

    def publish_camera_box(
        self,
        name: str,
        box: tuple[tuple[float, float], ...] | None,
    ) -> None:
        with self.condition:
            if box is None:
                self.camera_boxes.pop(name, None)
            else:
                self.camera_boxes[name] = box

    def camera_box(
        self,
        name: str,
    ) -> tuple[tuple[float, float], ...] | None:
        with self.condition:
            return self.camera_boxes.get(name)

    def publish_robot_state(self, state: tuple[float, ...]) -> None:
        with self.condition:
            self.robot_state_data = state
            self.condition.notify_all()

    def wait_robot_state(
        self,
        timeout: float,
    ) -> tuple[float, ...] | None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.robot_state_data is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            return self.robot_state_data

    def publish_policy_trajectories(
        self,
        trajectories: tuple[Trajectory3D, ...],
        pickup_steps: tuple[int, ...] = (),
    ) -> None:
        with self.condition:
            self.policy_trajectories_data = trajectories
            self.policy_pickup_steps_data = pickup_steps

    def policy_trajectories(self) -> tuple[Trajectory3D, ...]:
        with self.condition:
            if not self.policy_trajectories_visible:
                return ()
            return self.policy_trajectories_data

    def set_policy_trajectories_visible(self, visible: bool) -> None:
        with self.condition:
            self.policy_trajectories_visible = visible

    def policy_trajectory_visibility(self) -> bool:
        with self.condition:
            return self.policy_trajectories_visible

    def policy_pickup_steps(self) -> tuple[int, ...]:
        with self.condition:
            if not self.policy_trajectories_visible:
                return ()
            return self.policy_pickup_steps_data

    def publish_camera_trajectories(
        self,
        name: str,
        trajectories: tuple[Trajectory2D, ...],
    ) -> None:
        with self.condition:
            self.camera_trajectories_data[name] = trajectories

    def camera_trajectories(self, name: str) -> tuple[Trajectory2D, ...]:
        with self.condition:
            return self.camera_trajectories_data.get(name, ())

    def publish_policy_status(self, status: Mapping[str, object]) -> None:
        with self.condition:
            self.policy_status_data = dict(status)

    def policy_status(self) -> dict[str, object]:
        with self.condition:
            return dict(self.policy_status_data)

    def status_bytes(self) -> bytes:
        with self.condition:
            return self.status

    def request_realign(self) -> None:
        with self.condition:
            self.realign_requested = True

    def consume_realign_request(self) -> bool:
        with self.condition:
            requested = self.realign_requested
            self.realign_requested = False
            return requested

    def set_outlier_trim_percent(self, percentage: float) -> None:
        with self.condition:
            self.trim_percent = percentage

    def outlier_trim_percent(self) -> float:
        with self.condition:
            return self.trim_percent


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Obstacle perception</title>
<style>
:root {
  color-scheme: dark;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  background: #080a0d;
  color: #edf1f7;
}
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; overflow: hidden; }
body {
  margin: 0;
  padding: 14px;
  display: flex;
  flex-direction: column;
}
header {
  flex: 0 0 47px;
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 20px;
}
h1 { margin: 0; font-size: 20px; letter-spacing: -0.02em; }
.subtitle { margin: 2px 0 0; color: #778294; font-size: 12px; }
.overall {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #a5afbd;
  font-size: 13px;
}
.header-actions { display: flex; align-items: center; gap: 12px; }
.header-filter {
  height: 30px;
  display: grid;
  grid-template-columns: auto 130px 32px;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid #293342;
  border-radius: 7px;
  background: #10151d;
  color: #9ba7b8;
  font-size: 10px;
  white-space: nowrap;
}
.header-filter input {
  width: 130px;
  margin: 0;
  accent-color: #51d88a;
  cursor: ew-resize;
}
.header-filter output {
  color: #51d88a;
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.recalibrate, .trajectory-toggle {
  height: 30px;
  padding: 0 12px;
  border: 1px solid #334154;
  border-radius: 7px;
  background: #151b24;
  color: #cbd3df;
  font: inherit;
  font-size: 11px;
  font-weight: 650;
  cursor: pointer;
}
.recalibrate:hover, .trajectory-toggle:hover {
  border-color: #51d88a;
  color: #edf8f1;
}
.recalibrate:disabled, .trajectory-toggle:disabled {
  cursor: wait;
  opacity: 0.55;
}
.trajectory-toggle.active {
  border-color: #51d88a;
  background: #10231b;
  color: #77eaa7;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #687385; }
.dot.live { background: #51d88a; box-shadow: 0 0 12px #51d88a88; }
.workspace {
  flex: 1 1 auto;
  min-height: 0;
}
.stage {
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(3, minmax(0, 1fr));
  gap: 11px;
}
.tile, .sidebar, .reference-card {
  border: 1px solid #252b35;
  background: #10141a;
  box-shadow: 0 14px 35px #0003;
}
.tile {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 12px;
}
.tile-head {
  flex: 0 0 34px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 11px;
  border-bottom: 1px solid #242a33;
}
.tile-title { font-size: 13px; font-weight: 650; }
.tile-tag {
  color: #51d88a;
  font-size: 10px;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.overlay {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
  background: #050608;
}
.cloud-wrap {
  flex: 1 1 auto;
  min-height: 0;
  position: relative;
  display: grid;
  place-items: center;
  overflow: hidden;
  background:
    radial-gradient(circle at 50% 45%, #182131 0, #0b0e13 65%);
}
.cloud-stream {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.overview-body {
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  grid-template-rows: repeat(2, minmax(0, 1fr));
  gap: 5px;
  padding: 5px;
}
.diagnostic {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid #222a35;
  border-radius: 6px;
  background: #06080b;
}
.diagnostic-label {
  flex: 0 0 17px;
  display: flex;
  align-items: center;
  padding: 0 6px;
  color: #748196;
  font-size: 8px;
  font-weight: 750;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
.diagnostic-image {
  display: block;
  width: 100%;
  height: calc(100% - 17px);
  min-height: 0;
  object-fit: contain;
}
.cloud-canvas {
  height: 86%;
  max-width: 86%;
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  text-align: center;
  color: #657083;
  border: 1px solid #293241;
  border-radius: 10px;
  background-color: #090c11;
  background-image:
    linear-gradient(#17202c 1px, transparent 1px),
    linear-gradient(90deg, #17202c 1px, transparent 1px);
  background-size: 28px 28px;
  box-shadow: inset 0 0 50px #0008;
}
.cloud-icon {
  width: 42px;
  height: 42px;
  margin: 0 auto 10px;
  border: 1px solid #3a4658;
  border-radius: 50%;
  position: relative;
}
.cloud-icon::before, .cloud-icon::after {
  content: '';
  position: absolute;
  background: #3a4658;
}
.cloud-icon::before { width: 1px; height: 56px; left: 20px; top: -8px; }
.cloud-icon::after { height: 1px; width: 56px; top: 20px; left: -8px; }
.cloud-name { color: #8995a7; font-size: 12px; font-weight: 650; }
.cloud-note { margin-top: 3px; font-size: 10px; }
.sidebar {
  min-height: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) minmax(0, 1fr) auto auto;
  gap: 8px;
  padding: 10px;
  border-radius: 12px;
}
.sidebar-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.sidebar-title { font-size: 12px; font-weight: 700; }
.sidebar-note { color: #697588; font-size: 10px; }
.reference-card {
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 9px;
  box-shadow: none;
}
.reference-head {
  flex: 0 0 31px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0 9px;
  border-bottom: 1px solid #242a33;
}
.reference-name { font-size: 12px; font-weight: 650; }
.metrics { color: #788496; font-size: 10px; white-space: nowrap; }
.reference-body {
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(0, 0.75fr);
  gap: 6px;
  padding: 6px;
}
.raw-group, .mini-group {
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.mini-group { gap: 6px; }
.small-label {
  flex: 0 0 18px;
  color: #687488;
  font-size: 9px;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.thumb-box {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  border: 1px solid #222a35;
  border-radius: 6px;
  background-color: #06080b;
}
.checker {
  background-image:
    linear-gradient(45deg, #11161d 25%, transparent 25%),
    linear-gradient(-45deg, #11161d 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #11161d 75%),
    linear-gradient(-45deg, transparent 75%, #11161d 75%);
  background-size: 14px 14px;
  background-position: 0 0, 0 7px, 7px -7px, -7px 0;
}
.thumbnail {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.reference-status {
  flex: 0 0 22px;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 8px;
  color: #7d899a;
  font-size: 10px;
}
.reference-status.good { color: #55d98c; }
.reference-status.bad { color: #ef7a7a; }
.filter-control {
  padding: 8px 9px;
  border: 1px solid #252d39;
  border-radius: 8px;
  background: #0c1016;
}
.filter-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #aab4c3;
  font-size: 10px;
  font-weight: 650;
}
.filter-value { color: #51d88a; font-variant-numeric: tabular-nums; }
.filter-slider {
  width: 100%;
  height: 14px;
  margin: 4px 0 0;
  accent-color: #51d88a;
  cursor: ew-resize;
}
.filter-note { color: #626e80; font-size: 9px; }
.legend {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
  color: #687487;
  font-size: 9px;
}
.legend-item { display: flex; align-items: center; gap: 5px; }
.swatch { width: 8px; height: 8px; border-radius: 2px; background: #51d88a; }
.swatch.white { background: white; }
.swatch.checker {
  background: repeating-conic-gradient(#303846 0 25%, #11161d 0 50%) 50% / 6px 6px;
}
@media (max-height: 760px) {
  body { padding: 9px; }
  header { flex-basis: 38px; }
  .subtitle { display: none; }
  .workspace, .stage { gap: 7px; }
  .sidebar { gap: 5px; padding: 7px; }
  .tile-head { flex-basis: 28px; }
  .reference-head { flex-basis: 25px; }
  .reference-status { display: none; }
}
</style>
</head>
<body>
<header>
  <div><h1>Obstacle perception</h1>
    <p class="subtitle">Tracking workspace · dual RGB-D views</p></div>
  <div class="header-actions">
    <div class="header-filter"><label for="outlier-trim">Outlier trim</label>
      <input id="outlier-trim" type="range" min="0" max="50"
        step="1" value="40">
      <output id="outlier-value">40%</output></div>
    <button class="trajectory-toggle active" id="trajectory-toggle">
      Hide demo paths</button>
    <button class="recalibrate" id="recalibrate">Recalibrate</button>
    <div class="overall"><span class="dot" id="overall-dot"></span>
      <span id="overall-text">Waiting for tracker</span></div>
  </div>
</header>

<main class="workspace">
  <section class="stage">
    <article class="tile">
      <div class="tile-head"><span class="tile-title">Top tracking view</span>
        <span class="tile-tag" id="top-metrics">Waiting</span></div>
      <img class="overlay live-image" data-file="top_overlay.jpg">
    </article>
    <article class="tile">
      <div class="tile-head"><span class="tile-title">Left wrist tracking view</span>
        <span class="tile-tag" id="left-metrics">Waiting</span></div>
      <img class="overlay live-image" data-file="left_overlay.jpg">
    </article>
    <article class="tile">
      <div class="tile-head"><span class="tile-title">Merged cloud · top POV</span>
        <span class="tile-tag">Fixed camera projection</span></div>
      <div class="cloud-wrap"><img class="cloud-stream live-image"
        data-file="top_cloud.jpg"></div>
    </article>
    <article class="tile">
      <div class="tile-head"><span class="tile-title">Merged cloud · left POV</span>
        <span class="tile-tag">Fixed camera projection</span></div>
      <div class="cloud-wrap"><img class="cloud-stream live-image"
        data-file="left_cloud.jpg"></div>
    </article>
    <article class="tile">
      <div class="tile-head"><span class="tile-title">
        Shared 3D scene · right robot base</span>
        <span class="tile-tag" id="policy-metrics">Waiting for policy</span></div>
      <div class="cloud-wrap"><img class="cloud-stream live-image"
        data-file="scene_cloud.jpg"></div>
    </article>
    <article class="tile">
      <div class="tile-head"><span class="tile-title">Perception overview</span>
        <span class="tile-tag">Raw · target · mask</span></div>
      <div class="overview-body">
        <div class="diagnostic"><div class="diagnostic-label">Top raw</div>
          <img class="diagnostic-image live-image" data-file="top_raw.jpg"></div>
        <div class="diagnostic checker">
          <div class="diagnostic-label">Top target</div>
          <img class="diagnostic-image target-image"
            data-file="top_reference.png"></div>
        <div class="diagnostic"><div class="diagnostic-label">Top mask</div>
          <img class="diagnostic-image live-image" data-file="top_mask.png"></div>
        <div class="diagnostic"><div class="diagnostic-label">Left raw</div>
          <img class="diagnostic-image live-image" data-file="left_raw.jpg"></div>
        <div class="diagnostic checker">
          <div class="diagnostic-label">Left target</div>
          <img class="diagnostic-image target-image"
            data-file="left_reference.png"></div>
        <div class="diagnostic"><div class="diagnostic-label">Left mask</div>
          <img class="diagnostic-image live-image" data-file="left_mask.png"></div>
      </div>
    </article>
  </section>
</main>

<script>
function refreshImages() {
  const stamp = Date.now();
  document.querySelectorAll('.live-image').forEach(image => {
    image.src = `/data/${image.dataset.file}?t=${stamp}`;
  });
}

function refreshTargets() {
  const stamp = Date.now();
  document.querySelectorAll('.target-image').forEach(image => {
    image.src = `/targets/${image.dataset.file}?t=${stamp}`;
  });
}

async function recalibrate() {
  const button = document.getElementById('recalibrate');
  button.disabled = true;
  button.textContent = 'Realigning…';
  try {
    const response = await fetch('/actions/recalibrate', {method: 'POST'});
    if (!response.ok) throw new Error('Request failed');
    refreshTargets();
    button.textContent = 'Requested';
  } catch (error) {
    button.textContent = 'Failed';
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = 'Recalibrate';
    }, 1200);
  }
}

async function togglePolicyTrajectories() {
  const button = document.getElementById('trajectory-toggle');
  const visible = !button.classList.contains('active');
  button.disabled = true;
  try {
    const response = await fetch('/actions/policy-trajectories', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({visible}),
    });
    if (!response.ok) throw new Error('Request failed');
    button.classList.toggle('active', visible);
    button.textContent = visible ? 'Hide demo paths' : 'Show demo paths';
  } finally {
    button.disabled = false;
  }
}

let outlierRequestTimer;

function requestOutlierTrim() {
  const slider = document.getElementById('outlier-trim');
  const value = Number(slider.value);
  document.getElementById('outlier-value').textContent = `${value}%`;
  clearTimeout(outlierRequestTimer);
  outlierRequestTimer = setTimeout(async () => {
    try {
      await fetch('/actions/outlier-trim', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({percentage: value}),
      });
    } catch (error) {
      document.getElementById('outlier-value').textContent = 'failed';
    }
  }, 80);
}

function updateCamera(name, data, fresh) {
  const metrics = document.getElementById(`${name}-metrics`);
  if (!data) {
    metrics.textContent = 'Waiting';
    return;
  }
  const fps = data.fps ? `${data.fps.toFixed(1)} fps` : '— fps';
  const area = data.mask_area
    ? `${data.mask_area.toLocaleString()} px` : 'no mask';
  const match = data.match_score ? ` · match ${data.match_score.toFixed(2)}` : '';
  metrics.textContent = fresh ? `${fps} · ${area}${match}` : 'Stale';
}

async function refreshStatus() {
  try {
    const response = await fetch('/status.json', {cache: 'no-store'});
    const status = await response.json();
    const fresh = status.running
      && Date.now() / 1000 - status.updated_at < 3;
    document.getElementById('overall-dot').className
      = `dot ${fresh ? 'live' : ''}`;
    document.getElementById('overall-text').textContent = fresh
      ? status.phase : 'Tracker offline';
    updateCamera('top', status.cameras?.top, fresh);
    updateCamera('left', status.cameras?.left, fresh);
    const policy = status.policy;
    const policyMetrics = document.getElementById('policy-metrics');
    if (policy?.state === 'ready') {
      const episodes = policy.episodes?.join('/') ?? '—';
      const lengths = policy.lengths?.join('/') ?? '—';
      policyMetrics.textContent = `train episodes ${episodes} · ${lengths} steps`;
    } else if (policy?.state === 'error') {
      policyMetrics.textContent = 'Policy error';
    } else {
      policyMetrics.textContent = policy?.state ?? 'Waiting for policy';
    }
    const trajectoryToggle = document.getElementById('trajectory-toggle');
    const trajectoriesVisible = status.policy_trajectories_visible ?? true;
    trajectoryToggle.classList.toggle('active', trajectoriesVisible);
    trajectoryToggle.textContent = trajectoriesVisible
      ? 'Hide demo paths' : 'Show demo paths';
    const slider = document.getElementById('outlier-trim');
    if (document.activeElement !== slider) {
      const percentage = status.outlier_trim_percent ?? 0;
      slider.value = percentage;
      document.getElementById('outlier-value').textContent = `${percentage}%`;
    }
  } catch (error) {
    document.getElementById('overall-text').textContent = 'Tracker offline';
    document.getElementById('overall-dot').className = 'dot';
  }
}

refreshImages();
refreshTargets();
refreshStatus();
document.getElementById('recalibrate').addEventListener('click', recalibrate);
document.getElementById('trajectory-toggle').addEventListener(
  'click', togglePolicyTrajectories);
document.getElementById('outlier-trim').addEventListener('input', requestOutlierTrim);
setInterval(refreshImages, 200);
setInterval(refreshStatus, 800);
</script>
</body>
</html>
"""


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    live_state: LiveState
    target_dir: Path

    def handle_error(self, request, client_address) -> None:
        del request, client_address


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: DashboardServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/status.json":
            self.send_bytes(self.server.live_state.status_bytes(), "application/json")
            return
        if path.startswith("/stream/"):
            name = path.removeprefix("/stream/")
            if FILES.get(name) != "image/jpeg":
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.stream_jpeg(name)
            return
        if path.startswith("/targets/"):
            name = path.removeprefix("/targets/")
            source = self.server.target_dir / name
            if name not in REFERENCE_FILES or not source.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(source.read_bytes(), "image/png")
            return
        if path.startswith("/data/"):
            name = path.removeprefix("/data/")
            content_type = FILES.get(name)
            current = self.server.live_state.image(name)
            if content_type is None or current is None:
                self.send_error(HTTPStatus.NOT_FOUND)
            else:
                self.send_bytes(current[1], content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/actions/recalibrate":
            self.server.live_state.request_realign()
            self.send_bytes(b'{"accepted":true}', "application/json")
            return
        if path == "/actions/policy-trajectories":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                visible = payload["visible"]
                if not isinstance(visible, bool):
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid visibility")
                return
            self.server.live_state.set_policy_trajectories_visible(visible)
            self.send_bytes(
                json.dumps({"visible": visible}).encode(),
                "application/json",
            )
            return
        if path == "/actions/outlier-trim":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                percentage = float(payload["percentage"])
                if not 0.0 <= percentage <= 50.0:
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid trim percentage")
                return
            self.server.live_state.set_outlier_trim_percent(percentage)
            self.send_bytes(
                json.dumps({"percentage": percentage}).encode(),
                "application/json",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def stream_jpeg(self, name: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while True:
                current = self.server.live_state.wait_image(name, version, 1.0)
                if current is None:
                    continue
                version, payload = current
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode())
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        del format, args


def start_dashboard(
    live_state: LiveState,
    target_dir: Path,
    bind: str,
    port: int,
) -> DashboardServer:
    server = DashboardServer((bind, port), Handler)
    server.live_state = live_state
    server.target_dir = target_dir
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
