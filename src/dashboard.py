import os
import json
import asyncio
import threading
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.simulator import run as simulator_run

app = FastAPI()

# Track active websockets & dismissed alarms in memory
active_websockets = set()
dismissed_alarms = set()
main_loop = None

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_CHANNEL = 'icu_vitals'

def broadcast_record_from_thread(record):
    global main_loop
    if main_loop and active_websockets:
        asyncio.run_coroutine_threadsafe(broadcast_to_clients(record), main_loop)

async def broadcast_to_clients(record):
    if not active_websockets:
        return
    data = dict(record)
    if data['subject_id'] in dismissed_alarms:
        data['warning'] = 0
        data['dismissed'] = True
    else:
        data['dismissed'] = False

    disconnected = set()
    for ws in list(active_websockets):
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        active_websockets.discard(ws)

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    threading.Thread(
        target=simulator_run,
        kwargs={'broadcast_callback': broadcast_record_from_thread},
        daemon=True
    ).start()
    print("ICU Simulator Started")

# Allow Lovable/React frontend to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def live_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ICU Live Patient Risk Monitoring Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: #0b0f19; color: #f3f4f6; min-height: 100vh; padding: 20px; }
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid #1f2937; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; }
    .title-group h1 { font-size: 24px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }
    .title-group p { font-size: 14px; color: #9ca3af; margin-top: 4px; }
    .badge-live { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #059669; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
    .badge-live::before { content: ''; width: 8px; height: 8px; background: #10b981; border-radius: 50%; display: inline-block; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }
    .stats-bar { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .stat-card { background: #111827; border: 1px solid #1f2937; padding: 15px 20px; border-radius: 12px; }
    .stat-card .label { font-size: 13px; color: #9ca3af; }
    .stat-card .val { font-size: 24px; font-weight: 700; margin-top: 5px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 18px; }
    .patient-card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 18px; transition: transform 0.15s, border-color 0.15s; position: relative; }
    .patient-card:hover { transform: translateY(-2px); border-color: #374151; }
    .patient-card.warning-active { border-color: #ef4444; background: rgba(239, 68, 68, 0.05); box-shadow: 0 0 15px rgba(239, 68, 68, 0.15); }
    .patient-card.warning-active::before { content: '⚠️ HIGH RISK ALERT'; position: absolute; top: -10px; right: 15px; background: #ef4444; color: #fff; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .patient-id { font-size: 16px; font-weight: 600; color: #fff; }
    .state-tag { font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px; text-transform: uppercase; }
    .state-stable { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .state-deteriorating { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .state-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .state-recovering { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .vitals-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
    .vital-box { background: #1f2937; padding: 10px; border-radius: 8px; }
    .vital-box .v-name { font-size: 11px; color: #9ca3af; text-transform: uppercase; }
    .vital-box .v-val { font-size: 18px; font-weight: 700; margin-top: 3px; color: #fff; }
    .vital-box .v-unit { font-size: 11px; font-weight: 400; color: #6b7280; margin-left: 2px; }
    .risk-section { margin-top: 10px; }
    .risk-header { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 5px; font-weight: 600; }
    .risk-bar-bg { width: 100%; height: 8px; background: #374151; border-radius: 9999px; overflow: hidden; }
    .risk-bar-fill { height: 100%; width: 0%; border-radius: 9999px; transition: width 0.4s ease, background-color 0.4s ease; }
    .btn-dismiss { width: 100%; margin-top: 12px; padding: 8px; border-radius: 6px; background: #374151; border: none; color: #e5e7eb; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
    .btn-dismiss:hover { background: #4b5563; }
    .links-bar { display: flex; gap: 12px; align-items: center; }
    .btn-link { font-size: 13px; color: #60a5fa; text-decoration: none; padding: 6px 12px; border: 1px solid #1e3a8a; border-radius: 6px; background: rgba(30, 58, 138, 0.2); }
    .btn-link:hover { background: rgba(30, 58, 138, 0.4); }
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>🏥 ICU Patient Risk Telemetry Dashboard</h1>
      <p>Real-time ML deterioration forecasting on streaming vital signs</p>
    </div>
    <div class="links-bar">
      <div id="statusPill" class="badge-live">Connecting to stream...</div>
      <a href="/docs" target="_blank" class="btn-link">Swagger Docs</a>
      <a href="http://localhost:8080" target="_blank" class="btn-link">React UI (Port 8080)</a>
    </div>
  </header>

  <div class="stats-bar">
    <div class="stat-card">
      <div class="label">Monitored Patients</div>
      <div class="val" id="totalPatients">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Active Deterioration Alerts</div>
      <div class="val" id="activeAlerts" style="color: #ef4444;">0</div>
    </div>
    <div class="stat-card">
      <div class="label">Average Risk Score</div>
      <div class="val" id="avgRisk" style="color: #60a5fa;">0.00</div>
    </div>
    <div class="stat-card">
      <div class="label">Telemetry Events Processed</div>
      <div class="val" id="eventCount" style="color: #10b981;">0</div>
    </div>
  </div>

  <div id="patientGrid" class="grid"></div>

  <script>
    const patientData = {};
    let totalEvents = 0;
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws`;

    function connect() {
      const ws = new WebSocket(wsUrl);
      const pill = document.getElementById('statusPill');

      ws.onopen = () => {
        pill.innerText = '🟢 Stream Active (50 Beds)';
        pill.className = 'badge-live';
      };

      ws.onmessage = (event) => {
        const record = JSON.parse(event.data);
        totalEvents++;
        patientData[record.subject_id] = record;
        updateCard(record);
        updateSummary();
      };

      ws.onclose = () => {
        pill.innerText = '🔴 Reconnecting in 2s...';
        setTimeout(connect, 2000);
      };

      ws.onerror = () => { ws.close(); };
    }

    function updateCard(r) {
      const grid = document.getElementById('patientGrid');
      let card = document.getElementById('card-' + r.subject_id.replace(/[^a-zA-Z0-9]/g, ''));

      if (!card) {
        card = document.createElement('div');
        card.id = 'card-' + r.subject_id.replace(/[^a-zA-Z0-9]/g, '');
        grid.appendChild(card);
      }

      const isAlert = r.warning === 1 && !r.dismissed;
      card.className = 'patient-card' + (isAlert ? ' warning-active' : '');

      const riskPct = Math.round((r.risk_score || 0.05) * 100);
      let riskColor = '#10b981';
      if (riskPct >= 65) riskColor = '#ef4444';
      else if (riskPct >= 35) riskColor = '#f59e0b';

      card.innerHTML = `
        <div class="card-head">
          <div class="patient-id">${r.subject_id}</div>
          <div class="state-tag state-${r.state}">${r.state}</div>
        </div>
        <div class="vitals-grid">
          <div class="vital-box">
            <div class="v-name">Heart Rate</div>
            <div class="v-val">${r.heart_rate}<span class="v-unit">bpm</span></div>
          </div>
          <div class="vital-box">
            <div class="v-name">Oxygen SpO2</div>
            <div class="v-val">${r.spo2}<span class="v-unit">%</span></div>
          </div>
          <div class="vital-box">
            <div class="v-name">Blood Pressure</div>
            <div class="v-val">${r.systolic_bp}/${r.diastolic_bp}<span class="v-unit">mmHg</span></div>
          </div>
          <div class="vital-box">
            <div class="v-name">Mean Arterial</div>
            <div class="v-val">${r.mean_bp}<span class="v-unit">MAP</span></div>
          </div>
        </div>
        <div class="risk-section">
          <div class="risk-header">
            <span>Deterioration Risk Score</span>
            <span style="color: ${riskColor}; font-weight: 700;">${(r.risk_score || 0.05).toFixed(2)} (${riskPct}%)</span>
          </div>
          <div class="risk-bar-bg">
            <div class="risk-bar-fill" style="width: ${riskPct}%; background-color: ${riskColor};"></div>
          </div>
        </div>
        ${isAlert ? `<button class="btn-dismiss" onclick="dismissAlarm('${r.subject_id}')">Acknowledge & Dismiss Alarm</button>` : ''}
      `;
    }

    function updateSummary() {
      const records = Object.values(patientData);
      document.getElementById('totalPatients').innerText = records.length;
      document.getElementById('eventCount').innerText = totalEvents;

      const activeAlerts = records.filter(r => r.warning === 1 && !r.dismissed).length;
      document.getElementById('activeAlerts').innerText = activeAlerts;

      if (records.length > 0) {
        const avg = records.reduce((acc, r) => acc + (r.risk_score || 0.05), 0) / records.length;
        document.getElementById('avgRisk').innerText = avg.toFixed(2);
      }
    }

    async function dismissAlarm(subjId) {
      await fetch('/dismiss/' + encodeURIComponent(subjId), { method: 'POST' });
      if (patientData[subjId]) {
        patientData[subjId].dismissed = true;
        patientData[subjId].warning = 0;
        updateCard(patientData[subjId]);
        updateSummary();
      }
    }

    connect();
  </script>
</body>
</html>
"""

@app.post("/dismiss/{subject_id}")
async def dismiss_alarm(subject_id: str):
    dismissed_alarms.add(subject_id)
    return {"status": "dismissed", "subject_id": subject_id}

@app.delete("/dismiss/{subject_id}")
async def undismiss_alarm(subject_id: str):
    dismissed_alarms.discard(subject_id)
    return {"status": "active", "subject_id": subject_id}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_websockets.add(ws)
    print("Frontend connected via WebSocket")

    use_redis = False
    r = None
    try:
        r_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        await r_client.ping()
        r = r_client
        use_redis = True
    except Exception:
        use_redis = False

    if use_redis and r:
        pubsub = r.pubsub()
        await pubsub.subscribe(REDIS_CHANNEL)
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    if data['subject_id'] in dismissed_alarms:
                        data['warning'] = 0
                        data['dismissed'] = True
                    else:
                        data['dismissed'] = False
                    await ws.send_json(data)
        except WebSocketDisconnect:
            print("Frontend disconnected")
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            active_websockets.discard(ws)
            await pubsub.unsubscribe(REDIS_CHANNEL)
            await r.aclose()
    else:
        # In-memory streaming fallback
        try:
            while True:
                # Keep alive until disconnect
                _ = await ws.receive_text()
        except WebSocketDisconnect:
            print("Frontend disconnected")
        except Exception:
            pass
        finally:
            active_websockets.discard(ws)