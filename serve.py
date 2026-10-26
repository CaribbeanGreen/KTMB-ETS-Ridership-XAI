"""
serve.py  -  open the dashboard with one-click data updates (no terminal commands needed)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Starts a small local web server (Python standard library only) that
  - serves dashboard/ at http://localhost:8765/ and opens it in the browser
  - lets the dashboard run the update scripts with a button, and shows their progress

Two kinds of update, both started from the dashboard:
  Update forecast   python 07_forecast.py --reuse
                    fresh 16-day weather forecast, forecasts moved on to today  (a few minutes)
  Full data update  python 01_collect.py --latest      new ETS ridership days + observed weather
                    python 02_build_dataset.py         rebuild the dataset
                    python 06_prepare_dashboard.py --reuse   past results incl. the new days
                    python 07_forecast.py              retrain the forecast model on all data
                    (about 30-60 minutes; the evaluated 2023-2025 model is not retrained)

Run:   double-click "Start Dashboard.bat"   or   python serve.py
Stop:  close the window (or Ctrl + C).
The server only listens on this computer (127.0.0.1); nothing is exposed to the network.
"""
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import date, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
DASH = BASE / "dashboard"
STATE = BASE / "data" / "processed" / "update_state.json"
LOG = BASE / "data" / "processed" / "update_log.txt"
PORT = 8765
PY = sys.executable

JOBS = {
    "forecast": {"label": "Update forecast",
                 "steps": [["07_forecast.py", "--reuse"]]},
    "full": {"label": "Full data update",
             "steps": [["01_collect.py", "--latest"], ["02_build_dataset.py"],
                       ["06_prepare_dashboard.py", "--reuse"], ["07_forecast.py"]]},
}

job = {"running": False, "name": None, "step": 0, "steps": 0, "script": None, "started": None,
       "finished": None, "ok": None, "lines": [], "progress": 0.0, "eta": None, "phase": ""}

# typical seconds per step on a normal PC, used for the time estimate until a step reports its own
TYPICAL = {"01_collect.py": 120, "02_build_dataset.py": 60, "06_prepare_dashboard.py": 240, "07_forecast.py": 600}
LABEL = {"01_collect.py": "Downloading new ridership and weather", "02_build_dataset.py": "Rebuilding the dataset",
         "06_prepare_dashboard.py": "Explaining the newest past days", "07_forecast.py": "Forecasting"}
TREE = re.compile(r"tree (\d+)/(\d+).*?about (\d+)s left")


def track(line, step_frac, step_eta):
    """Read a progress line from a script: returns the step's (fraction done, seconds left, phase label)."""
    m = TREE.search(line)
    if m:
        t, n, left = map(int, m.groups())
        return 0.15 + 0.8 * t / n, left, "Explaining each forecast with TreeSHAP"
    if line.startswith("1) Training"):
        return max(step_frac, 0.03), step_eta, "Training the model"
    if line.startswith("3) Weather"):
        return max(step_frac, 0.08), step_eta, "Getting the weather forecast"
    if "TreeSHAP" in line and ("4)" in line or "3)" in line):
        return max(step_frac, 0.15), step_eta, "Explaining each forecast with TreeSHAP"
    if "rows computed" in line:
        return max(step_frac, 0.95), 5, "Saving the results"
    return step_frac, step_eta, None
lock = threading.Lock()


def read_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def run_job(name):
    spec = JOBS[name]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ok = True
    with open(LOG, "w", encoding="utf-8") as log:
        for i, step in enumerate(spec["steps"], 1):
            later = sum(TYPICAL.get(st[0], 120) for st in spec["steps"][i:])
            step_frac, step_eta = 0.0, TYPICAL.get(step[0], 120)
            with lock:
                job.update(step=i, script=step[0], phase=LABEL.get(step[0], step[0]))
            line = f"== step {i}/{len(spec['steps'])}: python {' '.join(step)}"
            log.write(line + "\n"); log.flush()
            with lock:
                job["lines"].append(line)
            p = subprocess.Popen([PY, "-u", *step], cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for out in p.stdout:
                out = out.rstrip()
                log.write(out + "\n"); log.flush()
                step_frac, step_eta, phase = track(out, step_frac, step_eta)
                with lock:
                    job["lines"] = (job["lines"] + [out])[-200:]
                    job["progress"] = round(((i - 1) + min(step_frac, 0.99)) / len(spec["steps"]), 3)
                    job["eta"] = int(step_eta + later)
                    if phase:
                        job["phase"] = phase
            if p.wait() != 0:
                ok = False
                msg = f"!! {step[0]} stopped with an error (exit code {p.returncode}); later steps were skipped."
                log.write(msg + "\n")
                with lock:
                    job["lines"].append(msg)
                break
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if ok:
        state = read_state()
        state[name] = now
        state["last"] = now
        state["lastDate"] = date.today().isoformat()
        STATE.write_text(json.dumps(state, indent=2))
    with lock:
        job.update(running=False, finished=now, ok=ok, progress=1.0 if ok else job["progress"], eta=0)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DASH), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # always load the newest data.js / forecast.js
        super().end_headers()

    def log_message(self, fmt, *args):                   # keep the console quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            with lock:
                j = dict(job, lines=job["lines"][-12:])
            return self.send_json({"server": True, "today": date.today().isoformat(), "state": read_state(),
                                   "job": j, "jobs": {k: v["label"] for k, v in JOBS.items()}})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/update"):
            return self.send_json({"error": "not found"}, 404)
        name = self.path.split("mode=")[-1] if "mode=" in self.path else "forecast"
        if name not in JOBS:
            return self.send_json({"error": f"unknown update '{name}'"}, 400)
        with lock:
            if job["running"]:
                return self.send_json({"error": "an update is already running"}, 409)
            job.update(running=True, name=name, step=0, steps=len(JOBS[name]["steps"]), script=None,
                       started=datetime.now().strftime("%Y-%m-%d %H:%M"), finished=None, ok=None, lines=[],
                       progress=0.0, eta=sum(TYPICAL.get(st[0], 120) for st in JOBS[name]["steps"]), phase="Starting")
        threading.Thread(target=run_job, args=(name,), daemon=True).start()
        return self.send_json({"started": name})


def main():
    if not (DASH / "index.html").exists():
        raise SystemExit(f"dashboard/index.html not found next to {Path(__file__).name}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"Dashboard running at {url}  (keep this window open; close it to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
