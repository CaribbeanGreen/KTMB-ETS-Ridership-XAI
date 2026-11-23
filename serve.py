"""
serve.py  -  open the dashboard with one-click data updates (no terminal commands needed)
FYP: An Explainable Prediction Model Using Random Forest for KTMB ETS Ridership Surges

Starts a small local web server (Python standard library only) that
  - serves dashboard/ at http://127.0.0.1:8800/ (or the next free port) and opens it in the browser
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
import os
import socket
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
PORT = 8800                 # first port tried; if another program uses it, the next free one is taken
PORT_TRIES = 50
PY = sys.executable

# Public mode (hosted server, e.g. the NAS behind Cloudflare Tunnel): set ETS_PUBLIC=1.
# Visitors can't start updates; they run on a schedule instead and the page shows their progress.
PUBLIC = os.environ.get("ETS_PUBLIC") == "1"
HOST = os.environ.get("ETS_HOST", "0.0.0.0" if PUBLIC else "127.0.0.1")
SCHEDULE = PUBLIC and os.environ.get("ETS_SCHEDULE", "1") == "1"
FORECAST_AT = os.environ.get("ETS_FORECAST_AT", "06:00")   # daily: fresh 16-day weather forecast
FULL_DAY = int(os.environ.get("ETS_FULL_DAY", "1"))         # monthly: new ridership + retrain forecast model
FULL_AT = os.environ.get("ETS_FULL_AT", "03:00")

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
            st = {"server": True, "today": date.today().isoformat(), "state": read_state(),
                  "job": j, "jobs": {k: v["label"] for k, v in JOBS.items()}, "public": PUBLIC}
            if SCHEDULE:
                st["schedule"] = {"forecast": FORECAST_AT, "full_day": FULL_DAY, "full": FULL_AT}
            return self.send_json(st)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/update"):
            return self.send_json({"error": "not found"}, 404)
        if PUBLIC:
            return self.send_json({"error": "Updates run automatically on this server."}, 403)
        name = self.path.split("mode=")[-1] if "mode=" in self.path else "forecast"
        if name not in JOBS:
            return self.send_json({"error": f"unknown update '{name}'"}, 400)
        err = start_job(name)
        if err:
            return self.send_json({"error": err}, 409)
        return self.send_json({"started": name})


def start_job(name, wait=False):
    """Start an update unless one is already running. Returns an error message, or None when started."""
    with lock:
        if job["running"]:
            return "an update is already running"
        job.update(running=True, name=name, step=0, steps=len(JOBS[name]["steps"]), script=None,
                   started=datetime.now().strftime("%Y-%m-%d %H:%M"), finished=None, ok=None, lines=[],
                   progress=0.0, eta=sum(TYPICAL.get(st[0], 120) for st in JOBS[name]["steps"]), phase="Starting")
    if wait:
        run_job(name)
    else:
        threading.Thread(target=run_job, args=(name,), daemon=True).start()
    return None


def due_job(now, state):
    """Which scheduled update is due now, if any. Missed runs (server was off) catch up on the next check."""
    hhmm, today = now.strftime("%H:%M"), now.date().isoformat()
    if state.get("full", "")[:7] != today[:7] and (now.day > FULL_DAY or (now.day == FULL_DAY and hhmm >= FULL_AT)):
        return "full"
    if state.get("forecast", "")[:10] != today and hhmm >= FORECAST_AT:
        return "forecast"
    return None


def scheduler():
    """Public mode: run the updates automatically. A failed update is retried after an hour, not in a loop."""
    failed = {}
    while True:
        name = due_job(datetime.now(), read_state())
        if name and time.time() - failed.get(name, 0) > 3600:
            start_job(name, wait=True)
            if job["ok"]:
                if name == "full":                       # the full update ends with 07_forecast.py
                    state = read_state()
                    state["forecast"] = state["full"]
                    STATE.write_text(json.dumps(state, indent=2))
            else:
                failed[name] = time.time()
        time.sleep(60)


class Server(ThreadingHTTPServer):
    """Refuses a port another program already uses. Python's default (address reuse) lets two programs
    share a port on Windows, so the browser could reach the other program instead of this dashboard."""
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self):
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def port_in_use(port):
    """True if any program on this computer already answers on the port, over IPv4 or IPv6.
    Windows sends "localhost" to IPv6 (::1) first, so a program listening only there (e.g. Streamlit)
    would not be seen by an IPv4-only check, and the browser would open that program instead."""
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            pass
    return False


def open_server():
    for port in range(PORT, PORT + PORT_TRIES):
        if port_in_use(port):
            continue
        try:
            return Server(("127.0.0.1", port), Handler), port
        except OSError:
            continue
    raise SystemExit(f"No free port between {PORT} and {PORT + PORT_TRIES - 1}. Close another local server and try again.")


def main():
    if not (DASH / "index.html").exists():
        raise SystemExit(f"dashboard/index.html not found next to {Path(__file__).name}")
    if PUBLIC:
        port = int(os.environ.get("ETS_PORT", "8080"))
        server = ThreadingHTTPServer((HOST, port), Handler)
        print(f"Public mode on {HOST}:{port}; update buttons off; "
              + (f"forecast daily {FORECAST_AT}, full update day {FULL_DAY} at {FULL_AT}" if SCHEDULE else "no schedule"),
              flush=True)
        if SCHEDULE:
            threading.Thread(target=scheduler, daemon=True).start()
    else:
        server, port = open_server()
        url = f"http://127.0.0.1:{port}/"   # not "localhost": it can resolve to IPv6, where another app may answer
        if port != PORT:
            print(f"Port {PORT} is used by another program, so port {port} is used instead.")
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
