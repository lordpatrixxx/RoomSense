"""Start or stop this project's local server without an extra console window."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
STATE = WORK / "server.json"


def owned_process(state):
    try:
        proc = psutil.Process(state["pid"])
        args = proc.cmdline()
        if str(ROOT / "app.py") in args and "streamlit" in args:
            return proc
    except (psutil.Error, KeyError):
        pass
    return None


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    options = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    existing = owned_process(state)
    if options.stop:
        if existing:
            existing.terminate()
            try:
                existing.wait(timeout=10)
            except psutil.TimeoutExpired:
                existing.kill()
            print("RoomSense stopped.")
        else:
            print("No RoomSense server from this folder is running.")
        STATE.unlink(missing_ok=True)
        return
    if existing and healthy(state["port"]):
        port = state["port"]
    else:
        port = None
        for candidate in range(8501, 8511):
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        if port is None:
            raise RuntimeError("Ports 8501–8510 are in use. Close an unused local app and retry.")
        with (WORK / "server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                "--server.address", "127.0.0.1", "--server.port", str(port), "--server.headless", "true"],
                cwd=ROOT, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        STATE.write_text(json.dumps({"pid":process.pid, "port":port}), encoding="utf-8")
        for _ in range(80):
            if healthy(port):
                break
            if process.poll() is not None:
                raise RuntimeError("The server stopped during startup. See work/server.log.")
            time.sleep(.25)
        else:
            raise RuntimeError("The server is taking longer than expected. See work/server.log.")
    url = f"http://127.0.0.1:{port}"
    print(f"RoomSense is ready: {url}")
    if not options.no_browser:
        webbrowser.open(url)


if __name__ == "__main__":
    main()
