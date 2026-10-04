"""
Reports the agent's decisions to the dashboard server. Fire-and-forget from
the agent's point of view: send() never blocks, and protection on this
machine never depends on the server being reachable. Events queue up
(bounded) while the server is down and are delivered when it comes back.

What leaves this machine: host/user identity, model metadata (thresholds,
test-set performance, the user's typical range per timing feature), each
incident's level + score + top reasons, the user's response, and a status
heartbeat with the latest scores and timing statistics. Never keystrokes,
never key or key-combination frequencies, never the model itself.

Server API (see server/biometrics_server/app.py):
  POST /api/hello   {host, model}              once per (re)connect / new model
  POST /api/report  {host_id, status, events}  heartbeat + event batch;
                    the reply carries dashboard commands (lockout, refine)
  POST /api/bye     {host_id}                  on clean shutdown
"""
import collections
import logging
import threading

import requests

log = logging.getLogger(__name__)

HEARTBEAT_SECONDS = 5  # when idle; every scored window and event is sent immediately
MAX_BUFFERED_EVENTS = 1000
MAX_BATCH = 100
TIMEOUT_SECONDS = 5


class Reporter:
    def __init__(self, base_url: str, hello: dict, status_fn, command_fn):
        self.base_url = base_url.rstrip("/")
        self.hello = hello
        self.host_id = hello["host"]["host_id"]
        self.status_fn = status_fn
        self.command_fn = command_fn

        self._events = collections.deque(maxlen=MAX_BUFFERED_EVENTS)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._connected = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="agent-reporter", daemon=True)
        self._thread.start()

    def send(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)
        self._wake.set()

    def poke(self) -> None:
        """Send the status now (e.g. after a scored window) instead of waiting for the heartbeat."""
        self._wake.set()

    def update_hello(self, hello: dict) -> None:
        """New model: announce it again on the next flush."""
        self.hello = hello
        self._connected = False
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=TIMEOUT_SECONDS * 2)
        if self._connected:
            self._post("/api/bye", {"host_id": self.host_id})

    def _run(self) -> None:
        while True:
            self._flush()
            if self._stop.is_set():
                return
            self._wake.wait(HEARTBEAT_SECONDS)
            self._wake.clear()

    def _flush(self) -> None:
        if not self._connected:
            if self._post("/api/hello", self.hello) is None:
                return
            self._connected = True
            log.info("Connected to dashboard server %s as %s", self.base_url, self.host_id)

        with self._lock:
            batch = [self._events.popleft() for _ in range(min(MAX_BATCH, len(self._events)))]

        resp = self._post("/api/report", {"host_id": self.host_id, "status": self.status_fn(), "events": batch})
        if resp is None or resp.status_code == 409:  # 409: server restarted and forgot us
            self._connected = False
            with self._lock:
                self._events.extendleft(reversed(batch))
            return
        if resp.status_code == 200:
            for command in resp.json().get("commands", []):
                self.command_fn(command)
        if batch and len(self._events):
            self._wake.set()  # more queued -- keep draining

    def _post(self, path: str, body: dict):
        try:
            resp = requests.post(self.base_url + path, json=body, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            if self._connected:
                log.warning("Dashboard server unreachable (%s); buffering events", e)
            return None
        if resp.status_code >= 400 and resp.status_code != 409:
            log.error("Dashboard server rejected %s: %s %s", path, resp.status_code, resp.text[:200])
        return resp
