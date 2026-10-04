"""
Always-on-top tkinter windows the human answers (human-in-the-loop):

  challenge  "Re-authenticate"; closing it or letting the 30 s countdown run
             out counts as ignoring it (the agent escalates to an alert)
  alert      informational; closing it without re-authenticating arms the host
  lockout    no close button, no timeout -- only a successful re-auth closes it

Nothing here blocks typing at the OS level. Re-authentication uses the OS's
own password prompt; the agent never sees or stores the password.

tkinter needs exactly one Tk root + mainloop per process, owned by one thread
for its whole lifetime. PopupManager owns that thread; its methods are safe
to call from any thread (requests go through a queue the Tk thread polls).
"""
import gc
import logging
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

log = logging.getLogger(__name__)

CHALLENGE_TIMEOUT_SECONDS = 30
POLL_INTERVAL_MS = 100
WRAP = 400

TITLES = {
    "challenge": ("Identity check", "Your typing doesn't look like the enrolled user's."),
    "alert": ("Security alert", "High-confidence mismatch with the enrolled user's typing."),
    "lockout": ("Locked", "An alert was dismissed without re-authenticating, and the typing is "
                          "still suspicious. Re-authenticate to continue."),
}

_SECONDS = {"hold_time", "seek_time", "interval", "time_diff", "start_time", "end_time",
            "mean_ngram_dwell_time", "std_ngram_dwell_time", "mean_ngram_flight_time", "std_ngram_flight_time"}
_SHARES = {"error", "error_rate", "accuracy", "is_burst"}


def format_value(feature: str, value) -> str:
    if value is None:
        return "?"
    if feature in _SECONDS:
        return f"{value * 1000:.0f} ms"
    if feature in _SHARES or (feature.startswith(("key_", "combination_")) and feature != "key_per_second"):
        return f"{value:.0%}"
    return f"{value:.2f}"


def reason_line(reason: dict) -> str:
    f = reason["feature"]
    line = f"• {reason['label']}: {format_value(f, reason['value'])}"
    if reason.get("typical") is not None:
        if reason.get("typical_low") is not None:
            usual = f"{format_value(f, reason['typical_low'])}–{format_value(f, reason['typical_high'])}"
        else:
            usual = format_value(f, reason["typical"])
        line += f"  (usually {usual})"
    return line


def _os_reauth_prompt() -> bool:
    """Runs the OS's own authentication prompt; True if it succeeded."""
    if sys.platform == "darwin":
        cmd = ["osascript", "-e", 'do shell script "true" with administrator privileges']
    elif sys.platform.startswith("linux"):
        cmd = ["pkexec", "true"]
    else:
        log.warning("OS re-auth prompt not implemented for %s", sys.platform)
        return False
    return subprocess.run(cmd, capture_output=True).returncode == 0


class IncidentWindow(tk.Toplevel):
    def __init__(self, master, kind: str, incident, on_reauth, on_dismiss, on_destroy):
        super().__init__(master)
        self.kind = kind
        self._on_reauth = on_reauth
        self._on_dismiss = on_dismiss
        self._on_destroy = on_destroy
        self._remaining = CHALLENGE_TIMEOUT_SECONDS
        self._tick_job = None

        title, headline = TITLES[kind]
        if incident.trigger == "supervisor_lockout":
            headline = "Your supervisor locked this computer from the dashboard. Re-authenticate to continue."
        self.title(title)
        self.attributes("-topmost", True)
        self.minsize(420, 0)
        self.protocol("WM_DELETE_WINDOW", self._closed if kind != "lockout" else (lambda: None))

        pad = {"padx": 12, "pady": 5}
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, pady=6)
        ttk.Label(frame, text=headline, font=("", 11, "bold"), wraplength=WRAP).pack(anchor="w", **pad)
        if incident.trigger != "supervisor_lockout":
            ttk.Label(frame, text=f"Confidence this is not the enrolled user: {incident.impostor_score:.0%}",
                      wraplength=WRAP).pack(anchor="w", **pad)

        if incident.reasons:
            ttk.Label(frame, text="What looked different:").pack(anchor="w", padx=12, pady=(6, 0))
            for reason in incident.reasons:
                ttk.Label(frame, text=reason_line(reason), wraplength=WRAP).pack(anchor="w", padx=20)

        self.status = ttk.Label(frame, text="", wraplength=WRAP)
        self.status.pack(anchor="w", **pad)
        ttk.Button(frame, text="Re-authenticate", command=self._reauth_clicked).pack(anchor="w", **pad)

        if kind == "challenge":
            self.countdown = ttk.Label(frame, text="")
            self.countdown.pack(anchor="w", **pad)
            self._tick()
        elif kind == "alert":
            ttk.Label(frame, text="Closing this without re-authenticating means the next suspicious "
                                  "typing locks this screen.", wraplength=WRAP).pack(anchor="w", **pad)

        self.lift()
        self.focus_force()

    def _tick(self):
        if self._remaining <= 0:
            self._finish(self._on_dismiss, "timed_out")
            return
        self.countdown.config(text=f"Escalates to an alert in {self._remaining}s")
        self._remaining -= 1
        self._tick_job = self.after(1000, self._tick)

    def _reauth_clicked(self):
        self.status.config(text="Waiting for OS authentication...")
        self.update_idletasks()
        if _os_reauth_prompt():
            self._finish(self._on_reauth)
        else:
            self.status.config(text="Authentication failed or cancelled.")

    def _closed(self):
        self._finish(self._on_dismiss, "closed")

    def _finish(self, callback, *args):
        if callback:
            callback(*args)
        self.close_silently()

    def close_silently(self):
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
            self._tick_job = None
        self._on_destroy()
        self.destroy()


class PopupManager:
    def __init__(self):
        self._thread = None
        self._root = None
        self._requests: queue.Queue = queue.Queue()
        self._windows: dict[str, IncidentWindow] = {}
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="agent-popups", daemon=True)
            self._thread.start()
            self._ready.wait(timeout=10)

    def show(self, kind: str, incident, on_reauth, on_dismiss) -> None:
        self._requests.put(("show", kind, incident, on_reauth, on_dismiss))

    def close(self, incident_id: str) -> None:
        """Close without firing any callback (the incident was superseded)."""
        self._requests.put(("close", incident_id))

    def close_all(self) -> None:
        self._requests.put(("close_all",))

    def stop(self) -> None:
        """Tear Tk down on its own thread; Tk objects destroyed from another
        thread at interpreter exit abort with 'Tcl_AsyncDelete'."""
        if self._thread is not None:
            self._requests.put(("quit",))
            self._thread.join(timeout=5)

    def _run(self):
        self._root = tk.Tk()
        self._root.withdraw()  # only the popups are visible
        self._poll()
        self._ready.set()
        self._root.mainloop()
        # The Tcl interpreter is freed when its last Python reference is
        # collected; make that happen here, on the thread that created it.
        self._windows.clear()
        self._root.destroy()
        self._root = None
        gc.collect()

    def _poll(self):
        try:
            while True:
                request = self._requests.get_nowait()
                if request[0] == "show":
                    _, kind, incident, on_reauth, on_dismiss = request
                    self._windows[incident.id] = IncidentWindow(
                        self._root, kind, incident, on_reauth, on_dismiss,
                        on_destroy=lambda i=incident.id: self._windows.pop(i, None))
                elif request[0] == "close":
                    window = self._windows.get(request[1])
                    if window:
                        window.close_silently()
                elif request[0] == "close_all":
                    for window in list(self._windows.values()):
                        window.close_silently()
                elif request[0] == "quit":
                    self._root.quit()
                    return
        except queue.Empty:
            pass
        self._root.after(POLL_INTERVAL_MS, self._poll)
