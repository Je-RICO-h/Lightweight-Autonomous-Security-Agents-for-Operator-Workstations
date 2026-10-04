#!/usr/bin/env python
"""Interactive menu entry point for the biometrics CLI.

Run with no arguments, then pick an option by number.
"""
import os
import pwd
import signal
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

# pynput defaults to its X11 backend on Linux, which under a native Wayland
# session only sees XWayland-routed input (laggy, desynced modifiers, and
# native-Wayland apps like JetBrains IDEs invisible to it entirely). The
# uinput backend reads raw kernel input events instead and works correctly
# under Wayland, but needs root (or udev-granted access to /dev/uinput and
# /dev/input/event*) to open those devices.
#
# Scoped to PYNPUT_BACKEND_KEYBOARD, not the generic PYNPUT_BACKEND: pynput
# has no mouse._uinput backend at all, and this project never touches
# pynput.mouse, but pynput/__init__.py imports it unconditionally -- setting
# the generic var would make that import crash before keyboard is ever used.
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and "PYNPUT_BACKEND_KEYBOARD" not in os.environ:
    os.environ["PYNPUT_BACKEND_KEYBOARD"] = "uinput"

sys.path.insert(0, str(Path(__file__).parent / "src"))

from biometrics.paths import hand_to_user, owner
from biometrics.cli.main import (
    cmd_collect,
    cmd_describe_model,
    cmd_infer,
    cmd_process,
    cmd_refine,
    cmd_train,
)

# The dashboard server lives in its own folder and doesn't import the
# biometrics package; its runtime state (SQLite DB, PID, log) goes in server/data/.
SERVER_DIR = Path(__file__).parent / "server"
SERVER_DATA_DIR = SERVER_DIR / "data"
SERVER_PID_FILE = SERVER_DATA_DIR / "server.pid"
SERVER_LOG_FILE = SERVER_DATA_DIR / "server.log"
DEFAULT_SERVER_URL = "http://127.0.0.1:8642"


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def ask_bool(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


def menu_collect():
    cmd_collect(Namespace(label=None, threshold=int(ask("Keystrokes per saved session file", "50"))))


def menu_process():
    cmd_process(Namespace())


def menu_train():
    cmd_train(Namespace(n_estimators=int(ask("Number of estimators", "100")),
                        gpu=ask_bool("Train using CUDA if available?")))


def menu_infer():
    server = ask("Dashboard server URL to report to ('-' for none)", DEFAULT_SERVER_URL)
    cmd_infer(Namespace(label=None, threshold=int(ask("Keystrokes per prediction window", "15")),
                        server=None if server == "-" else server, host_id=None))


def menu_refine():
    cmd_refine(Namespace(yes=False))


def menu_describe_model():
    cmd_describe_model(Namespace())


def _server_running_pid() -> int | None:
    if not SERVER_PID_FILE.exists():
        return None
    try:
        pid = int(SERVER_PID_FILE.read_text().strip())
        os.kill(pid, 0)
    except (ValueError, OSError):
        return None
    return pid


def menu_start_server():
    existing = _server_running_pid()
    if existing is not None:
        print(f"Dashboard server already running (pid {existing}). Log: {SERVER_LOG_FILE}")
        return

    SERVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hand_to_user(SERVER_DATA_DIR)
    # The server doesn't need root: run it as the project's owner -- with all of
    # their groups, not just the primary one (e.g. the venv's interpreter may
    # only be readable through a group such as "conda").
    uid, gid = owner()
    as_user = {}
    if os.geteuid() == 0 and uid != 0:
        as_user = {"user": uid, "group": gid, "extra_groups": os.getgrouplist(pwd.getpwuid(uid).pw_name, gid)}
    host = ask("Listen on host", "127.0.0.1")
    port = ask("Port", "8642")
    refresh = ask("Dashboard refresh interval in seconds", "1")

    log_file = open(SERVER_LOG_FILE, "ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", "biometrics_server", "--host", host, "--port", port,
         "--refresh", refresh, "--data-dir", str(SERVER_DATA_DIR)],
        cwd=SERVER_DIR,
        env={**os.environ, "PYTHONPATH": str(SERVER_DIR)},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        **as_user,
    )
    hand_to_user(SERVER_DATA_DIR)
    print(f"Dashboard server started in background (pid {proc.pid}): http://{host}:{port}/ "
          f"(refresh every {refresh} s)")
    print(f"Log: {SERVER_LOG_FILE}")


def menu_stop_server():
    pid = _server_running_pid()
    SERVER_PID_FILE.unlink(missing_ok=True)
    if pid is None:
        print("Dashboard server is not running.")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):  # wait until it has really exited (and closed its database)
        try:
            # started from this menu, it's our child: reap it, or it lingers as a
            # zombie that still "exists" for os.kill(pid, 0)
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                break
        except ChildProcessError:
            pass
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    print(f"Stopped dashboard server (pid {pid}).")


def menu_reset_dashboard():
    """Wipe everything the dashboard has stored. Agents simply reconnect."""
    if not ask_bool("Delete ALL dashboard data: hosts, incidents, logs, statistics, server log?"):
        print("Cancelled.")
        return
    was_running = _server_running_pid() is not None
    if was_running:
        menu_stop_server()
    removed = [p for p in SERVER_DATA_DIR.glob("*") if p.name.startswith("dashboard.sqlite3") or p.name == "server.log"]
    for path in removed:
        path.unlink()
    print(f"Dashboard reset ({len(removed)} file(s) removed). Running agents reappear on their next report.")
    if was_running and ask_bool("Start the dashboard server again?", default=True):
        menu_start_server()


MENU_OPTIONS = [
    ("Collect labeled keystroke data", menu_collect),
    ("Process raw sessions into training dataset", menu_process),
    ("Train the XGBoost user/other classifier", menu_train),
    ("Run live inference + protection agent", menu_infer),
    ("Refine model (learn sessions flagged for review)", menu_refine),
    ("Start dashboard server (background)", menu_start_server),
    ("Stop dashboard server", menu_stop_server),
    ("Reset dashboard (delete all its data)", menu_reset_dashboard),
    ("Regenerate model info/thresholds without retraining", menu_describe_model),
    ("Exit", None),
]


def main():
    while True:
        print("\nBiometrics CLI")
        for i, (label, _) in enumerate(MENU_OPTIONS, start=1):
            print(f"  {i}) {label}")

        choice = input("Select an option: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(MENU_OPTIONS)):
            print("Invalid choice, try again.")
            continue

        index = int(choice) - 1
        label, action = MENU_OPTIONS[index]
        if action is None:
            break

        try:
            action()
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    main()
