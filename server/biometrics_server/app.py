"""
Dashboard server: receives decisions from the protection agents running on
monitored machines, stores them, and shows them to whoever supervises.

It makes no detection decisions and never sees keystrokes, key frequencies or
models -- every agent decides locally and reports only the outcome plus live
timing statistics. The supervisor can send two commands back: Lock and
Refine model. There is no authentication: run it where only your own
machines and people can reach it (localhost, LAN, or VPN).

    python -m biometrics_server --host 127.0.0.1 --port 8642
"""
import argparse
import atexit
import datetime
import logging
import math
import os
import re
import signal
import sys
import time

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from biometrics_server import glossary
from biometrics_server.store import Store

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ONLINE_WITHIN_SECONDS = 20
DEFAULT_REFRESH_SECONDS = 1.0
HOST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
# Timing-feature names only: key_* / combination_* frequencies would reveal what was typed.
STAT_FEATURE_RE = re.compile(r"^(?!key_(?!per_second$))(?!combination_)[a-z_]{1,40}$")
LEVELS = ("challenge", "alert", "critical")
OUTCOMES = ("reauthenticated", "closed", "timed_out", "superseded")
COMMANDS = ("lockout", "refine")


class BadRequest(ValueError):
    pass


# ---- input validation (agents are trusted-network, but never trust shapes) --

def _text(value, limit=200) -> str | None:
    return None if value is None else str(value)[:limit]


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"not a number: {value!r}")
    if not math.isfinite(number):
        raise BadRequest("numbers must be finite")
    return number


def _host_id(value) -> str:
    if not isinstance(value, str) or not HOST_ID_RE.match(value):
        raise BadRequest("host_id must match [A-Za-z0-9_.-]{1,64}")
    return value


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _stat_features(value) -> dict:
    return {k: _num(v) for k, v in list(_dict(value).items())[:40] if STAT_FEATURE_RE.match(str(k))}


def _clean_event(e: dict) -> dict:
    kind = e.get("kind")
    base = {"kind": kind, "time": _num(e.get("time")) or time.time(),
            "incident_id": _text(e.get("incident_id"), 32)}
    if kind == "incident":
        if e.get("level") not in LEVELS:
            raise BadRequest(f"unknown level {e.get('level')!r}")
        return {**base, "level": e["level"], "impostor_score": _num(e.get("impostor_score")),
                "confidence": _num(e.get("confidence")), "label": _text(e.get("label"), 16),
                "trigger": _text(e.get("trigger"), 40), "follows": _text(e.get("follows"), 32),
                "thresholds": {k: _num(_dict(e.get("thresholds")).get(k)) for k in ("tau_warn", "tau_critical")},
                "reasons": [{"feature": _text(r.get("feature")), "label": _text(r.get("label")),
                             "value": _num(r.get("value")), "typical": _num(r.get("typical")),
                             "typical_low": _num(r.get("typical_low")), "typical_high": _num(r.get("typical_high")),
                             "direction": _text(r.get("direction"), 20), "impact": _num(r.get("impact"))}
                            for r in (e.get("reasons") or [])[:10] if isinstance(r, dict)]}
    if kind == "response":
        if e.get("outcome") not in OUTCOMES:
            raise BadRequest(f"unknown outcome {e.get('outcome')!r}")
        return {**base, "level": _text(e.get("level"), 16), "outcome": e["outcome"]}
    if kind == "note":
        return {**base, "text": _text(e.get("text"), 2000)}
    raise BadRequest(f"unknown event kind {kind!r}")


def _clean_status(status: dict) -> dict:
    refine = _dict(status.get("refine"))
    return {"windows": int(_num(status.get("windows")) or 0),
            "last_score": _num(status.get("last_score")),
            "recent_scores": [_num(v) for v in (status.get("recent_scores") or [])[-30:]],
            "last_level": _text(status.get("last_level"), 16),
            "features": _stat_features(status.get("features")),
            "armed": bool(status.get("armed")),
            "open_incident": _text(status.get("open_incident"), 16),
            "pending_review": int(_num(status.get("pending_review")) or 0),
            "refine": {"state": _text(refine.get("state"), 16), "message": _text(refine.get("message"), 400),
                       "time": _num(refine.get("time")), "since": _num(refine.get("since"))},
            "started": _num(status.get("started"))}


# ---- presentation helpers -----------------------------------------------------

def _dt(ts) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "–"


def _ago(ts) -> str:
    if not ts:
        return "never"
    s = int(time.time() - ts)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h {s % 3600 // 60}m ago"
    return f"{s // 86400}d ago"


def _online(host: dict) -> bool:
    return bool(host["connected"]) and time.time() - (host["last_seen"] or 0) < ONLINE_WITHIN_SECONDS


def _summarize(event: dict) -> str:
    d = event["data"]
    kind = event["kind"]
    if kind == "connected":
        text = f"Agent connected (user {d.get('user')}, model {d.get('model_hash')})"
        return text + " — new model" if d.get("model_changed") else text
    if kind == "disconnected":
        return "Agent stopped"
    if kind == "incident":
        return (f"{d['level'].upper()} — impostor score {d['impostor_score']:.0%}. "
                f"{glossary.TRIGGERS.get(d.get('trigger'), '')}")
    if kind == "response":
        return f"{(d.get('level') or '').capitalize()} → {d['outcome'].replace('_', ' ')}. " \
               f"{glossary.OUTCOMES.get(d['outcome'], '')}"
    if kind == "command":
        return {"lockout": "Lock sent from the dashboard",
                "refine": "Refine model requested from the dashboard"}.get(d.get("type"), d.get("type"))
    if kind == "note":
        return d.get("text") or ""
    return kind


def create_app(data_dir: str = DEFAULT_DATA_DIR, refresh_seconds: float = DEFAULT_REFRESH_SECONDS) -> Flask:
    app = Flask(__name__)
    store = Store(os.path.join(data_dir, "dashboard.sqlite3"))

    app.jinja_env.filters.update(dt=_dt, ago=_ago, pct=lambda v: "–" if v is None else f"{v:.0%}")
    app.jinja_env.globals.update(
        glossary=glossary, online=_online, summarize=_summarize, fmt=glossary.format_value,
        feature=glossary.feature, explain=glossary.explain_reason, scale=glossary.scale,
        refresh_seconds=refresh_seconds)

    @app.errorhandler(BadRequest)
    def bad_request(e):
        return jsonify({"error": str(e)}), 400

    # ---- agent API ------------------------------------------------------------

    @app.post("/api/hello")
    def api_hello():
        body = _dict(request.get_json(silent=True))
        host, model = _dict(body.get("host")), _dict(body.get("model"))
        thresholds = _dict(model.get("thresholds"))
        clean_host = {"host_id": _host_id(host.get("host_id")), "user": _text(host.get("user"), 64),
                      "hostname": _text(host.get("hostname"), 128), "platform": _text(host.get("platform"), 128)}
        clean_model = {
            "hash": _text(model.get("hash"), 64),
            "thresholds": {k: _num(thresholds.get(k)) for k in ("tau_warn", "tau_critical", "eer", "eer_far", "auc")},
            "info": _dict(model.get("info")),
            "profile": {k: {p: _num(_dict(v).get(p)) for p in ("typical", "low", "high")}
                        for k, v in list(_dict(model.get("profile")).items())[:40] if STAT_FEATURE_RE.match(str(k))},
            "loaded_at": _text(model.get("loaded_at"), 32),
        }
        if not clean_model["hash"] or clean_model["thresholds"]["tau_warn"] is None:
            raise BadRequest("model.hash and model.thresholds are required")
        store.hello(clean_host, clean_model)
        log.info("Agent connected: %s (user %s)", clean_host["host_id"], clean_host["user"])
        return jsonify({"status": "ok"})

    @app.post("/api/report")
    def api_report():
        body = _dict(request.get_json(silent=True))
        host_id = _host_id(body.get("host_id"))
        events = [_clean_event(_dict(e)) for e in (body.get("events") or [])[:200]]
        commands = store.report(host_id, _clean_status(_dict(body.get("status"))), events)
        if commands is None:
            return jsonify({"error": "unknown host, send /api/hello first"}), 409
        return jsonify({"status": "ok", "commands": commands})

    @app.post("/api/bye")
    def api_bye():
        store.bye(_host_id(_dict(request.get_json(silent=True)).get("host_id")))
        return jsonify({"status": "ok"})

    @app.get("/api/live")
    def api_live():
        """For the dashboard's own pages: new incidents since `after` (toasts, flags)."""
        latest, incidents = store.incidents_after(request.args.get("after", 0, type=int))
        return jsonify({"db": store.instance, "latest": latest, "incidents": [
            {"id": i["id"], "host_id": i["host_id"], "level": i["level"], "time": _dt(i["time"]),
             "score": i["data"].get("impostor_score"),
             "trigger": glossary.TRIGGERS.get(i["data"].get("trigger"), ""),
             "url": url_for("incident", incident_id=i["incident_id"])} for i in incidents]})

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    # ---- dashboard --------------------------------------------------------------

    def _host_or_404(host_id):
        h = store.host(host_id)
        if h is None:
            abort(404)
        return h

    @app.get("/")
    def overview():
        hosts = store.hosts()
        return render_template("overview.html", hosts=hosts, summary=store.summary(),
                               online_count=sum(_online(h) for h in hosts))

    @app.get("/hosts/<host_id>")
    def host(host_id):
        h = _host_or_404(host_id)
        return render_template("host.html", host=h, incidents=store.incidents(host_id, limit=5),
                               log=store.events(host_id=host_id, limit=30))

    @app.post("/hosts/<host_id>/<command>")
    def host_command(host_id, command):
        _host_or_404(host_id)
        if command not in COMMANDS:
            abort(404)
        store.command(host_id, command)
        log.info("Dashboard command %s -> %s", command, host_id)
        return redirect(request.referrer or url_for("host", host_id=host_id))

    @app.get("/incidents")
    def incidents():
        host_id = request.args.get("host") or None
        hosts = store.hosts()
        return render_template("incidents.html", incidents=store.incidents(host_id), hosts=hosts,
                               users={h["host_id"]: h["user"] for h in hosts}, host_id=host_id)

    @app.get("/incidents/<incident_id>")
    def incident(incident_id):
        i = store.incident(incident_id)
        if i is None:
            abort(404)
        return render_template("incident.html", i=i, host=store.host(i["host_id"]))

    @app.get("/statistics")
    def statistics():
        order = list(glossary.FEATURES)
        hosts = store.hosts()
        for h in hosts:
            profile = h["model"].get("profile") or {}
            h["model"]["profile"] = dict(sorted(profile.items(),
                                                key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))
        return render_template("statistics.html", hosts=hosts)

    @app.get("/logs")
    def logs():
        host_id = request.args.get("host") or None
        kind = request.args.get("kind") or None
        before = request.args.get("before", type=int)
        events = store.events(host_id=host_id, kind=kind, before=before, limit=100)
        return render_template("logs.html", events=events, hosts=store.hosts(), host_id=host_id, kind=kind,
                               older=events[-1]["id"] if len(events) == 100 else None)

    @app.get("/help")
    def help_page():
        return render_template("help.html")

    return app


def _own_pid_file(path: str) -> None:
    """The server records its own PID (whoever started it), so run.py's Stop
    and Reset always find the right process; removed again on exit."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(os.getpid()))

    def remove():
        try:
            with open(path) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(path)
        except OSError:
            pass

    atexit.register(remove)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # run atexit on a plain `kill` too


def main():
    parser = argparse.ArgumentParser(description="Biometrics dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Where the SQLite database is kept")
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH_SECONDS,
                        help="How often the dashboard pages update themselves, in seconds (default: 1)")
    args = parser.parse_args()
    if args.refresh <= 0:
        parser.error("--refresh must be a positive number of seconds")

    _own_pid_file(os.path.join(args.data_dir, "server.pid"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # skip a line per heartbeat
    print(f"Dashboard: http://{args.host}:{args.port}/  (refresh every {args.refresh:g} s, "
          f"data: {os.path.abspath(args.data_dir)})")
    create_app(args.data_dir, args.refresh).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
