"""
The protection agent, running on the owner's machine.

  Sensory      KeyLogger captures keystrokes; XGBoost scores each window
  Cognitive    policy.Policy turns the score into a level
  Explainable  explain.Explainer says which typing features looked wrong
  Action       actions.py (on_challenge / on_alert / on_critical) responds,
               and the human answers through the popup (human-in-the-loop)

Everything is decided locally. The dashboard server only receives the outcome
(see reporter.py for exactly what is sent) and can send back two commands:
"lockout" (show the lock window now) and "refine" (retrain with the reviewed
sessions). Without a server the agent works the same.
"""
import collections
import datetime
import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field

from biometrics import paths
from biometrics.agent import actions
from biometrics.agent.explain import Explainer
from biometrics.agent.policy import Level, Policy
from biometrics.training import review_queue
from biometrics.training.train_xgboost import is_one_hot

log = logging.getLogger(__name__)

TRIGGER_SUPERVISOR_LOCKOUT = "supervisor_lockout"
MAX_SUSPECT_WINDOWS = 40   # cap on keystroke windows kept from one suspicious stretch
RECENT_SCORES = 30


@dataclass
class Incident:
    level: str
    impostor_score: float
    trigger: str
    reasons: list[dict]
    label: str | None = None
    confidence: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    time: float = field(default_factory=time.time)
    follows: str | None = None  # id of the incident this one escalated from


def _local_user() -> str:
    # entrypoint.sh runs us as root; report the human (the project's owner), not root.
    import pwd
    user = getpass.getuser()
    if user == "root":
        uid = paths.owner()[0]
        if uid != 0:
            user = pwd.getpwuid(uid).pw_name
    return user


def _read_json(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class Agent:
    def __init__(self, host_id: str = None, server_url: str = None):
        from biometrics.agent.popup import PopupManager

        self.host_id = host_id or socket.gethostname()
        self.popups = PopupManager()
        self.on_model_reloaded = []  # callbacks, e.g. the KeyLogger reloading its Inference

        self._lock = threading.Lock()
        self._open: Incident | None = None      # incident whose popup is on screen
        self._suspect: list[list[dict]] = []    # keystroke windows of the current suspicious stretch
        self._recent = collections.deque(maxlen=RECENT_SCORES)
        self._last_features: dict = {}
        self._windows = 0
        self._last_level = Level.ALLOW.value
        self._started = time.time()
        self._refine = {"state": "idle"}
        self._load_model()

        self.reporter = None
        if server_url:
            from biometrics.agent.reporter import Reporter
            self.reporter = Reporter(server_url, self._hello(), self._status, self._on_command)

    def _load_model(self) -> None:
        thresholds = _read_json(paths.THRESHOLDS_FILE)
        if not thresholds:
            raise FileNotFoundError(f"{paths.THRESHOLDS_FILE} missing -- train the model first")
        self.thresholds = thresholds
        self.explainer = Explainer()
        self.stat_features = [c for c in self.explainer.columns if not is_one_hot(c)]
        if hasattr(self, "policy"):
            self.policy.tau_warn, self.policy.tau_critical = thresholds["tau_warn"], thresholds["tau_critical"]
            self.policy.reauthenticated()  # new model, start detection over
        else:
            self.policy = Policy(thresholds["tau_warn"], thresholds["tau_critical"])

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self.popups.start()
        if self.reporter:
            self.reporter.start()

    def stop(self) -> None:
        self.popups.stop()
        if self.reporter:
            self.reporter.stop()

    # ---- Sensory -> Cognitive -> Explainable -> Action -------------------

    def on_prediction(self, label: str, confidence: float, feature_row: dict = None,
                      keystroke_window: list[dict] = None) -> None:
        """KeyLogger callback, once per scored window."""
        confidence = float(confidence)
        feature_row = feature_row or {}
        try:
            self._decide(label, confidence, feature_row, keystroke_window)
        finally:
            if self.reporter:
                self.reporter.poke()  # live score and statistics on the dashboard, now

    def _decide(self, label: str, confidence: float, feature_row: dict, keystroke_window) -> None:
        with self._lock:
            decision = self.policy.decide(label, confidence)
            self._windows += 1
            self._last_level = decision.level.value
            self._recent.append(round(decision.impostor_score, 4))
            self._last_features = {f: float(feature_row[f]) for f in self.stat_features if f in feature_row}

            # Keep the windows of a suspicious stretch: if the person turns out to be
            # the owner (re-auth at challenge/alert), they go to the review queue.
            if decision.impostor_score >= self.policy.tau_warn or self._open:
                if keystroke_window:
                    self._suspect = (self._suspect + [list(keystroke_window)])[-MAX_SUSPECT_WINDOWS:]
            else:
                self._suspect = []

            if decision.level == Level.ALLOW:
                return
            open_incident = self._open
            if open_incident and decision.level.rank <= Level(open_incident.level).rank:
                return  # already on screen at this level or higher -- don't stack popups

        incident = Incident(
            level=decision.level.value,
            impostor_score=decision.impostor_score,
            trigger=decision.trigger,
            reasons=self.explainer.reasons(feature_row),
            label=label,
            confidence=confidence,
            follows=open_incident.id if open_incident else None,
        )
        log.info("%s: impostor score %.2f (%s)", incident.level.upper(), incident.impostor_score, incident.trigger)
        self._run_hook(incident, supersedes=open_incident)

    def _run_hook(self, incident: Incident, supersedes: Incident = None) -> None:
        if supersedes:
            self.popups.close(supersedes.id)
            self._respond(supersedes, "superseded")
        with self._lock:
            self._open = None
        self._report({"kind": "incident", "incident_id": incident.id, **asdict(incident),
                      "thresholds": self._thresholds_now()})
        try:
            getattr(actions, f"on_{incident.level}")(incident, self)
        except Exception:
            log.exception("actions.on_%s raised", incident.level)
            self.note(f"actions.on_{incident.level} raised: {traceback.format_exc(limit=2)}",
                      incident_id=incident.id)

    # ---- the API actions.py uses ------------------------------------------

    def show_challenge(self, incident: Incident) -> None:
        self._show("challenge", incident)

    def show_alert(self, incident: Incident) -> None:
        self._show("alert", incident)

    def show_lockout(self, incident: Incident) -> None:
        self._show("lockout", incident)

    def shutdown_hook(self, incident: Incident, dry_run: bool = True) -> None:
        """
        Placeholder for real enforcement (end the session, cut the network, ...).
        Only ever logs: wiring a real action is a deliberate, reviewed decision.
        """
        if not dry_run:
            raise NotImplementedError("No real enforcement action is configured")
        log.info("[DRY RUN] shutdown_hook for incident %s -- no action taken", incident.id)
        self.note("shutdown_hook called (dry run, no action taken)", incident_id=incident.id)

    def note(self, text: str, incident_id: str = None) -> None:
        log.info("note: %s", text)
        self._report({"kind": "note", "time": time.time(), "text": str(text)[:2000], "incident_id": incident_id})

    # ---- human-in-the-loop responses (called from the popup thread) ---------

    def _show(self, popup_kind: str, incident: Incident) -> None:
        with self._lock:
            self._open = incident
        self.popups.show(
            popup_kind, incident,
            on_reauth=lambda: self._in_background(self._reauthenticated, incident),
            on_dismiss=lambda how: self._in_background(self._dismissed, incident, how),
        )

    @staticmethod
    def _in_background(fn, *args) -> None:
        # Popup callbacks run on the single Tk thread; never block it.
        threading.Thread(target=fn, args=args, daemon=True).start()

    def _reauthenticated(self, incident: Incident) -> None:
        with self._lock:
            self.policy.reauthenticated()
            self._open = None
            windows, self._suspect = self._suspect, []
        self.popups.close_all()
        self._respond(incident, "reauthenticated")

        if incident.level == Level.CRITICAL.value:
            return  # see review_queue's docstring: critical re-auth is not learned from
        saved = review_queue.save_for_review(windows, asdict(incident))
        if saved:
            self.note(f"The model was wrong: {sum(map(len, windows))} keystrokes from this "
                      f"{incident.level} flagged for review (use 'refine model' to learn them)",
                      incident_id=incident.id)

    def _dismissed(self, incident: Incident, how: str) -> None:
        """how: "closed" (window X) or "timed_out" (challenge countdown ran out)."""
        with self._lock:
            still_current = self._open is incident
            if still_current:
                self._open = None
            if incident.level == Level.ALERT.value:
                self.policy.alert_dismissed()
        self._respond(incident, how)

        if incident.level == Level.CHALLENGE.value and still_current:
            decision = self.policy.challenge_ignored(incident.impostor_score)
            self._run_hook(Incident(level=decision.level.value, impostor_score=decision.impostor_score,
                                    trigger=decision.trigger, reasons=incident.reasons,
                                    label=incident.label, confidence=incident.confidence,
                                    follows=incident.id))

    def _respond(self, incident: Incident, outcome: str) -> None:
        log.info("%s %s: %s", incident.level, incident.id, outcome)
        self._report({"kind": "response", "time": time.time(), "incident_id": incident.id,
                      "level": incident.level, "outcome": outcome})

    # ---- commands from the dashboard ------------------------------------------

    def _on_command(self, command: dict) -> None:
        kind = command.get("type")
        if kind == "lockout":
            self._in_background(self._supervisor_lockout)
        elif kind == "refine":
            self._in_background(self.refine_model)
        else:
            log.warning("Unknown command from dashboard: %r", command)

    def _supervisor_lockout(self) -> None:
        with self._lock:
            supersedes = self._open
            score = self._recent[-1] if self._recent else 0.0
        incident = Incident(level=Level.CRITICAL.value, impostor_score=score,
                            trigger=TRIGGER_SUPERVISOR_LOCKOUT, reasons=[],
                            follows=supersedes.id if supersedes else None)
        if supersedes:
            self.popups.close(supersedes.id)
            self._respond(supersedes, "superseded")
        self._report({"kind": "incident", "incident_id": incident.id, **asdict(incident),
                      "thresholds": self._thresholds_now()})
        self.show_lockout(incident)

    def refine_model(self) -> None:
        """Learn the reviewed sessions (review_queue.refine), then switch to the new model."""
        with self._lock:
            if self._refine["state"] == "running":
                return
            self._refine = {"state": "running", "since": time.time()}
        self.note("Refining the model with the reviewed sessions...")
        try:
            result = review_queue.refine(log=self.note)
            with self._lock:
                self._load_model()
            for callback in self.on_model_reloaded:
                callback()
            t = result["thresholds"]
            message = (f"Refined with {result['sessions_added']} session(s) / {result['keystrokes_added']} "
                       f"keystrokes: accuracy {result['accuracy']:.2%}, τ warn {t['tau_warn']:.3f}, "
                       f"τ critical {t['tau_critical']:.3f}. The new model is active.")
            state = "done"
        except Exception as e:
            log.exception("Refine failed")
            message, state = f"Refine failed: {e}", "failed"
        with self._lock:
            self._refine = {"state": state, "message": message, "time": time.time()}
        self.note(message)
        if self.reporter and state == "done":
            self.reporter.update_hello(self._hello())

    # ---- reporting ------------------------------------------------------------

    def _report(self, event: dict) -> None:
        if self.reporter:
            self.reporter.send(event)

    def _thresholds_now(self) -> dict:
        return {"tau_warn": self.policy.tau_warn, "tau_critical": self.policy.tau_critical}

    def _status(self) -> dict:
        with self._lock:
            return {
                "windows": self._windows,
                "last_score": self._recent[-1] if self._recent else None,
                "recent_scores": list(self._recent),
                "last_level": self._last_level,
                "features": dict(self._last_features),
                "armed": self.policy.armed,
                "open_incident": self._open.level if self._open else None,
                "pending_review": len(review_queue.pending()),
                "refine": dict(self._refine),
                "started": self._started,
            }

    def _hello(self) -> dict:
        profile = _read_json(paths.FEATURE_PROFILE_FILE)
        return {
            "host": {
                "host_id": self.host_id,
                "user": _local_user(),
                "hostname": socket.gethostname(),
                "platform": f"{platform.system()} {platform.release()}",
            },
            "model": {
                "hash": hashlib.sha256(paths.MODEL_FILE.read_bytes()).hexdigest()[:12],
                "thresholds": self.thresholds,
                "info": _read_json(paths.MODEL_INFO_FILE),
                # the user's typical range per timing feature -- never key/combination features
                "profile": {f: profile[f] for f in self.stat_features if f in profile},
                "loaded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        }
