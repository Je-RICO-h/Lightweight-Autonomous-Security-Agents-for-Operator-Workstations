# Autonomous Access-Control Agent — Design

> **v2 architecture (2026-09-29, current). Read this first.** The agent now
> runs entirely on the owner's machine: inference, the policy decision, SHAP
> explanation, and the popups (`src/biometrics/agent/`). The server moved to
> its own folder (`server/biometrics_server/`). It is a Flask + SQLite
> dashboard that only receives decisions (level, score, top reasons, the
> user's response, a heartbeat) and makes none.
>
> This supersedes the parts of this document that put the policy and SHAP on
> the server:
> - §0: the server no longer decides.
> - §2a / §7: no `/register` model upload, no `/predict`, no long-polling, and no notifier. Instead the agent pushes `/api/hello`, `/api/report` and `/api/bye`.
> - §6: SHAP runs locally, so the "feature row leaves the machine" exception is gone.
>
> Retraining (§5, updated): windows from a stretch where the owner
> re-authenticated at challenge/alert are flagged for review. "Refine model"
> (menu, `biometrics refine`, dashboard button) learns them as the main user.
> Critical re-auth is never learned. Nothing retrains without that explicit
> trigger. `review-retrain` is gone. It also had a bug: accepted files kept
> a non-"User" label, so they would have trained as impostors.
>
> Still in force: thresholds from ROC/EER (§3), the dry-run `shutdown_hook` (now `Agent.shutdown_hook`, called
> from `actions.on_critical`), and no client/server auth (§8).
>
> Tier names are now allow / challenge / alert / **critical**. The old
> "Lockout" became "critical", whose default action is the lockout window.
> What each level *does* lives in `src/biometrics/agent/actions.py`.

Status: **design finalized, not implemented**. All open questions in the
original draft (§9), plus a follow-up architecture change (§0), have been
resolved with the user; see the "Resolved decisions" note at the end of each
affected section. This document specifies the agent described in Lakatos
Róbert's note (2026-09-18): extending the XGBoost classifier from a passive
"who's typing" model into the cognitive core of an autonomous agent that
reacts when the current typist is judged not to be the enrolled ("main")
user. Framed against the paper's PEP-AI (Precise, Explainable, Provable)
requirements.

**Important scope correction from the original draft:** this first version
does **not** lock the OS session or take any blocking action. "Lock" tier
means *surface a high-confidence alert*, not enforce anything. The agent is
strictly observe-and-inform in v1 — see §4.

## 0. Architecture change: the agent is a server, not an in-process callback

The original draft (§1-§2 below, as first written) assumed `PolicyAgent`
lives inside the same `biometrics infer` process as the keylogger, wired
through an in-process `on_prediction` callback, with a possible future
fleet/cloud supervisor mode as separate follow-up work. **That assumption no
longer holds.** The user has since specified the agent must run "always in
the background," and clarified what that means concretely: the agent *is* a
persistent Flask/Django HTTP server, always running, and the machine doing
keylogging + XGBoost inference is a **separate client** that POSTs each
prediction to it over HTTP. The server owns the policy decision and the
`shutdown_hook` call; the client owns the keyboard, the model, and anything
that touches that machine's screen/OS.

This pulls what was sketched as future "fleet/cloud mode" (old §8) forward
to be the v1 default, not a later phase. The rest of this document has been
rewritten around that split. Sections below marked "Resolved (architecture
change)" reflect this; anything not marked that way was decided before this
change and still holds.

## 1. Why an agent, not just a classifier

Today `biometrics infer` calls `Inference.predict()` every `threshold`
keystrokes and prints a label + confidence. That's it — no action is taken,
and nothing survives past the lifetime of that one terminal command. The
agent turns that output into a decision loop that runs independently of any
single keylogging session:

```
Client (this machine)                       Server (always running)
Sensory (keylogger + feature pipeline)
        -> Cognitive (XGBoost prediction)
        -> POST /predict  ------------------>  Decision (policy over thresholds + state)
                                                -> Action (alert / challenge / notify / no-op)
                                                -> Explainable (SHAP-backed justification)
        <---- long-poll for pending command ----  (Challenge/Alert -> show PySide6 popup)
```

The XGBoost model itself does not change and never leaves the client. It
stays a lightweight binary classifier (`User` vs `Other`) running wherever
the keylogger runs. The server only ever receives
`(timestamp, host_id, label, confidence)` plus the feature row needed to
compute SHAP server-side (see §6) — never raw keystrokes. This keeps the
server's job cheap enough to run continuously and, per the original note,
"even in the cloud," since it never needs the sensitive biometric timing
data itself.

## 2. Scope boundary (what ships now vs. later)

In scope for a first implementation (not built this session):
- A Flask/Django `agent server` process: `POST /predict` receives a
  prediction from a client, runs the policy, and returns/queues a command;
  a long-poll endpoint the client uses to receive Challenge/Alert commands
  (see §7). Runs as an OS service so it survives terminal close/reboot (§2a).
- A thin `agent client` in the `biometrics` package: replaces the current
  in-process `on_prediction` callback with an HTTP POST to the server, plus
  a background long-poll loop that shows the PySide6 popup when told to.
- Two thresholds, `tau_warn` and `tau_critical`, **computed dynamically**
  from the trained model's own ROC/EER on its held-out test set — not fixed
  constants (see §3). Computed client-side at train time, sent to the server
  per host (or bundled with the model artifacts the server also has access
  to — see §7 open question).
- Three tiers: **allow** (no-op), **challenge** (HITL popup on the client),
  **alert** (high-confidence notice — renamed from "lock"; it does not lock
  anything in v1, see §4).
- SHAP explanation generation for any prediction that crosses `tau_warn`.
- A local audit log (JSONL, on the server) of every decision + its SHAP
  justification — this is the "Provable" leg of PEP-AI.
- A pluggable `Notifier` interface on the server so a real alerting channel
  can be swapped in later without touching decision logic (§7).
- A `remote_command` error path: if the server's callback to a specific
  client fails (client unreachable, long-poll dead), it logs the failure
  explicitly and marks that host's channel halted rather than silently
  assuming the popup was shown (§7).
- A `shutdown_hook(reason, dry_run=True)` placeholder on the server —
  proves the policy layer can call a "take this offline" action without
  wiring one in yet (§4, §7).
- A human-reviewed retrain queue: confirmed-via-re-auth sessions get saved
  to a review folder on the client; folding them into training requires an
  explicit, separate command (§5).

Explicitly out of scope for v1:
- Any OS-level session lock or network/TCP suspension. Both are real
  privileged actions with destructive blast radius; v1 only informs.
- Auth between client and server (§2a) — v1 trusts the network (localhost
  or a private/VPN segment). Not safe to expose on an open network as-is.
- A real client-side listening port for server-initiated callbacks — v1
  uses client-initiated long-polling instead specifically to avoid needing
  inbound network access to every client (§7).
- Re-enrollment / profile drift handling beyond the reviewed retrain queue
  in §5.

### 2a. Resolved (architecture change): client/server trust and transport

- **Auth**: none in v1. The agent server is assumed to run somewhere only
  reachable by its own clients (localhost, private LAN, or VPN segment) —
  not exposed to an open network. This is an explicit, accepted limitation
  for a first version, not an oversight; adding a per-host API token is the
  natural next step whenever this needs to run somewhere less trusted.
- **Callback direction**: the client always initiates. It long-polls a
  server endpoint (e.g. `GET /commands/<host_id>?wait=30`) for a pending
  Challenge/Alert command, rather than the server opening a connection to
  the client. This works through NAT/firewalls without any inbound port
  needing to be open on the client machine — the tradeoff accepted for that
  is added latency (up to the long-poll timeout) before a popup appears,
  which is fine at the scale of "seconds," not fine at "sub-second."
- **Server lifecycle**: "always running in the background" is satisfied by
  the server being a registered OS service (systemd user/system service on
  Linux, Windows Service or Scheduled Task on Windows, launchd on macOS) on
  whatever machine hosts it — auto-start on boot, auto-restart on crash.
  Writing the actual service-unit files is implementation work, not a design
  question; the server process itself must be written to run cleanly under
  a service manager (proper signal handling for shutdown, no dependency on
  an interactive terminal, structured logging instead of only stdout prints).

## 3. Decision policy

Two thresholds on the XGBoost confidence that the current typist is **not**
the main user, `f_XGB`:

| Range | Meaning | Action |
|---|---|---|
| `f_XGB < tau_warn` | Consistent with enrolled user | **Allow** — log only |
| `tau_warn <= f_XGB < tau_critical` | Suspicious, not conclusive | **Challenge** (HITL) |
| `f_XGB >= tau_critical` | High-confidence impostor | **Alert** (informational, see §4) |

### Resolved: thresholds are computed, not hardcoded

Rather than fixed constants, both thresholds are derived from the *current*
trained model's own ROC curve on its held-out test set, every time a model
is (re)trained:

- `tau_critical` = the ROC operating point at the Equal Error Rate (EER) —
  where false-accept rate equals false-reject rate. This is the standard
  biometric-systems operating point and is exactly the `eer_threshold`
  already computed (but not persisted) in `Model_Training/XgBoost.ipynb`'s
  ROC cell.
- `tau_warn` = a configurable margin below `tau_critical` (default: the ROC
  threshold at a target false-accept rate a few points looser than EER,
  e.g. FAR ≈ 2× the EER's FAR — concretely: walk the `roc_curve` output
  until `fpr` first exceeds `2 * eer_far`, take that threshold). This gives
  a genuine "suspicious but not conclusive" band instead of an arbitrary gap.

Implementation shape: `training/train_xgboost.py`'s `train()` already
computes `y_proba` via `predict_proba` internally for its classification
report; extending it to also run `roc_curve`/`eer` and **write
`thresholds.json` (`{tau_warn, tau_critical, eer, auc}`) into `model_dir`
alongside the model artifacts** means every retrain automatically
recalibrates the policy — no manual tuning step, no stale thresholds after
retraining on new data.

```python
# Sketch, not final code — thresholds.py, called from train()
def compute_thresholds(y_test, y_proba_other, warn_margin_multiplier=2.0):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_other)
    eer_idx = np.argmin(np.abs(fpr + tpr - 1))
    tau_critical = thresholds[eer_idx]
    eer_far = fpr[eer_idx]

    warn_idx = np.argmax(fpr > warn_margin_multiplier * eer_far)
    tau_warn = thresholds[warn_idx] if warn_idx > 0 else tau_critical * 0.85

    return {"tau_warn": float(tau_warn), "tau_critical": float(tau_critical),
            "eer": float(eer_far), "auc": float(auc(fpr, tpr))}
```

### Resolved (architecture change): thresholds travel with the client, policy state lives per-host on the server

The client sends its own `thresholds.json` contents (computed at train time,
§3 above) as part of each `POST /predict` payload — or, equivalently, once
at client registration/startup and cached server-side keyed by `host_id`.
Either way the server ends up with a per-host `{tau_warn, tau_critical}` it
applies to that host's predictions; it does not assume one global threshold
pair, since different deployments' models (and thus EER) can differ.

State also matters, not just the instantaneous score: a single borderline
window (one `threshold`-sized keystroke batch, e.g. 15–50 keys) should not
trigger an alert. The server keeps a short rolling window of the last N
predictions **per `host_id`** (e.g. last 3) and requires two consecutive
`challenge`-tier results, or one `critical`-tier result, before acting —
this avoids flapping on a single noisy batch (user reaching for coffee
mid-sentence, etc). Because the server is now handling potentially many
clients, this state is a dict keyed by host, not a single instance.

```python
# Sketch, not final code — this is the server's request handler logic
class PolicyEngine:
    def __init__(self, window=3):
        self.recent_by_host: dict[str, deque] = {}  # host_id -> deque of impostor_scores

    def handle_prediction(self, host_id, label, confidence, feature_row, thresholds):
        tau_warn, tau_critical = thresholds["tau_warn"], thresholds["tau_critical"]
        recent = self.recent_by_host.setdefault(host_id, deque(maxlen=self.window))
        impostor_score = confidence if label == "Other" else 1 - confidence
        recent.append(impostor_score)

        if impostor_score >= tau_critical:
            return self._act(host_id, Action.ALERT, impostor_score, feature_row)
        elif impostor_score >= tau_warn and self._consecutive_warns(recent, tau_warn) >= 2:
            return self._act(host_id, Action.CHALLENGE, impostor_score, feature_row)
        else:
            return self._act(host_id, Action.ALLOW, impostor_score, feature_row)
```

`_act` is what queues a command for that `host_id` to be picked up by the
client's long-poll (§7), writes to the audit log, and calls the `Notifier`.

## 4. Actions

### Resolved: v1 is observe-and-inform, not enforce

The original draft had "Lock" actually locking the OS session. **That is
removed from v1.** The agent never blocks input, locks the screen, or
suspends the network on its own. The highest tier ("Alert") only makes the
situation *visible* — to the user via the same HITL window used for
Challenge, and to whatever is watching the audit log / notifier. Enforcement
is a decision for a human or a future, separately-reviewed version — not
something this agent does autonomously yet.

### Allow
No user-visible effect. Server writes to the audit log at debug level. No
command is queued for the client.

### Challenge (Human-in-the-Loop)

Split across both sides, per the §0 architecture change:

**Server side**, when `PolicyEngine` decides Challenge:
1. Writes a `challenge` event (with SHAP top features, §6) to the audit log.
2. Queues a `show_challenge` command for that `host_id`, picked up by the
   client's next long-poll response (§7).
3. Calls the configured `Notifier` (§7).

**Client side**, on receiving a `show_challenge` command:
1. Opens a small always-on-top **PySide6** window. Typing is not blocked at
   the OS level — this is advisory, not enforcement.
2. Window shows: the SHAP-derived top contributing features in plain
   language (e.g. "seek_time between SHIFT and F is 40% longer than usual"),
   a re-authenticate option (OS password prompt — never a plaintext field
   owned by the agent or sent to the server), and an escalate/notify-
   supervisor option.
3. A countdown (default 30s, configurable) auto-escalates: the client POSTs
   a `challenge_timeout` event back to the server, which the server treats
   as equivalent to an **Alert**-tier event for that host. Nothing is locked
   — it only raises the severity of the log entry and notification.
4. On successful local re-auth: the client POSTs a `confirmed_user` event to
   the server (which clears that host's rolling window), and **writes the
   triggering feature row + session context to a local review queue**
   (`data/retrain_review/` — on the client, since that's where the raw
   keystroke data lives; it is never uploaded to the server). See §5.

### Alert (renamed from "Lock")

**Server side**, when `PolicyEngine` decides Alert (or a Challenge times out):
1. Writes an `alert` event with full SHAP justification to the audit log.
2. Calls the configured `Notifier` (§7) — local JSONL by default, pluggable
   for a real webhook/server later.
3. Calls `shutdown_hook(reason, dry_run=True)` (§7) — logs that this is the
   point where a real enforcement action *could* plug in, takes no action.
4. Queues a `show_alert` command for that `host_id`, same delivery path as
   Challenge (§7).

**Client side**, on receiving a `show_alert` command: same PySide6 window
as Challenge, styled as higher-severity, with no auto-dismiss countdown
pressuring a response — it's informational only.

OS session lock and network suspension are **not** implemented in v1 — see
the scope correction at the top of this document and §2.

## 5. Retrain review queue

### Resolved: confirmed sessions are queued for human-approved retraining

A challenge that ends in successful re-authentication is evidence the
enrolled user's typing pattern may be drifting (fatigue, injury, new
keyboard, etc) — genuinely useful retraining signal. But auto-folding it
into the model is a poisoning vector: anyone who obtains the OS password
once (the same credential the agent's re-auth relies on) could then
repeatedly "confirm" fabricated sessions and gradually walk the model's
decision boundary toward accepting their own typing pattern.

So: **no automatic retraining, ever, from live traffic.** Instead:

1. On confirmed re-auth, the **client** (not the server — raw keystrokes
   never leave the client machine, per §0/§1) writes the session's raw
   keystroke data (in the same format `biometrics collect` produces) to
   `data/retrain_review/<timestamp>_<session_id>.csv`, plus a small
   sidecar `.json` with the triggering `impostor_score` and SHAP top
   features (received back from the server's Challenge response, §6), for a
   human reviewer's context.
2. A new CLI command, `biometrics review-retrain`, run on the client, lists
   pending files in that folder and lets a person accept/reject each one.
   Accepted files are moved into `data/raw/` (so they flow through the
   normal `process` → `train` pipeline next time those are run) or into a
   `data/retrain_review/rejected/` subfolder otherwise.
3. Nothing in `data/raw/` changes, and no retraining happens, without that
   explicit human step. `biometrics train` itself is unchanged by any of
   this — it only ever sees data that's already in `data/raw/`. The server
   never sees this data at all; it is entirely a client-local concern.

## 6. Explainability (SHAP)

### Resolved (architecture change): SHAP runs server-side, on a feature row sent from the client

`Inference.predict()` (client-side) already builds the exact feature row fed
to the model (`__prepare_live_data`). Since the server needs this to compute
SHAP but must never receive raw keystrokes, the client's `POST /predict`
payload includes the **already-engineered feature row** (609 numeric/OHE
columns, same shape as the model's training input) — not raw keystroke
timings. This is a deliberate, narrow exception to "the server never gets
biometric data": the engineered feature row is a derived, aggregated
statistical summary (dwell/flight-time means, rolling stats, burst counts),
not the raw per-keystroke sequence an attacker could replay or a court would
treat as directly identifying biometric data the way raw timing sequences
are. If this distinction turns out not to be acceptable for a given
deployment (stricter privacy regime, regulatory requirement), the fallback
is computing SHAP client-side instead and sending only the top-k result —
noted here as a documented tradeoff, not decided away.

Given that row, the server runs `shap.TreeExplainer(model)` — the server
needs a copy of the trained model artifacts for this, kept in sync with
whatever `thresholds.json`/model version the client is running (see the
open question at the end of this section) — whenever `impostor_score >=
tau_warn`, and keeps only the top-k (e.g. 5) contributing features,
translated to human-readable labels via a small lookup table (`seek_time` →
"typing rhythm between keys", `accuracy` → "typing accuracy", etc.) matching
the illustration in the pasted note. SHAP is only computed on-demand (not
every window) since it's the most expensive step in the loop — this keeps
the "Green AI" framing (the note's own point: XGBoost is cheap, so is SHAP
on a single row; running it on every 15-key window unconditionally would
not be). The resulting top-k features are included in the server's queued
`show_challenge`/`show_alert` command payload so the client's PySide6
window can display them (§4).

**Open question carried into implementation:** the server needs the same
model + `thresholds.json` the client trained, to run SHAP and to double-
check the client-reported thresholds are legitimate. Simplest v1 approach:
the client uploads its model directory (or the server is pointed at a
shared filesystem/artifact store) at registration time, keyed by `host_id`.
This needs to be nailed down when the server is actually implemented, but
doesn't change anything else in this design.

## 7. Client/server transport, notifier, and the shutdown hook

### Resolved (architecture change): the HTTP contract between client and server

Three endpoints, no auth in v1 (§2a):

```
POST /predict
  body: {host_id, timestamp, label, confidence, feature_row, thresholds}
  -> server runs PolicyEngine.handle_prediction (§3), returns {tier: "allow"}
     immediately (the actual Challenge/Alert command is delivered via long-poll,
     not in this response, so a slow/blocked client POST never blocks on a
     human responding to a popup)

GET /commands/<host_id>?wait=30
  -> long-poll, held open up to `wait` seconds. Returns immediately if a
     command is already queued for this host; otherwise blocks until one
     arrives or the timeout elapses (then returns {command: null} and the
     client immediately re-polls). This is the client-initiated delivery
     path from §2a for show_challenge / show_alert commands, including
     their SHAP top-k features (§6).

POST /events
  body: {host_id, event: "challenge_timeout"|"confirmed_user", ...}
  -> client reports back what happened in response to a delivered command
     (§4). Feeds back into PolicyEngine's per-host state (§3).
```

### Resolved: pluggable notifier, local JSONL for now

```python
# Sketch, not final code — server-side
class Notifier(Protocol):
    def notify(self, event: dict) -> None: ...

class LocalLogNotifier:
    """v1 default. Appends to data/agent_audit.jsonl. No network calls."""
    def notify(self, event: dict) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

class WebhookNotifier:
    """Not wired in by default. Same interface, so swapping the config
    value is the only change needed once a real endpoint exists."""
    def notify(self, event: dict) -> None:
        requests.post(self.url, json=event, timeout=5)
```

The server's `PolicyEngine` is constructed with a `Notifier` instance
(default `LocalLogNotifier`) and never imports a specific transport. Every
Allow/Challenge/Alert decision calls `notifier.notify(event)` where `event`
includes: timestamp, `host_id`, tier, `impostor_score`, and top-k SHAP
features. This is the same seam a future multi-server or upstream-alerting
setup would plug into — the interface doesn't need to change, only which
`Notifier` implementation is configured.

### Resolved (architecture change): a client that can't reach the server halts its own popups, not the other way around

The original draft's `remote_command` was written for a vague "agent calls
out to a supervisor" case. With the concrete client/server split, the
failure mode that actually matters is the **client's long-poll loop losing
its connection to the server** — if that happens, the client has no way to
know whether it's still being monitored, so it must not pretend everything
is fine:

```python
# Sketch, not final code — client-side long-poll loop
def poll_loop(self):
    consecutive_failures = 0
    while True:
        try:
            command = self.transport.get_command(self.host_id, wait=30)
            consecutive_failures = 0
        except RemoteAgentError as e:
            consecutive_failures += 1
            log.error("Long-poll to agent server failed (%d in a row): %s", consecutive_failures, e)
            if consecutive_failures >= 3:
                self._halted = True
                self._show_halted_notice()  # small, low-urgency local indicator only
            time.sleep(min(2 ** consecutive_failures, 60))
            continue

        if command:
            self.handle_command(command)
```

If the client is halted, it keeps retrying in the background but does not
fabricate a "you're clear" state — it shows a small local indicator that
monitoring is currently unreachable (distinct from, and much lower-urgency
than, a Challenge/Alert popup) rather than staying silent, since a security
agent that goes dark without saying so is worse than one that visibly says
"I can't currently reach the policy server." Predictions keep being computed
locally and queued/retried against `POST /predict` — nothing about local
XGBoost inference depends on the server being reachable, only the
Challenge/Alert delivery does.

On the server side, the equivalent failure is simpler: if a `host_id`
hasn't long-polled in an unusually long time, its queued commands just sit
there until it reconnects (or the operator notices via the audit log that a
host has gone quiet, which is itself worth surfacing operationally, though
that's future dashboard work, not v1).

### Resolved: shutdown hook is a logged no-op placeholder

```python
# Sketch, not final code — server-side, on PolicyEngine
def shutdown_hook(self, host_id: str, reason: str, dry_run: bool = True) -> None:
    """
    Placeholder for a future enforcement action (network suspension,
    process termination on the client, etc). v1 always calls this with
    dry_run=True. No real target is wired in - deployments that need real
    enforcement implement their own subclass/callback and pass
    dry_run=False explicitly, which is itself a separate, reviewed decision
    outside this design.
    """
    event = {"action": "shutdown_hook", "host_id": host_id, "reason": reason, "dry_run": dry_run}
    if dry_run:
        log.info("[NOOP] shutdown_hook called for %s: %s (dry_run=True, no action taken)", host_id, reason)
    else:
        raise NotImplementedError("No real shutdown target configured for this deployment")
    self.notifier.notify(event)
```

This is called from the **Alert** tier (§4) every time, always with
`dry_run=True` in v1. It exists purely to prove the policy layer has a
place to call into for a real enforcement action later — network
suspension, process kill, whatever a specific deployment needs — without
that decision being made, or that code being written, now. Because the
server already knows which `host_id` triggered it, this is also the natural
place a real future implementation would send a command down to that
specific client for local enforcement, reusing the same command-delivery
path as Challenge/Alert (§7) rather than needing a new mechanism.

## 8. Trust boundary

Per the §0 architecture change, the client/server split described
throughout this document — server always running, clients POST predictions
to it and long-poll for commands — **is the v1 architecture**, not a future
mode layered on top of a local-only default. The note's "even in the cloud"
framing is satisfied directly: the server only ever receives
`(timestamp, host_id, label, confidence, feature_row)` — never raw
keystrokes — which matters both for privacy and because keystroke dynamics
are themselves sensitive biometric data in most jurisdictions (GDPR Art.
9-adjacent). See §6 for the one deliberate, documented exception (the
engineered feature row, not raw keystrokes, travels to the server for SHAP).

The server can decide Challenge/Alert for any client, but per §4/§7 it can
only ever *deliver a command* the client already knows how to handle
(`show_challenge`, `show_alert`) — it cannot execute anything on the client
directly. A compromised or spoofed server could at most cause a client to
show a misleading popup or go into its own "can't reach server" halted
state (§7); it cannot, in this v1 design, lock a screen, kill a process, or
touch the network on any client, because no client-side handler exists for
that yet. Extending this to real enforcement, when that's decided, should
be scoped as its own reviewed change — not an incremental addition assumed
by this document.

## 9. Resolved decisions (was: open questions)

Two rounds of decisions are recorded here: the original five open questions
from the first draft, and the follow-up architecture questions from the
"always running in the background" requirement.

**Original five (§3-§7):**

1. **Thresholds** — not a fixed pair; computed from ROC/EER on the model's
   own held-out test set at train time, sent to the server as part of
   (or alongside) each prediction (§3).
2. **Challenge UI** — PySide6 native always-on-top window on the client,
   matching the existing `environment.yml` dependency (§4).
3. **Retrain feedback** — confirmed sessions go to a client-local
   human-reviewed queue (`data/retrain_review/`, new `biometrics
   review-retrain` command); nothing retrains automatically (§5).
4. **Notification target** — local JSONL log on the server by default,
   behind a `Notifier` interface so a real webhook can be swapped in later
   without touching decision logic (§7).
5. **Network suspension** — not implemented. Replaced with two safer seams:
   a `shutdown_hook(dry_run=True)` no-op placeholder on the server (§7),
   and a client-side halt (not a server-side one — see the architecture
   change below) that stops showing popups and surfaces an "unreachable"
   indicator if it loses its connection to the server, rather than
   silently degrading (§7).

**Follow-up architecture questions (§0, §2a):**

6. **Process model** — the agent is a persistent Flask/Django server, not
   an in-process callback or a foreground CLI command.
7. **Client/server split** — separate machines. The keylogger + XGBoost
   inference runs on each monitored client; the policy decision, SHAP, and
   `shutdown_hook` live on one central server.
8. **Idle detection** — out of scope; the agent only reacts to actual
   XGBoost predictions, not gaps in typing activity.
9. **Callback delivery** — client-initiated long-polling (§2a, §7), not the
   server opening a connection to each client, to avoid needing inbound
   network access to every monitored machine.
10. **Client/server auth** — none in v1; the deployment is expected to
    restrict network reachability itself (localhost/private LAN/VPN). Not
    safe to expose on an open network as-is (§2a).
