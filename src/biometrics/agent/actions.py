"""
WHAT THE AGENT DOES AT EACH LEVEL -- edit this file freely.

The policy (policy.py) decides the level; these functions decide the
response. Each one receives:

  incident  what happened. Fields:
              incident.level           "challenge" | "alert" | "critical"
              incident.impostor_score  0..1, the model's confidence it's NOT the user
              incident.trigger         why this level was reached (see policy.py)
              incident.reasons         top features that looked wrong, strongest first:
                                       {label, value, typical, direction, impact, ...}
              incident.id, incident.time

  agent     what the agent can do:
              agent.show_challenge(incident)  popup: "re-authenticate", auto-escalates to
                                              on_alert after 30 s without an answer
              agent.show_alert(incident)      popup: informational; closing it without
                                              re-authenticating ARMS the host, so the next
                                              suspicious window goes straight to on_critical
              agent.show_lockout(incident)    popup that only closes after re-authentication
              agent.shutdown_hook(incident)   enforcement placeholder -- logs only (dry run)
              agent.note(text)                adds a line to this host's log on the dashboard

Every incident and every user response is reported to the dashboard server
automatically -- nothing here needs to (or can) skip that.

An exception raised here is logged and reported to the dashboard; it never
stops the keylogger.
"""


def on_challenge(incident, agent):
    """Suspicious: impostor score >= tau_warn for 2 windows in a row."""
    agent.show_challenge(incident)


def on_alert(incident, agent):
    """High-confidence mismatch: score >= tau_critical for 3 windows in a row,
    or a challenge was left unanswered."""
    agent.show_alert(incident)


def on_critical(incident, agent):
    """An alert was dismissed without re-authenticating, and the typing is
    still suspicious."""
    agent.show_lockout(incident)
    agent.shutdown_hook(incident)
