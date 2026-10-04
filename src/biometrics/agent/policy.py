"""
The agent's decision policy (Cognitive step). Pure logic, no I/O: it turns
each XGBoost window into a level, and the user's responses feed back into it.

Impostor score = the model's probability that the typist is NOT the enrolled
user. Levels, checked top-down:

  critical   The host is ARMED (an Alert was dismissed without re-authenticating)
             and this window is "Other" with score >= tau_warn. No run needed.
             Stays armed until a successful re-auth.
  alert      score >= tau_critical for ALERT_RUN consecutive windows,
             or a Challenge was ignored until it timed out.
  challenge  score >= tau_warn for CHALLENGE_RUN consecutive windows.
  allow      anything else.

A successful re-auth clears the recent-score window and disarms, so detection
starts over from allow.
"""
import enum
from collections import deque
from dataclasses import dataclass

CHALLENGE_RUN = 2
ALERT_RUN = 3


class Level(str, enum.Enum):
    ALLOW = "allow"
    CHALLENGE = "challenge"
    ALERT = "alert"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(Level).index(self)


# Why a level was reached -- shown on the dashboard next to each incident.
TRIGGER_SUSTAINED_WARN = "sustained_warn"          # CHALLENGE_RUN windows >= tau_warn
TRIGGER_SUSTAINED_CRITICAL = "sustained_critical"  # ALERT_RUN windows >= tau_critical
TRIGGER_CHALLENGE_IGNORED = "challenge_ignored"    # challenge timed out / closed without re-auth
TRIGGER_AFTER_DISMISSED_ALERT = "after_dismissed_alert"  # armed host crossed tau_warn again


@dataclass
class Decision:
    level: Level
    impostor_score: float
    trigger: str | None = None


class Policy:
    def __init__(self, tau_warn: float, tau_critical: float):
        self.tau_warn = tau_warn
        self.tau_critical = tau_critical
        self.recent: deque[float] = deque(maxlen=max(CHALLENGE_RUN, ALERT_RUN))
        self.armed = False

    @staticmethod
    def impostor_score(label: str, confidence: float) -> float:
        return confidence if label == "Other" else 1.0 - confidence

    def _run_at_or_above(self, threshold: float) -> int:
        count = 0
        for score in reversed(self.recent):
            if score < threshold:
                break
            count += 1
        return count

    def decide(self, label: str, confidence: float) -> Decision:
        score = self.impostor_score(label, confidence)
        self.recent.append(score)

        if self.armed and label == "Other" and score >= self.tau_warn:
            return Decision(Level.CRITICAL, score, TRIGGER_AFTER_DISMISSED_ALERT)
        if self._run_at_or_above(self.tau_critical) >= ALERT_RUN:
            return Decision(Level.ALERT, score, TRIGGER_SUSTAINED_CRITICAL)
        if self._run_at_or_above(self.tau_warn) >= CHALLENGE_RUN:
            return Decision(Level.CHALLENGE, score, TRIGGER_SUSTAINED_WARN)
        return Decision(Level.ALLOW, score)

    def challenge_ignored(self, score: float) -> Decision:
        return Decision(Level.ALERT, score, TRIGGER_CHALLENGE_IGNORED)

    def alert_dismissed(self) -> None:
        self.armed = True

    def reauthenticated(self) -> None:
        self.recent.clear()
        self.armed = False
