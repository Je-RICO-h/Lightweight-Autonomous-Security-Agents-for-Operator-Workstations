"""
SQLite storage for the dashboard: connected hosts (each with its one model),
an append-only event log (connects, incidents, user responses, notes), and
the queue of commands waiting for an agent (lockout, refine).
One short-lived connection per call, so it's safe from Flask's request threads.
"""
import json
import os
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    host_id     TEXT PRIMARY KEY,
    user        TEXT,
    hostname    TEXT,
    platform    TEXT,
    model_hash  TEXT,
    model       TEXT DEFAULT '{}',
    first_seen  REAL,
    last_seen   REAL,
    connected   INTEGER DEFAULT 0,
    status      TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    time        REAL,
    host_id     TEXT,
    kind        TEXT,
    level       TEXT,
    incident_id TEXT,
    data        TEXT
);
CREATE INDEX IF NOT EXISTS events_host ON events (host_id, id);
CREATE INDEX IF NOT EXISTS events_incident ON events (incident_id);
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT
);
CREATE TABLE IF NOT EXISTS commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id     TEXT,
    type        TEXT,
    created     REAL,
    delivered   REAL
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)
            db.execute("PRAGMA journal_mode=WAL")
            # databases from before "one model per host" kept a separate models table
            if "model" not in {r["name"] for r in db.execute("PRAGMA table_info(hosts)")}:
                db.execute("ALTER TABLE hosts ADD COLUMN model TEXT DEFAULT '{}'")
            db.execute("DROP TABLE IF EXISTS models")
            db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('instance', lower(hex(randomblob(8))))")
            self.instance = db.execute("SELECT value FROM meta WHERE key = 'instance'").fetchone()[0]

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    # ---- writes -------------------------------------------------------------

    def hello(self, host: dict, model: dict) -> None:
        now = time.time()
        with self._db() as db:
            previous = db.execute("SELECT model_hash FROM hosts WHERE host_id = ?", (host["host_id"],)).fetchone()
            db.execute("""
                INSERT INTO hosts (host_id, user, hostname, platform, model_hash, model, first_seen, last_seen, connected)
                VALUES (:host_id, :user, :hostname, :platform, :model_hash, :model, :now, :now, 1)
                ON CONFLICT (host_id) DO UPDATE SET user = :user, hostname = :hostname, platform = :platform,
                    model_hash = :model_hash, model = :model, last_seen = :now, connected = 1
            """, {**host, "model_hash": model["hash"], "model": json.dumps(model), "now": now})
            model_changed = bool(previous and previous["model_hash"] != model["hash"])
            self._insert(db, now, host["host_id"], "connected",
                         data={"user": host["user"], "model_hash": model["hash"], "model_changed": model_changed})

    def report(self, host_id: str, status: dict, events: list[dict]) -> list[dict] | None:
        """Stores the heartbeat + events and hands back undelivered commands.
        None if the host never said hello (e.g. the server was started with a fresh DB)."""
        now = time.time()
        with self._db() as db:
            updated = db.execute("UPDATE hosts SET last_seen = ?, connected = 1, status = ? WHERE host_id = ?",
                                 (now, json.dumps(status), host_id)).rowcount
            if not updated:
                return None
            for e in events:
                self._insert(db, e["time"], host_id, e["kind"], e.get("level"), e.get("incident_id"), e)
            commands = [dict(r) for r in db.execute(
                "SELECT id, type FROM commands WHERE host_id = ? AND delivered IS NULL ORDER BY id", (host_id,))]
            if commands:
                db.execute("UPDATE commands SET delivered = ? WHERE host_id = ? AND delivered IS NULL", (now, host_id))
        return commands

    def bye(self, host_id: str) -> None:
        with self._db() as db:
            if db.execute("UPDATE hosts SET connected = 0 WHERE host_id = ?", (host_id,)).rowcount:
                self._insert(db, time.time(), host_id, "disconnected")

    def command(self, host_id: str, kind: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO commands (host_id, type, created) VALUES (?, ?, ?)", (host_id, kind, time.time()))
            self._insert(db, time.time(), host_id, "command", data={"type": kind})

    @staticmethod
    def _insert(db, t, host_id, kind, level=None, incident_id=None, data=None):
        db.execute("INSERT INTO events (time, host_id, kind, level, incident_id, data) VALUES (?, ?, ?, ?, ?, ?)",
                   (t, host_id, kind, level, incident_id, json.dumps(data or {})))

    # ---- reads --------------------------------------------------------------

    def hosts(self) -> list[dict]:
        day_ago = time.time() - 86400
        with self._db() as db:
            rows = db.execute("""
                SELECT h.*,
                  (SELECT COUNT(*) FROM events e WHERE e.host_id = h.host_id AND e.kind = 'incident'
                       AND e.time > :day) AS incidents_24h,
                  (SELECT COUNT(*) FROM events e WHERE e.host_id = h.host_id AND e.kind = 'incident'
                       AND e.level = 'critical' AND e.time > :day) AS critical_24h,
                  (SELECT MAX(time) FROM events e WHERE e.host_id = h.host_id AND e.kind = 'incident') AS last_incident,
                  (SELECT COUNT(*) FROM commands c WHERE c.host_id = h.host_id AND c.delivered IS NULL) AS queued
                FROM hosts h ORDER BY h.host_id
            """, {"day": day_ago}).fetchall()
        return [self._host(r) for r in rows]

    def host(self, host_id: str) -> dict | None:
        return next((h for h in self.hosts() if h["host_id"] == host_id), None)

    @staticmethod
    def _host(row) -> dict:
        host = dict(row)
        host["status"] = json.loads(host["status"] or "{}")
        host["model"] = json.loads(host["model"] or "{}")
        return host

    def incidents(self, host_id: str = None, limit: int = 200) -> list[dict]:
        """Newest first, each with the user's response and agent notes attached."""
        where, params = ("AND host_id = ?", [host_id]) if host_id else ("", [])
        with self._db() as db:
            rows = db.execute(f"SELECT * FROM events WHERE kind = 'incident' {where} ORDER BY id DESC LIMIT ?",
                              (*params, limit)).fetchall()
            incidents = [self._event(r) for r in rows]
            self._attach_related(db, incidents)
        return incidents

    def incident(self, incident_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM events WHERE kind = 'incident' AND incident_id = ?",
                             (incident_id,)).fetchone()
            if row is None:
                return None
            incident = self._event(row)
            self._attach_related(db, [incident])
        return incident

    def _attach_related(self, db, incidents: list[dict]) -> None:
        ids = [i["incident_id"] for i in incidents if i["incident_id"]]
        related = {}
        if ids:
            marks = ",".join("?" * len(ids))
            for r in db.execute(f"SELECT * FROM events WHERE kind IN ('response', 'note') "
                                f"AND incident_id IN ({marks}) ORDER BY id", ids):
                e = self._event(r)
                related.setdefault(e["incident_id"], []).append(e)
        for incident in incidents:
            events = related.get(incident["incident_id"], [])
            incident["response"] = next((e for e in events if e["kind"] == "response"), None)
            incident["notes"] = [e for e in events if e["kind"] == "note"]

    def events(self, host_id: str = None, kind: str = None, before: int = None, limit: int = 100) -> list[dict]:
        where, params = [], []
        if host_id:
            where.append("host_id = ?")
            params.append(host_id)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if before:
            where.append("id < ?")
            params.append(before)
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "")
        with self._db() as db:
            rows = db.execute(sql + " ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
        return [self._event(r) for r in rows]

    def incidents_after(self, after_id: int) -> tuple[int, list[dict]]:
        """(latest event id, incidents newer than after_id) -- for the dashboard's live toasts."""
        with self._db() as db:
            latest = db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
            rows = db.execute("SELECT * FROM events WHERE kind = 'incident' AND id > ? ORDER BY id LIMIT 20",
                              (after_id,)).fetchall()
        return latest, [self._event(r) for r in rows]

    @staticmethod
    def _event(row) -> dict:
        event = dict(row)
        event["data"] = json.loads(event["data"] or "{}")
        return event

    def summary(self) -> dict:
        day_ago = time.time() - 86400
        with self._db() as db:
            counts = db.execute("""
                SELECT COUNT(*) AS incidents,
                       SUM(level = 'critical') AS critical,
                       SUM(incident_id NOT IN (SELECT incident_id FROM events
                                               WHERE kind = 'response' AND incident_id IS NOT NULL)) AS unanswered
                FROM events WHERE kind = 'incident' AND time > ?
            """, (day_ago,)).fetchone()
        return {k: counts[k] or 0 for k in ("incidents", "critical", "unanswered")}
