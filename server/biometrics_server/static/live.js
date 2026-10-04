// Live dashboard: re-renders <main> every --refresh seconds (keeping open incident panels
// open), shows a toast for every new alert/critical incident, and flags new
// incidents in the nav and on the page until the Incidents page is visited.
(() => {
  const body = document.body;
  const PAGE_MS = Number(body.dataset.refreshMs) || 1000, FEED_MS = PAGE_MS;
  const store = {
    get(k) { try { return Number(localStorage.getItem(k)) || 0; } catch { return 0; } },
    raw(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, String(v)); } catch { /* private mode */ } },
  };
  let newIncidents = [];  // incidents newer than the last Incidents-page visit

  let busy = false;  // at short intervals, never stack requests behind a slow one
  async function refreshPage() {
    if (busy || body.dataset.live !== "1" || document.hidden) return;
    const active = document.activeElement;
    if (active && ["SELECT", "INPUT", "TEXTAREA", "BUTTON"].includes(active.tagName)) return;
    busy = true;
    try {
      const resp = await fetch(location.href, { cache: "no-store" });
      if (!resp.ok) return;
      const next = new DOMParser().parseFromString(await resp.text(), "text/html").querySelector("main");
      if (!next) return;
      const main = document.querySelector("main");
      main.querySelectorAll("details[id]").forEach(d => {
        const twin = next.querySelector(`details[id="${d.id}"]`);
        if (twin) twin.open = d.open;
      });
      main.innerHTML = next.innerHTML;
      flagPage();
    } catch { /* server restarting -- try again next tick */ }
    finally { busy = false; }
  }

  function flagPage() {
    const ids = new Set(newIncidents.map(i => String(i.id)));
    const hosts = new Set(newIncidents.map(i => i.host_id));
    document.querySelectorAll("[data-event-id]").forEach(el => el.classList.toggle("is-new", ids.has(el.dataset.eventId)));
    document.querySelectorAll("[data-host]").forEach(el => el.classList.toggle("is-new", hosts.has(el.dataset.host)));
    const flag = document.getElementById("new-incidents");
    flag.hidden = newIncidents.length === 0;
    flag.textContent = newIncidents.length >= 20 ? "20+" : newIncidents.length;
  }

  function toast(i) {
    const el = document.createElement("a");
    el.className = `toast ${i.level}`;
    el.href = i.url;
    const part = (tag, text, cls) => {
      const node = document.createElement(tag);
      node.textContent = text;
      if (cls) node.className = cls;
      return el.appendChild(node);
    };
    part("button", "×", "toast-x").addEventListener("click", e => { e.preventDefault(); el.remove(); });
    part("strong", `${i.level.toUpperCase()} on ${i.host_id}`);
    part("span", `Impostor score ${i.score == null ? "–" : Math.round(i.score * 100) + "%"} · ${i.time}`);
    part("span", i.trigger, "toast-why");
    document.getElementById("toasts").prepend(el);
    if (i.level !== "critical") setTimeout(() => el.remove(), 12000);  // critical stays until dismissed
  }

  async function pollFeed() {
    try {
      let seen = store.get("incidentsSeen");
      let toasted = store.get("incidentsToasted");
      let feed = await (await fetch(`${body.dataset.liveUrl}?after=${seen}`, { cache: "no-store" })).json();
      if (store.raw("incidentsDb") !== feed.db) {
        // the dashboard was reset: counters from the old database mean nothing now
        seen = toasted = 0;
        store.set("incidentsDb", feed.db);
        feed = await (await fetch(`${body.dataset.liveUrl}?after=0`, { cache: "no-store" })).json();
      }
      if (!toasted || body.dataset.incidentsPage === "1") {
        // first visit ever, or looking at the incidents right now: everything so far counts as seen
        if (!toasted) toasted = feed.latest;
        if (body.dataset.incidentsPage === "1") seen = feed.latest;
        store.set("incidentsSeen", seen);
      }
      newIncidents = feed.incidents.filter(i => i.id > seen);
      feed.incidents.filter(i => i.id > toasted && i.level !== "challenge").forEach(toast);
      store.set("incidentsToasted", Math.max(toasted, feed.latest));
      flagPage();
    } catch { /* ignore */ }
  }

  pollFeed();
  setInterval(pollFeed, FEED_MS);
  setInterval(refreshPage, PAGE_MS);
})();
