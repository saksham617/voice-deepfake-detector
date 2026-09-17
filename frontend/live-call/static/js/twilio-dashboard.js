// Live dashboard for auto-answered Twilio calls. Subscribes to /twilio/dashboard and renders
// a card per active call with a confidence meter that slides toward "real" or "AI-cloned".
// No accept/reject — cards appear on their own when a call hits the protected number.
(() => {
  const $ = (id) => document.getElementById(id);
  const FLAG_THRESHOLD = 0.5; // P(AI-cloned) at/above which a call is flagged. Tune here.

  const cards = new Map(); // call_id -> card element
  const pct = (x) => Math.round(x * 100);
  const fmtTime = (epoch) => new Date(epoch * 1000).toLocaleString();

  function ensureCard(callId, fromNumber) {
    let el = cards.get(callId);
    if (el) return el;
    $("emptyState").hidden = true;
    const frag = $("callCardTpl").content.cloneNode(true);
    el = frag.querySelector(".call-card");
    el.querySelector(".call-from").textContent = fromNumber || "Unknown caller";
    $("liveCalls").prepend(el);
    cards.set(callId, el);
    return el;
  }

  function onScore(msg) {
    const el = ensureCard(msg.call_id, msg.from);
    const p = msg.fake_prob; // P(AI-cloned), already smoothed by the risk engine
    const flagged = p >= FLAG_THRESHOLD;

    const fill = el.querySelector(".meter-fill");
    fill.style.width = `${pct(p)}%`;
    el.dataset.state = flagged ? "flagged" : "ok";

    const badge = el.querySelector(".call-badge");
    badge.textContent = flagged ? "⚠ Flagged" : "Likely genuine";

    el.querySelector(".call-verdict").textContent = flagged
      ? `Flagged: ${pct(p)}% confidence AI-generated voice`
      : `Likely genuine — ${pct(1 - p)}% confidence`;
  }

  function onEnded(msg) {
    const el = cards.get(msg.call_id);
    if (el) {
      el.dataset.state = msg.final_verdict === "fake" ? "flagged" : "ended";
      el.querySelector(".call-badge").textContent = "Call ended";
      // leave the final verdict text as-is; fade the card so live ones stand out
      el.classList.add("done");
      setTimeout(() => { el.remove(); cards.delete(msg.call_id); if (!cards.size) $("emptyState").hidden = false; }, 8000);
    }
    loadHistory();
  }

  function handle(msg) {
    if (msg.type === "call_started") ensureCard(msg.call_id, msg.from);
    else if (msg.type === "score") onScore(msg);
    else if (msg.type === "call_ended") onEnded(msg);
  }

  // -- call history ---------------------------------------------------------
  async function loadHistory() {
    try {
      const res = await fetch("/twilio/calls");
      const { calls } = await res.json();
      const body = $("historyBody");
      if (!calls.length) { body.innerHTML = '<tr><td colspan="4" class="hint">No calls yet.</td></tr>'; return; }
      body.innerHTML = calls.map((c) => {
        const verdict = c.final_verdict || (c.status === "active" ? "in progress" : "—");
        const conf = c.final_confidence != null ? `${pct(c.final_confidence)}%` : "—";
        const cls = verdict === "fake" ? "v-fake" : verdict === "real" ? "v-real" : "";
        return `<tr><td>${c.from_number || "Unknown"}</td><td>${fmtTime(c.started_at)}</td>` +
               `<td class="${cls}">${verdict}</td><td>${conf}</td></tr>`;
      }).join("");
    } catch (e) { /* history is best-effort */ }
  }

  // -- websocket with auto-reconnect ---------------------------------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/twilio/dashboard`);
    ws.onopen = () => { $("conn").textContent = "● live"; $("conn").className = "conn on"; };
    ws.onclose = () => {
      $("conn").textContent = "● reconnecting…"; $("conn").className = "conn off";
      setTimeout(connect, 2000);
    };
    ws.onmessage = (ev) => { try { handle(JSON.parse(ev.data)); } catch {} };
  }

  connect();
  loadHistory();
})();
