// PlacidWay chatbot frontend
(function () {
  const messagesEl = document.getElementById("messages");
  const form = document.getElementById("chatForm");
  const input = document.getElementById("question");
  const statusText = document.getElementById("statusText");
  const statusDot = document.getElementById("statusDot");
  const refreshBtn = document.getElementById("refreshBtn");

  const sessionId =
    "pw_" + Math.random().toString(36).slice(2, 10);

  function setStatus(text, ok) {
    statusText.textContent = text;
    statusDot.className = "dot" + (ok ? " ok" : "");
  }

  function scrollToEnd() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMsg(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.textContent = text;
    messagesEl.appendChild(div);
    scrollToEnd();
    return div;
  }

  // Sources now render inside the bot's bubble as a collapsible <details>.
  function addSources(parent, sources) {
    if (!sources || !sources.length) return;
    const wrap = document.createElement("details");
    wrap.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = "Sources (" + sources.length + ")";
    wrap.appendChild(summary);
    const ul = document.createElement("ul");
    sources.forEach((s) => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = s.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = s.title || s.url;
      li.appendChild(a);
      ul.appendChild(li);
    });
    wrap.appendChild(ul);
    parent.appendChild(wrap);
    scrollToEnd();
  }

  async function health() {
    try {
      const r = await fetch("/api/health");
      const j = await r.json();
      setStatus(
        j.status === "ok" ? "Online" : "Degraded",
        j.status === "ok"
      );
    } catch (e) {
      setStatus("Offline", false);
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    addMsg("user", question);
    input.value = "";

    const botMsg = addMsg("bot", "…");
    botMsg.classList.add("typing");

    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session_id: sessionId }),
      });
      const j = await r.json();

      botMsg.classList.remove("typing");

      if (j.error) {
        botMsg.textContent = j.error;
        botMsg.classList.add("error-msg");
        return;
      }
      botMsg.textContent = j.answer || "(no answer)";
      addSources(botMsg, j.sources);
    } catch (err) {
      botMsg.classList.remove("typing");
      botMsg.textContent = "Sorry, something went wrong. Please try again.";
      botMsg.classList.add("error-msg");
    }
  });

  refreshBtn.addEventListener("click", async () => {
    const original = refreshBtn.textContent;
    refreshBtn.textContent = "Refreshing…";
    refreshBtn.disabled = true;
    try {
      const r = await fetch("/api/refresh", { method: "POST" });
      const j = await r.json();
      setStatus(j.status === "started" ? "Refreshing" : "Degraded", j.status === "started");
      addMsg("bot", "Knowledge base refresh started. It re-scrapes all pages and rebuilds the index — this takes about a minute.");
    } catch (err) {
      addMsg("bot", "Refresh failed to start.");
    } finally {
      refreshBtn.textContent = original;
      refreshBtn.disabled = false;
    }
  });

  health();
  // poll health every 20s
  setInterval(health, 20000);
})();