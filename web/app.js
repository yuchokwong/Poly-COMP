"use strict";

const PAGE_SIZE = 12;
const MAX_RESULTS = 100;

const el = {
  form: document.getElementById("search-form"),
  query: document.getElementById("query"),
  button: document.getElementById("search-btn"),
  topk: document.getElementById("topk"),
  topkValue: document.getElementById("topk-value"),
  hybrid: document.getElementById("hybrid"),
  split: document.getElementById("split"),
  minscore: document.getElementById("minscore"),
  shareBtn: document.getElementById("share-btn"),
  status: document.getElementById("status"),
  results: document.getElementById("results"),
  galleryHead: document.getElementById("gallery-head"),
  resultMeta: document.getElementById("result-meta"),
  stats: document.getElementById("stats"),
  loadMore: document.getElementById("load-more"),
  loadMoreBtn: document.getElementById("load-more-btn"),
  lightbox: document.getElementById("lightbox"),
  lightboxImg: document.getElementById("lightbox-img"),
  lightboxInfo: document.getElementById("lightbox-info"),
  lightboxClose: document.getElementById("lightbox-close"),
  settingsBtn: document.getElementById("settings-btn"),
  settingsModal: document.getElementById("settings-modal"),
  settingsClose: document.getElementById("settings-close"),
  settingsFields: document.getElementById("settings-fields"),
  settingsSave: document.getElementById("settings-save"),
  settingsReload: document.getElementById("settings-reload"),
  settingsMsg: document.getElementById("settings-msg"),
};

const state = { maxScore: 1, stats: null, k: PAGE_SIZE, data: null };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function setStatus(message, kind) {
  if (!message) {
    el.status.classList.add("hidden");
    el.status.innerHTML = "";
    return;
  }
  el.status.className = "status" + (kind ? ` ${kind}` : "");
  el.status.innerHTML = message;
  el.status.classList.remove("hidden");
}

async function loadStats() {
  try {
    const resp = await fetch("/api/stats");
    const data = await resp.json();
    state.stats = data;
    const ready = data.ready;
    el.stats.innerHTML =
      `<span class="dot ${ready ? "" : "bad"}">●</span> ` +
      `${ready ? "ready" : "not indexed"}` +
      `<span class="pill">${data.num_images.toLocaleString()} images</span>`;
    if (!ready) {
      setStatus(
        "The gallery index is not built yet. Run <code>python -m tsr build</code> " +
        "then restart the server.",
        "warn"
      );
    }
  } catch (err) {
    el.stats.innerHTML = `<span class="dot bad">●</span> offline`;
    setStatus(`Cannot reach the API: ${escapeHtml(err.message)}`, "error");
  }
}

function urlState() {
  const p = new URLSearchParams(location.search);
  return {
    q: p.get("q") || "",
    k: Math.min(MAX_RESULTS, Math.max(PAGE_SIZE, Number(p.get("k")) || PAGE_SIZE)),
    hybrid: p.get("hybrid") !== "false",
    split: p.get("split") || "",
    minscore: Number(p.get("minscore")) || 0,
  };
}

function applyUrlState() {
  const s = urlState();
  el.query.value = s.q;
  el.topk.value = s.k;
  el.topkValue.textContent = s.k;
  el.hybrid.checked = s.hybrid;
  el.split.value = s.split;
  el.minscore.value = s.minscore || "";
  state.k = s.k;
}

function syncUrl(push) {
  const p = new URLSearchParams();
  if (el.query.value.trim()) p.set("q", el.query.value.trim());
  p.set("k", String(state.k));
  p.set("hybrid", String(el.hybrid.checked));
  if (el.split.value) p.set("split", el.split.value);
  if (el.minscore.value) p.set("minscore", el.minscore.value);
  const url = `${location.pathname}?${p.toString()}`;
  history[push ? "pushState" : "replaceState"]({}, "", url);
}

async function runSearch(push) {
  const query = el.query.value.trim();
  if (!query) {
    el.query.focus();
    return;
  }
  state.k = Math.max(PAGE_SIZE, Number(el.topk.value) || PAGE_SIZE);
  const hybrid = el.hybrid.checked;
  const urlK = Math.max(state.k, Number(new URLSearchParams(location.search).get("k")) || 0);

  el.button.disabled = true;
  el.button.textContent = "Searching…";
  setStatus("Searching the gallery…");

  try {
    const params = new URLSearchParams({ q: query, k: String(Math.min(MAX_RESULTS, urlK)), hybrid: String(hybrid) });
    const resp = await fetch(`/api/search?${params.toString()}`);
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    state.data = data;
    syncUrl(push);
    renderResults(data);
  } catch (err) {
    setStatus(`Search failed: ${escapeHtml(err.message)}`, "error");
  } finally {
    el.button.disabled = false;
    el.button.textContent = "Search";
  }
}

function filtered(items) {
  const split = el.split.value;
  const min = Number(el.minscore.value) || 0;
  return items.filter((r) => (!split || r.split === split) && r.score >= min);
}

function renderResults(data) {
  el.results.innerHTML = "";
  const items = filtered(data.results);
  state.maxScore = Math.max(...items.map((r) => r.score), 1e-6);

  if (!items.length) {
    setStatus(
      data.results.length
        ? "No results match the current filters."
        : "No results. Try a different query.",
      "warn"
    );
    el.galleryHead.classList.add("hidden");
    el.loadMore.classList.add("hidden");
    return;
  }

  setStatus("");
  el.galleryHead.classList.remove("hidden");
  const filterNote = items.length !== data.results.length ? ` (${items.length} shown)` : "";
  el.resultMeta.textContent =
    `${items.length} results for “${data.query}” · ${data.latency_ms} ms` +
    (data.hybrid ? " · hybrid" : "") + filterNote;

  const frag = document.createDocumentFragment();
  items.forEach((item) => frag.appendChild(buildCard(item)));
  el.results.appendChild(frag);

  const canLoadMore = data.count >= state.k && state.k < MAX_RESULTS;
  el.loadMore.classList.toggle("hidden", !canLoadMore);
}

function buildCard(item) {
  const card = document.createElement("article");
  card.className = "card";

  const pct = Math.max(0.02, item.score / state.maxScore);
  const caption = item.captions && item.captions.length ? item.captions[0] : "(no caption)";

  card.innerHTML = `
    <span class="rank">#${item.rank}</span>
    <span class="score">${item.score.toFixed(3)}</span>
    <img class="thumb" loading="lazy" src="/api/thumbnail/${item.id}?size=480" alt="${escapeHtml(caption)}" />
    <div class="card-body">
      <p class="caption">${escapeHtml(caption)}</p>
      <div class="meta">
        ${item.split ? `<span class="tag">${escapeHtml(item.split)}</span>` : ""}
        ${item.category ? `<span class="tag">${escapeHtml(item.category)}</span>` : ""}
        <span class="tag">${escapeHtml(item.image_id)}</span>
      </div>
      <div class="bar"><span style="width:${(pct * 100).toFixed(1)}%"></span></div>
    </div>`;

  card.addEventListener("click", () => openLightbox(item));
  return card;
}

function openLightbox(item) {
  el.lightboxImg.src = item.url;
  const lines = [
    `rank #${item.rank} · score ${item.score.toFixed(4)}`,
    item.image_id,
  ];
  if (item.captions && item.captions.length) {
    lines.push(...item.captions.slice(0, 3).map((c) => `“${escapeHtml(c)}”`));
  }
  el.lightboxInfo.innerHTML =
    lines.join("<br/>") +
    `<br/><button type="button" class="chip" id="similar-btn">Find similar images</button>`;
  const btn = document.getElementById("similar-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      closeLightbox();
      runSimilar(item.id);
    });
  }
  el.lightbox.classList.remove("hidden");
}

async function runSimilar(pk) {
  el.button.disabled = true;
  setStatus("Finding similar images…");
  try {
    const k = Math.max(state.k, PAGE_SIZE);
    const resp = await fetch(`/api/similar/${pk}?k=${Math.min(MAX_RESULTS, k)}`);
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    state.data = data;
    renderResults(data);
  } catch (err) {
    setStatus(`Similar search failed: ${escapeHtml(err.message)}`, "error");
  } finally {
    el.button.disabled = false;
  }
}

function closeLightbox() {
  el.lightbox.classList.add("hidden");
  el.lightboxImg.src = "";
}

async function openSettings() {
  el.settingsMsg.textContent = "Loading…";
  el.settingsModal.classList.remove("hidden");
  await loadSettings();
}

function closeSettings() {
  el.settingsModal.classList.add("hidden");
}

async function loadSettings() {
  try {
    const resp = await fetch("/api/settings");
    const data = await resp.json();
    renderSettings(data.fields || []);
    el.settingsMsg.textContent = "";
  } catch (err) {
    el.settingsMsg.textContent = `Load failed: ${err.message}`;
  }
}

function renderSettings(fields) {
  el.settingsFields.innerHTML = "";
  const groups = {};
  fields.forEach((f) => {
    (groups[f.group] = groups[f.group] || []).push(f);
  });
  Object.entries(groups).forEach(([group, items]) => {
    const section = document.createElement("section");
    section.className = "settings-group";
    const heading = document.createElement("h3");
    heading.textContent = group;
    section.appendChild(heading);
    items.forEach((f) => section.appendChild(buildSettingRow(f)));
    el.settingsFields.appendChild(section);
  });
}

function buildSettingRow(field) {
  const row = document.createElement("label");
  row.className = "setting-row" + (field.editable ? "" : " readonly");
  const name = document.createElement("span");
  name.className = "setting-label";
  name.textContent = field.label || field.key;
  if (field.help) name.title = field.help;
  row.appendChild(name);

  let control;
  if (field.type === "bool") {
    control = document.createElement("input");
    control.type = "checkbox";
    control.checked = !!field.value;
  } else if (field.type === "enum") {
    control = document.createElement("select");
    (field.choices || []).forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      control.appendChild(opt);
    });
    control.value = field.value;
  } else if (field.type === "list") {
    control = document.createElement("textarea");
    control.rows = 4;
    control.value = Array.isArray(field.value) ? field.value.join("\n") : (field.value || "");
  } else if (field.type === "int" || field.type === "float") {
    control = document.createElement("input");
    control.type = "number";
    if (field.min != null) control.min = field.min;
    if (field.max != null) control.max = field.max;
    control.step = field.type === "int" ? "1" : (field.step || 0.05);
    control.value = field.value ?? "";
  } else {
    control = document.createElement("input");
    control.type = "text";
    control.value = field.value ?? "";
  }
  control.dataset.key = field.key;
  control.dataset.type = field.type;
  if (!field.editable) control.disabled = true;
  row.appendChild(control);
  return row;
}

function collectSettings() {
  const out = {};
  el.settingsFields.querySelectorAll("[data-key]").forEach((c) => {
    if (c.disabled) return;
    const type = c.dataset.type;
    if (type === "bool") out[c.dataset.key] = c.checked;
    else if (type === "list") out[c.dataset.key] = c.value;
    else if (type === "int") out[c.dataset.key] = parseInt(c.value, 10);
    else if (type === "float") out[c.dataset.key] = parseFloat(c.value);
    else out[c.dataset.key] = c.value;
  });
  return out;
}

async function saveSettings() {
  el.settingsMsg.textContent = "Saving…";
  try {
    const resp = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectSettings()),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    renderSettings(data.fields || []);
    el.settingsMsg.textContent = "Saved.";
  } catch (err) {
    el.settingsMsg.textContent = `Save failed: ${escapeHtml(err.message)}`;
  }
}

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  runSearch(true);
});
el.topk.addEventListener("input", () => {
  el.topkValue.textContent = el.topk.value;
});
el.split.addEventListener("change", () => {
  if (state.data) renderResults(state.data);
  syncUrl(false);
});
el.minscore.addEventListener("input", () => {
  if (state.data) renderResults(state.data);
  syncUrl(false);
});
el.hybrid.addEventListener("change", () => {
  if (el.query.value.trim()) runSearch(true);
});
el.shareBtn.addEventListener("click", async () => {
  syncUrl(false);
  const url = location.href;
  try {
    await navigator.clipboard.writeText(url);
    setStatus("Link copied to clipboard.", "warn");
    setTimeout(() => setStatus(""), 1500);
  } catch (err) {
    setStatus(`Copy this link: ${escapeHtml(url)}`);
  }
});
el.loadMoreBtn.addEventListener("click", () => {
  state.k = Math.min(MAX_RESULTS, state.k + PAGE_SIZE);
  el.topk.value = state.k;
  el.topkValue.textContent = state.k;
  runSearch(false);
});
el.lightboxClose.addEventListener("click", closeLightbox);
el.lightbox.addEventListener("click", (e) => {
  if (e.target === el.lightbox) closeLightbox();
});
el.settingsBtn.addEventListener("click", openSettings);
el.settingsClose.addEventListener("click", closeSettings);
el.settingsReload.addEventListener("click", loadSettings);
el.settingsSave.addEventListener("click", saveSettings);
el.settingsModal.addEventListener("click", (e) => {
  if (e.target === el.settingsModal) closeSettings();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeLightbox();
    closeSettings();
  }
});

loadStats();
applyUrlState();
if (el.query.value.trim()) {
  runSearch(false);
} else {
  el.query.focus();
}
