"use strict";
const $ = (id) => document.getElementById(id);
let apiKey = sessionStorage.getItem("assistant-key") || "";
let currentDraft = "";

function notice(message, success = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("success", success);
  $("notice").hidden = !message;
}

async function api(path, options = {}) {
  if (!apiKey) throw new Error("Введите ключ доступа в левой панели.");
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", "Bearer " + apiKey);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(path, {...options, headers});
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error?.message || "Сервис временно недоступен.");
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

function setTab(tab) {
  $("assistant-tab").hidden = tab !== "assistant";
  $("knowledge-tab").hidden = tab !== "knowledge";
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
    if (button.dataset.tab === tab) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  notice("");
  if (tab === "knowledge" && apiKey) loadDocuments().catch((e) => notice(e.message));
}
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => setTab(button.dataset.tab)));

async function connect() {
  try {
    const state = await api("/health/ready");
    $("service-status").textContent = state.generation_mode === "extractive" ? "Режим выдержек · без LLM" : "Готов к работе";
    $("status-dot").className = "dot ready";
  } catch (error) {
    $("status-dot").className = "dot error";
    if (error.status !== 503) throw error;
    $("service-status").textContent = "Модель пока недоступна";
    notice("Ключ принят. Сервис модели ещё не готов к работе. Проверьте запуск и загрузку моделей.");
  }
  sessionStorage.setItem("assistant-key", apiKey);
  $("api-key").value = "";
  $("access-panel").open = false;
  $("logout").hidden = false;
  await loadDocuments();
}
$("access-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  apiKey = $("api-key").value.trim();
  notice("");
  try { await connect(); } catch (error) {
    notice(error.message);
    $("service-status").textContent = "Не удалось подключиться";
    sessionStorage.removeItem("assistant-key");
  }
});
$("logout").addEventListener("click", () => {
  sessionStorage.removeItem("assistant-key");
  location.reload();
});

$("example").addEventListener("click", () => {
  $("question").value = "Что уточнить для нестандартной комплектации и можно ли обещать срок поставки? Подготовь ответ клиенту.";
  $("lead-reference").value = "DEMO-42";
  $("lead-message").value = "Нужны 12 комплектов оборудования с нестандартными разъёмами. Сможете поставить на следующей неделе?";
  $("products").value = "Комплект оборудования";
  document.querySelector(".lead-details").open = true;
  notice("Это учебная заявка. Для ответа загрузите документы из демонстрационной базы знаний.", true);
});

function listBlock(id, title, items) {
  const target = $(id);
  target.replaceChildren();
  if (!items.length) return;
  target.className = "result-list";
  const heading = document.createElement("h3"); heading.textContent = title;
  const list = document.createElement("ul");
  items.forEach((text) => { const item = document.createElement("li"); item.textContent = text; list.append(item); });
  target.append(heading, list);
}

function renderDraftText(text, sources) {
  const target = $("draft-text"); target.replaceChildren();
  const ids = new Set(sources.map((s) => s.id));
  text.split(/(\[S\d+\])/g).forEach((part) => {
    const id = part.slice(1, -1);
    if (/^\[S\d+\]$/.test(part) && ids.has(id)) {
      const link = document.createElement("a"); link.href = "#source-" + id; link.textContent = part;
      target.append(link);
    } else target.append(document.createTextNode(part));
  });
}

function renderSources(sources) {
  $("sources").replaceChildren();
  $("sources-section").hidden = sources.length === 0;
  $("source-count").textContent = "Фрагментов: " + sources.length;
  sources.forEach((source) => {
    const card = document.createElement("article"); card.className = "source-card"; card.id = "source-" + source.id;
    const heading = document.createElement("h3"); heading.className = "source-title"; heading.textContent = `[${source.id}] ${source.title}`;
    const meta = document.createElement("div"); meta.className = "source-meta"; meta.textContent = `Версия ${source.version}${source.page ? " · Страница " + source.page : ""} · ${source.source_key}`;
    const details = document.createElement("details");
    const summary = document.createElement("summary"); summary.textContent = "Посмотреть фрагмент";
    const text = document.createElement("pre"); text.textContent = source.text;
    details.append(summary, text); card.append(heading, meta, details);
    if (source.source_url && /^https?:\/\//.test(source.source_url)) {
      const link = document.createElement("a"); link.href = source.source_url; link.textContent = "Открыть источник ↗";
      link.target = "_blank"; link.rel = "noopener noreferrer"; card.append(link);
    }
    $("sources").append(card);
  });
}

$("draft-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("lead-message").value.trim();
  if (!message && ($("lead-reference").value.trim() || $("products").value.trim())) {
    notice("Добавьте текст обращения или очистите дополнительные поля заявки."); return;
  }
  const data = {question: $("question").value.trim()};
  if (message) data.lead = {reference: $("lead-reference").value.trim() || null, message,
    products: $("products").value.split(",").map((s) => s.trim()).filter(Boolean)};
  notice(""); currentDraft = "";
  $("generate").disabled = true; $("loading").hidden = false;
  $("empty-result").hidden = true; $("result").hidden = true; $("copy").hidden = true;
  $("sources-section").hidden = true;
  try {
    const result = await api("/v1/drafts", {method: "POST", body: JSON.stringify(data)});
    currentDraft = result.draft;
    $("result-label").textContent = result.status === "insufficient_evidence" ? "Недостаточно данных" : result.generation_mode === "extractive" ? "Выдержки из документов · без генерации" : "Черновик готов";
    renderDraftText(result.draft || "По найденным документам нельзя подготовить обоснованный ответ.", result.sources);
    listBlock("clarifications", "Что уточнить у клиента", result.clarifying_questions);
    listBlock("missing", "Чего не хватает", result.missing_information);
    $("timing").textContent = `Подготовлено за ${(result.timings_ms.total / 1000).toFixed(1)} с`;
    renderSources(result.sources);
    $("result").hidden = false; $("copy").hidden = !result.draft;
  } catch (error) { notice(error.message); $("empty-result").hidden = false; }
  finally { $("loading").hidden = true; $("generate").disabled = false; }
});
$("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(currentDraft); notice("Текст скопирован вместе с обозначениями источников.", true); }
  catch { notice("Не удалось скопировать автоматически. Выделите текст ответа."); }
});

async function loadDocuments() {
  const documents = await api("/v1/documents");
  $("document-count").textContent = String(documents.length);
  $("documents").replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p"); empty.className = "muted";
    empty.textContent = "Документов пока нет. Добавьте первую инструкцию или описание товара.";
    $("documents").append(empty);
  }
  documents.forEach((documentInfo) => {
    const row = document.createElement("div"); row.className = "document-row";
    const body = document.createElement("div");
    const title = document.createElement("h3"); title.textContent = documentInfo.title;
    const meta = document.createElement("p"); meta.textContent = `${documentInfo.source_key} · Версия ${documentInfo.version} · Фрагментов: ${documentInfo.chunk_count}`;
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "text-button delete-button"; remove.textContent = "Удалить";
    remove.setAttribute("aria-label", "Удалить документ «" + documentInfo.title + "»");
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Удалить «${documentInfo.title}» из базы знаний?`)) return;
      remove.disabled = true;
      try { await api("/v1/documents/" + documentInfo.id, {method: "DELETE"}); await loadDocuments(); }
      catch (error) { notice(error.message); remove.disabled = false; }
    });
    body.append(title, meta); row.append(body, remove); $("documents").append(row);
  });
}
$("refresh").addEventListener("click", () => loadDocuments().catch((e) => notice(e.message)));
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault(); notice("");
  const file = $("document-file").files[0];
  if (!file) return;
  if (file.size > 2000000) { notice("Выберите файл размером до 2 МБ."); return; }
  const data = new FormData();
  data.append("file", file); data.append("title", $("document-title").value.trim());
  data.append("source_key", $("document-key").value.trim());
  $("upload-button").disabled = true; $("upload-button").textContent = "Индексируем…";
  try {
    const result = await api("/v1/documents/upload", {method: "POST", body: data});
    await loadDocuments();
    notice(result.changed ? `Документ сохранён. Версия ${result.document.version}.` : "Этот документ уже загружен, содержимое не изменилось.", true);
    $("upload-form").reset();
  } catch (error) { notice(error.message); }
  finally { $("upload-button").disabled = false; $("upload-button").textContent = "Добавить в базу ↗"; }
});

if (apiKey) connect().catch((error) => { notice(error.message); $("access-panel").open = true; });
