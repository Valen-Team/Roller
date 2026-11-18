(() => {
  "use strict";

  const state = {
    run: null,
    tasks: null,
    page: 1,
    pageSize: 80,
    query: "",
    status: "",
    model: "",
    score: "",
    sort: "",
    selectedId: "",
    detail: null,
    steps: new Map(),
    expanded: new Set(),
    inspectorTab: "overview",
  };

  const elements = {
    runName: document.querySelector("#run-name"),
    runPath: document.querySelector("#run-path"),
    metrics: document.querySelector("#metrics"),
    language: document.querySelector("#language-switch"),
    refresh: document.querySelector("#refresh-button"),
    toolbar: document.querySelector("#toolbar"),
    search: document.querySelector("#search-input"),
    status: document.querySelector("#status-filter"),
    model: document.querySelector("#model-filter"),
    score: document.querySelector("#score-filter"),
    sort: document.querySelector("#sort-order"),
    taskRange: document.querySelector("#task-range"),
    taskCount: document.querySelector("#task-count"),
    taskList: document.querySelector("#task-list"),
    pagination: document.querySelector("#pagination"),
    workspace: document.querySelector(".workspace"),
    taskPanel: document.querySelector(".task-panel"),
    timelinePanel: document.querySelector("#timeline-panel"),
    timelineEmpty: document.querySelector("#timeline-empty"),
    timeline: document.querySelector("#timeline-content"),
    inspectorPanel: document.querySelector("#inspector-panel"),
    inspector: document.querySelector("#inspector-content"),
    toast: document.querySelector("#toast"),
    error: document.querySelector("#error-banner"),
  };

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const jsonText = (value) => {
    try { return JSON.stringify(value, null, 2); }
    catch { return String(value); }
  };

  const statusClass = (value) => ["ok", "skipped", "error", "timeout", "partial"].includes(value)
    ? value
    : "partial";

  // The language and the layout are remembered per browser; storage can be
  // unavailable, in which case they only last for this page.
  const readStored = (key) => {
    try { return JSON.parse(localStorage.getItem(key) || "null"); }
    catch { return null; }
  };
  const writeStored = (key, value) => {
    try {
      if (value == null) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(value));
    } catch { /* Keep the choice for this page only. */ }
  };

  // English, or Chinese when the browser prefers it; a choice made with the
  // switch in the top bar wins over both.
  const LANGUAGES = ["en", "zh-CN"];
  const LANGUAGE_KEY = "roller-viewer:language";
  const storedLanguage = readStored(LANGUAGE_KEY);
  let language = LANGUAGES.includes(storedLanguage)
    ? storedLanguage
    : /^zh\b/i.test(navigator.language || "") ? "zh-CN" : "en";

  const formatNumber = (value) => {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat(language).format(number) : "—";
  };

  const formatScore = (value) => new Intl.NumberFormat(language, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  }).format(value);

  const counted = (value, one, many) => `${formatNumber(value)} ${Number(value) === 1 ? one : many}`;

  // Every string of the UI, as [English, Chinese]; a function takes the values
  // it interpolates. index.html names its entries in data-i18n attributes.
  const STRINGS = {
    // Top bar
    languageName: ["English", "简体中文"],
    languageShort: ["EN", "中文"],
    language: ["Language", "语言"],
    loadingRun: ["Loading run…", "正在读取 run…"],
    refresh: ["Refresh", "刷新"],
    refreshTitle: ["Reload the run from disk", "重新读取 run 目录"],
    refreshed: ["Run reloaded", "已重新读取 run"],
    copied: ["Copied", "已复制"],
    invalidResponse: [
      (status) => `The Viewer API returned invalid JSON (HTTP ${status})`,
      (status) => `Viewer API 返回了无效 JSON（HTTP ${status}）`,
    ],
    unknown: ["unknown", "未知"],

    // Run summary
    runSummary: ["Run summary", "运行概览"],
    resizeSummary: ["Resize the run summary", "调整运行概览高度"],
    resizeHint: ["Drag to resize, double-click to reset", "拖动调整大小，双击恢复默认"],
    metricRun: ["Run", "Run"],
    processed: [
      (done, total) => `${formatNumber(done)} / ${formatNumber(total)} processed`,
      (done, total) => `已处理 ${formatNumber(done)} / ${formatNumber(total)}`,
    ],
    noCheckpoint: ["no checkpoint", "无 checkpoint"],
    metricTasks: ["Tasks", "Task"],
    uniqueTasks: ["unique task IDs", "不重复的 task ID"],
    metricRollouts: ["Rollouts", "Rollout"],
    modelCount: [(n) => counted(n, "model", "models"), (n) => `${formatNumber(n)} 个模型`],
    metricCompleted: ["Completed", "已完成"],
    skippedOnResume: [(n) => `${formatNumber(n)} skipped on resume`, (n) => `${formatNumber(n)} 个由 Resume 跳过`],
    statusOk: ["status ok", "状态为 ok"],
    metricFailed: ["Failed", "失败"],
    timedOut: [(n) => `${formatNumber(n)} timed out`, (n) => `${formatNumber(n)} 个超时`],
    errorsAndTimeouts: ["errors and timeouts", "error 与 timeout"],
    metricTokens: ["Tokens", "Token"],
    requestCount: [(n) => counted(n, "request", "requests"), (n) => `${formatNumber(n)} 次请求`],
    metricReasoning: ["Reasoning", "Reasoning"],
    reasoningKept: ["kept", "保留"],
    reasoningStripped: ["stripped", "移除"],
    modelDefault: ["model default", "模型默认"],

    // Search and filters
    searchPlaceholder: ["Search tasks, titles, projects, models or errors", "搜索 task、标题、项目、模型或错误信息"],
    status: ["Status", "状态"],
    allStatuses: ["All statuses", "全部状态"],
    model: ["Model", "模型"],
    allModels: ["All models", "全部模型"],
    score: ["Score", "分数"],
    allScores: ["All scores", "全部分数"],
    scoreAboveZero: ["Above zero", "大于 0"],
    scoreZero: ["Zero", "0 分"],
    unscored: ["Unscored", "未评分"],
    sort: ["Sort", "排序"],
    sortTaskOrder: ["Task order", "Task 顺序"],
    sortLowestScore: ["Lowest score first", "分数从低到高"],
    sortHighestScore: ["Highest score first", "分数从高到低"],
    taskRange: [
      (start, end, total) => `${start}–${end} of ${counted(total, "task", "tasks")}`,
      (start, end, total) => `${start}–${end} / 共 ${formatNumber(total)} 个 task`,
    ],
    noMatchingTasks: ["No matching tasks", "没有匹配的 task"],

    // Task navigator
    tasksEyebrow: ["TASKS", "TASKS"],
    tasksHeading: ["Task navigator", "任务导航"],
    resizeTasks: ["Resize the task navigator", "调整任务导航宽度"],
    loading: ["Loading", "加载中"],
    noTasks: ["No tasks match the current search and filters.", "没有符合当前搜索和筛选条件的任务。"],
    unknownModel: ["Unknown model", "未知模型"],
    noTrace: ["no trace", "无轨迹"],
    previousPage: ["← Previous", "← 上一页"],
    nextPage: ["Next →", "下一页 →"],

    // Rollout header
    selectEyebrow: ["SELECT A ROLLOUT", "SELECT A ROLLOUT"],
    selectHeading: ["Select a rollout to follow the agent step by step", "选择一个 rollout，展开它的完整思考与行动轨迹"],
    selectHint: [
      "Reasoning, tool calls, tool results and the final reply appear in step order.",
      "Reasoning、工具调用、工具结果和最终回复会依照 step 顺序呈现。",
    ],
    loadingEyebrow: ["LOADING", "LOADING"],
    loadingTrajectory: ["Loading the trajectory…", "正在读取 ACTF 轨迹…"],
    noActfEyebrow: ["NO ACTF", "NO ACTF"],
    noActfHeading: ["This rollout has no ACTF trajectory", "该 rollout 没有可用的 ACTF 轨迹"],
    noActfHint: [
      "The sample has not finished, or its trajectory could not be converted.",
      "sample 尚未完成，或轨迹转换未成功。",
    ],
    scoreValue: [(value) => `Score ${value}`, (value) => `分数 ${value}`],
    stepCount: [(n) => counted(n, "step", "steps"), (n) => `${formatNumber(n)} 个 step`],
    inputTokens: [(n) => `${formatNumber(n)} input tokens`, (n) => `输入 ${formatNumber(n)} token`],
    outputTokens: [(n) => `${formatNumber(n)} output tokens`, (n) => `输出 ${formatNumber(n)} token`],
    cacheReadTokens: [(n) => `${formatNumber(n)} cache read tokens`, (n) => `缓存读取 ${formatNumber(n)} token`],
    collapseAll: ["Collapse all", "折叠全部"],
    toggleReasoning: ["Toggle reasoning", "展开/收起 reasoning"],
    inspector: ["Inspector", "Inspector"],

    // Steps
    step: ["step", "step"],
    emptyStep: ["Empty step", "空内容"],
    reasoningTag: [(n) => `Reasoning ${formatNumber(n)}`, (n) => `Reasoning ${formatNumber(n)}`],
    toolCallCount: [(n) => counted(n, "tool call", "tool calls"), (n) => `${formatNumber(n)} 次工具调用`],
    loadingStep: ["Loading step…", "正在加载 step 内容…"],
    noStepContent: ["This step has no text or tool events.", "该 step 没有可展示的文本或工具事件。"],
    userMessage: ["User message", "用户消息"],
    reasoning: ["Reasoning", "Reasoning"],
    showReasoning: ["Show reasoning", "展开 reasoning"],
    hideReasoning: ["Hide reasoning", "收起 reasoning"],
    toolCall: ["Tool call", "工具调用"],
    toolResult: ["Tool result", "工具结果"],
    toolError: ["Tool error", "工具错误"],
    assistantMessage: ["Assistant message", "助手回复"],
    userImage: ["Image in the user message", "用户消息中的图片"],
    toolImage: ["Image returned by the tool", "工具返回的图片"],
    copy: ["Copy", "复制"],
    charCount: [(n) => `${formatNumber(n)} chars`, (n) => `${formatNumber(n)} 字符`],
    showingChars: [
      (shown, total) => `Showing ${formatNumber(shown)} of ${formatNumber(total)} characters`,
      (shown, total) => `当前显示 ${formatNumber(shown)} / ${formatNumber(total)} 字符`,
    ],
    loadMore: ["Load more", "加载更多"],

    // Inspector
    inspectorEyebrow: ["INSPECTOR", "INSPECTOR"],
    inspectorHeading: ["Rollout details", "上下文"],
    resizeInspector: ["Resize the Inspector", "调整 Inspector 宽度"],
    inspectorEmpty: [
      "Select a rollout to see its run details, system prompt, tool schemas and raw metadata.",
      "选择 rollout 后可查看运行信息、system prompt、tools schema 和原始 metadata。",
    ],
    tabOverview: ["Overview", "概览"],
    tabConversation: ["Conversation", "对话"],
    tabSystem: ["System prompt", "System prompt"],
    tabTools: ["Tools", "工具"],
    tabJson: ["JSON", "JSON"],
    task: ["Task", "Task"],
    bench: ["Bench", "Bench"],
    category: ["Category", "类别"],
    harness: ["Harness", "Harness"],
    request: ["Request", "请求序号"],
    attempts: ["Attempts", "尝试次数"],
    totalTokens: ["Total tokens", "总 token"],
    duration: ["Duration", "耗时"],
    actf: ["ACTF", "ACTF"],
    available: ["available", "可用"],
    missing: ["missing", "缺失"],
    path: ["Path", "路径"],
    error: ["Error", "错误"],
    graderError: ["Grader error", "评分错误"],
    noConversation: ["This task has no User Simulator conversation.", "该 task 没有 User Simulator 对话。"],
    termination: ["Termination", "终止原因"],
    rounds: ["Rounds", "轮数"],
    visibleConversation: ["Visible conversation", "可见对话"],
    round: [(n) => `Round ${n}`, (n) => `第 ${n} 轮`],
    noRounds: ["No completed rounds.", "没有已完成的对话轮次。"],
    noSystemPrompt: ["No system prompt was captured.", "未捕获 system prompt。"],
    noTools: ["No tool schemas were captured.", "未捕获 tools schema。"],
  };

  const t = (key, ...values) => {
    const text = STRINGS[key]?.[LANGUAGES.indexOf(language)] ?? key;
    return typeof text === "function" ? text(...values) : text;
  };
  // t() for HTML templates.
  const th = (key, ...values) => escapeHtml(t(key, ...values));

  // Fill the static text of index.html and the language switch.
  const localize = () => {
    document.documentElement.lang = language;
    for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
    for (const attribute of ["aria-label", "placeholder", "title"]) {
      for (const node of document.querySelectorAll(`[data-i18n-${attribute}]`)) {
        node.setAttribute(attribute, t(node.getAttribute(`data-i18n-${attribute}`)));
      }
    }
    elements.language.innerHTML = LANGUAGES.map((code, index) => `
      <button type="button" data-language="${code}" lang="${code}" title="${escapeHtml(STRINGS.languageName[index])}" aria-pressed="${code === language}">${escapeHtml(STRINGS.languageShort[index])}</button>`).join("");
  };

  const formatDuration = (value) => {
    const seconds = Number(value);
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
    if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
    const minutes = Math.floor(seconds / 60);
    const rest = Math.round(seconds % 60);
    return `${minutes}m ${rest}s`;
  };

  const api = async (path) => {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error(t("invalidResponse", response.status)); }
    if (!response.ok) throw new Error(payload.error || response.statusText);
    return payload;
  };

  const showError = (error) => {
    elements.error.textContent = error instanceof Error ? error.message : String(error);
    elements.error.hidden = false;
    window.setTimeout(() => { elements.error.hidden = true; }, 8000);
  };

  let toastTimer = null;
  const toast = (message) => {
    elements.toast.textContent = message;
    elements.toast.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => elements.toast.classList.remove("show"), 1800);
  };

  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    toast(t("copied"));
  };

  const metric = (label, value, note = "") => `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      ${note ? `<small>${escapeHtml(note)}</small>` : ""}
    </div>`;

  const renderRun = () => {
    const run = state.run;
    if (!run) return;
    elements.runName.textContent = run.name || "Trajectory Viewer";
    elements.runPath.textContent = run.path || "";
    elements.runPath.title = run.path || "";
    const counts = run.status_counts || {};
    const completed = Number(counts.ok || 0) + Number(counts.skipped || 0);
    const failed = Number(counts.error || 0) + Number(counts.timeout || 0);
    const usage = run.usage || {};
    const progress = run.progress || {};
    elements.metrics.innerHTML = [
      metric(
        t("metricRun"),
        run.state || (run.finished_at ? "finished" : t("unknown")),
        progress.total == null ? t("noCheckpoint") : t("processed", progress.processed || 0, progress.total),
      ),
      metric(t("metricTasks"), formatNumber(run.task_count), t("uniqueTasks")),
      metric(t("metricRollouts"), formatNumber(run.rollout_count), t("modelCount", run.models?.length || 0)),
      metric(t("metricCompleted"), formatNumber(completed), counts.skipped ? t("skippedOnResume", counts.skipped) : t("statusOk")),
      metric(t("metricFailed"), formatNumber(failed), counts.timeout ? t("timedOut", counts.timeout) : t("errorsAndTimeouts")),
      metric(t("metricTokens"), formatNumber(usage.total_tokens), t("requestCount", usage.request_count)),
      metric(
        t("metricReasoning"),
        run.configuration?.keep_thinking === true
          ? t("reasoningKept")
          : run.configuration?.keep_thinking === false ? t("reasoningStripped") : t("unknown"),
        run.configuration?.thinking || t("modelDefault"),
      ),
    ].join("");

    const currentStatus = state.status;
    elements.status.innerHTML = `<option value="">${th("allStatuses")}</option>` + Object.keys(counts)
      .sort()
      .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)} (${counts[value]})</option>`)
      .join("");
    elements.status.value = currentStatus;
    const currentModel = state.model;
    elements.model.innerHTML = `<option value="">${th("allModels")}</option>` + (run.models || [])
      .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
      .join("");
    elements.model.value = currentModel;

    // The score filter and the sort order exist only for a run that was graded.
    const scoreCounts = run.score_counts || {};
    const graded = Object.keys(scoreCounts).length > 0;
    elements.toolbar.classList.toggle("graded", graded);
    for (const field of elements.toolbar.querySelectorAll("[data-graded]")) field.hidden = !graded;
    elements.score.innerHTML = `<option value="">${th("allScores")}</option>` + Object.keys(SCORE_GROUPS)
      .filter((group) => scoreCounts[group])
      .map((group) => `<option value="${group}">${th(SCORE_GROUPS[group])} (${scoreCounts[group]})</option>`)
      .join("");
    elements.score.value = state.score;
  };

  // The groups of the score filter (see _score_group in app.py) and their labels.
  const SCORE_GROUPS = { positive: "scoreAboveZero", zero: "scoreZero", unscored: "unscored" };
  const scoreGroup = (bench) => bench.score == null ? "unscored" : bench.score > 0 ? "positive" : "zero";
  const scoreText = (bench) => bench.score == null ? t("unscored") : formatScore(bench.score);

  // The score of a graded rollout; a grader error is its tooltip.
  const scoreBadge = (bench) => bench
    ? `<b class="score-badge ${scoreGroup(bench)}" title="${escapeHtml(bench.error)}">${escapeHtml(scoreText(bench))}</b>`
    : "";

  const rolloutMeta = (rollout) => {
    const tokens = rollout.usage?.total_tokens;
    return tokens ? `${formatNumber(tokens)} tok` : rollout.has_actf ? "ACTF" : t("noTrace");
  };

  // An imported session shows its title, with its project and start date below.
  const sessionLine = (item) => [item.project, (item.started_at || "").slice(0, 10)].filter(Boolean).join(" · ");

  const renderTasks = () => {
    const payload = state.tasks;
    if (!payload) return;
    elements.taskCount.textContent = formatNumber(payload.total);
    const start = payload.total ? (payload.page - 1) * payload.page_size + 1 : 0;
    const end = Math.min(payload.total, payload.page * payload.page_size);
    elements.taskRange.textContent = payload.total ? t("taskRange", start, end, payload.total) : t("noMatchingTasks");
    if (!payload.items.length) {
      elements.taskList.innerHTML = `<div class="empty-list">${th("noTasks")}</div>`;
    } else {
      elements.taskList.innerHTML = payload.items.map((task) => `
        <article class="task-group">
          <div class="task-title">
            <span class="task-index">${escapeHtml(task.source_index ?? "—")}</span>
            <strong title="${escapeHtml(task.task_id)}">${escapeHtml(task.title || task.task_id)}</strong>
          </div>
          ${task.title ? `<div class="task-session">${escapeHtml(sessionLine(task))}</div>` : ""}
          ${task.rollouts.map((rollout) => `
            <button class="rollout-row ${state.selectedId === rollout.id ? "selected" : ""}" data-rollout="${rollout.id}" type="button">
              <span class="status-dot ${statusClass(rollout.status)}" title="${escapeHtml(rollout.status)}"></span>
              <span class="rollout-model" title="${escapeHtml(rollout.model)}">${escapeHtml(rollout.model || t("unknownModel"))}</span>
              <span class="rollout-meta">${scoreBadge(rollout.bench)}${escapeHtml(rolloutMeta(rollout))}</span>
            </button>`).join("")}
        </article>`).join("");
    }
    const pageCount = Math.max(1, Math.ceil(payload.total / payload.page_size));
    elements.pagination.innerHTML = `
      <button type="button" data-page="${payload.page - 1}" ${payload.page <= 1 ? "disabled" : ""}>${th("previousPage")}</button>
      <span>${payload.page} / ${pageCount}</span>
      <button type="button" data-page="${payload.page + 1}" ${payload.page >= pageCount ? "disabled" : ""}>${th("nextPage")}</button>`;
  };

  const stepHeadline = (step) => step.content_preview || step.reasoning_preview || step.user_preview || (step.tool_call_count ? t("toolCall") : t("emptyStep"));

  const stepTags = (step) => {
    const tags = [];
    if (step.reasoning_chars) tags.push(`<i class="reasoning">${th("reasoningTag", step.reasoning_chars)}</i>`);
    if (step.tool_call_count) tags.push(`<i class="tools">${th("toolCallCount", step.tool_call_count)}</i>`);
    if (step.has_tool_error) tags.push(`<i class="error">${th("toolError")}</i>`);
    const input = step.metric?.prompt_tokens_len;
    const output = step.metric?.completion_tokens_len;
    if (input || output) tags.push(`<i>${formatNumber(input || 0)} / ${formatNumber(output || 0)} tok</i>`);
    if (step.duration_seconds != null) tags.push(`<i>${escapeHtml(formatDuration(step.duration_seconds))}</i>`);
    return tags.join("");
  };

  const truncation = (view, index) => view?.truncated ? `
    <div class="truncation-note">
      <span>${th("showingChars", view.text.length, view.length)}</span>
      <button type="button" data-full-step="${index}">${th("loadMore")}</button>
    </div>` : "";

  // The text of a message, with its images where they occur in it.
  const contentHtml = (view, imageAlt) => view?.blocks
    ? view.blocks.map((block) => block.type === "image"
      ? `<img class="trace-image" src="${escapeHtml(block.url)}" alt="${escapeHtml(imageAlt)}" loading="lazy">`
      : `<pre>${escapeHtml(block.text)}</pre>`).join("")
    : `<pre>${escapeHtml(view?.text || "")}</pre>`;

  const textSection = (kind, title, view, index) => {
    if (!view?.text) return "";
    if (kind === "reasoning") {
      return `<section class="trace-section reasoning">
        <header><span>${escapeHtml(title)}</span><span>${th("charCount", view.length)}</span></header>
        <details data-reasoning>
          <summary><span class="when-closed">${th("showReasoning")}</span><span class="when-open">${th("hideReasoning")}</span></summary>
          <pre>${escapeHtml(view.text)}</pre>
          ${truncation(view, index)}
        </details>
      </section>`;
    }
    return `<section class="trace-section ${kind}">
      <header><span>${escapeHtml(title)}</span><button type="button" data-copy="${kind}">${th("copy")}</button></header>
      ${contentHtml(view, t("userImage"))}
      ${truncation(view, index)}
    </section>`;
  };

  const observationSection = (observation, index) => `
    <section class="trace-section tool-result ${observation.is_error ? "error" : ""}">
      <header>
        <span>${observation.is_error ? th("toolError") : th("toolResult")} · ${escapeHtml(observation.tool_name || observation.tool_call_id || t("unknown"))}</span>
        <span>${th("charCount", observation.content?.length || 0)}</span>
      </header>
      ${contentHtml(observation.content, t("toolImage"))}
      ${truncation(observation.content, index)}
    </section>`;

  const renderStepBody = (step, index) => {
    const observations = [...(step.observations || [])];
    const toolHtml = (step.assistant?.tool_calls || []).map((call) => {
      const matches = observations.filter((item) => item.tool_call_id && item.tool_call_id === call.id);
      for (const match of matches) observations.splice(observations.indexOf(match), 1);
      return `<div>
        <section class="trace-section tool-call">
          <header><span>${th("toolCall")} · ${escapeHtml(call.name || t("unknown"))}</span><span>${escapeHtml(call.id || "")}</span></header>
          <pre>${escapeHtml(call.arguments?.text || "{}")}</pre>
          ${truncation(call.arguments, index)}
        </section>
        ${matches.map((item) => observationSection(item, index)).join("")}
      </div>`;
    }).join("");
    return [
      textSection("user", t("userMessage"), step.user, index),
      textSection("reasoning", t("reasoning"), step.assistant?.reasoning, index),
      toolHtml,
      observations.map((item) => observationSection(item, index)).join(""),
      textSection("assistant", t("assistantMessage"), step.assistant?.content, index),
    ].filter(Boolean).join("") || `<div class="empty-list">${th("noStepContent")}</div>`;
  };

  const renderTimeline = () => {
    const detail = state.detail;
    if (!detail) {
      elements.timelineEmpty.hidden = false;
      elements.timeline.hidden = true;
      return;
    }
    elements.timelineEmpty.hidden = true;
    elements.timeline.hidden = false;
    const rollout = detail.rollout;
    const trajectory = detail.trajectory;
    const usage = rollout.usage || {};
    const bench = rollout.bench;
    const benchLabel = [bench?.name, bench?.category].filter(Boolean).join(" · ");
    const header = `
      <header class="trajectory-header">
        <div class="trajectory-title-row">
          <div>
            <span class="eyebrow">${escapeHtml(rollout.model || "ROLLOUT")}</span>
            <h2>${escapeHtml(rollout.title || rollout.task_id)}</h2>
            ${rollout.title ? `<p>${escapeHtml(sessionLine(rollout))}</p>` : ""}
            <p>${escapeHtml(rollout.path)}</p>
          </div>
          <div class="trajectory-chips">
            ${bench ? `<span class="status-chip ${scoreGroup(bench)}" title="${escapeHtml(bench.error)}">${bench.score == null ? th("unscored") : th("scoreValue", formatScore(bench.score))}</span>` : ""}
            <span class="status-chip ${statusClass(rollout.status)}">${escapeHtml(rollout.status)}</span>
          </div>
        </div>
        <div class="trajectory-stats">
          ${benchLabel ? `<span>${escapeHtml(benchLabel)}</span>` : ""}
          <span>${th("stepCount", trajectory?.step_count || 0)}</span>
          <span>${th("inputTokens", usage.input_tokens)}</span>
          <span>${th("outputTokens", usage.output_tokens)}</span>
          <span>${th("cacheReadTokens", usage.cache_read_tokens)}</span>
          <span>${escapeHtml(formatDuration(rollout.timing?.elapsed_seconds ?? usage.elapsed_seconds))}</span>
        </div>
        <div class="trajectory-actions">
          <button type="button" data-action="collapse">${th("collapseAll")}</button>
          <button type="button" data-action="reasoning">${th("toggleReasoning")}</button>
          <button type="button" data-action="inspector">${th("inspector")}</button>
        </div>
      </header>`;
    if (!trajectory) {
      elements.timeline.innerHTML = header + `
        <div class="empty-state no-actf">
          <p class="eyebrow">${th("noActfEyebrow")}</p>
          <h2>${th("noActfHeading")}</h2>
          <p>${escapeHtml(rollout.error || t("noActfHint"))}</p>
        </div>`;
      return;
    }
    const steps = trajectory.steps || [];
    const scrollTop = elements.timelinePanel.scrollTop;
    elements.timeline.innerHTML = header + `<ol class="step-list">${steps.map((step) => {
      const expanded = state.expanded.has(step.index);
      const loaded = state.steps.get(step.index);
      return `<li class="step-card ${expanded ? "expanded" : ""}">
        <button class="step-toggle" type="button" data-step="${step.index}" aria-expanded="${expanded}">
          <span class="step-number">${escapeHtml(step.step_id)}</span>
          <span class="step-summary">
            <strong>${escapeHtml(stepHeadline(step))}</strong>
            <span>${escapeHtml(step.metric?.stop_reason || t("step"))} · ${escapeHtml(step.started_at || "")}</span>
          </span>
          <span class="step-tags">${stepTags(step)}<b class="step-chevron">⌄</b></span>
        </button>
        ${expanded ? `<div class="step-body">${loaded ? renderStepBody(loaded, step.index) : `<div class="loading-step">${th("loadingStep")}</div>`}</div>` : ""}
      </li>`;
    }).join("")}</ol>`;
    elements.timelinePanel.scrollTop = scrollTop;
  };

  const infoRows = (rollout) => {
    const usage = rollout.usage || {};
    const bench = rollout.bench;
    return [
      [t("task"), rollout.task_id], [t("model"), rollout.model], [t("status"), rollout.status],
      ...(bench ? [[t("bench"), bench.name || "—"], [t("category"), bench.category || "—"], [t("score"), scoreText(bench)]] : []),
      [t("harness"), rollout.harness || "—"], [t("request"), rollout.request_index ?? "—"],
      [t("attempts"), rollout.attempt_count], [t("totalTokens"), formatNumber(usage.total_tokens)],
      [t("duration"), formatDuration(rollout.timing?.elapsed_seconds ?? usage.elapsed_seconds)],
      [t("actf"), rollout.has_actf ? t("available") : t("missing")], [t("path"), rollout.path],
    ];
  };

  const conversationHtml = (conversation) => {
    if (!conversation) return `<div class="empty-list">${th("noConversation")}</div>`;
    const rounds = Array.isArray(conversation.rounds) ? conversation.rounds : [];
    return `
      <dl class="info-table">
        <dt>${th("status")}</dt><dd>${escapeHtml(conversation.status || "—")}</dd>
        <dt>${th("termination")}</dt><dd>${escapeHtml(conversation.termination?.reason || "—")}</dd>
        <dt>${th("rounds")}</dt><dd>${formatNumber(rounds.length)}</dd>
      </dl>
      <h3>${th("visibleConversation")}</h3>
      ${rounds.map((round, index) => `<div class="conversation-round">
        <strong>${th("round", index + 1)}</strong>
        <pre class="code-block">${escapeHtml(jsonText(round))}</pre>
      </div>`).join("") || `<div class="empty-list">${th("noRounds")}</div>`}`;
  };

  const renderInspector = () => {
    const detail = state.detail;
    if (!detail) {
      elements.inspector.className = "inspector-empty";
      elements.inspector.textContent = t("inspectorEmpty");
      return;
    }
    const tabs = [
      ["overview", "tabOverview"], ["system", "tabSystem"], ["tools", "tabTools"], ["json", "tabJson"],
    ];
    if (detail.conversation) tabs.splice(1, 0, ["conversation", "tabConversation"]);
    if (!tabs.some(([key]) => key === state.inspectorTab)) state.inspectorTab = "overview";
    let body = "";
    if (state.inspectorTab === "overview") {
      body = `<dl class="info-table">${infoRows(detail.rollout).map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join("")}</dl>`;
      if (detail.rollout.error) body += `<h3>${th("error")}</h3><pre class="code-block inspector-code">${escapeHtml(detail.rollout.error)}</pre>`;
      if (detail.rollout.bench?.error) body += `<h3>${th("graderError")}</h3><pre class="code-block inspector-code">${escapeHtml(detail.rollout.bench.error)}</pre>`;
    } else if (state.inspectorTab === "conversation") {
      body = detail.conversation_error
        ? `<div class="empty-list">${escapeHtml(detail.conversation_error)}</div>`
        : conversationHtml(detail.conversation);
    } else if (state.inspectorTab === "system") {
      const prompt = detail.trajectory?.system_prompt?.text || "";
      body = prompt ? `<pre class="code-block inspector-code">${escapeHtml(prompt)}</pre>` : `<div class="empty-list">${th("noSystemPrompt")}</div>`;
    } else if (state.inspectorTab === "tools") {
      const tools = detail.trajectory?.tools || [];
      body = tools.length ? `<pre class="code-block inspector-code">${escapeHtml(jsonText(tools))}</pre>` : `<div class="empty-list">${th("noTools")}</div>`;
    } else {
      body = `<pre class="code-block inspector-code">${escapeHtml(jsonText(detail.info))}</pre>`;
    }
    elements.inspector.className = "inspector-wrap";
    elements.inspector.innerHTML = `
      <nav class="tabs">${tabs.map(([key, label]) => `<button type="button" data-tab="${key}" class="${state.inspectorTab === key ? "active" : ""}">${th(label)}</button>`).join("")}</nav>
      <div class="inspector-body">${body}</div>`;
  };

  const loadTasks = async () => {
    elements.taskList.innerHTML = '<div class="skeleton-list"><i></i><i></i><i></i><i></i></div>';
    const params = new URLSearchParams({ page: String(state.page), page_size: String(state.pageSize) });
    if (state.query) params.set("query", state.query);
    if (state.status) params.set("status", state.status);
    if (state.model) params.set("model", state.model);
    if (state.score) params.set("score", state.score);
    if (state.sort) params.set("sort", state.sort);
    state.tasks = await api(`/api/v1/tasks?${params}`);
    renderTasks();
  };

  const selectRollout = async (rolloutId, updateHash = true) => {
    if (!rolloutId) return;
    state.selectedId = rolloutId;
    state.detail = null;
    state.steps.clear();
    state.expanded.clear();
    state.inspectorTab = "overview";
    renderTasks();
    elements.timelineEmpty.hidden = true;
    elements.timeline.hidden = false;
    elements.timeline.innerHTML = `<div class="empty-state"><p class="eyebrow">${th("loadingEyebrow")}</p><h2>${th("loadingTrajectory")}</h2></div>`;
    try {
      state.detail = await api(`/api/v1/rollouts/${encodeURIComponent(rolloutId)}`);
      if (updateHash) history.replaceState(null, "", `#/rollout/${encodeURIComponent(rolloutId)}`);
      elements.timelinePanel.scrollTop = 0;
      renderTimeline();
      renderInspector();
    } catch (error) {
      showError(error);
      state.selectedId = "";
      renderTimeline();
    }
  };

  const loadStep = async (index, limit = 40_000) => {
    try {
      const step = await api(`/api/v1/rollouts/${encodeURIComponent(state.selectedId)}/steps/${index}?limit=${limit}`);
      state.steps.set(index, step);
      renderTimeline();
    } catch (error) {
      state.expanded.delete(index);
      renderTimeline();
      showError(error);
    }
  };

  const toggleStep = (index) => {
    if (state.expanded.has(index)) {
      state.expanded.delete(index);
      renderTimeline();
      return;
    }
    state.expanded.add(index);
    renderTimeline();
    if (!state.steps.has(index)) void loadStep(index);
  };

  const loadRun = async (refresh = false) => {
    elements.refresh.classList.add("busy");
    try {
      state.run = await api(`/api/v1/run${refresh ? "?refresh=1" : ""}`);
      renderRun();
      // The summary's full height is known only once its cards are rendered.
      applyMetrics();
      await loadTasks();
      if (refresh) toast(t("refreshed"));
    } catch (error) {
      showError(error);
    } finally {
      elements.refresh.classList.remove("busy");
    }
  };

  elements.taskList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-rollout]");
    if (button) void selectRollout(button.dataset.rollout);
  });

  elements.pagination.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page]");
    if (!button || button.disabled) return;
    state.page = Number(button.dataset.page);
    void loadTasks();
  });

  elements.timeline.addEventListener("click", (event) => {
    const stepButton = event.target.closest("[data-step]");
    if (stepButton) {
      toggleStep(Number(stepButton.dataset.step));
      return;
    }
    const fullButton = event.target.closest("[data-full-step]");
    if (fullButton) {
      void loadStep(Number(fullButton.dataset.fullStep), 1_000_000);
      return;
    }
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "collapse") {
      state.expanded.clear();
      renderTimeline();
    } else if (action === "reasoning") {
      const details = [...elements.timeline.querySelectorAll("details[data-reasoning]")];
      const shouldOpen = details.some((item) => !item.open);
      details.forEach((item) => { item.open = shouldOpen; });
    } else if (action === "inspector") {
      elements.inspectorPanel.classList.toggle("open");
    }
    const copyButton = event.target.closest("[data-copy]");
    if (copyButton) {
      const pre = copyButton.closest(".trace-section")?.querySelector("pre");
      if (pre) void copyText(pre.textContent || "");
    }
  });

  elements.inspector.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab]");
    if (!button) return;
    state.inspectorTab = button.dataset.tab;
    renderInspector();
  });

  let searchTimer = null;
  elements.search.addEventListener("input", () => {
    window.clearTimeout(searchTimer);
    searchTimer = window.setTimeout(() => {
      state.query = elements.search.value.trim();
      state.page = 1;
      void loadTasks();
    }, 220);
  });
  elements.status.addEventListener("change", () => {
    state.status = elements.status.value;
    state.page = 1;
    void loadTasks();
  });
  elements.model.addEventListener("change", () => {
    state.model = elements.model.value;
    state.page = 1;
    void loadTasks();
  });
  elements.score.addEventListener("change", () => {
    state.score = elements.score.value;
    state.page = 1;
    void loadTasks();
  });
  elements.sort.addEventListener("change", () => {
    state.sort = elements.sort.value;
    state.page = 1;
    void loadTasks();
  });
  elements.refresh.addEventListener("click", () => void loadRun(true));

  elements.language.addEventListener("click", (event) => {
    const code = event.target.closest("[data-language]")?.dataset.language;
    if (!code || code === language) return;
    language = code;
    writeStored(LANGUAGE_KEY, code);
    localize();
    renderRun();
    // The run summary can change height with the length of its text.
    applyMetrics();
    renderTasks();
    renderTimeline();
    renderInspector();
  });

  // Track a pointer drag that started on a resizer; onMove gets the offset from
  // where it started, onEnd runs only if the pointer actually moved.
  const followDrag = (handle, event, axis, onMove, onEnd) => {
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    let moved = false;
    const move = (moveEvent) => {
      moved = true;
      onMove(moveEvent.clientX - startX, moveEvent.clientY - startY);
    };
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      handle.classList.remove("dragging");
      delete document.body.dataset.resizing;
      if (moved) onEnd();
    };
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    document.body.dataset.resizing = axis;
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  };

  // Arrow keys move a focused resizer: 16px a press, 64px with Shift.
  const arrowStep = (event, decrease, increase) => {
    if (event.key !== decrease && event.key !== increase) return 0;
    event.preventDefault();
    return (event.key === increase ? 1 : -1) * (event.shiftKey ? 64 : 16);
  };

  // Side panels keep the stylesheet's proportional widths until the user drags
  // a resizer; from then on both are pinned in px.
  const COLUMNS_KEY = "roller-viewer:columns";
  const MIN_TIMELINE = 360;
  const columns = {
    task: { panel: elements.taskPanel, variable: "--task-width", min: 200, fallback: 300 },
    inspector: { panel: elements.inspectorPanel, variable: "--inspector-width", min: 240, fallback: 340 },
  };
  let columnWidths = null;

  const resizer = (side) => elements.workspace.querySelector(`[data-resize="${side}"]`);
  // A panel is a grid column only while its resizer is displayed (see the media queries).
  const inGrid = (side) => resizer(side).offsetParent !== null;

  const applyColumns = () => {
    const style = elements.workspace.style;
    if (!columnWidths) {
      for (const column of Object.values(columns)) style.removeProperty(column.variable);
      return;
    }
    // Shrink the saved widths together when the window cannot fit them.
    const sides = Object.keys(columns).filter(inGrid);
    const handles = sides.reduce((sum, side) => sum + resizer(side).offsetWidth, 0);
    const wanted = sides.reduce((sum, side) => sum + columnWidths[side], 0);
    const room = elements.workspace.clientWidth - handles - MIN_TIMELINE;
    const scale = wanted > room ? Math.max(room, 0) / wanted : 1;
    for (const side of sides) {
      const width = Math.max(columns[side].min, Math.floor(columnWidths[side] * scale));
      style.setProperty(columns[side].variable, `${width}px`);
    }
  };

  // Pin both panels at their rendered widths so a drag moves only one edge.
  const pinColumns = () => {
    const width = (side) => inGrid(side)
      ? Math.round(columns[side].panel.getBoundingClientRect().width)
      : columnWidths?.[side] ?? columns[side].fallback;
    columnWidths = { task: width("task"), inspector: width("inspector") };
  };

  // Widest the panel may get while the timeline keeps MIN_TIMELINE.
  const columnLimit = (side) => columnWidths[side] + elements.timelinePanel.getBoundingClientRect().width - MIN_TIMELINE;

  const resizeColumn = (side, width, limit) => {
    const { min } = columns[side];
    columnWidths[side] = Math.round(Math.min(Math.max(width, min), Math.max(limit, min)));
    applyColumns();
  };

  // Dragging right widens the task panel but narrows the inspector.
  const towards = (side) => (side === "task" ? 1 : -1);

  elements.workspace.addEventListener("pointerdown", (event) => {
    const handle = event.target.closest("[data-resize]");
    if (!handle || event.button !== 0) return;
    const side = handle.dataset.resize;
    let start = null;
    followDrag(handle, event, "column", (dx) => {
      if (!start) {
        pinColumns();
        start = { width: columnWidths[side], limit: columnLimit(side) };
      }
      resizeColumn(side, start.width + towards(side) * dx, start.limit);
    }, () => writeStored(COLUMNS_KEY, columnWidths));
  });

  elements.workspace.addEventListener("dblclick", (event) => {
    if (!event.target.closest("[data-resize]")) return;
    columnWidths = null;
    applyColumns();
    writeStored(COLUMNS_KEY, null);
  });

  elements.workspace.addEventListener("keydown", (event) => {
    const handle = event.target.closest("[data-resize]");
    const step = handle ? arrowStep(event, "ArrowLeft", "ArrowRight") : 0;
    if (!step) return;
    const side = handle.dataset.resize;
    pinColumns();
    resizeColumn(side, columnWidths[side] + towards(side) * step, columnLimit(side));
    writeStored(COLUMNS_KEY, columnWidths);
  });

  const savedColumns = readStored(COLUMNS_KEY);
  if (Number.isFinite(savedColumns?.task) && Number.isFinite(savedColumns?.inspector)) {
    columnWidths = { task: savedColumns.task, inspector: savedColumns.inspector };
  }

  // The run summary shrinks from the bottom (down to nothing) by dragging the
  // resizer under it; at its full height it is left to size itself again.
  const METRICS_KEY = "roller-viewer:metrics-height";
  const metricsResizer = document.querySelector('[data-resize="metrics"]');
  let metricsHeight = null;

  // Full height of the run summary, measured with any pinned height lifted.
  const naturalMetricsHeight = () => {
    const { metrics } = elements;
    const pinned = metrics.style.getPropertyValue("height");
    const collapsed = metrics.classList.contains("collapsed");
    metrics.style.removeProperty("height");
    metrics.classList.remove("collapsed");
    const height = metrics.offsetHeight;
    if (pinned) metrics.style.setProperty("height", pinned);
    metrics.classList.toggle("collapsed", collapsed);
    return height;
  };

  const applyMetrics = () => {
    const pinned = metricsHeight != null && metricsHeight < naturalMetricsHeight();
    elements.metrics.classList.toggle("collapsed", pinned && metricsHeight === 0);
    if (pinned) elements.metrics.style.setProperty("height", `${metricsHeight}px`);
    else elements.metrics.style.removeProperty("height");
  };

  const resizeMetrics = (height) => {
    const full = naturalMetricsHeight();
    const clamped = Math.round(Math.min(Math.max(height, 0), full));
    metricsHeight = clamped < full ? clamped : null;
    applyMetrics();
  };

  metricsResizer.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const startHeight = elements.metrics.offsetHeight;
    followDrag(metricsResizer, event, "row", (dx, dy) => resizeMetrics(startHeight + dy),
      () => writeStored(METRICS_KEY, metricsHeight));
  });

  metricsResizer.addEventListener("dblclick", () => {
    metricsHeight = null;
    applyMetrics();
    writeStored(METRICS_KEY, null);
  });

  metricsResizer.addEventListener("keydown", (event) => {
    const step = arrowStep(event, "ArrowUp", "ArrowDown");
    if (!step) return;
    resizeMetrics(elements.metrics.offsetHeight + step);
    writeStored(METRICS_KEY, metricsHeight);
  });

  const savedMetrics = readStored(METRICS_KEY);
  if (Number.isFinite(savedMetrics)) metricsHeight = savedMetrics;

  window.addEventListener("resize", () => {
    applyColumns();
    applyMetrics();
  });
  applyColumns();

  const routeRollout = () => {
    const match = location.hash.match(/^#\/rollout\/([^/?]+)/);
    return match ? decodeURIComponent(match[1]) : "";
  };

  window.addEventListener("hashchange", () => {
    const rolloutId = routeRollout();
    if (rolloutId && rolloutId !== state.selectedId) void selectRollout(rolloutId, false);
  });

  localize();
  renderInspector();
  void (async () => {
    await loadRun(false);
    const rolloutId = routeRollout();
    if (rolloutId) await selectRollout(rolloutId, false);
  })();
})();
