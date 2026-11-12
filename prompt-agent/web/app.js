const state = {
  status: {},
  profiles: [],
  promptTypeProfiles: [],
  packs: [],
  requests: [],
  referenceImages: [],
  imageResults: [],
  imageReviews: [],
  settings: {},
  generationReviewedImages: [],
  generationReviewGroup: "included",
  selectedReferenceImagePath: "",
  currentRunId: "",
  pollTimer: null,
  humanDecisions: new Map(),
  incubation: {
    requestId: "",
    referenceImages: [],
    extractRunId: "",
    candidatePromptIds: [],
    validationRunId: "",
    incubationRunId: "",
    reviewedImages: [],
    reviewed: false,
    activeReviewGroup: "included",
  },
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function refreshAll() {
  const [status, profiles, promptTypes, requests, packs, refs, results, reviews, settings] = await Promise.all([
    api("/api/status"),
    api("/api/provider-profiles"),
    api("/api/prompt-type-profiles"),
    api("/api/idea-requests"),
    api("/api/prompt-packs"),
    api("/api/reference-images"),
    api("/api/image-results?latest=24"),
    api("/api/image-reviews?latest=100"),
    api("/api/settings"),
  ]);
  state.status = status;
  state.profiles = profiles.provider_profiles || [];
  state.promptTypeProfiles = promptTypes.prompt_type_profiles || [];
  state.requests = requests.idea_requests || [];
  state.packs = packs.prompt_packs || [];
  state.referenceImages = refs.reference_images || [];
  state.imageResults = results.image_results || [];
  state.imageReviews = reviews.image_reviews || [];
  state.settings = settings || {};
  renderStatus();
  renderControls();
  renderPromptTypeControls();
  renderReferenceImages();
  renderPackTable();
  renderImageResults();
  renderSettings();
  loadGenerationReviewGallery().catch(console.warn);
  $("lastUpdated").textContent = new Date().toLocaleTimeString();
}

function renderStatus() {
  $("packCount").textContent = state.status.prompt_pack_count || 0;
  $("itemCount").textContent = state.status.prompt_pack_item_count || 0;
  $("imageCount").textContent = state.status.image_result_count || 0;
  $("reviewCount").textContent = state.status.image_review_count || 0;
  $("currentProfile").textContent = $("profileSelect")?.value || "默认 / Mock Provider";
  const provider = state.status.aesthetic_review_provider || "mock";
  const url = state.status.aesthetic_review_api_url || "";
  $("tasteLensStatus").textContent = provider === "tastelens_http" && url
    ? `tastelens_http + ${url}`
    : provider === "mock"
      ? "mock"
      : "请在终端环境变量中配置 AESTHETIC_REVIEW_PROVIDER / AESTHETIC_REVIEW_API_URL";
}

function renderControls() {
  syncSelect(
    $("requestSelect"),
    state.requests,
    (request) => request.request_id,
    (request) => `${request.request_id} | ${request.title || request.topic || request.summary || request.raw_idea || ""}`,
  );
  syncSelect(
    $("packSelect"),
    state.packs,
    (pack) => pack.pack_id,
    (pack) => `${pack.pack_name || pack.pack_id} (${pack.item_count || 0})`,
  );
  const currentProfile = $("profileSelect").value;
  $("profileSelect").innerHTML = `<option value="">默认 / Mock Provider</option>`;
  for (const profile of state.profiles) {
    const option = document.createElement("option");
    option.value = profile.profile_path;
    option.textContent = `${profile.display_name || profile.file_name} | ${profile.provider || "unknown"}`;
    $("profileSelect").appendChild(option);
  }
  $("profileSelect").value = currentProfile;
  const incProfile = $("incProfileSelect");
  if (incProfile) {
    const currentIncProfile = incProfile.value;
    incProfile.innerHTML = `<option value="">默认 / Mock Provider</option>`;
    for (const profile of state.profiles) {
      const option = document.createElement("option");
      option.value = profile.profile_path;
      option.textContent = `${profile.display_name || profile.file_name} | ${profile.provider || "unknown"}`;
      incProfile.appendChild(option);
    }
    incProfile.value = currentIncProfile;
  }
}

function renderSettings() {
  if (!$("settingsWorkspace")) return;
  const settings = state.settings.settings || {};
  const secrets = state.settings.secrets || {};
  $("settingsMode").textContent = state.settings.settings_mode || "local";
  $("settingsImageProvider").value = settings.image_provider || "mock";
  $("settingsImageModel").value = settings.image_model || "";
  $("settingsReviewProvider").value = settings.aesthetic_review_provider || "mock";
  $("settingsReviewUrl").value = settings.aesthetic_review_api_url || "";
  $("settingsReviewImageField").value = settings.aesthetic_review_image_field || "image";
  $("settingsReviewScoreField").value = settings.aesthetic_review_score_field || "displayScore";
  $("settingsReviewRatingField").value = settings.aesthetic_review_rating_field || "rating";
  $("settingsReviewReasonField").value = settings.aesthetic_review_reason_field || "analysis";
  renderSettingsProfileSelect(settings.image_provider_profile_path || "");
  updateSecretStatus(secrets);
}

function renderSettingsProfileSelect(currentValue = "") {
  const select = $("settingsProfileSelect");
  if (!select) return;
  select.innerHTML = `<option value="">默认 / 环境变量 Provider</option>`;
  for (const profile of state.profiles) {
    const option = document.createElement("option");
    option.value = profile.profile_path;
    option.textContent = `${profile.display_name || profile.file_name} | ${profile.provider || "unknown"}`;
    select.appendChild(option);
  }
  select.value = currentValue;
}

function updateSecretStatus(secrets) {
  const imageProvider = $("settingsImageProvider")?.value || "mock";
  const imageKeyName = imageProvider === "wuyin_async" ? "WUYIN_API_KEY" : imageProvider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY";
  const imageSecret = secrets[imageKeyName] || {};
  $("settingsImageKeyStatus").textContent = imageSecret.exists
    ? `已配置 ${imageKeyName}：${imageSecret.masked}（${imageSecret.source}）`
    : `未配置 ${imageKeyName}`;
  const tasteLensSecret = secrets.TASTELENS_API_KEY || {};
  $("settingsTasteLensKeyStatus").textContent = tasteLensSecret.exists
    ? `已配置 TASTELENS_API_KEY：${tasteLensSecret.masked}（${tasteLensSecret.source}）`
    : "未配置 TASTELENS_API_KEY（本地 TasteLens 可不需要）";
}

function syncSelect(select, items, valueFn, labelFn) {
  const current = select.value;
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = valueFn(item);
    option.textContent = labelFn(item);
    select.appendChild(option);
  }
  select.value = current;
}

function renderPromptTypeControls() {
  const select = $("promptTypeSelect");
  if (!select) return;
  const current = select.value || "portrait";
  select.innerHTML = "";
  for (const profile of state.promptTypeProfiles) {
    const option = document.createElement("option");
    option.value = profile.type_id;
    option.textContent = profile.label || profile.type_id;
    select.appendChild(option);
  }
  select.value = state.promptTypeProfiles.some((item) => item.type_id === current) ? current : "portrait";
  applyPromptTypeProfile();
}

function selectedPromptTypeProfile() {
  const typeId = $("promptTypeSelect")?.value || "portrait";
  return state.promptTypeProfiles.find((item) => item.type_id === typeId) || state.promptTypeProfiles[0] || {};
}

function applyPromptTypeProfile() {
  const profile = selectedPromptTypeProfile();
  if (!profile.type_id) return;
  $("incTitle").placeholder = profile.title_placeholder || "";
  $("incDescription").placeholder = profile.description_placeholder || "";
  $("incRatio").value = profile.default_ratio || "3:4";
  $("candidateCount").placeholder = `默认 ${profile.candidate_count_default || 8}`;
  $("incImagesPerPrompt").placeholder = `默认 ${profile.images_per_prompt_default || 1}`;
  const hint = profile.requires_reference_images ? "该类型需要上传样图。" : "该类型样图可选。";
  $("incReferenceList").dataset.requirementHint = hint;
}

function renderReferenceImages() {
  const select = $("referenceSelect");
  const current = state.selectedReferenceImagePath || select.value;
  select.innerHTML = `<option value="">不使用样图</option>`;
  for (const image of state.referenceImages) {
    const option = document.createElement("option");
    option.value = image.image_path;
    option.textContent = image.file_name;
    select.appendChild(option);
  }
  select.value = current;
  state.selectedReferenceImagePath = select.value;
  updateReferencePreview();
}

function updateReferencePreview() {
  const selected = state.referenceImages.find((item) => item.image_path === state.selectedReferenceImagePath);
  if (!selected) {
    $("referencePreview").className = "reference-preview empty";
    $("referencePreview").textContent = "未选择样图";
    return;
  }
  $("referencePreview").className = "reference-preview";
  $("referencePreview").innerHTML = `<img src="${escapeHtml(selected.file_url || toFileUrl(selected.image_path) || "")}" alt="${escapeHtml(selected.file_name)}" />`;
}

function renderPackTable() {
  const table = $("packTable");
  table.innerHTML = "";
  for (const pack of state.packs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(pack.pack_name || pack.pack_id)}</strong></td>
      <td><span class="meta">${escapeHtml(pack.pack_id)}</span></td>
      <td><span class="status-pill ${escapeHtml(pack.status || "")}">${escapeHtml(pack.status || "")}</span></td>
      <td>${pack.item_count || 0}</td>
      <td><span class="muted">${escapeHtml([pack.domain, pack.platform, pack.use_case, pack.style].filter(Boolean).join(" / "))}</span></td>
      <td>${(pack.match_tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join(" ")}</td>
      <td><button class="delete-btn" data-delete="${escapeHtml(pack.pack_id)}">删除</button></td>
    `;
    table.appendChild(tr);
  }
  table.querySelectorAll("[data-delete]").forEach((button) => button.addEventListener("click", () => deletePack(button.dataset.delete)));
}

function renderImageResults() {
  const reviewByImage = new Map(state.imageReviews.map((review) => [review.image_result_id, review]));
  const grid = $("imageGrid");
  grid.innerHTML = "";
  for (const result of state.imageResults) {
    const review = reviewByImage.get(result.image_result_id) || {};
    const aesthetic = review.aesthetic || {};
    const savedDecision = state.humanDecisions.get(result.image_result_id);
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      ${renderThumb(result)}
      <div class="card-body">
        <div class="card-title">
          <strong>${escapeHtml(result.prompt_id || "无 prompt_id")}</strong>
          <span class="status-pill ${escapeHtml(result.status || "")}">${escapeHtml(result.status || "")}</span>
        </div>
        <div class="meta">${escapeHtml(result.image_result_id || "")}</div>
        <div class="meta">Provider ${escapeHtml(result.provider || "")} / ${escapeHtml(result.model || "")}</div>
        <div class="review">
          <div class="review-score">
            <span>审美 ${aesthetic.score ?? "-"}</span>
            <span>${escapeHtml(aesthetic.rating || "")}</span>
          </div>
          <div class="muted">${escapeHtml(aesthetic.reason || result.error_message || result.file_error || "")}</div>
          <div class="actions-grid">
            ${["keep", "reject", "revise", "high_potential"]
              .map((decision) => `<button class="${savedDecision === decision ? "saved" : ""}" data-human="${decision}" data-image="${escapeHtml(result.image_result_id)}" data-prompt="${escapeHtml(result.prompt_id || "")}">${savedDecision === decision ? "已保存 " : ""}${decision}</button>`)
              .join("")}
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  }
  grid.querySelectorAll("[data-human]").forEach((button) => {
    button.addEventListener("click", () => submitHumanReview(button.dataset.image, button.dataset.prompt, button.dataset.human, button));
  });
}

function renderThumb(result) {
  if (result.status === "failed") {
    return `<div class="failed-thumb">Failed</div>`;
  }
  const url = result.file_url || result.image_url || toFileUrl(result.image_path);
  if (url) {
    return `<div class="thumb"><img src="${escapeHtml(url)}" alt="${escapeHtml(result.image_result_id || "")}" loading="lazy" /></div>`;
  }
  if (result.provider === "mock") {
    return `<pre class="empty">${escapeHtml(result.mock_preview || result.prompt_preview || "Mock image result")}</pre>`;
  }
  return `<div class="empty">${escapeHtml(result.file_error || "图片文件不存在")}</div>`;
}

function renderReviewGallery(options) {
  const {
    containerId,
    images,
    mode,
    activeGroup,
    countIds,
    summaryId,
    onDecisionSaved,
  } = options;
  const grid = $(containerId);
  grid.innerHTML = "";
  const included = images.filter((item) => item.auto_group !== "filtered_out");
  const filtered = images.filter((item) => item.auto_group === "filtered_out");
  if (countIds?.included) $(countIds.included).textContent = included.length;
  if (countIds?.filtered) $(countIds.filtered).textContent = filtered.length;
  document.querySelectorAll(`[data-gallery="${mode}"]`).forEach((button) => {
    button.classList.toggle("active", button.dataset.reviewGroup === activeGroup);
  });
  const visible = activeGroup === "filtered_out" ? filtered : included;
  for (const result of visible) {
    const decision = state.humanDecisions.get(result.image_result_id) || result.human_decision || "";
    const filteredOut = result.auto_group === "filtered_out";
    const sourcePrefix = mode === "generation" ? "generation_" : "";
    const actions = filteredOut
      ? [
          ["rescue", "救回"],
          ["revise", "需要修改"],
          ["reject", "确认淘汰"],
        ]
      : [
          ["keep", "keep"],
          ["high_potential", "high_potential"],
          ["revise", "revise"],
          ["reject", "reject"],
        ];
    const reasons = result.auto_reject_reasons || [];
    const card = document.createElement("article");
    card.className = `card ${filteredOut ? "filtered-card" : ""} ${decision === "rescue" ? "rescued-card" : ""}`;
    card.innerHTML = `
      ${renderThumb(result)}
      <div class="card-body">
        <div class="card-title">
          <strong>${escapeHtml(result.prompt_id || "无 prompt_id")} · v${escapeHtml(result.variant_index || 1)}</strong>
          <span class="status-pill ${escapeHtml(result.status || "")}">${escapeHtml(result.status || "")}</span>
        </div>
        ${filteredOut ? `<div class="status-pill failed">AI 已筛掉</div>` : ""}
        ${decision ? `<div class="status-pill ${decision === "rescue" ? "success" : ""}">${decision === "rescue" ? "人工救回" : `已 ${escapeHtml(decision)}`}</div>` : ""}
        <div class="meta">${escapeHtml(result.image_result_id || "")}</div>
        <div class="meta">Provider ${escapeHtml(result.provider || "")} / ${escapeHtml(result.model || "")}</div>
        <div class="review">
          <div class="review-score">
            <span>审美 ${result.aesthetic_score ?? "-"}</span>
            <span>${escapeHtml(result.rating || "")}</span>
          </div>
          ${filteredOut ? `<div class="reject-reason">原因：${escapeHtml(reasons.join("；") || result.auto_reject_reason || "自动审核未通过")}</div>` : ""}
          <div class="muted">请同时判断：这张图是否保持了本轮主体一致性。</div>
          <div class="muted">${escapeHtml(result.prompt_preview || result.reason || result.error_message || "")}</div>
          <div class="actions-grid">
            ${actions
              .map(([value, label]) => {
                const source = filteredOut && value === "rescue" ? `${sourcePrefix}filtered_out_rescue` : `${sourcePrefix}included_review`;
                return `<button class="${decision === value ? "saved" : ""}" data-gallery-human="${value}" data-source="${source}" data-image="${escapeHtml(result.image_result_id)}" data-prompt="${escapeHtml(result.prompt_id || "")}">${decision === value ? "已保存 " : ""}${label}</button>`;
              })
              .join("")}
          </div>
        </div>
      </div>
    `;
    grid.appendChild(card);
  }
  grid.querySelectorAll("[data-gallery-human]").forEach((button) => {
    button.addEventListener("click", async () => {
      await submitHumanReview(button.dataset.image, button.dataset.prompt, button.dataset.galleryHuman, button, button.dataset.source || "");
      if (onDecisionSaved) onDecisionSaved();
    });
  });
  if (summaryId) {
    const stats = reviewGalleryStats(images);
    $(summaryId).textContent = mode === "generation"
      ? `当前人工选择 ${stats.selectedCount} 张图片，包含 rescue ${stats.rescue} 张。生成工作台暂不自动写入新 Prompt 包。`
      : `最终将写入 Prompt 包：${stats.finalPromptCount} 条 Prompt，其中人工救回：${stats.rescue} 条`;
  }
}

function reviewGalleryStats(images) {
  const counts = { keep: 0, high_potential: 0, rescue: 0, revise: 0, reject: 0 };
  const finalPromptIds = new Set();
  let scoreTotal = 0;
  let scored = 0;
  for (const image of images) {
    const decision = state.humanDecisions.get(image.image_result_id) || image.human_decision || "";
    if (counts[decision] !== undefined) counts[decision] += 1;
    if (["keep", "high_potential", "rescue"].includes(decision) || ($("includeRevise")?.checked && decision === "revise")) {
      if (image.prompt_id) finalPromptIds.add(image.prompt_id);
    }
    const score = Number(image.aesthetic_score || 0);
    if (score) {
      scoreTotal += score;
      scored += 1;
    }
  }
  return {
    ...counts,
    filteredTotal: images.filter((item) => item.auto_group === "filtered_out").length,
    selectedCount: counts.keep + counts.high_potential + counts.rescue + counts.revise + counts.reject,
    pass: counts.keep + counts.high_potential + counts.rescue,
    avg: scored ? Math.round(scoreTotal / scored) : 0,
    finalPromptCount: finalPromptIds.size,
  };
}

function selectionPayload() {
  const mode = $("selectionMode").value;
  const value = $("selectionValue").value.trim();
  if (mode === "all") return { mode: "all" };
  if (mode === "range") {
    const parts = value.split("-").map((item) => item.trim()).filter(Boolean);
    return { mode: "range", start: parts[0] || "", end: parts[1] || parts[0] || "" };
  }
  if (mode === "list") {
    return { mode: "list", prompt_ids: value.split(",").map((item) => item.trim()).filter(Boolean) };
  }
  return { mode: "first_n", n: Number(value || 3) || 3 };
}

async function runPromptPack(confirmRun = false) {
  setProgress({ status: "running", run_id: state.currentRunId || "--", total: 0, completed: 0, success: 0, failed: 0, percent: 4, message: "正在提交生成任务..." });
  const body = {
    request_id: $("requestSelect").value,
    pack_id: $("packSelect").value,
    provider_profile_path: $("profileSelect").value,
    prompt_selection: selectionPayload(),
    images_per_prompt: Number($("imagesPerPrompt").value || 1) || 1,
    reference_image_path: state.selectedReferenceImagePath || "",
    subject_lock: {
      enabled: $("subjectLockEnabled").checked,
      subject_description: $("subjectLockDescription").value.trim(),
      reference_image_path: state.selectedReferenceImagePath || "",
    },
    confirm: confirmRun,
  };
  const result = await api("/api/run-prompt-pack", { method: "POST", body: JSON.stringify(body) });
  if (result.require_confirm) {
    if (window.confirm(`当前将生成 ${result.total_task_count || result.selected_item_count || "多"} 张图片，可能产生费用，确认继续？`)) {
      return runPromptPack(true);
    }
    setProgress({ status: "idle", total: 0, completed: 0, success: 0, failed: 0, percent: 0, message: "已取消运行。" });
    return;
  }
  showLastRun(result);
  state.currentRunId = result.run_id || "";
  setProgress(result.run_status || { ...result.summary, run_id: state.currentRunId, status: "completed", message: "生成完成" });
  startProgressPolling(state.currentRunId);
  await refreshResultsAndStatus();
  await loadGenerationReviewGallery();
}

function startProgressPolling(runId) {
  if (!runId) return;
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  const poll = async () => {
    const status = await api(`/api/run-status/${encodeURIComponent(runId)}`);
    setProgress(status);
    if (["completed", "failed"].includes(status.status)) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      await refreshResultsAndStatus();
    }
  };
  state.pollTimer = window.setInterval(() => poll().catch(console.warn), 1000);
  poll().catch(console.warn);
}

function setProgress(status) {
  const card = $("progressCard");
  const normalized = status.status || "idle";
  card.className = `progress-card ${normalized}`;
  $("heroStatus").textContent = normalized;
  $("heroStatus").className = `status-pill ${normalized === "completed" ? "success" : normalized}`;
  $("progressTitle").textContent = status.message || (normalized === "completed" ? "生成完成" : normalized === "running" ? "正在生成" : "等待运行");
  $("progressRunId").textContent = status.run_id || "--";
  $("progressPercent").textContent = Math.max(0, Math.min(100, Number(status.percent || 0)));
  $("progressBar").style.width = `${Math.max(0, Math.min(100, Number(status.percent || 0)))}%`;
  $("progressCompleted").textContent = `${status.completed || 0}/${status.total || 0}`;
  $("progressSuccess").textContent = status.success || 0;
  $("progressFailed").textContent = status.failed || 0;
  $("progressPrompt").textContent = status.current_prompt_id
    ? `${status.current_prompt_id} · v${status.current_variant_index || 1}/${status.images_per_prompt || 1}`
    : "--";
  $("progressLog").textContent = `[${new Date().toLocaleTimeString()}] ${status.message || "idle"}`;
  renderProgressTicks(status);
}

function renderProgressTicks(status) {
  const box = $("progressTicks");
  box.innerHTML = "";
  const items = status.items || [];
  if (items.length) {
    for (const item of items) {
      const tick = document.createElement("span");
      tick.className = `tick ${item.status === "success" ? "done" : item.status === "failed" ? "failed" : "running"}`;
      tick.textContent = `${item.prompt_id || "?"} v${item.variant_index || 1}`;
      box.appendChild(tick);
    }
    return;
  }
  const total = Math.min(Number(status.total || 0), 60);
  const completed = Number(status.completed || 0);
  const failed = Number(status.failed || 0);
  for (let index = 0; index < total; index += 1) {
    const tick = document.createElement("span");
    tick.className = `tick ${index < completed ? "done" : index === completed && status.status === "running" ? "running" : ""}`;
    if (failed && index >= Math.max(0, completed - failed) && index < completed) tick.className = "tick failed";
    tick.textContent = String(index + 1).padStart(2, "0");
    box.appendChild(tick);
  }
}

async function autoReview() {
  const button = $("autoReviewBtn");
  const oldText = button.textContent;
  button.textContent = "正在自动审核...";
  try {
    const result = await api("/api/auto-review", { method: "POST", body: JSON.stringify({ pack_id: $("packSelect").value }) });
    showLastRun(result);
    await refreshResultsAndStatus();
    await loadGenerationReviewGallery();
  } finally {
    button.textContent = oldText;
  }
}

async function loadGenerationReviewGallery() {
  const query = state.currentRunId ? `run_id=${encodeURIComponent(state.currentRunId)}` : "latest=24";
  const payload = await api(`/api/reviewed-images?${query}`);
  state.generationReviewedImages = payload.images || [];
  renderReviewGallery({
    containerId: "generationReviewGrid",
    images: state.generationReviewedImages,
    mode: "generation",
    activeGroup: state.generationReviewGroup,
    countIds: { included: "generationIncludedCount", filtered: "generationFilteredCount" },
    summaryId: "generationReviewSummary",
    onDecisionSaved: () => loadGenerationReviewGallery().catch(showError),
  });
  $("generationReviewStatus").textContent = state.currentRunId ? state.currentRunId : "latest";
}

async function submitHumanReview(imageResultId, promptId, decision, button, source = "") {
  const scores = { keep: 5, high_potential: 4, rescue: 4, revise: 3, reject: 1 };
  const result = await api("/api/human-review", {
    method: "POST",
    body: JSON.stringify({
      image_result_id: imageResultId,
      prompt_id: promptId,
      human_decision: decision,
      human_score: scores[decision] || 3,
      reject_reasons: [],
      notes: "web console quick mark",
      source: source || (decision === "rescue" ? "filtered_out_rescue" : "included_review"),
    }),
  });
  state.humanDecisions.set(imageResultId, decision);
  button.classList.add("saved");
  button.textContent = `已保存 ${decision}`;
  showLastRun(result);
}

function switchWorkspace(name) {
  const incubate = name === "incubate";
  const settings = name === "settings";
  document.querySelector(".content").classList.toggle("incubation-mode", incubate);
  document.querySelector(".content").classList.toggle("settings-mode", settings);
  document.querySelectorAll(".generation-view").forEach((node) => node.classList.toggle("hidden", incubate || settings));
  $("incubationWorkspace").classList.toggle("hidden", !incubate);
  $("settingsWorkspace").classList.toggle("hidden", !settings);
  $("workspaceGenerateBtn").classList.toggle("active", !incubate);
  $("workspaceGenerateBtn").classList.toggle("active", !incubate && !settings);
  $("workspaceIncubateBtn").classList.toggle("active", incubate);
  $("workspaceSettingsBtn").classList.toggle("active", settings);
}

function setIncProgress(status, percent = 0, message = "") {
  $("incubationStatus").textContent = status;
  $("incProgressTitle").textContent = message || status;
  $("incProgressPercent").textContent = percent;
  $("incProgressBar").style.width = `${percent}%`;
  $("incProgressLog").textContent = `[${new Date().toLocaleTimeString()}] ${message || status}`;
  $("incubationProgress").className = `progress-card compact ${status}`;
}

function buildIncubationRequestPayload() {
  const title = $("incTitle").value.trim();
  const description = $("incDescription").value.trim();
  const ratio = $("incRatio").value || "3:4";
  const profile = selectedPromptTypeProfile();
  return {
    prompt_type: profile.type_id || "portrait",
    title,
    description,
    requirements: description,
    domain: profile.domain || "",
    platform: profile.platform || "general",
    use_case: profile.use_case || "",
    style: profile.style || "",
    review_profile: profile.review_profile || "",
    ratio,
    topic: title || profile.label || "主题",
  };
}

function optionalPositiveNumber(elementId) {
  const raw = ($(elementId).value || "").trim();
  if (!raw) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

async function createIncubationRequest() {
  setIncProgress("running", 10, "正在保存需求...");
  const payload = buildIncubationRequestPayload();
  const result = await api("/api/idea-requests", { method: "POST", body: JSON.stringify(payload) });
  state.incubation.requestId = result.request_id;
  showLastRun(result);
  setIncProgress("completed", 100, `需求已创建：${result.request_id}`);
  await refreshAll();
}

async function uploadIncubationReferences() {
  if (!state.incubation.requestId) throw new Error("请先新建需求");
  const files = $("incReferenceFiles").files;
  if (!files.length) throw new Error("请选择样图或 zip");
  setIncProgress("running", 20, "正在上传样图...");
  const form = new FormData();
  form.append("request_id", state.incubation.requestId);
  for (const file of files) form.append("files", file);
  const result = await api("/api/reference-images/upload", { method: "POST", body: form });
  state.incubation.referenceImages = result.reference_images || [];
  renderIncReferenceList();
  showLastRun(result);
  setIncProgress("completed", 100, `已上传 ${result.uploaded_count || 0} 张样图`);
}

function renderIncReferenceList() {
  $("incReferenceList").innerHTML = state.incubation.referenceImages
    .map((image) => `<img src="${escapeHtml(image.url || image.file_url || "")}" title="${escapeHtml(image.filename || image.file_name || "")}" />`)
    .join("");
}

async function extractIncubationPrompts() {
  if (!state.incubation.requestId) throw new Error("请先新建需求");
  if (!state.incubation.referenceImages.length) throw new Error("请先上传样图");
  setIncProgress("running", 35, "正在提取样图 Prompt 结构...");
  const result = await api("/api/extract-image-prompts", {
    method: "POST",
    body: JSON.stringify({
      request_id: state.incubation.requestId,
      reference_image_ids: state.incubation.referenceImages.map((item) => item.reference_image_id),
    }),
  });
  state.incubation.extractRunId = result.extract_run_id;
  $("extractSummary").textContent = `提取完成：样图 ${result.summary.reference_image_count} 张，Skeleton ${result.summary.prompt_skeleton_count} 条`;
  showLastRun(result);
  setIncProgress("completed", 100, "Prompt 结构提取完成");
}

async function generateIncubationCandidates() {
  if (!state.incubation.extractRunId) throw new Error("请先提取 Prompt 结构");
  setIncProgress("running", 45, "正在生成候选 Prompt...");
  const result = await api("/api/generate-candidate-prompts", {
    method: "POST",
    body: JSON.stringify({
      request_id: state.incubation.requestId,
      extract_run_id: state.incubation.extractRunId,
      candidate_count: optionalPositiveNumber("candidateCount"),
    }),
  });
  state.incubation.candidatePromptIds = result.candidate_prompt_ids || [];
  $("candidatePreview").innerHTML = (result.previews || [])
    .slice(0, 8)
    .map((item) => `<div><b>${escapeHtml(item.prompt_id)}</b> ${escapeHtml(item.preview)}</div>`)
    .join("");
  showLastRun(result);
  setIncProgress("completed", 100, `已生成 ${result.candidate_prompt_count} 条候选 Prompt`);
}

async function validateIncubationCandidates() {
  if (!state.incubation.candidatePromptIds.length) throw new Error("请先生成候选 Prompt");
  setIncProgress("running", 55, "正在验证候选 Prompt...");
  const result = await api("/api/validate-candidate-prompts", {
    method: "POST",
    body: JSON.stringify({
      request_id: state.incubation.requestId,
      candidate_prompt_ids: state.incubation.candidatePromptIds,
      provider_profile_path: $("incProfileSelect").value,
      images_per_prompt: optionalPositiveNumber("incImagesPerPrompt"),
      reference_image_path: state.incubation.referenceImages[0]?.path || "",
    }),
  });
  state.incubation.validationRunId = result.run_id;
  showLastRun(result);
  setProgress(result.run_status || { status: "completed", percent: 100, message: "验证完成" });
  startProgressPolling(result.run_id);
  setIncProgress("completed", 100, "候选 Prompt 验证完成，请进行自动审核");
}

async function autoReviewIncubationResults() {
  if (!state.incubation.requestId) throw new Error("缺少 request");
  setIncProgress("running", 70, "正在自动审核验证图片...");
  const result = await api("/api/auto-review", { method: "POST", body: JSON.stringify({ request_id: state.incubation.requestId }) });
  $("incReviewSummary").textContent = `自动审核完成：${result.review_count || 0} 条`;
  showLastRun(result);
  await refreshResultsAndStatus();
  renderIncubationReviewGrid();
  $("incubationReviewGrid").classList.remove("hidden");
  state.incubation.reviewed = true;
  setIncProgress("completed", 100, "自动审核完成，请进行人工审核");
}

function renderIncubationReviewGrid() {
  const ids = new Set(state.incubation.candidatePromptIds);
  const oldResults = state.imageResults;
  state.imageResults = oldResults.filter((item) => ids.has(item.prompt_id));
  renderImageResultsInto($("incubationReviewGrid"));
  state.imageResults = oldResults;
}

function renderImageResultsInto(grid) {
  const original = $("imageGrid");
  const oldId = original.id;
  original.id = "__imageGridTemp";
  grid.id = "imageGrid";
  renderImageResults();
  grid.id = "";
  original.id = oldId;
}

async function buildIncubatedPack() {
  if (!state.incubation.reviewed) throw new Error("请先完成孵化和人工审核");
  setIncProgress("running", 85, "正在生成新 Prompt 包...");
  const accepted = ["keep"];
  if ($("includeHighPotential").checked) accepted.push("high_potential");
  accepted.push("rescue");
  if ($("includeRevise").checked) accepted.push("revise");
  const result = await api("/api/build-pack-from-incubation", {
    method: "POST",
    body: JSON.stringify({
      incubation_run_id: state.incubation.incubationRunId,
      min_aesthetic_score: Number($("incMinScore").value || 70),
      accepted_decisions: accepted,
      include_revise: $("includeRevise").checked,
      include_unmarked_high_score: false,
      pack_name: $("incPackName").value,
      status: "candidate",
    }),
  });
  $("buildPackSummary").textContent = `新 Prompt 包已生成：${result.built_pack_id}，其中包含人工救回 Prompt：${result.rescued_prompt_count || 0} 条`;
  showLastRun(result);
  setIncProgress("completed", 100, `已生成 Prompt 包 ${result.built_pack_id}`);
  await refreshAll();
}

async function startIncubation() {
  setIncProgress("running", 5, "正在准备孵化输入...");
  const files = $("incReferenceFiles").files;
  const requestPayload = buildIncubationRequestPayload();
  if (!state.incubation.requestId) {
    const created = await api("/api/idea-requests", { method: "POST", body: JSON.stringify(requestPayload) });
    state.incubation.requestId = created.request_id;
  }
  if (files.length) {
    const form = new FormData();
    form.append("request_id", state.incubation.requestId);
    for (const file of files) form.append("files", file);
    const upload = await api("/api/reference-images/upload", { method: "POST", body: form });
    state.incubation.referenceImages = upload.reference_images || [];
    renderIncReferenceList();
  }
  if (selectedPromptTypeProfile().requires_reference_images && !state.incubation.referenceImages.length) {
    throw new Error("当前 Prompt 类型需要上传样图。");
  }
  const result = await api("/api/incubate-prompt-pack", {
    method: "POST",
    body: JSON.stringify({
      request_id: state.incubation.requestId,
      reference_image_ids: state.incubation.referenceImages.map((item) => item.reference_image_id),
      provider_profile_path: $("incProfileSelect").value,
      candidate_count: optionalPositiveNumber("candidateCount") || selectedPromptTypeProfile().candidate_count_default,
      images_per_prompt: optionalPositiveNumber("incImagesPerPrompt") || selectedPromptTypeProfile().images_per_prompt_default,
      subject_lock: {
        enabled: $("incSubjectLockEnabled").checked,
        subject_description: $("incSubjectLockDescription").value.trim() || $("incDescription").value.trim(),
      },
    }),
  });
  state.incubation.incubationRunId = result.incubation_run_id;
  state.incubation.reviewed = false;
  state.incubation.reviewedImages = [];
  state.incubation.activeReviewGroup = "included";
  $("incubationReviewTabs").classList.add("hidden");
  $("incubationReviewGrid").classList.add("hidden");
  showLastRun(result);
  pollIncubationStatus(result.incubation_run_id);
}

function pollIncubationStatus(incubationRunId) {
  const poll = async () => {
    const status = await api(`/api/incubation-status/${encodeURIComponent(incubationRunId)}`);
    setIncProgress(status.stage || status.status, Number(status.percent || 0), status.message || "");
    if (status.validation_run_id) state.incubation.validationRunId = status.validation_run_id;
    if (status.status === "waiting_human_review") {
      await loadIncubationReviewedImages();
      return;
    }
    if (status.status === "failed" || status.status === "completed") return;
    window.setTimeout(() => poll().catch(showError), 1000);
  };
  poll().catch(showError);
}

async function loadIncubationReviewedImages() {
  const payload = await api(`/api/incubation-reviewed-images/${encodeURIComponent(state.incubation.incubationRunId)}`);
  state.incubation.reviewedImages = payload.images || [];
  state.incubation.reviewed = true;
  state.incubation.activeReviewGroup = "included";
  renderIncubationReviewedImages();
  updateFeedbackPanel();
  $("incubationReviewTabs").classList.remove("hidden");
  $("incubationReviewGrid").classList.remove("hidden");
}

function renderIncubationReviewedImages() {
  renderReviewGallery({
    containerId: "incubationReviewGrid",
    images: state.incubation.reviewedImages,
    mode: "incubation",
    activeGroup: state.incubation.activeReviewGroup,
    countIds: { included: "includedCount", filtered: "filteredCount" },
    summaryId: "buildPackSummary",
    onDecisionSaved: () => {
      updateFeedbackPanel();
      renderIncubationReviewedImages();
    },
  });
}

function updateFeedbackPanel() {
  const counts = reviewGalleryStats(state.incubation.reviewedImages);
  const total = state.incubation.reviewedImages.length || 1;
  $("fbKeep").textContent = counts.keep;
  $("fbHigh").textContent = counts.high_potential;
  $("fbRevise").textContent = counts.revise;
  $("fbReject").textContent = counts.reject;
  $("fbFiltered").textContent = counts.filteredTotal;
  $("fbRescue").textContent = counts.rescue;
  $("fbAvg").textContent = counts.avg;
  $("fbPassRate").textContent = `${Math.round((counts.pass / total) * 100)}%`;
  $("fbFinal").textContent = counts.finalPromptCount;
  $("buildPackSummary").textContent = `最终将写入 Prompt 包：${counts.finalPromptCount} 条 Prompt，其中人工救回：${counts.rescue} 条`;
}

async function deletePack(packId) {
  if (!window.confirm("确认删除该 Prompt 包及其 prompt items？历史 image_results / reviews 会保留。")) return;
  const result = await api(`/api/prompt-packs/${encodeURIComponent(packId)}?confirm=true`, { method: "DELETE" });
  showLastRun(result);
  await refreshAll();
}

async function uploadReferenceImage() {
  const file = $("referenceFile").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("image", file);
  const result = await api("/api/reference-images/upload", { method: "POST", body: form });
  showLastRun(result);
  state.selectedReferenceImagePath = result.reference_image?.image_path || "";
  await refreshReferenceImages();
}

async function refreshReferenceImages() {
  const refs = await api("/api/reference-images");
  state.referenceImages = refs.reference_images || [];
  renderReferenceImages();
}

async function refreshResultsAndStatus() {
  const [status, results, reviews] = await Promise.all([
    api("/api/status"),
    api("/api/image-results?latest=24"),
    api("/api/image-reviews?latest=100"),
  ]);
  state.status = status;
  state.imageResults = results.image_results || [];
  state.imageReviews = reviews.image_reviews || [];
  renderStatus();
  renderImageResults();
}

async function saveImageSettings() {
  const result = await api("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      image_provider: $("settingsImageProvider").value,
      image_provider_profile_path: $("settingsProfileSelect").value,
      image_model: $("settingsImageModel").value,
    }),
  });
  showLastRun(result);
  await refreshAll();
}

async function saveReviewSettings() {
  const result = await api("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      aesthetic_review_provider: $("settingsReviewProvider").value,
      aesthetic_review_api_url: $("settingsReviewUrl").value,
      aesthetic_review_image_field: $("settingsReviewImageField").value || "image",
      aesthetic_review_score_field: $("settingsReviewScoreField").value || "displayScore",
      aesthetic_review_rating_field: $("settingsReviewRatingField").value || "rating",
      aesthetic_review_reason_field: $("settingsReviewReasonField").value || "analysis",
    }),
  });
  showLastRun(result);
  await refreshAll();
}

async function saveImageKey() {
  const provider = $("settingsImageProvider").value;
  const keyName = provider === "wuyin_async" ? "WUYIN_API_KEY" : provider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY";
  const value = $("settingsImageKey").value.trim();
  if (!value) throw new Error("Key 为空时不会覆盖已有配置");
  const result = await api("/api/settings/secrets", { method: "POST", body: JSON.stringify({ [keyName]: value }) });
  $("settingsImageKey").value = "";
  showLastRun(result);
  await refreshAll();
}

async function saveTasteLensKey() {
  const value = $("settingsTasteLensKey").value.trim();
  if (!value) throw new Error("Key 为空时不会覆盖已有配置");
  const result = await api("/api/settings/secrets", { method: "POST", body: JSON.stringify({ TASTELENS_API_KEY: value }) });
  $("settingsTasteLensKey").value = "";
  showLastRun(result);
  await refreshAll();
}

async function clearImageKey() {
  const provider = $("settingsImageProvider").value;
  const keyName = provider === "wuyin_async" ? "WUYIN_API_KEY" : provider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY";
  const result = await api(`/api/settings/secrets/${encodeURIComponent(keyName)}`, { method: "DELETE" });
  showLastRun(result);
  await refreshAll();
}

async function checkPorts() {
  const result = await api("/api/ports/check");
  $("portsResult").textContent = JSON.stringify(result, null, 2);
  showLastRun(result);
}

async function runDiagnostics(options = {}) {
  const result = await api("/api/diagnostics/run", { method: "POST", body: JSON.stringify(options) });
  $("diagnosticsSummary").textContent = result.overall_status || "unknown";
  $("diagnosticsList").innerHTML = (result.checks || [])
    .map((check) => `
      <div class="diagnostic-row ${escapeHtml(check.status)}">
        <strong>${escapeHtml(check.name)}</strong>
        <span class="status-pill">${escapeHtml(check.status)}</span>
        <div>${escapeHtml(check.message || "")}</div>
        <div class="muted">${escapeHtml(check.suggestion || "")}</div>
      </div>
    `)
    .join("");
  showLastRun(result);
}

function showLastRun(payload) {
  $("lastRunJson").textContent = JSON.stringify(payload, null, 2);
}

function toFileUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `/files/${encodeURIComponent(String(path).replaceAll("\\", "/")).replaceAll("%2F", "/")}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

$("refreshBtn").addEventListener("click", () => refreshAll().catch(showError));
$("refreshImagesBtn").addEventListener("click", () => refreshResultsAndStatus().catch(showError));
$("runBtn").addEventListener("click", () => runPromptPack(false).catch(showError));
$("autoReviewBtn").addEventListener("click", () => autoReview().catch(showError));
$("uploadReferenceBtn").addEventListener("click", () => uploadReferenceImage().catch(showError));
$("clearReferenceBtn").addEventListener("click", () => {
  state.selectedReferenceImagePath = "";
  $("referenceSelect").value = "";
  updateReferencePreview();
});
$("referenceSelect").addEventListener("change", () => {
  state.selectedReferenceImagePath = $("referenceSelect").value;
  updateReferencePreview();
});
$("profileSelect").addEventListener("change", renderStatus);
$("selectionMode").addEventListener("change", () => {
  const placeholders = {
    first_n: "3",
    range: "P001-P003",
    list: "P001,P003,P007",
    all: "",
  };
  $("selectionValue").value = placeholders[$("selectionMode").value] || "";
});
$("workspaceGenerateBtn").addEventListener("click", () => switchWorkspace("generate"));
$("workspaceIncubateBtn").addEventListener("click", () => switchWorkspace("incubate"));
$("workspaceSettingsBtn").addEventListener("click", () => switchWorkspace("settings"));
$("settingsImageProvider").addEventListener("change", () => updateSecretStatus(state.settings.secrets || {}));
$("saveImageSettingsBtn").addEventListener("click", () => saveImageSettings().catch(showError));
$("saveImageKeyBtn").addEventListener("click", () => saveImageKey().catch(showError));
$("clearImageKeyBtn").addEventListener("click", () => clearImageKey().catch(showError));
$("saveReviewSettingsBtn").addEventListener("click", () => saveReviewSettings().catch(showError));
$("saveTasteLensKeyBtn").addEventListener("click", () => saveTasteLensKey().catch(showError));
$("checkPortsBtn").addEventListener("click", () => checkPorts().catch(showError));
$("quickDiagnosticsBtn").addEventListener("click", () => runDiagnostics({ include_real_image_test: false, include_aesthetic_test: false }).catch(showError));
$("aestheticDiagnosticsBtn").addEventListener("click", () => runDiagnostics({ include_real_image_test: false, include_aesthetic_test: true }).catch(showError));
$("realImageDiagnosticsBtn").addEventListener("click", () => {
  if (!window.confirm("真实生图自检可能消耗额度，确认继续？")) return;
  runDiagnostics({ include_real_image_test: true, include_aesthetic_test: true }).catch(showError);
});
$("promptTypeSelect").addEventListener("change", applyPromptTypeProfile);
document.querySelectorAll("[data-review-group]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.gallery === "generation") {
      state.generationReviewGroup = button.dataset.reviewGroup || "included";
      loadGenerationReviewGallery().catch(showError);
      return;
    }
    state.incubation.activeReviewGroup = button.dataset.reviewGroup || "included";
    renderIncubationReviewedImages();
  });
});
$("startIncubationBtn").addEventListener("click", () => startIncubation().catch(showError));
$("buildIncPackBtn").addEventListener("click", () => buildIncubatedPack().catch(showError));
$("includeRevise").addEventListener("change", updateFeedbackPanel);

function showError(error) {
  console.error(error);
  showLastRun({ ok: false, error: error.message || String(error) });
}

setProgress({ status: "idle", total: 0, completed: 0, success: 0, failed: 0, percent: 0, message: "尚未开始。" });
refreshAll().catch(showError);
