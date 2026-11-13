"use strict";

// Browser storage holds display conveniences. Session generations and history
// are reconstructed from Agent whenever a session is opened or a turn completes.
const STORAGE_KEY = "agent.web.presentation.v1";
const RECENT_SESSIONS_KEY = "agent.web.recent-sessions.v1";
const RESOURCE_STORAGE_KEY = "agent.web.selected-resource.v1";
const MULTI_RESOURCE_STORAGE_KEY = "agent.web.selected-resources.v1";
const MAX_SELECTED_RESOURCES = 8;
const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATES = new Set([
  "answered", "clarification", "navigated", "succeeded", "failed", "cancelled",
  "interrupted", "unavailable", "stale", "activated", "planned",
]);
const element = (id) => document.getElementById(id);
const state = {
  session: null, models: [], activeTurn: null, submission: null, initializing: true,
  durableTurnIds: new Set(),
  posting: false, opening: false, timer: null, epoch: 0, pollEpoch: 0,
  viewedRevisionId: null, revision: null, revisionEpoch: 0,
  evidenceEpoch: 0, evidence: null, artifactUrl: null, artifactEpoch: 0,
  attachments: [], attachmentSession: null, restoringAttachments: false,
  resources: [], resourcesEnabled: false, uploadMaxBytes: null, upload: null,
  selectedResourceIds: [],
  parameterEditor: null, requestOptions: null, optionsEpoch: 0,
  fastqMembers: [], fastqCompletion: null,
  inputSets: [], epizooResources: [],
  scientificFiles: new Map(),
  artifactPreview: false,
};

class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

function readConveniences() {
  try {
    const value = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "{}");
    return value && typeof value === "object" ? value : {};
  } catch (_) {
    return {};
  }
}

function recentSessions() {
  try {
    const rows = JSON.parse(localStorage.getItem(RECENT_SESSIONS_KEY) || "[]");
    return Array.isArray(rows) ? rows.filter((r) => r && typeof r.id === "string"
      && r.id.length > 0 && r.id.length <= 256 && typeof r.label === "string").slice(0, 30) : [];
  } catch (_) { return []; }
}

function rememberSession(view) {
  const rows = recentSessions();
  const first = view.turns.find((t) => t.utterance);
  const previous = rows.find((r) => r.id === view.session_id);
  const row = {id: view.session_id, label: previous && previous.label !== "New conversation"
    ? previous.label : first ? first.utterance.slice(0, 80) : "New conversation"};
  const updated = [row, ...rows.filter((r) => r.id !== row.id)].slice(0, 30);
  try { localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(updated)); } catch (_) {}
  const select = element("recent-sessions");
  select.replaceChildren(...updated.map((r) => {
    const option = document.createElement("option");
    option.value = r.id; option.textContent = r.label; option.title = r.id;
    return option;
  }));
  select.value = view.session_id;
}

function saveConveniences() {
  if (state.initializing) return;
  const value = {
    sessionId: state.session ? state.session.session_id : "",
    profileId: element("model-choice").value,
    polledTurnId: state.activeTurn ? state.activeTurn.turn_id : "",
    draft: element("utterance").value,
    inputSetId: element("input-choice").value,
    epizooResourceId: element("epizoo-resource-choice").value,
    viewedRevisionId: state.session && state.viewedRevisionId !== state.session.active_revision_id ? state.viewedRevisionId : null,
    workspacePanel: element("research-layout").dataset.panel,
  };
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(value)); } catch (_) {}
}

async function api(path, method = "GET", body = null) {
  let response;
  try {
    response = await fetch(`/api/v1${path}`, {
      method, credentials: "same-origin", cache: "no-store",
      headers: body === null ? {} : { "Content-Type": "application/json" },
      body: body === null ? undefined : JSON.stringify(body),
    });
  } catch (_) {
    throw new ApiError("CONNECTION_UNAVAILABLE", "Agent could not be reached. Your submitted turn ID is retained for polling or an exact retry.");
  }
  let data;
  try { data = await response.json(); } catch (_) {
    throw new ApiError("RESPONSE_UNAVAILABLE", "Agent returned an unreadable response.", response.status);
  }
  if (!response.ok) {
    const error = data && data.error;
    throw new ApiError(error && error.code || "HTTP_ERROR",
      error && error.message || "Agent could not complete this request.", response.status);
  }
  return data;
}

function sessionPath(sessionId) {
  return `/sessions/${encodeURIComponent(sessionId)}`;
}

function turnPath(sessionId, turnId) {
  return `${sessionPath(sessionId)}/turns/${encodeURIComponent(turnId)}`;
}

function revisionPath(sessionId, revisionId) {
  return `${sessionPath(sessionId)}/revisions/${encodeURIComponent(revisionId)}`;
}

function showError(error) {
  const banner = element("client-error");
  banner.textContent = error ? `${error.code || "CLIENT_ERROR"}: ${error.message}` : "";
  banner.hidden = !error;
}

function terminal(turn) {
  return Boolean(turn.response) || TERMINAL_STATES.has(turn.status);
}

function checkpointText(turn) {
  return `Turn ${turn.turn_id} · ${turn.status}${turn.profile_id ? ` · ${turn.profile_id}` : ""}`;
}

function shortIdentity(value) {
  // Compact labels are display only; requests and expanded metadata keep full IDs.
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-4)}` : value;
}

function messageNode(label, text, className) {
  const message = document.createElement("div");
  message.className = `message ${className}`;
  const title = document.createElement("div");
  title.className = "message-label";
  title.textContent = label;
  const content = document.createElement("p");
  content.className = "message-text";
  // Persisted assistant wording is displayed exactly, without HTML or Markdown
  // interpretation. Neither historical text nor browser state is evidence.
  content.textContent = text;
  message.append(title, content);
  return message;
}

function guidanceNode(guidance) {
  const section = document.createElement("section");
  section.className = "guidance";
  const heading = document.createElement("h3");
  heading.textContent = "Proposed next steps";
  section.append(heading);
  for (const candidate of guidance.candidates) {
    const card = document.createElement("div");
    card.className = "guidance-candidate";
    card.dataset.candidateId = candidate.candidate_id;
    const title = document.createElement("h3");
    title.textContent = readableLabel(candidate.capability);
    const rationale = document.createElement("p");
    rationale.textContent = candidate.text;
    const metadata = document.createElement("span");
    metadata.className = "candidate-metadata";
    metadata.textContent = `Support: ${candidate.support} · Candidate ${candidate.candidate_id} · Origin turn ${candidate.origin_turn_id}`;
    card.append(title, rationale);
    const context = document.createElement("p");
    context.className = "muted small";
    context.textContent = candidate.base_revision_id ? `Captured revision: ${candidate.base_revision_id}` : "Captured revision metadata was not stored in this historical display.";
    const diagnostics = document.createElement("details");
    diagnostics.dataset.detailKey = `candidate:${candidate.candidate_id}`;
    const diagnosticsLabel = document.createElement("summary");
    diagnosticsLabel.textContent = "Scientific details";
    diagnosticsLabel.dataset.actionKey = `candidate-details:${candidate.candidate_id}`;
    diagnostics.append(diagnosticsLabel, metadata, context, textList(candidate.limitations));
    card.append(diagnostics);
    if (candidate.readiness) {
      const readiness = document.createElement("details");
      readiness.dataset.detailKey = `readiness:${candidate.candidate_id}`;
      const readinessLabel = document.createElement("summary");
      readinessLabel.dataset.actionKey = `readiness-details:${candidate.candidate_id}`;
      readinessLabel.textContent = `Technical readiness: ${candidate.readiness.readiness}`;
      const rows = document.createElement("dl");
      rows.className = "revision-summary";
      rows.replaceChildren(...summaryRows([
        ["Registered", candidate.readiness.capability_registered],
        ["Request scope", candidate.readiness.request_scope],
        ["Evidence handles", candidate.readiness.accepted_evidence_handles.join(", ")],
        ["Required ports", candidate.readiness.required_ports_without_supplied_request_source.join(", ")],
        ["Required parameters", candidate.readiness.required_scientific_parameters.join(", ")],
        ["Explicit inputs", candidate.readiness.explicit_request_source_choices.join(", ")],
      ]));
      readiness.append(readinessLabel, rows);
      card.append(readiness);
    } else {
      const readiness = document.createElement("p");
      readiness.className = "muted small";
      readiness.textContent = "Readiness metadata was not stored for this historical candidate.";
      diagnostics.append(readiness);
    }
    const action = document.createElement("button");
    action.className = "guidance-select";
    action.dataset.actionKey = `guidance:${candidate.candidate_id}`;
    action.type = "button";
    action.textContent = "Continue with this";
    action.addEventListener("click", () => {
      submitRequest(candidate.capability, candidate.origin_turn_id, null,
        {origin_turn_id: candidate.origin_turn_id, candidate_id: candidate.candidate_id});
    });
    card.append(action);
    section.append(card);
  }
  if (guidance.limitations.length) {
    const details = document.createElement("details");
    const label = document.createElement("summary"); label.textContent = "Guidance scope";
    details.append(label, textList(guidance.limitations)); section.append(details);
  }
  return section;
}

function textList(values) {
  const list = document.createElement("ul");
  list.className = "compact-list";
  for (const value of values || []) {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  }
  return list;
}

function summaryRows(rows) {
  const nodes = [];
  for (const [label, value] of rows) {
    const name = document.createElement("dt");
    name.textContent = label;
    const content = document.createElement("dd");
    content.textContent = value === null || value === undefined ? "Unavailable" : String(value);
    nodes.push(name, content);
  }
  return nodes;
}

function formatBytes(bytes) {
  if (!Number.isSafeInteger(bytes) || bytes < 0) return "";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = bytes ? Math.min(3, Math.floor(Math.log(bytes) / Math.log(1024))) : 0;
  return `${Number((bytes / 1024 ** index).toFixed(1))} ${units[index]}`;
}

function resultLabel(revisionId) {
  const index = state.session ? state.session.revisions.findIndex((r) => r.revision_id === revisionId) : -1;
  return index < 0 ? "Result outside available history" : `${state.session.history_truncated ? "Recent result" : "Result"} ${index + 1}`;
}

function showWorkspacePanel(panel, focus = false) {
  element("research-layout").dataset.panel = panel;
  element("show-conversation").setAttribute("aria-pressed", String(panel === "conversation"));
  element("show-results").setAttribute("aria-pressed", String(panel === "results"));
  if (focus) {
    const target = element(panel === "results" ? "scientific-details" : "utterance");
    target.focus({ preventScroll: true });
  }
  saveConveniences();
}

function renderResultNavigation() {
  const hasResults = Boolean(state.session?.revisions.length);
  element("research-layout").classList.toggle("has-results", hasResults);
  element("result-navigation").hidden = !hasResults;
  element("show-results").disabled = !hasResults;
  if (!hasResults) showWorkspacePanel("conversation");
}

function showResult(revisionId) {
  showWorkspacePanel("results");
  element("artifact-panel").open = true;
  loadRevision(revisionId);
  element("scientific-details").focus({ preventScroll: true });
}

function renderRevisionHistory() {
  const revisions = state.session ? state.session.revisions : [];
  element("result-choice").disabled = !revisions.length;
  element("result-choice").replaceChildren(...revisions.map((revision) => {
    const option = document.createElement("option");
    option.value = revision.revision_id;
    option.textContent = `${resultLabel(revision.revision_id)}${revision.is_active ? " · Current" : ""}`;
    return option;
  }));
  element("result-choice").value = state.viewedRevisionId || state.session?.active_revision_id || "";

}

function clearArtifactView() {
  state.artifactEpoch += 1;
  if (state.artifactUrl) URL.revokeObjectURL(state.artifactUrl);
  state.artifactUrl = null;
  element("artifact-viewer").hidden = true;
  element("figure-description").hidden = true;
  element("figure-content").hidden = true;
  element("figure-content").removeAttribute("src");
  element("artifact-view-error").hidden = true;
}

function clearRevisionView() {
  state.revisionEpoch += 1;
  state.evidenceEpoch += 1;
  state.revision = null;
  state.evidence = null;
  state.viewedRevisionId = null;
  element("result-method").textContent = "Scientific results";
  clearArtifactView();
  element("revision-view").hidden = true;
  element("scientific-empty").hidden = false;
  element("scientific-empty").textContent = "Your accepted results will appear here. Start with a question or attach data in the conversation.";
  updateControls();
}

async function loadRevision(revisionId) {
  if (!state.session) return;
  const sessionId = state.session.session_id;
  const epoch = state.epoch;
  const revisionEpoch = ++state.revisionEpoch;
  const previousOutput = state.revision?.revision_id === revisionId ? element("evidence-output").value : null;
  if (state.viewedRevisionId !== revisionId) element("scientific-panel").scrollTop = 0;
  state.evidenceEpoch += 1;
  state.revision = null;
  state.evidence = null;
  state.viewedRevisionId = revisionId;
  state.artifactPreview = false;
  element("result-choice").value = revisionId;
  renderResultNavigation();
  saveConveniences();
  clearArtifactView();
  element("revision-view").hidden = true;
  element("scientific-empty").hidden = false;
  element("scientific-empty").textContent = "Loading accepted scientific state…";
  updateControls();
  try {
    const revision = await api(revisionPath(sessionId, revisionId));
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch) return;
    state.revision = revision;
    renderRevision(revision);
    if (previousOutput && revision.evidence_outputs.includes(previousOutput)) element("evidence-output").value = previousOutput;
    renderResultSelection();
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch) return;
    element("scientific-empty").textContent = `${error.code}: ${error.message}`;
  }
  updateControls();
}

function renderRevisionContext(revision) {
  element("result-heading").textContent = `${resultLabel(revision.revision_id)} · ${revision.is_active ? "Current" : "Earlier result"} · ${statusLabel(revision.status)}`;
  element("result-heading").title = revision.revision_id;

}

function renderRevision(revision) {
  element("scientific-empty").hidden = true;
  element("revision-view").hidden = false;
  renderRevisionContext(revision);
  element("result-operations").replaceChildren(...(revision.steps || []).map((step) => {
    const item = document.createElement("li");
    item.textContent = `${stepLabel(step)} · ${readableLabel(step.status.toLowerCase())}`;
    if (step.training_notice) item.textContent += ` — ${step.training_notice}`;
    return item;
  }));
  const groups = revision.result_groups || [];
  element("detail-comparison").replaceChildren(...[["", "Same result"],
    ...groups.map((group) => [group.canonical_output, group.label])].map(([value, label]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = label; return option;
    }));
  element("evidence-output").replaceChildren(...revision.evidence_outputs.map((output) => {
    const group = groups.find((g) => g.outputs.some((entry) => entry.name === output));
    const field = group?.outputs.find((entry) => entry.name === output)?.field;
    const option = document.createElement("option");
    option.value = output;
    option.textContent = group ? `${group.label} · ${readableLabel(field)} (${output})` : output;
    return option;
  }));
  element("evidence-output").disabled = !revision.evidence_outputs.length;
  element("load-evidence").disabled = !revision.evidence_outputs.length;
  element("result-method").textContent = "Scientific results";
  element("result-facts").textContent = "No accepted summary is available.";
  element("evidence-content").replaceChildren();
  element("detail-content").replaceChildren();
  element("detail-form").hidden = true;
  element("detail-subject").value = "";
  if (!revision.evidence_outputs.length) {
    element("evidence-content").textContent = "Accepted evidence is unavailable for these output names.";
  }
}

function artifactPath(sessionId, revisionId, handle, turnId = null) {
  const owner = revisionId === null ? turnPath(sessionId, turnId) : revisionPath(sessionId, revisionId);
  return `${owner}/artifacts/${encodeURIComponent(handle)}`;
}

function distinctFigureArtifacts(artifacts) {
  const seen = new Set();
  return artifacts.filter((artifact) => {
    const key = artifact.figure ? `${artifact.revision_id}:${artifact.figure.manifest_sha256}:${artifact.figure.figure_id}`
      : `${artifact.revision_id}:${artifact.turn_id}:${artifact.handle}`;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
}

// Display association uses the published source references, never captured base
// context, filenames, prose, cell counts or the currently active revision.
function presentationChoices() {
  if (!state.revision || !state.session) return [];
  const revisionId = state.revision.revision_id;
  return state.session.turns.flatMap((turn) => {
    const entry = state.scientificFiles.get(JSON.stringify([state.session.session_id, turn.turn_id, turn.revision_id]));
    if (!entry) return [];
    const presentations = entry.presentations.filter((artifact) => artifact.turn_id === turn.turn_id
      && artifact.revision_id === null && artifact.figure?.sources.some((source) =>
        ["coordinates", "qc"].includes(source.role) && source.output?.revision_id === revisionId));
    const artifacts = entry.artifacts.filter((artifact) => {
      if (artifact.artifact_type !== "scientific_composition" || artifact.turn_id !== turn.turn_id) return false;
      const contents = artifact.contents || {};
      return [contents.measurement, ...(contents.observations || []), ...(contents.representations || [])]
        .some((source) => source?.accepted_output?.revision_id === revisionId);
    });
    return presentations.length || artifacts.length ? [{turn, presentations, artifacts}] : [];
  });
}

function resultFigureArtifacts(group = null, presentation = null) {
  if (presentation?.presentations.length) {
    return distinctFigureArtifacts([...presentation.presentations, ...state.revision.artifacts.filter((a) => a.figure)]);
  }
  const names = group ? new Set(group.outputs.map((o) => o.name)) : null;
  const matching = state.revision.artifacts.filter((artifact) => artifact.figure
    && artifact.artifact_type === "analysis_figure"
    && (!names || artifact.figure.output_names.some((name) => names.has(name))));
  return distinctFigureArtifacts(matching);
}

function resultFigureGallery(artifacts) {
  if (!artifacts.length) return null;
  const section = document.createElement("section"); section.className = "result-figures";
  const heading = document.createElement("h3"); heading.textContent = "Scientific figures";
  section.append(heading);
  const captions = {
    qc_metrics: "Saved native QC measurements and recorded producer populations. Missing values are explicit; profiles are not recomputed after filtering. Thresholds describe the selected filtering result.",
    annotation_composition: "Native groups and predicted cell-type composition, including unassigned coverage. Candidate labels are computational annotations.",
    marker_activity_dot: "Accepted Gene Activity: color = log1p(mean Gene Activity), size = fraction activity > 0. Representative positive native markers; accessibility-derived activity, not measured RNA expression or independent validation.",
    marker_activity_heatmap: "Color = log1p(mean accepted Gene Activity). Representative positive native markers; accessibility-derived activity, not measured RNA expression. Bounded groups and genes are named on the plot.",
    annotation_hierarchy: "Accepted branch relationships and actual stopping outcomes. Computational subdivisions are not independently validated cell types.",
    marker_activity_umap: "Color = log1p(Gene Activity), without additional normalization. All cells use the same accepted UMAP. Accessibility-derived activity is not measured RNA expression. Native markers can help inspect candidate annotations and within-cluster heterogeneity; they are dependent evidence, not independent biological validation.",
    annotation_umap: "Accepted UMAP colored by hierarchical annotations using exact original-source ancestry and ordered cells. Gray = unassigned. Coordinates were reused.",
    umap_leiden: "UMAP colored by accepted native clustering groups. Separation is descriptive, not independent validation of cell identity.",
    umap_coordinates: "Accepted UMAP coordinates. No cluster or cell-type assignment is implied by this coordinate plot.",
    umap_categorical: "Accepted UMAP coordinates colored by the recorded categorical field. Supplied or predicted labels are not independent validation.",
    clustering_metrics: "Clustering agreement with the recorded reference labels. Interpretation depends on the reference's provenance and independence.",
  };
  const levels = [...new Set(artifacts.filter(a => a.figure.kind === "annotation_umap")
    .map(a => String(a.figure.annotation_level)))];
  const levelCards = [];
  if (levels.length > 1) {
    const label = document.createElement("label"); label.textContent = "Annotation level ";
    const select = document.createElement("select"); select.setAttribute("aria-label", "Annotation level");
    for (const level of levels) {
      const option = document.createElement("option"); option.value = level;
      option.textContent = level === "final" ? "Final annotation" : `Level ${level}`;
      select.append(option);
    }
    select.value = "final";
    select.addEventListener("change", () => {
      for (const [card, level] of levelCards) card.hidden = level !== select.value;
    });
    label.append(select); section.append(label);
  }
  // Bound image loading, not scientific availability. Every figure and full-size
  // PNG remains in the source-bound inventory below.
  for (const artifact of artifacts) {
    const card = document.createElement("figure");
    const title = document.createElement("h4"); title.textContent = readableLabel(artifact.figure.kind) + (artifact.figure.annotation_level !== undefined ? ` · ${artifact.figure.annotation_level === 'final' ? 'Final annotation' : 'Level ' + artifact.figure.annotation_level}` : '');
    const image = document.createElement("img"); image.className = "analysis-figure";
    image.loading = "lazy"; image.alt = title.textContent;
    image.src = `/api/v1${artifactPath(state.session.session_id, artifact.revision_id, artifact.handle, artifact.turn_id)}`;
    const caption = document.createElement("figcaption"); caption.className = "small";
    caption.textContent = captions[artifact.figure.kind] || "Published scientific figure; exact sources are available below.";
    if (artifact.figure.compatible_umap_available === false) caption.textContent += " No compatible accepted UMAP was available for this publication.";
    image.addEventListener("error", () => {
      image.hidden = true;
      caption.textContent = "Figure preview unavailable. Download the saved PNG to inspect it; no analysis was rerun.";
    });
    if (artifact.figure.kind === "annotation_umap" && levels.length > 1) {
      const level = String(artifact.figure.annotation_level);
      levelCards.push([card, level]); card.hidden = level !== "final";
    }
    card.append(title, image, caption); section.append(card);
  }
  return section;
}

function workflowSummaryFacts(groups) {
  const rows = new Map();
  const byHandle = new Map(groups.map((group) => [group.handle, group]));
  const derivesFrom = (group, ancestor) => {
    const pending = [...(group.derived_from || [])], seen = new Set();
    while (pending.length) {
      const handle = pending.pop();
      if (handle === ancestor) return true;
      if (seen.has(handle)) continue;
      seen.add(handle);
      pending.push(...(byHandle.get(handle)?.derived_from || []));
    }
    return false;
  };
  const add = (field, label, value, source) => {
    const key = JSON.stringify([field, value]);
    if (!rows.has(key)) rows.set(key, {field, label, value, sources: []});
    const row = rows.get(key);
    if (!row.sources.includes(source)) row.sources.push(source);
  };
  const populations = groups.flatMap((group) => {
    const parameters = group.evidence?.facts.find((f) => f.field === "parameters" && f.status === "available")?.value;
    const population = parameters?.population;
    return population?.stage === "native_cell_filter" ? [{group, population}] : [];
  });
  for (const {group, population} of populations) {
    if (typeof population.input_barcodes === "number") add("input_barcodes", "QC input barcodes", population.input_barcodes, group.label);
    if (typeof population.output_barcodes === "number") add("retained_barcodes", "QC retained barcodes", population.output_barcodes, group.label);
  }
  for (const group of groups) {
    for (const fact of group.evidence?.facts || []) {
      if (!group.summary_fields.includes(fact.field) || fact.status !== "available" || typeof fact.value !== "number") continue;
      if (fact.field === "n_features" && fact.value === 0) continue;
      if (fact.field === "n_cells" && populations.some(({group: filter, population}) =>
        (group.handle === filter.handle || derivesFrom(group, filter.handle) || derivesFrom(filter, group.handle))
        && [population.input_barcodes, population.output_barcodes].includes(fact.value))) continue;
      add(fact.field, group.evidence.field_labels[fact.field] || readableLabel(fact.field), fact.value, group.label);
    }
  }
  return [...rows.values()];
}

function renderResultSelection() {
  if (!state.revision) return;
  state.evidenceEpoch += 1;
  const groups = state.revision.result_groups || [];
  const presentations = presentationChoices();
  const presentation = {presentations: presentations.flatMap((p) => p.presentations), artifacts: presentations.flatMap((p) => p.artifacts)};
  const retained = new Set(state.revision.retained_outputs || []);
  const visible = groups.filter((g) => g.outputs.some((o) => !retained.has(o.name)));
  element("result-method").textContent = "Scientific results";
  const nodes = [];
  const technicalNodes = [];
  const limitations = [];
  limitations.push(...(state.revision.result?.scientific?.limitations || []));
  const summaryFacts = workflowSummaryFacts(visible);
  if (summaryFacts.length) {
    const values = document.createElement("dl"); values.className = "result-values workflow-summary";
    for (const fact of summaryFacts) {
      const name = document.createElement("dt"); name.textContent = fact.label;
      const value = document.createElement("dd"); value.append(evidenceValueNode(fact.value));
      if (fact.sources.length > 1) value.title = `Recorded for ${fact.sources.length} accepted outputs: ${fact.sources.join("; ")}. Equal counts alone do not establish shared cells.`;
      values.append(name, value);
    }
    nodes.push(values);
  }
  limitations.push(...visible.flatMap((group) => group.summary_limitations || []));
  limitations.push(...visible.flatMap((group) => group.evidence?.limitations || []));
  if (limitations.length) {
    const record = document.createElement("details");
    const title = document.createElement("summary"); title.textContent = "Evidence scope & technical qualifications";
    const text = document.createElement("p"); text.textContent = [...new Set(limitations)].join(" ");
    record.append(title, text); technicalNodes.push(record);
  }
  const gallery = resultFigureGallery(resultFigureArtifacts(null, presentation));
  element("result-figures").replaceChildren(...(gallery ? [gallery] : []));
  for (const group of visible) {
    const section = document.createElement("section");
    section.className = "scientific-result";
    const heading = document.createElement("h3");
    const fn = group.evidence?.facts.find((f) => f.field === "function_name" && f.status === "available");
    heading.textContent = fn?.value || group.label;
    section.append(heading);
    if (group.evidence?.facts.some((f) => f.field === "n_features" && f.value === 0 && f.status === "available")) {
      const context = document.createElement("p"); context.className = "muted small";
      context.textContent = "This native observation stage has no feature matrix yet. Its feature count describes this step, not the final count matrix.";
      section.append(context);
    }
    if (group.factual_report) {
      section.append(scientificReportNode(group.factual_report));
    }
    if (group.evidence) {
      const view = group.evidence;
      const facts = (group.factual_report ? [] : group.summary_fields)
        .flatMap((field) => view.facts.filter((fact) => fact.field === field))
        .filter((fact) => !summaryFacts.some((shown) => shown.field === fact.field && shown.value === fact.value));
      const values = document.createElement("dl"); values.className = "result-values";
      for (const fact of facts) {
        const label = document.createElement("dt"); label.textContent = view.field_labels[fact.field] || readableLabel(fact.field);
        const value = document.createElement("dd");
        if (fact.status !== "available") value.textContent = `${fact.status}: ${fact.reason || "No accepted value"}`;
        else if (["branch_summary", "population_summary", "metrics"].includes(fact.field)) {
          const preview = fact.value;
          value.append(evidenceValueNode(preview));
          if (fact.field === "branch_summary") {
            const note = document.createElement("p"); note.className = "muted small";
            note.textContent = "Recorded branch summary; omitted branches do not establish absence. Full results remain available in downloads.";
            value.append(note);
          }
        } else if (fact.value !== null && typeof fact.value === "object") {
          const records = document.createElement("details");
          records.open = fact.field === "score_examples";
          const summary = document.createElement("summary"); summary.textContent = "Structured accepted records";
          const preview = fact.field === "score_examples" && Array.isArray(fact.value)
            ? fact.value.map(({sequence_id, condition, mean, fold_std, fold_0, fold_1, fold_3}) =>
                ({sequence_id, condition, mean, fold_std, fold_0, fold_1, fold_3})) : fact.value;
          records.append(summary, evidenceValueNode(preview)); value.append(records);
        } else value.append(evidenceValueNode(fact.value));
        values.append(label, value);
      }
      if (values.childElementCount) section.append(values);
      if (!group.factual_report && !view.facts.some((fact) => group.summary_fields.includes(fact.field))) {
        const empty = document.createElement("p");
        empty.textContent = `Summary: ${view.reason || view.status}. Accepted fields remain in details.`;
        section.append(empty);
      }
      if (view.coverage.length) section.append(...evidenceFactNodes(view.coverage.filter((fact) =>
        !facts.some((shown) => shown.field === fact.field)), view.field_labels || {}));
    } else {
      const omitted = document.createElement("p");
      omitted.textContent = "Summary omitted from this bounded view. Select an accepted field in details to read it.";
      section.append(omitted);
    }
    const methods = document.createElement("details");
    const title = document.createElement("summary"); title.textContent = group.label;
    methods.append(title, evidenceValueNode({parameters: group.parameters, derived_from: group.derived_from,
      source: group.evidence?.source || "Unavailable", limitations: group.evidence?.limitations || []}));
    if (group.evidence) methods.append(...evidenceFactNodes(group.evidence.facts.filter((fact) =>
      !group.summary_fields.includes(fact.field)), group.evidence.field_labels || {}));
    if (section.childElementCount > 1) nodes.push(section);
    else { const method = document.createElement("p"); method.className = "accepted-operation"; method.textContent = heading.textContent; nodes.push(method); }
    technicalNodes.push(methods);
  }
  element("native-results").replaceChildren(...technicalNodes);
  element("result-facts").replaceChildren(...nodes);
  if (!nodes.length) element("result-facts").textContent = "Accepted summary unavailable. Detailed historical outputs remain accessible.";
  state.evidence = null;
  element("evidence-content").textContent = "Select an accepted field to inspect its exact value and provenance.";
  element("detail-section").replaceChildren();
  element("detail-form").hidden = true;
  element("detail-content").replaceChildren();
  renderArtifacts({...state.revision,
    artifacts: [...(presentation?.presentations || []), ...state.revision.artifacts],
    scientific_artifacts: [...(presentation?.artifacts || []), ...(state.revision.scientific_artifacts || [])]});
  if (!state.artifactPreview) clearArtifactView();
}

function scientificArtifactPath(sessionId, revisionId, handle, turnId = null) {
  const owner = revisionId === null ? turnPath(sessionId, turnId) : revisionPath(sessionId, revisionId);
  return `${owner}/scientific-artifacts/${encodeURIComponent(handle)}`;
}

function supportedScientificFile(artifact) {
  // The application validates the closed descriptors. Wording and file-purpose
  // policies belong there, shared by the conversation and result inventory.
  return ["scientific_native_result", "scientific_motif_matches", "scientific_peaks", "scientific_motifs", "scientific_sequence", "scientific_matrix", "scientific_qc_table", "scientific_selection_table",
    "scientific_analysis", "scientific_integrated", "scientific_annotation", "scientific_annotation_evidence", "scientific_model", "scientific_composition", "scientific_statistics"]
    .includes(artifact.artifact_type) && typeof artifact.filename === "string";
}

const recentDownloads = new Map();

function guardDownload(link) {
  link.addEventListener("click", (event) => {
    const now = Date.now();
    for (const [url, time] of recentDownloads) if (now - time >= 2000) recentDownloads.delete(url);
    if (now - (recentDownloads.get(link.href) || 0) < 2000) {
      event.preventDefault();
      return;
    }
    recentDownloads.set(link.href, now);
    link.textContent = "Requested — retry if needed";
    // Streaming attachments have no reliable browser completion event. This
    // indicates a request only; never claim that preparation or transfer passed.
  });
}

function scientificFileNode(sessionId, artifact) {
  const row = document.createElement("div");
  row.className = "scientific-file";
  const title = document.createElement("span");
  const name = document.createElement("strong");
  name.textContent = artifact.title || artifact.label;
  const description = document.createElement("span");
  description.className = "file-description";
  description.textContent = `${artifact.filename}${formatBytes(artifact.size_bytes) ? ` · Source size: ${formatBytes(artifact.size_bytes)}` : ""} · ${artifact.description || "Accepted scientific output"}`;
  title.append(name, description);
  const download = document.createElement("a");
  download.textContent = "Download";
  download.href = `/api/v1${scientificArtifactPath(sessionId, artifact.revision_id, artifact.handle, artifact.turn_id)}`;
  download.download = artifact.filename;
  download.target = "_blank";
  download.rel = "noopener";
  download.dataset.actionKey = `scientific-download:${artifact.handle}`;
  download.setAttribute("aria-label", `Download ${artifact.filename}`);
  guardDownload(download);
  // The browser streams an attachment directly; scientific bytes stay out of JS.
  row.append(title, download);
  if (artifact.artifact_type === "scientific_model") {
    const note = document.createElement("span");
    note.className = "scientific-export-note";
    note.textContent = "Accepted model. Read the configuration for initialization or training status and reuse requirements; these downloads are not a standalone inference bundle.";
    row.append(note);
  }
  return row;
}

function scientificFileGroups(sessionId, artifacts, selectedOutputs = []) {
  const files = [...new Map(artifacts.filter(supportedScientificFile).map((a) => [a.handle, a])).values()];
  const primaryFiles = primaryScientificFiles(files, selectedOutputs);
  const primary = document.createElement("li"); primary.className = "download-group primary-downloads";
  const additional = document.createElement("details"); additional.className = "additional-downloads";
  const title = document.createElement("summary"); title.textContent = "Additional files"; additional.append(title);
  for (const file of files) (primaryFiles.has(file.handle) ? primary : additional).append(scientificFileNode(sessionId, file));
  const nodes = [primary];
  if (additional.childElementCount > 1) { const item = document.createElement("li"); item.append(additional); nodes.push(item); }
  return nodes;
}

function primaryScientificFiles(artifacts, selectedOutputs = []) {
  // Presentation ranking only. Every exact handle stays in the advanced list.
  const files = artifacts.filter(supportedScientificFile);
  return new Set(files.filter((a) => {
    if (a.artifact_type === "scientific_annotation_evidence") return false;
    if (a.source_step?.data_output === false) return false;
    if (selectedOutputs.length && a.source_step) return a.source_step.output_names?.some((name) => selectedOutputs.includes(name));
    if (a.artifact_type === "scientific_integrated") return !files.some((other) =>
      other.artifact_type === "scientific_integrated" && a.source_step?.original_sha256
      && other.source_step?.original_sha256 === a.source_step.original_sha256
      && other.source_step.run_id === a.source_step.run_id
      && other.source_step.ancestors?.includes(a.source_step.step_id));
    if (["scientific_annotation", "scientific_matrix", "scientific_composition", "scientific_statistics", "scientific_model"].includes(a.artifact_type)) return true;
    if (a.source_step) return a.source_step.terminal;
    return a.artifact_type !== "scientific_analysis";
  }).map((a) => a.handle));
}

function loadRequestedArtifacts(turn) {
  const presentation = Boolean(turn.response.artifacts?.length);
  const recipe = turn.export_requested === true;
  if (!presentation && !recipe) return;
  const sessionId = state.session.session_id;
  const key = JSON.stringify([sessionId, turn.turn_id, turn.revision_id]);
  const cached = state.scientificFiles.get(key);
  if (cached) return;
  const epoch = state.epoch;
  const entry = { artifacts: [], presentations: [] };
  state.scientificFiles.set(key, entry);
  // Inventory is metadata only and requested once per captured result. Regular
  // conversation polling reuses this display cache; refresh reads it afresh.
  Promise.allSettled([
    recipe ? api(`${turnPath(sessionId, turn.turn_id)}/scientific-artifacts`) : Promise.resolve({ artifacts: [] }),
    presentation ? api(`${turnPath(sessionId, turn.turn_id)}/artifacts`) : Promise.resolve({ artifacts: [] }),
  ]).then(([inventory, figures]) => {
    if (epoch !== state.epoch || state.scientificFiles.get(key) !== entry) return;
    if (inventory.status === "fulfilled") {
      entry.artifacts = inventory.value.artifacts.filter((artifact) => supportedScientificFile(artifact)
        && artifact.turn_id === turn.turn_id && (artifact.revision_id === turn.revision_id
          || recipe && artifact.artifact_type === "scientific_composition" && artifact.revision_id === null));
    }
    if (figures.status === "fulfilled") entry.presentations = figures.value.artifacts;
    renderHistory();
    if (!state.artifactPreview) renderResultSelection();
  }).catch(() => {
    // Missing or unqualified historical associations expose no download.
  });
}

function renderArtifacts(revision, target = element("artifact-inventory")) {
  const sessionId = state.session.session_id;
  const artifacts = distinctFigureArtifacts(revision.artifacts);
  const artifactNode = (artifact) => {
    const item = document.createElement("div"); item.className = "scientific-file";
    const supported = ["analysis_report", "analysis_figure"].includes(artifact.artifact_type);
    const title = document.createElement("span");
    const kind = artifact.artifact_type;
    const siblings = artifacts.filter((entry) => entry.artifact_type === kind);
    const number = siblings.length > 1 ? ` ${siblings.indexOf(artifact) + 1}` : "";
    const name = kind === "analysis_report" ? `Technical report${number} (TXT)`
      : kind === "analysis_figure" ? `${artifact.figure ? readableLabel(artifact.figure.kind) : "Analysis figure"}${number} (PNG)` : "Accepted artifact";
    const heading = document.createElement("strong"); heading.textContent = name;
    title.append(heading);
    if (kind === "analysis_report") {
      const description = document.createElement("span"); description.className = "file-description";
      description.textContent = "Accepted methods, parameters and execution/provenance record; saved with this analysis.";
      title.append(description);
    }
    item.append(title);
    if (supported) {
      const actions = document.createElement("div");
      actions.className = "artifact-actions";
      if (kind === "analysis_figure") {
        const view = document.createElement("button");
        view.type = "button"; view.className = "secondary"; view.textContent = "View figure";
        view.setAttribute("aria-label", `${view.textContent}: ${name}`);
        view.dataset.artifactHandle = artifact.handle;
        view.dataset.actionKey = `artifact-view:${artifact.handle}`;
        view.addEventListener("click", () => viewArtifact(sessionId, artifact.revision_id, artifact));
        actions.append(view);
      }
      const download = document.createElement("a");
      download.textContent = "Download";
      download.setAttribute("aria-label", `Download ${name}`);
      download.dataset.actionKey = `artifact-download:${artifact.handle}`;
      download.href = `/api/v1${artifactPath(sessionId, artifact.revision_id, artifact.handle, artifact.turn_id)}?download=true`;
      download.target = "_blank";
      download.rel = "noopener";
      guardDownload(download);
      actions.append(download);
      item.append(actions);
    }
    return item;
  };
  const nodes = scientificFileGroups(sessionId, revision.scientific_artifacts || []);
  const primary = nodes[0];
  for (const artifact of artifacts) primary.append(artifactNode(artifact));
  if (!primary.childElementCount) nodes.shift();
  if (!nodes.length) {
    const empty = document.createElement("li"); empty.textContent = "No accepted scientific downloads are available."; nodes.push(empty);
  }
  target.replaceChildren(...nodes);
}

async function viewArtifact(sessionId, revisionId, artifact, reveal = true) {
  const epoch = state.epoch;
  if (!state.session || state.session.session_id !== sessionId) return;
  if (reveal) {
    showWorkspacePanel("results");
    element("artifact-panel").open = true;
  }
  const coordinateRevision = artifact.figure?.sources?.find((source) => source.role === "coordinates")?.output?.revision_id;
  if (revisionId === null && coordinateRevision && state.viewedRevisionId !== coordinateRevision) await loadRevision(coordinateRevision);
  if (revisionId !== null && (!state.revision || state.revision.revision_id !== revisionId)) await loadRevision(revisionId);
  if (epoch !== state.epoch) return;
  if (!state.session || state.session.session_id !== sessionId
      || (revisionId === null && coordinateRevision && state.viewedRevisionId !== coordinateRevision)
      || (revisionId !== null && (!state.revision || state.revision.revision_id !== revisionId))) {
    if (state.session && state.session.session_id === sessionId) element("scientific-details").scrollIntoView({ block: "nearest" });
    return;
  }
  if (reveal) state.artifactPreview = true;
  clearArtifactView();
  const artifactEpoch = state.artifactEpoch;
  const revisionEpoch = state.revisionEpoch;
  element("artifact-viewer").hidden = false;
  const figures = revisionId === null ? [artifact] : state.revision.artifacts.filter((entry) => entry.artifact_type === "analysis_figure");
  element("artifact-title").textContent = artifact.figure ? readableLabel(artifact.figure.kind) : `Analysis figure${figures.length > 1 ? ` ${figures.findIndex((entry) => entry.handle === artifact.handle) + 1} of ${figures.length}` : ""}`;
  try {
    const response = await fetch(`/api/v1${artifactPath(sessionId, revisionId, artifact.handle, artifact.turn_id)}`, {
      credentials: "same-origin", cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json();
      throw new ApiError(data.error.code, data.error.message, response.status);
    }
    const contentType = response.headers.get("Content-Type") || "";
    if (artifact.artifact_type === "analysis_figure") {
      if (!contentType.startsWith("image/png")) throw new ApiError("ARTIFACT_TYPE_UNSUPPORTED", "This artifact is not a supported PNG figure.");
      const blob = await response.blob();
      if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
      state.artifactUrl = URL.createObjectURL(blob);
      element("figure-content").src = state.artifactUrl;
      element("figure-content").hidden = false;
      element("figure-description").hidden = false;
      element("figure-description").textContent = artifact.figure
        ? `Accepted ${readableLabel(artifact.figure.kind)} · Source runs: ${[...new Set([artifact.figure.run_id, ...(artifact.figure.sources || []).map((s) => s.run_id)].filter(Boolean))].join(", ")} · Operations: ${artifact.figure.step_ids.join(", ")}. Exact sources are saved in the technical report.`
        : "Accepted historical PNG. Figure subject metadata is unavailable; no association with the selected field is asserted.";
      if (artifact.figure?.coordinate_reused) {
        const coordinates = artifact.figure.sources.find((source) => source.role === "coordinates");
        const categories = artifact.figure.sources.find((source) => source.role === "categories");
        const labelSource = categories.output ? resultLabel(categories.output.revision_id) : "the registered input";
        element("figure-description").textContent = `Accepted UMAP coordinates from ${resultLabel(coordinates.output.revision_id)}. Categories: ${categories.field}, from ${labelSource}. No UMAP fit was performed to render this figure; exact sources are saved in the technical report.`;
      }
    }
    if (reveal) element("artifact-viewer").scrollIntoView({ block: "nearest" });
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
    element("artifact-view-error").textContent = `${error.code || "ARTIFACT_UNAVAILABLE"}: ${error instanceof ApiError ? error.message : "The accepted artifact could not be read."}`;
    element("artifact-view-error").hidden = false;
  }
}

function readableLabel(value) {
  const scientificLabels = {umap_leiden: "UMAP · clusters", umap_coordinates: "UMAP coordinates",
    umap_categorical: "UMAP · categorical field", annotation_umap: "UMAP · cell-type annotation",
    qc_metrics: "QC measurements", annotation_composition: "Cell-type composition",
    marker_activity_umap: "UMAP · marker activity", marker_activity_dot: "Marker activity",
    marker_activity_heatmap: "Marker activity heatmap", annotation_hierarchy: "Annotation hierarchy"};
  if (scientificLabels[value]) return scientificLabels[value];
  const label = String(value).replace(/_/g, " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function statusLabel(status, responseKind = null, revisionId = null, runId = undefined, steps = undefined) {
  if (status === "finalizing") {
    if (responseKind) return "Saved progress requires recovery";
    return revisionId ? "Scientific analysis completed. Preparing the results and explanation."
      : "Preparing the scientific response.";
  }
  if (status === "failed" && responseKind === "answer") return "Answer unavailable";
  if (status === "failed" && (runId === null || (Array.isArray(steps) && steps.every((step) => step.attempts === 0)))) return "Request failed before scientific execution";
  const labels = {
    activated: "Analysis completed", succeeded: "Analysis completed",
    stale: "Analysis completed · Historical result", failed: "Analysis failed",
    cancelled: "Cancelled", clarification: "Clarification needed",
    answered: "Answer available", navigated: "Revision selected",
    unavailable: "Result unavailable", running: "Analysis in progress",
    planning: "Planning", validated: "Ready for execution", submitted: "Request received",
    planned: "Plan prepared", interrupted: "Interrupted",
  };
  return labels[status] || readableLabel(status || "Unavailable");
}

function scientificReportNode(text) {
  // Format only the deterministic report owner's headings/tables. Every value
  // uses textContent; this does not interpret model prose or execute raw HTML.
  const report = document.createElement("div"); report.className = "factual-report";
  const lines = text.split("\n");
  const cells = line => line.split(/(?<!\\)\s*\|\s*/).map(value => value.replace(/\\\|/g, "|"));
  for (let i = 0; i < lines.length; i += 1) {
    if (!lines[i].trim()) continue;
    if (lines[i + 1] && cells(lines[i + 1]).every(value => value.trim() === "---")) {
      const wrapper = document.createElement("div"); wrapper.className = "scientific-table";
      const table = document.createElement("table"); const head = document.createElement("thead");
      const header = document.createElement("tr");
      for (const value of cells(lines[i])) { const th = document.createElement("th"); th.textContent = value; header.append(th); }
      head.append(header); table.append(head); i += 2;
      const body = document.createElement("tbody");
      while (i < lines.length && lines[i].trim()) {
        const row = document.createElement("tr");
        for (const value of cells(lines[i])) { const td = document.createElement("td"); td.textContent = value; row.append(td); }
        body.append(row); i += 1;
      }
      table.append(body); wrapper.append(table); report.append(wrapper);
    } else {
      const heading = lines[i].match(/^(#{3,4}) (.*)$/);
      const node = document.createElement(heading ? (heading[1].length === 3 ? "h3" : "h4") : "p");
      node.textContent = heading ? heading[2] : lines[i]; report.append(node);
    }
  }
  return report;
}

function evidenceValueNode(value) {
  if (Array.isArray(value)) {
    if (value.length && value.every((entry) => entry !== null && !Array.isArray(entry) && typeof entry === "object")) {
      const columns = [...new Set(value.flatMap((entry) => Object.keys(entry)))];
      const wrapper = document.createElement("div"); wrapper.className = "scientific-table";
      const table = document.createElement("table");
      const header = document.createElement("tr");
      for (const key of columns) { const cell = document.createElement("th"); cell.scope = "col"; cell.textContent = readableLabel(key); header.append(cell); }
      const head = document.createElement("thead"); head.append(header); table.append(head);
      const body = document.createElement("tbody");
      for (const row of value) {
        const line = document.createElement("tr");
        for (const key of columns) { const cell = document.createElement("td"); cell.append(Object.hasOwn(row, key) ? evidenceValueNode(row[key]) : document.createTextNode("Not recorded")); line.append(cell); }
        body.append(line);
      }
      table.append(body); wrapper.append(table); return wrapper;
    }
    const list = document.createElement("ul");
    list.className = "compact-list";
    for (const entry of value) {
      const item = document.createElement("li");
      item.append(evidenceValueNode(entry));
      list.append(item);
    }
    return list;
  }
  if (value !== null && typeof value === "object") {
    const container = document.createElement("div");
    if (value.complete === false) {
      const coverage = document.createElement("p"); coverage.className = "partial-coverage";
      coverage.textContent = "Partial coverage — omitted records do not establish absence.";
      container.append(coverage);
    }
    const list = document.createElement("dl");
    list.className = "fact-values";
    const entries = Object.entries(value);
    // Owner record tables are the primary structured result; retain every
    // denominator, source role and coverage value alongside them.
    entries.sort(([a], [b]) => (a === "records" ? -1 : b === "records" ? 1 : 0));
    for (const [key, entry] of entries) {
      const label = document.createElement("dt");
      label.textContent = readableLabel(key);
      const content = document.createElement("dd");
      content.append(evidenceValueNode(entry));
      list.append(label, content);
    }
    container.append(list); return container;
  }
  const content = document.createElement("span");
  content.textContent = value === null ? "Recorded null" : String(value);
  return content;
}

function evidenceFactNodes(facts, fieldLabels = {}) {
  return (facts || []).map((fact) => {
    const technical = fact.field.endsWith("_sha256") || fact.field.includes("contract");
    const item = document.createElement(technical ? "details" : "section");
    item.className = "scientific-fact";
    if (fact.field === "branch_summary") item.classList.add("annotation-groups");
    const title = document.createElement(technical ? "summary" : "h4");
    title.textContent = `${fieldLabels[fact.field] || readableLabel(fact.field)}${fact.status === "available" ? "" : ` · ${fact.status}`}`;
    const content = fact.status === "available" ? evidenceValueNode(fact.value) : document.createElement("p");
    if (fact.status !== "available") content.textContent = fact.reason || "No accepted value is available.";
    item.append(title, content);
    if (fact.derivation) {
      const provenance = document.createElement("details");
      const label = document.createElement("summary"); label.textContent = "Descriptive read sources & coverage";
      provenance.append(label, evidenceValueNode(fact.derivation)); item.append(provenance);
    }
    return item;
  });
}

function renderEvidence(view, target) {
  const status = document.createElement("p");
  status.className = "small";
  status.textContent = `Availability: ${view.status}${view.reason ? ` · ${view.reason}` : ""}\nScope: ${view.evidence_scope} · Generation: ${view.generation === null ? "Unavailable" : view.generation}`;
  const nodes = [status];
  if (view.status === "available" && view.source) {
    const operation = document.createElement("p");
    operation.className = "accepted-operation";
    operation.textContent = `Accepted operation: ${readableLabel(view.source.tool_name)}.`;
    nodes.push(operation);
    for (const prior of view.source.prior_outputs) {
      const reused = document.createElement("p");
      reused.className = "accepted-reuse";
      reused.textContent = `Prior accepted input consumed: ${readableLabel(prior.argument)}. This binding identifies historical input reuse.`;
      nodes.push(reused);
    }
    if (view.source.authority_sha256 === null) {
      const authority = document.createElement("p");
      authority.className = "muted small";
      authority.textContent = "Reusable scientific authority is unavailable for this accepted result.";
      nodes.push(authority);
    }
  }
  for (const [heading, facts] of [["Coverage", view.coverage], ["Accepted bounded facts", view.facts], ["Detailed evidence", view.detail]]) {
    if (!facts.length) continue;
    const title = document.createElement("h3");
    title.textContent = heading;
    if (heading === "Accepted bounded facts" && view.detail.length) {
      const summary = document.createElement("details");
      const label = document.createElement("summary"); label.textContent = heading;
      summary.append(label, ...evidenceFactNodes(facts, view.field_labels || {})); nodes.push(summary);
    } else nodes.push(title, ...evidenceFactNodes(facts, view.field_labels || {}));
  }
  if (view.source) {
    const provenance = document.createElement("details");
    const label = document.createElement("summary");
    label.textContent = "Verification & provenance";
    const source = view.source;
    const rows = document.createElement("dl");
    rows.className = "revision-summary";
    rows.replaceChildren(...summaryRows([
      ["Tool", source.tool_name], ["Result contract", source.result_contract],
      ["Source run", source.output_locator.run_id], ["Source step", source.output_locator.step_id],
      ["Source output", source.output_locator.name], ["Accepted step SHA", source.output_locator.accepted_step_sha256],
      ["Recovery contract", source.recovery_identity], ["Evidence SHA", source.evidence_sha256],
      ["Run result SHA", source.source_run_result_sha256], ["Authority scope", source.authority_scope],
    ]));
    const checks = document.createElement("h3");
    checks.textContent = "Accepted verification checks";
    const lineage = document.createElement("h3");
    lineage.textContent = "Accepted lineage";
    const prior = source.prior_outputs.map((output) => `${output.argument}: ${output.run_id} / ${output.step_id} / ${output.output_key}`);
    provenance.append(label, rows, checks, textList(source.verification_checks), lineage,
      textList([...source.depends_on.map((step) => `Step dependency: ${step}`), ...prior]));
    nodes.push(provenance);
  }
  const limitations = document.createElement("h3");
  limitations.textContent = "Limitations";
  nodes.push(limitations, textList(view.limitations));
  target.replaceChildren(...nodes);
}

async function loadEvidence(detail = false) {
  if (!state.session || !state.revision || !element("evidence-output").value) return;
  const sessionId = state.session.session_id;
  const revisionId = state.revision.revision_id;
  const epoch = state.epoch;
  const revisionEpoch = state.revisionEpoch;
  const evidenceEpoch = ++state.evidenceEpoch;
  const query = new URLSearchParams({ output_name: element("evidence-output").value });
  if (!detail) {
    const group = (state.revision.result_groups || []).find((g) => g.outputs.some((o) => o.name === element("evidence-output").value));
    const binding = group?.outputs.find((o) => o.name === element("evidence-output").value);
    if (binding) query.append("fields", binding.field);
  }
  if (detail) {
    const section = element("detail-section").value;
    if (["category_counts", "category_association", "categorical_fields"].includes(section)) {
      if (element("detail-field").value) query.set("field", element("detail-field").value);
      query.set("offset", element("detail-offset").value);
      if (section === "category_association") {
        query.set("other_field", element("detail-other-field").value);
        if (element("detail-comparison").value) query.set("comparison_output_name", element("detail-comparison").value);
      }
    }
    query.set("detail_section", element("detail-section").value);
    query.set("limit", element("detail-limit").value);
    if (!["category_counts", "category_association", "categorical_fields"].includes(section) && element("detail-subject").value) query.set("subject", element("detail-subject").value);
  }
  const target = element(detail ? "detail-content" : "evidence-content");
  if (!detail) {
    state.evidence = null;
    element("detail-form").hidden = true;
    element("detail-content").replaceChildren();
  }
  target.textContent = "Reading accepted evidence…";
  let view;
  try {
    view = await api(`${revisionPath(sessionId, revisionId)}/evidence?${query}`);
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || evidenceEpoch !== state.evidenceEpoch) return;
    target.textContent = `${error.code || "EVIDENCE_REQUEST_FAILED"}: ${error.message}`;
    return;
  }
  if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || evidenceEpoch !== state.evidenceEpoch) return;
  try {
    renderEvidence(view, target);
    if (!detail) {
      element("detail-section").replaceChildren(...view.supported_details.map((section) => {
        const option = document.createElement("option");
        option.value = section;
        option.textContent = section;
        return option;
      }));
      element("detail-form").hidden = !view.supported_details.length;
      if (!view.supported_details.length) {
        element("detail-content").textContent = "No reviewed detailed-evidence section is offered for this output.";
      }
      state.evidence = view;
      saveConveniences();
    }
  } catch (error) {
    console.error("Accepted evidence rendering failed.", error);
    target.textContent = "EVIDENCE_RENDER_FAILED: Accepted evidence was received, but this browser could not display it. Reload the page to try again.";
  }
}

function captureConversationUi(history) {
  const expanded = new Set(Array.from(history.querySelectorAll("details[open][data-detail-key]")).map((details) =>
    JSON.stringify([details.closest(".turn").dataset.turnId, details.dataset.detailKey])));
  const focused = document.activeElement;
  const turn = focused && history.contains(focused) && focused.closest(".turn");
  const known = new Set(Array.from(history.querySelectorAll("details[data-detail-key]")).map((details) =>
    JSON.stringify([details.closest(".turn").dataset.turnId, details.dataset.detailKey])));
  return { expanded, known, focus: turn && focused.dataset.actionKey
    ? { turnId: turn.dataset.turnId, actionKey: focused.dataset.actionKey } : null };
}

function restoreConversationUi(history, captured) {
  for (const details of history.querySelectorAll("details[data-detail-key]")) {
    const key = JSON.stringify([details.closest(".turn").dataset.turnId, details.dataset.detailKey]);
    if (captured.known.has(key)) details.open = captured.expanded.has(key);
  }
  if (!captured.focus) return;
  const turn = Array.from(history.children).find((node) => node.dataset.turnId === captured.focus.turnId);
  const target = turn && Array.from(turn.querySelectorAll("[data-action-key]")).find((node) =>
    node.dataset.actionKey === captured.focus.actionKey);
  if (target) target.focus({ preventScroll: true });
}

// Draft storage contains opaque references and display labels, never file bytes.
function saveAttachments() {
  if (!state.attachmentSession) return;
  try { sessionStorage.setItem(`agent.web.attachments.${state.attachmentSession}`, JSON.stringify(
    state.attachments.map((a) => ({ resource: a.resource, label: a.label, size: a.size })))); } catch (_) {}
}

function renderAttachments() {
  element("chat-attachments").replaceChildren(...state.attachments.map((a) => {
    const row = document.createElement("li");
    const label = document.createElement("span");
    row.className = `attachment attachment-${a.status}`;
    const name = document.createElement("strong");
    name.textContent = a.label;
    const detail = document.createElement("span");
    detail.className = "attachment-state";
    detail.textContent = [formatBytes(a.size), a.status === "uploaded" ? "Uploaded · eligibility checked during analysis"
      : a.status === "uploading" ? (a.percent === 100 ? "Transfer complete · registering…" : `Uploading ${a.percent || 0}%`)
        : a.status === "failed" ? a.error || "Upload failed" : "Waiting to upload"].filter(Boolean).join(" · ");
    label.append(name, detail);
    row.append(label);
    if (a.status === "uploading") {
      const progress = document.createElement("progress");
      progress.max = 100; progress.value = a.percent || 0;
      progress.setAttribute("aria-label", `Upload progress for ${a.label}`);
      row.append(progress);
    }
    if (a.status === "failed" && a.file) {
      const retry = document.createElement("button"); retry.type = "button"; retry.className = "secondary";
      retry.textContent = "Retry"; retry.setAttribute("aria-label", `Retry upload ${a.label}`); retry.addEventListener("click", () => uploadChatAttachment(a)); row.append(retry);
    }
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "secondary";
    remove.textContent = "Remove"; remove.setAttribute("aria-label", `Remove attachment ${a.label}`);
    remove.disabled = state.posting || Boolean(state.activeTurn && !terminal(state.activeTurn));
    remove.addEventListener("click", () => {
      state.attachments = state.attachments.filter((item) => item !== a);
      if (a.request) a.request.abort();
      saveAttachments(); renderAttachments(); inputContext(); updateControls();
      element("attach-chat-files").focus();
    });
    row.append(remove); return row;
  }));
  if (state.session?.turns.at(-1)?.clarification_choices?.some((row) => row.action === "input_role")) renderHistory();
  element("chat-attachment-status").textContent = state.attachments.some((a) => a.status !== "uploaded")
    ? "Wait for uploads to finish, or retry/remove failed files before sending."
    : state.attachments.length ? "Files accompany this message. Uploading does not qualify a file or complete its analysis." : "";
}

async function restoreAttachments(sessionId) {
  for (const a of state.attachments) if (a.request) a.request.abort();
  state.attachments = []; state.attachmentSession = sessionId; state.restoringAttachments = true;
  renderAttachments(); updateControls();
  let saved = [];
  try { saved = JSON.parse(sessionStorage.getItem(`agent.web.attachments.${sessionId}`) || "[]"); } catch (_) {}
  if (!Array.isArray(saved)) saved = [];
  try {
    const catalog = await api(`${sessionPath(sessionId)}/attachments`);
    if (state.attachmentSession !== sessionId) return;
    const used = new Set(state.session.turns.flatMap((t) => (t.attachments || []).map((a) => a.resource_id)));
    if (Array.isArray(saved)) state.attachments = saved.slice(0, 16).filter((a) => a && typeof a.label === "string")
      .filter((a) => !(a.resource && used.has(a.resource.resource_id))).map((a) => {
        const resource = catalog.choices.find((r) => a.resource && r.resource_id === a.resource.resource_id);
        return { label: resource ? resource.label : a.label, size: a.size, resource, status: resource ? "uploaded" : "failed",
          error: "Transfer unavailable after refresh. Remove and select the file again." };
      });
    saveAttachments();
  } catch (error) {
    if (state.attachmentSession === sessionId) {
      state.attachments = saved.slice(0, 16).filter((a) => a && typeof a.label === "string").map((a) => ({
        label: a.label, status: "failed", error: "Cannot confirm this attachment. Reopen the session to retry, or remove it." }));
      showError(error);
    }
  } finally {
    if (state.attachmentSession === sessionId) {
      state.restoringAttachments = false; renderAttachments(); inputContext(); updateControls();
    }
  }
}

async function uploadChatAttachment(a) {
  if (a.status === "uploading") return;
  if (state.uploadMaxBytes && a.file.size > state.uploadMaxBytes) {
    a.status = "failed";
    a.error = `Exceeds this server’s ${formatBytes(state.uploadMaxBytes)} per-file limit. Remove this file or ask the operator to raise the limit, then retry.`;
    renderAttachments(); updateControls(); saveAttachments(); return;
  }
  a.status = "uploading"; a.percent = 0;
  renderAttachments(); updateControls(); saveAttachments();
  try {
    a.resource = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest(); a.request = request;
      // Extension is an explicit H5AD candidate hint only; the existing owner validates bytes.
      const inputType = a.file.name.toLowerCase().endsWith(".h5ad") ? "h5ad" : "attachment";
      request.open("PUT", `/api/v1/sessions/${encodeURIComponent(a.sessionId)}/uploads/${encodeURIComponent(a.uploadId)}?filename=${encodeURIComponent(a.file.name)}&input_type=${inputType}`);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) a.percent = Math.floor(event.loaded * 100 / event.total);
        if (state.attachments.includes(a)) renderAttachments();
      });
      request.addEventListener("load", () => {
        let data; try { data = JSON.parse(request.responseText); } catch (_) {}
        if (request.status >= 200 && request.status < 300 && data && data.status === "registered"
            && /^local-[0-9a-f]{64}$/.test(data.resource_id) && data.input_type === inputType) resolve(data);
        else reject(new Error(data && data.error ? data.error.message : "Upload failed. Retry this file."));
      });
      for (const event of ["error", "abort", "timeout"]) request.addEventListener(event,
        () => reject(new Error("Upload interrupted. Retry this file.")));
      request.send(a.file);
    });
    a.status = "uploaded";
  } catch (error) { a.status = "failed"; a.error = error.message; }
  if (state.attachments.includes(a)) { saveAttachments(); renderAttachments(); inputContext(); updateControls(); }
}

element("attach-chat-files").addEventListener("click", () => element("attachment-dialog").showModal());
element("close-attachments").addEventListener("click", () => element("attachment-dialog").close());
element("browse-chat-files").addEventListener("click", () => element("chat-files").click());
element("chat-files").addEventListener("change", () => {
  if (!state.session || element("attach-chat-files").disabled) return;
  const files = Array.from(element("chat-files").files);
  element("chat-files").value = "";
  if (element("upload-type").value !== "attachment") {
    prepareDeclaredUpload(files[0]);
    return;
  }
  element("attachment-dialog").close();
  if (state.attachments.length + files.length > 16) {
    showError(new Error("Attach at most 16 files to one message. Remove a file before adding more.")); return;
  }
  const additions = files.map((file) => ({file, label: file.name, size: file.size, status: "selected", uploadId: newTurnId(), sessionId: state.session.session_id}));
  state.attachments.push(...additions);
  // Sequential transfers use the existing bounded server capacity; failures are independent.
  (async () => { for (const a of additions) if (state.attachments.includes(a)) await uploadChatAttachment(a); })();
  renderAttachments(); updateControls(); saveAttachments();
});

function renderHistory() {
  const history = element("conversation-history");
  const captured = captureConversationUi(history);
  const followLatest = history.scrollHeight - history.scrollTop - history.clientHeight < 80;
  const scrollTop = history.scrollTop;
  const turns = state.session ? state.session.turns.filter((turn) => turn.utterance || turn.response || turn.rerun_revision_id) : [];
  const visibleFileKeys = new Set(turns.map((turn) =>
    JSON.stringify([state.session.session_id, turn.turn_id, turn.revision_id])));
  for (const key of state.scientificFiles.keys()) {
    if (!visibleFileKeys.has(key)) state.scientificFiles.delete(key);
  }
  if (!turns.length) {
    const empty = document.createElement("p");
    empty.className = "empty-history";
    const welcome = document.createElement("strong");
    welcome.textContent = state.session ? "Start with your data, or a question." : "Opening your scientific workspace…";
    const help = document.createElement("span");
    help.textContent = "Attach files or describe what you want to learn.";
    empty.append(welcome, help);
    history.replaceChildren(empty);
    return;
  }
  history.replaceChildren(...turns.map((turn) => {
    const article = document.createElement("article");
    article.className = "turn";
    article.dataset.turnId = turn.turn_id;
    if (turn.utterance) article.append(messageNode("You", turn.utterance, "user"));
    else if (turn.rerun_revision_id) article.append(messageNode("You", "Run with changes", "user"));
    if (turn.attachments && turn.attachments.length) {
      const attachments = document.createElement("ul"); attachments.className = "message-attachments";
      for (const a of turn.attachments) {
        const item = document.createElement("li");
        item.textContent = `${a.label} · ${a.use === "staged" ? "Not yet used in an analysis" : a.use === "consumed" ? "Used in the accepted analysis" : "Attached for analysis"}`;
        attachments.append(item);
      }
      (article.querySelector(".user") || article).append(attachments);
    }
    let response = null;
    if (turn.response) {
      response = messageNode("Agent", turn.response.text, "assistant");
      article.append(response);
      if (turn.response.guidance) response.append(guidanceNode(turn.response.guidance));
      if (turn.clarification_choices?.length) {
        const choices = document.createElement("div"); choices.className = "clarification-choices";
        for (const option of turn.clarification_choices) {
          if (option.action === "input_role") {
            const label = document.createElement("label"); label.textContent = option.label + " ";
            const select = document.createElement("select"); select.setAttribute("aria-label", option.label + " file");
            const files = new Map((option.sources || []).map((row) => [row.resource_id, row.label]));
            state.attachments.filter((row) => row.status === "uploaded").forEach((row) => files.set(row.resource.resource_id, row.resource.label));
            selectOptions(select, [["", "Choose file"], ...files]);
            const apply = document.createElement("button"); apply.type = "button"; apply.className = "secondary"; apply.textContent = "Use this file";
            apply.disabled = state.posting || !files.size;
            apply.addEventListener("click", () => { if (select.value) submitRequest("", null,
              {question_turn_id: turn.turn_id, handle: option.handle + ":" + select.value}); });
            label.append(select, apply); choices.append(label);
            continue;
          }
          const button = document.createElement("button"); button.type = "button"; button.className = "secondary";
          button.textContent = option.description || option.label;
          button.dataset.actionKey = `clarification-choice:${option.handle}`;
          button.disabled = state.posting || Boolean(state.activeTurn && !terminal(state.activeTurn));
          button.addEventListener("click", () => {
            if (option.action === "parameter") {
              openRequestedParameter(option, turn.turn_id);
            } else submitRequest("", null, {question_turn_id: turn.turn_id, handle: option.handle});
          });
          choices.append(button);
        }
        response.append(choices);
      }

      loadRequestedArtifacts(turn);
    }
    const status = document.createElement("p");
    status.className = "turn-status-note";
    status.textContent = statusLabel(turn.status, turn.response && turn.response.kind, turn.revision_id, turn.run_id, turn.steps) + ((turn.attachments || []).some((a) => a.use === "staged") ? " · Some attachments remain unresolved" : "");
    const metadata = document.createElement("details");
    metadata.className = "turn-metadata";
    metadata.dataset.detailKey = "turn-details";
    const metadataLabel = document.createElement("summary");
    metadataLabel.dataset.actionKey = "turn-details";
    metadataLabel.textContent = "Turn details";
    metadata.append(metadataLabel);
    const checkpoint = document.createElement("div");
    checkpoint.className = "turn-checkpoint";
    checkpoint.textContent = checkpointText(turn);
    metadata.append(checkpoint);
    if (Object.prototype.hasOwnProperty.call(turn, "base_revision_id")) {
      const captured = document.createElement("div");
      captured.className = "turn-checkpoint";
      captured.textContent = `Captured base revision: ${turn.base_revision_id || "None"} · Generation: ${turn.base_generation}`;
      metadata.append(captured);
    }
    const scientific = turn.response && turn.response.scientific;
    if (scientific) {
      if (scientific.limitations?.length) {
        const note = document.createElement("p"); note.className = "result-caution small";
        note.textContent = scientific.limitations.join(" "); metadata.append(note);
      }
      const targets = document.createElement("div");
      targets.className = "turn-checkpoint";
      if (scientific.targets) {
        targets.append(textList(scientific.targets.map((target) => `Scientific target: Revision ${target.revision_id} · Output ${target.output_name} · Subject ${target.subject === null ? "None specified" : target.subject}`)));
      } else {
        targets.textContent = "Scientific target metadata was not stored in this historical display.";
      }
      metadata.append(targets);
      if (scientific.availability) {
        const explanation = document.createElement("p");
        explanation.textContent = `Scientific explanation: ${scientific.availability}. Model identity is the captured turn profile.`;
        metadata.append(explanation);
      }
      if (scientific.citations?.length) {
        const references = document.createElement("details");
        const label = document.createElement("summary"); label.textContent = "Exact scientific citations";
        references.append(label, evidenceValueNode(scientific.citations)); metadata.append(references);
      }
    }
    const discussed = [...new Set((scientific?.targets || []).map((target) => target.revision_id))];
    const resultRevision = turn.revision_id || (discussed.length === 1 ? discussed[0] : null);
    if (resultRevision) {
      const result = document.createElement("button");
      result.className = "secondary view-result";
      result.type = "button";
      result.dataset.actionKey = "view-result";
      result.textContent = "View result details";
      result.title = turn.response && turn.response.kind === "execute"
        ? `View created result revision ${resultRevision}` : `View revision ${resultRevision}`;
      result.addEventListener("click", () => {
        showWorkspacePanel("results");
        element("evidence-panel").open = false;
        element("artifact-panel").open = true;
        loadRevision(resultRevision);
        const details = element("scientific-details");
        details.focus({ preventScroll: true });
      });
      (response || article).append(result);
      if (turn.revision_id && ["activated", "stale", "succeeded"].includes(turn.status)) {
        for (const [label, action, key] of [
          ["Continue from this result", () => continueFromResult(resultRevision), "continue-result"],
          ["Adjust parameters", () => openResultParameters(resultRevision), "adjust-parameters"],
        ]) {
          const button = document.createElement("button"); button.type = "button"; button.className = "secondary result-action";
          button.textContent = label; button.dataset.actionKey = key;
          button.addEventListener("click", () => action().catch(showError));
          (response || article).append(button);
        }
      }
    }
    article.append(status, metadata);
    const error = turn.error || turn.response && turn.response.error;
    if (error) {
      const note = document.createElement("div");
      note.className = "turn-error";
      note.textContent = `${error.code}: ${error.message}`;
      article.append(note);
      if ((!turn.response && turn.status === "interrupted") || error.code === "INTERACTIVE_PROCESSING_INTERRUPTED" || (turn.status === "finalizing" && turn.response)) {
        const recover = document.createElement("button");
        recover.type = "button"; recover.className = "secondary";
        recover.textContent = "Recover saved progress";
        recover.addEventListener("click", async () => {
          recover.disabled = true;
          const sessionId = state.session.session_id, epoch = state.epoch;
          try {
            await api(`${turnPath(sessionId, turn.turn_id)}/recover`, "POST");
            await refreshCurrentSession(sessionId, epoch);
          } catch (failure) { showError(failure); }
          finally { recover.disabled = false; }
        });
        article.append(recover);
      }
    }
    return article;
  }));
  restoreConversationUi(history, captured);
  history.scrollTop = followLatest ? history.scrollHeight : scrollTop;
}

function updateControls() {
  const busy = state.posting || Boolean(state.activeTurn && !terminal(state.activeTurn));
  const cannotSubmit = !state.session || !state.models.length || busy || state.opening || state.restoringAttachments || state.attachments.some((a) => a.status !== "uploaded") || Boolean(state.upload) || Boolean(state.fastqCompletion && state.fastqCompletion.pending);
  element("submit-turn").disabled = cannotSubmit;
  element("attach-chat-files").disabled = !state.session || !state.resourcesEnabled || busy || state.opening || state.restoringAttachments;
  document.querySelectorAll("#chat-attachments button").forEach((button) => { button.disabled = busy; });
  document.querySelectorAll(".guidance-select").forEach((button) => { button.disabled = cannotSubmit; });
  element("utterance").disabled = !state.session;
  document.querySelectorAll(".result-action").forEach((button) => { button.disabled = cannotSubmit; });
  element("request-result-choices").querySelectorAll("input").forEach((input) => { input.disabled = cannotSubmit; });
  element("new-session").disabled = state.opening || state.posting;
  element("reopen-session").disabled = state.opening;
  element("recent-sessions").disabled = state.opening || state.posting;
  const cancel = element("cancel-turn");
  cancel.hidden = !(state.activeTurn && state.activeTurn.run_id &&
    ["planning", "validated", "running"].includes(state.activeTurn.status) && !terminal(state.activeTurn));
  cancel.disabled = cancel.hidden || state.posting;
  const uploading = Boolean(state.upload && state.upload.status === "uploading")
    || Boolean(state.fastqCompletion && state.fastqCompletion.pending);
  element("resource-choice").disabled = !state.resourcesEnabled || !state.resources.length || uploading;
  element("add-scientific-input").disabled = !canAddScientificInput() || uploading;
  document.querySelectorAll(".remove-scientific-input").forEach((button) => { button.disabled = uploading; });
  element("chat-files").disabled = !state.resourcesEnabled || uploading;
  element("upload-type").disabled = !state.resourcesEnabled || uploading;
  element("upload-index").disabled = !state.resourcesEnabled || uploading || !state.upload;
  element("clear-upload-index").hidden = !(state.upload && state.upload.index);
  element("clear-upload-index").disabled = uploading;
  element("upload-file-button").disabled = !state.resourcesEnabled || !state.upload || uploading;
  element("upload-file-button").textContent = state.upload && state.upload.status === "failed" ? "Retry upload" : "Upload";
  element("clear-upload").hidden = !state.upload;
  element("input-choice").disabled = !element("input-choice").options.length || element("input-choice").options.length === 1 || uploading;
  element("epizoo-resource-choice").disabled = selectedResourceType() !== "h5ad" || !state.epizooResources.length || uploading;
  const completing = Boolean(state.fastqCompletion && state.fastqCompletion.pending);
  for (const id of ["fastq-library", "fastq-layout", "fastq-role", "fastq-lane", "fastq-chunk", "fastq-compression", "fastq-collection-label"]) {
    element(id).disabled = !state.resourcesEnabled || uploading || completing;
  }
  document.querySelectorAll(".fastq-member input").forEach((node) => { node.disabled = uploading || completing; });
  element("complete-fastq-collection").disabled = !state.resourcesEnabled || !state.fastqMembers.length || uploading || completing || Boolean(state.upload);
  element("complete-fastq-collection").textContent = completing ? "Completing library…" : "Complete FASTQ library";
}

function stepLabel(step) {
  return step.function_name || readableLabel(step.tool_name);
}

function stepProgress(steps, status, expanded = false) {
  const section = document.createElement("div");
  const completed = steps.filter((s) => s.status === "SUCCEEDED").length;
  const running = steps.filter((s) => s.status === "RUNNING");
  const summary = document.createElement("p"); summary.className = "step-progress-summary";
  const interrupted = status === "interrupted";
  const cancelled = status === "cancelled";
  summary.textContent = `${completed} of ${steps.length} steps completed`;
  if (interrupted) summary.textContent += " · Interrupted; no automatic retry";
  else if (cancelled) summary.textContent += " · Cancelled";
  else if (status === "failed") summary.textContent += " · Failed";
  else if (running.length) summary.textContent += ` · Running: ${running.map(stepLabel).join(", ")}`;
  const segments = document.createElement("ol"); segments.className = "step-segments";
  const details = document.createElement("details"); details.open = expanded;
  details.dataset.detailKey = "execution-steps";
  const title = document.createElement("summary"); title.textContent = "Scientific step details";
  const list = document.createElement("ol"); list.className = "step-detail-list";
  for (const step of steps) {
    const display = cancelled && ["PENDING", "RUNNING", "SKIPPED"].includes(step.status) ? "CANCELLED"
      : interrupted && step.status === "RUNNING" ? "INTERRUPTED" : step.status;
    const text = `${stepLabel(step)} · ${readableLabel(display.toLowerCase())}`;
    const segment = document.createElement("li"); segment.className = `step-segment state-${display.toLowerCase()}`;
    segment.textContent = ({SUCCEEDED:"✓", RUNNING:"●", PENDING:"…", FAILED:"×", SKIPPED:"—", CANCELLED:"■", INTERRUPTED:"!"})[display] || "?";
    segment.title = text; segment.tabIndex = 0; segment.setAttribute("aria-label", text);
    segment.dataset.stepId = step.step_id; segment.dataset.recordedStatus = step.status;
    segments.append(segment);
    const item = document.createElement("li"); item.textContent = text;
    if (step.training_notice) item.textContent += ` — ${step.training_notice}`;
    if (step.error_message) { const error = document.createElement("p"); error.className = "turn-error"; error.textContent = step.error_message; item.append(error); }
    if (step.annotation_progress) {
      const progress = step.annotation_progress;
      const note = document.createElement("p");
      note.textContent = `Annotation: ${progress.completed_children} native child steps completed; ${progress.pending_branches ?? "unknown"} known pending branches. Future total is unknown.`;
      if (progress.level) note.textContent += ` Level ${progress.level}.`;
      if (progress.branch_id) note.textContent += ` Branch ${progress.branch_id}.`;
      if (progress.function_name) note.textContent += ` Native function: ${progress.function_name}.`;
      if (progress.outcome) note.textContent += ` Outcome: ${readableLabel(progress.outcome)}.`;
      note.textContent += ` ${progress.running.length} native child steps running.`;
      item.append(note);
    }
    list.append(item);
  }
  details.append(title, list); section.append(summary, segments, details); return section;
}

function renderExecution() {
  element("turn-status").parentElement.hidden = !state.posting && !state.activeTurn;
  element("turn-status").textContent = state.posting ? "Submitting request…" : state.activeTurn ? statusLabel(state.activeTurn.status, state.activeTurn.response && state.activeTurn.response.kind, state.activeTurn.revision_id, state.activeTurn.run_id, state.activeTurn.steps) : "Ready";
  element("turn-status").title = state.activeTurn ? checkpointText(state.activeTurn) : "";
  const steps = state.activeTurn && state.activeTurn.steps || [];
  const list = element("step-status");
  list.hidden = !steps.length;
  const expanded = list.querySelector("details")?.open || false;
  list.replaceChildren(...(steps.length ? [stepProgress(steps, state.activeTurn.status, expanded)] : []));
  updateControls();
}

function mergeTurn(turn) {
  if (!state.session || turn.session_id !== state.session.session_id) return;
  const turns = [...state.session.turns];
  const index = turns.findIndex((item) => item.turn_id === turn.turn_id);
  if (index === -1) turns.push(turn); else turns[index] = turn;
  state.session = { ...state.session, turns };
  // A 202 receipt is transient. Clear only references echoed by a captured turn.
  if (state.submission && state.submission.body.turn_id === turn.turn_id && turn.attachments) {
    const captured = new Set(turn.attachments.map((a) => a.resource_id));
    state.attachments = state.attachments.filter((a) => !a.resource || !captured.has(a.resource.resource_id));
    saveAttachments(); renderAttachments(); inputContext();
  }
  renderHistory();
}

function stopPolling() {
  clearTimeout(state.timer);
  state.timer = null;
  state.pollEpoch += 1;
}

// Display choice only: revisions and their active identity always come from Agent.
function revisionToDisplay(previous, view, viewedRevisionId) {
  const offered = (id) => view.revisions.some((revision) => revision.revision_id === id);
  const sameSession = previous && previous.session_id === view.session_id;
  const historicalPreview = sameSession && viewedRevisionId
    && viewedRevisionId !== previous.active_revision_id && offered(viewedRevisionId);
  if (historicalPreview) return viewedRevisionId;
  if (view.active_revision_id && offered(view.active_revision_id)) return view.active_revision_id;
  // No active revision is not permission to select an arbitrary historical output.
  return sameSession && offered(viewedRevisionId) ? viewedRevisionId : null;
}

function displaySession(view, preferredRevisionId = null) {
  const previous = state.session;
  const nextRevisionId = preferredRevisionId && view.revisions.some((revision) => revision.revision_id === preferredRevisionId)
    ? preferredRevisionId : revisionToDisplay(previous, view, state.viewedRevisionId);
  const refreshRevision = !state.revision || state.revision.revision_id !== nextRevisionId
    || !previous || previous.session_id !== view.session_id
    || (!state.revision.result && view.turns.some((turn) => turn.turn_id === state.revision.turn_id && turn.response));
  state.session = view;
  refreshRequestOptions(view.session_id);
  rememberSession(view);
  if (state.attachmentSession !== view.session_id) restoreAttachments(view.session_id);
  state.durableTurnIds = new Set(view.turns.map((turn) => turn.turn_id));
  element("session-input").value = view.session_id;
  element("history-notice").hidden = !view.history_truncated;
  renderHistory();
  renderRevisionHistory();
  if (nextRevisionId) {
    if (refreshRevision) loadRevision(nextRevisionId);
    else {
      // Immutable result content stays in place while only Session context changes.
      state.revision = { ...state.revision, is_active: nextRevisionId === view.active_revision_id,
        session_generation: view.generation };
      renderRevisionContext(state.revision);
    }
  } else clearRevisionView();
  renderResultNavigation();
  updateControls();
}

async function refreshCurrentSession(sessionId, epoch) {
  const view = await api(sessionPath(sessionId));
  if (epoch === state.epoch && state.session && sessionId === state.session.session_id) displaySession(view);
}

function watchTurn(sessionId, turnId) {
  stopPolling();
  const pollEpoch = state.pollEpoch;
  const epoch = state.epoch;
  const tick = async () => {
    if (epoch !== state.epoch || pollEpoch !== state.pollEpoch) return;
    try {
      const turn = await api(`${turnPath(sessionId, turnId)}/status`);
      if (epoch !== state.epoch || pollEpoch !== state.pollEpoch) return;
      state.activeTurn = turn;
      mergeTurn(turn);
      renderExecution();
      saveConveniences();
      showError(turn.error || turn.response && turn.response.error || null);
      if (!terminal(turn) && turn.revision_id && state.session && !state.session.revisions.some((revision) => revision.revision_id === turn.revision_id)) {
        // Scientific acceptance can precede the immutable response. Refresh its
        // authoritative result view without publishing pending delivery bytes.
        try {
          await refreshCurrentSession(sessionId, epoch);
        } catch (error) {
          if (epoch === state.epoch) showError(error);
        }
        if (epoch !== state.epoch || pollEpoch !== state.pollEpoch) return;
      }
      if (terminal(turn)) {
        stopPolling();
        state.submission = null;
        element("retry-turn").hidden = true;
        try {
          await refreshCurrentSession(sessionId, epoch);
        } catch (error) {
          if (epoch === state.epoch) showError(error);
        }
        if (epoch !== state.epoch) return;
        state.activeTurn = null;
        // Keep the completed status visible; the persisted turn also remains
        // in conversation history, while the next request becomes available.
        updateControls();
        saveConveniences();
        return;
      }
    } catch (error) {
      if (epoch !== state.epoch || pollEpoch !== state.pollEpoch) return;
      showError(error);
      if (state.submission) element("retry-turn").hidden = false;
      if (error.status && error.status !== 503) {
        stopPolling();
        if (!state.durableTurnIds.has(turnId)) {
          state.activeTurn = null;
          if (state.session) {
            state.session = { ...state.session, turns: state.session.turns.filter((turn) => turn.turn_id !== turnId) };
            renderHistory();
          }
          element("turn-status").textContent = state.submission ? "Submission status unavailable. Retry the same submission or reopen the session." : "No durable turn was found. Reopen the session to continue.";
          updateControls();
          saveConveniences();
        }
        return;
      }
    }
    state.timer = setTimeout(tick, POLL_INTERVAL_MS);
  };
  tick();
}

async function openSession(sessionId, resumeTurnId = "", preferredRevisionId = null) {
  if (state.session?.session_id !== sessionId) {
    element("utterance").value = "";
  }
  state.attachmentSession = null;
  state.revision = null;
  const epoch = ++state.epoch;
  state.scientificFiles.clear();
  stopPolling();
  state.opening = true;
  state.submission = null;
  state.activeTurn = null;
  element("retry-turn").hidden = true;
  renderExecution();
  showError(null);
  try {
    const view = await api(sessionPath(sessionId));
    if (epoch !== state.epoch) return;
    displaySession(view, preferredRevisionId);
    const current = [...view.turns].reverse().find((turn) => (turn.utterance || turn.response || turn.rerun_revision_id) && !terminal(turn));
    const requested = resumeTurnId && view.turns.find((turn) => turn.turn_id === resumeTurnId);
    const requestedConversation = requested && (requested.utterance || requested.response);
    const pendingId = requestedConversation && !terminal(requested) ? requested.turn_id : current ? current.turn_id : resumeTurnId && !requested ? resumeTurnId : "";
    if (pendingId) {
      state.activeTurn = requested || current || { session_id: sessionId, turn_id: pendingId, status: "submitted", steps: [] };
      renderExecution();
      watchTurn(sessionId, pendingId);
    }
    saveConveniences();
  } catch (error) {
    if (epoch === state.epoch) showError(error);
  } finally {
    if (epoch === state.epoch) {
      state.opening = false;
      updateControls();
    }
  }
}

function newTurnId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function postSubmission(submission) {
  const epoch = state.epoch;
  state.posting = true;
  element("retry-turn").hidden = true;
  showError(null);
  renderExecution();
  try {
    await api(`${sessionPath(submission.sessionId)}/turns`, "POST", submission.body);
    if (epoch !== state.epoch) return;
    if (element("utterance").value === submission.body.utterance) element("utterance").value = "";
    closeParameters();
    element("request-result-choices").querySelectorAll("input").forEach((input) => { input.checked = false; });
    watchTurn(submission.sessionId, submission.body.turn_id);
  } catch (error) {
    if (epoch !== state.epoch) return;
    showError(error);
    if (!error.status || error.status >= 500) {
      element("retry-turn").hidden = false;
      watchTurn(submission.sessionId, submission.body.turn_id);
    } else {
      state.submission = null;
      state.activeTurn = null;
      element("turn-status").textContent = "Request rejected";
      try { await refreshCurrentSession(submission.sessionId, epoch); } catch (_) {}
    }
  } finally {
    if (epoch === state.epoch) {
      state.posting = false;
      if (state.activeTurn) renderExecution(); else updateControls();
      saveConveniences();
    }
  }
}

function modelDescription() {
  const choice = state.models.find((m) => m.profile_id === element("model-choice").value);
  element("model-choice").title = choice ? `${choice.provider_id} · ${choice.model_id}` : "";
  saveConveniences();
}

function inputContext() {
  // Repeat the operator-admitted display label without deriving scientific context.
  const choice = element("input-choice").selectedOptions[0];
  const resources = [...selectedResourceIds().map((id) => state.resources.find((choice) => choice.resource_id === id)),
    ...state.attachments.filter((a) => a.resource && a.resource.input_type === "h5ad").map((a) => a.resource)];
  const multiple = resources.length > 1 || state.selectedResourceIds.length > 0;
  const epizoo = element("epizoo-resource-choice").selectedOptions[0];
  const labels = [...resources.map((resource, index) => `${multiple ? `${index + 1}. ` : ""}${resource.label}`),
    choice && choice.value ? choice.textContent : "",
    selectedResourceType() === "h5ad" && epizoo && epizoo.value ? epizoo.textContent : ""].filter(Boolean);
  const label = labels.length ? labels.join(" · ") : "none selected";
  element("selected-input-context").textContent = `Additional configured inputs: ${label}`;
  element("selected-input-context").hidden = !selectedResourceIds().length && !(choice && choice.value) && !(epizoo && epizoo.value);
  const fragments = selectedResourceType() === "external_fragments";
  const bam = selectedResourceType() === "bam";
  const fastq = selectedResourceType() === "fastq";
  const configuredDefaults = state.inputSets.filter((item) => item.fragments_default === true);
  const bamDefaults = state.inputSets.filter((item) => item.bam_default === true);
  const fastqDefaults = state.inputSets.filter((item) => item.fastq_default === true);
  element("input-choice").options[0].textContent = fragments
    ? configuredDefaults.length === 1 ? "Use explicitly configured fragments default" : "Choose approved fragments declarations"
    : bam ? bamDefaults.length === 1 ? "Use explicitly configured BAM context" : "Choose approved BAM context or inspect"
      : fastq ? fastqDefaults.length === 1 ? "Use explicitly configured FASTQ context" : "Choose approved FASTQ context or inspect"
        : "No structured inputs";
  element("input-choice-help").textContent = fragments
    ? "Choose an approved fragments declaration set with a reference, source profile and namespace. Missing prerequisites are reported when you send the request."
    : bam ? "Inspect the BAM without a producer context. To prepare fragments, choose an approved paired-ATAC source context and compatible reference."
      : fastq ? "Inspect the completed FASTQ library without producer resources. Fragment production requires an approved library context, barcode policy and compatible reference/runtime."
        : multiple ? "Describe each selected input's scientific role in your message. Configured inputs may supply approved companion values."
          : "Choose configured inputs or declarations for an uploaded source.";
  for (const option of element("input-choice").options) {
    const configured = state.inputSets.find((item) => item.input_set_id === option.value);
    option.disabled = Boolean(option.value) && (multiple && configured && (configured.h5ad_companion === true
      || configured.fragments_companion === true || configured.bam_companion === true || configured.fastq_companion === true)
      || fragments && !(configured && configured.fragments_companion === true)
      || bam && !(configured && configured.bam_companion === true)
      || fastq && !(configured && configured.fastq_companion === true));
    option.hidden = option.disabled;
  }
  element("configured-input-controls").hidden = !Array.from(element("input-choice").options)
    .some((option) => option.value && !option.disabled);
  element("epizoo-resource-controls").hidden = selectedResourceType() !== "h5ad" || !state.epizooResources.length;
  const irrelevant = element("configured-input-controls").hidden && element("epizoo-resource-controls").hidden;
  element("scientific-input-settings").hidden = irrelevant;

}

function selectedResourceIds() {
  const current = element("resource-choice").value;
  return [...state.selectedResourceIds, ...(current && !state.selectedResourceIds.includes(current) ? [current] : [])];
}

function canAddScientificInput() {
  const ids = selectedResourceIds();
  return Boolean(element("resource-choice").value) && ids.length < MAX_SELECTED_RESOURCES
    && ids.every((id) => state.resources.some((resource) => resource.resource_id === id && ["h5ad", "groups"].includes(resource.input_type)));
}

function renderSelectedResources() {
  const list = element("selected-scientific-inputs");
  const ids = selectedResourceIds();
  list.hidden = !ids.length;
  list.replaceChildren(...ids.map((id) => {
    const resource = state.resources.find((choice) => choice.resource_id === id);
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = resource.label;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary remove-scientific-input";
    remove.textContent = "Remove";
    remove.setAttribute("aria-label", `Remove ${resource.label} from selected inputs`);
    remove.addEventListener("click", () => {
      state.selectedResourceIds = state.selectedResourceIds.filter((value) => value !== id);
      if (element("resource-choice").value === id) element("resource-choice").value = "";
      renderResources(element("resource-choice").value);
    });
    item.append(label, remove);
    return item;
  }));
}

function selectedResourceType() {
  if (selectedResourceIds().length > 1) return null;
  const resource = state.resources.find((choice) => choice.resource_id === selectedResourceIds()[0]);
  const attached = state.attachments.filter((a) => a.resource && a.resource.input_type === "h5ad");
  return resource ? resource.input_type : state.upload ? state.upload.inputType : attached.length === 1 ? "h5ad" : null;
}

function selectedCompanion(inputType = selectedResourceType()) {
  if (selectedResourceIds().length > 1) return false;
  return state.inputSets.some((choice) => choice.input_set_id === element("input-choice").value &&
    (inputType === "external_fragments" ? choice.fragments_companion === true :
      inputType === "h5ad" ? choice.h5ad_companion === true :
        inputType === "bam" ? choice.bam_companion === true :
          inputType === "fastq" ? choice.fastq_companion === true :
            choice.h5ad_companion === true || choice.fragments_companion === true || choice.bam_companion === true || choice.fastq_companion === true));
}

function saveResourceSelection() {
  // This convenience is rechecked against the server catalog and sent per turn.
  try { sessionStorage.setItem(RESOURCE_STORAGE_KEY, element("resource-choice").value); } catch (_) {}
  try { sessionStorage.setItem(MULTI_RESOURCE_STORAGE_KEY, JSON.stringify(selectedResourceIds())); } catch (_) {}
}

function uploadStatus(text, failed = false) {
  element("upload-status").textContent = text;
  element("upload-status").classList.toggle("upload-error", failed);
  element("upload-status").hidden = !failed && !state.upload;
}

function renderResources(selectedId = "", restoreSavedSelection = false) {
  const selected = state.resources.find((choice) => choice.resource_id === selectedId);
  const canSelect = selected && (restoreSavedSelection || !state.selectedResourceIds.length || ["h5ad", "groups"].includes(selected.input_type));
  if (selected && !canSelect) state.selectedResourceIds = selectedResourceIds();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "No uploaded input selected";
  element("resource-choice").replaceChildren(none, ...state.resources.map((choice) => {
    const option = document.createElement("option");
    option.value = choice.resource_id;
    const role = choice.input_type === "external_fragments" ? `Fragments${choice.has_source_index === true ? " + TBI" : ""}` : choice.input_type === "bam" ? "BAM" : choice.input_type === "fastq" ? "FASTQ library" : choice.input_type === "groups" ? "Cell groups" : "H5AD";
    option.textContent = `${choice.label} · ${role} · registered`;
    option.disabled = state.selectedResourceIds.includes(choice.resource_id)
      || state.selectedResourceIds.length > 0 && !["h5ad", "groups"].includes(choice.input_type);
    return option;
  }));
  if (canSelect) {
    element("resource-choice").value = selectedId;
  }
  if (selectedResourceIds().length === 1 && !state.selectedResourceIds.length && !selectedCompanion()) element("input-choice").value = "";
  renderSelectedResources();
  saveResourceSelection();
  saveConveniences();
  inputContext();
  updateControls();
  return Boolean(selectedId && element("resource-choice").value === selectedId);
}

async function loadResources() {
  let selectedId = "";
  let savedIds = [];
  try { selectedId = sessionStorage.getItem(RESOURCE_STORAGE_KEY) || ""; } catch (_) {}
  try {
    const saved = JSON.parse(sessionStorage.getItem(MULTI_RESOURCE_STORAGE_KEY) || "[]");
    if (Array.isArray(saved) && saved.length <= MAX_SELECTED_RESOURCES
        && new Set(saved).size === saved.length && saved.every((id) => typeof id === "string")) savedIds = saved;
  } catch (_) {}
  try {
    const catalog = await api("/resources");
    state.resourcesEnabled = catalog.enabled === true;
    state.uploadMaxBytes = Number.isSafeInteger(catalog.upload_max_bytes) && catalog.upload_max_bytes > 0 ? catalog.upload_max_bytes : null;
    element("upload-limit").textContent = state.resourcesEnabled
      ? `Up to 16 files per message${state.uploadMaxBytes ? ` · ${formatBytes(state.uploadMaxBytes)} per file` : ""}`
      : "File uploads are not configured on this server.";
    state.resources = state.resourcesEnabled ? catalog.choices : [];
    state.fastqMembers = state.resourcesEnabled && Array.isArray(catalog.fastq_members) ? catalog.fastq_members : [];
    savedIds = savedIds.filter((id) => state.resources.some((choice) => choice.resource_id === id));
    if (savedIds.length) {
      selectedId = savedIds[savedIds.length - 1];
      state.selectedResourceIds = savedIds.slice(0, -1);
    }
    renderFastqMembers();
    const selected = state.resources.find((choice) => choice.resource_id === selectedId);
    renderResources(element("input-choice").value && !state.selectedResourceIds.length
      && !selectedCompanion(selected && selected.input_type) ? "" : selectedId, true);
    uploadStatus(state.resourcesEnabled ? "Attach a file or choose a registered input. Uploading does not start an analysis." : "Scientific attachments are unavailable on this server.");
  } catch (error) {
    state.resourcesEnabled = false;
    state.resources = [];
    state.selectedResourceIds = [];
    state.fastqMembers = [];
    renderFastqMembers();
    renderResources();
    uploadStatus(`${error.code}: ${error.message}`, true);
  }
}

function clearPendingUpload(keepFile = false) {
  const upload = state.upload;
  state.upload = null;
  if (upload && upload.request) upload.request.abort();
  if (!keepFile) element("chat-files").value = "";
  element("upload-index").value = "";
  element("upload-progress").hidden = true;
  uploadStatus("No file awaiting upload. You can send a message without an attachment.");
  updateControls();
}

function renderUploadType() {
  const automatic = element("upload-type").value === "attachment";
  element("chat-files").multiple = automatic;
  element("upload-file-button").hidden = automatic;
  const fasta = element("upload-type").value === "fasta";
  const fragments = element("upload-type").value === "external_fragments";
  const bam = element("upload-type").value === "bam";
  const fastq = element("upload-type").value === "fastq";
  const groups = element("upload-type").value === "groups";
  element("upload-file-label").textContent = automatic ? "Choose files, then describe their roles in your request." : fasta ? "Attach the plain reference-genome FASTA for your pending task" : fragments ? "Attach a fragments source (TSV, gzip or BGZF)" : bam ? "Attach a scATAC-seq BAM" : fastq ? "Attach one declared FASTQ member" : groups ? "Attach exact ordered cell_id / group TSV" : "Attach an H5AD file";
  element("chat-files").accept = automatic ? "" : fasta ? ".fa,.fasta,.fna" : fragments ? ".tsv,.gz,.bgz,.bgzf" : bam ? ".bam" : fastq ? ".fastq,.fq,.fastq.gz,.fq.gz" : groups ? ".tsv" : ".h5ad";
  element("upload-index-fields").hidden = !(fragments || fasta);
  document.querySelector('label[for="upload-index"]').textContent = fasta ? "FAI index (optional; generated if omitted)" : "Paired TBI index (optional)";
  element("upload-index").accept = fasta ? ".fai" : ".tbi";
  element("fragments-input-help").hidden = !fragments;
  element("bam-input-help").hidden = !bam;
  element("fastq-fields").hidden = !fastq;
  element("fastq-collection-fields").hidden = !fastq;
}

function fastqAttribution() {
  return Object.freeze({ library_id: element("fastq-library").value,
    fastq_layout: element("fastq-layout").value, role: element("fastq-role").value,
    lane: element("fastq-lane").value, chunk: element("fastq-chunk").value,
    compression: element("fastq-compression").value });
}

function selectedFastqMembers() {
  return Array.from(document.querySelectorAll(".fastq-member input:checked")).map((node) => node.value);
}

function renderFastqMembers(selectId = "") {
  const selected = new Set(selectedFastqMembers());
  if (selectId) selected.add(selectId);
  element("fastq-members").replaceChildren(...state.fastqMembers.map((member) => {
    const row = document.createElement("label");
    row.className = "fastq-member";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = member.resource_id;
    checkbox.checked = selected.has(member.resource_id);
    const text = document.createElement("span");
    const facts = member.fastq_attribution;
    text.textContent = `${member.label} · ${facts.library_id} · ${facts.role} · lane ${facts.lane || "unspecified"} / chunk ${facts.chunk}`;
    row.append(checkbox, text);
    return row;
  }));
}

function fastqCollectionStatus(text, failed = false) {
  element("fastq-collection-status").textContent = text;
  element("fastq-collection-status").classList.toggle("upload-error", failed);
}

async function completeFastqCollection() {
  if (!state.resourcesEnabled || state.upload || state.fastqCompletion && state.fastqCompletion.pending) return;
  if (!state.fastqCompletion) {
    state.fastqCompletion = { pending: false, body: Object.freeze({ collection_id: newTurnId(),
      label: element("fastq-collection-label").value, member_ids: Object.freeze(selectedFastqMembers()) }) };
  }
  const completion = state.fastqCompletion;
  completion.pending = true;
  fastqCollectionStatus("Checking declared members and completing the library…");
  updateControls();
  try {
    const collection = await api("/uploads/fastq-collections", "POST", completion.body);
    if (state.fastqCompletion !== completion) return;
    state.resources = [...state.resources.filter((choice) => choice.resource_id !== collection.resource_id), collection];
    state.fastqCompletion = null;
    const selected = renderResources(collection.resource_id);
    fastqCollectionStatus(selected
      ? `${collection.label} · library registered. Describe the analysis in your message. Fragment production requires qualified context.`
      : `${collection.label} · library registered, not selected. Remove the current selected inputs before choosing this library.`);
    if (!selected) uploadStatus(`${collection.label} · library registered, not selected. Remove the current selected inputs before choosing this library.`);
  } catch (error) {
    if (state.fastqCompletion !== completion) return;
    completion.pending = false;
    fastqCollectionStatus(`${error.code || "COLLECTION_FAILED"}: ${error.message}`, true);
  }
  updateControls();
}

function transferUpload(upload) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    upload.request = request;
    let query = `?filename=${encodeURIComponent(upload.file.name)}&input_type=${encodeURIComponent(upload.inputType)}`;
    if (upload.index) query += `&index_filename=${encodeURIComponent(upload.index.name)}&source_size=${upload.file.size}`;
    if (upload.fastq) {
      for (const [key, value] of Object.entries(upload.fastq)) query += `&${key}=${encodeURIComponent(value)}`;
    }
    request.open("PUT", `/api/v1/uploads/${encodeURIComponent(upload.uploadId)}${query}`);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.responseType = "json";
    request.upload.addEventListener("progress", (event) => {
      if (state.upload !== upload || !event.lengthComputable || event.total <= 0) return;
      const percent = Math.min(100, Math.round(event.loaded * 100 / event.total));
      element("upload-progress").value = percent;
      uploadStatus(percent === 100 ? "Transfer complete. Waiting for registration…" : `Uploading ${upload.file.name} · ${percent}%`);
    });
    request.onload = () => {
      const data = request.response;
      if (request.status < 200 || request.status >= 300) {
        const error = data && data.error;
        reject(new ApiError(error && error.code || "UPLOAD_FAILED", error && error.message || "The file could not be registered.", request.status));
      } else if (!data || typeof data.resource_id !== "string" || typeof data.label !== "string" || data.input_type !== upload.inputType || data.status !== "registered") {
        reject(new ApiError("RESPONSE_UNAVAILABLE", "The upload response could not be read. Retry the same file to recover its registration."));
      } else {
        resolve(data);
      }
    };
    const interrupted = () => reject(new ApiError("UPLOAD_INTERRUPTED", "The upload was interrupted. Retry the same file to recover its registration."));
    request.onerror = interrupted;
    request.onabort = interrupted;
    request.ontimeout = interrupted;
    request.send(upload.index ? new Blob([upload.file, upload.index]) : upload.file);
  });
}

async function uploadSelectedFile() {
  const upload = state.upload;
  if (!state.resourcesEnabled || !upload || upload.status === "uploading") return;
  upload.status = "uploading";
  element("upload-progress").hidden = false;
  element("upload-progress").value = 0;
  uploadStatus(`Uploading ${upload.file.name}…`);
  updateControls();
  try {
    const sessionId = state.session && state.session.session_id;
    let referencePending = null;
    if (upload.inputType === "fasta") {
      if (!sessionId) throw new Error("Start the scientific request before uploading its reference.");
      referencePending = (await api(`${sessionPath(sessionId)}/reference-prerequisite`)).pending;
      if (!referencePending) throw new Error("No genome-reference request is pending in this Session.");
    }
    const resource = await transferUpload(upload);
    if (upload.inputType === "fasta") {
      uploadStatus("Reference uploaded. Verifying the genome and feature coordinates…");
      await api(`${sessionPath(sessionId)}/reference-prerequisite`, "POST", {
        resource_id: resource.resource_id, expected_generation: referencePending.expected_generation,
        origin_turn_id: referencePending.origin_turn_id });
      if (state.upload !== upload) return;
      state.upload = null;
      element("chat-files").value = "";
      element("upload-index").value = "";
      element("upload-progress").hidden = true;
      uploadStatus("Reference qualified for the original task. Send ‘Use the qualified reference and continue’ to resume.");
      updateControls();
      return;
    }
    if (state.upload !== upload) return;
    if (resource.input_type === "fastq") {
      state.fastqMembers = [...state.fastqMembers.filter((choice) => choice.resource_id !== resource.resource_id), resource];
      state.fastqCompletion = null;
      renderFastqMembers(resource.resource_id);
    } else {
      state.resources = [...state.resources.filter((choice) => choice.resource_id !== resource.resource_id), resource];
    }
    state.upload = null;
    element("chat-files").value = "";
    element("upload-index").value = "";
    element("upload-progress").hidden = true;
    if (resource.input_type === "fastq") {
      renderResources();
      uploadStatus(`${resource.label} · member registered. Upload the remaining declared roles, then complete the FASTQ library.`);
    } else {
      const selected = renderResources(resource.resource_id);
      uploadStatus(selected
        ? `${resource.label} · registered. Describe what you want Agent to do, then send your message.`
        : `${resource.label} · registered, not selected. Remove the current selected inputs before choosing this file.`);
    }
  } catch (error) {
    if (state.upload !== upload) return;
    upload.status = "failed";
    element("upload-progress").hidden = true;
    uploadStatus(`${error.code || "UPLOAD_FAILED"}: ${error.message}`, true);
  }
  updateControls();
}

function prepareDeclaredUpload(file) {
  clearPendingUpload(true);
  if (!file) return;
  // Selection creates a new transfer identity; Retry keeps this same File and ID.
  state.upload = { uploadId: newTurnId(), file, inputType: element("upload-type").value, index: null, status: "selected", request: null,
    fastq: element("upload-type").value === "fastq" ? fastqAttribution() : null };
  if (state.selectedResourceIds.length && state.upload.inputType !== "h5ad") {
    state.selectedResourceIds = selectedResourceIds();
  }
  if (state.upload.inputType !== "fasta") {
    element("resource-choice").value = "";
    if (!selectedCompanion()) element("input-choice").value = "";
  }
  saveResourceSelection();
  saveConveniences();
  inputContext();
  uploadStatus(`${file.name} · selected. Upload this file before sending your request.`);
  updateControls();
}

element("upload-type").addEventListener("change", () => {
  clearPendingUpload();
  renderUploadType();
});
for (const id of ["fastq-library", "fastq-layout", "fastq-role", "fastq-lane", "fastq-chunk", "fastq-compression"]) {
  element(id).addEventListener("change", () => {
    if (!state.upload || state.upload.inputType !== "fastq" || state.upload.status === "uploading") return;
    state.upload = { ...state.upload, uploadId: newTurnId(), fastq: fastqAttribution(), status: "selected", request: null };
    updateControls();
  });
}
element("fastq-members").addEventListener("change", () => { state.fastqCompletion = null; });
element("fastq-collection-label").addEventListener("input", () => { state.fastqCompletion = null; });
element("complete-fastq-collection").addEventListener("click", completeFastqCollection);
function changePairedIndex(index) {
  const upload = state.upload;
  if (!upload || !(["external_fragments", "fasta"].includes(upload.inputType)) || upload.status === "uploading") return;
  // Changing the explicit pair creates a new unit identity. Retry keeps it exact.
  state.upload = { ...upload, uploadId: newTurnId(), index, status: "selected", request: null };
  uploadStatus(`${upload.file.name}${index ? ` + ${index.name}` : ""} · selected. Upload this source before sending your request.`);
  updateControls();
}
element("upload-index").addEventListener("change", () => changePairedIndex(element("upload-index").files[0] || null));
element("clear-upload-index").addEventListener("click", () => {
  element("upload-index").value = "";
  changePairedIndex(null);
});
element("upload-file-button").addEventListener("click", uploadSelectedFile);
element("clear-upload").addEventListener("click", () => clearPendingUpload());
element("resource-choice").addEventListener("change", () => {
  if (state.upload) clearPendingUpload();
  if (element("resource-choice").value && !selectedCompanion()
      && (selectedResourceIds().length === 1 || state.inputSets.some((choice) =>
        choice.input_set_id === element("input-choice").value && (choice.h5ad_companion === true
          || choice.fragments_companion === true || choice.bam_companion === true || choice.fastq_companion === true)))) {
    element("input-choice").value = "";
  }
  if (selectedResourceType() !== "h5ad") element("epizoo-resource-choice").value = "";
  renderSelectedResources();
  saveResourceSelection();
  saveConveniences();
  inputContext();
  updateControls();
});
element("add-scientific-input").addEventListener("click", () => {
  const current = element("resource-choice").value;
  if (!current || !canAddScientificInput()) return;
  state.selectedResourceIds = selectedResourceIds();
  element("resource-choice").value = "";
  renderResources();
});

element("new-session").addEventListener("click", () => createSession());
element("recent-sessions").addEventListener("change", (event) => {
  if (event.target.value) openSession(event.target.value);
});
element("history-notice").addEventListener("click", async () => {
  if (!state.session || !state.session.turns.length) return;
  const button = element("history-notice"), sessionId = state.session.session_id, epoch = state.epoch;
  button.disabled = true;
  try {
    const first = state.session.turns[0].turn_id;
    const page = await api(`${sessionPath(sessionId)}?before_turn_id=${encodeURIComponent(first)}`);
    if (epoch !== state.epoch) return;
    const existing = new Set(state.session.turns.map((turn) => turn.turn_id));
    state.session = {...state.session, turns: [...page.turns.filter((turn) => !existing.has(turn.turn_id)), ...state.session.turns],
      history_truncated: page.history_truncated};
    button.hidden = !page.history_truncated;
    renderHistory();
  } catch (error) { showError(error); }
  finally { button.disabled = false; }
});

async function createSession() {
  const epoch = ++state.epoch;
  state.scientificFiles.clear();
  stopPolling();
  state.opening = true;
  state.submission = null;
  state.activeTurn = null;
  element("retry-turn").hidden = true;
  renderExecution();
  showError(null);
  try {
    const view = await api("/sessions", "POST", {});
    if (epoch !== state.epoch) return;
    element("utterance").value = "";
    closeParameters();
    state.selectedResourceIds = [];
    element("resource-choice").value = "";
    element("input-choice").value = "";
    element("epizoo-resource-choice").value = "";
    saveResourceSelection(); renderSelectedResources();
    displaySession(view);
    saveConveniences();
    element("utterance").focus();
  } catch (error) {
    if (epoch === state.epoch) showError(error);
  } finally {
    if (epoch === state.epoch) { state.opening = false; updateControls(); }
  }
}

element("reopen-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const id = element("session-input").value.trim();
  if (id) openSession(id);
});


function selectOptions(select, rows) {
  select.replaceChildren(...rows.map(([value, label]) => {
    const option = document.createElement("option"); option.value = value; option.textContent = label; return option;
  }));
}

function closeParameters() {
  element("parameter-dialog").close();
  state.parameterEditor = null;
  element("parameter-steps").replaceChildren();
}

async function continueFromResult(revisionId) {
  const sessionId = state.session.session_id;
  element("request-result-choices").querySelectorAll("input").forEach((input) => { input.checked = false; });
  state.requestOptionsKey = null;
  if (state.session.active_revision_id !== revisionId) {
    const view = await api(`${sessionPath(sessionId)}/activate`, "POST", {
      turn_id: newTurnId(), revision_id: revisionId, expected_generation: state.session.generation,
    });
    if (state.session?.session_id !== sessionId) return false;
    displaySession(view, revisionId);
  } else {
    displaySession(state.session, revisionId);
  }
  element("utterance").value = "";
  saveConveniences();
  inputContext();
  showWorkspacePanel("conversation");
  element("utterance").focus();
  return true;
}

async function openResultParameters(revisionId) {
  const sessionId = state.session.session_id;
  const view = await api(`${sessionPath(sessionId)}/revisions/${encodeURIComponent(revisionId)}/parameters`);
  if (state.session?.session_id !== sessionId) return;
  showParameters({sessionId, revisionId, generation: view.generation, steps: view.steps});
}

async function openRequestedParameter(option, questionTurnId) {
  try {
    const sessionId = state.session.session_id;
    const view = await api(`/parameters?tool=${encodeURIComponent(option.tool)}` +
      (option.native_function ? `&function=${encodeURIComponent(option.native_function)}` : ""));
    const recorded = option.previous_revision_id && !option.source
      ? await api(`${sessionPath(sessionId)}/revisions/${encodeURIComponent(option.previous_revision_id)}/parameters`)
      : null;
    if (state.session?.session_id !== sessionId) return;
    const tool = view.tools[0];
    const specs = option.argument === "kwargs"
      ? (tool.kwargs || []).filter((p) => !option.member || p.argument === option.member).map((p) => ({...p, kwarg: true}))
      : tool.parameters.filter((p) => p.argument === option.argument);
    if (!specs.length) throw new Error("The requested parameter scope is unavailable. Specify the scientific operation in your reply.");
    showParameters({sessionId: state.session.session_id, generation: state.session.generation,
      question: {question_turn_id: questionTurnId, handle: option.handle},
      steps: [{tool: option.tool, native_function: option.native_function, functions: tool.functions,
        port: option.port, source: option.source, parameters: specs.map((spec) => {
          let previous = spec.kwarg && !option.member ? option.previous_value?.[spec.argument] : option.previous_value;
          const matches = (recorded?.steps || []).filter((step) => step.tool === option.tool &&
            step.native_function === option.native_function).flatMap((step) => step.parameters.filter((p) =>
              p.argument === spec.argument && Boolean(p.kwarg) === Boolean(spec.kwarg) && p.recorded));
          if (previous === undefined && matches.length === 1) previous = matches[0].value;
          return {...spec, ...(previous !== undefined ? {previous_value: previous} : {})};
        })}]});
  } catch (error) { showError(error); }
}

function showParameters(editor) {
  editor.rows = [];
  state.parameterEditor = editor;
  element("parameter-heading").textContent = editor.revisionId ? "Adjust parameters" : "Supply requested parameter";
  element("parameter-context").textContent = editor.revisionId
    ? `Selected analysis ${editor.revisionId.slice(0, 12)} · Only changes are submitted. Unchanged steps are reused where supported.`
    : "Supply the requested value to continue your pending request.";
  element("run-parameters").textContent = editor.revisionId ? "Run with changes" : "Apply and continue";
  element("parameter-error").textContent = "";
  element("parameter-steps").replaceChildren(...editor.steps.map((step) => {
    const section = document.createElement("section"), title = document.createElement("h3");
    title.textContent = `${step.native_function || step.tool}${step.branch_id ? " · Branch " + step.branch_id + " · " + step.operation_id : step.step_id ? " · " + step.step_id : ""}`;
    section.append(title);
    if (!editor.revisionId && step.functions?.length && !step.native_function) {
      const label = document.createElement("label"); label.textContent = "Requested native function";
      const select = document.createElement("select");
      selectOptions(select, [["", "Choose the function for this value"], ...step.functions.map((name) => [name,name])]);
      select.addEventListener("change", () => {
        editor.rows.filter((row) => row.tool === step.tool).forEach((row) => { row.nativeFunction = select.value; });
      });
      label.append(select); section.append(label);
    }
    if (!step.parameters.length) {
      const note = document.createElement("p"); note.className = "muted small";
      note.textContent = "This step has no adjustable scientific parameters."; section.append(note);
    }
    for (const spec of step.parameters) {
      if (spec.editable === false) {
        const note = document.createElement("p"); note.textContent = `${spec.argument} (read only)`;
        section.append(note, evidenceValueNode(spec.value)); continue;
      }
      const value = spec.value;
      const valueType = spec.choices ? "choice" : value === null && spec.recorded ? "null" :
        typeof value === "boolean" ? "boolean" : typeof value === "number" ? "number" :
        Array.isArray(value) ? (value.every((v) => typeof v === "number") ? "numbers" : "strings") :
        typeof value === "string" ? "string" : "auto";
      const raw = !spec.recorded ? "" : spec.choices ? String(spec.choices.indexOf(value)) :
        Array.isArray(value) ? value.join(", ") : value === null ? "" : String(value);
      const row = {tool: step.tool, port: step.port, source: step.source, needsFunction: Boolean(step.functions?.length), nativeFunction: step.native_function, stepId: step.step_id, spec, valueType, raw};
      editor.rows.push(row);
      const field = document.createElement("div"); field.className = "parameter-row";
      const label = document.createElement("label"); label.textContent = spec.argument;
      const input = document.createElement(spec.choices || valueType === "boolean" ? "select" : "input");
      input.setAttribute("aria-label", `${step.step_id || step.tool} ${spec.argument}`);
      if (spec.choices) selectOptions(input, [["", "Choose value"], ...spec.choices.map((v,i) => [String(i), String(v)])]);
      else if (valueType === "boolean") selectOptions(input, [["", "Choose value"], ["true", "true"], ["false", "false"]]);
      else { input.type = valueType === "number" ? "number" : "text"; input.step = "any"; }
      input.value = raw; row.input = input;
      const typeInput = document.createElement("select"); row.typeInput = typeInput;
      typeInput.setAttribute("aria-label", `${step.step_id || step.tool} ${spec.argument} value type`);
      selectOptions(typeInput, spec.choices ? [["choice", "Choice"]] : [["auto", "Number or text"], ["string", "Text"], ["number", "Number"], ["boolean", "Boolean"], ["null", "null"], ["strings", "Text list"], ["numbers", "Number list"]]);
      typeInput.value = valueType;
      const help = document.createElement("p"); help.className = "muted small";
      help.textContent = spec.description + (spec.recorded ? "" :
        Object.hasOwn(spec, "default_when_omitted") ? ` · Native/default when omitted: ${JSON.stringify(spec.default_when_omitted)}` :
        editor.question ? " · Enter a value to resolve this question." : " · Value not recorded; leave empty to preserve omission.");
      if (Object.hasOwn(spec, "previous_value")) help.textContent += ` · Recorded previous value: ${JSON.stringify(spec.previous_value)}`;
      label.append(input); field.append(label, typeInput, help); section.append(field);
    }
    if (step.recorded) {
      const details = document.createElement("details"), summary = document.createElement("summary");
      summary.textContent = "Recorded execution details (read only)";
      details.append(summary, evidenceValueNode(step.recorded)); section.append(details);
    }
    return section;
  }));
  element("run-parameters").disabled = !editor.rows.length;
  element("parameter-dialog").showModal();
}

function parameterValue(row) {
  const raw = row.input.value, type = row.typeInput.value;
  if (type === "null") return null;
  if (!raw.trim()) throw new Error(`Enter ${row.spec.argument}, or keep its recorded value unchanged.`);
  if (type === "choice") return row.spec.choices[Number(raw)];
  if (type === "string") return raw;
  if (type === "boolean") {
    if (!["true", "false"].includes(raw)) throw new Error("Choose true or false.");
    return raw === "true";
  }
  if (type === "strings") return raw.split(",").map((s) => s.trim());
  if (type === "numbers") {
    const values = raw.split(",").map((s) => Number(s.trim()));
    if (values.some((v) => !Number.isFinite(v))) throw new Error("Enter comma-separated numbers.");
    return values;
  }
  if (type === "auto") {
    if (["true", "false"].includes(raw)) return raw === "true";
    return Number.isFinite(Number(raw)) ? Number(raw) : raw;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || type === "integer" && !Number.isInteger(value)) throw new Error(`Parameter ${row.spec.argument} requires a ${type === "integer" ? "whole number" : "number"}.`);
  return value;
}

function collectedParameters() {
  const rows = new Map();
  for (const row of state.parameterEditor.rows) {
    if (row.input.value === row.raw && row.typeInput.value === row.valueType) continue;
    if (row.needsFunction && !row.nativeFunction) throw new Error("Choose the requested native function for this parameter.");
    const value = parameterValue(row);
    if (row.spec.recorded && JSON.stringify(value) === JSON.stringify(row.spec.value)) continue;
    const argument = row.spec.kwarg ? "kwargs" : row.spec.argument;
    const key = JSON.stringify([row.stepId, row.tool, row.nativeFunction, row.port, row.source, argument]);
    let setting = rows.get(key);
    if (!setting) {
      setting = {tool: row.tool, argument, value: row.spec.kwarg ? {} : null};
      if (row.nativeFunction) setting.native_function = row.nativeFunction;
      if (row.stepId) setting.step_id = row.stepId;
      if (row.port) { setting.port = row.port; setting.source = row.source; }
      rows.set(key, setting);
    }
    if (row.spec.kwarg) setting.value[row.spec.argument] = value;
    else setting.value = value;
  }
  return Array.from(rows.values());
}

async function refreshRequestOptions(sessionId = state.session?.session_id) {
  if (!sessionId) return;
  const query = new URLSearchParams();
  const ids = [...new Set([...selectedResourceIds(), ...state.attachments.map((a) => a.resource.resource_id)])];
  ids.forEach((id) => query.append("resource_ids", id));
  const key = JSON.stringify([sessionId, state.session?.generation, ids]);
  if (state.requestOptionsKey === key) return;
  const epoch = ++state.optionsEpoch;
  try {
    const view = await api(`${sessionPath(sessionId)}/interaction-options?${query}`);
    if (epoch !== state.optionsEpoch || state.session?.session_id !== sessionId) return;
    const checkedResults = new Set(Array.from(element("request-result-choices").querySelectorAll("input:checked")).map((input) => input.value));
    state.requestOptions = view;
    state.requestOptionsKey = key;
    element("request-results-panel").hidden = !view.results.length;
    element("request-result-choices").replaceChildren(...view.results.map((result, index) => {
      const label = document.createElement("label"), input = document.createElement("input");
      input.type = "checkbox"; input.value = JSON.stringify(result.identity);
      input.checked = checkedResults.has(input.value);
      label.append(input, document.createTextNode(`${result.label} · ${result.tool}${result.current ? " · Current" : ""}`));
      return label;
    }));
  } catch (error) { showError(error); }
}

element("cancel-parameters").addEventListener("click", closeParameters);
element("parameter-dialog").addEventListener("cancel", closeParameters);
element("parameter-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const editor = state.parameterEditor;
  try {
    if (!editor || editor.sessionId !== state.session?.session_id || editor.generation !== state.session.generation)
      throw new Error("The conversation changed. Reopen the parameter dialog for the result you want.");
    const overrides = collectedParameters();
    if (!overrides.length) throw new Error("Change at least one parameter before running.");
    element("run-parameters").disabled = true;
    if (editor.revisionId && !await continueFromResult(editor.revisionId)) return;
    const body = {turn_id: newTurnId(), expected_generation: state.session.generation, utterance: "",
      profile_id: element("model-choice").value, parameter_overrides: overrides};
    if (editor.revisionId) body.rerun_revision_id = editor.revisionId;
    if (editor.question) body.clarification_choice = editor.question;
    closeParameters();
    startSubmission(body);
  } catch (error) {
    element("parameter-error").textContent = error.message;
    element("run-parameters").disabled = false;
  }
});

function startSubmission(body) {
  state.submission = Object.freeze({ sessionId: state.session.session_id, body: Object.freeze(body) });
  state.activeTurn = { session_id: state.session.session_id, turn_id: body.turn_id, status: "submitted", steps: [] };
  saveConveniences();
  postSubmission(state.submission);
}

function submitRequest(utterance, predecessorTurnId = null, clarificationChoice = null, candidateSelection = null) {
  if (element("submit-turn").disabled || !state.session) return;
  if (!utterance.trim() && !clarificationChoice) return;
  const body = {
    turn_id: newTurnId(), expected_generation: state.session.generation,
    utterance, profile_id: element("model-choice").value,
  };
  if (candidateSelection !== null) body.candidate_selection = Object.freeze(candidateSelection);
  const selectedResults = Array.from(element("request-result-choices").querySelectorAll("input:checked"))
    .map((input) => JSON.parse(input.value));
  if (selectedResults.length) body.selected_results = selectedResults;
  if (predecessorTurnId !== null) body.predecessor_turn_id = predecessorTurnId;
  if (clarificationChoice !== null) body.clarification_choice = Object.freeze(clarificationChoice);
  const selectedIds = selectedResourceIds();
  const h5ads = state.attachments.filter((a) => a.resource.input_type === "h5ad").map((a) => a.resource.resource_id);
  if (state.attachments.length) body.attachment_ids = Object.freeze(state.attachments.map((a) => a.resource.resource_id));
  if (h5ads.length && selectedIds.length) {
    showError(new Error("New H5AD attachments and previously selected inputs are both present. Remove the earlier input selection or the new H5AD attachments before sending.")); return;
  }
  const lastTurn = state.session.turns[state.session.turns.length - 1];
  const pendingRoles = lastTurn && lastTurn.response && lastTurn.response.clarification && ["missing_input_roles", "missing_fastq_declarations", "missing_bam_declarations", "missing_fragments_declarations", "missing_annotation_declarations"].includes(lastTurn.response.clarification.reason);
  if (h5ads.length === 1 && state.attachments.length === 1 && !pendingRoles) {
    body.resource_id = h5ads[0];
    if (element("epizoo-resource-choice").value) body.epizoo_resource_id = element("epizoo-resource-choice").value;
    if (element("input-choice").value) body.input_set_id = element("input-choice").value;
  }
  else if (selectedIds.length > 1 || state.selectedResourceIds.length || selectedResourceType() === "groups") {
    body.resource_ids = Object.freeze(selectedIds);
    if (element("input-choice").value) body.input_set_id = element("input-choice").value;
  }
  else if (selectedIds.length) {
    if (!element("resource-choice").value) element("resource-choice").value = selectedIds[0];
    body.resource_id = element("resource-choice").value;
    if (selectedCompanion()) body.input_set_id = element("input-choice").value;
    if (selectedResourceType() === "h5ad" && element("epizoo-resource-choice").value) body.epizoo_resource_id = element("epizoo-resource-choice").value;
  }
  else if (element("input-choice").value) body.input_set_id = element("input-choice").value;
  startSubmission(body);
}

element("turn-form").addEventListener("submit", (event) => {
  event.preventDefault();
  submitRequest(element("utterance").value);
});

element("retry-turn").addEventListener("click", () => {
  if (state.submission && !state.posting) postSubmission(state.submission);
});

element("cancel-turn").addEventListener("click", async () => {
  if (!state.activeTurn || element("cancel-turn").disabled) return;
  const turn = state.activeTurn;
  const epoch = state.epoch;
  element("cancel-turn").disabled = true;
  try {
    const view = await api(`${turnPath(turn.session_id, turn.turn_id)}/cancel`, "POST", {});
    if (epoch !== state.epoch) return;
    state.activeTurn = view;
    mergeTurn(view);
    renderExecution();
    // Cooperative cancellation is a request; poll the persisted terminal state.
    watchTurn(turn.session_id, turn.turn_id);
  } catch (error) {
    if (epoch === state.epoch) { showError(error); updateControls(); }
  }
});

element("load-evidence").addEventListener("click", () => loadEvidence());
element("evidence-output").addEventListener("change", () => loadEvidence());
element("evidence-panel").addEventListener("toggle", () => {
  if (element("evidence-panel").open && state.revision && !state.evidence) {
    element("evidence-content").textContent = "Choose an accepted field and select Show evidence to read its exact value.";
  }
});
element("detail-form").addEventListener("submit", (event) => {
  event.preventDefault();
  loadEvidence(true);
});

element("model-choice").addEventListener("change", modelDescription);
element("input-choice").addEventListener("change", () => {
  if (state.upload) clearPendingUpload();
  if (element("input-choice").value && !selectedCompanion()) {
    if (selectedResourceIds().length < 2 && !state.selectedResourceIds.length) {
      state.selectedResourceIds = [];
      element("resource-choice").value = "";
    }
    element("epizoo-resource-choice").value = "";
  }
  renderSelectedResources();
  saveResourceSelection();
  saveConveniences();
  inputContext();
  updateControls();
});
element("epizoo-resource-choice").addEventListener("change", () => {
  inputContext();
  saveConveniences();
});
element("show-conversation").addEventListener("click", () => showWorkspacePanel("conversation", true));
element("show-results").addEventListener("click", () => showWorkspacePanel("results", true));


element("result-choice").addEventListener("change", () => {
  if (element("result-choice").value) loadRevision(element("result-choice").value);
});
element("utterance").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)
      && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    element("turn-form").requestSubmit();
  }
});
element("utterance").addEventListener("input", saveConveniences);

async function initialize() {
  const conveniences = readConveniences();
  let resources;
  if (typeof conveniences.draft === "string") element("utterance").value = conveniences.draft.slice(0, 4096);
  try {
    const [, models, inputs, epizoo] = await Promise.all([api("/health"), api("/models"), api("/input-sets"), api("/epizoo-resources")]);
    state.models = models.choices;
    state.inputSets = inputs.choices;
    state.epizooResources = epizoo.choices;
    element("model-choice").replaceChildren(...state.models.map((choice) => {
      const option = document.createElement("option");
      option.value = choice.profile_id;
      option.textContent = `${choice.display_label}${choice.is_default ? " (default)" : ""}`;
      option.title = choice.display_label;
      return option;
    }));
    const selected = state.models.find((choice) => choice.profile_id === conveniences.profileId) || state.models.find((choice) => choice.is_default) || state.models[0];
    if (selected) element("model-choice").value = selected.profile_id;
    element("model-choice").disabled = !state.models.length;
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "No structured inputs";
    element("input-choice").replaceChildren(none, ...inputs.choices.map((choice) => {
      const option = document.createElement("option");
      option.value = choice.input_set_id;
      option.textContent = `${choice.display_label}${choice.fragments_default === true ? " (fragments default)" : choice.bam_default === true ? " (BAM default)" : ""}`;
      return option;
    }));
    element("input-choice").disabled = !inputs.choices.length;
    if (state.inputSets.some((choice) => choice.input_set_id === conveniences.inputSetId)) {
      element("input-choice").value = conveniences.inputSetId;
    }
    const resourceDefault = document.createElement("option");
    resourceDefault.value = "";
    resourceDefault.textContent = "Use applicable configured default";
    element("epizoo-resource-choice").replaceChildren(resourceDefault, ...state.epizooResources.map((choice) => {
      const option = document.createElement("option");
      option.value = choice.resource_id;
      option.textContent = `${choice.display_label} · ${choice.species}${choice.is_default ? " (default)" : ""}`;
      return option;
    }));
    resources = loadResources();
    inputContext();
    modelDescription();
    if (typeof conveniences.sessionId === "string" && conveniences.sessionId) {
      await openSession(conveniences.sessionId, typeof conveniences.polledTurnId === "string" ? conveniences.polledTurnId : "",
        typeof conveniences.viewedRevisionId === "string" ? conveniences.viewedRevisionId : null);
        if (state.revision) renderResultSelection();
      if (typeof conveniences.draft === "string") element("utterance").value = conveniences.draft.slice(0, 4096);
        if (conveniences.workspacePanel === "results" && state.viewedRevisionId) showWorkspacePanel("results");
    } else {
      await createSession();
      if (typeof conveniences.draft === "string") element("utterance").value = conveniences.draft.slice(0, 4096);
    }
  } catch (error) {
    showError(error);
  }
  await resources;
  if (selectedResourceType() === "h5ad" && state.epizooResources.some((choice) =>
      choice.resource_id === conveniences.epizooResourceId)) {
    element("epizoo-resource-choice").value = conveniences.epizooResourceId;
  }
  inputContext();
  state.initializing = false;
  if (state.session) saveConveniences();
  updateControls();
}

initialize();

// Explicit sharing is separate from ordinary attachment upload and analysis.
element("shared-reference-controls").addEventListener("toggle", async (event) => {
  if (!event.target.open) return;
  const target = element("private-reference-choices");
  target.replaceChildren();
  const sessionId = state.session && state.session.session_id;
  if (!sessionId) { target.textContent = "Open a session to see its references."; return; }
  try {
    const result = await api(`${sessionPath(sessionId)}/references`);
    if (!state.session || state.session.session_id !== sessionId) return;
    if (!result.enabled) { target.textContent = "A shared reference library is not configured."; return; }
    if (!result.choices.length) { target.textContent = "This session has no reusable references yet."; return; }
    for (const reference of result.choices) {
      const row = document.createElement("p");
      const name = document.createElement("span"); name.textContent = reference.label + " ";
      const button = document.createElement("button");
      button.type = "button"; button.className = "secondary";
      button.textContent = "Share for future laboratory analyses";
      button.addEventListener("click", async () => {
        if (!state.session || state.session.session_id !== sessionId) return;
        button.disabled = true;
        try {
          await api(`${sessionPath(sessionId)}/references/${encodeURIComponent(reference.resource_id)}/share`, "POST", {shared_use: true});
          button.textContent = "Available in the shared library";
        } catch (error) { button.disabled = false; showError(error); }
      });
      row.append(name, button); target.append(row);
    }
  } catch (error) { showError(error); }
});

// Native interaction holds display conveniences only. Scientific state lives in
// the same server Session/Revision and RunStore owners as AI Research.
const nativeView = {session: null, fields: [], resources: [], metadataEpoch: 0, timer: null, active: null};
const nativeOption = (value, label) => { const e = document.createElement('option'); e.value = value; e.textContent = label; return e; };
const nativeText = (tag, text, css = '') => { const e = document.createElement(tag); e.textContent = text; e.className = css; return e; };
function nativeError(error) { element('native-error').textContent = error ? error.message || String(error) : ''; }
async function nativeOpen() {
  if (nativeView.session) return;
  const saved = sessionStorage.getItem('agent.native.session');
  nativeView.session = saved ? await api(`/native/sessions/${encodeURIComponent(saved)}`) : await api('/native/sessions', 'POST', {});
  sessionStorage.setItem('agent.native.session', nativeView.session.session_id);
  const attachments = await api(`${sessionPath(nativeView.session.session_id)}/attachments`);
  nativeView.resources = attachments.choices;
  nativeRenderResults();
  const running = nativeView.session.calls.find(c => ['submitted','planning','validated','running','finalizing'].includes(c.status));
  if (running) { nativeView.active = running.call_id; nativePoll(); }
}
async function selectAgentMode(mode) {
  document.body.dataset.mode = mode;
  element('mode-research').setAttribute('aria-pressed', String(mode === 'research'));
  element('mode-native').setAttribute('aria-pressed', String(mode === 'native'));
  element('mode-research').classList.toggle('secondary', mode !== 'research');
  element('mode-native').classList.toggle('secondary', mode !== 'native');
  element('native-workspace').hidden = mode !== 'native';
  sessionStorage.setItem('agent.mode', mode);
  if (mode !== 'native') return;
  try {
    await nativeOpen();
    if (element('native-package').options.length === 1) {
      const catalog = await api('/native/packages');
      for (const p of catalog.packages) { const option = nativeOption(p.name, p.name); option.dataset.language = p.language; element('native-package').append(option); }
    }
  } catch (error) { nativeError(error); }
}
element('mode-native').addEventListener('click', () => selectAgentMode('native'));
element('mode-research').addEventListener('click', () => selectAgentMode('research'));
element('native-new').addEventListener('click', async () => {
  if (nativeView.active) return;
  try {
    nativeView.session = await api('/native/sessions', 'POST', {});
    nativeView.resources = []; sessionStorage.setItem('agent.native.session', nativeView.session.session_id);
    nativeRenderResults(); nativeRefreshSources();
  } catch (error) { nativeError(error); }
});
element('native-package').addEventListener('change', async () => {
  const epoch = ++nativeView.metadataEpoch;
  nativeError(null); element('native-function').disabled = true; element('native-run').disabled = true;
  element('native-inputs').replaceChildren(); element('native-fields').replaceChildren();
  element('native-description').textContent = 'Reading installed functions…';
  try {
    const packageName = element('native-package').value;
    if (!packageName) return;
    const data = await api(`/native/functions?package=${encodeURIComponent(packageName)}`);
    if (epoch !== nativeView.metadataEpoch) return;
    element('native-function').replaceChildren(nativeOption('', 'Choose a function…'), ...data.functions.map(f => nativeOption(f.name, `${packageName}.${f.name}`)));
    element('native-function').disabled = false;
    element('native-description').textContent = `${packageName} ${data.version}`;
  } catch (error) { nativeError(error); }
});
element('native-function').addEventListener('change', async () => {
  const epoch = ++nativeView.metadataEpoch;
  nativeError(null); element('native-run').disabled = true;
  try {
    const fn = element('native-function').value;
    if (!fn) return;
    const data = await api(`/native/functions?package=${encodeURIComponent(element('native-package').value)}&function=${encodeURIComponent(fn)}`);
    if (epoch !== nativeView.metadataEpoch) return;
    const metadata = data.functions[0];
    element('native-description').textContent = metadata.description || metadata.limitation || '';
    nativeView.fields = []; element('native-inputs').replaceChildren(); element('native-fields').replaceChildren();
    if (metadata.limitation) { nativeError(new Error(metadata.limitation)); return; }
    for (const spec of metadata.parameters) {
      if (['VAR_KEYWORD', 'VAR_POSITIONAL'].includes(spec.parameter_kind)) continue;
      nativeView.fields.push(nativeField(spec));
    }
    nativeRefreshSources(); element('native-run').disabled = false;
  } catch (error) { nativeError(error); }
});
function nativeField(spec) {
  const row = nativeText('div', '', 'native-argument');
  const enabled = document.createElement('input'); enabled.type = 'checkbox'; enabled.checked = spec.required;
  const label = nativeText('label', `${spec.name}${spec.required ? ' · required' : ''}`); label.prepend(enabled); row.append(label);
  if (!spec.required) row.append(nativeText('p', `Default: ${spec.default_known ? JSON.stringify(spec.default) : 'Native default'}`, 'muted small'));
  const controls = document.createElement('div'); controls.hidden = !enabled.checked;
  enabled.addEventListener('change', () => { controls.hidden = !enabled.checked; });
  const mode = document.createElement('select'); mode.setAttribute('aria-label', `${spec.name} input type`);
  for (const [value, name] of [['value', 'Parameter value'], ['null', 'Explicit null'], ['anndata', 'H5AD → AnnData'], ['r_matrix', 'H5AD → R matrix'], ['dataframe', 'CSV / TSV → DataFrame'], ['column', 'CSV / TSV column → list'], ['matrix', 'CSV / TSV → matrix'], ['array', 'Numeric array'], ['sparse', 'Sparse matrix'], ['numpy', 'NumPy array file'], ['file', 'Native file input'], ['accepted', 'Accepted result'], ['output', 'Managed output filename']]) mode.append(nativeOption(value, name));
  const objectInput = spec.required && (/AnnData/.test(spec.annotation) || ['adata', 'data'].includes(spec.name));
  if (objectInput) mode.value = 'anndata';
  if (spec.input_kind === 'file') mode.value = 'file';
  if (!spec.required && spec.default_known && spec.default === null) mode.value = 'null';
  const value = document.createElement(spec.kind === 'list' || spec.kind === 'value' ? 'textarea' : 'input');
  value.setAttribute('aria-label', `${spec.name} value`);
  if (value.tagName === 'INPUT') value.type = spec.kind === 'integer' || spec.kind === 'number' ? 'number' : 'text';
  if (spec.kind === 'number') value.step = 'any';
  value.value = spec.default_known ? (spec.kind === 'string' ? spec.default : JSON.stringify(spec.default)) : '';
  value.placeholder = spec.kind === 'value' || spec.kind === 'list' ? 'JSON value, list or object' : spec.kind;
  const boolean = document.createElement('select'); boolean.setAttribute('aria-label', `${spec.name} boolean`);
  boolean.append(nativeOption('true', 'True'), nativeOption('false', 'False')); boolean.value = String(spec.default ?? true);
  const choice = document.createElement('select'); choice.setAttribute('aria-label', `${spec.name} choice`);
  for (const item of spec.choices || []) choice.append(nativeOption(JSON.stringify(item), String(item)));
  if (spec.default_known) choice.value = JSON.stringify(spec.default);
  const source = document.createElement('select'); source.setAttribute('aria-label', `${spec.name} source`);
  const upload = document.createElement('input'); upload.type = 'file'; upload.setAttribute('aria-label', `${spec.name} upload`);
  const separator = document.createElement('select'); separator.setAttribute('aria-label', `${spec.name} table format`);
  separator.append(nativeOption(',', 'CSV (comma separated)'), nativeOption('\t', 'TSV (tab separated)'));
  const column = document.createElement('input'); column.placeholder = 'Exact column name'; column.setAttribute('aria-label', `${spec.name} column`);
  const sparseFormat = document.createElement('select'); sparseFormat.setAttribute('aria-label', `${spec.name} sparse format`);
  sparseFormat.append(nativeOption('mtx', 'Matrix Market'), nativeOption('npz', 'SciPy NPZ'));
  const orientation = document.createElement('select'); orientation.setAttribute('aria-label', `${spec.name} matrix orientation`);
  orientation.append(nativeOption('unchanged', 'Cells × features'), nativeOption('transpose', 'Features × cells (transpose)'));
  const accepted = document.createElement('select'); accepted.setAttribute('aria-label', `${spec.name} accepted result`);
  const update = () => {
    const file = ['anndata','r_matrix','dataframe','column','matrix','sparse','numpy','file'].includes(mode.value);
    source.hidden = upload.hidden = !file;
    separator.hidden = !['dataframe','column','matrix'].includes(mode.value); column.hidden = mode.value !== 'column';
    orientation.hidden = mode.value !== 'r_matrix';
    sparseFormat.hidden = mode.value !== 'sparse'; accepted.hidden = mode.value !== 'accepted';
    value.hidden = !['value','array','output'].includes(mode.value) || mode.value === 'value' && spec.kind === 'boolean';
    boolean.hidden = mode.value !== 'value' || spec.kind !== 'boolean' || Boolean(spec.choices?.length);
    choice.hidden = mode.value !== 'value' || !spec.choices?.length;
    if (!choice.hidden) value.hidden = true;
  };
  mode.addEventListener('change', update);
  upload.addEventListener('change', async () => {
    const file = upload.files[0]; if (!file) return;
    element('native-run').disabled = true; nativeError(null);
    try {
      const path = `${sessionPath(nativeView.session.session_id)}/uploads/${crypto.randomUUID()}?input_type=attachment&filename=${encodeURIComponent(file.name)}`;
      const response = await fetch('/api/v1' + path, {method:'PUT', headers:{'Content-Type':'application/octet-stream'}, body:file});
      const data = await response.json(); if (!response.ok) throw new Error(data.error?.message || 'Upload failed');
      nativeView.resources.push(data); nativeRefreshSources(); source.value = data.resource_id;
    } catch (error) { nativeError(error); }
    finally { element('native-run').disabled = false; }
  });
  controls.append(mode, value, boolean, choice, source, upload, separator, column, sparseFormat, orientation, accepted); row.append(controls);
  (spec.required ? element('native-inputs') : element('native-fields')).append(row); update();
  return {spec, enabled, mode, value, boolean, choice, source, separator, column, sparseFormat, orientation, accepted};
}
function nativeRefreshSources() {
  for (const row of nativeView.fields) {
    const previous = row.source.value;
    row.source.replaceChildren(nativeOption('', 'Choose an uploaded file…'), ...nativeView.resources.map(r => nativeOption(r.resource_id, r.label)));
    row.source.value = previous;
    const selected = row.accepted.value;
    const language = element('native-package').selectedOptions[0]?.dataset.language;
    const choices = (nativeView.session?.calls || []).flatMap((call, i) => (call.outputs || []).filter(output => output.encoding !== 'r' || language === 'R').map(output => nativeOption(JSON.stringify({revision_id:call.revision_id, output:output.key}), `${i+1}. ${call.function} · ${output.key.startsWith('modified:') ? 'Modified ' + output.key.slice(9) : output.key === 'return' ? 'Return value' : output.key} · ${output.type.split('.').pop()}${output.shape ? ' ' + output.shape.join(' × ') : ''}`)));
    row.accepted.replaceChildren(nativeOption('', 'Choose an accepted output…'), ...choices); row.accepted.value = selected;
  }
}
function nativeValue(row) {
  if (row.spec.choices?.length) return JSON.parse(row.choice.value);
  if (row.spec.kind === 'boolean') return row.boolean.value === 'true';
  if (row.spec.kind === 'string') return row.value.value;
  const value = JSON.parse(row.value.value);
  if (row.spec.kind === 'integer' && !Number.isSafeInteger(value)) throw new Error(`${row.spec.name}: enter an integer representable exactly by this browser`);
  return value;
}
element('native-form').addEventListener('submit', async event => {
  event.preventDefault(); nativeError(null);
  const body = {call_id:crypto.randomUUID(), expected_generation:nativeView.session.generation, package:element('native-package').value, function:element('native-function').value, inputs:{}, parameters:{}, outputs:{}};
  try {
    for (const row of nativeView.fields) {
      if (!row.enabled.checked) continue;
      const key = row.spec.name, mode = row.mode.value;
      if (mode === 'null') body.parameters[key] = null;
      else if (mode === 'value') body.parameters[key] = nativeValue(row);
      else if (mode === 'output') body.outputs[key] = row.value.value;
      else if (mode === 'array') body.inputs[key] = {adapter:mode, value:JSON.parse(row.value.value)};
      else if (mode === 'accepted') body.inputs[key] = {adapter:mode, ...JSON.parse(row.accepted.value)};
      else {
        if (!row.source.value) throw new Error(`Choose a file for ${key}`);
        body.inputs[key] = {adapter:mode, resource_id:row.source.value};
        if (['dataframe','matrix','column'].includes(mode)) body.inputs[key].separator = row.separator.value;
        if (mode === 'column') body.inputs[key].column = row.column.value;
        if (mode === 'sparse') body.inputs[key].format = row.sparseFormat.value;
        if (mode === 'r_matrix') { body.inputs[key].format = 'h5ad'; body.inputs[key].orientation = row.orientation.value; }
      }
    }
    element('native-run').disabled = true; element('native-status').textContent = 'Submitting function…';
    await api(`/native/sessions/${nativeView.session.session_id}/calls`, 'POST', body);
    nativeView.active = body.call_id; element('native-cancel').hidden = false;
    nativePoll();
  } catch (error) { nativeError(error); element('native-run').disabled = false; element('native-status').textContent = ''; }
});
async function nativePoll() {
  try {
    nativeView.session = await api(`/native/sessions/${nativeView.session.session_id}`); nativeRenderResults();
    const call = nativeView.session.calls.find(r => r.call_id === nativeView.active);
    element('native-status').textContent = call?.status || 'Submitted';
    if (call && ['succeeded','failed','cancelled','interrupted','awaiting_acceptance'].includes(call.status)) {
      nativeView.active = null; element('native-run').disabled = !element('native-function').value; element('native-cancel').hidden = true; nativeRefreshSources(); return;
    }
    nativeView.timer = setTimeout(nativePoll, POLL_INTERVAL_MS);
  } catch (error) { nativeError(error); }
}
element('native-cancel').addEventListener('click', async () => {
  try { await api(`/native/sessions/${nativeView.session.session_id}/calls/${nativeView.active}/cancel`, 'POST', {}); }
  catch (error) { nativeError(error); }
});
function nativeRenderResults() {
  const container = element('native-call-results'); container.replaceChildren();
  if (!nativeView.session.calls.length) { container.append(nativeText('p', 'Your function results will appear here.', 'muted')); return; }
  for (const call of [...nativeView.session.calls].reverse()) {
    const section = nativeText('article', '', 'native-result');
    section.append(nativeText('h3', call.function || 'Function call'), nativeText('p', call.status));
    if (call.status === 'awaiting_acceptance') { const recover = nativeText('button', 'Finish saved result'); recover.type = 'button'; recover.addEventListener('click', async () => { try { nativeView.session = await api(`/native/sessions/${nativeView.session.session_id}/calls/${call.call_id}/recover`, 'POST', {}); nativeRenderResults(); nativeRefreshSources(); } catch(error) { nativeError(error); } }); section.append(recover); }
    if (call.native_error) section.append(nativeText('p', `Native execution: ${call.native_error.native_execution}. ${call.native_error.message}`, 'turn-error'));
    if (call.error) section.append(nativeText('p', call.error, 'turn-error'));
    if (!call.native_error) for (const message of new Set((call.errors || []).map(e => e.message || e.code))) section.append(nativeText('p', message, 'turn-error'));
    const downloads = new Map((call.downloads || []).map(d => [d.filename, d]));
    const shownImages = new Set(), shownFiles = new Set();
    const image = (container, file) => {
      if (shownImages.has(file) || !/\.(png|jpg|jpeg|webp)$/i.test(file) || !downloads.has(file)) return;
      const img = document.createElement('img'); img.className = 'analysis-figure'; img.alt = 'Native graphical output'; img.loading = 'lazy';
      img.src = nativeDownloadUrl(call, downloads.get(file)); container.append(img); shownImages.add(file);
    };
    if (call.preview) {
      for (const [name, value] of [['Return value', call.preview.return_value], ...Object.entries(call.preview.modified_inputs).map(([k,v]) => ['Modified input: '+k,v])]) {
        section.append(nativeText('h4', name));
        if (value.encoding === 'json') section.append(nativeText('p', value.value === null ? 'None' : String(value.value)));
        else nativeRenderValue(section, value, image);
      }
      if (call.preview.produced_files.length) {
        section.append(nativeText('h4', 'Produced files'));
        for (const file of call.preview.produced_files) {
          const download = downloads.get(file.file);
          if (download && !shownFiles.has(file.file)) nativeRenderDownload(section, call, download, file.name);
          shownFiles.add(file.file); image(section, file.file);
        }
      }
      if (call.preview.return_value.encoding === 'json' && call.preview.return_value.value === null && !Object.keys(call.preview.modified_inputs).length && !call.preview.produced_files.length) section.append(nativeText('p', 'Completed without a data result.'));
    }
    const remaining = (call.downloads || []).filter(d => !shownFiles.has(d.filename));
    if (remaining.length) section.append(nativeText('h4', 'Downloads'));
    for (const download of remaining) nativeRenderDownload(section, call, download, download.label);
    container.append(section);
  }
}
selectAgentMode(sessionStorage.getItem('agent.mode') || 'research');

function nativeDownloadUrl(call, download) {
  return `/api/v1/sessions/${nativeView.session.session_id}/revisions/${call.revision_id}/scientific-artifacts/${download.handle}`;
}
function nativeRenderDownload(container, call, download, label) {
  const link = nativeText('a', label); link.href = nativeDownloadUrl(call, download); link.download = download.filename;
  const line = document.createElement('p'); line.append(link); container.append(line);
}
function nativeRenderValue(container, value, image) {
  container.append(nativeText('p', `${value.type}${value.shape ? ' · ' + value.shape.join(' × ') : ''}${value.dtype ? ' · ' + value.dtype : ''}`, 'muted small'));
  if (value.encoding === 'matplotlib') { image(container, value.file); container.append(nativeText('p', value.limitation, 'muted small')); }
  if (value.encoding === 'r') {
    container.append(nativeText('p', `R type: ${value.r_type}; classes: ${(value.classes || []).join(', ')}${value.slots?.length ? '; slots: ' + value.slots.join(', ') : ''}`, 'muted small'));
    if (value.names?.length) container.append(nativeText('p', 'Names: ' + value.names.join(', '), 'muted small'));
  }
  if (['dataframe','series'].includes(value.encoding) && value.preview) {
    const data = JSON.parse(value.preview), table = document.createElement('table');
    const header = document.createElement('tr');
    for (const label of ['Index', ...(data.columns || [data.name || 'Value'])]) header.append(nativeText('th', String(label)));
    table.append(header);
    data.data.forEach((values, i) => { const row = document.createElement('tr'); for (const item of [data.index[i], ...(Array.isArray(values) ? values : [values])]) row.append(nativeText('td', item === null ? 'null' : String(item))); table.append(row); });
    container.append(table);
  } else if (value.preview !== undefined) container.append(nativeText('pre', typeof value.preview === 'string' ? value.preview : JSON.stringify(value.preview, null, 2)));
  if (value.items) { for (const [i, item] of value.items.entries()) { container.append(nativeText('h5', value.encoding === 'dict' ? String(item.key.value ?? i) : String(i))); nativeRenderValue(container, value.encoding === 'dict' ? item.value : item, image); } }
  if (value.encoding === 'json') container.append(nativeText('p', value.value === null ? 'None' : String(value.value)));
  if (value.length !== undefined) container.append(nativeText('p', `${value.length} members; full values are saved.`, 'muted small'));
}
