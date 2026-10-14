"use strict";

// Browser storage holds display conveniences. Session generations and history
// are reconstructed from Agent whenever a session is opened or a turn completes.
const STORAGE_KEY = "agent.web.presentation.v1";
const RESOURCE_STORAGE_KEY = "agent.web.selected-resource.v1";
const MULTI_RESOURCE_STORAGE_KEY = "agent.web.selected-resources.v1";
const MAX_SELECTED_RESOURCES = 8;
const POLL_INTERVAL_MS = 2000;
const PRESENTATION_POLL_INTERVAL_MS = 5000;
const TERMINAL_STATES = new Set([
  "answered", "clarification", "navigated", "succeeded", "failed", "cancelled",
  "interrupted", "unavailable", "stale", "activated", "planned",
]);
const PRESENTATION_PENDING_STATES = new Set([
  "succeeded", "planned", "activated", "stale", "failed", "cancelled",
]);
const RESPONSE_LABELS = {
  answer: "Agent · Answer", clarify: "Agent · Clarification",
  execute: "Agent · Result", navigate: "Agent · Navigation",
};
const element = (id) => document.getElementById(id);
const state = {
  session: null, models: [], activeTurn: null, submission: null,
  durableTurnIds: new Set(),
  posting: false, opening: false, timer: null, epoch: 0, pollEpoch: 0,
  navigating: false, viewedRevisionId: null, revision: null, revisionEpoch: 0,
  evidenceEpoch: 0, evidence: null, artifactUrl: null, artifactEpoch: 0,
  resources: [], resourcesEnabled: false, upload: null,
  selectedResourceIds: [],
  fastqMembers: [], fastqCompletion: null,
  inputSets: [], epizooResources: [],
  scientificFiles: new Map(),
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
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return value && typeof value === "object" ? value : {};
  } catch (_) {
    return {};
  }
}

function saveConveniences() {
  const value = {
    sessionId: state.session ? state.session.session_id : "",
    profileId: element("model-choice").value,
    polledTurnId: state.activeTurn ? state.activeTurn.turn_id : "",
    draft: element("utterance").value,
    inputSetId: element("input-choice").value,
    epizooResourceId: element("epizoo-resource-choice").value,
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(value)); } catch (_) {}
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

function awaitingPresentation(turn) {
  // Answer and scientific completion precede immutable display publication.
  // An explicit submission failure has no display to await.
  const submissionFailure = ["failed", "cancelled"].includes(turn.status) && !turn.run_id
    && turn.error && turn.error.code !== "INTERACTIVE_PRESENTATION_UNAVAILABLE";
  return typeof turn.profile_id === "string" && !turn.response
    && PRESENTATION_PENDING_STATES.has(turn.status) && !submissionFailure;
}

function terminal(turn) {
  return Boolean(turn.response) || (!awaitingPresentation(turn) && TERMINAL_STATES.has(turn.status));
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
  const explanation = document.createElement("p");
  explanation.className = "muted small";
  explanation.textContent = "Continuing with an option starts a new Agent turn. Agent checks the captured scientific context before execution.";
  section.append(heading, explanation);
  for (const candidate of guidance.candidates) {
    const card = document.createElement("div");
    card.className = "guidance-candidate";
    card.dataset.candidateId = candidate.candidate_id;
    const title = document.createElement("h3");
    title.textContent = `Option ${candidate.option} · ${readableLabel(candidate.capability)}`;
    const rationale = document.createElement("p");
    rationale.textContent = candidate.text;
    const metadata = document.createElement("span");
    metadata.className = "candidate-metadata";
    metadata.textContent = `Support: ${candidate.support} · Candidate ${candidate.candidate_id} · Origin turn ${candidate.origin_turn_id}`;
    card.append(title, rationale, textList(candidate.limitations));
    const context = document.createElement("p");
    context.className = "muted small";
    context.textContent = candidate.base_revision_id ? `Captured revision: ${candidate.base_revision_id}` : "Captured revision metadata was not stored in this historical display.";
    const diagnostics = document.createElement("details");
    diagnostics.dataset.detailKey = `candidate:${candidate.candidate_id}`;
    const diagnosticsLabel = document.createElement("summary");
    diagnosticsLabel.textContent = "Scientific details";
    diagnosticsLabel.dataset.actionKey = `candidate-details:${candidate.candidate_id}`;
    diagnostics.append(diagnosticsLabel, metadata, context);
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
      card.append(readiness);
    }
    const action = document.createElement("button");
    action.className = "guidance-select";
    action.dataset.actionKey = `guidance:${candidate.candidate_id}`;
    action.type = "button";
    action.textContent = "Continue with this";
    action.addEventListener("click", () => {
      // The click supplies a command and persisted discussion referent only.
      // The existing semantic interpreter and M17.2 admit the exact candidate.
      submitRequest(`Run option ${candidate.option}.`, candidate.origin_turn_id);
    });
    card.append(action);
    section.append(card);
  }
  if (guidance.limitations.length) section.append(textList(guidance.limitations));
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

function renderRevisionHistory() {
  const revisions = state.session ? state.session.revisions : [];
  element("revision-history").replaceChildren(...revisions.map((revision) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `Revision ${shortIdentity(revision.revision_id)}${revision.is_active ? " · Active" : ""}`;
    button.title = revision.revision_id;
    button.dataset.revisionId = revision.revision_id;
    button.setAttribute("aria-current", String(revision.is_active));
    button.addEventListener("click", () => loadRevision(revision.revision_id));
    const metadata = document.createElement("details");
    const label = document.createElement("summary");
    label.textContent = "Revision details";
    const identity = document.createElement("span");
    identity.className = "revision-metadata";
    identity.textContent = `Revision: ${revision.revision_id}\nParent: ${revision.parent_revision_id || "None"} · Turn: ${revision.turn_id} · Outputs: ${revision.outputs.join(", ") || "None"}`;
    metadata.append(label, identity);
    item.append(button, metadata);
    return item;
  }));
}

function clearArtifactView() {
  state.artifactEpoch += 1;
  if (state.artifactUrl) URL.revokeObjectURL(state.artifactUrl);
  state.artifactUrl = null;
  element("artifact-viewer").hidden = true;
  element("report-content").hidden = true;
  element("report-content").textContent = "";
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
  clearArtifactView();
  element("revision-view").hidden = true;
  element("scientific-empty").hidden = false;
  element("scientific-empty").textContent = "Accepted revisions will appear here after an analysis.";
  updateControls();
}

async function loadRevision(revisionId) {
  if (!state.session) return;
  const sessionId = state.session.session_id;
  const epoch = state.epoch;
  const revisionEpoch = ++state.revisionEpoch;
  state.evidenceEpoch += 1;
  state.revision = null;
  state.evidence = null;
  state.viewedRevisionId = revisionId;
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
    if (element("evidence-panel").open && revision.evidence_outputs.length) loadEvidence();
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch) return;
    element("scientific-empty").textContent = `${error.code}: ${error.message}`;
  }
  updateControls();
}

function renderRevision(revision) {
  element("scientific-empty").hidden = true;
  element("revision-view").hidden = false;
  element("result-heading").textContent = `${revision.is_active ? "Current" : "Historical"} result · ${statusLabel(revision.status)}`;
  element("result-heading").title = revision.revision_id;
  const originatingTurn = state.session.turns.find((turn) => turn.turn_id === revision.turn_id);
  element("result-origin").textContent = originatingTurn && originatingTurn.utterance
    ? `Requested in this turn: ${originatingTurn.utterance}`
    : "Originating analysis identifiers are available in Revision details.";
  element("result-origin").title = `Originating turn: ${revision.turn_id}. Viewing this result does not change the active revision.`;
  element("result-panel-heading").textContent = revision.is_active ? "Current result" : "Historical result";
  element("result-output-summary").textContent = `${revision.outputs.length} accepted outputs${revision.outputs.length ? `: ${revision.outputs.join(", ")}` : ""}`;
  element("viewed-revision").textContent = revision.revision_id;
  element("revision-summary").replaceChildren(...summaryRows([
    ["State", revision.is_active ? "Active revision" : "Historical revision"],
    ["Parent", revision.parent_revision_id || "None"], ["Turn", revision.turn_id],
    ["Run", revision.run_id], ["Checkpoint", revision.status],
    ["Base generation", revision.base_generation], ["Session generation", revision.session_generation],
  ]));
  element("activate-revision").hidden = revision.is_active;
  element("result-presentation").textContent = revision.result ? revision.result.text : "Stored result presentation is unavailable.";
  element("result-operations").replaceChildren(...(revision.steps || []).map((step) => {
    const item = document.createElement("li");
    item.textContent = `${readableLabel(step.tool_name)} · ${readableLabel(step.status.toLowerCase())}`;
    return item;
  }));
  const retained = new Set(revision.retained_outputs);
  element("revision-outputs").replaceChildren(...revision.outputs.map((output) => {
    const item = document.createElement("li");
    item.textContent = `${output}${retained.has(output) ? " · Retained from an earlier result" : ""}`;
    return item;
  }));
  renderArtifacts(revision);
  element("evidence-output").replaceChildren(...revision.evidence_outputs.map((output) => {
    const option = document.createElement("option");
    option.value = output;
    option.textContent = readableLabel(output);
    return option;
  }));
  element("evidence-output").disabled = !revision.evidence_outputs.length;
  element("load-evidence").disabled = !revision.evidence_outputs.length;
  element("evidence-content").replaceChildren();
  element("detail-content").replaceChildren();
  element("detail-form").hidden = true;
  element("detail-subject").value = "";
  if (!revision.evidence_outputs.length) {
    element("evidence-content").textContent = "Accepted evidence is unavailable for these output names.";
  }
}

function artifactPath(sessionId, revisionId, handle) {
  return `${revisionPath(sessionId, revisionId)}/artifacts/${encodeURIComponent(handle)}`;
}

function scientificArtifactPath(sessionId, revisionId, handle) {
  return `${revisionPath(sessionId, revisionId)}/scientific-artifacts/${encodeURIComponent(handle)}`;
}

function supportedScientificFile(artifact) {
  const files = {
    scientific_matrix: { "matrix.h5ad": "Original accepted matrix" },
    scientific_qc_table: {
      "barcodes.tsv.gz": "Barcode QC",
      "lengths.tsv.gz": "Fragment length distribution",
    },
    scientific_selection_table: {
      "decisions.tsv.gz": "Cell-selection decisions",
      "selected.tsv.gz": "Selected cells",
    },
    scientific_analysis: { "neighbors.h5ad": "Cell neighbors — safe export" },
    scientific_annotation: { "annotated.h5ad": "Annotated cells — safe export" },
  };
  return Object.hasOwn(files, artifact.artifact_type)
    && Object.hasOwn(files[artifact.artifact_type], artifact.filename)
    && files[artifact.artifact_type][artifact.filename] === artifact.label;
}

function scientificFileNode(sessionId, artifact) {
  const row = document.createElement("div");
  row.className = "scientific-file";
  const title = document.createElement("span");
  title.textContent = `${artifact.filename} — ${artifact.label}`;
  const download = document.createElement("a");
  download.textContent = "Download";
  download.href = `/api/v1${scientificArtifactPath(sessionId, artifact.revision_id, artifact.handle)}`;
  download.download = artifact.filename;
  download.dataset.actionKey = `scientific-download:${artifact.handle}`;
  download.setAttribute("aria-label", `Download ${artifact.filename}`);
  // The browser streams an attachment directly; scientific bytes stay out of JS.
  row.append(title, download);
  if (artifact.artifact_type === "scientific_analysis") {
    const note = document.createElement("span");
    note.className = "scientific-export-note";
    note.textContent = "Privacy-safe export; accepted source remains unchanged.";
    row.append(note);
  }
  return row;
}

function appendScientificFiles(response, turn) {
  if (turn.response.kind !== "execute" || !turn.revision_id
      || !["succeeded", "activated", "stale"].includes(turn.status)) return;
  const sessionId = state.session.session_id;
  const key = JSON.stringify([sessionId, turn.turn_id, turn.revision_id]);
  const cached = state.scientificFiles.get(key);
  if (cached) {
    for (const artifact of cached.artifacts) {
      response.append(scientificFileNode(sessionId, artifact));
    }
    if (cached.revision && cached.revision.artifacts.some((artifact) =>
      ["analysis_report", "analysis_figure"].includes(artifact.artifact_type))) {
      const details = document.createElement("details");
      details.className = "message-artifacts";
      details.dataset.detailKey = "result-artifacts";
      const summary = document.createElement("summary");
      summary.textContent = "Figures & report";
      summary.dataset.actionKey = "result-artifacts";
      const inventory = document.createElement("ul");
      inventory.className = "compact-list";
      renderArtifacts(cached.revision, inventory, true);
      details.append(summary, inventory);
      response.append(details);
    }
    return;
  }
  const epoch = state.epoch;
  const entry = { artifacts: [], revision: null };
  state.scientificFiles.set(key, entry);
  // Inventory is metadata only and requested once per captured result. Regular
  // conversation polling reuses this display cache; refresh reads it afresh.
  Promise.allSettled([
    api(`${turnPath(sessionId, turn.turn_id)}/scientific-artifacts`),
    api(revisionPath(sessionId, turn.revision_id)),
  ]).then(([inventory, revision]) => {
    if (epoch !== state.epoch || state.scientificFiles.get(key) !== entry) return;
    if (inventory.status === "fulfilled") {
      entry.artifacts = inventory.value.artifacts.filter((artifact) => supportedScientificFile(artifact)
        && artifact.turn_id === turn.turn_id && artifact.revision_id === turn.revision_id);
    }
    if (revision.status === "fulfilled" && revision.value.revision_id === turn.revision_id
        && revision.value.turn_id === turn.turn_id) entry.revision = revision.value;
    renderHistory();
  }).catch(() => {
    // Missing or unqualified historical associations expose no download.
  });
}

function renderArtifacts(revision, target = element("artifact-inventory"), inConversation = false) {
  const sessionId = state.session.session_id;
  const artifacts = inConversation ? revision.artifacts.filter((artifact) =>
    ["analysis_report", "analysis_figure"].includes(artifact.artifact_type)) : revision.artifacts;
  const nodes = artifacts.map((artifact) => {
    const item = document.createElement("li");
    const supported = ["analysis_report", "analysis_figure"].includes(artifact.artifact_type);
    const title = document.createElement("span");
    title.textContent = artifact.artifact_type === "analysis_report" ? "Report projection" : artifact.artifact_type === "analysis_figure" ? "PNG figure" : artifact.artifact_type;
    const metadata = document.createElement("details");
    metadata.dataset.detailKey = `artifact:${artifact.handle}`;
    const label = document.createElement("summary");
    label.dataset.actionKey = `artifact-identity:${artifact.handle}`;
    label.textContent = "Artifact identity";
    const identity = document.createElement("p");
    identity.className = "small";
    identity.textContent = `Handle: ${artifact.handle}\nSource SHA-256: ${artifact.sha256}\nRevision: ${artifact.revision_id}`;
    metadata.append(label, identity);
    item.append(title, metadata);
    if (supported) {
      const actions = document.createElement("div");
      actions.className = "artifact-actions";
      const view = document.createElement("button");
      view.type = "button";
      view.className = "secondary";
      view.textContent = "View";
      view.dataset.artifactHandle = artifact.handle;
      view.dataset.actionKey = `artifact-view:${artifact.handle}`;
      view.addEventListener("click", () => viewArtifact(sessionId, revision.revision_id, artifact));
      const download = document.createElement("a");
      download.textContent = "Download";
      download.dataset.actionKey = `artifact-download:${artifact.handle}`;
      download.href = `/api/v1${artifactPath(sessionId, revision.revision_id, artifact.handle)}?download=true`;
      actions.append(view, download);
      item.append(actions);
    }
    return item;
  });
  for (const artifact of inConversation ? [] : revision.scientific_artifacts || []) {
    if (!supportedScientificFile(artifact)) continue;
    const item = document.createElement("li");
    item.append(scientificFileNode(sessionId, artifact));
    nodes.push(item);
  }
  if (!nodes.length) {
    const empty = document.createElement("li");
    empty.textContent = "No accepted reports, figures or eligible scientific downloads are available.";
    nodes.push(empty);
  }
  if (!inConversation && !revision.artifacts.some((artifact) => artifact.artifact_type === "analysis_figure")) {
    const note = document.createElement("li");
    note.className = "muted small";
    note.textContent = "No reviewed figure is available for this result. Figure availability is separate from analysis success.";
    nodes.push(note);
  }
  target.replaceChildren(...nodes);
}

async function viewArtifact(sessionId, revisionId, artifact) {
  const epoch = state.epoch;
  if (!state.session || state.session.session_id !== sessionId) return;
  element("scientific-panel").open = true;
  element("artifact-panel").open = true;
  if (!state.revision || state.revision.revision_id !== revisionId) await loadRevision(revisionId);
  if (epoch !== state.epoch) return;
  if (!state.session || state.session.session_id !== sessionId
      || !state.revision || state.revision.revision_id !== revisionId) {
    if (state.session && state.session.session_id === sessionId) element("scientific-details").scrollIntoView({ block: "nearest" });
    return;
  }
  clearArtifactView();
  const artifactEpoch = state.artifactEpoch;
  const revisionEpoch = state.revisionEpoch;
  element("artifact-viewer").hidden = false;
  element("artifact-title").textContent = artifact.artifact_type === "analysis_report" ? "Client-safe report projection" : "Accepted PNG figure";
  try {
    const response = await fetch(`/api/v1${artifactPath(sessionId, revisionId, artifact.handle)}`, {
      credentials: "same-origin", cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json();
      throw new ApiError(data.error.code, data.error.message, response.status);
    }
    const contentType = response.headers.get("Content-Type") || "";
    if (artifact.artifact_type === "analysis_report") {
      if (!contentType.startsWith("text/plain")) throw new ApiError("ARTIFACT_TYPE_UNSUPPORTED", "This artifact is not a supported text report projection.");
      const text = await response.text();
      if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
      element("report-content").textContent = text;
      element("report-content").hidden = false;
    } else if (artifact.artifact_type === "analysis_figure") {
      if (!contentType.startsWith("image/png")) throw new ApiError("ARTIFACT_TYPE_UNSUPPORTED", "This artifact is not a supported PNG figure.");
      const blob = await response.blob();
      if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
      state.artifactUrl = URL.createObjectURL(blob);
      element("figure-content").src = state.artifactUrl;
      element("figure-content").hidden = false;
    }
    element("artifact-viewer").scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
    element("artifact-view-error").textContent = `${error.code || "ARTIFACT_UNAVAILABLE"}: ${error instanceof ApiError ? error.message : "The accepted artifact could not be read."}`;
    element("artifact-view-error").hidden = false;
  }
}

function readableLabel(value) {
  const label = String(value).replace(/_/g, " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function statusLabel(status) {
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

function evidenceValueNode(value) {
  if (Array.isArray(value)) {
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
    const list = document.createElement("dl");
    list.className = "fact-values";
    for (const [key, entry] of Object.entries(value)) {
      const label = document.createElement("dt");
      label.textContent = readableLabel(key);
      const content = document.createElement("dd");
      content.append(evidenceValueNode(entry));
      list.append(label, content);
    }
    return list;
  }
  const content = document.createElement("span");
  const origins = {
    explicit_user_instruction: "Explicit user instruction",
    explicit_execution_argument: "Explicit execution argument; user origin is not established by this fact",
    existing_owner_default: "Existing scientific owner default",
  };
  content.textContent = typeof value === "string" && Object.hasOwn(origins, value)
    ? origins[value] : value === null ? "Recorded null" : String(value);
  return content;
}

function evidenceFactNodes(facts, fieldLabels = {}) {
  return (facts || []).map((fact) => {
    const technical = fact.field.endsWith("_sha256") || fact.field.includes("contract");
    const item = document.createElement(technical ? "details" : "section");
    item.className = "scientific-fact";
    const title = document.createElement(technical ? "summary" : "h4");
    title.textContent = `${fieldLabels[fact.field] || readableLabel(fact.field)}${fact.status === "available" ? "" : ` · ${fact.status}`}`;
    const content = fact.status === "available" ? evidenceValueNode(fact.value) : document.createElement("p");
    if (fact.status !== "available") content.textContent = fact.reason || "No accepted value is available.";
    item.append(title, content);
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
    nodes.push(title, ...evidenceFactNodes(facts, view.field_labels || {}));
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
  if (detail) {
    query.set("detail_section", element("detail-section").value);
    query.set("limit", element("detail-limit").value);
    if (element("detail-subject").value) query.set("subject", element("detail-subject").value);
  }
  const target = element(detail ? "detail-content" : "evidence-content");
  if (!detail) {
    state.evidence = null;
    element("detail-form").hidden = true;
    element("detail-content").replaceChildren();
  }
  target.textContent = "Reading accepted evidence…";
  try {
    const view = await api(`${revisionPath(sessionId, revisionId)}/evidence?${query}`);
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || evidenceEpoch !== state.evidenceEpoch) return;
    renderEvidence(view, target);
    if (!detail) {
      state.evidence = view;
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
    }
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || evidenceEpoch !== state.evidenceEpoch) return;
    target.textContent = `${error.code}: ${error.message}`;
  }
}

async function activateViewedRevision(continueFromHere = false) {
  if (!state.session || !state.revision || element("continue-revision").disabled) return;
  const sessionId = state.session.session_id;
  const revisionId = state.revision.revision_id;
  const epoch = state.epoch;
  if (state.session.active_revision_id !== revisionId) {
    state.navigating = true;
    updateControls();
    showError(null);
    try {
      const view = await api(`${sessionPath(sessionId)}/activate`, "POST", {
        turn_id: newTurnId(), revision_id: revisionId, expected_generation: state.session.generation,
      });
      if (epoch !== state.epoch) return;
      displaySession(view);
      saveConveniences();
      if (state.session.active_revision_id !== revisionId) {
        throw new ApiError("REVISION_ACTIVATION_CHANGED", "Agent confirmed a different active revision. Reopen the session before continuing.");
      }
    } catch (error) {
      if (epoch !== state.epoch) return;
      showError(error);
      try { await refreshCurrentSession(sessionId, epoch); } catch (_) {}
      return;
    } finally {
      if (epoch === state.epoch) { state.navigating = false; updateControls(); }
    }
  }
  if (continueFromHere && epoch === state.epoch && state.session.active_revision_id === revisionId) {
    if (!element("utterance").value.trim()) element("utterance").value = "Continue from this result.";
    element("utterance").focus();
    element("utterance").scrollIntoView({ block: "center", behavior: "smooth" });
    saveConveniences();
  }
}

function captureConversationUi(history) {
  const expanded = new Set(Array.from(history.querySelectorAll("details[open][data-detail-key]")).map((details) =>
    JSON.stringify([details.closest(".turn").dataset.turnId, details.dataset.detailKey])));
  const focused = document.activeElement;
  const turn = focused && history.contains(focused) && focused.closest(".turn");
  return { expanded, focus: turn && focused.dataset.actionKey
    ? { turnId: turn.dataset.turnId, actionKey: focused.dataset.actionKey } : null };
}

function restoreConversationUi(history, captured) {
  for (const details of history.querySelectorAll("details[data-detail-key]")) {
    details.open = captured.expanded.has(JSON.stringify([details.closest(".turn").dataset.turnId, details.dataset.detailKey]));
  }
  if (!captured.focus) return;
  const turn = Array.from(history.children).find((node) => node.dataset.turnId === captured.focus.turnId);
  const target = turn && Array.from(turn.querySelectorAll("[data-action-key]")).find((node) =>
    node.dataset.actionKey === captured.focus.actionKey);
  if (target) target.focus({ preventScroll: true });
}

function renderHistory() {
  const history = element("conversation-history");
  const captured = captureConversationUi(history);
  const turns = state.session ? state.session.turns.filter((turn) => turn.utterance || turn.response) : [];
  const visibleFileKeys = new Set(turns.map((turn) =>
    JSON.stringify([state.session.session_id, turn.turn_id, turn.revision_id])));
  for (const key of state.scientificFiles.keys()) {
    if (!visibleFileKeys.has(key)) state.scientificFiles.delete(key);
  }
  if (!turns.length) {
    const empty = document.createElement("p");
    empty.className = "empty-history";
    empty.textContent = state.session ? "Your session is ready. Enter a request below." : "Create a session or reopen one to begin.";
    history.replaceChildren(empty);
    return;
  }
  history.replaceChildren(...turns.map((turn) => {
    const article = document.createElement("article");
    article.className = "turn";
    article.dataset.turnId = turn.turn_id;
    if (turn.utterance) article.append(messageNode("You", turn.utterance, "user"));
    let response = null;
    if (turn.response) {
      response = messageNode(RESPONSE_LABELS[turn.response.kind] || "Agent", turn.response.text, "assistant");
      article.append(response);
      if (turn.response.guidance) response.append(guidanceNode(turn.response.guidance));
      appendScientificFiles(response, turn);
    }
    const status = document.createElement("p");
    status.className = "turn-status-note";
    status.textContent = statusLabel(turn.status);
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
      const targets = document.createElement("div");
      targets.className = "turn-checkpoint";
      if (scientific.targets) {
        targets.append(textList(scientific.targets.map((target) => `Scientific target: Revision ${target.revision_id} · Output ${target.output_name} · Subject ${target.subject === null ? "None specified" : target.subject}`)));
      } else {
        targets.textContent = "Scientific target metadata was not stored in this historical display.";
      }
      metadata.append(targets);
    }
    if (turn.revision_id) {
      const result = document.createElement("button");
      result.className = "secondary view-result";
      result.type = "button";
      result.dataset.actionKey = "view-result";
      result.textContent = "View result details";
      result.title = turn.response && turn.response.kind === "execute"
        ? `View created result revision ${turn.revision_id}` : `View revision ${turn.revision_id}`;
      result.addEventListener("click", () => {
        element("scientific-panel").open = true;
        element("result-panel").open = true;
        element("evidence-panel").open = true;
        element("artifact-panel").open = true;
        loadRevision(turn.revision_id);
        const details = element("scientific-details");
        details.scrollIntoView({ block: "start" });
        details.focus({ preventScroll: true });
      });
      (response || article).append(result);
    }
    article.append(status, metadata);
    const error = turn.error || turn.response && turn.response.error;
    if (error) {
      const note = document.createElement("div");
      note.className = "turn-error";
      note.textContent = `${error.code}: ${error.message}`;
      article.append(note);
    }
    return article;
  }));
  restoreConversationUi(history, captured);
}

function updateControls() {
  const busy = state.posting || Boolean(state.activeTurn && !terminal(state.activeTurn));
  const cannotSubmit = !state.session || !state.models.length || busy || state.opening || state.navigating || Boolean(state.upload) || Boolean(state.fastqCompletion && state.fastqCompletion.pending);
  element("submit-turn").disabled = cannotSubmit;
  document.querySelectorAll(".guidance-select").forEach((button) => { button.disabled = cannotSubmit; });
  element("activate-revision").disabled = !state.revision || busy || state.opening || state.navigating;
  element("continue-revision").disabled = !state.revision || busy || state.opening || state.navigating;
  element("utterance").disabled = !state.session;
  element("new-session").disabled = state.opening;
  element("reopen-session").disabled = state.opening;
  const cancel = element("cancel-turn");
  cancel.hidden = !(state.activeTurn && state.activeTurn.run_id &&
    ["planning", "validated", "running"].includes(state.activeTurn.status) && !terminal(state.activeTurn));
  cancel.disabled = cancel.hidden || state.posting;
  const uploading = Boolean(state.upload && state.upload.status === "uploading")
    || Boolean(state.fastqCompletion && state.fastqCompletion.pending);
  element("resource-choice").disabled = !state.resourcesEnabled || !state.resources.length || uploading;
  element("add-scientific-input").disabled = !canAddScientificInput() || uploading;
  document.querySelectorAll(".remove-scientific-input").forEach((button) => { button.disabled = uploading; });
  element("upload-file").disabled = !state.resourcesEnabled || uploading;
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

function renderExecution() {
  element("turn-status").textContent = state.posting ? "Submitting request…" : state.activeTurn ? statusLabel(state.activeTurn.status) : "Ready";
  element("turn-status").title = state.activeTurn ? checkpointText(state.activeTurn) : "";
  const steps = state.activeTurn && state.activeTurn.steps || [];
  const list = element("step-status");
  list.hidden = !steps.length;
  list.replaceChildren(...steps.map((step) => {
    const item = document.createElement("li");
    item.textContent = `${readableLabel(step.tool_name)} · ${readableLabel(step.status.toLowerCase())}`;
    return item;
  }));
  updateControls();
}

function mergeTurn(turn) {
  if (!state.session || turn.session_id !== state.session.session_id) return;
  const turns = [...state.session.turns];
  const index = turns.findIndex((item) => item.turn_id === turn.turn_id);
  if (index === -1) turns.push(turn); else turns[index] = turn;
  state.session = { ...state.session, turns };
  renderHistory();
}

function stopPolling() {
  clearTimeout(state.timer);
  state.timer = null;
  state.pollEpoch += 1;
}

function displaySession(view) {
  const changedActive = !state.session || state.session.session_id !== view.session_id || state.session.active_revision_id !== view.active_revision_id;
  state.session = view;
  state.durableTurnIds = new Set(view.turns.map((turn) => turn.turn_id));
  element("current-session").textContent = view.session_id;
  element("session-input").value = view.session_id;
  element("current-generation").textContent = String(view.generation);
  element("current-revision").textContent = view.active_revision_id || "None";
  element("session-context").textContent = `Session ${shortIdentity(view.session_id)} · Generation ${view.generation} · Active revision ${view.active_revision_id ? shortIdentity(view.active_revision_id) : "None"}`;
  element("session-context").title = view.session_id;
  element("history-notice").hidden = !view.history_truncated;
  renderHistory();
  renderRevisionHistory();
  if (changedActive || !view.revisions.some((revision) => revision.revision_id === state.viewedRevisionId)) {
    state.viewedRevisionId = view.active_revision_id || (view.revisions.length ? view.revisions[view.revisions.length - 1].revision_id : null);
  }
  if (state.viewedRevisionId) loadRevision(state.viewedRevisionId); else clearRevisionView();
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
      element("connection-state").textContent = "Connected";
      showError(turn.error || turn.response && turn.response.error || null);
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
      element("connection-state").textContent = "Status unavailable";
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
    const interval = state.activeTurn && awaitingPresentation(state.activeTurn)
      ? PRESENTATION_POLL_INTERVAL_MS : POLL_INTERVAL_MS;
    state.timer = setTimeout(tick, interval);
  };
  tick();
}

async function openSession(sessionId, resumeTurnId = "") {
  const epoch = ++state.epoch;
  state.scientificFiles.clear();
  stopPolling();
  state.opening = true;
  state.navigating = false;
  state.submission = null;
  state.activeTurn = null;
  element("retry-turn").hidden = true;
  renderExecution();
  showError(null);
  try {
    const view = await api(sessionPath(sessionId));
    if (epoch !== state.epoch) return;
    displaySession(view);
    const current = [...view.turns].reverse().find((turn) => (turn.utterance || turn.response) && !terminal(turn));
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
  const choice = state.models.find((item) => item.profile_id === element("model-choice").value);
  element("model-description").textContent = choice ? [choice.is_default ? "Configured default" : "", choice.provider_id, choice.model_id].filter(Boolean).join(" · ") : "";
  saveConveniences();
}

function inputContext() {
  // Repeat the operator-admitted display label without deriving scientific context.
  const choice = element("input-choice").selectedOptions[0];
  const resources = selectedResourceIds().map((id) => state.resources.find((choice) => choice.resource_id === id));
  const multiple = resources.length > 1;
  const epizoo = element("epizoo-resource-choice").selectedOptions[0];
  const labels = [...resources.map((resource, index) => `${multiple ? `${index + 1}. ` : ""}${resource.label}`),
    choice && choice.value ? choice.textContent : "",
    selectedResourceType() === "h5ad" && epizoo && epizoo.value ? epizoo.textContent : ""].filter(Boolean);
  const label = labels.length ? labels.join(" · ") : "none selected";
  element("selected-input-context").textContent = `Scientific inputs for next turn: ${label}`;
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
  element("toggle-scientific-inputs").hidden = irrelevant;
}

function selectedResourceIds() {
  const current = element("resource-choice").value;
  return [...state.selectedResourceIds, ...(current && !state.selectedResourceIds.includes(current) ? [current] : [])];
}

function canAddScientificInput() {
  const ids = selectedResourceIds();
  return Boolean(element("resource-choice").value) && ids.length < MAX_SELECTED_RESOURCES
    && ids.every((id) => state.resources.some((resource) => resource.resource_id === id && resource.input_type === "h5ad"));
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
  return resource ? resource.input_type : state.upload ? state.upload.inputType : null;
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
  try { localStorage.setItem(RESOURCE_STORAGE_KEY, element("resource-choice").value); } catch (_) {}
  try { localStorage.setItem(MULTI_RESOURCE_STORAGE_KEY, JSON.stringify(selectedResourceIds())); } catch (_) {}
}

function uploadStatus(text, failed = false) {
  element("upload-status").textContent = text;
  element("upload-status").classList.toggle("upload-error", failed);
}

function renderResources(selectedId = "", restoreSavedSelection = false) {
  const selected = state.resources.find((choice) => choice.resource_id === selectedId);
  const canSelect = selected && (restoreSavedSelection || !state.selectedResourceIds.length || selected.input_type === "h5ad");
  if (selected && !canSelect) state.selectedResourceIds = selectedResourceIds();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "No uploaded input selected";
  element("resource-choice").replaceChildren(none, ...state.resources.map((choice) => {
    const option = document.createElement("option");
    option.value = choice.resource_id;
    const role = choice.input_type === "external_fragments" ? `Fragments${choice.has_source_index === true ? " + TBI" : ""}` : choice.input_type === "bam" ? "BAM" : choice.input_type === "fastq" ? "FASTQ library" : "H5AD";
    option.textContent = `${choice.label} · ${role} · registered`;
    option.disabled = state.selectedResourceIds.includes(choice.resource_id)
      || state.selectedResourceIds.length > 0 && choice.input_type !== "h5ad";
    return option;
  }));
  if (canSelect) {
    element("resource-choice").value = selectedId;
  }
  if (selectedResourceIds().length === 1 && !selectedCompanion()) element("input-choice").value = "";
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
  try { selectedId = localStorage.getItem(RESOURCE_STORAGE_KEY) || ""; } catch (_) {}
  try {
    const saved = JSON.parse(localStorage.getItem(MULTI_RESOURCE_STORAGE_KEY) || "[]");
    if (Array.isArray(saved) && saved.length <= MAX_SELECTED_RESOURCES
        && new Set(saved).size === saved.length && saved.every((id) => typeof id === "string")) savedIds = saved;
  } catch (_) {}
  try {
    const catalog = await api("/resources");
    state.resourcesEnabled = catalog.enabled === true;
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
  if (!keepFile) element("upload-file").value = "";
  element("upload-index").value = "";
  element("upload-progress").hidden = true;
  uploadStatus("No file awaiting upload. You can send a message without an attachment.");
  updateControls();
}

function renderUploadType() {
  const fragments = element("upload-type").value === "external_fragments";
  const bam = element("upload-type").value === "bam";
  const fastq = element("upload-type").value === "fastq";
  element("upload-file-label").textContent = fragments ? "Attach a fragments source (TSV, gzip or BGZF)" : bam ? "Attach a scATAC-seq BAM" : fastq ? "Attach one declared FASTQ member" : "Attach an H5AD file";
  element("upload-file").accept = fragments ? ".tsv,.gz,.bgz,.bgzf" : bam ? ".bam" : fastq ? ".fastq,.fq,.fastq.gz,.fq.gz" : ".h5ad";
  element("upload-index-fields").hidden = !fragments;
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
    const resource = await transferUpload(upload);
    if (state.upload !== upload) return;
    if (resource.input_type === "fastq") {
      state.fastqMembers = [...state.fastqMembers.filter((choice) => choice.resource_id !== resource.resource_id), resource];
      state.fastqCompletion = null;
      renderFastqMembers(resource.resource_id);
    } else {
      state.resources = [...state.resources.filter((choice) => choice.resource_id !== resource.resource_id), resource];
    }
    state.upload = null;
    element("upload-file").value = "";
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

element("upload-file").addEventListener("change", () => {
  const file = element("upload-file").files[0];
  clearPendingUpload(true);
  if (!file) return;
  // Selection creates a new transfer identity; Retry keeps this same File and ID.
  state.upload = { uploadId: newTurnId(), file, inputType: element("upload-type").value, index: null, status: "selected", request: null,
    fastq: element("upload-type").value === "fastq" ? fastqAttribution() : null };
  if (state.selectedResourceIds.length && state.upload.inputType !== "h5ad") {
    state.selectedResourceIds = selectedResourceIds();
  }
  element("resource-choice").value = "";
  if (!selectedCompanion()) element("input-choice").value = "";
  saveResourceSelection();
  saveConveniences();
  inputContext();
  uploadStatus(`${file.name} · selected. Upload this file before sending your request.`);
  updateControls();
});
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
  if (!upload || upload.inputType !== "external_fragments" || upload.status === "uploading") return;
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

element("new-session").addEventListener("click", async () => {
  const epoch = ++state.epoch;
  state.scientificFiles.clear();
  stopPolling();
  state.opening = true;
  state.navigating = false;
  state.submission = null;
  state.activeTurn = null;
  element("retry-turn").hidden = true;
  renderExecution();
  showError(null);
  try {
    const view = await api("/sessions", "POST", {});
    if (epoch !== state.epoch) return;
    displaySession(view);
    saveConveniences();
    element("utterance").focus();
  } catch (error) {
    if (epoch === state.epoch) showError(error);
  } finally {
    if (epoch === state.epoch) { state.opening = false; updateControls(); }
  }
});

element("reopen-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const id = element("session-input").value.trim();
  if (id) openSession(id);
});

function submitRequest(utterance, predecessorTurnId = null) {
  if (element("submit-turn").disabled || !state.session) return;
  if (!utterance.trim()) return;
  const body = {
    turn_id: newTurnId(), expected_generation: state.session.generation,
    utterance, profile_id: element("model-choice").value,
  };
  if (predecessorTurnId !== null) body.predecessor_turn_id = predecessorTurnId;
  const selectedIds = selectedResourceIds();
  if (selectedIds.length > 1) {
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
  state.submission = Object.freeze({ sessionId: state.session.session_id, body: Object.freeze(body) });
  state.activeTurn = { session_id: state.session.session_id, turn_id: body.turn_id, status: "submitted", steps: [] };
  saveConveniences();
  postSubmission(state.submission);
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

element("activate-revision").addEventListener("click", () => activateViewedRevision());
element("continue-revision").addEventListener("click", () => activateViewedRevision(true));
element("load-evidence").addEventListener("click", () => loadEvidence());
element("evidence-output").addEventListener("change", () => loadEvidence());
element("evidence-panel").addEventListener("toggle", () => {
  if (element("evidence-panel").open && state.revision && !state.evidence) loadEvidence();
});
element("detail-form").addEventListener("submit", (event) => {
  event.preventDefault();
  loadEvidence(true);
});

element("model-choice").addEventListener("change", modelDescription);
element("input-choice").addEventListener("change", () => {
  if (state.upload) clearPendingUpload();
  if (element("input-choice").value && !selectedCompanion()) {
    if (selectedResourceIds().length < 2) {
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
for (const [buttonId, detailsId] of [["toggle-attachments", "attachment-panel"],
  ["toggle-scientific-inputs", "scientific-input-settings"]]) {
  element(buttonId).addEventListener("click", () => {
    const details = element(detailsId);
    details.open = !details.open;
  });
  element(detailsId).addEventListener("toggle", () => {
    element(buttonId).setAttribute("aria-expanded", String(element(detailsId).open));
  });
}
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
    element("connection-state").textContent = "Connected";
    modelDescription();
    if (typeof conveniences.sessionId === "string" && conveniences.sessionId) {
      await openSession(conveniences.sessionId, typeof conveniences.polledTurnId === "string" ? conveniences.polledTurnId : "");
    }
  } catch (error) {
    element("connection-state").textContent = "Connection unavailable";
    showError(error);
  }
  await resources;
  if (selectedResourceType() === "h5ad" && state.epizooResources.some((choice) =>
      choice.resource_id === conveniences.epizooResourceId)) {
    element("epizoo-resource-choice").value = conveniences.epizooResourceId;
  }
  inputContext();
  if (resources) saveConveniences();
  updateControls();
}

initialize();
