"use strict";

// Browser storage holds display conveniences. Session generations and history
// are reconstructed from Agent whenever a session is opened or a turn completes.
const STORAGE_KEY = "agent.web.presentation.v1";
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
  // Scientific completion and immutable displayed completion are separate
  // checkpoints. Keep reading until display publication; never replay work.
  return !turn.response && PRESENTATION_PENDING_STATES.has(turn.status)
    && (!["failed", "cancelled"].includes(turn.status) || Boolean(turn.run_id));
}

function terminal(turn) {
  return Boolean(turn.response) || (!awaitingPresentation(turn) && TERMINAL_STATES.has(turn.status));
}

function checkpointText(turn) {
  return `Turn ${turn.turn_id} · ${turn.status}${turn.profile_id ? ` · ${turn.profile_id}` : ""}`;
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
    title.textContent = `Option ${candidate.option} · ${candidate.capability}`;
    const rationale = document.createElement("p");
    rationale.textContent = candidate.text;
    const metadata = document.createElement("span");
    metadata.className = "candidate-metadata";
    metadata.textContent = `Support: ${candidate.support} · Candidate ${candidate.candidate_id} · Origin turn ${candidate.origin_turn_id}`;
    card.append(title, rationale, metadata, textList(candidate.limitations));
    const context = document.createElement("p");
    context.className = "muted small";
    context.textContent = candidate.base_revision_id ? `Captured revision: ${candidate.base_revision_id}` : "Captured revision metadata was not stored in this historical display.";
    card.append(context);
    if (candidate.readiness) {
      const readiness = document.createElement("details");
      const readinessLabel = document.createElement("summary");
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
    button.textContent = `${revision.revision_id}${revision.is_active ? " · Active" : ""}`;
    button.dataset.revisionId = revision.revision_id;
    button.setAttribute("aria-current", String(revision.is_active));
    button.addEventListener("click", () => loadRevision(revision.revision_id));
    const metadata = document.createElement("span");
    metadata.className = "revision-metadata";
    metadata.textContent = `Parent: ${revision.parent_revision_id || "None"} · Turn: ${revision.turn_id} · Outputs: ${revision.outputs.join(", ") || "None"}`;
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
  element("viewed-revision").textContent = revision.revision_id;
  element("revision-summary").replaceChildren(...summaryRows([
    ["State", revision.is_active ? "Active revision" : "Historical revision"],
    ["Parent", revision.parent_revision_id || "None"], ["Turn", revision.turn_id],
    ["Run", revision.run_id], ["Checkpoint", revision.status],
    ["Base generation", revision.base_generation], ["Session generation", revision.session_generation],
  ]));
  element("activate-revision").hidden = revision.is_active;
  element("result-presentation").textContent = revision.result ? revision.result.text : "Stored result presentation is unavailable.";
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
    option.textContent = output;
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

function renderArtifacts(revision) {
  const sessionId = state.session.session_id;
  const nodes = revision.artifacts.map((artifact) => {
    const item = document.createElement("li");
    const supported = ["analysis_report", "analysis_figure"].includes(artifact.artifact_type);
    const title = document.createElement("span");
    title.textContent = artifact.artifact_type === "analysis_report" ? "Report projection" : artifact.artifact_type === "analysis_figure" ? "PNG figure" : artifact.artifact_type;
    const metadata = document.createElement("details");
    const label = document.createElement("summary");
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
      view.addEventListener("click", () => viewArtifact(sessionId, revision.revision_id, artifact));
      const download = document.createElement("a");
      download.textContent = "Download";
      download.href = `/api/v1${artifactPath(sessionId, revision.revision_id, artifact.handle)}?download=true`;
      actions.append(view, download);
      item.append(actions);
    }
    return item;
  });
  if (!nodes.length) {
    const empty = document.createElement("li");
    empty.textContent = "No accepted reports or figures are available.";
    nodes.push(empty);
  }
  element("artifact-inventory").replaceChildren(...nodes);
}

async function viewArtifact(sessionId, revisionId, artifact) {
  clearArtifactView();
  const artifactEpoch = state.artifactEpoch;
  const epoch = state.epoch;
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
  } catch (error) {
    if (epoch !== state.epoch || revisionEpoch !== state.revisionEpoch || artifactEpoch !== state.artifactEpoch) return;
    element("artifact-view-error").textContent = `${error.code || "ARTIFACT_UNAVAILABLE"}: ${error instanceof ApiError ? error.message : "The accepted artifact could not be read."}`;
    element("artifact-view-error").hidden = false;
  }
}

function evidenceFactNodes(facts) {
  return (facts || []).map((fact) => {
    const item = document.createElement("details");
    const title = document.createElement("summary");
    title.textContent = `${fact.field} · ${fact.status}`;
    const content = document.createElement("pre");
    content.className = "evidence-value";
    content.textContent = fact.status === "available" ? JSON.stringify(fact.value, null, 2) : fact.reason || "No accepted value is available.";
    item.append(title, content);
    return item;
  });
}

function renderEvidence(view, target) {
  const status = document.createElement("p");
  status.className = "small";
  status.textContent = `Availability: ${view.status}${view.reason ? ` · ${view.reason}` : ""}\nScope: ${view.evidence_scope} · Generation: ${view.generation === null ? "Unavailable" : view.generation}`;
  const nodes = [status];
  for (const [heading, facts] of [["Coverage", view.coverage], ["Accepted bounded facts", view.facts], ["Detailed evidence", view.detail]]) {
    if (!facts.length) continue;
    const title = document.createElement("h3");
    title.textContent = heading;
    nodes.push(title, ...evidenceFactNodes(facts));
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

function renderHistory() {
  const history = element("conversation-history");
  const turns = state.session ? state.session.turns.filter((turn) => turn.utterance || turn.response) : [];
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
    if (turn.response) {
      article.append(messageNode(RESPONSE_LABELS[turn.response.kind] || "Agent", turn.response.text, "assistant"));
      if (turn.response.guidance) article.append(guidanceNode(turn.response.guidance));
    }
    const checkpoint = document.createElement("div");
    checkpoint.className = "turn-checkpoint";
    checkpoint.textContent = checkpointText(turn);
    article.append(checkpoint);
    if (Object.prototype.hasOwnProperty.call(turn, "base_revision_id")) {
      const captured = document.createElement("div");
      captured.className = "turn-checkpoint";
      captured.textContent = `Captured base revision: ${turn.base_revision_id || "None"} · Generation: ${turn.base_generation}`;
      article.append(captured);
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
      article.append(targets);
    }
    if (turn.revision_id) {
      const result = document.createElement("button");
      result.className = "secondary view-result";
      result.textContent = turn.response && turn.response.kind === "execute"
        ? `View created result revision ${turn.revision_id}` : `View revision ${turn.revision_id}`;
      result.addEventListener("click", () => loadRevision(turn.revision_id));
      article.append(result);
    }
    const error = turn.error || turn.response && turn.response.error;
    if (error) {
      const note = document.createElement("div");
      note.className = "turn-error";
      note.textContent = `${error.code}: ${error.message}`;
      article.append(note);
    }
    return article;
  }));
}

function updateControls() {
  const busy = state.posting || Boolean(state.activeTurn && !terminal(state.activeTurn));
  const cannotSubmit = !state.session || !state.models.length || busy || state.opening || state.navigating;
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
}

function renderExecution() {
  element("turn-status").textContent = state.posting ? "Submitting request…" : state.activeTurn ? checkpointText(state.activeTurn) : "Ready";
  const steps = state.activeTurn && state.activeTurn.steps || [];
  const list = element("step-status");
  list.hidden = !steps.length;
  list.replaceChildren(...steps.map((step) => {
    const item = document.createElement("li");
    item.textContent = `${step.step_id} · ${step.tool_name} · ${step.status.toLowerCase()}`;
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

element("new-session").addEventListener("click", async () => {
  const epoch = ++state.epoch;
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
  if (element("input-choice").value) body.input_set_id = element("input-choice").value;
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
element("utterance").addEventListener("input", saveConveniences);

async function initialize() {
  const conveniences = readConveniences();
  if (typeof conveniences.draft === "string") element("utterance").value = conveniences.draft.slice(0, 4096);
  try {
    const [, models, inputs] = await Promise.all([api("/health"), api("/models"), api("/input-sets")]);
    state.models = models.choices;
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
      option.textContent = choice.display_label;
      return option;
    }));
    element("input-choice").disabled = !inputs.choices.length;
    element("connection-state").textContent = "Connected";
    modelDescription();
    if (typeof conveniences.sessionId === "string" && conveniences.sessionId) {
      await openSession(conveniences.sessionId, typeof conveniences.polledTurnId === "string" ? conveniences.polledTurnId : "");
    }
  } catch (error) {
    element("connection-state").textContent = "Connection unavailable";
    showError(error);
  }
  updateControls();
}

initialize();
