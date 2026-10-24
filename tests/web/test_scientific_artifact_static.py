"""Chat placement and direct browser delivery for reviewed scientific files."""
from pathlib import Path


STATIC = Path(__file__).resolve().parents[2] / 'src' / 'agent' / 'web' / 'static'


def test_download_is_inside_corresponding_assistant_message():
    source = (STATIC / 'app.js').read_text()
    history = source[source.index('function renderHistory'):source.index('function updateControls')]
    assert 'appendScientificFiles(response, turn)' in history
    assert history.index('article.append(response)') < history.index('appendScientificFiles(response, turn)')
    assert history.index('appendScientificFiles(response, turn)') < history.index('View result details')
    assert '(response || article).append(result)' in history


def test_download_uses_server_descriptor_identity_and_normal_browser_attachment():
    source = (STATIC / 'app.js').read_text()
    files = source[source.index('function scientificArtifactPath'):source.index('function renderArtifacts')]
    assert 'scientificArtifactPath(sessionId, artifact.revision_id, artifact.handle, artifact.turn_id)' in files
    assert 'download.download = artifact.filename' in files
    assert 'artifact.title || artifact.label' in files
    assert 'artifact.description ||' in files and 'artifact.purpose ||' in files
    assert '${artifact.filename}' in files
    assert 'formatBytes(artifact.size_bytes)' in files
    assert 'artifact.turn_id === turn.turn_id && (artifact.revision_id === turn.revision_id' in files
    assert 'active_revision_id' not in files
    assert 'fetch(' not in files and '.blob(' not in files and 'createObjectURL' not in files
    assert 'scientific-artifacts/${encodeURIComponent(handle)}' in files


def test_inventory_is_cached_per_exact_accepted_execute_message_and_refreshed_on_reopen():
    source = (STATIC / 'app.js').read_text()
    files = source[source.index('function appendScientificFiles'):source.index('function renderArtifacts')]
    assert 'turn.response.kind === "execute"' in files
    assert '["succeeded", "activated", "stale"].includes(turn.status)' in files
    assert 'JSON.stringify([sessionId, turn.turn_id, turn.revision_id])' in files
    assert files.index('if (cached)') < files.index('api(`${turnPath')
    assert 'epoch !== state.epoch' in files
    assert 'setInterval' not in files and 'setTimeout' not in files
    reopen = source[source.index('async function openSession'):source.index('function newTurnId')]
    assert 'state.scientificFiles.clear()' in reopen
    assert 'localStorage' not in files


def test_details_report_and_figure_controls_remain_available():
    source = (STATIC / 'app.js').read_text()
    artifacts = source[source.index('function renderArtifacts'):source.index('async function viewArtifact')]
    assert 'revision.scientific_artifacts || []' in artifacts
    assert '["analysis_report", "analysis_figure"].includes(artifact.artifact_type)' in artifacts
    assert 'viewArtifact(sessionId, revision.revision_id, artifact)' in artifacts
    assert 'innerHTML' not in source


def test_chat_and_details_use_server_file_descriptions_without_a_duplicate_label_policy():
    source = (STATIC / 'app.js').read_text()
    policy = source[source.index('function supportedScientificFile'):source.index('function scientificFileNode')]
    for artifact_type in ('scientific_matrix', 'scientific_qc_table', 'scientific_selection_table',
                          'scientific_integrated', 'scientific_analysis'):
        assert artifact_type in policy
    assert 'Original accepted matrix' not in policy and 'safe export' not in policy
    assert 'scientificFileGroups(sessionId,' in source[source.index('function appendScientificFiles'):]


def test_report_and_figure_controls_reuse_the_existing_renderer_and_viewer():
    source = (STATIC / 'app.js').read_text()
    messages = source[source.index('function appendScientificFiles'):source.index('function renderArtifacts')]
    assert 'renderArtifacts(cached.revision, inventory, true)' in messages
    assert 'api(revisionPath(sessionId, turn.revision_id))' in messages
    assert 'revision.value?.revision_id === turn.revision_id' in messages
    assert 'revision.value.turn_id === turn.turn_id' in messages
    assert 'Promise.allSettled' in messages
    renderer = source[source.index('function renderArtifacts'):source.index('async function viewArtifact')]
    assert 'target.replaceChildren(...nodes)' in renderer
    assert 'viewArtifact(sessionId, revision.revision_id, artifact)' in renderer
    viewer = source[source.index('async function viewArtifact'):source.index('function readableLabel')]
    assert 'await loadRevision(revisionId)' in viewer
    assert viewer.index('const epoch = state.epoch') < viewer.index('await loadRevision(revisionId)')
    assert viewer.index('await loadRevision(revisionId)') < viewer.index('if (epoch !== state.epoch) return')
    assert 'showWorkspacePanel("results")' in viewer
    assert '/activate' not in messages + renderer + viewer


def test_downloads_use_scientific_figure_identity_and_honest_request_feedback():
    source = (STATIC / 'app.js').read_text()
    identity = source[source.index('function distinctFigureArtifacts'):source.index('function selectResultFigure')]
    key = next(line for line in identity.splitlines() if 'const key =' in line)
    assert 'artifact.revision_id' in key and 'artifact.figure.manifest_sha256' in key and 'artifact.figure.figure_id' in key
    assert 'artifact.sha256' not in key
    guard = source[source.index('function guardDownload'):source.index('function scientificFileNode')]
    assert 'event.preventDefault()' in guard and 'Requested — retry if needed' in guard
    assert '.blob(' not in guard and 'complete' not in guard.lower().split('//')[0]
