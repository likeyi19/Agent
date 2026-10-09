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
    assert 'scientificArtifactPath(sessionId, artifact.revision_id, artifact.handle)' in files
    assert 'download.download = artifact.filename' in files
    assert '`${artifact.filename} — ${artifact.label}`' in files
    assert 'artifact.turn_id === turn.turn_id && artifact.revision_id === turn.revision_id' in files
    assert 'active_revision_id' not in files
    assert 'fetch(' not in files and '.blob(' not in files and 'createObjectURL' not in files
    assert 'scientific-artifacts/${encodeURIComponent(handle)}' in files


def test_inventory_is_cached_per_exact_accepted_execute_message_and_refreshed_on_reopen():
    source = (STATIC / 'app.js').read_text()
    files = source[source.index('function appendScientificFiles'):source.index('function renderArtifacts')]
    assert 'turn.response.kind !== "execute"' in files
    assert '!["succeeded", "activated", "stale"].includes(turn.status)' in files
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


def test_chat_and_details_share_reviewed_matrix_qc_selection_file_allowlist():
    source = (STATIC / 'app.js').read_text()
    policy = source[source.index('function supportedScientificFile'):source.index('function scientificFileNode')]
    for artifact_type, filename, label in (
        ('scientific_matrix', 'matrix.h5ad', 'Original accepted matrix'),
        ('scientific_qc_table', 'barcodes.tsv.gz', 'Barcode QC'),
        ('scientific_qc_table', 'lengths.tsv.gz', 'Fragment length distribution'),
        ('scientific_selection_table', 'decisions.tsv.gz', 'Cell-selection decisions'),
        ('scientific_selection_table', 'selected.tsv.gz', 'Selected cells'),
    ):
        assert artifact_type in policy and filename in policy and label in policy
    assert 'Object.hasOwn' in policy
    assert 'supportedScientificFile(artifact)' in source[source.index('function appendScientificFiles'):]
