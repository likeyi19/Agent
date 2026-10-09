"""Browser contracts for optional H5AD transfer and ordinary chat submission."""
from test_static import HierarchyPage, STATIC


def source():
    return (STATIC / 'app.js').read_text()


def section(start, end):
    code = source()
    return code[code.index(start):code.index(end, code.index(start))]


def test_attachment_is_optional_single_file_and_lives_in_the_chat_composer():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    tags = {attrs['id']: (tag, attrs) for tag, attrs in page.tags if 'id' in attrs}
    tag, picker = tags['upload-file']
    assert tag == 'input' and picker['type'] == 'file' and picker['accept'] == '.h5ad'
    assert 'multiple' not in picker and 'required' not in picker
    assert 'turn-form' in page.ancestors['upload-file']
    assert 'turn-form' in page.ancestors['resource-choice']
    ids = [attrs['id'] for _, attrs in page.tags if 'id' in attrs]
    assert ids.index('turn-form') < ids.index('upload-file') < ids.index('utterance')
    assert tags['upload-file-button'][1]['type'] == 'button'
    assert tags['clear-upload'][1]['type'] == 'button'
    assert tags['upload-status'][1]['role'] == 'status'
    assert tags['upload-progress'][1]['max'] == '100'


def test_resource_discovery_uses_server_registration_without_readiness_inference():
    catalog = section('async function loadResources()', 'function clearPendingUpload')
    assert 'api("/resources")' in catalog
    assert 'catalog.enabled === true' in catalog
    assert 'catalog.choices' in catalog
    assert 'H5AD attachments are unavailable on this server.' in catalog
    assert 'submitRequest(' not in catalog and 'postSubmission(' not in catalog
    render = section('function renderResources(', 'async function loadResources()')
    assert 'option.value = choice.resource_id' in render
    assert 'option.textContent' in render and 'choice.label' in render
    assert 'registered' in render and 'readiness' not in render and 'compatible' not in render
    assert 'innerHTML' not in source() and 'insertAdjacentHTML' not in source()


def test_binary_transfer_never_submits_science_or_buffers_the_file_in_json():
    transfer = section('function transferUpload(', 'async function uploadSelectedFile()')
    assert 'request.open("PUT"' in transfer
    assert 'encodeURIComponent(upload.uploadId)' in transfer
    assert 'encodeURIComponent(upload.file.name)' in transfer
    assert 'application/octet-stream' in transfer
    assert 'request.send(upload.file)' in transfer
    assert '"progress"' in transfer and 'event.loaded' in transfer
    assert 'data.status !== "registered"' in transfer and 'data.input_type !== "h5ad"' in transfer
    assert not any(value in transfer for value in (
        'JSON.stringify(upload.file)', 'FileReader', '.arrayBuffer(',
        'submitRequest(', 'postSubmission(', '/sessions/',
    ))


def test_upload_retry_preserves_file_identity_and_selection_creates_a_new_identity():
    upload = section('async function uploadSelectedFile()', 'element("new-session")')
    retry = upload[:upload.index('element("upload-file").addEventListener')]
    assert 'const upload = state.upload' in retry and 'await transferUpload(upload)' in retry
    assert 'upload.status = "failed"' in retry
    assert 'newTurnId()' not in retry
    picker = upload[upload.index('element("upload-file").addEventListener'):]
    assert 'const file = element("upload-file").files[0]' in picker
    assert 'uploadId: newTurnId(), file' in picker
    assert 'state.upload !== upload' in retry
    assert 'Retry upload' in source()
    clear = section('function clearPendingUpload(', 'function transferUpload(')
    assert clear.index('state.upload = null') < clear.index('upload.request.abort()')


def test_turn_resource_binding_admits_only_declared_companions_and_retry_keeps_the_frozen_body():
    request = section('function submitRequest(', 'element("turn-form").addEventListener')
    assert 'body.resource_id = element("resource-choice").value' in request
    assert 'else if (element("input-choice").value) body.input_set_id' in request
    assert 'if (selectedCompanion()) body.input_set_id' in request
    assert 'body.epizoo_resource_id = element("epizoo-resource-choice").value' in request
    assert 'Object.freeze(body)' in request
    assert 'predecessor_turn_id = predecessorTurnId' in request
    assert 'expected_generation: state.session.generation' in request
    assert 'profile_id: element("model-choice").value' in request
    assert 'postSubmission(state.submission)' in source()
    resource = section('element("resource-choice").addEventListener', 'element("new-session")')
    assert 'element("input-choice").value = ""' in resource
    assert '!selectedCompanion()' in resource
    configured = section('element("input-choice").addEventListener', 'element("utterance").addEventListener')
    assert 'element("resource-choice").value = ""' in configured
    assert '!selectedCompanion()' in configured
    assert 'execution_inputs' not in source() and 'source_sha256' not in source()


def test_resource_and_companion_controls_use_only_safe_configured_choices():
    initialize = source()[source().index('async function initialize()'):]
    assert 'api("/epizoo-resources")' in initialize
    assert 'state.inputSets = inputs.choices' in initialize
    assert 'option.value = choice.resource_id' in initialize
    assert 'choice.display_label' in initialize and 'choice.is_default' in initialize
    assert 'checkpoint_path' not in source() and 'frequencies_sha256' not in source()
    helper = section('function selectedCompanion()', 'function saveResourceSelection()')
    assert 'choice.input_set_id === element("input-choice").value' in helper
    assert 'choice.h5ad_companion === true' in helper


def test_resource_selection_is_only_a_rechecked_display_convenience():
    convenience = section('function saveResourceSelection()', 'function uploadStatus(')
    assert 'localStorage.setItem(RESOURCE_STORAGE_KEY, element("resource-choice").value)' in convenience
    render = section('function renderResources(', 'async function loadResources()')
    assert 'state.resources.some((choice) => choice.resource_id === selectedId)' in render
    initialize = source()[source().index('async function initialize()'):]
    assert 'loadResources()' in initialize
    assert 'uploadSelectedFile(' not in initialize and 'postSubmission(' not in initialize
    assert 'await openSession(conveniences.sessionId' in initialize


def test_pending_upload_has_separate_status_and_can_be_removed_for_text_only_chat():
    controls = section('function updateControls()', 'function renderExecution()')
    assert 'cannotSubmit' in controls and 'Boolean(state.upload)' in controls
    assert 'element("resource-choice").disabled' in controls
    assert 'element("upload-file").disabled' in controls
    assert 'element("clear-upload").hidden = !state.upload' in controls
    status = section('function uploadStatus(', 'function renderResources(')
    assert 'element("upload-status").textContent = text' in status
    assert 'element("turn-status")' not in status
    assert 'You can send a message without an attachment.' in source()
