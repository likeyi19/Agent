"""Browser contracts for optional scientific transfer and ordinary chat submission."""
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
    assert tags['upload-type'][0] == 'select'
    assert 'turn-form' in page.ancestors['upload-type']
    assert 'turn-form' in page.ancestors['input-choice']
    assert tags['upload-index'][1]['accept'] == '.tbi'
    assert 'multiple' not in tags['upload-index'][1] and 'required' not in tags['upload-index'][1]
    assert 'hidden' in tags['upload-index-fields'][1]


def test_resource_discovery_uses_server_registration_without_readiness_inference():
    catalog = section('async function loadResources()', 'function clearPendingUpload')
    assert 'api("/resources")' in catalog
    assert 'catalog.enabled === true' in catalog
    assert 'catalog.choices' in catalog
    assert 'Scientific attachments are unavailable on this server.' in catalog
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
    assert 'request.send(upload.index ? new Blob([upload.file, upload.index]) : upload.file)' in transfer
    assert 'index_filename=${encodeURIComponent(upload.index.name)}&source_size=${upload.file.size}' in transfer
    assert 'input_type=${encodeURIComponent(upload.inputType)}' in transfer
    assert '"progress"' in transfer and 'event.loaded' in transfer
    assert 'data.status !== "registered"' in transfer and 'data.input_type !== upload.inputType' in transfer
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
    helper = section('function selectedCompanion(', 'function saveResourceSelection()')
    assert 'choice.input_set_id === element("input-choice").value' in helper
    assert 'choice.h5ad_companion === true' in helper
    assert 'choice.fragments_companion === true' in helper
    assert 'choice.has_source_index === true' in source()


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


def test_fragment_role_and_pair_are_explicit_and_never_infer_a_scientific_workflow():
    picker = section('element("upload-file").addEventListener', 'element("upload-file-button").addEventListener')
    assert 'inputType: element("upload-type").value' in picker
    assert 'index: null' in picker
    assert 'clearPendingUpload(true)' in picker
    assert 'changePairedIndex(element("upload-index").files[0] || null)' in picker
    assert 'uploadId: newTurnId(), index' in picker
    assert 'clearPendingUpload()' in picker
    assert 'renderUploadType()' in picker
    assert not any(value in picker for value in ('endsWith(', '.match(', 'import_scATAC', 'submitRequest(', 'postSubmission('))
    controls = section('function updateControls()', 'function renderExecution()')
    assert 'element("upload-index").disabled' in controls
    assert 'element("clear-upload-index").disabled = uploading' in controls
    request = section('function submitRequest(', 'element("turn-form").addEventListener')
    assert 'selectedResourceType() === "h5ad"' in request
    assert 'reference_bundle_path' not in source() and 'source_profile' not in source()


def test_reopened_companion_choice_is_rechecked_against_safe_catalog_and_bound_per_turn():
    initialize = source()[source().index('async function initialize()'):]
    assert 'choice.input_set_id === conveniences.inputSetId' in initialize
    assert initialize.index('state.inputSets = inputs.choices') < initialize.index('resources = loadResources()')
    assert 'inputSetId: element("input-choice").value' in source()
    assert 'scientific format and compatibility are checked during import' in (STATIC / 'index.html').read_text()


def test_bam_has_one_source_and_context_selection_keeps_inspection_available():
    html = (STATIC / 'index.html').read_text()
    assert '<option value="bam">scATAC-seq BAM</option>' in html
    assert 'Upload one BAM; no index is required.' in html
    page = HierarchyPage(html)
    assert 'turn-form' in page.ancestors['bam-input-help']
    upload = section('function renderUploadType()', 'function transferUpload(')
    assert 'bam ? ".bam" : fastq ? ".fastq,.fq,.fastq.gz,.fq.gz" : ".h5ad"' in upload
    assert 'element("upload-index-fields").hidden = !fragments' in upload
    context = section('function inputContext()', 'function selectedResourceType()')
    assert 'No qualified BAM context — inspection available' in context
    assert 'item.bam_default === true' in context
    assert 'configured.bam_companion === true' in context
    companion = section('function selectedCompanion(', 'function saveResourceSelection()')
    assert 'inputType === "bam" ? choice.bam_companion === true' in companion
    assert not any(value in upload + context + companion for value in (
        'inspect_raw_scATAC', 'prepare_scATAC_bam_fragments', 'source_sha256', 'library_context_path',
    ))


def test_fastq_attribution_and_completion_stay_in_the_existing_composer():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    tags = {attrs['id']: (tag, attrs) for tag, attrs in page.tags if 'id' in attrs}
    for identifier in ('fastq-library', 'fastq-layout', 'fastq-role', 'fastq-lane',
                       'fastq-chunk', 'fastq-compression', 'fastq-members', 'complete-fastq-collection'):
        assert 'turn-form' in page.ancestors[identifier]
    assert tags['complete-fastq-collection'][1]['type'] == 'button'
    assert tags['fastq-collection-status'][1]['role'] == 'status'
    html = (STATIC / 'index.html').read_text()
    assert 'tenx-atac-r1-r2-r3.v1' in html and 'tenx-atac-r1-i2-r2.v1' in html
    assert 'Filenames do not assign roles.' in html
    assert 'I1 (optional index)' in html
    assert 'value="plain">Plain FASTQ' in html and 'value="gzip">gzip FASTQ' in html


def test_fastq_completed_members_are_separate_from_selectable_libraries():
    catalog = section('async function loadResources()', 'function clearPendingUpload')
    assert 'catalog.fastq_members' in catalog and 'renderFastqMembers()' in catalog
    uploaded = section('async function uploadSelectedFile()', 'element("upload-file").addEventListener')
    assert 'resource.input_type === "fastq"' in uploaded
    assert 'state.fastqMembers =' in uploaded and 'renderFastqMembers(resource.resource_id)' in uploaded
    assert 'Upload the remaining declared roles, then complete the FASTQ library.' in uploaded
    member_view = section('function renderFastqMembers(', 'function fastqCollectionStatus(')
    assert 'checkbox.value = member.resource_id' in member_view
    assert 'facts.role' in member_view and 'facts.library_id' in member_view
    assert 'facts.lane' in member_view and 'facts.chunk' in member_view
    assert 'renderResources(' not in member_view


def test_fastq_transfer_pins_explicit_attribution_without_filename_inference():
    attribution = section('function fastqAttribution()', 'function selectedFastqMembers()')
    for identifier in ('fastq-library', 'fastq-layout', 'fastq-role', 'fastq-lane', 'fastq-chunk', 'fastq-compression'):
        assert f'element("{identifier}").value' in attribution
    assert 'Object.freeze' in attribution
    transfer = section('function transferUpload(', 'async function uploadSelectedFile()')
    assert 'Object.entries(upload.fastq)' in transfer
    assert 'encodeURIComponent(value)' in transfer
    assert not any(value in attribution + transfer for value in ('endsWith(', '.match(', 'FileReader', '.arrayBuffer('))
    edits = section('for (const id of ["fastq-library"', 'function changePairedIndex(')
    assert 'uploadId: newTurnId(), fastq: fastqAttribution()' in edits


def test_fastq_completion_sends_opaque_members_and_does_not_select_a_workflow():
    complete = section('async function completeFastqCollection()', 'function transferUpload(')
    assert 'api("/uploads/fastq-collections", "POST", completion.body)' in complete
    assert 'member_ids: Object.freeze(selectedFastqMembers())' in complete
    assert 'collection_id: newTurnId()' in complete and 'if (!state.fastqCompletion)' in complete
    assert 'completion.pending = false' in complete
    assert 'fastqCollectionStatus(`${error.code' in complete
    assert not any(value in complete for value in ('submitRequest(', 'postSubmission(',
        'raw_input_paths', 'source_path', 'record_sha256', '/sessions/', 'prepare_scATAC'))
    helper = section('function selectedCompanion(', 'function saveResourceSelection()')
    assert 'inputType === "fastq" ? choice.fastq_companion === true' in helper
    context = section('function inputContext()', 'function selectedResourceType()')
    assert 'item.fastq_default === true' in context
    assert 'configured.fastq_companion === true' in context
    assert 'No qualified FASTQ producer context — inspection available' in context
