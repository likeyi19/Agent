"""Shared composer selection remains separate from biological role admission."""
from test_static import HierarchyPage, STATIC
from test_upload_static import section, source


def test_ordered_input_controls_are_small_and_stay_in_the_chat_composer():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    tags = {attrs['id']: (tag, attrs) for tag, attrs in page.tags if 'id' in attrs}
    assert tags['add-scientific-input'][1]['type'] == 'button'
    assert tags['selected-scientific-inputs'][0] == 'ol'
    for identifier in ('add-scientific-input', 'selected-scientific-inputs'):
        assert 'turn-form' in page.ancestors[identifier]
    assert 'Selecting inputs does not start an analysis.' in (STATIC / 'index.html').read_text()


def test_selection_keeps_explicit_order_and_only_submits_opaque_catalog_ids():
    code = source()
    selected = section('function selectedResourceIds()', 'function renderSelectedResources()')
    assert '[...state.selectedResourceIds' in selected
    assert '.sort(' not in selected
    add = section('element("add-scientific-input").addEventListener', 'element("new-session")')
    assert 'state.selectedResourceIds = selectedResourceIds()' in add
    assert 'postSubmission(' not in add and 'submitRequest(' not in add
    rendered = section('function renderSelectedResources()', 'function selectedResourceType()')
    assert 'label.textContent = resource.label' in rendered
    assert 'Remove' in rendered and '.filter((value) => value !== id)' in rendered
    assert not any(term in rendered for term in ('reference', 'query', 'ground_truth', 'innerHTML'))
    assert 'MAX_SELECTED_RESOURCES = 8' in code


def test_multi_source_retry_freezes_captured_selection_and_singles_keep_the_old_wire():
    submitted = section('function submitRequest(', 'element("turn-form").addEventListener')
    assert 'body.resource_ids = Object.freeze(selectedIds)' in submitted
    assert 'body.resource_id = element("resource-choice").value' in submitted
    assert 'Object.freeze(body)' in submitted
    assert 'expected_generation: state.session.generation' in submitted
    assert 'predecessor_turn_id = predecessorTurnId' in submitted
    assert not any(term in submitted for term in ('source_sha256', 'record_sha256', 'roles', 'tool_name'))


def test_reopen_checks_display_selection_against_the_catalog_without_execution():
    loaded = section('async function loadResources()', 'function clearPendingUpload')
    assert 'new Set(saved).size === saved.length' in loaded
    assert 'saved.length <= MAX_SELECTED_RESOURCES' in loaded
    assert 'savedIds.filter((id) => state.resources.some' in loaded
    assert 'savedIds.slice(0, -1)' in loaded
    assert 'submitRequest(' not in loaded and 'postSubmission(' not in loaded
    assert 'checkpoint_path' not in source() and 'record_sha256' not in source()


def test_group_add_affordance_matches_existing_h5ad_only_contract():
    helper = section('function canAddScientificInput()', 'function renderSelectedResources()')
    assert 'resource.input_type === "h5ad"' in helper
    assert 'ids.length < MAX_SELECTED_RESOURCES' in helper
    assert not any(term in helper for term in ('reference', 'query', 'ground_truth', 'species'))
    rendered = section('function renderResources(', 'async function loadResources()')
    assert 'state.selectedResourceIds.length > 0 && choice.input_type !== "h5ad"' in rendered
    assert 'Multiple inputs support 2–8 H5AD files.' in (STATIC / 'index.html').read_text()


def test_incompatible_new_registration_does_not_replace_or_extend_queued_h5ad_selection():
    rendered = section('function renderResources(', 'async function loadResources()')
    assert 'restoreSavedSelection || !state.selectedResourceIds.length || selected.input_type === "h5ad"' in rendered
    assert 'return Boolean(selectedId && element("resource-choice").value === selectedId)' in rendered
    assert rendered.index('if (selected && !canSelect) state.selectedResourceIds = selectedResourceIds()') < rendered.index('replaceChildren')
    assert 'state.selectedResourceIds = []' not in rendered and '.sort(' not in rendered
    loaded = section('async function loadResources()', 'function clearPendingUpload')
    assert ': selectedId, true)' in loaded
    uploaded = section('async function uploadSelectedFile()', 'element("upload-file").addEventListener')
    assert 'const selected = renderResources(resource.resource_id)' in uploaded
    assert 'registered, not selected' in uploaded
    completed = section('async function completeFastqCollection()', 'function transferUpload(')
    assert 'const selected = renderResources(collection.resource_id)' in completed
    assert 'library registered, not selected' in completed
    picker = section('element("upload-file").addEventListener', 'element("upload-type").addEventListener')
    assert 'state.selectedResourceIds.length && state.upload.inputType !== "h5ad"' in picker
    assert picker.index('state.selectedResourceIds = selectedResourceIds()') < picker.index('element("resource-choice").value = ""')
