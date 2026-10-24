"""Resolve categorical reads through accepted outputs and recorded dependencies.

Identity and access live here; observation decoding and statistics live with the
scientific owner. Optional figure delivery uses the existing composer and pins.
"""
from agent.schemas import StepOutputRef, PriorOutputRef
from agent.orchestration.prior_outputs import accepted_step_digest, validate_binding
from agent.orchestration.verification_authority import _accepted_step

CATEGORICAL_TOOLS = ('pca_embed_cells', 'build_cell_neighbors', 'cluster_cells',
                     'compute_cell_umap', 'annotate_scATAC_cell_types')


def dependency(store, run_id, step_id, argument):
    stored, planned, _ = _accepted_step(store, run_id, step_id)
    ref = planned.arguments.get(argument)
    if isinstance(ref, StepOutputRef):
        if ref.step_id not in planned.depends_on:
            raise ValueError('Scientific dependency is not declared.')
        source_run, source_step, output = run_id, ref.step_id, ref.output_key
    elif isinstance(ref, PriorOutputRef):
        validate_binding(ref.binding, store=store)
        source_run, source_step, output = ref.binding.run_id, ref.binding.step_id, ref.output_key
    else:
        raise ValueError('Reading requires recorded scientific source lineage.')
    source, _, state = _accepted_step(store, source_run, source_step)
    if (state.lifecycle_status.value != 'SUCCEEDED'
            or source.result[output] != stored.resolved_arguments[argument]):
        raise ValueError('Accepted scientific dependency changed.')
    return source_run, source_step


def analysis_lineage(store, run_id, step_id):
    """The recorded cell-analysis chain, shared by source and method readers."""
    stored, planned, _ = _accepted_step(store, run_id, step_id)
    lineage = [(run_id, stored, planned)]
    for tool, argument in (('compute_cell_umap', 'analysis_path'), ('cluster_cells', 'analysis_path'),
                           ('build_cell_neighbors', 'embedding_path')):
        if stored.tool_name != tool:
            continue
        if not isinstance(planned.arguments.get(argument), (StepOutputRef, PriorOutputRef)):
            break
        run_id, step_id = dependency(store, run_id, step_id, argument)
        stored, planned, _ = _accepted_step(store, run_id, step_id)
        lineage.append((run_id, stored, planned))
    return tuple(lineage)


def original_source(store, run_id, step_id):
    """Follow only exact accepted lineage; legacy EpiZoo has no input digest."""
    run_id, stored, _ = analysis_lineage(store, run_id, step_id)[-1]
    if stored.tool_name == 'annotate_scATAC_cell_types':
        if stored.resolved_arguments.get('source_path'):
            return dict(path=stored.resolved_arguments['source_path'],
                        sha256=stored.resolved_arguments['source_sha256'])
        return None
    if stored.tool_name != 'pca_embed_cells':
        return None
    from agent.orchestration.verification_authority import _accepted
    _accepted(store, run_id, stored.step_id)  # Existing authority anchor; no publication reconstruction.
    return dict(path=stored.result['input_path'], sha256=stored.result['source_sha256'])


def _files(sessions, run, step):
    result = step.result
    files = {}
    if step.tool_name == 'annotate_scATAC_cell_types':
        files['result'] = dict(path=result['annotated_h5ad_path'], sha256=result['annotated_h5ad_sha256'],
            role='accepted_result_observation')
    elif 'analysis_sha256' in result:
        files['result'] = dict(path=result['analysis_path'], sha256=result['analysis_sha256'],
            role='computational_assignment')
    try:
        original = original_source(sessions._application.run_store, run.run_id, step.step_id)
    except (ValueError, OSError, KeyError):
        original = None  # Missing lineage limits association, not independent facts.
    if original is not None:
        files['source'] = dict(original, role='supplied_source_observation; biological validity not assessed')
    return files


def _role(step, location, name, file):
    if location == 'result' and step.tool_name == 'annotate_scATAC_cell_types':
        from agent.tools.analysis.annotation_contract import OBS_COLUMNS
        return 'computational_assignment' if name in OBS_COLUMNS else 'copied_supplied_source_observation; biological validity not assessed'
    return file['role']


def inventory(sessions, run, step, *, offset=0, limit=32, field=None):
    from agent.tools.analysis.categorical import fields
    if step.tool_name not in CATEGORICAL_TOOLS:
        return dict(fields=[], complete=True)
    from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
    entries, unavailable = [], []
    for location, file in _files(sessions, run, step).items():
        # Metadata discovery is shallow. Selected reading checks the exact bytes.
        try:
            names = fields(file['path'])
        except (ValueError, OSError, KeyError):
            unavailable.append(location)
            continue
        for name in names:
            if _unsafe_fact_field(name) or _unsafe_fact_value(name):
                unavailable.append(location)
                continue
            handle = location + ':' + name
            if field is None or field == handle:
                entries.append(dict(field=handle, label=name, role=_role(step, location, name, file)))
    page = entries[offset:offset+limit]
    return dict(fields=page, total_fields=len(entries), offset=offset, returned_fields=len(page),
                next_offset=offset+len(page) if offset+len(page) < len(entries) else None,
                complete=offset == 0 and len(entries) <= limit and not unavailable,
                unavailable_locations=sorted(set(unavailable)))


def _identity(sessions, run, step, location, name, file, files, *, associate):
    origin = files.get('source')
    # Same artifact fields need no original-source claim. Across publications,
    # exact common source identity AND ordered axes are required.
    identity = dict(artifact_sha256=file['sha256'], field=name, location=location, role=_role(step, location, name, file),
                    accepted_step_sha256=accepted_step_digest(step),
                    run_id=run.run_id, step_id=step.step_id,
                    source_sha256=None if origin is None else origin['sha256'])
    if associate and origin is None:
        # A shared qualified graph already identifies its cell population. It
        # need not identify the original count file to compare two assignments.
        from .dialogue_evidence import _clustering_neighbors_facts
        planned = next(p for p in run.plan.steps if p.step_id == step.step_id)
        facts = _clustering_neighbors_facts(sessions, run, step, planned)
        identity['neighbors_identity_sha256'] = next((f.value for f in facts
            if f.status == 'available' and f.field in (
                'clustering_neighbors_identity_sha256', 'umap_neighbors_identity_sha256')), None)
    return identity


def _column(sessions, run, step, selected, *, associate=False):
    from agent.tools.analysis.categorical import observations
    location, name = selected.split(':', 1)
    files = _files(sessions, run, step)
    file = files[location]
    ids, values = observations(file['path'], file['sha256'], name)
    identity = _identity(sessions, run, step, location, name, file, files, associate=associate)
    if location == 'source':
        # A source field must describe the exact entity axis of this result.
        own = files.get('result')
        if own is not None:
            from agent.tools.analysis.categorical import axis, require_same_cells
            require_same_cells(ids, axis(own['path'], own['sha256']))
    return ids, values, identity


def qualified_sources(left, right):
    """The existing W2 publication relationship, without entity matching."""
    if (left['artifact_sha256'] == right['artifact_sha256']
            and all(left[k] == right[k] for k in ('run_id', 'step_id', 'accepted_step_sha256'))):
        return 'same_accepted_artifact'
    if left['source_sha256'] is not None and left['source_sha256'] == right['source_sha256']:
        return 'exact_original_source'
    if (left.get('neighbors_identity_sha256') is not None
            and left.get('neighbors_identity_sha256') == right.get('neighbors_identity_sha256')):
        return 'same_qualified_neighbors'
    raise ValueError('Accepted sources have no qualified scientific population relationship.')


def cell_overlay(sessions, coordinate, categories, field):
    """Read exact accepted coordinates and explicitly qualified categories."""
    from agent.tools.analysis import categorical as owner
    from agent.tools.analysis.embedding_analysis import _cell_order_digest
    store = sessions._application.run_store
    cs, _, cr = _accepted_step(store, coordinate['run_id'], coordinate['step_id'])
    if (accepted_step_digest(cs) != coordinate['accepted_step_sha256']
            or coordinate['output_key'] not in cs.result):
        raise ValueError('Changed accepted visualization input.')
    if cs.tool_name != 'compute_cell_umap':
        raise ValueError('Select an accepted UMAP coordinate result.')
    files = _files(sessions, cr, cs)
    file = files['result']
    ids, xy = owner.coordinates(file['path'], file['sha256'])
    identity = _identity(sessions, cr, cs, 'result', 'X_umap', file, files, associate=True)
    coordinate_source = dict(producing_step_id=cs.step_id, tool_name=cs.tool_name,
        result_field='analysis_path', artifact_kind='cell_umap_h5ad',
        artifact_path=file['path'], integrity=dict(sha256=file['sha256']),
        role='coordinates', accepted_output=dict(coordinate), field='obsm:X_umap',
        source_sha256=identity['source_sha256'],
        **({'neighbors_identity_sha256': identity['neighbors_identity_sha256']}
           if identity.get('neighbors_identity_sha256') is not None else {}))
    if 'registered_source' in categories:
        if not field.startswith('source:'):
            raise ValueError('A registered source selects an exact original observation field.')
        from .local_resources import LocalResourceAdmission
        selected = categories['registered_source']
        record = LocalResourceAdmission(sessions._application._workspace).load(selected['resource_id'])
        if record.record_sha256 != selected['record_sha256'] or record.source_sha256 != selected['source_sha256']:
            raise ValueError('Changed registered visualization source.')
        name = field.split(':', 1)[1]
        from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
        if _unsafe_fact_field(name) or _unsafe_fact_value(name) or name not in owner.fields(record.source_path):
            raise ValueError('Select an exact eligible categorical field.')
        other_ids, labels = owner.observations(record.source_path, record.source_sha256, name)
        other_identity = dict(artifact_sha256=record.source_sha256, source_sha256=record.source_sha256,
                              run_id=None, step_id=None, accepted_step_sha256=None)
        category_source = dict(role='categories', field=field, artifact_kind='categorical_observations',
            artifact_path=record.source_path, integrity=dict(sha256=record.source_sha256),
            registered_source=dict(selected), source_sha256=record.source_sha256,
            observation_role='supplied_source_observation; biological validity not assessed')
    else:
        ls, _, lr = _accepted_step(store, categories['run_id'], categories['step_id'])
        if (accepted_step_digest(ls) != categories['accepted_step_sha256']
                or categories['output_key'] not in ls.result):
            raise ValueError('Changed accepted visualization input.')
        if not inventory(sessions, lr, ls, field=field)['fields']:
            raise ValueError('Select an exact eligible categorical field.')
        other_ids, labels, other_identity = _column(sessions, lr, ls, field, associate=True)
        artifact = _files(sessions, lr, ls)[field.split(':', 1)[0]]
        category_source = dict(producing_step_id=ls.step_id, tool_name=ls.tool_name,
            **({'result_field': 'annotated_h5ad_path' if ls.tool_name == 'annotate_scATAC_cell_types'
                 else 'analysis_path'} if field.startswith('result:') else {}),
            artifact_kind='categorical_observations', artifact_path=artifact['path'],
            integrity=dict(sha256=artifact['sha256']), role='categories',
            accepted_output=dict(categories), field=field, source_sha256=other_identity['source_sha256'],
            **({'neighbors_identity_sha256': other_identity['neighbors_identity_sha256']}
               if other_identity.get('neighbors_identity_sha256') is not None else {}))
    basis = qualified_sources(identity, other_identity)
    owner.require_same_cells(ids, other_ids)
    from .dialogue_evidence import _unsafe_fact_value
    if _unsafe_fact_value(labels.tolist()):
        raise ValueError('Selected labels are ineligible for public figure delivery.')
    return dict(cell_ids=ids, coordinates=xy, labels=labels, field=field,
        sources=[coordinate_source, category_source],
        correspondence=dict(basis=basis, n_cells=len(ids), cell_order_sha256=_cell_order_digest(ids)))


def _captured_output(sessions, state, origin, handle):
    """Resolve one offered W1 group through its existing accepted locator."""
    from dataclasses import asdict
    from .turn_context import result_groups
    groups = {g['handle']:g['canonical'] for g in result_groups(origin.snapshot, state)}
    if handle not in groups:
        raise ValueError('Select an explicitly captured accepted result.')
    item = groups[handle]
    revision = next(r for r in state.revisions if r.revision_id == item['revision_id'])
    output = next(o for o in revision.outputs if o.name == item['output_name'])
    sessions._validate_locator(output)
    return dict(asdict(output), revision_id=revision.revision_id)


def _selected_output(sessions, state, interaction, purpose):
    """Resolve an explicit output purpose without tool/order inference."""
    from dataclasses import asdict
    from .session_state import selection_uses
    turn = state.turn(interaction.turn_id)
    selected = [s for s in turn.selections if purpose in selection_uses(s)]
    if len(selected) != 1:
        raise ValueError('Select exactly one new result for the requested purpose.')
    run = sessions._application.run_store.load(turn.run_id)
    outputs = [o for o in sessions._locators(turn, run) if o.name == selected[0].name]
    if len(outputs) != 1:
        raise ValueError('The selected result is unavailable.')
    output = outputs[0]
    revisions = [r for r in state.revisions if r.revision_id == turn.revision_id
                 and r.turn_id == interaction.turn_id and r.run_id == turn.run_id]
    if len(revisions) != 1 or output not in revisions[0].outputs:
        raise ValueError('The requested new result has not been accepted.')
    return dict(asdict(output), revision_id=revisions[0].revision_id)


def _captured_source_binding(origin):
    """Use the exact ordered source offer shown to the interpreter."""
    from .local_resources import available_input_view, LocalResourceAdmission
    if (origin.submission or {}).get('candidate_input') is not None:
        from .raw_handoff import _candidate_view
        viewed = _candidate_view(origin)
    else:
        viewed = available_input_view(origin, allow_empty=True)
    return LocalResourceAdmission._submission_binding(viewed.submission)


def _captured_source(sessions, session_id, origin, handle):
    """Resolve the same captured registration offered by typed role admission."""
    from .local_resources import LocalResourceAdmission
    binding = _captured_source_binding(origin)
    members = (binding.members if hasattr(binding, 'members') else
               (dict(resource_id=binding.resource_id, record_sha256=binding.record_sha256),))
    index = int(handle.split('.')[1])
    if not 0 <= index < len(members):
        raise ValueError('Select an explicitly captured source.')
    member = members[index]
    owner = LocalResourceAdmission(sessions._application._workspace)
    record = owner.load(member['resource_id'])
    if record.record_sha256 != member['record_sha256'] or record.session_id not in (None, session_id):
        raise ValueError('Changed or unavailable captured visualization source.')
    return dict(registered_source=dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256, source_sha256=record.source_sha256))


def comparison_inputs(sessions, session_id, interaction):
    """Bind captured/new comparison operands to exact accepted targets.

    This selects sources only. Existing evidence readers still own scientific
    field eligibility, population correspondence and ordered-axis checks.
    """
    from .dialogue_execution import _received_origin
    from agent.schemas.orchestration import _serialize
    state = sessions.load(session_id)
    origin = _received_origin(interaction, state.interactions)
    request = (origin.admitted or {}).get('comparison')
    if request is None:
        return None
    def target(operand):
        output = (_selected_output(sessions, state, interaction, 'comparison_result')
                  if operand == 'new' else _captured_output(sessions, state, origin, operand))
        return dict(revision_id=output['revision_id'], output_name=output['name'],
                    accepted_step_sha256=output['accepted_step_sha256'], subject=None)
    return target(request['left']), target(request['right']), _serialize(request['detail'])


def visualization_inputs(sessions, session_id, interaction):
    """Resolve one precise request against its accepted outputs and capture."""
    from .dialogue_execution import _received_origin
    state = sessions.load(session_id)
    origin = _received_origin(interaction, state.interactions)
    drawing = (origin.admitted or {}).get('visualization')
    if drawing is None:
        return None
    coordinates = (_selected_output(sessions, state, interaction, 'visualization_coordinates')
                   if drawing['coordinates'] == 'new' else _captured_output(sessions, state, origin, drawing['coordinates']))
    if drawing['labels'].startswith('source.'):
        labels = _captured_source(sessions, session_id, origin, drawing['labels'])
    else:
        labels = (_selected_output(sessions, state, interaction, 'visualization_labels')
                  if drawing['labels'] == 'new' else _captured_output(sessions, state, origin, drawing['labels']))
    return coordinates, labels, drawing['field']


def read(sessions, state, run, step, request):
    from .dialogue_evidence import EvidenceFact, _accepted, _one, _bytes, MAX_FACT_BYTES, _Unavailable
    from agent.tools.analysis import categorical as owner
    if request.section == 'categorical_fields':
        value = inventory(sessions, run, step, offset=request.offset, limit=request.limit, field=request.field)
        return (EvidenceFact('categorical_fields', 'available', value,
                            reason='on_demand_accepted_result_metadata'),)
    left_ids, left, left_source = _column(sessions, run, step, request.field,
                                       associate=request.section == 'category_association')
    sources = [left_source]
    if request.section == 'category_counts':
        value = owner.counts(left, offset=request.offset, limit=request.limit)
    else:
        other_run, other_step = run, step
        if request.comparison is not None:
            other = request.comparison
            other_revision = _one([r for r in state.revisions if r.revision_id == other['revision_id']])
            other_output = _one([o for o in other_revision.outputs if o.name == other['output_name']
                and o.accepted_step_sha256 == other['accepted_step_sha256']])
            other_run, other_step, _, _, _ = _accepted(sessions, state, other_revision, other_output)
        right_ids, right, right_source = _column(sessions, other_run, other_step, request.other_field, associate=True)
        try:
            qualified_sources(left_source, right_source)
        except ValueError as exc:
            raise _Unavailable('categorical_source_mismatch') from exc
        try:
            value = owner.association(left_ids, left, right_ids, right, offset=request.offset, limit=request.limit)
        except ValueError as exc:
            raise _Unavailable('categorical_axis_mismatch') from exc
        sources.append(right_source)
    value['fields'] = [dict(field=s['field'], location=s['location'], role=s['role']) for s in sources]
    from agent.tools.analysis.embedding_analysis import _cell_order_digest
    derivation = dict(read_version=owner.READ_VERSION, sources=sources,
                      request=dict(section=request.section, field=request.field, other_field=request.other_field,
                                   offset=request.offset, limit=request.limit),
                      cell_order_sha256=_cell_order_digest(left_ids),
                      comparison=None if request.comparison is None else dict(request.comparison))
    # Whole values retain the existing budget. A too-large record is explicit;
    # callers can select a smaller page without pretending it was complete.
    omitted = len(_bytes(value)) > MAX_FACT_BYTES
    return (EvidenceFact(request.section, 'omitted' if omitted else 'available', None if omitted else value,
        source_pointer='/obs', reason='field_size_limit' if omitted else 'on_demand_accepted_result_read',
        artifact_name='accepted_categorical_observations', artifact_sha256=left_source['artifact_sha256'],
        derivation=derivation),)
