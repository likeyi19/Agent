"""Offer accepted clustering through the shared transient result handoff.

No grouping algorithm, source alignment or adapter publication lives here.
"""
from agent.orchestration.prior_outputs import accepted_step_digest, bind_output, validate_binding
from agent.orchestration.verification_authority import _accepted_step
from agent.schemas import PriorOutputRef
from agent.schemas.verification_authority import AuthorityError
from .session_state import digest
from .result_reading import handoff


def accepted_producer(sessions, reference):
    """Resolve a planned accepted dataset to its recorded grouping producer."""
    if not isinstance(reference, PriorOutputRef) or reference.output_key != 'matrix_path':
        return None
    app = sessions._application
    binding = reference.binding
    validate_binding(binding, store=app.run_store, registry=app.registry)
    view = handoff(app.run_store, binding.run_id, binding.step_id)
    return None if view is None else view['groups']


def same_producer(grouping, selected):
    return grouping is not None and all(grouping[key] == selected[key]
        for key in ('run_id', 'step_id', 'accepted_step_sha256'))


def producer_reference(sessions, producer):
    app = sessions._application
    binding = bind_output(app.run_store, app.registry, producer['run_id'], producer['step_id'], 'dataset')
    if binding.accepted_step_sha256 != producer['accepted_step_sha256']:
        raise AuthorityError('Accepted grouping producer anchor changed.')
    return PriorOutputRef(binding, 'matrix_path')


def choices(sessions, interaction, *, state=None):
    """Offer only clusters on explicit captured-result ancestry, without file IO."""
    state = state if state is not None else sessions.load(sessions._interaction_session_id)
    if interaction.base_revision_id is None:
        return []
    revision = next(r for r in state.revisions if r.revision_id == interaction.base_revision_id)
    store = sessions._application.run_store
    found = {}
    for output in revision.outputs:
        stored, planned, _ = _accepted_step(store, output.run_id, output.step_id)
        if accepted_step_digest(stored) != output.accepted_step_sha256:
            raise ValueError('Captured grouping result anchor changed')
        capability = handoff(store, output.run_id, output.step_id)
        if capability is None or capability['groups'] is None:
            continue
        grouping = capability['groups']
        run_id, step_id = grouping['run_id'], grouping['step_id']
        stored, _, _ = _accepted_step(store, run_id, step_id)
        identity = dict(run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(stored))
        handle = 'groups.' + digest(identity)
        found[handle] = dict(handle=handle, **identity, n_cells=stored.result['n_cells'],
            n_groups=stored.result.get('n_clusters'), resolution=stored.result.get('resolution',
                stored.result.get('parameters', {}).get('kwargs', {}).get('resolution')))
    return list(found.values())


def public_choices(sessions, interaction):
    return [{k: row[k] for k in ('handle', 'n_cells', 'n_groups', 'resolution')}
            for row in choices(sessions, interaction)]


def selected_results(sessions, interaction, handle, *, state=None):
    """Lower a typed grouping selection to its captured accepted result operands.

    Only actual accepted grouping ancestry can make an operand a scientific
    input. Presentation references and similarly named observation fields cannot.
    """
    from .turn_context import result_groups
    from .turn_decisions import IntentError
    state = state if state is not None else sessions.load(sessions._interaction_session_id)
    offered = [row for row in choices(sessions, interaction, state=state) if row['handle'] == handle]
    if len(offered) != 1:
        raise IntentError('unavailable_context')
    selected = offered[0]
    result = []
    for group in result_groups(interaction.snapshot, state):
        source = group['canonical']
        if source['revision_id'] != interaction.base_revision_id:
            continue
        revision = next(r for r in state.revisions if r.revision_id == source['revision_id'])
        output = next(o for o in revision.outputs if o.name == source['output_name'])
        if output.accepted_step_sha256 != source['accepted_step_sha256']:
            raise IntentError('unavailable_context')
        view = handoff(sessions._application.run_store, output.run_id, output.step_id)
        lineage = None if view is None else view['groups']
        if same_producer(lineage, selected):
            result.append(group['handle'])
    if not result:
        raise IntentError('unavailable_context')
    return tuple(result)
