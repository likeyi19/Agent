"""Metadata-only turn snapshots. Executable trust is checked later by M15.3."""
from dataclasses import asdict
from agent.schemas.orchestration import _serialize
from .session_state import OutputLocator
from .turn_decisions import IntentError

SELECTION = 'select_scATAC_cells'
MATRIX = 'build_scATAC_cell_by_ccre'


def parameter_specs(registry):
    spec = registry.get(SELECTION)
    return {k:v for k,v in {**spec.required_arguments, **spec.optional_arguments}.items()
            if v.planning is not None and v.planning.scientific_parameter}


def stored_step(sessions, locator):
    locator = OutputLocator(**locator) if isinstance(locator, dict) else locator
    sessions._validate_locator(locator)
    state = sessions._application.run_store.load(locator.run_id)
    return next(s for s in state.steps if s.step_id == locator.step_id)


def snapshot(sessions, state, *, tool_names=None):
    revisions = {r.revision_id:r for r in state.revisions}
    current = revisions.get(state.active_revision_id)
    if current is None: raise IntentError('unavailable_context')
    relations = {'current':current.revision_id}
    if current.parent_revision_id is not None: relations['parent'] = current.parent_revision_id
    previous = next((e.from_revision_id for e in reversed(state.navigation)
                     if e.to_revision_id != e.from_revision_id), None)
    if previous is not None: relations['previous_active'] = previous
    bases = {}
    specs = parameter_specs(sessions._application.registry)
    for rid in dict.fromkeys(relations.values()):
        revision = revisions[rid]
        operations = []
        matrix_source = None
        for locator in revision.outputs:
            step = stored_step(sessions, locator)
            if tool_names is not None:
                tool_names[rid, locator.name] = step.tool_name
            if step.tool_name == MATRIX:
                if matrix_source is not None: raise IntentError('unavailable_context')
                matrix_source = asdict(locator)
            if step.tool_name != SELECTION: continue
            if any(o['source']['run_id'] == locator.run_id and o['source']['step_id'] == locator.step_id for o in operations):
                continue
            parameters = {k:_serialize(step.resolved_arguments[k]) for k in specs if k in step.resolved_arguments}
            required = set(specs).intersection(sessions._application.registry.get(SELECTION).required_arguments)
            if not required <= set(parameters): raise IntentError('unavailable_context')
            for key, value in parameters.items(): specs[key].validate(key, value)
            operations.append(dict(handle=f'op.{len(operations)}', source=asdict(locator), parameters=parameters))
        if len(operations) > 1 or len(revision.outputs) > 32:
            raise IntentError('unavailable_context')
        # Preserve an exact prior recipe pointer, not its stale matrix artifact or
        # a target to maintain. The pointer was captured by this revision's turn.
        origin = next((i for i in state.interactions if i.turn_id == revision.turn_id), None)
        focus = None
        if origin is not None and origin.admitted is not None:
            admitted = _serialize(origin.admitted)
            if matrix_source is None: matrix_source = admitted.get('matrix_source')
            focus = admitted.get('parameter')
        if matrix_source is not None: stored_step(sessions, matrix_source)
        bases[rid] = dict(revision_id=rid, outputs=[asdict(o) for o in revision.outputs],
                          operations=operations, matrix_source=matrix_source, focus=focus)
    return dict(relations=relations, bases=bases)


def public_context(captured):
    bases = {}
    for relation, rid in captured['relations'].items():
        base = captured['bases'][rid]
        bases[relation] = dict(operations=[dict(handle=o['handle'], tool=SELECTION,
            parameters=o['parameters']) for o in base['operations']],
            matrix_recipe_available=base['matrix_source'] is not None, parameter_focus=base['focus'])
    return dict(relations=tuple(bases) + ('previous',), bases=bases,
                previous_is_ambiguous=len(set(v for k,v in captured['relations'].items() if k != 'current')) > 1)
