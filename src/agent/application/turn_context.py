"""Metadata-only turn snapshots. Executable trust is checked later by M15.3."""
from dataclasses import asdict
from collections.abc import Mapping
import re
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from .session_state import OutputLocator
from .turn_decisions import IntentError



def supplied_input_context(inputs, registry):
    """Describe supplied Registry inputs without exposing values or arbitrary keys.

    Consumer descriptions are registered expectations, not observed scientific
    formats, validated resources, or execution readiness.
    """
    from agent.orchestration.semantic_prompt import _json_type
    values = freeze_json_mapping({} if inputs is None else inputs, 'execution_inputs')
    consumers = {}
    for tool_name in registry.names():
        tool = registry.get(tool_name)
        if tool.semantic_planning is None:
            continue
        for port in tool.semantic_planning.consumer_ports:
            fields = {m.name: m.field_name for m in port.members}
            for source in port.request_sources:
                for member in source.members:
                    name = member.input_name
                    if name not in values or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', name) is None:
                        continue
                    field = fields[member.name]
                    argument = tool.required_arguments.get(field, tool.optional_arguments.get(field))
                    planning = argument.planning
                    descriptor = dict(tool=tool_name, port=port.name, member=member.name,
                        lineage=None if source.lineage is None else source.lineage.value,
                        description='' if planning is None else planning.description,
                        expected_artifact_kinds=[] if planning is None else [k.value for k in planning.accepted_artifact_kinds],
                        scientific_parameter=False if planning is None else planning.scientific_parameter,
                        source_complete=all(m.input_name in values for m in source.members))
                    if descriptor not in consumers.setdefault(name, []):
                        consumers[name].append(descriptor)
    return dict(present=bool(values), fields=[dict(name=name, json_type=_json_type(values[name]),
        consumers=consumers[name]) for name in sorted(consumers)], omitted_count=len(values)-len(consumers))


def stored_step(sessions, locator):
    locator = OutputLocator(**locator) if isinstance(locator, dict) else locator
    sessions._validate_locator(locator)
    state = sessions._application.run_store.load(locator.run_id)
    return next(s for s in state.steps if s.step_id == locator.step_id)


def result_groups(captured, state):
    """Group only cooutputs with the same complete captured accepted identity."""
    revisions = {r.revision_id: r for r in state.revisions}
    grouped = {}
    for item in captured.get('dialogue', {}).get('outputs', ()):
        revision = revisions[item['revision_id']]
        locators = [o for o in revision.outputs if o.name == item['output_name']]
        if len(locators) != 1 or locators[0].accepted_step_sha256 != item['accepted_step_sha256']:
            raise IntentError('unavailable_context')
        locator = locators[0]
        key = (revision.revision_id, locator.run_id, locator.step_id, locator.accepted_step_sha256)
        grouped.setdefault(key, []).append((item, locator))
    groups = []
    for members in grouped.values():
        if len({item['tool'] for item, _ in members}) != 1:
            raise IntentError('unavailable_context')
        canonical, _ = min(members, key=lambda pair: (pair[1].name, pair[1].output_key))
        groups.append(dict(handle=f'g{len(groups)}', members=[item for item, _ in members], canonical=canonical))
    return groups


def result_descriptions(sessions, state, captured):
    """One metadata projection for interpretation and planning, never authority.

    Only accepted argument/result metadata is read. No native artifact is opened,
    no statistic is produced, and omitted historical parameters stay unknown.
    """
    from .dialogue_execution import scientific_parameters
    from .dialogue_evidence import _unsafe_fact_value
    from agent.schemas import StepOutputRef
    from agent.schemas.prior_output import PriorOutputRef
    parameters = scientific_parameters(sessions._application.registry)
    groups = result_groups(captured, state)
    locators, steps = {}, {}
    for group in groups:
        item = group['canonical']
        revision = next(r for r in state.revisions if r.revision_id == item['revision_id'])
        locator = next(o for o in revision.outputs if o.name == item['output_name'])
        locators[group['handle']] = locator
        steps[group['handle']] = stored_step(sessions, locator)
    result = []
    for group in groups:
        item, locator = group['canonical'], locators[group['handle']]
        step = steps[group['handle']]
        spec = sessions._application.registry.get(step.tool_name)
        values = []
        nested = step.result.get('parameters', {})
        if not isinstance(nested, Mapping):
            nested = {}
        for parameter in parameters:
            name = parameter['argument']
            if parameter['tool'] != step.tool_name:
                continue
            origin = 'accepted_argument'
            if name in step.resolved_arguments:
                value = _serialize(step.resolved_arguments[name])
            elif name in step.result:
                value, origin = _serialize(step.result[name]), 'owner_effective_value'
            elif name in nested:
                value, origin = _serialize(nested[name]), 'owner_effective_value'
            else:
                continue
            if name == 'kwargs' and spec.native_functions is not None:
                from .structured_parameters import recorded_native_kwargs
                value = recorded_native_kwargs(step)
                origin = 'owner_effective_value'
            argument = spec.required_arguments.get(name, spec.optional_arguments.get(name))
            try:
                argument.validate(name, value)
            except (ValueError, TypeError):
                continue
            if not _unsafe_fact_value(value):
                scope = ({'native_function': step.resolved_arguments['function_name']}
                         if spec.native_functions is not None and step.resolved_arguments.get('function_name') else {})
                values.append(dict(argument=name, value=value, origin=origin, **scope))
        run = sessions._application.run_store.load(locator.run_id)
        planned = next(s for s in run.plan.steps if s.step_id == locator.step_id)
        dependencies = {(locator.run_id, ref.step_id) for ref in planned.arguments.values()
                        if isinstance(ref, StepOutputRef)}
        dependencies.update((ref.binding.run_id, ref.binding.step_id) for ref in planned.arguments.values()
                            if isinstance(ref, PriorOutputRef))
        revision = next(r for r in state.revisions if r.revision_id == item['revision_id'])
        names = {m['output_name'] for m in group['members']}
        from .result_reading import handoff
        capability = handoff(sessions._application.run_store, locator.run_id, locator.step_id)
        result.append(dict(handle=group['handle'], outputs=[m['handle'] for m in group['members']],
            canonical_output=item['handle'], tool=step.tool_name, version=item['version'],
            relations=list(item['relations']), uses=['reference'],
            result_contract=spec.result_contract.name,
            cell_capabilities={} if capability is None else dict(
                coordinates=capability['coordinates'],
                grouping=None if capability['groups'] is None else capability['groups']['field']),
            available_outputs=sorted({o.output_key for o in revision.outputs if o.name in names}),
            parameters=values, derived_from=[g['handle'] for g in groups
                if (locators[g['handle']].run_id, locators[g['handle']].step_id) in dependencies]))
    return result


def revision_results(sessions, state, revision):
    """Project the existing W1 groups for one exact Revision, without recapture.

    The members remain accepted output locators. This live projection is neither
    another catalog nor a replacement for an Interaction's captured context.
    """
    tools = {}
    for output in revision.outputs:
        key = (output.run_id, output.step_id, output.accepted_step_sha256)
        if key not in tools:
            tools[key] = stored_step(sessions, output).tool_name
    captured = dict(dialogue=dict(outputs=[dict(handle=o.name, revision_id=revision.revision_id,
        output_name=o.name, accepted_step_sha256=o.accepted_step_sha256,
        tool=tools[o.run_id, o.step_id, o.accepted_step_sha256], version=state.revisions.index(revision) + 1,
        relations=['current'] if revision.revision_id == state.active_revision_id else [])
        for o in revision.outputs]))
    groups = result_groups(captured, state)
    descriptions = result_descriptions(sessions, state, captured)
    return tuple(dict(description, canonical=group['canonical'], members=group['members'])
                 for group, description in zip(groups, descriptions))


def snapshot(sessions, state, *, tool_names=None):
    revisions = {r.revision_id:r for r in state.revisions}
    current = revisions.get(state.active_revision_id)
    if current is None:
        if state.active_revision_id is not None: raise IntentError('unavailable_context')
        return dict(relations={}, bases={})
    relations = {'current':current.revision_id}
    if current.parent_revision_id is not None: relations['parent'] = current.parent_revision_id
    previous = next((e.from_revision_id for e in reversed(state.navigation)
                     if e.to_revision_id != e.from_revision_id), None)
    if previous is not None: relations['previous_active'] = previous
    bases = {}
    for rid in dict.fromkeys(relations.values()):
        revision = revisions[rid]
        if tool_names is not None:
            for locator in revision.outputs:
                tool_names[rid, locator.name] = stored_step(sessions, locator).tool_name
        bases[rid] = dict(revision_id=rid, outputs=[asdict(o) for o in revision.outputs])
    return dict(relations=relations, bases=bases)


def public_context(captured):
    relations = captured['relations']
    return dict(relations=tuple(relations) + ('previous',) if relations else ('current',),
                previous_is_ambiguous=len({v for k, v in relations.items() if k != 'current'}) > 1)
