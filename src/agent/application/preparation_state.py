"""Immutable scientific preparation records in existing Interaction history.

This module owns identities and answer reduction, not scientific qualification.
Reopening history validates recorded shapes/digests without consulting providers,
files, today's Registry or native science. Live reanalysis owns those checks.
"""
from collections.abc import Mapping
from dataclasses import replace

from agent.schemas.orchestration import _serialize
from agent.orchestration.semantic_compiler import candidate_record, candidate_from_record, candidate_identity
from .session_state import SessionError, SessionConflictError, digest

VERSION = 2


def origin_record(interactions, record):
    origin = next((i for i in interactions if i.turn_id == record['task_origin']), None)
    if origin is None or origin.preparation is None or 'task' not in origin.preparation:
        raise SessionError('Preparation task origin is unavailable.')
    return origin.preparation


def candidate_for(interactions, record):
    if record['candidate_origin'] is None:
        return None
    origin = next((i for i in interactions if i.turn_id == record['candidate_origin']), None)
    if origin is None or origin.preparation is None or 'candidate' not in origin.preparation:
        raise SessionError('Canonical candidate is unavailable.')
    return candidate_from_record(origin.preparation['candidate'])


def head(interactions, task_id):
    return next((i for i in reversed(interactions) if i.preparation is not None
                 and i.preparation['task_id'] == task_id), None)


def start(session_id, interaction, objective, *, candidate=None, contract_identity=None, sources=None, context=None):
    task_id = digest(dict(session=session_id, origin=interaction.turn_id))
    record = dict(version=VERSION, task_id=task_id, task_origin=interaction.turn_id,
        candidate_origin=None if candidate is None else interaction.turn_id,
        candidate_id=None if candidate is None else candidate_identity(candidate),
        previous=None, answers=[], task=dict(objective=objective, submission_turn_id=interaction.turn_id,
            sources=_serialize(sources or {}), context=_serialize(context or {}),
            base_generation=interaction.base_generation, base_revision_id=interaction.base_revision_id))
    if candidate is not None:
        record.update(candidate=candidate_record(candidate), contract_identity=contract_identity)
    return record


def question(record, interaction, requirements, *, blocked=(), status='NEEDS_INPUT', diagnostics=()):
    rows = [_serialize(r.to_dict() if hasattr(r, 'to_dict') else r) for r in requirements]
    basis = dict(task_id=record['task_id'], candidate_id=record['candidate_id'],
        binding_head=interaction.turn_id, base_generation=interaction.base_generation,
        base_revision_id=interaction.base_revision_id, requirements=rows)
    return dict(version=record['version'], status=status, requirements=rows, blocked_checks=_serialize(blocked),
                diagnostics=_serialize(diagnostics), question_version=digest(basis))


def observation(interaction):
    return (interaction.prerequisite or {}).get('preparation_question') or (interaction.admitted or {}).get('preparation_result')


def choices(interaction):
    record, result = interaction.preparation, observation(interaction)
    if record is None or result is None:
        return ()
    if result['status']=='READY':
        return (dict(handle='spp.'+result['question_version']+'.execute',task_id=record['task_id'],
            question_version=result['question_version'],action='execute_preparation',label='Run prepared analysis'),)
    if result['status']!='NEEDS_INPUT':
        return ()
    options = []
    for req in result['requirements']:
        token = 'spp.' + result['question_version'] + '.' + req['requirement_id']
        identity = dict(handle=token, task_id=record['task_id'], candidate_id=record['candidate_id'],
            requirement_id=req['requirement_id'], question_version=result['question_version'],
            step_id=req['step_id'], tool=req['tool_name'], port=req['port'], argument=req['argument'],
            label=req['step_id'] + ' · ' + req['argument'])
        if req.get('native_function') is not None:
            identity['native_function'] = req['native_function']
        if req.get('member') is not None:
            identity['member'] = req['member']
        if req['kind']=='external_input' or req.get('applicability',{}).get('source_input_types'):
            options.append(dict(identity, handle=token+'.upload', action='upload_source', label='Upload '+req['argument']))
        if req['kind'] == 'scalar' and not req.get('choices'):
            options.append(dict(identity, action='parameter'))
        offers = req.get('offers', ())
        if req['kind'] in ('scope', 'scalar') and req.get('choices'):
            offers = [dict(handle=str(value), label=str(value), value=value) for value in req['choices']]
        for index, option in enumerate(offers):
            public = {key: value for key, value in option.items() if key not in ({'component','source','role'} if req['kind'] == 'input_role' else {'component'})}
            options.append({**identity, **public, 'choice_identity': option['handle'], 'handle': token + '.' + str(index)})
    return tuple(options)


def validate_exact_submission(interactions, interaction, previous):
    """Replay received input identity as well as the exact offered answer."""
    received = interaction.submission or {}
    admitted = previous.admitted or {}
    record = previous.preparation
    if any(key not in admitted.get('inputs', {}) or _serialize(value) != _serialize(admitted['inputs'][key])
           for key, value in received.get('execution_inputs', {}).items()):
        raise SessionConflictError('Exact answer changed the captured scientific inputs.')
    if received.get('registered_input') is not None:
        original = next(i for i in interactions if i.turn_id == record['task_origin'])
        identities = [admitted.get('registered_input'), (original.submission or {}).get('registered_input')]
        offered = previous.snapshot.get('dialogue', {}).get('source_view', {}).get('registered_input')
        if (offered is not None and offered.get('composition') == 'scientific-role-selection.v1'
                and offered.get('attachment_candidates') is True and _serialize(offered.get('roles')) == []
                and offered.get('resource_choice') is None):
            # The Web may retransmit its still-staged catalog on an exact click.
            # Only this question's complete neutral offer identity is reusable;
            # the selected Answer and source owner still bind the actual input.
            identities.append(offered)
        if not any(_serialize(received['registered_input']) == _serialize(value) for value in identities):
            raise SessionConflictError('Exact answer changed the captured source selection.')


def answer_question(interactions, interaction, *, values, interpreted=False):
    """Reduce one exact offered question, with no interpretation or planning.

    Caller supplies already typed values indexed by requirement identity. A
    question is consumed once even if another Interaction uses its token.
    """
    selection = (interaction.submission or {}).get('clarification_choice')
    if interpreted:
        _, previous = intake_context(interaction, interactions)
    else:
        if selection is None:
            raise SessionConflictError('An exact preparation question is required.')
        previous = next((i for i in interactions if i.turn_id == selection['question_turn_id']), None)
    if previous is None or previous.preparation is None:
        raise SessionConflictError('Unknown scientific preparation question.')
    record, result = previous.preparation, observation(previous)
    if (head(interactions, record['task_id']) != previous or result is None
            or result['status'] != 'NEEDS_INPUT' or previous.status != 'clarification'
            or previous.base_generation != interaction.base_generation
            or previous.base_revision_id != interaction.base_revision_id):
        raise SessionConflictError('Scientific preparation question is stale or consumed.')
    if not all(r['kind'] == 'input_resource' for r in result['requirements']):
        validate_exact_submission(interactions, interaction, previous)
    offered = {row['handle']: row for row in choices(previous)}
    if interpreted:
        matches = [(r, option) for r in result['requirements'] if r['kind'] in ('input_resource', 'resource')
                   for option in r.get('offers', ()) if values == {r['requirement_id']: _serialize(option)}]
        if len(matches) != 1:
            raise SessionConflictError('Interpreted resource was not exactly offered.')
        req, option = matches[0]
        selected = next(c for c in offered.values() if c['requirement_id'] == req['requirement_id']
                        and c['choice_identity'] == option['handle'])
    else:
        selected = offered.get(selection['handle'])
    if selected is None:
        raise SessionConflictError('Preparation question version or choice changed.')
    requirements = {r['requirement_id']: r for r in result['requirements']}
    if set(values) != {selected['requirement_id']}:
        raise SessionConflictError('Answer does not address this preparation question.')
    selected_requirement = requirements[selected['requirement_id']]
    if selected_requirement['kind'] in ('resource','scope','input_role','input_resource','declaration') or selected_requirement.get('choices'):
        index = int(selected['handle'].rsplit('.', 1)[1])
        expected = (selected_requirement['choices'][index] if selected_requirement['kind'] in ('scope', 'scalar')
                    else selected_requirement['offers'][index])
        if set(values) != {selected['requirement_id']} or _serialize(values[selected['requirement_id']]) != _serialize(expected):
            raise SessionConflictError('Answer changed the exact offered identity.')
    return record_answers(previous, interaction, values, provenance='interpreted_resource' if interpreted else 'exact_submission')


def record_answers(previous, interaction, values, *, provenance):
    record, result = previous.preparation, observation(previous)
    requirements = {r['requirement_id']:r for r in result['requirements']}
    if set(values) - set(requirements):
        raise SessionError('Answer is outside the captured scientific question.')
    answers = [dict(requirement_id=key, value=_serialize(value), answering_interaction=interaction.turn_id,
        question_version=result['question_version'], question_turn_id=previous.turn_id,
        step_id=requirements[key]['step_id'], argument=requirements[key]['argument'],
        member=requirements[key].get('member'), kind=requirements[key]['kind'],
        provenance=provenance) for key, value in values.items()]
    return dict(version=VERSION, task_id=record['task_id'], task_origin=record['task_origin'],
        candidate_origin=record['candidate_origin'], candidate_id=record['candidate_id'],
        previous=previous.turn_id, answers=answers)


def continued_admission(interactions, record, request_id):
    """Replay exact scalar/scope admission from its immutable question owner.

    A scope click changes a consumer identity, never the original declaration's
    value or grounding text. A resource choice is replayed separately by its exact
    operation input owner; it cannot rewrite the admitted source or declarations.
    """
    previous = next(i for i in interactions if i.turn_id == record['previous'])
    admitted = {k: _serialize(v) for k, v in previous.admitted.items()
                if k not in {'preparation_result', 'native_resource_bindings', 'choices', 'resource_options'}}
    requirements = {r['requirement_id']: r for r in observation(previous)['requirements']}
    for answer in record['answers']:
        if answer['provenance'] not in ('exact_submission', 'interpreted_resource', 'interpreted_parameter') or answer['kind'] not in ('scope', 'scalar', 'resource'):
            raise SessionError('This answer requires its scientific input owner.')
        if answer['kind'] != 'scope':
            continue
        req = requirements[answer['requirement_id']]
        if req['step_id'].startswith('@request:'):
            index = int(req['step_id'].split(':')[1])
            row = admitted['requested_parameters'][index]
            if isinstance(row, str):
                row = dict(tool=admitted['tool'], argument=row)
                admitted['requested_parameters'][index] = row
            rows = [row]
        else:
            rows = [r for r in admitted.get('arguments', ()) if r['tool'] == req['tool_name']
                and r['argument'] == req['argument'] and r.get('native_function') == req['native_function']]
        if len(rows) != 1:
            raise SessionError('Scope answer requires one originating declaration.')
        rows[0]['step_id'] = answer['value']
    from .source_observations import apply_source_choices
    from agent.orchestration.registry import build_default_tool_registry
    admitted = apply_source_choices(admitted, record['answers'], requirements, build_default_tool_registry(),
        captured_group=previous.snapshot.get('dialogue', {}).get('source_view', {}).get('registered_input'))
    admitted['request_id'] = request_id
    return admitted


def argument_origin(interactions, interaction):
    """Find the text that actually admitted a declaration, not an empty click."""
    by_id = {i.turn_id: i for i in interactions}
    while interaction is not None and interaction.preparation is not None:
        record = interaction.preparation
        if not record['previous'] or ('resume' not in record and 'authorization' not in record and (not record['answers'] or any(
                a['provenance'] not in ('exact_submission', 'interpreted_resource', 'interpreted_parameter') or a['kind'] not in ('scope', 'scalar', 'resource')
                for a in record['answers']))):
            break
        interaction = by_id[record['previous']]
    return interaction


def scalar_bindings(interactions, record):
    """Replay only the selected task's exact predecessor chain."""
    bound = {}
    for answer, _ in active_answers(interactions, record):
        if answer['kind'] == 'resource' and answer['value'].get('argument') == answer['argument'] and 'value' in answer['value']:
            bound.setdefault(answer['step_id'], {})[answer['argument']] = _serialize(answer['value']['value'])
            continue
        if answer['kind'] != 'scalar':
            continue
        values = bound.setdefault(answer['step_id'], {})
        if answer['member'] is None:
            values[answer['argument']] = _serialize(answer['value'])
        else:
            values.setdefault(answer['argument'], {})[answer['member']] = _serialize(answer['value'])
    return bound


def validate_record(session_id, interaction, preceding):
    """Versioned historical decoding. Never requalifies scientific bindings."""
    record = interaction.preparation
    if 'preparation' in interaction.snapshot and not (interaction.submission or {}).get('resume_task_id'):
        intake_context(interaction, preceding, require_current=False)
    if record is None:
        return
    required = {'version','task_id','task_origin','candidate_origin','candidate_id','previous','answers'}
    if record.get('version') not in (1, VERSION) or set(record) - required - ({'task','candidate','contract_identity','source_pins','declarations','resume','authorization'} if record.get('version') == 2 else {'task','candidate','contract_identity'}) or not required <= set(record):
        raise SessionError('Invalid scientific preparation record.')
    if 'task' in record:
        if (record['task_origin'] != interaction.turn_id or record['previous'] is not None
                or record['task_id'] != digest(dict(session=session_id, origin=interaction.turn_id))
                or not isinstance(record['task'].get('objective'), str) or not record['task']['objective'].strip()):
            raise SessionError('Invalid preparation task origin.')
    else:
        original = origin_record(preceding, record)
        if original['task_id'] != record['task_id']:
            raise SessionError('Preparation changed task identity.')
    if 'candidate' in record:
        if (record['candidate_origin'] != interaction.turn_id
                or candidate_identity(candidate_from_record(record['candidate'])) != record['candidate_id']
                or not isinstance(record.get('contract_identity'), str)):
            raise SessionError('Canonical candidate identity changed.')
    elif record['candidate_origin'] is not None:
        if candidate_identity(candidate_for(preceding, record)) != record['candidate_id']:
            raise SessionError('Candidate reference changed.')
    elif record['candidate_id'] is not None:
        raise SessionError('Candidate has no canonical origin.')
    if 'source_pins' in record:
        import re
        pins = record['source_pins']
        if ('candidate' not in record or not isinstance(pins, (tuple, list))
                or any(set(pin) not in ({'source_path'}, {'source_path','source_sha256'})
                    or not isinstance(pin['source_path'], str) or not pin['source_path']
                    or ('source_sha256' in pin and (not isinstance(pin['source_sha256'], str)
                        or re.fullmatch('[0-9a-f]{64}', pin['source_sha256']) is None)) for pin in pins)
                or len({pin['source_path'] for pin in pins}) != len(pins)):
            raise SessionError('Invalid candidate source byte identities.')
    if record['previous'] is not None:
        previous = head(preceding, record['task_id'])
        if previous is None or previous.turn_id != record['previous']:
            raise SessionError('Preparation binding head changed.')
        if record['version'] < previous.preparation['version']:
            raise SessionError('Preparation record version cannot be downgraded.')
        if previous.preparation['candidate_origin'] is not None and record['version'] >= 2:
            if ('candidate' in record or record['candidate_origin'] != previous.preparation['candidate_origin']
                    or record['candidate_id'] != previous.preparation['candidate_id']):
                raise SessionError('Preparation cannot replace its immutable candidate.')
        if 'authorization' in record:
            result=observation(previous)
            selection=(interaction.submission or {}).get('clarification_choice') or {}
            if (record['answers'] or result is None or result['status']!='READY' or previous.status!='answered'
                    or record['authorization']!=dict(question_turn_id=previous.turn_id,question_version=result['question_version'])
                    or selection!=dict(question_turn_id=previous.turn_id,handle='spp.'+result['question_version']+'.execute')):
                raise SessionError('Prepared execution requires the exact current READY authorization.')
        if 'resume' in record:
            if (record['version'] < 2 or record['answers'] or record['resume'] != dict(
                    base_generation=previous.base_generation, base_revision_id=previous.base_revision_id)
                    or (interaction.submission or {}).get('resume_task_id') != record['task_id']):
                raise SessionError('Preparation resume has no exact task authority.')
        elif (previous.base_generation != interaction.base_generation
                or previous.base_revision_id != interaction.base_revision_id):
            raise SessionError('Preparation answer changed the captured context.')
        if record['version'] >= 2 and any(a['provenance'] == 'exact_submission' and a['kind'] not in {'input_resource', 'declaration'} for a in record['answers']):
            validate_exact_submission(preceding, interaction, previous)
        answer_ids = [a['requirement_id'] for a in record['answers']]
        if len(set(answer_ids)) != len(answer_ids):
            raise SessionError('Duplicate preparation answers.')
        for answer in record['answers']:
            if answer['provenance'] not in (('exact_submission', 'interpreted_declaration', 'interpreted_role', 'interpreted_resource', 'interpreted_source', 'interpreted_parameter') if record['version'] >= 2 else ('exact_submission', 'interpreted_declaration')):
                raise SessionError('Unknown preparation answer provenance.')
            if answer['answering_interaction'] != interaction.turn_id:
                raise SessionError('Answering Interaction changed.')
            result = observation(previous)
            if result is None or answer['question_version'] != result['question_version']:
                raise SessionError('Preparation answer question changed.')
            matches = [r for r in result['requirements'] if r['requirement_id'] == answer['requirement_id']]
            if len(matches) != 1 or any(answer[k] != matches[0].get(k) for k in ('step_id','argument','member','kind')):
                raise SessionError('Preparation answer scope changed.')
            if answer['question_turn_id'] != previous.turn_id:
                raise SessionError('Answer question identity changed.')
            if answer['provenance'] == 'interpreted_parameter':
                if (answer['kind'] != 'scalar' or interaction.snapshot.get('preparation', {}).get('binding_turn_id') != previous.turn_id
                        or parameter_value(interaction, matches[0], answer['grounding']) != _serialize(answer['value'])):
                    raise SessionError('Interpreted parameter changed its captured value or scope.')
            if answer['provenance'] == 'interpreted_source':
                if (answer['kind'] != 'declaration' or matches[0]['step_id'] != '@task'
                        or interaction.snapshot.get('preparation', {}).get('binding_turn_id') != previous.turn_id):
                    raise SessionError('Source declaration has no captured task authority.')
            if answer['provenance'] == 'interpreted_resource':
                if (answer['kind'] not in ('input_resource', 'resource') or answer['value'] not in matches[0].get('offers', ())
                        or interaction.snapshot.get('preparation', {}).get('binding_turn_id') != previous.turn_id):
                    raise SessionError('Interpreted resource changed its captured offer.')
            if answer['provenance'] == 'interpreted_role':
                value, requirement = answer['value'], matches[0]
                role = value.get('role', {})
                if (record['version'] < 2 or answer['kind'] != 'input_role'
                        or set(value) != {'handle','label','role','source'}
                        or role.get('tool') != requirement['tool_name']
                        or role.get('port') != requirement['port']
                        or role.get('source') != value['handle']
                        or value['label'] != value['source'].get('label')):
                    raise SessionError('Interpreted role answer changed its exact consumer or source.')
            if answer['provenance'] == 'exact_submission':
                selection = (interaction.submission or {}).get('clarification_choice') or {}
                valid = [c for c in choices(previous) if c['handle'] == selection.get('handle')]
                if selection.get('question_turn_id') != previous.turn_id or not valid:
                    raise SessionError('Answer has no exact submitted question token.')
                if answer['kind'] in ('resource','scope','input_role','input_resource','declaration') or matches[0].get('choices'):
                    index = int(selection['handle'].rsplit('.', 1)[1])
                    expected = (matches[0]['choices'][index] if answer['kind'] in ('scope', 'scalar') else matches[0]['offers'][index])
                    if (valid[0]['requirement_id'] != answer['requirement_id']
                            or _serialize(expected) != _serialize(answer['value'])):
                        raise SessionError('Answer changed the selected offered identity.')
                if answer['kind'] == 'scalar' and not matches[0].get('choices'):
                    settings = [r for r in (interaction.submission or {}).get('parameter_overrides', ())
                        if r.get('step_id') == answer['step_id'] and r.get('argument') == answer['argument']
                        and r.get('tool') == matches[0]['tool_name']
                        and (record['version'] == 1 or r.get('native_function') == matches[0]['native_function'])]
                    if len(settings) != 1:
                        raise SessionError('Scalar answer has no exact submitted parameter.')
                    supplied = settings[0]['value']
                    if answer['member'] is not None:
                        supplied = supplied[answer['member']]
                    if _serialize(supplied) != _serialize(answer['value']):
                        raise SessionError('Scalar answer differs from its submitted value.')
    elif record['answers']:
        raise SessionError('Preparation answer has no predecessor.')
    if record['version'] >= 2:
        if any(a['provenance'] == 'interpreted_declaration' for a in record['answers']):
            raise SessionError('Source declarations cannot satisfy resource requirements.')
        if 'declarations' in record and interaction.admitted is not None:
            previous = next(i for i in preceding if i.turn_id == record['previous'])
            field = (interaction.admitted.get('continuation') or {}).get('field') or intake_task(preceding, record)
            if field is None or _serialize(record['declarations']) != declaration_changes(
                    previous, interaction, interaction.admitted, field, origin_record(preceding, record)['task']['sources']):
                raise SessionError('Preparation declaration provenance changed.')
    result = observation(interaction)
    if result is not None:
        if result['status'] not in ('READY','NEEDS_INPUT','INVALID'):
            raise SessionError('Invalid preparation disposition.')
        expected = question(record, interaction, result['requirements'], blocked=result['blocked_checks'],
                            status=result['status'], diagnostics=result['diagnostics'])
        if _serialize(result) != expected:
            raise SessionError('Scientific preparation question changed.')
        if result['status'] == 'READY' and result['requirements']:
            raise SessionError('Unresolved requirements cannot be READY.')


from agent.orchestration.preparation_dependencies import operation_sources, operation_dependencies


def active_answers(interactions, record):
    """Reduce answers and source-scoped declaration deltas along one task ledger.

    Requirements carry dependencies from their scientific owner and source
    operands from the compiler graph. This is a projection of that graph, not
    another plan or lifecycle. Invalidated historical answers remain immutable.
    """
    chain, by_id, seen = [record], {i.turn_id:i for i in interactions}, set()
    while chain[-1]['previous'] is not None:
        ref = chain[-1]['previous']
        if ref in seen or ref not in by_id:
            raise SessionError('Broken preparation answer chain.')
        seen.add(ref)
        chain.append(by_id[ref].preparation)
    active = {}
    for row in reversed(chain):
        for change in row.get('declarations', ()):
            paths = source_identities(change['scope']['sources'])
            def affected(requirement):
                sources = set(requirement.get('source_dependencies', ()))
                if not sources and isinstance(requirement.get('applicability', {}).get('source'), str):
                    sources.add(requirement['applicability']['source'])
                # Pre-migration questions have no graph projection. A matching
                # declaration dependency must be reasked, never blindly replayed.
                source_match = (bool(paths & sources) if 'source_dependencies' in requirement
                                else not sources or bool(paths & sources))
                dependencies = requirement.get('dependencies', ())
                member = change['member']
                declared = any(d.endswith(':declaration:' + member) for d in dependencies)
                return source_match and (declared or member in {'source','members','resource','resource_preference'})
            active = {key:value for key,value in active.items() if not affected(value[1])}
        for answer in row['answers']:
            question_turn = by_id[answer['question_turn_id']]
            requirement = next(r for r in observation(question_turn)['requirements']
                               if r['requirement_id'] == answer['requirement_id'])
            if answer['kind'] == 'resource' and requirement['port'] == 'reference':
                paths = set(requirement.get('source_dependencies', ()))
                active = {key: value for key, value in active.items()
                    if not (any(d.endswith(':reference:fasta') for d in value[1].get('dependencies', ()))
                            and (answer['step_id'] in value[1].get('source_steps', ())
                                if 'source_steps' in value[1] else not paths or paths & set(value[1].get('source_dependencies', ()))))}
            active[(answer['step_id'], answer['argument'], answer.get('member'))] = (answer, requirement)
    return tuple(active.values())


def resource_bindings(interactions, record):
    result = {}
    for answer, _ in active_answers(interactions, record):
        if answer['kind'] == 'resource' and answer['provenance'] in ('exact_submission', 'interpreted_resource'):
            result.setdefault(answer['step_id'], {})[answer['argument']] = _serialize(answer['value'])
    return result


def intake_task(interactions, record):
    return origin_record(interactions, record)['task'].get('context', {}).get('intake_owner')


def capture_intake(interactions, generation, revision_id, submission, predecessor=None):
    """Capture a canonical input question, never an executable task selection."""
    if predecessor is not None or not interactions:
        return None
    selection = (submission or {}).get('clarification_choice') or {}
    prior = (next((i for i in interactions if i.turn_id == selection.get('question_turn_id')), None)
             if selection else interactions[-1])
    if (prior is not None and (prior.status == 'failed' or (prior.admitted or {}).get('intent') == 'guidance') and prior.preparation is None
            and (prior.snapshot.get('preparation') or {}).get('binding_turn_id')):
        reference = prior.snapshot['preparation']
        prior = next((i for i in interactions if i.turn_id == reference['binding_turn_id']), None)
    if (prior is None or prior.preparation is None or prior.status != 'clarification'
            or prior.base_generation != generation or prior.base_revision_id != revision_id
            or head(interactions, prior.preparation['task_id']) != prior
):
        return None
    return dict(task_id=prior.preparation['task_id'], binding_turn_id=prior.turn_id,
                question_version=observation(prior)['question_version'])


def intake_context(interaction, preceding, *, require_current=True):
    ref = interaction.snapshot.get('preparation')
    if ref is None:
        raise SessionConflictError('No captured preparation task.')
    previous = next((i for i in preceding if i.turn_id == ref['binding_turn_id']), None)
    if (previous is None or previous.preparation is None
            or ref != dict(task_id=previous.preparation['task_id'], binding_turn_id=previous.turn_id,
                           question_version=observation(previous)['question_version'])
            or require_current and head(preceding, ref['task_id']) != previous
            or previous.base_generation != interaction.base_generation
            or previous.base_revision_id != interaction.base_revision_id):
        raise SessionConflictError('Preparation intake question is stale or changed.')
    original = next(i for i in preceding if i.turn_id == previous.preparation['task_origin'])
    return original, previous


def declaration_members(admitted, field, captured_source, registry=None):
    """Resolve the intake's source through its actual registered consumer.

    A source omitted from a declaration may already be selected by a role. The
    attachment list is only an identity catalog, never selection authority.
    """
    declaration = admitted.get(field, {})
    captured = admitted.get(field + '_sources', {})
    handles = ([row['source'] for row in declaration.get('members', ())]
               if field == 'fastq' else [declaration.get('source')])
    if handles == [None] and admitted.get('roles'):
        if registry is None:
            from agent.orchestration.registry import build_default_tool_registry
            registry = build_default_tool_registry()
        from agent.orchestration.registry import ArtifactSemanticKind
        tool = registry.get(admitted['tool'])
        ports = {port.name for port in tool.semantic_planning.consumer_ports
            if any(member.field_name in tool.required_arguments
                and tool.required_arguments[member.field_name].planning is not None
                and ArtifactSemanticKind.RAW_SCATAC in
                    tool.required_arguments[member.field_name].planning.accepted_artifact_kinds
                for member in port.members)}
        handles = sorted({row['source'] for row in admitted['roles']
                          if row['tool'] == tool.name and row['port'] in ports})
        if len(handles) > 1:
            raise SessionConflictError('Declaration requires an exact consumer source selection.')
    members = []
    for handle in handles:
        if handle is None:
            continue
        index = int(handle.split('.')[1])
        catalog = captured.get('members', ())
        if index >= len(catalog):
            raise SessionError('Declaration source is outside its captured members.')
        member = catalog[index]
        # The source owner validates the immutable prefix and explicitly
        # authorized appends. Initial unselected offers are not an admission
        # gate for a file supplied later to answer the same task's prerequisite.
        members.append(member)
    return tuple(members)


def declaration_source(admitted, field, captured_source):
    """Identify only the declared original source, never another offered file."""
    pins = [{key: _serialize(member[key]) for key in
             ('resource_id', 'record_sha256', 'source_path', 'source_sha256') if key in member}
            for member in declaration_members(admitted, field, captured_source)]
    if not pins and captured_source.get('accepted_results'):
        pins = [dict(accepted_binding=_serialize(binding)) for binding in captured_source['accepted_results']]
    if not pins:
        # Accepted-result declarations retain their exact selected result/context
        # identities; an absent source is not an invented input identity.
        pins = [_serialize(admitted.get('annotation_source_evidence', {}))] if field == 'annotation' else []
    if (not pins or any(not pin for pin in pins)) and captured_source.get('source_path') == admitted.get('inputs', {}).get('input_path') and captured_source.get('source_sha256'):
        pins = [_serialize(captured_source)]
    if not pins or any(not pin for pin in pins):
        raise SessionConflictError('Declaration correction has no exact captured scientific source identity.')
    return dict(owner=field, sources=pins)


def declaration_changes(previous, interaction, admitted, field, captured_source):
    before = previous.admitted or {}
    if field not in before or field not in admitted:
        raise SessionError('Declaration has no captured scientific owner.')
    scope = declaration_source(before, field, captured_source)
    if scope != declaration_source(admitted, field, captured_source):
        raise SessionConflictError('Declaration correction changed its scientific source identity.')
    old, new = _serialize(before[field]), _serialize(admitted[field])
    evidence = dict(interaction_id=interaction.turn_id, utterance_sha256=digest(interaction.utterance),
                    addition=_serialize(admitted.get(field + '_addition', {})),
                    unknown_fields=_serialize(admitted.get('unknown_fields', ())))
    return [dict(scope=scope, member=member, old_value=old.get(member), value=new.get(member), evidence=evidence)
            for member in sorted(set(old) | set(new)) if old.get(member) != new.get(member)]


def captured_owner(interaction, preceding):
    """Resolve a captured canonical task's scientific intake owner, without a pending projection."""
    if 'preparation' not in interaction.snapshot:
        return None
    _, previous = intake_context(interaction, preceding)
    return intake_task(preceding, previous.preparation)


def source_context(interaction, preceding):
    """Canonical ancestry for scientific declaration replay, including fresh tasks."""
    record = interaction.preparation
    if record is not None:
        if record['previous'] is None:
            return interaction, None
        return (next(i for i in preceding if i.turn_id == record['task_origin']),
                next(i for i in preceding if i.turn_id == record['previous']))
    return intake_context(interaction, preceding)


def public_question(interaction, preceding):
    """Transient projection of the one captured preparation observation."""
    if 'preparation' not in interaction.snapshot:
        return None
    origin, previous = intake_context(interaction, preceding)
    field = intake_task(preceding, previous.preparation)
    result = dict(handle='@roles' if field in {'roles','fastq','bam','fragments','annotation','epizoo'} else '@resources',
        field=field or 'preparation', tool=previous.admitted.get('tool'),
        objective=origin.preparation['task']['objective'], task_id=previous.preparation['task_id'],
        missing=[m for r in observation(previous)['requirements'] for m in r.get('messages', ())],
        choices=[dict(handle=c['choice_identity'], label=c['label']) for c in choices(previous) if 'choice_identity' in c],
        resource_options=[_serialize(c) for c in choices(previous)])
    if field == 'roles':
        result.update(required_roles=_serialize(origin.admitted['required_roles']), roles=_serialize(previous.admitted['roles']))
    elif field in {'fastq','bam','fragments','annotation','epizoo'}:
        result['declarations'] = _serialize(previous.admitted[field])
    requirements = observation(previous)['requirements']
    resource_requirements = {r['requirement_id'] for r in requirements if r['kind'] in ('resource', 'input_resource')}
    result['resource_answer_choices'] = [dict(handle=c['handle'], label=c['label'])
        for c in choices(previous) if c['requirement_id'] in resource_requirements and 'choice_identity' in c]
    if previous.preparation['candidate_origin'] is not None and requirements and all(r['kind']=='scalar' for r in requirements):
        result.update(handle='@parameters', field='parameters',
            missing=[{k:r[k] for k in ('step_id','tool_name','native_function','argument','member')} for r in requirements])
    return result


def parameter_value(interaction, requirement, grounding, registry=None):
    """Replay grounded text using the existing scientific parameter owner."""
    from .dialogue_execution import _bind_arguments
    from .turn_decisions import _scientific_arguments, ExplicitArgument
    if registry is None:
        from agent.orchestration.registry import build_default_tool_registry
        registry = build_default_tool_registry()
    declaration, = _scientific_arguments([_serialize(grounding)])
    if (declaration.tool != requirement['tool_name'] or declaration.argument != requirement['argument']
            or getattr(declaration, 'native_function', None) != requirement['native_function']
            or getattr(declaration, 'step_id', requirement['step_id']) != requirement['step_id']):
        raise SessionError('Parameter declaration changed its consumer scope.')
    if isinstance(declaration, ExplicitArgument) and _serialize(grounding) not in _serialize((interaction.submission or {}).get('parameter_overrides', ())):
        raise SessionError('Structured parameter is not in the received submission.')
    values = _bind_arguments(registry, interaction.utterance, {}, (declaration,))
    if len(values) != 1:
        raise SessionError('Parameter declaration has no single registered operand.')
    value, = values.values()
    if requirement.get('member') is not None:
        value = value[requirement['member']]
    if requirement.get('choices') and value not in requirement['choices']:
        raise SessionError('Parameter value is outside the captured choices.')
    return _serialize(value)


def source_identities(pins):
    """Exact source paths and accepted atomic-output anchors, never display names."""
    result = {pin['source_path'] for pin in pins if pin.get('source_path')}
    from agent.orchestration.preparation_dependencies import accepted_source_identity
    result.update(accepted_source_identity(pin['accepted_binding']) for pin in pins if pin.get('accepted_binding'))
    return result
