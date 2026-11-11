"""Structured native decisions; shared scientific execution and accepted history.

This boundary has no interpreter, planner, prompt, model profile or provider.
"""
from pathlib import Path
from agent.schemas import AgentRequest, AgentPlan, PlanStep
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import digest
from agent.tools.analysis import native_callable as owner
from .session_state import OutputSelection
from agent.orchestration.run_store import RunNotFoundError
from .shared_resources import scientific_resource_scope


def catalog():
    return dict(packages=[dict(name=name, language='R' if name in owner.R_RUNTIMES else 'CLI' if name == 'MACS3' else 'Python') for name in owner.packages()])


def _accepted(app, session_id, revision_id):
    state = app._application.sessions.load(session_id)
    revision = next(r for r in state.revisions if r.revision_id == revision_id)
    output = next(o for o in revision.outputs if o.name == 'native_result')
    from .native_delivery import _publication
    origin = state.turn(revision.turn_id)
    data = _publication(app, revision, origin, output)
    manifest = data[7]
    if manifest['contract'] != owner.CONTRACT:
        raise ValueError('Selected result requires its own native input representation')
    return revision, manifest


def _descriptors(manifest):
    # Only bounded accepted previews enter the Web process. Full values remain
    # in result.json and are loaded by the next isolated scientific invocation.
    data = manifest['result']['preview']
    result = {'return': data['return_value']}
    result.update({'modified:' + k: v for k, v in data['modified_inputs'].items()})
    result.update({'file:' + str(i): dict(encoding='file', type='native_file', file=v['file'])
        for i, v in enumerate(data['produced_files'])})
    # Nested members retain their keys/order; select using stable structural paths.
    def walk(name, value):
        if value['encoding'] in ('tuple', 'list'):
            for i, v in enumerate(value.get('items', ())):
                result[name + '/' + str(i)] = v
                walk(name + '/' + str(i), v)
        elif value['encoding'] == 'dict':
            for i, entry in enumerate(value.get('items', ())):
                result[name + '/' + str(i)] = entry['value']
                walk(name + '/' + str(i), entry['value'])
    for name, value in tuple(result.items()): walk(name, value)
    return result


def bind(app, session_id, payload):
    """Resolve client identities to immutable input bytes before reserving science."""
    state = app._application.sessions.load(session_id)
    package, function = payload['package'], payload['function']
    with scientific_resource_scope(app.resources.library):
        metadata = owner.discover(package, function)
    if len(metadata['functions']) != 1: raise ValueError('Unknown installed public function')
    function_metadata = metadata['functions'][0]
    if function_metadata['limitation']: raise ValueError(function_metadata['limitation'])
    specs = {p['name']: p for p in function_metadata['parameters']}
    def file_parameter(key):
        annotation = specs[key]['annotation'].lower()
        return (specs[key].get('input_kind') == 'file' or 'path' in annotation or any(part in key.lower() for part in ('filename', 'directory', 'filepath', 'outdir', 'cache_dir', 'url'))
            or key.lower() in ('file', 'output', 'path'))
    parameters = payload.get('parameters', {})
    bindings = payload.get('inputs', {})
    outputs = payload.get('outputs', {})
    if set(parameters) & set(bindings) or set(outputs) & (set(parameters) | set(bindings)):
        raise ValueError('Each native argument needs exactly one explicit binding')
    supplied = set(parameters) | set(bindings) | set(outputs)
    if supplied - specs.keys(): raise ValueError('Unknown native parameter')
    missing = {k for k, p in specs.items() if p['required']} - supplied
    if missing: raise ValueError('Required native inputs: ' + ', '.join(sorted(missing)))
    for key in parameters:
        annotation = specs[key]['annotation'].lower()
        if file_parameter(key) or 'callable' in annotation:
            raise ValueError(key + ' requires a managed file/object binding; executable callbacks are unsupported')
        if key in ('save', 'cache_dir', 'backup_url') and isinstance(parameters[key], str):
            raise ValueError(key + ' requires a managed file binding')
        if package == 'MACS3' and isinstance(parameters[key], str) and ('/' in parameters[key] or '\\' in parameters[key]):
            raise ValueError('CLI file paths require managed file bindings')
    bound = {}
    for key, entry in bindings.items():
        adapter = entry['adapter']
        if adapter in ('value', 'array'):
            if file_parameter(key):
                raise ValueError(key + ' requires a managed file binding')
            if set(entry) != {'adapter', 'value'}: raise ValueError('Invalid inline object construction')
            bound[key] = dict(entry)
        elif adapter == 'accepted':
            if set(entry) != {'adapter', 'revision_id', 'output'}: raise ValueError('Invalid accepted output selection')
            revision, manifest = _accepted(app, session_id, entry['revision_id'])
            descriptor = _descriptors(manifest)[entry['output']]
            if descriptor.get('restorable') is False:
                raise ValueError('Rendered images cannot restore the original native object; use its downloadable visual artifact')
            if descriptor['encoding'] == 'r' and package not in owner.R_RUNTIMES:
                raise ValueError('The selected RDS object requires an installed R runtime')
            if file_parameter(key) and descriptor['encoding'] != 'file':
                raise ValueError(key + ' requires an accepted native-produced file')
            root = Path(manifest['arguments']['output_dir']) / 'objects'
            result_path = root.parent / 'result.json'
            result_pin = next(p for p in manifest['files'] if p['path'] == str(result_path))
            bound[key] = dict(adapter=adapter, descriptor={k: descriptor[k] for k in ('type', 'encoding')}, root=str(root),
                selection=entry['output'], result_path=str(result_path), result_sha256=result_pin['sha256'],
                files=[p for p in manifest['files'] if Path(p['path']).is_relative_to(root) or p == result_pin],
                source=dict(revision_id=revision.revision_id, run_id=revision.run_id, output=entry['output'],
                    manifest_sha256=digest(manifest)))
        else:
            if adapter not in ('anndata', 'dataframe', 'column', 'matrix', 'sparse', 'numpy', 'file', 'r_matrix'):
                raise ValueError('No input adapter for ' + adapter)
            allowed = {'adapter', 'resource_id', 'separator', 'column', 'format', 'options', 'orientation'}
            if set(entry) - allowed: raise ValueError('Unknown input construction option')
            if entry.get('orientation', 'unchanged') not in ('unchanged', 'transpose'):
                raise ValueError('Choose unchanged or transposed matrix orientation')
            record = app.resources.load(entry['resource_id'])
            if record.session_id != session_id: raise ValueError('Input belongs to another workspace Session')
            if entry.get('separator', ',') not in (',', '\t'): raise ValueError('Choose CSV or TSV explicitly')
            options = entry.get('options', {})
            if set(options) - {'header', 'index_col', 'dtype'}: raise ValueError('Unsupported table construction option')
            bound[key] = {k: v for k, v in entry.items() if k != 'resource_id'}
            bound[key].update(path=record.source_path, sha256=record.source_sha256,
                resource_id=record.resource_id)
    for key, name in outputs.items():
        if type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Choose a relative managed output filename')
    request_id = 'native-' + digest(dict(session=session_id, call=payload['call_id']))
    request = AgentRequest(request_id, None, dict(package=package, function_name=function,
        inputs=bound, parameters=parameters, outputs=outputs))
    _, run = app._application._prepare_request(request)
    args = dict(package=package, function_name=function, inputs=bound, parameters=parameters,
        outputs=outputs, output_dir=str(run.scientific / 'native'))
    plan = AgentPlan(request_id + ':native', request_id, 'structured-native',
        (PlanStep('native', owner.TOOL, args),))
    return request, plan


def execute(app, session_id, payload, prepared):
    request, plan = prepared
    with scientific_resource_scope(app.resources.library), app._application.sessions.processing_lease(session_id, payload['call_id']):
        app._application.sessions.run(session_id, payload['call_id'], request,
            (OutputSelection('native_result', 'native', 'manifest_path'),),
            expected_generation=payload['expected_generation'], prepared_plan=plan)
    return history(app, session_id)


def history(app, session_id):
    state = app._application.sessions.load(session_id)
    calls = []
    for turn in state.turns:
        if not turn.run_id: continue
        try: run = app._application.run_store.load(turn.run_id)
        except RunNotFoundError: continue
        if run.request.prompt is not None: continue
        row = dict(call_id=turn.turn_id, status=run.lifecycle_status.value.lower(), revision_id=turn.revision_id,
            function=run.request.inputs['package'] + '.' + run.request.inputs['function_name'],
            errors=[_serialize(e.to_dict()) for e in run.errors])
        if not turn.revision_id and run.lifecycle_status.value in ('PLANNING', 'VALIDATED', 'RUNNING', 'SUCCEEDED'):
            processing = app._application.sessions._store.is_processing(session_id, turn.turn_id)
            if not processing:
                row['status'] = 'awaiting_acceptance' if run.lifecycle_status.value == 'SUCCEEDED' else 'interrupted'
            elif run.lifecycle_status.value == 'SUCCEEDED':
                row['status'] = 'finalizing'
        if run.lifecycle_status.value == 'FAILED':
            failure = app._application._workspace.run_paths(turn.run_id).scientific / 'native' / 'failure.json'
            if failure.is_file():
                row['native_error'] = owner.read(failure)
        if turn.revision_id:
            revision, manifest = _accepted(app, session_id, turn.revision_id)
            row['preview'] = manifest['result']['preview']
            row['outputs'] = [dict(key=k, type=v['type'], encoding=v['encoding'], shape=v.get('shape'))
                for k, v in _descriptors(manifest).items() if v.get('restorable') is not False
                and not (v['encoding'] == 'json' and v['value'] is None)]
            from .native_delivery import accepted_artifacts
            row['downloads'] = [a.reference.to_dict() for a in accepted_artifacts(app, session_id, revision, turn, revision.outputs[0])]
        calls.append(row)
    return dict(session_id=session_id, generation=state.generation, calls=calls)
