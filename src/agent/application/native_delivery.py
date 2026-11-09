"""Registry-declared native files use the existing accepted/private snapshot path."""
import csv
import io
from pathlib import Path
from agent.schemas.orchestration import _serialize

PRIVATE_LABEL = '[private label omitted; use ordered index]'
CELL_MATRIX_LABELS = ('UMAP coordinates and retained matrix (H5AD)',
    'Leiden groups and retained matrix (H5AD)', 'Region fragment counts (H5AD)',
    'Peak counts (H5AD)', 'Binary accessibility (H5AD)')


def definitions(registry=None):
    if registry is None:
        from agent.orchestration import build_default_tool_registry
        registry = build_default_tool_registry()
    return tuple((name, registry.get(name).scientific_delivery) for name in registry.names()
                 if registry.get(name).scientific_delivery is not None)


def for_artifact(artifact):
    matches = [spec for name, spec in definitions() if spec.contract == artifact.contract
               and name == artifact.manifest.get('tool_name') and artifact.role in spec.files]
    return matches[0] if len(matches) == 1 else None


def _publication(app, revision, origin, output, step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    store = app._application.run_store
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, authority, execution, anchor = _accepted(store, run_id, step_id)
    spec = app._application.registry.accepted(step.tool_name).scientific_delivery
    if spec is None:
        raise ValueError('No reviewed scientific delivery declaration')
    result = _serialize(step.result); record = authority.to_dict(); path = Path(result['manifest_path'])
    workspace = app._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'; path.relative_to(root)
    manifest, sha = spec.publication(step.resolved_arguments, result, record)
    if (output is not None and accepted_step_digest(step) != output.accepted_step_sha256
            or record['artifact_contract'] != spec.contract or manifest['execution_identity'] != execution
            or record['resources']['manifest_identity'] != manifest
            or record['resources']['result_metadata'] != result):
        raise ValueError('Scientific artifacts differ from accepted publication')
    return store, run_id, step_id, step, authority, anchor, root, manifest, sha, record, spec


def accepted_artifacts(app, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .session_state import digest
    store, run_id, step_id, step, authority, anchor, root, manifest, sha, record, spec = _publication(
        app, revision, origin, output, step_id)
    items = []
    for pin in manifest['files']:
        path = Path(pin['path']); name = path.name; path.relative_to(root)
        if name not in spec.files:
            continue
        if pin not in record['files']:
            raise ValueError('Unbound scientific artifact')
        if pin['size_bytes'] == 0:
            continue
        identity = dict(session_id=session_id, revision_id=revision.revision_id, turn_id=origin.turn_id, run_id=run_id,
            step_id=step_id, accepted_step_sha256=accepted_step_digest(step), authority_sha256=authority.identity_sha256,
            manifest_sha256=sha, payload=pin)
        ref = ScientificArtifactHandle(digest(identity), spec.artifact_type, pin['sha256'], revision.revision_id,
            origin.turn_id, pin['size_bytes'], name, _label(step, name, spec.files[name][0]))
        delivery = dict(manifest, tool_name=step.tool_name, accepted_step_sha256=accepted_step_digest(step))
        items.append(_ScientificMatrixArtifact(ref, path, root, pin['sha256'], pin['size_bytes'], spec.contract, delivery, name, sha))
    if _accepted(store, run_id, step_id)[3] != anchor:
        raise ValueError('Scientific accepted anchor changed')
    return items


def _label(step, name, default):
    from agent.tools.analysis.cell_result import describe
    view = describe(step.tool_name, step.result)
    if view is None:
        return default
    if name == 'result.h5ad':
        if view['coordinates']:
            return 'UMAP coordinates and retained matrix (H5AD)'
        if view['grouping']:
            return 'Leiden groups and retained matrix (H5AD)'
        measurement = step.result.get('measurement')
        if measurement in ('fragment_counts', 'peak_counts', 'binary_accessibility'):
            return {'fragment_counts': 'Region fragment counts', 'peak_counts': 'Peak counts',
                    'binary_accessibility': 'Binary accessibility'}[measurement] + ' (H5AD)'
    return default


def prepare(stream, role):
    """Common JSON/TSV privacy projection; original publication bytes stay pinned."""
    if role.endswith('.json'):
        import json
        from .sequence_delivery import prepare as metadata_prepare
        value = json.load(stream)
        if isinstance(value, dict):
            value.pop('session', None)
            if isinstance(value.get('receipt'), dict):
                value['receipt'].pop('session', None)
        return metadata_prepare(io.BytesIO(json.dumps(value).encode()), 'metadata.json')
    raw = stream.read()
    if role.endswith('.tsv'):
        from .dialogue_evidence import _unsafe_fact_value
        rows = list(csv.reader(io.StringIO(raw.decode()), delimiter='\t'))
        # Native tables include both headered statistics and headerless genomic
        # tracks. Project every cell; a first row is not necessarily a header.
        for row in rows:
            for i, text in enumerate(row):
                try:
                    float(text); numeric = True
                except ValueError:
                    numeric = False
                if _unsafe_fact_value(text) or not numeric and text not in ('+', '-') and text.startswith(('=', '+', '-', '@')):
                    row[i] = PRIVATE_LABEL
        target = io.StringIO(); csv.writer(target, delimiter='\t').writerows(rows); raw = target.getvalue().encode()
    return raw


def accepted_figures(app, turn, revision):
    from agent.orchestration.verification_authority import _accepted
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.tools._file_identity import file_closure
    if revision is None:
        return ()
    figures = []; seen = set()
    for output in revision.outputs:
        key = (output.run_id, output.step_id)
        if output.run_id != turn.run_id or key in seen:
            continue
        seen.add(key); run = app._application.run_store.load(output.run_id)
        step = next(s for s in run.steps if s.step_id == output.step_id)
        spec = app._application.registry.accepted(step.tool_name).scientific_delivery
        if spec is None or spec.figure is None:
            continue
        store, run_id, step_id, step, authority, anchor, root, manifest, sha, record, spec = _publication(app, revision, turn, output)
        figure_name, figure_kind = spec.figure
        pin = next(p for p in manifest['files'] if Path(p['path']).name == figure_name)
        path = Path(pin['path']); path.relative_to(root)
        if pin not in record['files'] or file_closure([path]) != [pin]:
            raise ValueError('Scientific figure integrity changed')
        if _accepted(store, run_id, step_id)[3] != anchor:
            raise ValueError('Scientific figure anchor changed')
        source = dict(tool_name=step.tool_name, role='native_scientific_figure', output=dict(name=output.name,
            run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(step), revision_id=revision.revision_id))
        figures.append(dict(path=str(path), sha256=pin['sha256'], metadata=dict(kind=figure_kind, figure_id=figure_name,
            manifest_sha256=sha, revision_id=revision.revision_id, run_id=run_id, step_ids=[step_id],
            output_names=[o.name for o in revision.outputs if (o.run_id, o.step_id) == key], sources=[source])))
    return tuple(figures)
