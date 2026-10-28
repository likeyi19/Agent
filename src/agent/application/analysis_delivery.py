"""Closed safe exports of pinned embeddings, Leiden and UMAP results."""
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

CONTRACT = 'compact-analysis-delivery.v1'
ROLE = 'compact_analysis_export'
FILENAME = 'analysis.h5ad'
FILENAMES = {'embedding': 'embeddings.h5ad', 'clustering': 'clusters.h5ad', 'umap': FILENAME}
LABEL = 'Embeddings, Leiden and UMAP — safe export'
INTEGRATED_ROLE = 'integrated_analysis_export'
COMPOSED_ROLE = 'composed_analysis_export'
COMPOSED_CONTRACT = 'qualified-multi-result-h5ad.v1'
COMPOSED_FILENAME = 'composed-analysis.h5ad'
COMPOSED_METADATA_BYTES = 256 * 1024
COMPOSED_FIELD_OVERHEAD_BYTES = 8 * 1024
EXPORT_KEY = 'agent_scientific_export'


class ExportSelectionError(ValueError):
    """A closed export prerequisite/collision reason, without private details."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__('The selected scientific export is unavailable: ' + reason)


def accepted_analysis(service, session_id, revision, origin, output, *, step_id=None):
    from agent.orchestration.verification_authority import _accepted_step
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.application.session_state import digest
    from agent.application.interactive_schemas import ScientificArtifactHandle
    from agent.application.interactive import _ScientificMatrixArtifact
    from agent.application.matrix_delivery import _open_matrix_source
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    stored, planned, state = _accepted_step(service._application.run_store, run_id, step_id)
    result = stored.result
    anchor = accepted_step_digest(stored)
    stages = {'epizoo_embed_cells': 'embedding', 'pca_embed_cells': 'embedding', 'cluster_cells': 'clustering', 'compute_cell_umap': 'umap'}
    if (stored.tool_name not in stages or state.lifecycle_status.value != 'SUCCEEDED'
            or output is not None and output.accepted_step_sha256 != anchor
            or 'analysis_sha256' not in result):
        raise ValueError('No accepted pinned compact analysis is available.')
    sha = result['analysis_sha256']
    if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Invalid compact analysis digest.')
    workspace = service._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    path = Path(result['analysis_path'])
    with _open_matrix_source(path, root) as stream:
        size = os.fstat(stream.fileno()).st_size
    identity = dict(session_id=session_id, revision_id=revision.revision_id,
        turn_id=origin.turn_id, run_id=run_id, step_id=step_id, accepted_step=anchor,
        role=ROLE, sha256=sha, size_bytes=size)
    stage = stages[stored.tool_name]
    filename = FILENAMES[stage]
    label = {'embedding': 'Cell embeddings — safe export',
             'clustering': 'Embeddings and Leiden — safe export',
             'umap': 'Embeddings and UMAP — safe export'
                     if result.get('clustering_present') is False else LABEL}[stage]
    reference = ScientificArtifactHandle(digest(identity), 'scientific_analysis', sha,
        revision.revision_id, origin.turn_id, size, filename, label)
    from agent.schemas import StepOutputRef, PriorOutputRef
    from .result_reading import dependency, analysis_lineage
    dependencies = {}
    for argument, ref in planned.arguments.items():
        if isinstance(ref, (StepOutputRef, PriorOutputRef)):
            source_run, source_step = dependency(service._application.run_store, run_id, step_id, argument)
            producer, _, _ = _accepted_step(service._application.run_store, source_run, source_step)
            dependencies[argument] = dict(run_id=source_run, step_id=source_step,
                output_key=ref.output_key, accepted_step_sha256=accepted_step_digest(producer))
    provenance = dict(revision_id=revision.revision_id, run_id=run_id, step_id=step_id,
                      accepted_step_sha256=anchor, dependencies=dependencies)
    versions, execution = {}, {}
    lineage = []
    for source_run, producer, _ in analysis_lineage(service._application.run_store, run_id, step_id):
        lineage.append(dict(run_id=source_run, step_id=producer.step_id, tool_name=producer.tool_name,
            accepted_step_sha256=accepted_step_digest(producer),
            **({'analysis_sha256': producer.result['analysis_sha256']} if 'analysis_sha256' in producer.result else {})))
        key = {'pca_embed_cells': 'embedding', 'epizoo_embed_cells': 'embedding',
               'build_cell_neighbors': 'neighbors', 'cluster_cells': 'clustering', 'compute_cell_umap': 'umap'}.get(producer.tool_name)
        if key is not None and 'software_versions' in producer.result:
            versions[key] = dict(producer.result['software_versions'])
        if producer.tool_name == 'epizoo_embed_cells':
            # Legacy embedding files predate method metadata. Project only
            # actual accepted owner fields/arguments; absent defaults stay unknown.
            execution['embedding'] = {k: producer.result[k] for k in
                ('backend', 'species', 'device', 'embedding_dtype') if k in producer.result}
            execution['embedding']['recorded_execution_arguments'] = {k: producer.resolved_arguments[k]
                for k in ('device',)
                if k in producer.resolved_arguments}
            resources = producer.result.get('resource_provenance', {})
            if 'expected_resource_identity' in resources:
                execution['embedding']['resource_identity'] = dict(resources['expected_resource_identity'])
            if 'inference_settings' in resources:
                execution['embedding']['parameters'] = dict(resources['inference_settings'])
    provenance['lineage'] = {str(index): item for index, item in enumerate(lineage)}
    return _ScientificMatrixArtifact(reference, path, root, sha, size, CONTRACT,
        dict(n_cells=result['n_cells'], stage=stage, provenance=provenance,
             method_software_versions=versions, recorded_execution=execution,
             clustering_present=result.get('clustering_present', True)), ROLE)


def integrated_analysis(service, artifact):
    """Offer an optional derivative of this result and its recorded exact source.

    Inventory reads accepted lineage and file size only. Bytes, privacy and axes
    are checked on private snapshots when the researcher requests the download.
    """
    from .result_reading import original_source
    from .matrix_delivery import _open_matrix_source
    from .session_state import digest
    from .interactive_schemas import ScientificArtifactHandle
    provenance = artifact.manifest['provenance']
    source = original_source(service._application.run_store, provenance['run_id'], provenance['step_id'])
    if source is None:
        return None
    path = Path(source['path'])
    with _open_matrix_source(path, path.parent) as stream:
        size = os.fstat(stream.fileno()).st_size
    ref = artifact.reference
    filename = 'integrated-' + ref.filename
    reference = ScientificArtifactHandle(digest(dict(compact_handle=ref.handle, role=INTEGRATED_ROLE,
        original_sha256=source['sha256'], original_size_bytes=size)), 'scientific_integrated', artifact.sha256,
        ref.revision_id, ref.turn_id, None, filename, 'Original data with accepted analysis')
    return replace(artifact, reference=reference, role=INTEGRATED_ROLE,
        manifest=dict(artifact.manifest, original_source=dict(source, size_bytes=size)))


def export_descriptions(sessions, session_id, targets):
    """Project existing eligible matrix/analysis downloads without preparing bytes.

    The normal artifact owners establish inventory membership. Payload integrity,
    privacy, size admission and exact delivered bytes remain download-time work.
    No source is promoted to scientific authority by this presentation.
    """
    state = sessions.load(session_id)
    descriptions = []
    checked = set()
    for index, target in enumerate(targets):
        revisions = [r for r in state.revisions if r.revision_id == target['revision_id']]
        if len(revisions) != 1:
            raise ValueError('The export target is not an accepted revision.')
        revision = revisions[0]
        if revision.revision_id not in checked:
            sessions._validate_revision(revision)
            checked.add(revision.revision_id)
        outputs = [o for o in revision.outputs if o.name == target['output_name']
                   and o.accepted_step_sha256 == target['accepted_step_sha256']]
        if len(outputs) != 1:
            raise ValueError('The export target does not identify an accepted output.')
        output = outputs[0]
        sessions._validate_locator(output)
        origin = state.turn(revision.turn_id)
        from .matrix_delivery import MATRIX_CONTRACTS, accepted_matrix, selects_matrix_port
        run = sessions._application.run_store.load(output.run_id)
        step = next(s for s in run.steps if s.step_id == output.step_id)
        if step.tool_name in ('adapt_epizoo_species', 'posttrain_epizoo_model'):
            from .model_delivery import accepted_models
            selected = output.output_key in ('manifest_path', 'manifest_sha256')
            if not selected and output.run_id != origin.run_id:
                continue
            try:
                models = accepted_models(sessions, session_id, revision, origin,
                    output if selected else None, step_id=output.step_id)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
                continue
            for artifact in models:
                reference = artifact.reference
                descriptions.append(dict(target=index, status='available_for_preparation',
                    **{name:getattr(reference, name) for name in (
                        'handle', 'artifact_type', 'filename', 'title', 'purpose', 'description', 'size_bytes')},
                    model_state=step.result['purpose'], optimizer_updates=step.result['optimizer_steps'],
                    transfer_strategy=step.result['strategy'], payload_checks='performed_on_download',
                    actual_download_status='NOT_RECORDED'))
            continue
        if step.tool_name in MATRIX_CONTRACTS:
            selected = selects_matrix_port(sessions._application.registry, step.tool_name, output.output_key)
            if not selected and output.run_id != origin.run_id:
                continue
            try:
                artifact = accepted_matrix(sessions, session_id, revision, origin,
                    output if selected else None, step_id=output.step_id)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
                continue
            reference = artifact.reference
            descriptions.append(dict(target=index, status='available_for_preparation',
                **{name:getattr(reference, name) for name in (
                    'handle', 'artifact_type', 'filename', 'title', 'purpose', 'description', 'size_bytes')},
                original_measurements=True, original_feature_axis=True,
                selected_result=dict(cell_embeddings=False, leiden_assignments=False, umap_coordinates=False),
                payload_checks='performed_on_download', actual_download_status='NOT_RECORDED'))
            continue
        try:
            compact = accepted_analysis(sessions, session_id, revision, origin, output)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
            continue  # Optional delivery never removes independent accepted facts.
        artifacts = [compact]
        try:
            integrated = integrated_analysis(sessions, compact)
            if integrated is not None:
                artifacts.append(integrated)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, StopIteration):
            pass
        stage = compact.manifest['stage']
        selected = dict(cell_embeddings=True,
            leiden_assignments=stage == 'clustering' or stage == 'umap' and compact.manifest['clustering_present'],
            umap_coordinates=stage == 'umap')
        for artifact in artifacts:
            reference = artifact.reference
            integrated = artifact.role == INTEGRATED_ROLE
            descriptions.append(dict(target=index, status='available_for_preparation',
                **{name:getattr(reference, name) for name in (
                    'handle', 'artifact_type', 'filename', 'title', 'purpose', 'description', 'size_bytes')},
                original_measurements=integrated, original_feature_axis=integrated,
                selected_result=dict(selected), payload_checks='performed_on_download'))
    return tuple(descriptions)


def _methods(handle, manifest):
    from agent.tools.analysis.embedding_analysis import scientific_methods
    methods = scientific_methods(handle)
    for key, versions in manifest.get('method_software_versions', {}).items():
        if key in methods:
            methods[key]['software_versions'] = versions
    for key, record in manifest.get('recorded_execution', {}).items():
        if key in methods:
            methods[key].update(record)
    return methods


def _compact_result(source, manifest, max_bytes):
    from agent.tools.analysis.peak_h5ad import _PATH
    from agent.tools.analysis.neighbors_delivery import _safe_objects, _strings
    from agent.tools.analysis.embedding_analysis import _validate_umap_labels, _validate_cluster_labels
    with h5py.File(source, 'r') as handle:
        _safe_objects(handle, max_bytes, numeric_kinds=frozenset({'i', 'f', 'b'}), max_depth=6, max_objects=160)
        def bounded_strings(name, value):
            if isinstance(value, h5py.Dataset) and h5py.check_string_dtype(value.dtype) is not None:
                for _ in _strings(value, private=True):
                    pass
        handle.visititems(bounded_strings)
        from anndata.io import read_elem
        methods = _methods(handle, manifest)
        data = ad.AnnData(obs=read_elem(handle['obs']), obsm={key: read_elem(handle['obsm'][key])
            for key in (methods['embedding']['representation'], 'X_umap') if key in handle['obsm']})
        # Only the selected observation/coordinate values and small provenance
        # are needed. Accepted graph validity belongs to the scientific owner.
        for key in ('agent_cell_embedding', 'agent_milestone6'):
            if 'uns/' + key in handle:
                group = handle['uns'][key]
                data.uns[key] = {name: read_elem(group[name]) for name in group
                    if name not in ('source_embedding_path', 'source_cell_ids_path', 'source_analysis_path')}
    stage = manifest.get('stage', 'umap')
    if stage not in ('embedding', 'clustering', 'umap'):
        raise ValueError('Unsupported compact analysis stage.')
    if not data.obs_names.is_unique:
        raise ValueError('Invalid compact embedding structure.')
    clustered = (_validate_umap_labels(data) if stage == 'umap' else
                 bool(_validate_cluster_labels(data)) if stage == 'clustering' else False)
    if stage == 'umap' and clustered is not manifest.get('clustering_present', True):
        raise ValueError('Changed UMAP clustering variant.')
    if data.n_obs != manifest['n_cells']:
        raise ValueError('Changed analysis cell count.')
    labels = tuple(data.obs['leiden'].astype(str)) if clustered else ()
    for text in (*data.obs_names, *labels):
        if not text or _PATH.search(text) or any(ord(c) < 32 for c in text):
            raise ValueError('Cell identities or labels are ineligible for public delivery.')
    result = ad.AnnData(obs=data.obs[['leiden'] if clustered else []].copy())
    from agent.tools.analysis.embedding_analysis import representation
    if stage == 'embedding' and 'agent_cell_embedding' in data.uns:
        from agent.tools.analysis.pca_embedding import descriptor
        meta = descriptor(data); key, dimensions = meta['representation'], meta['embedding_dim']
    elif stage == 'embedding':
        key, dimensions = 'X_epizoo', 512
    else:
        key, values = representation(data); dimensions = values.shape[1]
    for key, dimensions in ((key, dimensions),) + ((('X_umap', 2),) if stage == 'umap' else ()):
        values = np.asarray(data.obsm[key])
        if values.shape != (data.n_obs, dimensions) or not np.isfinite(values).all():
            raise ValueError('Invalid compact analysis coordinates.')
        result.obsm[key] = values.copy()
    return result, methods


def _metadata(manifest, methods, source_sha256, *, integrated=False):
    from .dialogue_evidence import _unsafe_fact_value
    if _unsafe_fact_value(methods):
        raise ValueError('Method metadata is ineligible for public delivery.')
    metadata = dict(contract=CONTRACT, format='integrated_annotated_h5ad' if integrated else 'compact_analysis_result_h5ad',
        scope=('Original supplied scientific data with selected accepted cell analysis; no scientific transformation.'
               if integrated else 'Selected cell identities, assignments and coordinates; X=None and no original features.'),
        source_sha256=source_sha256, stage=manifest.get('stage', 'umap'), methods=methods,
        # Describe the selected, validated result, not any original-data labels
        # preserved by an integrated export. Keep the public label-free marker.
        clustering_present='clustering' in methods,
        accepted_result=manifest.get('provenance', {}),
        authority_scope='Delivery derivative; does not issue scientific authority.',
        omitted_information='Derived neighbor graphs and PCA loadings/fit arrays; private source locators. '
            'Unrecorded historical method settings are unavailable, not inferred.')
    # Preserve the original compact PCA fields through the shared method
    # projection; no method-specific export branch or invented defaults.
    embedding = methods.get('embedding', {})
    metadata.update({key: embedding[key] for key in ('method', 'representation', 'embedding_dim', 'parameters')
                     if key in embedding})
    if 'source_sha256' in embedding:
        metadata['input_sha256'] = embedding['source_sha256']
    return metadata


def create_analysis_export(source, destination, manifest, *, source_sha256,
                           source_size_bytes, manifest_sha256, max_bytes):
    from agent.tools.analysis.neighbors_delivery import _BoundedDestination
    result, methods = _compact_result(source, manifest, max_bytes)
    result.uns[EXPORT_KEY] = _metadata(manifest, methods, source_sha256)
    with h5py.File(_BoundedDestination(destination, max_bytes), 'w') as handle:
        from anndata.io import write_elem
        write_elem(handle, '/', result)
    destination.seek(0, 2)
    if destination.tell() > max_bytes:
        raise ValueError('Compact analysis export exceeds its size bound.')
    destination.seek(0)


def create_integrated_export(source, original, destination, manifest, *, source_sha256, max_bytes):
    """Copy a pinned source logically in bounded blocks, then add exact results."""
    from anndata.io import read_elem, write_elem
    from .matrix_delivery import _copy, _redactions
    from agent.tools.analysis.neighbors_delivery import _BoundedDestination
    from agent.tools.analysis.peak_h5ad import inspect
    from agent.tools.analysis.categorical import ordered_ids, require_same_cells
    result, methods = _compact_result(source, manifest, max_bytes)
    metadata = _metadata(manifest, methods, source_sha256, integrated=True)
    metadata['original_source_sha256'] = manifest['original_source']['sha256']
    metadata['fields'] = {}
    with h5py.File(original, 'r') as supplied:
        inspect(supplied)
        cells = read_elem(supplied['obs'][supplied['obs'].attrs['_index']])
        require_same_cells(ordered_ids(cells, source='Original data'), tuple(result.obs_names))
        omitted = _redactions(supplied, preserve_annotation=False)
        metadata['omitted_private_metadata_fields'] = len(omitted)
        with h5py.File(_BoundedDestination(destination, max_bytes), 'w') as exported:
            _copy(supplied, exported, omitted)
            for slot, values in (('obs', result.obs), ('obsm', result.obsm)):
                group = exported.require_group(slot)
                if slot == 'obsm' and not group.attrs:
                    group.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
                fields = {}
                for name in values:
                    # Original names always win. A collision at the one clearly
                    # differentiated derived name is genuinely ambiguous.
                    target = name if name not in group else 'agent_' + name
                    if target in group:
                        raise ValueError('Original and derived-result field names collide.')
                    value = values[name]
                    write_elem(group, target, value.array if slot == 'obs' else value)
                    fields[name] = target
                metadata['fields'][slot] = fields
                if slot == 'obs':
                    group.attrs['column-order'] = np.array([*group.attrs['column-order'], *fields.values()],
                                                         dtype=h5py.string_dtype())
            uns = exported.require_group('uns')
            if not uns.attrs:
                uns.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
            key = EXPORT_KEY if EXPORT_KEY not in uns else 'agent_integrated_export'
            if key in uns:
                raise ValueError('Original and export provenance keys collide.')
            write_elem(uns, key, metadata)
    destination.flush(); destination.seek(0)
    # Storage/privacy check only: never replay the accepted scientific method.
    inspect(destination, privacy=True)
    destination.seek(0)


def _json_values(value):
    """Normalize small recorded method values for immutable client metadata."""
    from collections.abc import Mapping
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_values(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {k: _json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(v) for v in value]
    return value


def _selected_output(sessions, state, locator):
    """Validate the exact captured accepted locator, without selecting history."""
    from dataclasses import asdict
    revisions = [r for r in state.revisions if r.revision_id == locator['revision_id']]
    if len(revisions) != 1:
        raise ValueError('The export source is not an accepted revision.')
    revision = revisions[0]
    outputs = [o for o in revision.outputs if dict(asdict(o), revision_id=revision.revision_id) == dict(locator)]
    if len(outputs) != 1:
        raise ValueError('The export source no longer identifies its exact accepted output.')
    sessions._validate_locator(outputs[0])
    return revision, state.turn(revision.turn_id), outputs[0]


def _source_record(path, sha256, root):
    from .matrix_delivery import _open_matrix_source
    path, root = Path(path), Path(root)
    with _open_matrix_source(path, root) as stream:
        size = os.fstat(stream.fileno()).st_size
    return dict(path=str(path), managed_root=str(root), sha256=sha256, size_bytes=size)


def _annotation_source(artifact):
    """Read the annotation owner's recorded original measurement binding."""
    from .matrix_delivery import _open_matrix_source
    from agent.tools.analysis import annotation_contract
    if annotation_contract.external(artifact.manifest):
        binding = artifact.manifest['matrix_binding']
        return dict(path=binding['source_path'], sha256=binding['source_sha256'])
    import hashlib
    from agent.tools.data import scatac_matrix_contract
    binding = artifact.manifest['matrix_binding']
    path = Path(binding['manifest_path'])
    with _open_matrix_source(path, path.parent) as stream:
        raw = stream.read(scatac_matrix_contract.MAX_MANIFEST_BYTES + 1)
    if (len(raw) > scatac_matrix_contract.MAX_MANIFEST_BYTES
            or hashlib.sha256(raw).hexdigest() != binding['manifest_sha256']):
        raise ValueError('The recorded annotation measurement manifest changed.')
    manifest = scatac_matrix_contract.load_manifest_bytes(raw)
    if manifest['matrix']['sha256'] != binding['matrix_sha256']:
        raise ValueError('The recorded annotation measurement identity changed.')
    return dict(path=str(path.parent / manifest['matrix']['path']), sha256=binding['matrix_sha256'])


def _component(sessions, session_id, state, locator):
    from agent.orchestration.verification_authority import _accepted_step
    from . import annotation_delivery
    from .result_reading import original_source
    revision, origin, output = _selected_output(sessions, state, locator)
    step, _, _ = _accepted_step(sessions._application.run_store, output.run_id, output.step_id)
    if step.tool_name == 'annotate_scATAC_cell_types':
        artifact = annotation_delivery.accepted_annotation(sessions, session_id, revision, origin, output)
        original = _annotation_source(artifact)
    else:
        artifact = accepted_analysis(sessions, session_id, revision, origin, output)
        original = original_source(sessions._application.run_store, output.run_id, output.step_id)
    if original is None:
        raise ValueError('This accepted result does not establish its exact original measurement identity.')
    return artifact, original


def _component_methods(artifact, handle):
    if artifact.role == ROLE:
        return _methods(handle, artifact.manifest)
    # Annotation retains its distinct policy and accepted method identity.
    args = artifact.manifest['arguments']
    return dict(annotation=dict(method=artifact.manifest['profile']['primary_profile'],
        parameters={k: args[k] for k in ('annotation_method', 'count_slot', 'group_column') if k in args},
        validation_state='not_assessed'))


@contextmanager
def _accepted_metadata(artifact):
    """Read selected native metadata only after its exact byte pin is checked."""
    import hashlib
    from .matrix_delivery import _open_matrix_source, _source_state, MATRIX_CHUNK_BYTES, MAX_MATRIX_BYTES
    if not 0 < artifact.size_bytes <= MAX_MATRIX_BYTES:
        raise ValueError('Accepted export metadata exceeds the supported source size.')
    with _open_matrix_source(artifact.path, artifact.managed_root) as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != artifact.size_bytes:
            raise ValueError('Accepted export metadata source size changed.')
        checksum, copied = hashlib.sha256(), 0
        while chunk := stream.read(MATRIX_CHUNK_BYTES):
            copied += len(chunk)
            if copied > artifact.size_bytes:
                raise ValueError('Accepted export metadata source changed.')
            checksum.update(chunk)
        if (copied != artifact.size_bytes or checksum.hexdigest() != artifact.sha256
                or _source_state(os.fstat(stream.fileno())) != _source_state(before)):
            raise ValueError('Accepted export metadata source digest changed.')
        stream.seek(0)
        with h5py.File(stream, 'r') as handle:
            yield handle
        if _source_state(os.fstat(stream.fileno())) != _source_state(before):
            raise ValueError('Accepted export metadata source changed while reading.')


def _representation_fields(artifact, handle):
    if artifact.role != ROLE:
        return ()
    methods = _component_methods(artifact, handle)
    embedding = methods.get('embedding', {})
    result = []
    name = embedding.get('representation')
    if name is not None and 'obsm/' + name in handle:
        result.append(dict(field='obsm:' + name, role='accepted_representation', method=_json_values(embedding)))
    if artifact.manifest['stage'] == 'umap' and 'obsm/X_umap' in handle:
        result.append(dict(field='obsm:X_umap', role='accepted_coordinates',
                           method=_json_values(methods.get('umap', {}))))
    return tuple(result)


def _observation_fields(artifact, handle, *, original=False):
    """Export policy follows the producer; numeric covariates are not labels."""
    from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
    names = handle['obs'].attrs['column-order']
    if not original and artifact.role == ROLE:
        names = [name for name in names if name == 'leiden' and (
            artifact.manifest['stage'] == 'clustering' or artifact.manifest['stage'] == 'umap'
            and artifact.manifest.get('clustering_present', True))]
    return tuple(name for name in names if not _unsafe_fact_field(name) and not _unsafe_fact_value(name))


def _field_logical_bytes(handle, slot, field):
    """Bound each destination's selected values, including legacy categories."""
    from agent.tools.analysis.peak_h5ad import strings
    from .matrix_delivery import MAX_MATRIX_BYTES
    seen = set()
    def size(item):
        address = h5py.h5o.get_info(item.id).addr
        if address in seen:
            return 0
        seen.add(address)
        if isinstance(item, h5py.Group):
            if any(not isinstance(item.get(k, getlink=True), h5py.HardLink) for k in item):
                raise ValueError('Indirect selected field storage is unsupported.')
            return sum(size(value) for value in item.values())
        if item.is_virtual or item.external:
            raise ValueError('Indirect selected field storage is unsupported.')
        if h5py.check_string_dtype(item.dtype) is not None:
            if item.ndim > 1:
                raise ValueError('Selected string field storage is unsupported.')
            selections = ((),) if item.shape == () else (
                slice(left, min(left + 4096, len(item))) for left in range(0, len(item), 4096))
            amount = 0
            for selection in selections:
                amount += sum(len(value.encode('utf-8')) + 8 for value in strings(item, selection))
                if amount > MAX_MATRIX_BYTES:
                    raise ValueError('Selected field exceeds the export resource limit.')
        elif item.dtype.kind in 'biuf':
            amount = int(item.size) * item.dtype.itemsize
        else:
            raise ValueError('Selected field has unsupported scientific storage.')
        if 'categories' in item.attrs:
            amount += size(handle[item.attrs['categories']])
        if amount > MAX_MATRIX_BYTES:
            raise ValueError('Selected field exceeds the export resource limit.')
        return amount
    group = handle[slot]
    if field not in group or not isinstance(group.get(field, getlink=True), h5py.HardLink):
        raise ExportSelectionError('selected_field_unavailable')
    return size(group[field])


def export_fields(sessions, session_id, target):
    """Offer accepted field names; selected reads verify the actual native bytes.

    Representation names come from exact accepted producers, never native method
    values. Observation discovery uses the existing bounded metadata owner and
    does not decode columns. These offers do not establish payload readiness.
    """
    from dataclasses import asdict
    from .matrix_delivery import _open_matrix_source
    from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
    from .result_reading import analysis_lineage
    from agent.tools.analysis.peak_h5ad import inspect
    state = sessions.load(session_id)
    revision = next(r for r in state.revisions if r.revision_id == target['revision_id'])
    output = next(o for o in revision.outputs if o.name == target['output_name']
                  and o.accepted_step_sha256 == target['accepted_step_sha256'])
    artifact, original = _component(sessions, session_id, state, dict(asdict(output), revision_id=revision.revision_id))
    rows = []
    if artifact.role == ROLE:
        provenance = artifact.manifest['provenance']
        lineage = analysis_lineage(sessions._application.run_store, provenance['run_id'], provenance['step_id'])
        # _component's original-source owner already qualifies this exact PCA
        # ancestry. Do not substitute an arbitrary native obsm or a default key.
        fields = [(lineage[-1][1].result['representation'], 'accepted_representation')]
        if artifact.manifest['stage'] == 'umap':
            fields.append((lineage[0][1].result['umap_key'], 'accepted_coordinates'))
        for name, role in fields:
            if _unsafe_fact_field(name) or _unsafe_fact_value(name):
                raise ValueError('Accepted representation name is not eligible for public delivery.')
            rows.append(dict(field='obsm:' + name, role=role))
    with _open_matrix_source(artifact.path, artifact.managed_root) as stream, h5py.File(stream, 'r') as handle:
        inspect(handle, observation_fields=())
        rows.extend(dict(field='result:' + name, role='accepted_observation')
            for name in _observation_fields(artifact, handle))
    with _open_matrix_source(original['path'], Path(original['path']).parent) as stream, h5py.File(stream, 'r') as handle:
        inspect(handle, observation_fields=())
        rows.extend(dict(field='source:' + name, role='supplied_source_observation')
            for name in _observation_fields(artifact, handle, original=True))
    if _unsafe_fact_value(rows):
        raise ValueError('Recorded representation metadata is not eligible for public delivery.')
    return tuple(rows)


def composed_analysis(sessions, session_id, resolved_export, *, turn_id):
    """Describe one explicit qualified recipe; no derivative is prepared here."""
    from .matrix_delivery import MATRIX_CONTRACTS, accepted_matrix, _open_matrix_source
    from .local_resources import LocalResourceAdmission
    from .interactive import _ScientificMatrixArtifact
    from .interactive_schemas import ScientificArtifactHandle
    from .session_state import digest
    from .dialogue_evidence import _unsafe_fact_field, _unsafe_fact_value
    from agent.orchestration.verification_authority import _accepted_step
    from agent.schemas.orchestration import _serialize
    state = sessions.load(session_id)
    measurement = resolved_export['measurement']
    if 'registered_source' in measurement:
        selected = measurement['registered_source']
        record = LocalResourceAdmission(sessions._application._workspace).load(selected['resource_id'])
        if (record.session_id not in (None, session_id) or record.record_sha256 != selected['record_sha256']
                or record.source_sha256 != selected['source_sha256']):
            raise ValueError('The captured original measurement registration changed.')
        original = dict(path=record.source_path, sha256=record.source_sha256)
        measurement_identity = dict(origin='registered_source', registered_source=dict(selected))
    else:
        locator = measurement['accepted_output']
        revision, origin, output = _selected_output(sessions, state, locator)
        step, _, _ = _accepted_step(sessions._application.run_store, output.run_id, output.step_id)
        if step.tool_name in MATRIX_CONTRACTS:
            artifact = accepted_matrix(sessions, session_id, revision, origin, output)
            original = dict(path=str(artifact.path), sha256=artifact.sha256)
        else:
            _, original = _component(sessions, session_id, state, locator)
        measurement_identity = dict(origin='accepted_output', accepted_output=dict(locator))
    base = _source_record(original['path'], original['sha256'], Path(original['path']).parent)
    measurement_identity['source_sha256'] = base['sha256']
    components, sources, cached, metadata_cache, destinations = [], [], {}, {}, {'obs': set(), 'obsm': set()}
    public = dict(observations=[], representations=[])
    for name, slot in (('observations', 'obs'), ('representations', 'obsm')):
        for selection in resolved_export[name]:
            locator, field, destination = selection['accepted_output'], selection['field'], selection['destination']
            if (type(destination) is not str or not destination.strip() or len(destination) > 256
                    or destination in ('.', '..', '_index', '__categories') or '/' in destination or '\\' in destination
                    or any(ord(c) < 32 or ord(c) == 127 for c in destination)
                    or _unsafe_fact_field(destination) or _unsafe_fact_value(destination)
                    or destination in destinations[slot]):
                raise ValueError('Select distinct safe destination field names.')
            destinations[slot].add(destination)
            identity = digest(dict(locator))
            if identity not in cached:
                artifact, relationship = _component(sessions, session_id, state, locator)
                if relationship['sha256'] != base['sha256']:
                    raise ExportSelectionError('unqualified_measurement_source')
                source = _source_record(artifact.path, artifact.sha256, artifact.managed_root)
                if source not in sources:
                    sources.append(source)
                source_key = tuple(source[k] for k in ('path', 'managed_root', 'sha256', 'size_bytes'))
                if source_key not in metadata_cache:
                    with _accepted_metadata(artifact) as handle:
                        methods = _json_values(_component_methods(artifact, handle))
                        representations = _representation_fields(artifact, handle)
                    metadata_cache[source_key] = methods, representations
                methods, representations = metadata_cache[source_key]
                if _unsafe_fact_value(methods):
                    raise ValueError('Recorded scientific methods are not eligible for public delivery.')
                cached[identity] = artifact, sources.index(source), methods, representations
            artifact, source_index, methods, representations = cached[identity]
            if slot == 'obsm' and field not in {row['field'] for row in representations}:
                raise ValueError('Select the actual declared accepted representation or UMAP field.')
            if slot == 'obs' and (not field.startswith(('result:', 'source:')) or not field.split(':', 1)[1]
                    or _unsafe_fact_field(field.split(':', 1)[1]) or _unsafe_fact_value(field)):
                raise ValueError('Select an exact eligible observation field.')
            if slot == 'obs':
                spec = base if field.startswith('source:') else sources[source_index]
                with _open_matrix_source(spec['path'], spec['managed_root']) as stream, h5py.File(stream, 'r') as h:
                    available = _observation_fields(artifact, h, original=field.startswith('source:'))
                if field.split(':', 1)[1] not in available:
                    raise ExportSelectionError('selected_field_unavailable')
            row = dict(accepted_output=dict(locator), field=field, destination=destination, method=methods)
            if slot == 'obs':
                if field.startswith('source:'):
                    row['observation_role'] = 'supplied_source_observation; biological validity not assessed'
                elif artifact.role == ROLE:
                    row['observation_role'] = 'computational_assignment'
                else:
                    from agent.tools.analysis.annotation_contract import OBS_COLUMNS
                    row['observation_role'] = ('computational_assignment' if field.split(':', 1)[1] in OBS_COLUMNS
                        else 'copied_supplied_source_observation; biological validity not assessed')
            public[name].append(row)
            spec = base if field.startswith('source:') else sources[source_index]
            with _open_matrix_source(spec['path'], spec['managed_root']) as stream, h5py.File(stream, 'r') as h:
                logical_bytes = _field_logical_bytes(h, slot, field.split(':', 1)[1])
            components.append(dict(row, slot=slot, source_index=source_index, logical_bytes=logical_bytes))
    if not 1 <= len(components) <= 32:
        raise ValueError('Select between one and 32 accepted export fields.')
    with _open_matrix_source(base['path'], base['managed_root']) as stream, h5py.File(stream, 'r') as supplied:
        for row in components:
            if row['destination'] in supplied.get(row['slot'], {}):
                raise ExportSelectionError('destination_collision')
    contents = dict(policy=COMPOSED_CONTRACT, measurement=measurement_identity, **public,
        omitted_information=['Only explicitly selected derived fields are added; other accepted results, derived neighbor graphs and fit arrays are not added.',
            'Original safe scientific content is retained; private metadata is omitted.',
            'Unrecorded method information remains unavailable.'],
        preparation='on_demand', payload_checks='performed_on_download')
    recipe_identity = dict(session_id=session_id, turn_id=turn_id, policy=COMPOSED_CONTRACT,
        measurement=measurement_identity, components=[{k:r[k] for k in (
            'accepted_output', 'field', 'destination', 'slot')} for r in components])
    handle = digest(recipe_identity)
    reference = ScientificArtifactHandle(handle, 'scientific_composition', base['sha256'], None,
        turn_id, None, COMPOSED_FILENAME, 'Original data with selected accepted results', contents=contents)
    return _ScientificMatrixArtifact(reference, Path(base['path']), Path(base['managed_root']), base['sha256'],
        base['size_bytes'], COMPOSED_CONTRACT, dict(original_source=base, sources=sources,
        components=components, contents=contents, recipe_identity=recipe_identity, recipe_handle=handle), COMPOSED_ROLE)


def create_composed_export(original, sources, destination, manifest, *, max_bytes):
    """Copy exact original storage and only explicitly mapped accepted fields."""
    from contextlib import ExitStack
    from anndata.io import read_elem, write_elem
    from .matrix_delivery import _copy, _redactions
    from agent.tools.analysis.peak_h5ad import inspect
    from agent.tools.analysis.categorical import ordered_ids, require_same_cells, _labels, _series
    import pandas as pd
    required = (manifest['original_source']['size_bytes']
        + sum(row['logical_bytes'] for row in manifest['components'])
        + COMPOSED_METADATA_BYTES + len(manifest['components']) * COMPOSED_FIELD_OVERHEAD_BYTES)
    if type(max_bytes) is not int or not required <= max_bytes:
        raise ValueError('Composed export exceeds the admitted preparation budget.')
    with ExitStack() as stack:
        supplied = stack.enter_context(h5py.File(original, 'r'))
        inspect(supplied)
        cells = ordered_ids(_series(supplied['obs'][supplied['obs'].attrs['_index']]), source='Original measurements')
        selected = [stack.enter_context(h5py.File(stream, 'r')) for stream in sources]
        for source in selected:
            obs = source['obs']
            inspect(source, observation_fields=(obs.attrs['_index'],))
            require_same_cells(cells, ordered_ids(_series(obs[obs.attrs['_index']]), source='Accepted result'))
        omitted = _redactions(supplied, preserve_annotation=False)
        # Collisions are checked against original names even when privacy policy
        # will omit one of those fields. No existing scientific field is renamed.
        for row in manifest['components']:
            if row['destination'] in supplied.get(row['slot'], {}):
                raise ExportSelectionError('destination_collision')
        # Keep the preparer's anonymous stream. Raising a custom size exception
        # from HDF5's write callback can mask errors or crash during h5py close.
        # Admit the resource budget before opening HDF5 instead, then enforce
        # the actual byte cap before this derivative can be streamed.
        with h5py.File(destination, 'w') as exported:
            _copy(supplied, exported, omitted)
            for row in manifest['components']:
                source = selected[row['source_index']]
                location, field = row['field'].split(':', 1)
                group = exported.require_group(row['slot'])
                if row['destination'] in group:
                    raise ValueError('Selected export destination fields collide.')
                if row['slot'] == 'obs':
                    field_source = supplied if location == 'source' else source
                    obs = field_source['obs']
                    if field not in obs.attrs['column-order']:
                        raise ValueError('The selected observation field is unavailable.')
                    check = [obs.attrs['_index'], field]
                    if 'categories' in obs[field].attrs:
                        check.append(obs[obs[field].attrs['categories']].name)
                    inspect(field_source, observation_fields=check)
                    value = pd.Series(_series(obs[field]))
                    if len(value) != len(cells):
                        raise ValueError('The selected observation field and cell axis differ.')
                    if row.get('observation_role') == 'computational_assignment' or not pd.api.types.is_numeric_dtype(value.dtype):
                        _labels(value)
                    elif pd.api.types.is_complex_dtype(value.dtype):
                        raise ValueError('Selected observation storage is unsupported.')
                    payload = value.to_numpy() if isinstance(value.dtype, np.dtype) else value.array
                    write_elem(group, row['destination'], payload)
                    group.attrs['column-order'] = np.array([*group.attrs['column-order'], row['destination']],
                        dtype=h5py.string_dtype())
                else:
                    if location != 'obsm' or 'obsm/' + field not in source:
                        raise ValueError('The selected accepted representation is unavailable.')
                    values = np.asarray(read_elem(source['obsm'][field]))
                    if (values.ndim != 2 or values.shape[0] != len(cells) or values.shape[1] < 1
                            or values.dtype.kind not in 'fi' or not np.isfinite(values).all()
                            or field == 'X_umap' and values.shape[1] != 2):
                        raise ValueError('Invalid selected accepted representation.')
                    if not group.attrs:
                        group.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
                    write_elem(group, row['destination'], values)
            uns = exported.require_group('uns')
            if not uns.attrs:
                uns.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
            if EXPORT_KEY in uns:
                raise ValueError('Original and composed export provenance keys collide.')
            from .session_state import canonical, digest
            selection_json = canonical(manifest['contents']).decode('utf-8')
            if len(selection_json.encode('utf-8')) > 65536:
                raise ValueError('Composed export provenance exceeds its bound.')
            # One portable bounded JSON value records all selected mappings and
            # methods, without duplicating deep HDF5 groups per component.
            metadata = dict(contract=COMPOSED_CONTRACT, selection_json=selection_json,
                recipe_handle=manifest.get('recipe_handle', digest(manifest['recipe_identity'])),
                original_source_sha256=manifest['original_source']['sha256'],
                authority_scope='Delivery derivative; does not issue scientific authority.',
                omitted_private_metadata_fields=len(omitted))
            write_elem(uns, EXPORT_KEY, metadata)
    destination.flush(); destination.seek(0)
    destination.seek(0, 2)
    if destination.tell() > max_bytes:
        raise ValueError('Prepared composed H5AD exceeds its admitted byte limit.')
    destination.seek(0)
    inspect(destination, privacy=True)
    destination.seek(0)
