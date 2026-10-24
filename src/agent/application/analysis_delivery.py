"""Closed safe exports of pinned embeddings, Leiden and UMAP results."""
import os
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
EXPORT_KEY = 'agent_scientific_export'


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
    """Project existing eligible cell-analysis downloads without preparing bytes.

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
        from agent.tools.analysis.embedding_analysis import scientific_methods
        methods = scientific_methods(handle)
        for key, versions in manifest.get('method_software_versions', {}).items():
            if key in methods:
                methods[key]['software_versions'] = versions
        for key, record in manifest.get('recorded_execution', {}).items():
            if key in methods:
                methods[key].update(record)
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
    from .annotation_delivery import _copy, _redactions
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
