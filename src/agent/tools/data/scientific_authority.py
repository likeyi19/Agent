"""Reviewed artifact adapters for the raw scientific authority DAG.

Descriptions only read artifact/resource metadata. Raw files are historical
identities after producer verification; current-source auditing is explicit.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os

from agent.schemas.verification_authority import AuthorityError, digest, VerifiedArtifactAuthority
from agent.schemas.orchestration import _serialize
from . import scatac_fragments_v2 as v2
from .fragments_authority_contract import CONTRACTS

TOOLS = {
    **{c.tool_name: k for k, c in CONTRACTS.items()},
    'compute_scATAC_qc': 'qc', 'select_scATAC_cells': 'selection',
    'build_scATAC_cell_by_ccre': 'matrix',
}
MODULES = {
    'fastq_fragment_production': 'scatac_fragments',
    'bam_fragment_production': 'scatac_bam_fragments',
    'external_fragment_adoption': 'scatac_fragment_import',
    'qc': 'scatac_barcode_qc', 'selection': 'scatac_cell_selection', 'matrix': 'scatac_matrix',
}


def _module(name):
    from importlib import import_module
    return import_module('.' + name, __package__)


def _resource_paths(value):
    """Traverse reviewed resource records, not arbitrary filesystem discovery."""
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            yield value['path']
        for item in value.values():
            yield from _resource_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _resource_paths(item)


def _reference(path, sha):
    from .scatac_reference import load_scatac_reference_bundle
    _, reference, _ = load_scatac_reference_bundle(path, expected_sha256=sha)
    return [str(path), *_resource_paths(asdict(reference))]


def _load(kind, path, sha):
    with path.open('rb') as stream:
        raw = stream.read(v2.MAX_BYTES + 1)
    if len(raw) > v2.MAX_BYTES:
        raise AuthorityError('Scientific manifest exceeds its resource bound.')
    actual = hashlib.sha256(raw).hexdigest()
    if sha is not None and actual != sha:
        raise AuthorityError('Manifest differs from the requested artifact identity.')
    if kind in CONTRACTS or kind == 'generic_fragments':
        value = v2.load_fragments_manifest_v2(path, expected_sha256=actual)
    elif kind == 'qc':
        from ._barcode_qc_contract import load_manifest
        value = load_manifest(path, actual).to_dict()
    elif kind == 'selection':
        from ._cell_selection_contract import load_manifest
        value = load_manifest(path, actual).to_dict()
    elif kind == 'matrix':
        from .scatac_matrix_contract import load_manifest_bytes
        value = load_manifest_bytes(raw)
    else:
        raise AuthorityError('Unsupported scientific authority contract.')
    return value, actual


def describe(kind, path, sha, context):
    path = Path(path)
    if path != path.resolve() or not path.is_file():
        raise AuthorityError('Unsafe scientific publication locator.')
    value, actual = _load(kind, path, sha)
    root = path.parent
    normal = deepcopy(value)
    owned = [path]
    resources = []
    sources = []
    dependencies = {}
    qualification = {}

    def dependency(label, dependency_kind, name, expected):
        described, identity = context.describe(dependency_kind, Path(name), expected)
        dependencies[label] = dict(kind=dependency_kind, manifest_path=str(name),
            manifest_sha256=expected, scientific_proof_sha256=identity,
            authority_sha256=context.accepted.get((str(name), expected), {}).get('identity'),
            required_scope='scientific_correctness.v1')
        resources.extend(f['path'] for f in described['files'])
        sources.extend(described['historical_sources'])

    if kind in CONTRACTS or kind == 'generic_fragments':
        kinds = {e['provenance']['kind'] for e in value['libraries']}
        if len(kinds) != 1 or not kinds <= CONTRACTS.keys():
            raise AuthorityError('No compatible producer qualification contract.')
        producer_kind = next(iter(kinds))
        if kind != 'generic_fragments' and producer_kind != kind:
            raise AuthorityError('Cross-producer scientific authority substitution.')
        contract = CONTRACTS[producer_kind]
        qualification = contract.qualification if kind != 'generic_fragments' else {}
        profile = contract.profile_sha256
        rp = value['reference']
        resources.extend(_reference(rp['manifest_path'], rp['manifest_sha256']))
        for entry, normalized in zip(value['libraries'], normal['libraries'], strict=True):
            p = entry['provenance']
            for key in ('bgzf', 'tabix'):
                owned.append(root / entry[key]['path'])
            for key, resource in (('profile', p['profile']['resource']), ('producer_record', p['producer_record'])):
                locator = Path(resource['path'])
                if locator.parent != root:
                    raise AuthorityError('Producer-owned resource escaped the publication.')
                owned.append(locator)
                target = normalized['provenance']['profile']['resource'] if key == 'profile' else normalized['provenance'][key]
                target['path'] = '@owned/' + locator.name
            if p['profile']['id'] != contract.profile_id or p['profile']['resource']['sha256'] != profile:
                raise AuthorityError('Incompatible producer science profile.')
        # Source differences are handled only by the reviewed producer adapter.
        record_path = Path(value['libraries'][0]['provenance']['producer_record']['path'])
        with record_path.open('rb') as stream:
            raw_record = stream.read(v2.MAX_BYTES + 1)
        if len(raw_record) > v2.MAX_BYTES:
            raise AuthorityError('Producer record exceeds its resource bound.')
        record = json.loads(raw_record, object_pairs_hook=v2._pairs)
        if producer_kind == 'fastq_fragment_production':
            from .scatac_library_context import load_scatac_library_processing_context
            from .chromap_reference_index import load_chromap_reference_index
            inputs = record['inputs']
            resources.extend(inputs[k] for k in ('intake_path', 'context_path', 'index_path'))
            _, library_context, _ = load_scatac_library_processing_context(inputs['context_path'], expected_sha256=inputs['context_sha256'])
            resources.extend(_resource_paths(asdict(library_context)))
            index = load_chromap_reference_index(inputs['index_path'], expected_sha256=inputs['index_sha256'])
            resources.append(str(Path(inputs['index_path']).parent / index.index_file))
            sources.extend(s for library in record['libraries'] for s in library['sources'])
        elif producer_kind == 'bam_fragment_production':
            sources.append(record['source'])
            resources.extend(record[k]['path'] for k in ('intake', 'context'))
        else:
            sources.append(record['source']['resource'])
            if record['source_index'] is not None:
                sources.append(record['source_index'])
        verifier = contract.verifier if kind != 'generic_fragments' else dict(id='agent.canonical-fragments', compatibility_version='1')
    elif kind == 'qc':
        from .scatac_qc_reference import load_scatac_qc_reference_bundle
        from .scatac_qc_profile import PROFILE_SHA256
        args = value['arguments']
        fp = args['fragments_manifest_path']; fs = args['fragments_manifest_sha256']
        fragments, _ = _load('generic_fragments', Path(fp), fs)
        producer_kind = fragments['libraries'][0]['provenance']['kind']
        dependency('fragments', producer_kind, fp, fs)
        owned.extend(root / value[k]['path'] for k in ('table', 'histogram'))
        qr = args['qc_reference_manifest_path']
        _, bundle, _ = load_scatac_qc_reference_bundle(qr, expected_sha256=args['qc_reference_manifest_sha256'])
        resources.extend([qr, *_resource_paths(asdict(bundle))])
        resources.extend(_reference(bundle.parent_manifest.path, bundle.parent_manifest.sha256))
        if os.environ.get('AGENT_QC_RESOURCE_CATALOG'):
            resources.append(os.environ['AGENT_QC_RESOURCE_CATALOG'])
        profile = PROFILE_SHA256
        verifier = dict(id='agent.barcode-qc-independent', compatibility_version='1')
    elif kind == 'selection':
        from .scatac_selection_profile import PROFILE_SHA256
        args = value['arguments']
        dependency('qc', 'qc', args['barcode_qc_manifest_path'], args['barcode_qc_manifest_sha256'])
        owned.extend(root / value[k]['path'] for k in ('decisions', 'selected'))
        profile = PROFILE_SHA256
        verifier = dict(id='agent.cell-selection-independent', compatibility_version='1')
    else:
        from .scatac_matrix_contract import PROFILE_SHA256
        fp, sp, rp = (value['upstream'][k] for k in ('fragments', 'selection', 'reference'))
        fragments, _ = _load('generic_fragments', Path(fp['manifest_path']), fp['manifest_sha256'])
        dependency('fragments', fragments['libraries'][0]['provenance']['kind'], fp['manifest_path'], fp['manifest_sha256'])
        dependency('selection', 'selection', sp['manifest_path'], sp['manifest_sha256'])
        resources.extend(_reference(rp['manifest_path'], rp['manifest_sha256']))
        owned.append(root / value['matrix']['path'])
        profile = PROFILE_SHA256
        verifier = dict(id='agent.cell-by-ccre-independent', compatibility_version='1')
    historical = {f['path']: dict(f) for f in sources}
    # Historical raw identities must never leak into ordinary resource validation.
    resources = [p for p in resources if str(p) not in historical]
    files = context.files([*owned, *resources])
    relative = []
    for entry in files:
        if entry['path'] == str(path):
            continue  # only reviewed owned locator fields differ on publication
        item = dict(entry)
        if Path(item['path']).is_relative_to(root) and Path(item['path']) in map(Path, owned):
            item['path'] = '@owned/' + str(Path(item['path']).relative_to(root))
        relative.append(item)
    identity = dict(kind=kind, manifest=normal, files=sorted(relative, key=lambda f:f['path']),
                    execution_identity=context.execution_for(kind,path),
                    upstream=dependencies, historical_sources=sorted(historical.values(), key=lambda f:f['path']),
                    profile=profile, verifier=verifier, qualification=qualification,
                    policy='historical_verified_sources.v1')
    return dict(kind=kind, manifest_path=str(path), manifest_sha256=actual, manifest=value,
                proof_identity=identity, files=files, historical_sources=identity['historical_sources'],
                upstream=dependencies, profile=profile, verifier=verifier, qualification=qualification)


def runtime_compatibility(kind, description, kwargs):
    if kind in CONTRACTS or kind == 'generic_fragments':
        if kind == 'fastq_fragment_production':
            from .fastq_fragments_verifier import verify_packaging
        elif kind == 'generic_fragments':
            from .scatac_fragments_v2_verifier import verify_packaging
        else:
            from ._external_fragment_io import verify_packaging
        verify_packaging(kwargs['runtime'])
        return
    if kind == 'qc':
        from ._barcode_qc_binding import backend_runtime, resource_qualification
        from .scatac_qc_reference import load_scatac_qc_reference_bundle
        value = description['manifest']; args = value['arguments']
        supplied = kwargs.get('fragments_authority')
        if supplied is not None:
            from .fragments_authority_contract import StoredFragmentAuthority
            if type(supplied) is not StoredFragmentAuthority:
                raise AuthorityError('QC requires an issued dependency authority.')
            supplied.validate(args['fragments_manifest_path'], args['fragments_manifest_sha256'],
                              producer_kind=description['upstream']['fragments']['kind'])
        _, bundle, _ = load_scatac_qc_reference_bundle(args['qc_reference_manifest_path'], expected_sha256=args['qc_reference_manifest_sha256'])
        if backend_runtime()[1] != value['backend_identity'] or resource_qualification(bundle) != value['resource_qualification']:
            raise AuthorityError('QC runtime/resource qualification changed.')
    if kind == 'matrix':
        from . import _matrix_bedtools as bed
        if bed.qualify_runtime(kwargs['bedtools_path']) != description['manifest']['backend']:
            raise AuthorityError('Matrix runtime qualification changed.')


def reuse_result(kind, description, proof, kwargs):
    path = Path(description['manifest_path']); sha = description['manifest_sha256']
    if kind == 'fastq_fragment_production':
        return deepcopy(description['manifest'])
    if kind in ('generic_fragments', 'bam_fragment_production', 'external_fragment_adoption'):
        from .scatac_fragments_v2_verifier import FragmentVerification, take_snapshots
        fragment = FragmentVerification(str(path), sha, path.read_bytes(),
            tuple(tuple(c) for c in proof.result_metadata['contigs']),
            take_snapshots(f['path'] for f in description['files']))
        if kind == 'generic_fragments': return fragment
        if kind == 'bam_fragment_production':
            from .bam_fragments_verifier import BamProductionVerification
            return BamProductionVerification(fragment)
        from .external_fragments_verifier import ExternalAdoptionVerification
        return ExternalAdoptionVerification(fragment)
    if kind in ('qc', 'selection'):
        module = _module('_barcode_qc_contract' if kind == 'qc' else '_cell_selection_contract')
        return module.load_manifest(path, sha)
    return _serialize(proof.result_metadata) | dict(manifest_path=str(path), manifest_sha256=sha,
        verification_seconds=0.0, verification_scratch_peak_bytes=0)


def issue(context, tool_name, arguments, result, execution_identity):
    kind = TOOLS[tool_name]
    _, key = context.describe(kind, Path(result['manifest_path']), result['manifest_sha256'])
    if key not in context.proofs:
        raise AuthorityError('No completed independent scientific proof.')
    return _publication_record(context, tool_name, arguments, result, execution_identity,
                               context.proofs[key].result_metadata)


def _publication_record(context, tool_name, arguments, result, execution_identity, result_metadata):
    """Pure provenance description, not a trust-issuing entry point."""
    kind = TOOLS[tool_name]; module = _module(MODULES[kind])
    publication = module._publication(arguments, execution_identity)
    destination, token = publication[:2] if kind == 'fastq_fragment_production' else publication[1:]
    normalized_arguments = arguments if kind == 'fastq_fragment_production' else publication[0]
    directory = 'fragments' if kind in CONTRACTS else 'artifact' if kind == 'matrix' else ''
    path = destination / directory / 'manifest.json'
    if str(path) != result['manifest_path']:
        raise AuthorityError('Publication belongs to a different execution.')
    receipt = (module._read_receipt if kind == 'fastq_fragment_production' else module._receipt)(destination, normalized_arguments, token)
    if receipt['manifest_sha256'] != result['manifest_sha256']:
        raise AuthorityError('Publication receipt identity mismatch.')
    description, key = context.describe(kind, path, result['manifest_sha256'])
    files = context.files([*(f['path'] for f in description['files']), destination / 'receipt.json'])
    record = dict(schema_version=2, artifact_type=result['artifact_type'], artifact_contract=result['contract_version'],
        publication_path=str(path), manifest_sha256=result['manifest_sha256'], files=files,
        execution_identity=execution_identity, arguments_sha256=receipt['arguments_sha256'],
        receipt_sha256=next(f['sha256'] for f in files if f['path']==str(destination/'receipt.json')),
        upstream=description['upstream'] or {'producer_inputs': digest(description['proof_identity'])},
        resources={'scientific_proof_sha256': key, 'manifest_identity': description['manifest'],
                   'result_metadata': _serialize(result_metadata)},
        science_profile=description['profile'], producer_qualification=description['qualification'],
        verifier=description['verifier'], scope='scientific_correctness.v1', completion='succeeded',
        integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1', historical_sources=description['historical_sources'])
    return VerifiedArtifactAuthority(record)
