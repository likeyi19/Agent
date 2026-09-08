import hashlib
from pathlib import Path

import pytest

from agent.tools.data import raw_scatac as raw
from agent.tools.data import raw_scatac_manifest as m
from agent.orchestration.registry import build_default_tool_registry, ToolResultContractError
from agent.schemas import ErrorCategory, RecoveryDisposition


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('selection', ['directory', 'single', 'tuple', 'list'])
def test_dispatch_publication_and_compact_summary(raw_factory, kind, selection):
    args = raw_factory(kind)
    folder = Path(args['raw_input_paths'])
    paths = sorted(folder.iterdir())
    if selection != 'directory':
        args['raw_input_paths'] = str(paths[0]) if selection == 'single' else (
            tuple(str(p) for p in paths) if selection == 'tuple' else [str(p) for p in paths])
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    result = raw.inspect_raw_scATAC(**args)
    build_default_tool_registry().validate_result('inspect_raw_scATAC', result)
    path, manifest, digest = m.load_raw_intake_manifest(result['manifest_path'])
    assert result == raw._summary(manifest, path, digest)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.name == f'raw-scatac-intake-{digest}.json'
    assert result['input_kind'] == kind and result['status'] == 'success'
    assert set(result) == set(raw.RawScATACInspection.__annotations__)
    assert 'PRIVATE_BARCODE' not in str(result) + path.read_text()
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in folder.iterdir()}
    snapshot = (path.read_bytes(), path.stat().st_mtime_ns)
    assert raw.inspect_raw_scATAC(**args) == result
    assert (path.read_bytes(), path.stat().st_mtime_ns) == snapshot
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize('species', ['human', 'mouse', 'rat'])
@pytest.mark.parametrize('kind', ['fastq', 'bam'])
def test_species_and_truthful_unsupported(raw_factory, species, kind):
    args = raw_factory(kind, species=species)
    result = raw.inspect_raw_scATAC(**args)
    assert result['status'] == 'success'
    assert result['readiness'] == ('UNSUPPORTED' if species == 'rat' else 'READY')


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('state', ['READY', 'NEEDS_USER_INPUT', 'INVALID'])
def test_readiness_is_not_execution_status(raw_factory, kind, state):
    args = raw_factory(kind, malformed=state == 'INVALID')
    if state == 'NEEDS_USER_INPUT':
        args['raw_assay'] = None
    result = raw.inspect_raw_scATAC(**args)
    assert result['readiness'] == state and result['status'] == 'success'


@pytest.mark.parametrize('kind,updates', [
    ('fastq', {'source_genome_assembly': 'hg38'}), ('fastq', {'raw_assay': 'SCATAC'}),
    ('bam', {'fastq_layout': 'tenx-atac-r1-r2-r3.v1'}),
    ('fastq', {'raw_assay': None, 'fastq_layout': 'tenx-atac-r1-r2-r3.v1'}),
    ('fastq', {'raw_assay': 'guessed'}), ('bam', {'fastq_layout': 'A'}),
    ('bam', {'species': 'Mouse'}), ('bam', {'source_genome_assembly': 42}),
])
def test_declaration_incompatibility_is_not_guessed(raw_factory, kind, updates):
    args = {**raw_factory(kind), **updates}
    with pytest.raises(ValueError) as exc:
        raw.inspect_raw_scATAC(**args)
    assert 'DECLARATION' in exc.value.code
    assert not Path(args['output_dir']).exists()


def test_bam_scatac_and_explicit_source(raw_factory):
    args = raw_factory('bam')
    args.update(raw_assay='SCATAC', source_genome_assembly='GRCh38')
    assert raw.inspect_raw_scATAC(**args)['readiness'] == 'READY'


def test_explicit_fastq_layout(raw_factory):
    args = raw_factory()
    args['fastq_layout'] = 'tenx-atac-r1-r2-r3.v1'
    assert raw.inspect_raw_scATAC(**args)['readiness'] == 'READY'


def test_mixed_selection_rejected_before_inspection(raw_factory, monkeypatch):
    a, b = raw_factory(), raw_factory('bam', root='bam')
    monkeypatch.setattr(raw.fastq, 'inspect_fastq_inputs', lambda *a, **k: pytest.fail('parsed mixed selection'))
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_KIND_MIXED'):
        raw.inspect_raw_scATAC((a['raw_input_paths'], b['raw_input_paths']), a['output_dir'])


@pytest.mark.parametrize('selection', [None, (), (42,), 'https://host/input.bam', {'path': 'x'}])
def test_invalid_selection(tmp_path, selection):
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_SELECTION_INVALID'):
        raw.inspect_raw_scATAC(selection, tmp_path / 'out')


def test_empty_directory(tmp_path):
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_NO_CANDIDATES'):
        raw.inspect_raw_scATAC(tmp_path, tmp_path / 'out')


def test_dispatch_bounds_before_any_content_inspection(raw_factory, monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    args = raw_factory()
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_SELECTION_INVALID'):
        raw.inspect_raw_scATAC([args['raw_input_paths']] * 129, args['output_dir'])
    entries = (SimpleNamespace(name='unrelated') for _ in range(16385))
    monkeypatch.setattr(raw.os, 'scandir', lambda path: nullcontext(entries))
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_DISCOVERY_LIMIT'):
        raw.inspect_raw_scATAC(**args)


def test_mixed_kinds_in_same_directory_are_rejected(raw_factory):
    args = raw_factory()
    Path(args['raw_input_paths'], 'extra.bam').write_bytes(b'not inspected')
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_KIND_MIXED'):
        raw.inspect_raw_scATAC(**args)


def test_new_candidate_during_dispatch_is_source_integrity_failure(raw_factory, monkeypatch):
    args = raw_factory()
    original = raw.fastq.inspect_fastq_inputs
    def changed(*a, **kw):
        manifest = original(*a, **kw)
        Path(args['raw_input_paths'], 'extra.fastq').write_bytes(b'@r\nA\n+\nI\n')
        return manifest
    monkeypatch.setattr(raw.fastq, 'inspect_fastq_inputs', changed)
    with pytest.raises(raw.RawScATACError, match='RAW_INPUT_SOURCE_CHANGED'):
        raw.inspect_raw_scATAC(**args)


def test_permission_failure_is_sanitized_resource_error(raw_factory, monkeypatch):
    args = raw_factory()
    def denied(*args):
        raise PermissionError('PRIVATE filesystem path')
    monkeypatch.setattr(raw.os, 'scandir', denied)
    with pytest.raises(raw.RawScATACError) as exc:
        raw.inspect_raw_scATAC(**args)
    error = build_default_tool_registry().classify_exception('inspect_raw_scATAC', exc.value)
    assert error.category is ErrorCategory.RESOURCE_ERROR
    assert error.code == 'RAW_INPUT_ACCESS_FAILED'
    assert 'PRIVATE' not in str(exc.value) + error.message


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
def test_multiple_independent_groups_and_aliases(raw_factory, kind):
    args = raw_factory(kind)
    second = raw_factory(kind, root='second')
    args['raw_input_paths'] = (args['raw_input_paths'], second['raw_input_paths'], args['raw_input_paths'])
    result = raw.inspect_raw_scATAC(**args)
    assert result['n_groups'] == 2
    assert result == raw.inspect_raw_scATAC(**{**args, 'raw_input_paths': list(reversed(args['raw_input_paths']))})


def test_source_directory_cannot_be_publication_directory(raw_factory):
    args = raw_factory()
    args['output_dir'] = args['raw_input_paths']
    before = sorted(Path(args['raw_input_paths']).iterdir())
    with pytest.raises(raw.RawScATACError, match='RAW_INTAKE_OUTPUT_CONFLICT'):
        raw.inspect_raw_scATAC(**args)
    assert sorted(Path(args['raw_input_paths']).iterdir()) == before


def test_conflicting_destination_is_preserved(raw_factory):
    args = raw_factory()
    result = raw.inspect_raw_scATAC(**args)
    path = Path(result['manifest_path'])
    path.write_bytes(b'unrelated artifact')
    with pytest.raises(raw.RawScATACError, match='RAW_INTAKE_OUTPUT_CONFLICT'):
        raw.inspect_raw_scATAC(**args)
    assert path.read_bytes() == b'unrelated artifact'


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'), ('readiness', 'unknown'), ('input_kind', 'h5ad'),
    ('artifact_schema_version', True), ('artifact_type', 'wrong'),
    ('intake_contract_version', 'wrong'), ('manifest_sha256', 'a' * 63),
    ('n_files', 0), ('n_groups', 129), ('n_issues', -1), ('n_repairs', True),
    ('manifest', {'large': 'object'}), ('manifest_path', 'relative.json'),
])
def test_result_contract_rejects_forgery(raw_factory, field, value):
    result = raw.inspect_raw_scATAC(**raw_factory())
    result[field] = value
    with pytest.raises(ToolResultContractError):
        build_default_tool_registry().validate_result('inspect_raw_scATAC', result)


@pytest.mark.parametrize('code,category', [
    ('RAW_INPUT_KIND_MIXED', ErrorCategory.USER_INPUT_ERROR),
    ('RAW_BAM_DEPENDENCY_UNAVAILABLE', ErrorCategory.ENVIRONMENT_ERROR),
    ('RAW_BAM_BACKEND_VERSION_UNSUPPORTED', ErrorCategory.ENVIRONMENT_ERROR),
    ('RAW_INPUT_SOURCE_CHANGED', ErrorCategory.VERIFICATION_ERROR),
    ('RAW_INTAKE_OUTPUT_CONFLICT', ErrorCategory.RESOURCE_ERROR),
    ('RAW_INTAKE_WRITE_FAILED', ErrorCategory.RESOURCE_ERROR),
])
def test_error_classification_is_stable_and_nonretryable(code, category):
    error = build_default_tool_registry().classify_exception('inspect_raw_scATAC', raw.RawScATACError(code))
    assert error.code == code and error.category is category and not error.recoverable
    assert error.recovery_disposition is not RecoveryDisposition.SAME_STEP_RETRY_ELIGIBLE
