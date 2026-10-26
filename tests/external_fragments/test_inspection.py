"""Simulated structure witnesses; no biological or preprocessing qualification."""
import gzip
import hashlib
from pathlib import Path
import pytest

from agent.tools.data.fragment_inspection import inspect_scATAC_fragments, verify_result, RECORD_LIMIT
from agent.tools.data.external_fragment_manifest import ExternalFragmentsError


def arguments(path):
    return dict(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('encoding', ['plain', 'gzip', 'bgzf'])
def test_no_reference_or_declarations_needed_and_no_authority(tmp_path, encoding):
    data = b'# structural example\nchrUn\t1\t7\tA-1\t3\nchrUn\t9\t15\tB-1\t2\n'
    source = tmp_path/'opaque'
    if encoding == 'bgzf':
        import pysam
        with pysam.BGZFile(str(source), 'wb') as f: f.write(data)
    else: source.write_bytes(gzip.compress(data) if encoding == 'gzip' else data)
    before = source.read_bytes()
    result = inspect_scATAC_fragments(**arguments(source))
    assert result['encoding'] == encoding
    assert result['n_records_observed'] == 2 and result['sum_support_observed'] == 5
    assert result['n_distinct_barcodes_observed'] == 2 and result['n_contigs_observed'] == 1
    assert result['reached_eof'] and result['index_status'] == 'not_supplied'
    assert not {'manifest_path', 'n_cells', 'species', 'assembly', 'authority'} & result.keys()
    assert sorted(tmp_path.iterdir()) == [source] and source.read_bytes() == before
    verify_result(arguments(source), result)
    with pytest.raises(ExternalFragmentsError):
        verify_result(arguments(source), dict(result, n_records_observed=3))


def test_prefix_does_not_qualify_unread_tail(tmp_path):
    source = tmp_path/'source'
    source.write_bytes(b'chr1\t0\t5\tA\t1\n' * RECORD_LIMIT + b'not inspected\n')
    result = inspect_scATAC_fragments(**arguments(source))
    assert result['n_records_observed'] == RECORD_LIMIT and not result['reached_eof']
    assert result['n_distinct_barcodes_observed'] == 1


@pytest.mark.parametrize('data', [b'', b'chr1\t5\t1\tA\t1\n', b'chr1\t0\t1\tA\t0\n',
    b'chr1\t0\t1\tA\t1', b'chr1\t0\t1\tA\t1\n# late\n',
    b'chr1\t0\t1\tA\t1\nchr1\t1\t2\tA\t1\t+\n'])
def test_malformed_observed_structure_fails(tmp_path, data):
    source = tmp_path/'source'; source.write_bytes(data)
    with pytest.raises(ValueError): inspect_scATAC_fragments(**arguments(source))


def test_exact_source_and_index_identity_without_false_pairing(source_factory):
    one = source_factory(encoding='bgzf', indexed=True)
    two = source_factory(encoding='bgzf', indexed=True, rows=[b'chr2\t10\t30\tOTHER\t9\n'])
    args = {k: one[k] for k in ('source_path', 'source_sha256', 'source_index_path', 'source_index_sha256')}
    result = inspect_scATAC_fragments(**args)
    assert result['index_status'] == 'tbi_header_only_pair_unverified'
    with pytest.raises(ValueError): inspect_scATAC_fragments(**dict(args, source_sha256='a'*64))
    with pytest.raises(ValueError): inspect_scATAC_fragments(**dict(args, source_index_path=two['source_index_path']))
    # A different structurally valid TBI can be described, but never certified as paired.
    other = inspect_scATAC_fragments(**dict(args, source_index_path=two['source_index_path'],
        source_index_sha256=two['source_index_sha256']))
    assert other['index_status'] == 'tbi_header_only_pair_unverified'
    assert other['source_index_sha256'] != result['source_index_sha256']


def test_structural_result_cannot_supply_canonical_or_qualified_input(monkeypatch):
    import json
    from agent.orchestration import LLMPlanner, PlannerError, build_default_tool_registry
    from agent.schemas import AgentRequest, RunMode
    from agent.tools.data import _external_fragment_io
    monkeypatch.setattr(_external_fragment_io, 'resource',
        lambda *a, **k: pytest.fail('Planning or failed compilation performed scientific IO'))
    def source(target, name): return dict(target=target, source=dict(kind='input', input=name))
    payload = dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id='inspect', tool='inspect_scATAC_fragments', control_dependencies=[],
             sources=[source('source', 'source_path')]),
        dict(step_id='import', tool='import_scATAC_fragments', control_dependencies=[], sources=[
            dict(target='source', source=dict(kind='step', step='inspect')),
            source('reference', 'reference_bundle_path'), source('source_profile', 'source_profile'),
            source('namespace', 'namespace')])]))
    class Model:
        model_id = 'scripted-test'
        def complete(self, **kwargs): return json.dumps(payload)
    request = AgentRequest('inspection-not-authority', 'Inspect then import.', dict(
        source_path='/unopened/source', source_sha256='1'*64,
        reference_bundle_path='/unopened/reference', reference_bundle_sha256='2'*64,
        source_profile='10x-atac-fragments.v1', namespace='sample'), RunMode.PLAN_ONLY)
    with pytest.raises(PlannerError): LLMPlanner(Model()).plan(request, build_default_tool_registry())
