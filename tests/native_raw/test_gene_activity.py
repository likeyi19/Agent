"""ICTA-2A: genuine native constructors and their shared publication boundary."""
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis.native_matrix import EPISCANPY, SNAPATAC2, SCANPY, NativeInvocationError
from agent.tools.analysis.native_gene_activity import axis_identity
from agent.schemas.verification_authority import AuthorityError


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def sample(tmp_path):
    gtf = tmp_path / 'genes.gtf'
    rows = []
    for kind, gid, tid, name, start, end, strand in (
        ('gene', 'G1', 'T1', 'Alpha', 101, 200, '+'),
        ('transcript', 'G1', 'T1', 'Alpha', 101, 200, '+'),
        ('transcript', 'G1', 'T1b', 'Alpha', 141, 200, '+'),
        ('gene', 'G2', 'T2', 'Beta', 801, 1000, '-'),
        ('transcript', 'G2', 'T2', 'Beta', 801, 1000, '-')):
        rows.append(f'chr1\tHAVANA\t{kind}\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gid}"; transcript_id "{tid}"; gene_name "{name}"; transcript_name "{tid}-name";\n')
    gtf.write_text(''.join(rows))
    fragments = tmp_path / 'fragments.tsv'
    fragments.write_text('chr1\t100\t120\tA\t1\nchr1\t150\t180\tA\t1\nchr1\t900\t920\tB\t1\n')
    peaks = tmp_path / 'peaks.h5ad'
    a = ad.AnnData(sparse.csr_matrix([[2, 0], [0, 3]], dtype=np.int32),
        obs=pd.DataFrame({'group': ['a', 'b']}, index=['A', 'B']),
        var=pd.DataFrame(index=['chr1:100-120', 'chr1:900-920']))
    a.uns.update(matrix_semantics='peak_counts', species='human', assembly='fixture', experiment='retained')
    a.obsm['X_peak'] = np.array([[1., 2.], [3., 4.]])
    a.obsp['peak_graph'] = sparse.eye(2, format='csr')
    a.write_h5ad(peaks)
    return dict(root=tmp_path, gtf=gtf, fragments=fragments, peaks=peaks)


@pytest.fixture
def native():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native runtime qualification')


def invoke(sample, owner, name, source, label, **kw):
    return owner.execute(dict(input_path=str(source), input_sha256=sha(source),
        output_dir=str(sample['root'] / label), function_name=name, **kw), 'icta2a-' + label)


def activity(sample, owner, source, label, **kwargs):
    name = 'pp.make_gene_matrix' if owner is SNAPATAC2 else 'tl.geneactivity'
    return invoke(sample, owner, name, source, label, gene_anno_path=str(sample['gtf']),
        gene_anno_sha256=sha(sample['gtf']), kwargs=kwargs)


def imported(sample):
    return invoke(sample, SNAPATAC2, 'pp.import_fragments', sample['fragments'], 'import',
        kwargs=dict(chrom_sizes={'chr1': 2000}, min_num_fragments=0), species='human', assembly='fixture')


def values(result):
    a = ad.read_h5ad(result['matrix_path'])
    return a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)


def test_catalog_and_function_scoped_resource_port(sample):
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.native_parameters import input_name
    registry = build_default_tool_registry()
    for tool, fn in [('run_snapatac2', 'pp.make_gene_matrix'), ('run_episcanpy', 'tl.geneactivity')]:
        spec = registry.get(tool)
        assert fn in dict(spec.native_functions.functions)
        port = next(p for p in spec.semantic_planning.consumer_ports if p.name == 'gene_anno')
        assert any(r.selector == input_name(tool, fn, 'gene_anno_path') for r in port.request_sources)
    for owner, fn in [(SNAPATAC2, 'pp.make_gene_matrix'), (EPISCANPY, 'tl.geneactivity')]:
        with pytest.raises(ValueError, match='gene_anno'):
            owner.native_arguments(dict(function_name=fn, input_path=str(sample['peaks'])))
        with pytest.raises(ValueError):
            owner.native_arguments(dict(function_name=fn, input_path=str(sample['peaks']), kwargs={'gtf_file': str(sample['gtf'])}))


def test_annotation_pin_precedes_execution(sample, monkeypatch):
    monkeypatch.setattr(EPISCANPY, 'native', lambda *a: pytest.fail('Changed annotation reached science'))
    with pytest.raises(AuthorityError, match='annotation bytes'):
        EPISCANPY._execute(dict(function_name='tl.geneactivity', input_path=str(sample['peaks']),
            output_dir=str(sample['root']/'changed'), gene_anno_path=str(sample['gtf']), gene_anno_sha256='0'*64))


def test_native_snap_counts_transcripts_use_x_and_source_preservation(native, sample):
    source = imported(sample)
    before = sha(source['matrix_path'])
    original = ad.read_h5ad(source['matrix_path'])
    assert 'fragment_paired' in original.obsm
    for strategy, expected in [('paired-insertion', [[2, 0], [0, 1]]), ('insertion', [[4, 0], [0, 2]]), ('fragment', [[2, 0], [0, 1]])]:
        out = activity(sample, SNAPATAC2, source['matrix_path'], strategy, upstream=0, counting_strategy=strategy)
        np.testing.assert_array_equal(values(out), expected)
        a = ad.read_h5ad(out['matrix_path'])
        assert a.obs_names.tolist() == ['A', 'B'] and a.var_names.tolist() == ['Alpha', 'Beta']
        assert not set(('chrom', 'start', 'end', 'gene_id')) & set(a.var)
        assert 'fragment_paired' not in a.obsm
        assert out['parameters']['gene_activity']['feature_type'] == 'gene'
        assert out['source_identity']['references'] == [dict(role='gene_anno', sha256=sha(sample['gtf']))]
        assert out['parameters']['effective_native_parameters']['counting_strategy'] == strategy
    tx = activity(sample, SNAPATAC2, source['matrix_path'], 'transcripts', upstream=0, id_type='transcript', backend=None)
    a = ad.read_h5ad(tx['matrix_path'])
    assert a.var_names.tolist() == ['T1', 'T1b', 'T2']
    np.testing.assert_array_equal(values(tx), [[2, 1, 0], [0, 0, 1]])
    aggregated = activity(sample, SNAPATAC2, sample['peaks'], 'use-x', upstream=0, use_x=True)
    np.testing.assert_array_equal(values(aggregated), [[2, 0], [0, 3]])
    desc = aggregated['parameters']['gene_activity']
    assert desc['source_slot'] == 'X' and desc['source_measurement'] == 'peak_counts'
    assert aggregated['measurement'] == 'snapatac2_gene_region_sum'
    assert sha(source['matrix_path']) == before
    np.testing.assert_array_equal(ad.read_h5ad(source['matrix_path']).obsm['fragment_paired'].data, original.obsm['fragment_paired'].data)


def test_native_snap_omitted_defaults_inplace_and_bad_representation(native, sample):
    source = imported(sample)
    out = activity(sample, SNAPATAC2, source['matrix_path'], 'defaults')
    effective = out['parameters']['effective_native_parameters']
    assert {k: effective[k] for k in ('upstream', 'downstream', 'id_type', 'include_gene_body', 'use_x', 'inplace')} == dict(
        upstream=2000, downstream=0, id_type='gene', include_gene_body=True, use_x=False, inplace=False)
    inplace = activity(sample, SNAPATAC2, source['matrix_path'], 'inplace', inplace=True, upstream=0)
    np.testing.assert_array_equal(values(inplace), [[2, 0], [0, 1]])
    with pytest.raises(NativeInvocationError):
        activity(sample, SNAPATAC2, sample['peaks'], 'no-fragments', upstream=0)
    assert not (sample['root']/'no-fragments/manifest.json').exists()
    assert (sample['root']/'no-fragments/native-failure.log').is_file()


@pytest.mark.parametrize('binary,coordinates', [(False, 'colon'), (True, 'underscore')])
def test_native_epi_supplied_measurement_dense_axis_and_inherited_slots(native, sample, binary, coordinates):
    a = ad.read_h5ad(sample['peaks'])
    if binary:
        a.X.data[:] = 1
        a.uns['matrix_semantics'] = 'binary_accessibility'
    if coordinates == 'underscore': a.var_names = ['chr1_100_120', 'chr1_900_920']
    a.write_h5ad(sample['peaks']); before = sha(sample['peaks'])
    out = activity(sample, EPISCANPY, sample['peaks'], 'epi', upstream=0)
    b = ad.read_h5ad(out['matrix_path'])
    assert isinstance(b.X, np.ndarray)
    np.testing.assert_array_equal(b.X, [[1, 0], [0, 1]] if binary else [[2, 0], [0, 3]])
    assert b.obs_names.tolist() == a.obs_names.tolist() and b.var_names.tolist() == ['Alpha', 'Beta']
    assert b.var.gene_id.tolist() == ['G1', 'G2']
    desc = out['parameters']['gene_activity']
    assert desc['source_measurement'] == a.uns['matrix_semantics']
    assert desc['axes'] == axis_identity(out['matrix_path'])
    assert {'obsm.X_peak', 'obsp.peak_graph', 'uns.experiment'} <= set(desc['inherited_slots'])
    assert not {'obsm.X_peak', 'obsp.peak_graph'} & set(out['result_slots'])
    np.testing.assert_array_equal(b.obsm['X_peak'], a.obsm['X_peak'])
    assert (b.obsp['peak_graph'] != a.obsp['peak_graph']).nnz == 0
    assert sha(sample['peaks']) == before


def test_native_epi_strand_windows_annotation_defaults_and_duplicate_names(native, sample):
    a = ad.read_h5ad(sample['peaks']); a.var_names = ['chr1:50-51', 'chr1:1050-1051']; a.write_h5ad(sample['peaks'])
    out = activity(sample, EPISCANPY, sample['peaks'], 'windows', upstream=50)
    np.testing.assert_array_equal(values(out), [[2, 0], [0, 3]])
    sample['gtf'].write_text(sample['gtf'].read_text().replace('HAVANA', 'ENSEMBL'))
    with pytest.raises(NativeInvocationError): activity(sample, EPISCANPY, sample['peaks'], 'havana-empty')
    assert not (sample['root']/'havana-empty/manifest.json').exists()
    out = activity(sample, EPISCANPY, sample['peaks'], 'all-sources', annotation=None)
    assert out['parameters']['effective_native_parameters']['upstream'] == 5000
    assert out['parameters']['effective_native_parameters']['annotation'] is None
    sample['gtf'].write_text(sample['gtf'].read_text().replace('Beta', 'Alpha'))
    out = activity(sample, EPISCANPY, sample['peaks'], 'duplicates', annotation='ENSEMBL', upstream=50, copy=False, layer_name='ignored')
    b = ad.read_h5ad(out['matrix_path'])
    assert b.var_names.tolist() == ['Alpha', 'Alpha'] and b.var.gene_id.tolist() == ['G1', 'G2']
    assert not b.layers


def test_native_epi_raw_exact_axis_and_missing_identifiers(native, sample):
    a = ad.read_h5ad(sample['peaks']); a.raw = a.copy(); a.X.data[:] = 1; a.write_h5ad(sample['peaks'])
    out = activity(sample, EPISCANPY, sample['peaks'], 'raw', raw=True, upstream=0)
    np.testing.assert_array_equal(values(out), [[2, 0], [0, 3]])
    assert out['parameters']['gene_activity']['source_slot'] == 'raw.X'
    a.var_names = a.var_names[::-1]; a.write_h5ad(sample['peaks'])
    out = activity(sample, EPISCANPY, sample['peaks'], 'different-raw', raw=True, upstream=0)
    np.testing.assert_array_equal(values(out), [[0, 2], [3, 0]])
    assert out['parameters']['gene_activity']['raw_current_features_equal'] is False
    sample['gtf'].write_text(sample['gtf'].read_text().replace('gene_name', 'display').replace('gene_id', 'identifier'))
    out = activity(sample, EPISCANPY, sample['peaks'], 'missing-ids', upstream=0)
    b = ad.read_h5ad(out['matrix_path'])
    assert b.var_names.tolist() == ['0', '1']
    assert b.var.index.name is None and 'gene_id' not in b.var
    assert b.var['display'].tolist() == ['Alpha', 'Beta']


@pytest.mark.parametrize('owner', [SNAPATAC2, EPISCANPY], ids=['snapatac2', 'episcanpy'])
def test_native_common_contract_scanpy_and_read_only_recovery(native, sample, owner, monkeypatch):
    source = imported(sample)['matrix_path'] if owner is SNAPATAC2 else sample['peaks']
    original_sha = sha(source)
    out = activity(sample, owner, source, 'activity', upstream=0)
    name = 'pp.make_gene_matrix' if owner is SNAPATAC2 else 'tl.geneactivity'
    args = dict(function_name=name, input_path=str(source), input_sha256=original_sha, output_dir=str(sample['root']/'activity'),
        gene_anno_path=str(sample['gtf']), gene_anno_sha256=sha(sample['gtf']), kwargs=dict(upstream=0))
    descriptor = out['parameters']['gene_activity']
    assert descriptor['interface'] == 'gene-activity.v1' and descriptor['matrix_slot'] == 'X'
    assert descriptor['declarations'] == dict(species='human', assembly='fixture')
    downstream = invoke(sample, SCANPY, 'pp.log1p', out['matrix_path'], 'log1p', kwargs=dict(copy=True))
    np.testing.assert_allclose(values(downstream), np.log1p(values(out)))
    assert axis_identity(downstream['matrix_path']) == axis_identity(out['matrix_path'])
    assert downstream['source_identity']['measurement'] == out['measurement']
    monkeypatch.setattr(owner, 'native', lambda *a: pytest.fail('Recovery repeated science'))
    assert owner.recover(args, 'icta2a-activity') == out
    assert sha(source) == original_sha
    # Same dimensions and cell names do not establish the native gene axis.
    a = ad.read_h5ad(out['matrix_path']); a.var_names = ['wrong-gene', 'Beta']; a.write_h5ad(out['matrix_path'])
    with pytest.raises(AuthorityError): owner.recover(args, 'icta2a-activity')


@pytest.mark.parametrize('owner', [SNAPATAC2, EPISCANPY], ids=['snapatac2', 'episcanpy'])
def test_native_accepted_source_prior_binding_and_download(native, sample, owner, monkeypatch):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
    from agent.orchestration.prior_outputs import bind_output, accepted_step_digest
    from agent.orchestration.verification_authority import _load_context
    from agent.schemas.prior_output import PriorOutputRef
    root = sample['root']/'workspace'
    scientific = root/'runs'/ManagedWorkspace.run_digest('activity:run')/'scientific'
    snap = owner is SNAPATAC2
    source = sample['fragments'] if snap else sample['peaks']
    before = sha(source)
    first = dict(input_path=str(source), output_dir=str(scientific/'source'),
        function_name='pp.import_fragments' if snap else 'pp.calculate_qc_metrics',
        kwargs=dict(chrom_sizes={'chr1': 2000}, min_num_fragments=0) if snap else dict(percent_top=None, log1p=False))
    plan = AgentPlan('activity', 'activity', 'Native Gene Activity', (
        PlanStep('source', 'run_snapatac2' if snap else 'run_scanpy', first),
        PlanStep('activity', owner.TOOL, dict(input_path=StepOutputRef('source', 'matrix_path'),
            function_name='pp.make_gene_matrix' if snap else 'tl.geneactivity', kwargs=dict(upstream=0),
            gene_anno_path=str(sample['gtf']), gene_anno_sha256=sha(sample['gtf']), output_dir=str(scientific/'activity')), ('source',))))
    class Planner:
        def plan(self, request, registry): return plan
    app = ResearchAgentApplication(root, planner=Planner()); app.sessions.create('genes')
    state = app.sessions.run('genes', 'activity', AgentRequest('activity', 'Calculate native gene activity', {}),
        (OutputSelection('dataset', 'activity', 'matrix_path'),), expected_generation=0)
    run = app.run_store.load('activity:run')
    assert state.turn('activity').status == 'activated', run.errors
    assert all(s.verification.passed and s.verification.artifact_authority for s in run.steps)
    original, result = [s.result for s in run.steps]
    assert result['source_identity']['accepted_publication'] == {
        k: original[k] for k in ('manifest_path', 'manifest_sha256')}
    restarted = ResearchAgentApplication(root)
    assert restarted.sessions.load('genes') == state
    _load_context(restarted.run_store, run.run_id, 'historical_verified_sources.v1').validate_anchors()
    binding = bind_output(restarted.run_store, restarted.registry, run.run_id, 'activity', 'dataset')
    assert binding.semantic_type == 'native_matrix_h5ad.v1'
    class NextPlanner:
        def plan(self, request, registry):
            return AgentPlan('log', request.request_id, 'Native transformation of accepted activity', (
                PlanStep('log', 'run_scanpy', dict(function_name='pp.log1p', input_path=PriorOutputRef(binding, 'matrix_path'),
                    output_dir=str(Path(request.inputs['output_dir'])/'log'))),))
    next_result = ResearchAgentApplication(root, planner=NextPlanner()).run(AgentRequest('log', 'Log transform the selected activity', {}))
    assert next_result.status.value == 'SUCCEEDED', next_result.run_result.errors
    np.testing.assert_allclose(values(next_result.run_result.steps[0].result), np.log1p(values(result)))
    # Delivery is the existing accepted H5AD export, with no numerical rerun.
    import io
    from agent.application.statistical_delivery import create_statistical_export
    monkeypatch.setattr(owner, 'native', lambda *a: pytest.fail('Delivery repeated science'))
    target = io.BytesIO()
    path = Path(result['matrix_path'])
    with path.open('rb') as stream:
        create_statistical_export(stream, target, dict(tool_name=owner.TOOL, accepted_step_sha256=accepted_step_digest(run.steps[1])),
            source_sha256=result['matrix_sha256'], source_size_bytes=path.stat().st_size,
            manifest_sha256=result['manifest_sha256'], max_bytes=1024**3)
    delivered = ad.read_h5ad(target)
    assert delivered.obs_names.tolist() == ['A', 'B']
    assert delivered.var_names.tolist() == ['Alpha', 'Beta']
    assert delivered.uns['matrix_semantics'] == result['measurement']
    portable = json.loads(delivered.uns['agent_gene_activity'])
    assert portable['description']['interface'] == 'gene-activity.v1'
    assert portable['source_identity']['accepted_publication'] == dict(manifest_sha256=original['manifest_sha256'])
    assert sha(source) == before


@pytest.mark.parametrize('change', ['cells', 'features', 'attributes', 'index_name'])
def test_gene_axis_completion_rejects_same_shape_substitution(sample, monkeypatch, change):
    from agent.tools.analysis.native_gene_activity import source_metadata
    def native(args, stage, matrix):
        a = matrix.copy(); a.var_names = ['Alpha', 'Beta']; a.var['gene_id'] = ['G1', 'G2']
        a.write_h5ad(stage/'result.h5ad')
        receipt = dict(package='episcanpy', function_name='tl.geneactivity', versions={'episcanpy': 'SIMULATED'},
            implementation=[], native_output_type='AnnData', result_slots=['X'], measurement='episcanpy_gene_domain_sum',
            effective_parameters={'feature_type': 'gene'}, gene_activity_axes=axis_identity(stage/'result.h5ad'),
            gene_activity_source=source_metadata(stage/'input.h5ad', args))
        if change == 'cells': a.obs_names = ['B', 'A']
        if change == 'features': a.var_names = ['wrong-gene', 'Beta']
        if change == 'attributes': a.var['gene_id'] = ['different-id', 'G2']
        if change == 'index_name': a.var.index.name = 'incorrect_namespace'
        a.write_h5ad(stage/'result.h5ad')
        return receipt
    monkeypatch.setattr(EPISCANPY, 'native', native)
    with pytest.raises(AuthorityError, match='axes|gene axis'):
        EPISCANPY._execute(dict(function_name='tl.geneactivity', input_path=str(sample['peaks']),
            output_dir=str(sample['root']/'substituted'), gene_anno_path=str(sample['gtf'])))
    assert not (sample['root']/'substituted/manifest.json').exists()


@pytest.mark.parametrize('change', ['features', 'X'])
def test_gene_exception_does_not_relax_other_functions(sample, monkeypatch, change):
    def native(args, stage, matrix):
        if change == 'features': matrix.var_names = ['Alpha', 'Beta']
        else: matrix.X.data *= 2
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(result_slots=[])
    monkeypatch.setattr(EPISCANPY, 'native', native)
    with pytest.raises(AuthorityError, match='axes|source X'):
        EPISCANPY._execute(dict(function_name='tl.pca', input_path=str(sample['peaks']),
            output_dir=str(sample['root']/'unrelated')))
    assert not (sample['root']/'unrelated/manifest.json').exists()


def test_native_snap_explicit_tss_windows_and_identifier_keys(native, sample):
    source = imported(sample)
    out = activity(sample, SNAPATAC2, source['matrix_path'], 'tss',
        upstream=20, downstream=20, include_gene_body=False, counting_strategy='insertion', gene_name_key='gene_id')
    assert ad.read_h5ad(out['matrix_path']).var_names.tolist() == ['G1', 'G2']
    np.testing.assert_array_equal(values(out), [[2, 0], [0, 0]])
    parameters = out['parameters']['effective_native_parameters']
    assert parameters['include_gene_body'] is False and parameters['gene_name_key'] == 'gene_id'


def test_native_epi_selected_layer_and_normalized_values_remain_explicit(native, sample):
    a = ad.read_h5ad(sample['peaks'])
    a.layers['fractional'] = a.X.astype(float) / 2
    a.write_h5ad(sample['peaks']); original = sha(sample['peaks'])
    out = EPISCANPY.execute(dict(function_name='tl.geneactivity', input_path=str(sample['peaks']),
        output_dir=str(sample['root']/'layer'), count_slot='fractional', matrix_semantics='declared_normalized_peak_values',
        gene_anno_path=str(sample['gtf']), kwargs=dict(upstream=0)))
    np.testing.assert_array_equal(values(out), [[1, 0], [0, 1.5]])
    assert out['parameters']['gene_activity']['source_slot'] == 'layers/fractional'
    assert out['source_identity']['measurement'] == 'declared_normalized_peak_values'
    assert sha(sample['peaks']) == original


def test_native_epi_raw_different_width_and_layer_reaches_native(native, sample):
    a = ad.read_h5ad(sample['peaks'])
    raw = ad.AnnData(sparse.csr_matrix([[2, 0, 9], [0, 3, 8]]), obs=a.obs.copy(),
        var=pd.DataFrame(index=['raw-1', 'raw-2', 'raw-extra']))
    a.raw = raw
    a.layers['other'] = a.X.astype(float) / 2
    a.write_h5ad(sample['peaks'])
    before = sha(sample['peaks'])
    out = invoke(sample, EPISCANPY, 'tl.geneactivity', sample['peaks'], 'raw-wide',
        count_slot='other', gene_anno_path=str(sample['gtf']), kwargs=dict(raw=True, upstream=0))
    np.testing.assert_array_equal(values(out), [[2, 0], [0, 3]])
    assert out['parameters']['gene_activity']['raw_current_features_equal'] is False
    assert out['parameters']['gene_activity']['source_slot'] == 'raw.X'
    assert sha(sample['peaks']) == before
    # Missing raw remains the installed function's error, without an Agent precheck.
    a.raw = None; a.write_h5ad(sample['peaks'])
    with pytest.raises(NativeInvocationError):
        activity(sample, EPISCANPY, sample['peaks'], 'raw-absent', raw=True, upstream=0)
    assert 'AttributeError' in (sample['root']/'raw-absent/native-failure.log').read_text()
