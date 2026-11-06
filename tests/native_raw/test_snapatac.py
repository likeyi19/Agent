"""Native raw dispatch, scientific semantics and authority without production replay."""
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import pytest

from agent.tools.analysis.native_matrix import SNAPATAC2, NativeInvocationError
from agent.schemas.verification_authority import AuthorityError
from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path/'source.data'
    source.write_text('# explicitly simulated; position/barcode order is arbitrary\n'
        'chr1\t4900\t5100\tB-1\t300\nchr1\t4940\t4960\tA-1\t2\nchr1\t4900\t5100\tA-1\t1\n'
        'chrM\t2\t20\tA-1\t1\n')
    sizes = tmp_path/'sizes'; sizes.write_text('chr1\t10000\nchr2\t100\nchrM\t50\n')
    peaks = tmp_path/'regions'; peaks.write_text('chr1\t4920\t5080\nchr1\t4890\t5110\nchr1\t5090\t5120\nchr2\t0\t10\n')
    annotation = tmp_path/'annotation.gtf'
    annotation.write_text('chr1\tfixture\ttranscript\t5001\t6000\t.\t+\t.\tgene_id "G"; transcript_id "T"; gene_name "G"; transcript_name "T"; transcript_type "protein_coding";\n')
    return dict(source=source, sizes=sizes, peaks=peaks, annotation=annotation, root=tmp_path)


@pytest.fixture
def native():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit native runtime qualification')
    assert os.environ.get('AGENT_SNAPATAC2_PYTHON'), 'Supply the isolated installed runtime'


def call(inputs, name, input_path=None, **kwargs):
    return SNAPATAC2.execute(dict(input_path=str(input_path or inputs['source']), output_dir=str(inputs['root']/('out-'+name.replace('.','-'))),
        function_name=name, **kwargs), 'test-execution')


def imported(inputs):
    return call(inputs, 'pp.import_fragments', chrom_sizes_path=str(inputs['sizes']),
        kwargs=dict(sorted_by_barcode=False, min_num_fragments=0), species='human', assembly='hg38')


def test_catalog_uses_one_native_owner_and_no_old_executable_registry():
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.semantic_compiler import build_semantic_compiler_contract
    registry = build_default_tool_registry(); catalog = registry.get('run_snapatac2').native_functions
    assert {'pp.make_fragment_file','pp.import_fragments','metrics.tsse','pp.filter_cells','pp.make_peak_matrix'} <= set(dict(catalog.functions))
    assert 'tl.spectral' in dict(catalog.functions)
    for name in ('prepare_scATAC_fragments','prepare_scATAC_bam_fragments','import_scATAC_fragments','compute_scATAC_qc','select_scATAC_cells','build_scATAC_cell_by_ccre','build_scATAC_cell_by_features'):
        assert not registry.contains(name)
        old = registry.accepted(name)
        assert not hasattr(old, 'function') and old.planning is None
        with pytest.raises(LookupError): registry.get(name)
    contract = build_semantic_compiler_contract(registry)
    assert any(rule.producer_tool_name == 'run_chromap' and rule.consumer_tool_name == 'run_snapatac2' for rule in contract.step_output_channels)


def test_raw_native_kwargs_are_forwarded_without_scientific_repair(inputs):
    for name, kwargs in [('pp.make_fragment_file',dict(min_mapq=-9,shift_left=17,shift_right=2)),
                         ('pp.filter_cells',dict(min_counts=-1,min_tsse=None)),
                         ('pp.make_peak_matrix',dict(counting_strategy='native-rejects-this'))]:
        args = dict(input_path=str(inputs['source']), output_dir=str(inputs['root']/'out'),function_name=name,kwargs=kwargs)
        assert SNAPATAC2.native_arguments(args) == kwargs


@pytest.mark.parametrize('kw', [dict(peak_file='/etc/passwd'),dict(tempdir='/etc'),dict(file='/etc/passwd'),dict(whitelist='/etc/passwd')])
def test_native_files_cannot_enter_through_kwargs(inputs, kw):
    with pytest.raises(ValueError): call(inputs,'pp.make_peak_matrix',kwargs=kw)


def test_source_and_resource_checksum_precede_native_execution(inputs, monkeypatch):
    monkeypatch.setattr(SNAPATAC2,'native',lambda *a: pytest.fail('Changed input reached native science'))
    with pytest.raises(AuthorityError,match='bytes changed'):
        call(inputs,'pp.import_fragments',input_sha256='0'*64,chrom_sizes_path=str(inputs['sizes']))
    with pytest.raises(AuthorityError,match='bytes changed'):
        call(inputs,'pp.import_fragments',chrom_sizes_path=str(inputs['sizes']),chrom_sizes_sha256='0'*64)


def test_native_default_and_explicit_declarations_do_not_require_legacy_profiles(inputs):
    from agent.application.raw_handoff import FragmentsDeclaration, BamDeclaration, requirements
    from agent.application.local_resources import RegisteredInputGroup
    d=FragmentsDeclaration(source='source.0',species='human',assembly='hg38')
    binding=RegisteredInputGroup((dict(resource_id='local-'+'a'*64,record_sha256='b'*64,source_path=str(inputs['source']),source_sha256='c'*64,input_type='attachment',label='simulated fragments'),),{},attachment_candidates=True)
    questions,_=requirements(binding,d,())
    assert not any('support' in q or 'library' in q or 'profile' in q for q in questions)


def test_unaccepted_old_receipt_cannot_gain_authority(inputs):
    from agent.tools.data.raw_history import accepted
    with pytest.raises(AuthorityError,match='existing accepted execution'):
        accepted('bam_fragment_production',inputs['source'],'0'*64)


def test_native_import_and_all_count_units(native, inputs):
    before=inputs['source'].read_bytes(); result=imported(inputs)
    raw=ad.read_h5ad(result['matrix_path'])
    assert list(raw.obs_names)==['A-1','B-1'] and raw.shape==(2,0)
    assert raw.obs.n_fragment.tolist()==[2,1]  # Mito excluded from native n_fragment.
    assert raw.obs.frac_dup.tolist()==pytest.approx([.25,299/300])
    assert result['parameters']['kwargs']==dict(sorted_by_barcode=False,min_num_fragments=0)
    for strategy,expected in [('fragment',[[2,2,1,0],[1,1,1,0]]),('insertion',[[2,4,1,0],[0,2,1,0]]),('paired-insertion',[[1,2,1,0],[0,1,1,0]])]:
        out=SNAPATAC2.execute(dict(input_path=result['matrix_path'],output_dir=str(inputs['root']/strategy),function_name='pp.make_peak_matrix',
            peaks_path=str(inputs['peaks']),kwargs=dict(counting_strategy=strategy)))
        matrix=ad.read_h5ad(out['matrix_path'])
        assert matrix.X.toarray().tolist()==expected
        assert list(matrix.obs_names)==list(raw.obs_names)
        assert matrix.var.start.tolist()==[4920,4890,5090,0] and matrix.var.end.tolist()==[5080,5110,5120,10]
        assert matrix.uns['species']=='human' and matrix.uns['assembly']=='hg38'
        assert 'fragment_paired' not in matrix.obsm  # Native matrix contains counts, not a duplicated fragment store.
    assert inputs['source'].read_bytes()==before


def test_native_errors_and_cancellation_publish_no_manifest(native,inputs):
    with pytest.raises(NativeInvocationError,match='barcode_tag'):
        call(inputs,'pp.make_fragment_file')
    assert not list(inputs['root'].rglob('manifest.json'))
    with cancellation_scope(lambda:True),pytest.raises(ToolWorkCancelled): imported(inputs)
    assert not list(inputs['root'].rglob('manifest.json'))
    assert not list(inputs['root'].rglob('.native-snapatac2-*'))


def test_native_qc_filtering_and_exact_source_preservation(native,inputs):
    result=imported(inputs); source=Path(result['matrix_path']); before=source.read_bytes()
    qc=call(inputs,'metrics.tsse',input_path=source,gene_anno_path=str(inputs['annotation']))
    native_qc=ad.read_h5ad(qc['matrix_path']); assert 'tsse' in native_qc.obs
    selected=call(inputs,'pp.filter_cells',input_path=qc['matrix_path'],kwargs=dict(min_counts=2,min_tsse=0))
    assert list(ad.read_h5ad(selected['matrix_path']).obs_names)==['A-1']
    assert selected['parameters']['effective_native_parameters']['min_tsse']==0
    assert source.read_bytes()==before


def test_native_recovery_is_read_only_and_bound_to_exact_execution(native,inputs,monkeypatch):
    result=imported(inputs)
    args=dict(input_path=str(inputs['source']),output_dir=str(inputs['root']/'out-pp-import_fragments'),function_name='pp.import_fragments',
        chrom_sizes_path=str(inputs['sizes']),kwargs=dict(sorted_by_barcode=False,min_num_fragments=0),species='human',assembly='hg38')
    monkeypatch.setattr(SNAPATAC2,'native',lambda *a:pytest.fail('Recovery repeated native science'))
    assert SNAPATAC2.recover(args,'test-execution')==result
    with pytest.raises(AuthorityError): SNAPATAC2.recover(args,'unrelated-execution')
    Path(result['matrix_path']).write_bytes(b'corrupt')
    with pytest.raises(AuthorityError): SNAPATAC2.recover(args,'test-execution')


def test_chromap_presence_switches_cannot_silently_become_defaults():
    from agent.tools.data.chromap_native import cli_arguments
    assert cli_arguments({'MAPQ-threshold':-9,'Tn5-shift':True})==['--MAPQ-threshold','-9','--Tn5-shift']
    for value in (False,None,0):
        with pytest.raises(ValueError,match='presence-only'):
            cli_arguments({'preset':'atac','Tn5-shift':value})
    with pytest.raises(ValueError,match='requires one'):
        cli_arguments({'preset':None})


def test_native_h5ad_handoff_cannot_follow_unbound_external_storage(inputs,monkeypatch):
    import h5py
    import pandas as pd
    path=inputs['root']/'linked.h5ad'
    ad.AnnData(obs=pd.DataFrame(index=['A'])).write_h5ad(path)
    with h5py.File(path,'a') as handle:
        handle['obsm']['unbound']=h5py.ExternalLink('/unoffered/reference.h5','/fragments')
    monkeypatch.setattr(SNAPATAC2,'native',lambda *a:pytest.fail('Unsafe storage reached native code'))
    with pytest.raises(ValueError,match='Indirect'):
        call(inputs,'pp.filter_cells',input_path=path)


def test_existing_peak_caller_reads_exact_explicit_cell_contract_without_raw_producer(tmp_path):
    from types import SimpleNamespace
    from agent.tools.data.explicit_cells import publish_explicit_cells
    from agent.tools.analysis.macs3 import selected_population
    result = publish_explicit_cells(cells=[('library', 'B'), ('library', 'A')],
        declaration='Explicit caller order', output_dir=tmp_path/'cells')
    source = SimpleNamespace(libraries=[SimpleNamespace(namespace='library')])
    args = dict(explicit_cells_manifest_path=result['manifest_path'],
        explicit_cells_manifest_sha256=result['manifest_sha256'])
    namespace, pairs, label, pins = selected_population(args, source)
    assert namespace == 'library' and pairs == [('library', 'B'), ('library', 'A')]
    assert label == 'explicit supplied ordered cells; no QC claim' and len(pins) == 2
    with pytest.raises(ValueError, match='changed'):
        selected_population(dict(args, explicit_cells_manifest_sha256='0'*64), source)
