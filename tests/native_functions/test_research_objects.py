"""Native research-object boundaries; real numerical parity is opt-in evidence."""
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pysam
import pytest

from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis import archr, monalisa
from agent.tools.data.authority_context import VerificationContext, authority_operation


@pytest.mark.parametrize('owner,function,kwargs', [
    (archr, 'addPeak2GeneLinks', {}), (archr, 'getMatrixFromProject', {'ArchRProj': 'evil.rds'}),
    (archr, 'createArrowFiles', {'geneAnnotation': 'eval()'}),
    (archr, 'createArrowFiles', {'outputNames': '/tmp/elsewhere'}),
    (archr, 'ArchRProject', {'copyArrows': False}),
    (monalisa, 'run_cgs_models', {}), (monalisa, 'calcBinnedMotifEnrR', {'pwmL': 'evil.rds'}),
    (monalisa, 'calcBinnedMotifEnrR', {'background': 'genome'}),
    (monalisa, 'calcBinnedMotifEnrR', {'maxKmerSize': 5}),
])
def test_unsupported_native_function_or_object_has_no_substitution(owner, function, kwargs):
    with pytest.raises(ValueError):
        owner.native_arguments(dict(function_name=function, kwargs=kwargs))


def test_native_numeric_errors_and_omissions_are_not_repaired():
    for owner, function, kwargs in [(archr,'createArrowFiles',{'minTSS':-1}),
        (monalisa,'calcBinnedMotifEnrR',{'pseudocount.log2enr':-2})]:
        assert owner.native_arguments(dict(function_name=function,kwargs=kwargs)) == kwargs
        assert owner.native_arguments(dict(function_name=function)) == {}


@pytest.mark.parametrize('context', [False, True])
def test_uploaded_snapshot_never_grants_object_deserialization(tmp_path, context):
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'native_completion':'succeeded'}))
    args=dict(input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),function_name='ArchRProject')
    with authority_operation(VerificationContext() if context else None):
        with pytest.raises(AuthorityError,match='accepted internal'):
            archr.accepted_snapshot(args,tmp_path)
    assert not (tmp_path/'snapshot').exists()


def test_no_missing_genome_or_shift_inference():
    args=dict(function_name='createArrowFiles')
    assert archr.parameter_prerequisites(args)==('sample_name','assembly','coordinate_system','fragment_endpoints')
    args.update(sample_name='A',assembly='GRCh38',coordinate_system='zero_based_half_open',fragment_endpoints='tn5_adjusted',second_path='second')
    assert archr.parameter_prerequisites(args)==('second_sample_name',)
    assert archr.parameter_prerequisites(dict(function_name='getMatrixFromProject'))==()


def test_fragment_packaging_preserves_decoded_records_and_source(tmp_path):
    genome=tmp_path/'genome.fa';genome.write_text('>chr1\n'+'A'*100+'\n');pysam.faidx(str(genome))
    source=tmp_path/'fragments';data=b'chr1\t0\t10\tBC1\t2\nchr1\t20\t30\tBC2\t1\n';source.write_bytes(data)
    target=tmp_path/'native.tsv.gz'
    assert archr.fragment_input(source,target,dict(genome=dict(fasta=dict(path=str(genome)))))==2
    with pysam.BGZFile(str(target),'rb') as f:assert f.read()==data
    assert source.read_bytes()==data and Path(str(target)+'.tbi').exists()


@pytest.fixture
def bins_source(tmp_path,monkeypatch):
    a=ad.AnnData(np.ones((1,3)),var=pd.DataFrame({'chrom':['chr1']*3,'start':[0,10,20],'end':[4,14,24],
        'bins':pd.Categorical(['z','a','z'],categories=['z','a'],ordered=True)},index=['r3','r1','r2']))
    a.uns.update(assembly='GRCh38',coordinate_system='zero_based_half_open');path=tmp_path/'input.h5ad';a.write_h5ad(path)
    fasta=tmp_path/'genome.fa';fasta.write_text('>chr1\n'+'ACGT'*25+'\n');pysam.faidx(str(fasta))
    monkeypatch.setattr(monalisa.sequence_reference,'resolve_motif_resources',lambda *a:(str(fasta),{}, {}, {}, []))
    stage=tmp_path/'stage';stage.mkdir()
    return dict(input_path=str(path),bin_key='bins'),stage


def test_sequence_bins_keep_categorical_order_and_exact_region_order(bins_source):
    args,stage=bins_source
    levels,*_=monalisa.prepare(args,stage)
    assert levels==['z','a']
    assert (stage/'sequences.fa').read_text()=='>r3\nACGT\n>r1\nGTAC\n>r2\nACGT\n'
    assert pd.read_csv(stage/'bins.tsv',sep='\t').feature_id.tolist()==['r3','r1','r2']


def test_bin_table_equal_counts_do_not_establish_correspondence(bins_source,tmp_path):
    args,stage=bins_source;args.pop('bin_key')
    table=tmp_path/'bins.tsv';table.write_text('feature_id\tbin\nr1\ta\nr2\tb\nr3\ta\n');args['bins_path']=str(table)
    with pytest.raises(ValueError,match='exact source feature order'):
        monalisa.prepare(args,stage)


def test_registry_native_identity_and_new_roles(tmp_path):
    from agent.orchestration import build_default_tool_registry
    from agent.application.local_resources import LocalResourceAdmission,group_choices
    r=build_default_tool_registry()
    assert r.get('run_archr').native_functions.package=='ArchR'
    assert r.get('run_monalisa').native_functions.package=='monaLisa'
    assert 'run_pycistopic' not in r.names()
    p=tmp_path/'input.tsv';p.write_text('feature_id\tbin\na\tb\n')
    local=LocalResourceAdmission(tmp_path/'roles',approved_source_roots=(tmp_path,),registry=r)
    e=local.register('first',p,label=p.name,attribution='test',input_type='attachment')
    g=local.compose_group([e.resource_id],attachment_candidates=True)
    offers={(p['tool'],p['port']) for p in group_choices(g,r)['roles']}
    assert {('run_archr','dataset'),('run_archr','second'),('run_monalisa','dataset'),('run_monalisa','bins')}<=offers


def test_archr_result_family_keeps_project_and_matrix_distinct():
    assert 'native-result.rds' not in archr.publication_files(dict(arguments={'function_name':'createArrowFiles'}))
    assert {'matrix.mtx','native-result.rds','features.tsv'}<=set(archr.publication_files(dict(arguments={'function_name':'getMatrixFromProject'})))


def test_archr_prerequisites_are_all_conversationally_representable():
    from agent.orchestration import build_default_tool_registry
    from agent.application.dialogue_execution import _parameter_input
    registry=build_default_tool_registry()
    for key in archr.parameter_prerequisites(dict(function_name='createArrowFiles',second_path='sample2')):
        selector, argument = _parameter_input(registry,'run_archr',key)
        assert selector and argument.planning.scientific_parameter
    assert archr.native_arguments(dict(function_name='createArrowFiles',second_path='sample2',kwargs={'minTSS':0})) == {'minTSS':0}
