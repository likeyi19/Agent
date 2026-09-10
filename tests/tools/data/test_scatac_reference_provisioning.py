import hashlib
import json
import os
from pathlib import Path

import h5py
import pytest

from agent.tools.data import scatac_reference_provisioning as p
from agent.tools.data import scatac_reference as r
from agent.tools.data.scatac_matrix_contract import ScATACMatrixError, canonical
from agent.tools._cancellation import cancellation_scope, ToolWorkCancelled


@pytest.fixture
def inputs(tmp_path):
    source=tmp_path/'source.h5ad'
    names=['z:5-8','a:0-2','z:1-3']
    with h5py.File(source,'w') as f:
        var=f.create_group('var');var.attrs['_index']='names'
        var.create_dataset('names',data=names,dtype=h5py.string_dtype())
        # No X needed or read by reference provisioning.
    fa=tmp_path/'genome.fa';fa.write_text('>z\nAAAAAAAAAA\n>a\nAAAAAAAAAA\n')
    fai=tmp_path/'genome.fa.fai';fai.write_text('z\t10\t3\t10\t11\na\t10\t17\t10\t11\n')
    return dict(source_h5ad_path=source,source_h5ad_sha256=p._hash(source),expected_feature_count=3,
        expected_ordered_feature_sha256=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest(),
        species='mouse',assembly='mm10',fasta_path=fa,fai_path=fai,output_dir=tmp_path/'reference',
        authority='explicit synthetic fixture')


def test_provision_reference_roundtrip_order_and_provenance(inputs):
    result=p.provision_h5ad_reference(**inputs)
    bundle=p.verify_h5ad_reference_derivation(result['manifest_path'],expected_sha256=result['manifest_sha256'])
    assert Path(bundle.ccre.bed.path).read_bytes()==b'z\t5\t8\na\t0\t2\nz\t1\t3\n'
    assert bundle.ccre.ordered_feature_sha256==inputs['expected_ordered_feature_sha256']
    assert bundle.ccre.bed.provenance.accession==result['derivation_sha256']
    assert bundle.species=='mouse' and bundle.target_assembly=='mm10'
    assert set(x.name for x in inputs['output_dir'].iterdir())=={'ccre.bed','reference.json','derivation.json'}
    assert r.reinspect_scatac_reference_bundle_sources(bundle)==bundle
    with pytest.raises(ScATACMatrixError,match='REFERENCE_OUTPUT_CONFLICT'):
        p.provision_h5ad_reference(**inputs)


@pytest.mark.parametrize('names',[['a:01-2'],['a:0-02'],['a:2-2'],['a:0-11'],['other:0-2'],
    ['a:0-2','a:0-2'],['a:+0-2'],['a:0-2\n'],['a:0-2 extra']])
def test_bad_source_vocabulary_cleanup(inputs,names):
    with h5py.File(inputs['source_h5ad_path'],'w') as f:
        v=f.create_group('var');v.attrs['_index']='names';v.create_dataset('names',data=names,dtype=h5py.string_dtype())
    inputs['source_h5ad_sha256']=p._hash(inputs['source_h5ad_path'])
    with pytest.raises(ScATACMatrixError):p.provision_h5ad_reference(**inputs)
    assert not inputs['output_dir'].exists()
    assert not list(inputs['output_dir'].parent.glob('.reference-provision-*'))


@pytest.mark.parametrize('key,value',[('source_h5ad_sha256','0'*64),('expected_feature_count',4),
    ('expected_ordered_feature_sha256','0'*64),('assembly','hg38'),('authority','')])
def test_exact_expectations(inputs,key,value):
    inputs[key]=value
    with pytest.raises(ScATACMatrixError):p.provision_h5ad_reference(**inputs)
    assert not inputs['output_dir'].exists()


def test_source_mutation_and_rehashed_false_derivation(inputs):
    result=p.provision_h5ad_reference(**inputs)
    with h5py.File(inputs['source_h5ad_path'],'r+') as f:f['var/names'][0]='z:4-8'
    with pytest.raises(ScATACMatrixError,match='REFERENCE_SOURCE_MISMATCH'):
        p.verify_h5ad_reference_derivation(result['manifest_path'],expected_sha256=result['manifest_sha256'])
    # Rehash both provenance layers; reconstruction must still reject the BED.
    receipt=inputs['output_dir']/'derivation.json'
    v=json.loads(receipt.read_bytes());v['source_h5ad_sha256']=p._hash(inputs['source_h5ad_path'])
    receipt.write_bytes(canonical(v))
    refpath=Path(result['manifest_path']);b=json.loads(refpath.read_bytes())
    b['ccre']['bed']['provenance']['accession']=p._hash(receipt)
    refpath.write_bytes(canonical(b)) # Portable identity excludes provenance.
    with pytest.raises(ScATACMatrixError,match='REFERENCE_SOURCE_MISMATCH'):
        p.verify_h5ad_reference_derivation(refpath,expected_sha256=p._hash(refpath))


def test_external_feature_link_rejected(inputs,tmp_path):
    with h5py.File(inputs['source_h5ad_path'],'r+') as f:
        del f['var/names'];f['var/names']=h5py.ExternalLink('/outside.h5','names')
    inputs['source_h5ad_sha256']=p._hash(inputs['source_h5ad_path'])
    with pytest.raises(ScATACMatrixError):p.provision_h5ad_reference(**inputs)


def test_cancel_private_provision(inputs):
    with cancellation_scope(lambda: bool(list(inputs['output_dir'].parent.glob('.reference-provision-*')))):
        with pytest.raises(ToolWorkCancelled):p.provision_h5ad_reference(**inputs)
    assert not inputs['output_dir'].exists()
    assert not list(inputs['output_dir'].parent.glob('.reference-provision-*'))


def test_species_independent_helper(inputs):
    inputs.update(species='human',assembly='hg38')
    result=p.provision_h5ad_reference(**inputs)
    assert p.verify_h5ad_reference_derivation(result['manifest_path'],expected_sha256=result['manifest_sha256']).species=='human'


@pytest.mark.skipif(os.environ.get('RUN_SCATAC_MOUSE_REFERENCE_ACCEPTANCE') != '1',
                    reason='Explicit local full mouse reference acceptance; no model execution.')
def test_exact_authoritative_mouse_ordered_feature_identity(tmp_path):
    from agent.tools.data._ordered_identity import ordered_identity_sha256
    source=Path('/home/likeyi/program/EpiZoo/data/Fang2021_downsampled_2000_cells.h5ad')
    expected='e2585fa28673779877fb3540d23ab6f36d929a11c901545e2459226dfa6b9fac'
    physical='16189d0dabd21cc1d9f51e5f39662da82037fe9f54938f8593a3ff49aa50cfab'
    assert p._hash(source)==physical
    with h5py.File(source,'r') as f:
        index=f['var'][f['var'].attrs['_index']]
        assert len(index)==1_341_077
        names=(name for start in range(0,len(index),4096)
               for name in index.asstr()[start:start+4096])
        # Independent existing identity helper: UTF-8 name + LF, no prefix.
        assert ordered_identity_sha256(names)==expected
    result=p.provision_h5ad_reference(source_h5ad_path=source,source_h5ad_sha256=physical,
        expected_feature_count=1_341_077,expected_ordered_feature_sha256=expected,
        species='mouse',assembly='mm10',fasta_path='/home/likeyi/program/mm10.fa',
        fai_path='/home/likeyi/program/mm10.fa.fai',output_dir=tmp_path/'mouse-reference',
        authority='Project-authoritative Fang2021 full mouse/mm10 ordered feature vocabulary.')
    bundle=p.verify_h5ad_reference_derivation(result['manifest_path'],expected_sha256=result['manifest_sha256'])
    assert bundle.ccre.feature_count==1_341_077
    assert bundle.ccre.ordered_feature_sha256==expected
    assert bundle.ccre.bed.sha256=='6094091b2767c6e0fc57ca1af19a96dc4ddb30ace8ed6291dcc5b0496a0875a5'
    assert bundle.reference_identity_sha256=='bf698c7842fd86043840e80e0744345bcb11b6a979d917d33c73305e9d23f179'
    with h5py.File(source,'r') as f, open(bundle.ccre.bed.path) as bed:
        index=f['var'][f['var'].attrs['_index']]
        names=(name for start in range(0,len(index),4096)
               for name in index.asstr()[start:start+4096])
        count=0
        for name,line in zip(names,bed,strict=True):
            chrom,start,end=line.rstrip('\n').split('\t')
            assert line==f'{chrom}\t{int(start)}\t{int(end)}\n'
            assert name==f'{chrom}:{int(start)}-{int(end)}'
            count+=1
        assert count==1_341_077
