"""Explicit FASTQ roles use original paths through the existing scientific owners."""
from dataclasses import replace
from pathlib import Path
import gzip
import json

import pytest

from agent.tools.data import _raw_fastq as f, raw_scatac as raw, raw_scatac_manifest as m
from agent.tools.data._fragments_binding import fresh_intake
from agent.tools.data._fragments_fastq import scan_group
from agent.tools.data._chromap import QualificationGroup
from agent.orchestration.registry import build_default_tool_registry
from agent.orchestration.raw_scatac_verifier import verify_raw_scatac_intake


def sources(root, *, layout=f.FastqLayout.A, gzip_roles=(), optional=False):
    rows=[]
    for role,meaning in f.LAYOUT_ROLES[layout].items():
        if role is f.ReadRole.I1 and not optional:
            continue
        sequence='ACGTACGTACGTACGT' if meaning is f.ReadMeaning.BARCODE else 'A'*50
        content=f'@pair0\n{sequence}\n+\n'+ 'I'*len(sequence)+'\n'
        path=root/(role.value+'.opaque')
        path.write_bytes(gzip.compress(content.encode(),mtime=0) if role.value in gzip_roles else content.encode())
        rows.append(dict(path=str(path),library_id='library',role=role.value,lane='001',chunk='001'))
    return rows


def inspect(root,rows,layout=f.FastqLayout.A):
    args=dict(raw_input_paths=rows,output_dir=str(root/'intake'),species='human',raw_assay='TENX_ATAC',fastq_layout=layout.value)
    result=raw.inspect_raw_scATAC(**args)
    verify_raw_scatac_intake(args,result,build_default_tool_registry())
    _,manifest,_=m.load_raw_intake_manifest(result['manifest_path'])
    return args,result,manifest


@pytest.mark.parametrize('layout',[f.FastqLayout.A,f.FastqLayout.B])
@pytest.mark.parametrize('gzip_roles',[(),('R1','R2','R3','I2','I1'),('R1',)])
@pytest.mark.parametrize('optional',[False,True])
def test_explicit_roles_content_compression_and_independent_full_scan(tmp_path,layout,gzip_roles,optional):
    rows=sources(tmp_path,layout=layout,gzip_roles=gzip_roles,optional=optional)
    args,result,manifest=inspect(tmp_path,rows,layout)
    assert result['readiness']=='READY'
    assert f.declared_sources(manifest)==f.explicit_sources(rows)
    assert manifest.groups[0].grouping_basis is m.GroupingBasis.EXPLICIT
    assert manifest.groups[0].syntactic_sample_token is None
    assert not any(e.source is m.EvidenceSource.FILENAME for e in manifest.evidence)
    assert {e.file_id for e in manifest.evidence if e.claim is m.EvidenceClaim.GROUPING}=={s.id for s in manifest.files}
    assert all(e.source is m.EvidenceSource.USER_DECLARATION for e in manifest.evidence if e.claim is m.EvidenceClaim.GROUPING)
    fresh_intake(manifest)
    group=QualificationGroup(manifest.groups[0].id.removeprefix('group:'),layout,
        tuple((f.ReadRole(r['role']),r['path']) for r in rows))
    assert scan_group(group,16)['record_count']==1
    assert {s.path for s in manifest.files}=={r['path'] for r in rows}


@pytest.mark.parametrize('case',['duplicate_path','duplicate_role','unknown_role','mixed','extra','relative','traversal','symlink','directory'])
def test_explicit_sources_reject_ambiguous_or_unsafe_binding(tmp_path,case):
    rows=sources(tmp_path)
    if case=='duplicate_path':rows[1]['path']=rows[0]['path']
    elif case=='duplicate_role':rows[1]['role']=rows[0]['role']
    elif case=='unknown_role':rows[0]['role']='guessed'
    elif case=='mixed':rows[1]=rows[1]['path']
    elif case=='extra':rows[0]['shell']='run'
    elif case=='relative':rows[0]['path']='relative.attachment'
    elif case=='traversal':rows[0]['path']=str(tmp_path)+'/../R1.opaque'
    elif case=='symlink':
        link=tmp_path/'alias';link.symlink_to(rows[0]['path']);rows[0]['path']=str(link)
    elif case=='directory':rows[0]['path']=str(tmp_path)
    with pytest.raises(ValueError):inspect(tmp_path,rows)


@pytest.mark.parametrize('case',['invalid_bytes','read_name','missing_member','wrong_layout','truncated_gzip'])
def test_explicit_labels_never_qualify_bad_scientific_content(tmp_path,case):
    rows=sources(tmp_path)
    layout=f.FastqLayout.A
    if case=='invalid_bytes':Path(rows[0]['path']).write_bytes(b'not FASTQ')
    elif case=='read_name':
        p=Path(rows[0]['path']);p.write_bytes(p.read_bytes().replace(b'pair0',b'other'))
    elif case=='missing_member':rows.pop()
    elif case=='wrong_layout':layout=f.FastqLayout.B
    elif case=='truncated_gzip':Path(rows[0]['path']).write_bytes(b'\x1f\x8b\x08')
    _,result,manifest=inspect(tmp_path,rows,layout)
    assert result['readiness']!='READY'
    assert manifest.groups[0].structure is not m.StructureState.SUPPORTED


def test_mutated_original_or_changed_declarations_invalidate_intake(tmp_path):
    rows=sources(tmp_path);args,result,manifest=inspect(tmp_path,rows)
    forged=[dict(row,library_id='another') for row in rows]
    with pytest.raises(ValueError):
        verify_raw_scatac_intake(dict(args,raw_input_paths=forged),result,build_default_tool_registry())
    p=Path(rows[0]['path']);p.write_bytes(p.read_bytes().replace(b'A',b'T'))
    with pytest.raises(ValueError):fresh_intake(manifest)
    with pytest.raises(ValueError):verify_raw_scatac_intake(args,result,build_default_tool_registry())


def test_explicit_evidence_cannot_fall_back_to_filename_inference(tmp_path):
    _,_,manifest=inspect(tmp_path,sources(tmp_path))
    group=manifest.groups[0]
    with pytest.raises(ValueError):
        f.declared_sources(replace(manifest,groups=(replace(group,grouping_evidence_ids=group.grouping_evidence_ids[:-1]),)))


def test_multiple_lanes_are_explicit_and_do_not_depend_on_directory(tmp_path):
    rows=sources(tmp_path)
    another=tmp_path/'unrelated-name';another.mkdir()
    rows+= [dict(row,lane='002') for row in sources(another)]
    _,result,manifest=inspect(tmp_path,rows)
    assert result['readiness']=='READY' and result['n_groups']==2
    assert {g.lane for g in manifest.groups}=={'001','002'}
    fresh_intake(manifest)
