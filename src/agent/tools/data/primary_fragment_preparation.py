"""Exact stream restriction to a reviewed primary-nuclear scope, before adoption.

A preparation record is not a fragment producer profile or matrix authority.
The existing 10x adoption owner still independently qualifies prepared fragments.
"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from . import primary_contigs as scope, _external_fragment_io as io
from . import external_fragments as external, external_fragment_manifest as m
from . import scatac_fragments_v2 as v2
from ._fragments_common import canonical, PACKAGING_POLICY, verify_packaging
from .scatac_fragments_v2_verifier import FragmentVerificationRuntime, take_snapshots, check_snapshots
from agent.tools._cancellation import cancellation_checkpoint

CONTRACT = 'primary-nuclear-fragment-preparation.v1'


def dependencies(value):
    """All original and derived bytes remain live dependencies for authority reuse."""
    scoped = scope.verify_scope(value['scope']['path'],value['scope']['sha256'])
    return [value['scope'],scoped['reference'],scoped['classification_source'],value['source'],
            *([value['source_index']] if value['source_index'] else []),value['prepared'],value['prepared_index']]


def load(path,sha):
    value=scope.read(path,sha)
    v2.shape(value,('contract_version','profile_sha256','scope','source','source_index','prepared',
                    'prepared_index','packaging','summary'))
    if value['contract_version']!=CONTRACT or value['profile_sha256']!=scope.PROFILE_SHA256:scope.fail()
    if value['packaging'] != PACKAGING_POLICY:scope.fail()
    for k in ('scope','source','prepared','prepared_index'):
        v2.resource(value[k])
    if value['source_index'] is not None:v2.resource(value['source_index'])
    v2.shape(value['summary'],('source_records','retained_records','excluded_records','excluded_by_contig',
                              'retained_by_contig','retained_stream_sha256'))
    return value


def _classes(scoped):
    return {c['name']:c['classification'] for c in scoped['contigs']} | scoped['extra_exclusions']


def verify_preparation(path,sha,*,reference_path=None,reference_sha256=None,deep=True):
    value=load(path,sha)
    resources=dependencies(value);before=take_snapshots([path,*[r['path'] for r in resources]])
    for r in resources:
        if io.resource(r['path'],r['sha256'])!=r:scope.fail()
    scoped=scope.verify_scope(value['scope']['path'],value['scope']['sha256'],
                              reference_path=reference_path,reference_sha256=reference_sha256)
    if deep:
        # Independent bytewise subsequence replay. No production filter call.
        classes=_classes(scoped);retained=Counter();excluded=Counter();digest=hashlib.sha256();seen_data=False
        original=io.lines(value['source']['path'],io.encoding(value['source']['path']))
        prepared=iter(io.lines(value['prepared']['path'],'bgzf'))
        for count,raw in enumerate(original):
            if len(raw)>m.MAX_LINE or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:scope.fail()
            if raw.startswith(b'#'):
                if seen_data or next(prepared,None)!=raw:scope.fail()
                continue
            seen_data=True;name=raw.split(b'\t',1)[0].decode('ascii')
            if name not in classes:scope.fail()
            if classes[name]=='primary_nuclear':
                if next(prepared,None)!=raw:scope.fail()
                retained[name]+=1;digest.update(raw)
            else:excluded[name]+=1
            if count%65536==0:cancellation_checkpoint()
        if next(prepared,None) is not None:scope.fail()
        summary=dict(source_records=sum(retained.values())+sum(excluded.values()),retained_records=sum(retained.values()),
            excluded_records=sum(excluded.values()),excluded_by_contig=dict(excluded),retained_by_contig=dict(retained),
            retained_stream_sha256=digest.hexdigest())
        if summary!=value['summary']:scope.fail()
    check_snapshots(before)
    return value


def prepare_primary_fragments(*,source_path,source_sha256,source_index_path,source_index_sha256,
                              scope_path,scope_sha256,output_dir):
    """Preserve every retained byte and its order; index afresh with BED semantics."""
    destination=Path(output_dir)
    if not destination.is_absolute() or destination!=destination.resolve() or destination.exists():scope.fail()
    scoped=scope.verify_scope(scope_path,scope_sha256);classes=_classes(scoped)
    source=io.resource(source_path,source_sha256)
    index=None if source_index_path is None else io.resource(source_index_path,source_index_sha256)
    if (source_index_path is None)!=(source_index_sha256 is None):scope.fail()
    scoped_resource=io.resource(scope_path,scope_sha256)
    before=take_snapshots([source_path,scope_path,*([source_index_path] if index else [])])
    runtime=FragmentVerificationRuntime();verify_packaging(runtime)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.primary-fragments-',dir=destination.parent) as temporary:
        stage=Path(temporary);plain=stage/'retained.tsv';prepared=stage/'fragments.tsv.gz'
        retained=Counter();excluded=Counter();digest=hashlib.sha256()
        dictionary={c['name']:(i,c['length']) for i,c in enumerate(scoped['contigs'])}
        previous=None;closed=set();width=None;headers=header_bytes=0
        with plain.open('xb') as out:
            for number,raw in enumerate(io.lines(source_path,io.encoding(source_path))):
                if len(raw)>m.MAX_LINE or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:scope.fail()
                if raw.startswith(b'#'):
                    headers+=1;header_bytes+=len(raw)
                    if width is not None or headers>m.MAX_HEADERS or header_bytes>m.MAX_HEADER_BYTES:scope.fail()
                    raw.decode('utf-8');out.write(raw);continue
                fields=raw[:-1].decode('ascii').split('\t');name=fields[0]
                if len(fields) not in (5,6) or (width is not None and width!=len(fields)):scope.fail()
                width=len(fields)
                if name not in classes:scope.fail()
                if classes[name]!='primary_nuclear':excluded[name]+=1;continue
                _,start,_,_,_,_,_=external.parse_record(raw,dictionary)
                if previous:
                    if name==previous[0] and start<previous[1]:scope.fail()
                    if name!=previous[0]:
                        closed.add(previous[0])
                        if name in closed:scope.fail()
                previous=(name,start);out.write(raw);digest.update(raw);retained[name]+=1
                if number%65536==0:cancellation_checkpoint()
        if not retained:scope.fail()
        with prepared.open('xb') as out:
            subprocess.run([runtime.bgzip,*PACKAGING_POLICY['bgzip'],str(plain)],stdout=out,check=True)
        subprocess.run([runtime.tabix,'-p','bed',str(prepared)],check=True)
        contigs=tuple((c['name'],c['length']) for c in scoped['contigs'])
        io.check_source_index(prepared,Path(str(prepared)+'.tbi'),'bgzf',contigs,stage,runtime)
        value=dict(contract_version=CONTRACT,profile_sha256=scope.PROFILE_SHA256,scope=scoped_resource,
            source=source,source_index=index,prepared=io.resource(prepared),prepared_index=io.resource(str(prepared)+'.tbi'),
            packaging=PACKAGING_POLICY,summary=dict(source_records=sum(retained.values())+sum(excluded.values()),
            retained_records=sum(retained.values()),excluded_records=sum(excluded.values()),
            excluded_by_contig=dict(excluded),retained_by_contig=dict(retained),retained_stream_sha256=digest.hexdigest()))
        manifest=stage/'preparation.json';manifest.write_bytes(canonical(value))
        verify_preparation(manifest,io.file_sha256(manifest))
        for key in ('prepared','prepared_index'):value[key]['path']=str(destination/Path(value[key]['path']).name)
        manifest.write_bytes(canonical(value));plain.unlink()
        for p in stage.iterdir():
            with p.open('rb') as stream:os.fsync(stream.fileno())
        check_snapshots(before)
        if destination.exists():scope.fail()
        os.rename(stage,destination)
        from .scatac_qc_reference import _fsync_dir
        _fsync_dir(destination);_fsync_dir(destination.parent)
    return io.resource(destination/'preparation.json')
