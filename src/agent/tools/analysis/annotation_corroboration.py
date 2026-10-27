"""Qualified local reference corroboration of an exact native annotation publication.

Candidate labels remain immutable. The existing marker comparison owns biology;
this owner binds native sidecars, reference context, publication and delivery.
"""
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import tempfile
from collections import Counter

from agent.schemas.verification_authority import AuthorityError, VerifiedArtifactAuthority, digest
from agent.schemas.orchestration import _serialize
from agent.tools.data.authority_context import current, owned_verification
from agent.tools.data.fastq_verification_authority import file_closure
from . import annotation_contract as annotation, marker_annotation as native, marker_validation as comparison

KIND = 'annotation_corroboration'
TOOL = 'corroborate_cell_type_annotation'
CONTRACT = 'cell-type-reference-corroboration.v1'
RESOURCE = 'cell-marker-reference.v1'
PROFILE = digest(dict(comparison=comparison.PROFILE, primary='unchanged native annotation',
    mapping='qualified signature-bound terminology or exact database term; no fuzzy matching',
    evidence='local pinned literature associations;not experimental ground truth'))
VERIFIER = dict(id='agent.marker-reference-integrity', compatibility_version='1')
POLICY = 'corroborate-cell-type-annotation-v1'
LIMITATIONS = ('Local pinned CellMarker records, not a live database query. RNA/protein literature '
    'associations are not direct validation of scATAC accessibility or matched experimental ground truth. '
    'Broad-lineage evidence does not establish finer subtypes or states. Positive competing markers '
    'are nonexclusive associations, not established contradictions. Missing evidence is not disproof. '
    'Native scores are not calibrated probabilities; primary labels remain unchanged.')
TABLE_FIELDS = ('group','candidate','native_score','assessment','reference_term','relationship',
    'mapping_basis','gene','avg_logFC','association','source_row','species','tissue','cancer_type',
    'marker','database_symbol','marker_source','PMID','Title','database_release','table_sha256')
RESULT_FIELDS = dict(status=(str,), manifest_path=(str,), manifest_sha256=(str,),
    contract_version=(str,), evidence_path=(str,), evidence_sha256=(str,),
    annotation_manifest_sha256=(str,), ordered_cells_sha256=(str,),
    database=(str,), database_release=(str,), database_source=(str,), database_table_sha256=(str,),
    species=(str,), context=(str,), tissues=(list,), cancer_type=(str,),
    n_groups=(int,), n_records=(int,), group_summary=(list,), groups_omitted=(int,),
    state_counts=(dict,), limitations=(str,))


def read(path, sha=None, limit=64*1024**2):
    return annotation.read_json(path, sha, limit=limit)


def reference(value):
    """Qualify an operator-supplied reference using the existing knowledge reader."""
    annotation.matrix.shape(value, ('schema','knowledge','species','context','tissues','cancer_type',
                                    'alignment','alignment_provenance'))
    if value['schema'] != RESOURCE or value['species'] not in ('Human','Mouse'):
        raise ValueError('Unsupported marker reference species or schema')
    for key in ('context','cancer_type','alignment_provenance'): native.identifier(value[key])
    resource = comparison.KnowledgeResource(**value['knowledge'])
    rows = comparison.load_knowledge(resource)
    tissues = value['tissues']
    if not isinstance(tissues,list) or not tissues or len(set(tissues)) != len(tissues):
        raise ValueError('Reference requires explicit unique tissue terms')
    selected = [r for r in rows if r['original']['species'] == value['species']
        and r['original']['tissue_type'] in tissues and r['original']['cancer_type'] == value['cancer_type']]
    if not selected or set(tissues) != {r['original']['tissue_type'] for r in selected}:
        raise ValueError('Reference context has no matching database records')
    alignment = value['alignment']
    if not isinstance(alignment,dict): raise ValueError('Invalid terminology alignment')
    types = {r['original']['cell_name'] for r in selected}
    for signature, candidates in alignment.items():
        annotation.matrix.sha(signature)
        if not isinstance(candidates,dict): raise ValueError('Invalid signature-bound terminology')
        for candidate, links in candidates.items():
            native.identifier(candidate)
            if not isinstance(links,list): raise ValueError('Invalid terminology links')
            seen = set()
            for link in links:
                annotation.matrix.shape(link, ('external_type','relationship','rationale'))
                native.identifier(link['rationale'])
                if (link['external_type'] not in types or link['external_type'] in seen
                        or link['relationship'] not in ('equivalent','broader','narrower','incompatible')):
                    raise ValueError('Invalid or unavailable reference terminology')
                seen.add(link['external_type'])
    return rows


def resolve_reference(primary, path=None, sha=None):
    if (path is None) != (sha is None): raise ValueError('Exact reference specification and checksum required together')
    if path is not None:
        value = read(path,sha)
    else:
        # One existing library, explicitly configured by the operator. No global
        # fallback, online search, first match, or per-dataset resource catalog.
        from agent.application.shared_resources import ScientificResourceLibrary, current_scientific_library
        library = current_scientific_library()
        if library is None:
            root = os.environ.get('AGENT_SCIENTIFIC_RESOURCE_ROOT')
            if not root: raise ValueError('The operator must deploy a qualified marker reference library for database corroboration')
            library = ScientificResourceLibrary(root)
        matches = [e for e in library.entries() if e['kind'] == 'marker_reference'
            and e['resource']['species'].lower() == primary['species']
            and e['resource']['context'] == primary['context']]
        if len(matches) != 1:
            raise ValueError('No unique compatible marker reference; supply or select an exact reference with matching species and tissue context')
        value = library.check(matches[0])
    if value['species'].lower() != primary['species'] or value['context'] != primary['context']:
        raise ValueError('Marker database species/tissue context differs from the annotation; cross-species mapping is not supported')
    reference(value)
    return value


def arguments(values):
    value = dict(values)
    required = {'annotation_manifest_path','annotation_manifest_sha256','output_dir'}
    if not required <= value.keys() or value.keys() - required - {'reference_spec_path','reference_spec_sha256'}:
        raise ValueError('Invalid reference corroboration arguments')
    for key in value:
        if key.endswith('_sha256'): annotation.matrix.sha(value[key])
        else:
            value[key] = str(value[key]); annotation.matrix.absolute_path(value[key])
    if ('reference_spec_path' in value) != ('reference_spec_sha256' in value):
        raise ValueError('Incomplete exact reference binding')
    return value


def _source(args):
    from .scatac_annotation import verify_annotation
    path = Path(args['annotation_manifest_path']); sha = args['annotation_manifest_sha256']
    # Existing operation proofs/accepted authority avoid any native replay.
    result = verify_annotation(path, expected_sha256=sha)
    manifest = annotation.load_manifest(path,sha)
    return result, manifest


def compare(args, spec):
    """Read native sidecars directly and invoke the shared comparison once."""
    manifest = annotation.load_manifest(args['annotation_manifest_path'],args['annotation_manifest_sha256'])
    root = Path(args['annotation_manifest_path']).parent / 'primary'
    def sidecar(name):
        return native.checked_file(root/name,manifest['sidecars']['primary/'+name]['sha256'])
    groups = read(sidecar('groups.json'))
    genes = sidecar('genes.tsv').read_text().splitlines()
    with sidecar('markers-signature-input.tsv').open(newline='') as stream:
        markers = list(csv.DictReader(stream,delimiter='\t',quoting=csv.QUOTE_NONE))
    knowledge = comparison.load_knowledge(comparison.KnowledgeResource(**spec['knowledge']))
    selected = [r for r in knowledge if r['original']['species']==spec['species']
        and r['original']['tissue_type'] in spec['tissues'] and r['original']['cancer_type']==spec['cancer_type']]
    terms = {r['original']['cell_name'] for r in selected}
    qualified = spec['alignment'].get(manifest['inputs']['signature_resource']['sha256'],{})
    mappings = {}
    for group in groups:
        candidate = group['candidate']
        if candidate is None: continue
        # Case/space normalization is equality, never a biological synonym guess.
        exact = [term for term in terms if ' '.join(term.casefold().split())==' '.join(candidate.casefold().split())]
        mappings[candidate] = qualified.get(candidate, [dict(external_type=term,relationship='equivalent',
            rationale='Exact database term after case and whitespace normalization') for term in exact])
    mapping = dict(schema='external-marker-alignment.v1', species=spec['species'],context=spec['context'],
        rationale=spec['alignment_provenance'],tissues=spec['tissues'],cancer_type=spec['cancer_type'],candidates=mappings)
    compared, selected = comparison.compare_marker_annotation(primary=dict(groups=groups),genes=genes,
        marker_rows=markers,knowledge=knowledge,mapping=mapping,species=spec['species'],context=spec['context'])
    effects = {(r['cluster'],r['gene'].upper()):float(r['avg_logFC']) for r in markers}
    native_groups = {g['group']:g for g in groups}
    records = []
    for group in compared:
        original = native_groups[group['group']]
        group.update(n_cells=original['n_cells'], native_score=original['top_score'])
        for link in group['evidence']:
            for record in selected:
                gene = record['normalized']; row = record['original']
                effect = effects.get((group['group'],gene),0)
                if row['cell_name'] != link['external_type'] or not gene or not effect: continue
                records.append(dict(group=group['group'],candidate=group['primary_candidate'],
                    native_score=original['top_score'],assessment=group['corroboration_state'],
                    reference_term=link['external_type'],relationship=link['relationship'],mapping_basis=link['rationale'],
                    gene=gene,avg_logFC=effect,association='positive' if effect>0 else 'lower_RP',
                    source_row=record['row'],species=spec['species'],tissue=row['tissue_type'],cancer_type=row['cancer_type'],
                    marker=row['marker'],database_symbol=row['Symbol'],marker_source=row['marker_source'],
                    PMID=row['PMID'],Title=row['Title'],database_release=spec['knowledge']['release'],
                    table_sha256=spec['knowledge']['table_sha256']))
    return dict(groups=compared, records=records, mapping=mapping, limitations=LIMITATIONS)


def _summary(evidence, spec, primary):
    groups = []
    for g in evidence['groups'][:50]:
        links = [dict(reference_term=e['external_type'],relationship=e['relationship'],mapping_basis=e['rationale'],
            positive_markers=e['positive_markers'][:20],positive_markers_omitted=max(0,len(e['positive_markers'])-20),
            lower_rp_markers=e['lower_rp_markers'][:20],reference_records=len(e['provenance_rows'])) for e in g['evidence']]
        rows = [r for r in evidence['records'] if r['group']==g['group'] and r['association']=='positive']
        groups.append(dict(group=g['group'],candidate=g['primary_candidate'],n_cells=g['n_cells'],
            native_score=g['native_score'],assessment=g['corroboration_state'],reference_links=links,
            positive_record_examples=[{k:r[k] for k in ('gene','reference_term','relationship','source_row','PMID')} for r in rows[:8]],
            positive_records_omitted=max(0,len(rows)-8)))
    return dict(status='success',contract_version=CONTRACT,annotation_manifest_sha256=primary['manifest_sha256'],
        ordered_cells_sha256=primary['ordered_cells_sha256'],database='CellMarker',database_release=spec['knowledge']['release'],
        database_source=spec['knowledge']['source'],database_table_sha256=spec['knowledge']['table_sha256'],
        species=spec['species'],context=spec['context'],tissues=spec['tissues'],cancer_type=spec['cancer_type'],
        n_groups=len(evidence['groups']),n_records=len(evidence['records']),group_summary=groups,
        groups_omitted=max(0,len(evidence['groups'])-50),state_counts=dict(Counter(g['corroboration_state'] for g in evidence['groups'])),
        limitations=LIMITATIONS)


def _write(path,value):
    data = annotation.matrix.canonical(value)
    with path.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return native.sha256(path)


def load(path,sha=None):
    value = read(path,sha)
    annotation.matrix.shape(value,('contract','profile','arguments','execution_identity','reference','result','files','sources'))
    if value['contract'] != CONTRACT or value['profile'] != PROFILE or arguments(value['arguments']) != value['arguments']:
        raise AuthorityError('Invalid corroboration publication')
    if Path(path).parent != Path(value['arguments']['output_dir']) or Path(path).name != 'corroboration.json':
        raise AuthorityError('Corroboration publication locator mismatch')
    expected = {str(Path(path).parent/name) for name in ('reference-evidence.json','reference-evidence.tsv')}
    if {f['path'] for f in value['files']} != expected or len(value['files']) != 2:
        raise AuthorityError('Invalid corroboration file membership')
    return value, native.sha256(path)


def result(value,path,sha):
    table = next(f for f in value['files'] if f['path'].endswith('.tsv'))
    return dict(value['result'],manifest_path=str(path),manifest_sha256=sha,
                evidence_path=table['path'],evidence_sha256=table['sha256'])


def execute(arguments_, execution_identity=None):
    args = arguments(arguments_); root = Path(args['output_dir']); root.mkdir(parents=True,exist_ok=True)
    if any((root/name).exists() for name in ('corroboration.json','reference-evidence.json','reference-evidence.tsv')):
        raise ValueError('Corroboration output already exists; accepted evidence cannot be overwritten')
    primary, manifest = _source(args)
    spec = resolve_reference(primary,args.get('reference_spec_path'),args.get('reference_spec_sha256'))
    paths = [args['annotation_manifest_path'], spec['knowledge']['snapshot_path'],spec['knowledge']['table_path'],
        *(str(Path(args['annotation_manifest_path']).parent/'primary'/name) for name in ('groups.json','genes.tsv','markers-signature-input.tsv'))]
    if 'reference_spec_path' in args: paths.append(args['reference_spec_path'])
    sources = file_closure(paths)
    evidence = compare(args,spec)
    # Only source-bound, path-free evidence is exported. Private locators remain
    # in the manifest. The database itself is never offered for redistribution.
    with tempfile.TemporaryDirectory(prefix='.corroboration-',dir=root) as temporary:
        stage = Path(temporary)
        _write(stage/'reference-evidence.json',evidence)
        with (stage/'reference-evidence.tsv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=TABLE_FIELDS,delimiter='\t'); writer.writeheader(); writer.writerows(evidence['records'])
            stream.flush();os.fsync(stream.fileno())
        if file_closure(paths) != sources: raise AuthorityError('Corroboration dependencies changed during comparison')
        for name in ('reference-evidence.json','reference-evidence.tsv'): os.link(stage/name,root/name)
    value=dict(contract=CONTRACT,profile=PROFILE,arguments=args,execution_identity=execution_identity,
        reference=spec,result=_summary(evidence,spec,primary),files=file_closure([root/'reference-evidence.json',root/'reference-evidence.tsv']),sources=sources)
    path=root/'corroboration.json';sha=_write(path,value)
    if current() is not None: current().register_execution(KIND,root,execution_identity)
    return verify_publication(path,expected_sha256=sha)


@owned_verification(KIND)
def verify_publication(path,*,expected_sha256):
    value,sha=load(path,expected_sha256)
    if file_closure(f['path'] for f in value['files']+value['sources']) != sorted(value['files']+value['sources'],key=lambda f:f['path']):
        raise AuthorityError('Corroboration publication or dependency integrity changed')
    args=value['arguments'];primary,_=_source(args)
    if value['reference']['species'].lower()!=primary['species'] or value['reference']['context']!=primary['context']:
        raise AuthorityError('Corroboration reference context changed')
    evidence=read(Path(path).parent/'reference-evidence.json')
    # Independent exact deterministic comparison of inexpensive database records;
    # annotation scoring and RP are never repeated here.
    if evidence != compare(args,value['reference']) or value['result'] != _summary(evidence,value['reference'],primary):
        raise AuthorityError('Corroboration records or summary differ from native/reference evidence')
    with (Path(path).parent/'reference-evidence.tsv').open(newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t');rows=list(reader)
    expected=[{k:str(v) if v is not None else '' for k,v in row.items()} for row in evidence['records']]
    if reader.fieldnames!=list(TABLE_FIELDS) or rows!=expected: raise AuthorityError('Reference table differs from published evidence')
    return result(value,path,sha)


def verify_result(args,output):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if value['arguments']!=arguments(args) or _serialize(output)!=verify_publication(output['manifest_path'],expected_sha256=sha):
        raise AuthorityError('Reference result differs from its exact request')


def recover(args, execution_identity):
    args=arguments(args);path=Path(args['output_dir'])/'corroboration.json';value,sha=load(path)
    if value['arguments']!=args or value['execution_identity']!=execution_identity:
        raise AuthorityError('Corroboration recovery identity differs')
    return verify_publication(path,expected_sha256=sha)


def corroborate_cell_type_annotation(annotation_manifest_path,annotation_manifest_sha256,output_dir,
                                     reference_spec_path=None,reference_spec_sha256=None):
    return execute({k:v for k,v in locals().items() if v is not None})


def describe(path,sha,context):
    value,actual=load(path,sha)
    files=context.files([path,*(f['path'] for f in value['files'])])
    if [f for f in files if f['path']!=str(path)]!=value['files']:
        raise AuthorityError('Corroboration file integrity changed')
    identity=dict(kind=KIND,manifest=value,files=files,historical_sources=value['sources'],
        execution_identity=context.execution_for(KIND,path),profile=PROFILE,verifier=VERIFIER)
    return dict(kind=KIND,manifest_path=str(path),manifest_sha256=actual,manifest=value,proof_identity=identity,
        files=files,historical_sources=value['sources'],upstream={},profile=PROFILE,verifier=VERIFIER,qualification={})


def reuse_result(kind,description,proof,kwargs):
    return result(description['manifest'],description['manifest_path'],description['manifest_sha256'])


def proof_metadata(kind,result,context): return dict(result)


def publication_record(context,tool_name,args,output,execution_identity,result_metadata):
    value,sha=load(output['manifest_path'],output['manifest_sha256'])
    if (tool_name!=TOOL or value['execution_identity']!=execution_identity or value['arguments']!=arguments(args)
            or _serialize(result_metadata)!=_serialize(output) or result(value,output['manifest_path'],sha)!=_serialize(output)):
        raise AuthorityError('Corroboration lacks the exact producing execution identity')
    description,key=context.describe(KIND,Path(output['manifest_path']),sha)
    return VerifiedArtifactAuthority(dict(schema_version=2,artifact_type='agent.cell-type-corroboration',artifact_contract=CONTRACT,
        publication_path=output['manifest_path'],manifest_sha256=sha,files=description['files'],
        execution_identity=execution_identity,arguments_sha256=digest(arguments(args)),receipt_sha256=sha,
        upstream={'producer_inputs':digest(description['proof_identity'])},
        resources=dict(scientific_proof_sha256=key,manifest_identity=value,result_metadata=_serialize(output)),
        science_profile=PROFILE,producer_qualification={},verifier=VERIFIER,scope='scientific_correctness.v1',
        completion='succeeded',integrity_scope='artifact_integrity_lineage_historical_sources.v1',
        source_policy='historical_verified_sources.v1',historical_sources=description['historical_sources']))
