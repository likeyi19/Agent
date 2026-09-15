"""Pinned, group-level external marker associations, separate from primary scoring."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .marker_annotation import checked_file, identifier, sha256

PROFILE = 'external-marker-associations.v1'
FIELDS = ['species','tissue_class','tissue_type','uberonongology_id','cancer_type',
          'cell_type','cell_name','cellontology_id','marker','Symbol','GeneID','Genetype',
          'Genename','UNIPROTID','technology_seq','marker_source','PMID','Title','journal','year']


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,
                                     separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class KnowledgeResource:
    snapshot_path: str
    snapshot_sha256: str
    table_path: str
    table_sha256: str
    source: str
    release: str
    retrieved_at: str
    license_statement: str
    normalization: str = 'CellMarker-Symbol-uppercase-unambiguous.v1'
    gene_namespace: str = 'gene_symbol'


def load_knowledge(resource):
    for value in asdict(resource).values():
        identifier(value)
    if resource.normalization != 'CellMarker-Symbol-uppercase-unambiguous.v1' or resource.gene_namespace != 'gene_symbol':
        raise ValueError('Unsupported normalization or namespace')
    checked_file(resource.snapshot_path,resource.snapshot_sha256)
    p=checked_file(resource.table_path,resource.table_sha256)
    with p.open(newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if reader.fieldnames != FIELDS:
            raise ValueError('Unexpected pinned CellMarker table schema')
        rows=list(reader)
    if not rows or any(None in r or None in r.values() for r in rows):
        raise ValueError('Malformed knowledge table')
    # Only native explicit marker -> Symbol associations. No alias guessing.
    def usable_symbol(symbol):
        return bool(symbol) and symbol not in ('NONE','NA','N/A','-') and all(c.isalnum() or c in '.-_' for c in symbol)
    aliases={}
    for r in rows:
        symbol=r['Symbol'].upper()
        if usable_symbol(symbol):
            aliases.setdefault((r['species'],r['marker'].upper()),set()).add(symbol)
    result=[]
    for i,r in enumerate(rows,2):
        symbol=r['Symbol'].upper()
        usable=usable_symbol(symbol)
        status=('unmappable' if not usable else 'ambiguous' if len(aliases[(r['species'],r['marker'].upper())])!=1
                else 'exact_symbol' if r['marker'].upper()==symbol else 'explicit_database_symbol')
        result.append(dict(row=i, original=r, normalized=symbol if status not in ('ambiguous','unmappable') else None,
                           mapping_status=status))
    return result


def validate_marker_annotation_candidates(*, primary_path, primary_sha256, genes_path, genes_sha256,
        markers_path, markers_sha256, resource: KnowledgeResource, mapping_path, mapping_sha256,
        species, context, profile):
    """Consume pinned M13.3 group evidence plus its retained genes/filtered markers.

    Supported means positive marker association evidence exists at the mapped
    external type's scope. Competing associations remain insufficient evidence:
    membership does not establish marker exclusivity or identity contradiction.
    This positive-only profile cannot emit a scientifically proven conflicting
    state. No state authorizes accepted identity or calibrated confidence.
    """
    if profile != PROFILE or species not in ('Human','Mouse'):
        raise ValueError('Unsupported explicit profile/species')
    identifier(context)
    primary=json.loads(checked_file(primary_path,primary_sha256).read_text())
    genes=json.loads(checked_file(genes_path,genes_sha256).read_text())
    if not isinstance(genes,list) or not genes or len({identifier(g).upper() for g in genes})!=len(genes):
        raise ValueError('Ambiguous primary gene space')
    gene_set={g.upper() for g in genes}
    mapping=json.loads(checked_file(mapping_path,mapping_sha256).read_text())
    if mapping['schema']!='external-marker-alignment.v1' or mapping['species']!=species or mapping['context']!=context:
        raise ValueError('Species/context mapping mismatch')
    if mapping['resource_table_sha256']!=resource.table_sha256 or mapping['primary_sha256']!=primary_sha256:
        raise ValueError('Mapping dependency mismatch')
    identifier(mapping['rationale'])
    tissues=mapping['tissues']
    if not tissues or len(set(tissues))!=len(tissues):
        raise ValueError('Explicit unique tissue vocabulary required')
    knowledge=load_knowledge(resource)
    available={r['original']['tissue_type'] for r in knowledge if r['original']['species']==species}
    if not set(tissues)<=available:
        raise ValueError('Unsupported resource tissue context')
    selected=[r for r in knowledge if r['original']['species']==species and r['original']['tissue_type'] in tissues
              and r['original']['cancer_type']==mapping['cancer_type']]
    if not selected:
        raise ValueError('No resource records for explicit context')
    types={r['original']['cell_name'] for r in selected}
    group_rows=primary['groups']
    group_ids=[identifier(r['group']) for r in group_rows]
    if len(set(group_ids))!=len(group_ids):
        raise ValueError('Duplicate primary groups')
    for g in group_rows:
        if g['status'] not in ('candidate','ambiguous','unresolved','insufficient_evidence') or g['accepted_identity'] is not None:
            raise ValueError('Unsupported primary annotation state')
        if (g['status']=='candidate') != (g['candidate'] is not None):
            raise ValueError('Inconsistent primary candidate state')
    effects={g:{} for g in group_ids}
    with checked_file(markers_path,markers_sha256).open(newline='') as f:
        import math
        for row in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
            group,gene,value=row['cluster'],row['gene'].upper(),float(row['avg_logFC'])
            if group not in effects or gene not in gene_set or gene in effects[group] or not math.isfinite(value):
                raise ValueError('Invalid primary marker identity/value')
            effects[group][gene]=value
    align=mapping['candidates']
    required={g['candidate'] for g in group_rows if g['candidate'] is not None}
    if set(align)!=required:
        raise ValueError('Mapping must explicitly cover exactly the primary candidate vocabulary')
    for candidate,entries in align.items():
        seen=set()
        for e in entries:
            if e['relationship'] not in ('equivalent','broader','narrower','incompatible') or e['external_type'] not in types or e['external_type'] in seen:
                raise ValueError('Invalid or duplicate cell-type alignment')
            identifier(e['rationale']);seen.add(e['external_type'])
    result=[]
    for g in group_rows:
        evidence=[]
        for link in align.get(g['candidate'],[]):
            records=[r for r in selected if r['original']['cell_name']==link['external_type']]
            markers={r['normalized'] for r in records if r['normalized']}
            support=sorted(x for x in markers if effects[g['group']].get(x,0)>0)
            lower=sorted(x for x in markers if effects[g['group']].get(x,0)<0)
            evidence.append(dict(**link, external_marker_set_size=len(markers),positive_markers=support,
                lower_rp_markers=lower,unmeasured=sorted(markers-gene_set),
                measured_without_retained_marker=sorted((markers&gene_set)-effects[g['group']].keys()),
                provenance_rows=[r['row'] for r in records], excluded_mapping_rows=[r['row'] for r in records if r['normalized'] is None]))
        positive=[e for e in evidence if e['relationship'] in ('equivalent','broader') and e['positive_markers']]
        competing=[e for e in evidence if e['relationship']=='incompatible' and e['positive_markers']]
        state=('not_applicable' if g['candidate'] is None else 'unmapped' if not evidence
               else 'insufficient_evidence' if competing else 'supported' if positive else 'insufficient_evidence')
        result.append(dict(group=g['group'],primary_candidate=g['candidate'],primary_state=g['status'],
            accepted_identity=None,validation_state=state,evidence=evidence,
            validation_scope='mapped_external_type_marker_associations_only',
            compatible_positive_associations=bool(positive),competing_positive_associations=bool(competing),
            conflict_assessment='not_established_by_positive_marker_membership',
            reason={'not_applicable':'primary_has_no_candidate','unmapped':'no_declared_alignment',
                                        'supported':'positive_compatible_associations_not_identity_acceptance',
                    'insufficient_evidence':('competing_nonexclusive_associations_require_review' if competing else 'no_positive_equivalent_or_broader_associations')}[state]))
    dependencies=dict(primary=dict(path=str(primary_path),sha256=primary_sha256),genes=dict(path=str(genes_path),sha256=genes_sha256),
                      markers=dict(path=str(markers_path),sha256=markers_sha256),resource=asdict(resource),
                      mapping=dict(path=str(mapping_path),sha256=mapping_sha256),species=species,context=context)
    output=dict(schema='agent.external-marker-validation.v1',profile=profile,dependencies=dependencies,groups=result,
                interpretation='RNA/protein literature associations with accessibility-derived evidence; not assay equivalence or independent trials',
                source_row_reference='1-based TSV line; header is line 1',normalization_audit=[dict(row=r['row'],original_marker=r['original']['marker'],
                normalized=r['normalized'],mapping_status=r['mapping_status']) for r in selected],
                accepted_identity_policy='none')
    for p,h in [(primary_path,primary_sha256),(genes_path,genes_sha256),(markers_path,markers_sha256),
                (mapping_path,mapping_sha256),(resource.snapshot_path,resource.snapshot_sha256),(resource.table_path,resource.table_sha256)]:
        checked_file(p,h)
    output['identity_sha256']=identity(output)
    return output
