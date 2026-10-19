"""Bounded source declarations for the existing direct peak annotation owner.

Observations are source-bound admission evidence, not proof of counting history
or biological validity. Full qualification remains in external_marker_annotation.
"""
import anndata as ad
from pathlib import Path
import csv
from . import marker_annotation as science, peak_h5ad
from .external_marker_annotation import group_rows


def observe(path, sha256):
    path = science.checked_file(path, sha256)
    if path.stat().st_size > peak_h5ad.MAX_BYTES:
        raise ValueError('Peak H5AD exceeds its supported size bound')
    peak_h5ad.inspect(path, privacy=True)
    a = ad.read_h5ad(path, backed='r')
    try:
        declarations = {}
        for name in ('species', 'assembly', 'measurement', 'context', 'count_slot'):
            if name in a.uns:
                science.identifier(a.uns[name])
                declarations[name] = a.uns[name]
        if len(a.obs.columns) > 64:
            raise ValueError('Annotation handoff supports at most 64 observation columns')
        columns = []
        for name in a.obs.columns:
            try:
                science.identifier(name)
                group_rows(a, dict(groups=dict(obs_column=name)))
            except ValueError:
                continue
            columns.append(name)
        # An explicit column/provenance association can qualify a declared
        # clustering role. Mere categorical values or column names cannot.
        grouping = a.uns.get('grouping')
        qualified = []
        if isinstance(grouping, dict):
            for column, descriptor in grouping.items():
                if not isinstance(descriptor, dict) or descriptor.get('role') != 'clustering':
                    continue
                if set(descriptor) != {'role', 'provenance'}:
                    raise ValueError('Clustering metadata requires role and derivation provenance')
                provenance = descriptor['provenance']
                if column not in columns:
                    raise ValueError('Declared grouping does not identify a valid obs grouping column')
                science.identifier(provenance)
                qualified.append(dict(column=column, provenance=provenance))
        return dict(source_sha256=sha256, declarations=declarations,
                    grouping_columns=columns, qualified_groupings=qualified)
    finally:
        a.file.close()


def resource_base(values):
    """Observe an existing resource or shared genome dependency, not a new catalog."""
    from . import annotation_contract as c
    if 'annotation_spec_path' in values:
        return c.specification(values['annotation_spec_path'], values['annotation_spec_sha256'])
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    from agent.tools.data._barcode_qc_binding import resource_qualification
    path, sha = values['qc_reference_manifest_path'], values['qc_reference_manifest_sha256']
    _, bundle, _ = load_scatac_qc_reference_bundle(path, expected_sha256=sha)
    resource_qualification(bundle)
    return dict(species=bundle.species, assembly=bundle.assembly,
                contig_scope=dict(path=path, sha256=sha))


def resource_specification(base, declared, members):
    """Pure binding to the one existing scientific resource contract.

    Provenance/applicability are explicit declarations, not biological validation.
    Paths and hashes come exclusively from the captured trusted registrations.
    """
    from . import annotation_contract as c
    from dataclasses import asdict
    spec = dict(base or {})
    problems = []
    uploaded = declared.uploaded
    if uploaded is None:
        return dict(spec, grouping_provenance=declared.grouping_provenance), problems
    for name in ('species', 'assembly', 'context', 'gene_namespace'):
        if getattr(uploaded, name) is None:
            problems.append('Specify the uploaded references\' ' + name.replace('_', ' ') + '.')
    for name in ('species', 'assembly', 'context'):
        if getattr(uploaded, name) is not None and getattr(uploaded, name) != getattr(declared, name):
            problems.append('Uploaded reference ' + name + ' conflicts with the annotation task; correct the declaration or supply compatible files.')
    if uploaded.gene_namespace not in (None, 'gene_symbol'):
        problems.append('Native MAESTRO refgenes and signatures require a matching gene_symbol namespace; identifier conversion is not automatic.')
    for role, semantics, normalization in (
            ('gene', 'maestro-refgenes-transcripts-exons.v1', 'none'),
            ('signature', 'candidate-gene-list.v1', 'uppercase')):
        source = getattr(uploaded, role + '_source')
        if source is None:
            if role + '_resource' not in spec:
                problems.append('Upload a MAESTRO refgenes transcript/exon table.' if role == 'gene' else
                                'Upload a marker-signature TSV with candidate and gene columns.')
            continue
        index = int(source.split('.')[1])
        if index >= len(members):
            raise ValueError('Uploaded reference is outside the captured attachments.')
        member = members[index]
        for name in ('provenance', 'release'):
            if getattr(uploaded, role + '_' + name) is None:
                problems.append('Describe the uploaded ' + role + ' resource ' + name + ' (publisher/derivation and release are required).')
        if all(getattr(uploaded, name) is not None for name in ('species', 'assembly', 'context', 'gene_namespace', role+'_provenance', role+'_release')):
            spec[role + '_resource'] = asdict(science.Resource(
                member['source_path'], member['source_sha256'], getattr(uploaded, role + '_provenance'),
                getattr(uploaded, role + '_release'), uploaded.species, uploaded.assembly,
                uploaded.context, uploaded.gene_namespace, normalization, semantics))
    spec.update(schema=c.EXTERNAL_SPEC, method=c.EXTERNAL_PRIMARY, measurement=declared.measurement,
                species=declared.species, assembly=declared.assembly, context=declared.context,
                coordinate_system='zero-based-half-open', grouping_provenance=declared.grouping_provenance)
    return spec, problems


def qualify_resources(spec):
    """Admission of user resource bytes under the existing native MAESTRO owner.

    This validates the supported refgenes table, signed-list inputs and their
    correspondence. It neither assigns labels nor certifies biological accuracy.
    The computation and independent result replay retain their existing owners.
    """
    from .external_marker_annotation import contig_lengths
    contigs = contig_lengths(spec)
    gene, signature = (science.Resource(**spec[k]) for k in ('gene_resource', 'signature_resource'))
    gene.check(spec['species'], spec['assembly'], 'maestro-refgenes-transcripts-exons.v1', 'none')
    signature.check(spec['species'], spec['assembly'], 'candidate-gene-list.v1', 'uppercase')
    if gene.gene_namespace != signature.gene_namespace or gene.gene_namespace != 'gene_symbol' or signature.context != spec['context']:
        raise ValueError('Reference gene namespaces or signature biological context do not match.')
    if any(Path(r.path).stat().st_size > 64 * 1024**2 for r in (gene, signature)):
        raise ValueError('Annotation resource exceeds the 64 MiB per-component admission bound.')
    genes = set()
    transcript_count = 0
    columns = 'bin name chrom strand txStart txEnd cdsStart cdsEnd exonCount exonStarts exonEnds score name2 cdsStartStat cdsEndStat exonFrames'.split()
    with Path(gene.path).open(newline='') as stream:
        rows = csv.DictReader(stream, delimiter='\t')
        if rows.fieldnames != columns:
            raise ValueError('Gene annotation requires the native MAESTRO refgenes transcript/exon columns.')
        for count, row in enumerate(rows, 1):
            if count > 200000 or None in row or None in row.values():
                raise ValueError('Malformed or oversized refgenes table.')
            science.identifier(row['name']); science.identifier(row['name2'])
            start, end, coding_start, coding_end, n = (int(row[k]) for k in ('txStart', 'txEnd', 'cdsStart', 'cdsEnd', 'exonCount'))
            starts = [int(v) for v in row['exonStarts'].rstrip(',').split(',')]
            ends = [int(v) for v in row['exonEnds'].rstrip(',').split(',')]
            science.identifier(row['chrom'])
            if (row['strand'] not in ('+', '-') or not 0 <= start < end
                    or row['chrom'] in contigs and end > contigs[row['chrom']]
                    or not start <= coding_start <= coding_end <= end
                    or n < 1 or len(starts) != n or len(ends) != n
                    or any(not start <= a < b <= end for a, b in zip(starts, ends))
                    or any(b > a for b, a in zip(ends, starts[1:]))):
                raise ValueError('Refgenes transcript/exon coordinates or genome chromosome compatibility are invalid.')
            genes.add(row['name2'].upper())
            transcript_count += 1
    if not genes:
        raise ValueError('Gene annotation contains no transcripts.')
    signatures = science.read_signatures(signature.path)
    if not any(g in genes for values in signatures.values() for g in values):
        raise ValueError('Signature gene identifiers have no correspondence with the supplied gene annotation.')
    return dict(gene_sha256=gene.sha256, signature_sha256=signature.sha256,
                gene_count=len(genes), transcript_count=transcript_count, candidate_count=len(signatures),
                interpretation='declared_provenance_and_applicability;not_independent_biological_validation')
