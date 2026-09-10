"""Physical BAM projection, artifact binding and bounded external sorting.

No pair eligibility, endpoint construction or support aggregation lives here.
Limits are post-decoding: htslib allocations for one record/header are not bounded
by the projection limit. Read sequences and qualities are never serialized.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import bam_fragment_manifest as m, _external_fragment_io as physical
from . import scatac_library_context as lc, scatac_reference as ref, raw_scatac_manifest as raw
from . import _raw_bam
from ._fragments_common import canonical, digest, run_stage
from .scatac_fragments_v2_verifier import file_sha256, take_snapshots, check_snapshots

resource = physical.resource


def _observed_runtime():
    ps = _raw_bam._backend()
    versions = dict(pysam=ps.__version__, htslib=ps.version.__htslib_version__, samtools=ps.__samtools_version__)
    if versions != m.PROFILE['runtime_versions']: m.fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
    root = Path(ps.__file__).parent
    # Hash the installed decoder extensions and bundled shared libraries, not
    # merely version strings. Runtime identity remains checked on fresh verify.
    files = {p.name:file_sha256(p) for p in sorted(root.glob('*.so'))}
    for p in sorted((root.parent / 'pysam.libs').glob('*')):
        if p.is_file(): files[p.name] = file_sha256(p)
    for name in ('__init__.py', 'version.py'):
        files[name] = file_sha256(root / name)
    for stem in ('libcalignedsegment', 'libcalignmentfile', 'libchtslib', 'libcfaidx'):
        extension = next(root.glob(stem + '.*.so'))
        result = subprocess.run(['/usr/bin/ldd', str(extension)], capture_output=True, check=False)
        if result.returncode: m.fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
        for line in result.stdout.decode().splitlines():
            fields = line.split()
            path = fields[2] if '=>' in fields else fields[0]
            if path.startswith('/'):
                key = 'linked.' + Path(path).name; sha = file_sha256(path)
                if key in files and files[key] != sha: m.fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
                files[key] = sha
    value = dict(versions=versions, files=files)
    return value | {'identity_sha256':digest(value)}


def runtime_identity():
    observed = _observed_runtime()
    expected = m.RUNTIME_QUALIFICATION
    if observed != expected: m.fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
    return observed


def bind(arguments):
    a = m.validate_arguments(arguments)
    _, intake, _ = raw.load_raw_intake_manifest(a['intake_manifest_path'], expected_sha256=a['intake_manifest_sha256'])
    _, context, _ = lc.load_scatac_library_processing_context(a['library_context_path'], expected_sha256=a['library_context_sha256'])
    # M11.1 explicitly permits an identical intake at another manifest locator.
    # Its content identity and exact raw source/group binding remain mandatory.
    if context.intake.manifest_sha256 != a['intake_manifest_sha256']: m.fail('BAM_FRAGMENTS_CONTEXT_MISMATCH')
    lc.validate_library_context_intake_binding(context, intake_manifest_path=a['intake_manifest_path'])
    if len(context.libraries) != 1: m.fail('BAM_FRAGMENTS_CLASS_UNSUPPORTED')
    lib = context.libraries[0]
    if (len(lib.groups) != 1 or lib.input_kind is not raw.InputKind.BAM
            or lib.barcode_interpretation is not lc.BarcodeInterpretation.CORRECTED_IDENTIFIER
            or lib.correction_policy is not lc.CorrectionPolicy.ALREADY_CORRECTED or lib.whitelist is not None
            or lib.groups[0].source is not raw.BarcodeSource.BAM_CELL_IDENTIFIER or lib.groups[0].locator != 'CB'):
        m.fail('BAM_FRAGMENTS_BARCODE_POLICY_UNSUPPORTED')
    group = next(g for g in intake.groups if g.id == lib.groups[0].group_id)
    if len(group.file_ids) != 1 or group.structure is not raw.StructureState.SUPPORTED: m.fail('BAM_FRAGMENTS_CLASS_UNSUPPORTED')
    source = next(f for f in intake.files if f.id == group.file_ids[0])
    if source.path != a['source_path'] or source.input_kind is not raw.InputKind.BAM: m.fail('BAM_FRAGMENTS_SOURCE_MISMATCH')
    reference, contigs, paths = physical.reference(a['reference_bundle_path'], a['reference_bundle_sha256'])
    _, bundle, _ = ref.load_scatac_reference_bundle(a['reference_bundle_path'], expected_sha256=a['reference_bundle_sha256'])
    if (group.species.value != reference['species'] or group.target_genome_assembly.value != reference['assembly']
            or group.source_genome_assembly.value != reference['assembly']): m.fail('BAM_FRAGMENTS_REFERENCE_MISMATCH')
    paths += [a['source_path'], a['intake_manifest_path'], a['library_context_path']]
    snapshots = take_snapshots(paths)
    # Recover only original explicit declarations, preserving absence and M10
    # authority. Reinspection is bounded; full qualification is separate.
    def declaration(field):
        values = {x.value for g in intake.groups for x in getattr(g, field).assertions
                  if x.source is raw.EvidenceSource.USER_DECLARATION}
        if len(values) > 1: m.fail('BAM_FRAGMENTS_INTAKE_MISMATCH')
        return next(iter(values), None)
    assays = set()
    for e in intake.evidence:
        if e.source is raw.EvidenceSource.USER_DECLARATION:
            try: item = json.loads(e.value)
            except (ValueError, TypeError): continue
            if isinstance(item, dict) and item.get('contract') == _raw_bam.CONTRACT and item.get('fact') == 'assay':
                assays.add(item['value'])
    if len(assays) > 1: m.fail('BAM_FRAGMENTS_INTAKE_MISMATCH')
    roots = sorted({f.selection_root if f.selection is raw.SelectionBasis.DIRECTORY_DISCOVERY else f.path for f in intake.files})
    current = _raw_bam.inspect_bam_inputs(roots, species=declaration('species'),
        assay=_raw_bam.BamAssay(next(iter(assays))) if assays else None,
        source_genome_assembly=declaration('source_genome_assembly'))
    if raw.canonical_manifest_bytes(current) != raw.canonical_manifest_bytes(intake): m.fail('BAM_FRAGMENTS_INTAKE_MISMATCH')
    check_snapshots(snapshots)
    return dict(reference=reference, contigs=contigs, bundle=bundle, library=lib, group=group,
        namespace=lib.namespace, group_id=group.id, context_identity_sha256=context.context_identity_sha256,
        source=resource(a['source_path'], a['source_sha256']),
        intake=resource(a['intake_manifest_path'], a['intake_manifest_sha256']),
        context=resource(a['library_context_path'], a['library_context_sha256']), snapshots=snapshots)


def project(path, target):
    """Raw decoder fields only, including CIGAR and sequence length for checking.

    Exact QNAME is hex encoded solely for collision-free sort framing. Header and
    all relevant tags are subsequently checked by each scientific implementation.
    """
    ps = _raw_bam._backend(); stream = hashlib.sha256(); count = 0
    try:
        with ps.AlignmentFile(path, 'rb', require_index=False, threads=1) as bam, target.open('xb') as output:
            if not bam.is_bam: m.fail('BAM_FRAGMENTS_FORMAT_UNSUPPORTED')
            bam.check_truncation()
            header = bam.header.to_dict()
            if len(canonical(header)) > _raw_bam.MAX_HEADER_BYTES: m.fail('BAM_FRAGMENTS_HEADER_INVALID')
            dictionary = list(zip(bam.references, bam.lengths))
            for r in bam.fetch(until_eof=True):
                q = r.query_name
                if not isinstance(q, str) or re.fullmatch(r'[!-?A-~]{1,254}', q) is None or q == '*':
                    m.fail('BAM_FRAGMENTS_QNAME_INVALID')
                tags = [(k,v,t) for k,v,t in r.get_tags(with_value_type=True) if k in ('CB','RG','SA')]
                # Refuse non-string relevant tag payloads before serialization;
                # arrays can otherwise escape bounded scalar projections.
                if any(t != 'Z' or not isinstance(v,str) for k,v,t in tags): m.fail('BAM_FRAGMENTS_TAG_INVALID')
                value = dict(flag=r.flag, rid=r.reference_id, start=r.reference_start,
                    end=r.reference_end, mrid=r.next_reference_id, mpos=r.next_reference_start,
                    tlen=r.template_length, mapq=r.mapping_quality, cigar=r.cigartuples,
                    seq_len=None if r.query_sequence is None else r.query_length, tags=tags)
                data = q.encode('ascii').hex().encode() + b'\t' + canonical(value) + b'\n'
                if len(data) > m.MAX_PROJECTION: m.fail('BAM_FRAGMENTS_RECORD_LIMIT')
                output.write(data); stream.update(data); count += 1
                if count > m.MAX_TOTAL: m.fail('BAM_FRAGMENTS_SUPPORT_OVERFLOW')
            return header, dictionary, stream.hexdigest(), count
    except m.BamFragmentsError: raise
    except (OSError, ValueError, TypeError, UnicodeError, OverflowError): m.fail('BAM_FRAGMENTS_DECODE_FAILED')


def sort_templates(source, target, directory, runtime):
    run_stage([runtime.sort,'--parallel=1','-S','64M','-t','\t','-k1,1','-T',directory,'-o',target,'--',source],
              cwd=directory, code='BAM_FRAGMENTS_SORT_FAILED')


def records(path):
    with path.open('rb') as source:
        for line in iter(lambda:source.readline(m.MAX_PROJECTION+1), b''):
            if len(line) > m.MAX_PROJECTION or not line.endswith(b'\n'): m.fail('BAM_FRAGMENTS_RECORD_LIMIT')
            key, data = line.split(b'\t',1)
            yield key, json.loads(data)


def sequence_md5(bundle, names):
    """Bounded reference reads through htslib, using the already existing FAI."""
    ps = _raw_bam._backend(); result = {}
    with ps.FastaFile(bundle.genome.fasta.path, filepath_index=bundle.genome.fai.path) as fasta:
        for name in names:
            h = hashlib.md5()
            for start in range(0, fasta.get_reference_length(name), 65536):
                h.update(fasta.fetch(name,start,start+65536).upper().encode('ascii'))
            result[name] = h.hexdigest()
    return result


def distinct_count(barcode_file, directory, runtime):
    target = directory / 'barcodes.sorted'
    run_stage([runtime.sort,'--parallel=1','-S','64M','-u','-T',directory,'-o',target,'--',barcode_file],
              cwd=directory, code='BAM_FRAGMENTS_SORT_FAILED')
    with target.open('rb') as source: return sum(1 for _ in source)
