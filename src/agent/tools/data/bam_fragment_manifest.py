"""Closed Agent BAM policy, producer record and pure public argument contract."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import scatac_fragments_v2 as v2
from ._fragments_common import canonical, digest, MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY

PROFILE_ID = 'agent-cb-paired-atac.v1'
PRODUCTION_CONTRACT = 'bam-fragment-production.v1'
MAX_PROJECTION = 1024 * 1024
EXCLUSIONS = ('unmapped', 'discordant', 'improper', 'qc_failed', 'split',
              'unsupported_cigar', 'geometry', 'mapq', 'missing_cb', 'short_shifted')
SUPPORT = {'unit': 'read_pairs', 'definition':
    'Number of distinct, well-formed primary read pairs present in the bound BAM that individually pass agent-cb-paired-atac.v1 and produce this exact shifted fragment key.'}
HISTORY = {'coordinates': 'declared_unshifted', 'duplicate_pairs': 'declared_retained',
           'upstream_history': 'not_independently_established'}
RUNTIME_QUALIFICATION = json.loads(Path(__file__).with_name('_bam_fragment_toolchain.json').read_bytes())
PROFILE = dict(artifact_type='agent.bam-fragment-profile', schema_version=1, profile_id=PROFILE_ID,
    source='one-BAM/group/library/namespace;human-hg38|mouse-mm10;no-index-required',
    barcode='CB:corrected_identifier:already_corrected;both-primary-mates-exact;preserve-case-suffix',
    templates='exact-QNAME;one-paired-primary-R1-and-R2;matching-RG-or-both-absent;reciprocal-metadata',
    eligibility='both-mapped/same-contig/proper/QC-pass;no-supplementary-or-primary-SA',
    cigar='M/I/D/S/H/=/X;legal-3prime-clips;5prime-terminal-M/=/X;no-N/P;no-repair',
    geometry='forward.start<=reverse.start;forward.end<=reverse.end;overlap-allowed',
    tlen='both-zero-or-opposite-signed-outer-reference-span;never-coordinate-source',
    mapq='both>=30;255-excluded;before-aggregation', contigs='all-exact-FAI-contigs;no-aliases-or-mito-filter',
    tn5='forward.reference_start+4;reverse.reference_end-5;once;0-based-half-open',
    duplicate_key=['namespace', 'barcode_identifier', 'contig', 'shifted_start', 'shifted_end'],
    duplicates='include-marked-and-unmarked;exact-complete-key-only;no-PCR/optical/molecule-inference',
    support=SUPPORT, strand='absent', source_history=HISTORY,
    exclusion_order=list(EXCLUSIONS), malformed='fail-before-exclusion-accounting',
    packaging=PACKAGING_POLICY, max_projection_bytes=MAX_PROJECTION,
    runtime_versions={'pysam': '0.24.1', 'htslib': '1.24', 'samtools': '1.24'},
    runtime_qualification_sha256=RUNTIME_QUALIFICATION['identity_sha256'])
PROFILE_BYTES = canonical(PROFILE)
ARGUMENTS = ('intake_manifest_path', 'intake_manifest_sha256', 'library_context_path',
    'library_context_sha256', 'reference_bundle_path', 'reference_bundle_sha256',
    'source_path', 'source_sha256', 'source_profile', 'output_dir')


class BamFragmentsError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code='BAM_FRAGMENTS_CONTRACT_INVALID'):
    raise BamFragmentsError(code) from None


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_arguments(arguments):
    try:
        value = dict(arguments); v2.shape(value, ARGUMENTS)
        for key in ARGUMENTS:
            if key.endswith('_path') or key == 'output_dir':
                if isinstance(value[key], Path): value[key] = str(value[key])
                v2.absolute_path(value[key])
            elif key.endswith('_sha256'):
                v2.sha(value[key])
        if value['source_profile'] != PROFILE_ID:
            fail('BAM_FRAGMENTS_PROFILE_UNSUPPORTED')
        return value
    except BamFragmentsError:
        raise
    except (ValueError, TypeError, KeyError):
        fail()


def validate_record(value):
    try:
        v2.shape(value, ('artifact_type', 'schema_version', 'contract_version', 'profile_sha256',
            'arguments', 'source', 'intake', 'context', 'reference', 'namespace', 'group_id',
            'context_identity_sha256', 'source_history', 'runtime', 'qualification', 'canonical', 'outputs'))
        if (value['artifact_type'] != 'agent.bam-fragment-production' or type(value['schema_version']) is not int
                or value['schema_version'] != 1 or value['contract_version'] != PRODUCTION_CONTRACT
                or value['profile_sha256'] != sha_bytes(PROFILE_BYTES) or value['source_history'] != HISTORY): fail()
        validate_arguments(value['arguments']); v2.token(value['namespace']); v2.text(value['group_id'])
        v2.sha(value['context_identity_sha256'])
        for k in ('source', 'intake', 'context'): v2.resource(value[k])
        r = value['reference']
        v2.shape(r, ('manifest_path', 'manifest_sha256', 'reference_identity_sha256', 'species', 'assembly', 'ordered_contig_sha256'))
        v2.absolute_path(r['manifest_path'])
        for k in ('manifest_sha256', 'reference_identity_sha256', 'ordered_contig_sha256'): v2.sha(r[k])
        if r['species'] not in ('human', 'mouse') or r['assembly'] != {'human':'hg38','mouse':'mm10'}[r['species']]: fail()
        runtime = value['runtime']; v2.shape(runtime, ('versions', 'files', 'identity_sha256'))
        if runtime != RUNTIME_QUALIFICATION: fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
        if runtime['versions'] != PROFILE['runtime_versions']: fail('BAM_FRAGMENTS_RUNTIME_MISMATCH')
        if type(runtime['files']) is not dict or not 1 <= len(runtime['files']) <= 128: fail()
        for name, sha in runtime['files'].items(): v2.text(name, 256); v2.sha(sha)
        if runtime['identity_sha256'] != digest({k:runtime[k] for k in ('versions','files')}): fail()
        q = value['qualification']
        v2.shape(q, ('header_sha256', 'sq_sha256', 'stream_sha256', 'n_records', 'n_templates',
                     'n_primary_pairs', 'n_secondary', 'n_supplementary', 'eligible_pairs', 'exclusions'))
        for k in ('header_sha256', 'sq_sha256', 'stream_sha256'): v2.sha(q[k])
        for k in ('n_records','n_templates','n_primary_pairs','n_secondary','n_supplementary','eligible_pairs'):
            v2.number(q[k], low=0)
        v2.shape(q['exclusions'], EXCLUSIONS)
        for count in q['exclusions'].values(): v2.number(count, low=0)
        if q['n_primary_pairs'] != q['n_templates'] or q['n_templates'] != q['eligible_pairs'] + sum(q['exclusions'].values()): fail()
        summary = value['canonical']; v2.shape(summary, v2.SUMMARY_KEYS)
        v2.sha(summary['canonical_record_stream_sha256'])
        for k in v2.SUMMARY_KEYS[1:]: v2.number(summary[k], high=MAX_SUPPORT if k == 'max_support' else MAX_TOTAL)
        if summary['sum_support'] != q['eligible_pairs']: fail('BAM_FRAGMENTS_CONSERVATION_MISMATCH')
        v2.shape(value['outputs'], ('bgzf', 'tabix'))
        for output in value['outputs'].values():
            v2.shape(output, ('sha256','size_bytes')); v2.sha(output['sha256']); v2.number(output['size_bytes'])
        if len(canonical(value)) > 65536: fail()
        return deepcopy(value)
    except BamFragmentsError: raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError): fail()


def load_record(path, expected_sha256):
    try:
        with Path(path).open('rb') as source: payload = source.read(65537)
        if len(payload) > 65536 or sha_bytes(payload) != expected_sha256: fail('BAM_FRAGMENTS_RECORD_MISMATCH')
        return validate_record(json.loads(payload, object_pairs_hook=v2._pairs, parse_constant=lambda _: fail()))
    except BamFragmentsError: raise
    except (ValueError, OSError, UnicodeError): fail('BAM_FRAGMENTS_RECORD_MISMATCH')


def provenance(record, profile_resource, record_resource):
    sources = [{'role':role, 'resource':record[key]} for role,key in
               (('bam','source'), ('intake_manifest','intake'), ('library_context','context'))]
    return dict(kind='bam_fragment_production', contract_version='fragment-producer-provenance.v1',
        profile={'id':PROFILE_ID,'resource':profile_resource}, sources=sources,
        producer_record=record_resource, support=deepcopy(SUPPORT),
        processing={
            'tn5': {'status':'declared','description':'Source declared unshifted; Agent +4/-5 transformation independently recomputed.'},
            'deduplication': {'status':'declared','description':'Source declares duplicates retained; exact Agent key and individually passing pair support independently recomputed.'},
            'mapq_filtering': {'status':'declared','description':'Agent both-mate MAPQ>=30 excluding 255, before aggregation; independently recomputed.'},
            'barcode_correction': {'status':'declared','description':'Context declares corrected CB; exact mate agreement verified; historical correction not established.'}},
        # Agent's eligible subset is exactly reconstructed in qualification.
        # Historical export/selection coverage of the supplied BAM is unknown.
        source_selection='unspecified', verification_requirement='producer-specific-verifier-required.v1')
