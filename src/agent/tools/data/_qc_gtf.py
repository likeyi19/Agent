"""Narrow GENCODE transcript decoding through the installed mature pysam parser.

No gene-model inference, GFF support, automatic installation, or source
qualification. GTF input is plain UTF-8; coordinates are decoded by asGTF.
"""
import importlib.util
import json
import re
from pathlib import Path
import subprocess

from ._qc_bedtools import _sha, ENVIRONMENT
from .scatac_qc_profile import fail, ScATACQCError

PARSER_PROFILE = 'pysam-0.24.1-gencode-transcript-gtf.v1'


def _qualified_attributes(record):
    """Reject permissive parser repairs by exact narrow-dialect round trip.

    pysam owns decoding. This is a syntax acceptance check, not an alternative
    attribute parser: GENCODE quoted strings or unquoted integers, separated
    by one space, each terminated by a semicolon. No escapes or dialect repair.
    """
    attrs = list(record.attribute_string2iterator(record.attributes))
    encoded = []
    for key, value in attrs:
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) is None:
            fail('QC_GTF_INVALID')
        if type(value) is str:
            if any(ord(c) < 32 or c in '\\";' for c in value):
                fail('QC_GTF_INVALID')
            encoded.append(f'{key} "{value}";')
        elif type(value) is int:
            encoded.append(f'{key} {value};')
        else:
            fail('QC_GTF_INVALID')
    if not attrs or ' '.join(encoded) != record.attributes.strip(' '):
        fail('QC_GTF_INVALID')
    return attrs


def verify_gtf_parser():
    from .scatac_qc_reference import GTF_RUNTIME_SHA256
    try:
        spec = importlib.util.find_spec('pysam')
        if spec is None or spec.origin is None:
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        root = Path(spec.origin).parent
        identity_path = Path(__file__).with_name('_qc_gtf_toolchain.json')
        if _sha(identity_path) != GTF_RUNTIME_SHA256:
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        expected = json.loads(identity_path.read_text())
        if any(_sha(root / p) != sha for p, sha in expected['files'].items()):
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        found = {}
        for extension in expected['extensions']:
            r = subprocess.run(['/usr/bin/ldd', str(root / extension)], env=ENVIRONMENT,
                               capture_output=True, timeout=15, check=False)
            if r.returncode:
                fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
            for line in r.stdout.decode().splitlines():
                words = line.split()
                path = words[2] if '=>' in words else words[0]
                if path.startswith('/'):
                    key = Path(path).name
                    digest = _sha(path)
                    if key in found and found[key] != digest:
                        fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
                    found[key] = digest
        if found != expected['libraries']:
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        import pysam
        if pysam.__version__ != expected['version']:
            fail('QC_ANNOTATION_PARSER_UNQUALIFIED')
        return pysam
    except (OSError, ValueError, TypeError, ImportError, subprocess.SubprocessError):
        fail('QC_ANNOTATION_PARSER_UNQUALIFIED')


def gencode_transcripts(path):
    from .scatac_qc_reference import _lines, _decimal, _text, MAX_TRANSCRIPTS
    ps = verify_gtf_parser()
    n = 0
    try:
        # _lines enforces a pre-parser byte bound. Only syntax is delegated;
        # GENCODE attribute/coordinate eligibility remains an explicit contract.
        parser = ps.asGTF()
        for line in _lines(path):
            line.decode('utf-8')
            if line.startswith(b'#'):
                continue
            raw = line[:-1]
            record = parser(raw, len(raw))
            columns = tuple(record)
            if len(columns) != 9:
                fail('QC_GTF_INVALID')
            left, right = _decimal(columns[3]), _decimal(columns[4])
            if not 1 <= left <= right or record.start != left-1 or record.end != right:
                fail('QC_GTF_INVALID')
            if record.strand not in ('+', '-'):
                fail('QC_GTF_INVALID')
            attrs = _qualified_attributes(record)
            if record.feature != 'transcript':
                continue
            n += 1
            if n > MAX_TRANSCRIPTS:
                fail('QC_RESOURCE_LIMIT')
            # GENCODE legitimately repeats 'tag'. Scientific identity attributes
            # must be unique even when a permissive parser would keep the last.
            values = {}
            for key in ('gene_id', 'transcript_id', 'gene_type'):
                matches = [value for name, value in attrs if name == key]
                if len(matches) != 1:
                    fail('QC_GTF_INVALID')
                _text(matches[0]); values[key] = matches[0]
            _text(record.contig)
            yield (record.contig, record.start, record.end, record.strand,
                   values['gene_id'], values['transcript_id'], values['gene_type'])
    except ScATACQCError:
        raise
    except (ValueError, TypeError, AttributeError, IndexError, UnicodeError):
        fail('QC_GTF_INVALID')
