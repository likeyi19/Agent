"""Frozen explicit selection policy; no statistical cell calling."""
import base64
from fractions import Fraction
import re

from . import _barcode_qc_contract as qc

METHOD = 'explicit_qc_thresholds.v1'
REASONS = ('QC_FRAGMENT_COUNT_BELOW_MIN', 'TSS_ENRICHMENT_UNDEFINED',
    'TSS_ENRICHMENT_BELOW_MIN', 'TSS_FLANK_EVIDENCE_BELOW_MIN',
    'QC_FRAGMENT_COUNT_ABOVE_MAX', 'NUCLEOSOME_SIGNAL_UNDEFINED',
    'NUCLEOSOME_SIGNAL_ABOVE_MAX')
PROFILE = dict(selection_method=METHOD, cell_call_method='none', cell_call_state='not_assessed',
    comparisons='minimum_inclusive_maximum_inclusive', undefined_tss='fail',
    undefined_enabled_nucleosome='fail', absent_optional='disabled',
    flank_metric='tss_left_flank_count+tss_right_flank_count', depth_metric='n_qc_fragment_records',
    threshold_encoding='nonnegative-int-or-exact-decimal-or-rational-string.v1',
    identity_encoding='scatac-cell.v1.base64url-unpadded', reason_order=list(REASONS))
PROFILE_SHA256 = qc.digest(PROFILE)
REQUIRED = ('min_qc_fragment_records', 'min_tss_enrichment')
OPTIONAL = ('min_tss_flank_evidence', 'max_qc_fragment_records', 'max_nucleosome_signal')
MAX_INTEGER = 10**18


class SelectionError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code):
    raise SelectionError(code)


def integer(value):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        fail('SELECTION_THRESHOLD_INVALID')
    return value


def rational(value):
    if type(value) is int:
        return Fraction(integer(value))
    if type(value) is not str or len(value) > 80 or not re.fullmatch(
            r'(0|[1-9][0-9]{0,18})(?:\.[0-9]{1,18}|/[1-9][0-9]{0,18})?', value):
        fail('SELECTION_THRESHOLD_INVALID')
    result = Fraction(value)
    if max(result.numerator, result.denominator) > MAX_INTEGER:
        fail('SELECTION_THRESHOLD_INVALID')
    return result


def thresholds(args):
    out = {}
    for key in (*REQUIRED, *OPTIONAL):
        value = args.get(key)
        if value is None:
            if key in REQUIRED: fail('SELECTION_THRESHOLD_REQUIRED')
            out[key] = None
        elif key in ('min_tss_enrichment', 'max_nucleosome_signal'):
            out[key] = qc.fraction_json(rational(value))
        else:
            out[key] = integer(value)
    return out


def encode_cell_id(namespace, barcode):
    qc.identity(namespace, barcode)
    return 'scatac-cell.v1.' + '.'.join(base64.urlsafe_b64encode(s.encode('ascii')).decode('ascii').rstrip('=')
        for s in (namespace, barcode))


def decode_cell_id(value):
    if type(value) is not str or len(value) > 529 or not value.startswith('scatac-cell.v1.'):
        fail('SELECTION_CELL_ID_INVALID')
    parts = value[len('scatac-cell.v1.'):].split('.')
    if len(parts) != 2 or any(not re.fullmatch('[A-Za-z0-9_-]+', p) for p in parts):
        fail('SELECTION_CELL_ID_INVALID')
    try:
        pair = tuple(base64.b64decode(p + '=' * (-len(p) % 4), altchars=b'-_', validate=True).decode('ascii') for p in parts)
        if encode_cell_id(*pair) != value: fail('SELECTION_CELL_ID_INVALID')
    except (ValueError, UnicodeError): fail('SELECTION_CELL_ID_INVALID')
    return pair


def decide(counts, policy):
    _, depth, c, l, r, free, mono, _ = counts
    reasons = []
    if depth < policy['min_qc_fragment_records']: reasons.append(REASONS[0])
    t = policy['min_tss_enrichment']
    if l+r == 0: reasons.append(REASONS[1])
    elif 200*c*t['denominator'] < 101*(l+r)*t['numerator']: reasons.append(REASONS[2])
    if policy['min_tss_flank_evidence'] is not None and l+r < policy['min_tss_flank_evidence']: reasons.append(REASONS[3])
    if policy['max_qc_fragment_records'] is not None and depth > policy['max_qc_fragment_records']: reasons.append(REASONS[4])
    t = policy['max_nucleosome_signal']
    if t is not None:
        if free == 0: reasons.append(REASONS[5])
        elif mono*t['denominator'] > free*t['numerator']: reasons.append(REASONS[6])
    return tuple(reasons)
