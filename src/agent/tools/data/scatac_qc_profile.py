"""Frozen M11.4a science. No fragment processing or cell selection."""
from dataclasses import asdict, dataclass
import hashlib
import json
from fractions import Fraction


class ScATACQCError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code='QC_CONTRACT_INVALID'):
    raise ScATACQCError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class QCScientificProfile:
    contract_version: str = 'scatac-qc-science.v1'
    depth: str = 'canonical-records;all-retained-contigs'
    selection_depth: str = 'canonical-records;declared-primary-nuclear-qc-contigs'
    endpoints: str = '[start,start+1);[end-1,end);weight-one;no-shift'
    tss_method: str = 'agent-tss-incidence-101-200.v1'
    center: tuple = (-50, 51)
    left_flank: tuple = (-2000, -1900)
    right_flank: tuple = (1901, 2001)
    multiplicity: str = 'every-distinct-contig-position-strand-TSS;no-sampling'
    ratio: str = '200*center/(101*(left_flank+right_flank))'
    zero_background: str = 'null;ZERO_TSS_BACKGROUND;no-population-borrowing'
    length_bins: tuple = (1, 147, 294)
    length_scope: str = 'declared-primary-nuclear-qc-contigs'
    histogram: str = 'length-1-through-1000;overflow-ge-1001;record-counts'
    nucleosome_ratio: str = 'mononucleosomal/nucleosome_free;null-if-zero'
    selection: str = 'not-implemented;no-default-thresholds;no-doublet-inference'
    support: str = 'never-weight-QC;producer-specific-diagnostics-only'


PROFILE = QCScientificProfile()
PROFILE_SHA256 = hashlib.sha256(canonical(asdict(PROFILE))).hexdigest()


def unsigned(value):
    if type(value) is not int or not 0 <= value <= 2**128 - 1:
        fail('QC_COUNT_INVALID')
    return value


def tss_windows(position, contig_length):
    """TSS is a zero-based base; both strands use symmetric genomic windows."""
    if (type(position) is not int or type(contig_length) is not int
            or not 2000 <= position < contig_length - 2000):
        fail('QC_TSS_WINDOW_INVALID')
    return tuple((position + a, position + b)
                 for a, b in (PROFILE.center, PROFILE.left_flank, PROFILE.right_flank))


def tss_enrichment(center, left_flank, right_flank):
    """Exact rational sufficient-statistic result, not rounded authoritative float."""
    center, left_flank, right_flank = map(unsigned, (center, left_flank, right_flank))
    flank = unsigned(left_flank + right_flank)
    return (Fraction(200 * center, 101 * flank), None) if flank else (None, 'ZERO_TSS_BACKGROUND')


def fragment_length_bin(length):
    if type(length) is not int or not 1 <= length <= 2**63 - 1:
        fail('QC_FRAGMENT_LENGTH_INVALID')
    return 'nucleosome_free' if length < 147 else 'mononucleosomal' if length < 294 else 'longer'


def nucleosome_signal(nucleosome_free, mononucleosomal):
    a, b = unsigned(nucleosome_free), unsigned(mononucleosomal)
    return (Fraction(b, a), None) if a else (None, 'ZERO_NUCLEOSOME_FREE')
