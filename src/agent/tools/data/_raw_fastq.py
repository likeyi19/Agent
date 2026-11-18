"""Explicit FASTQ member and read-layout declarations shared by native intake.

No source discovery, record inspection, readiness or manifest publication.
"""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType




class FastqAssay(str, Enum):
    TENX_ATAC = "TENX_ATAC"


class ReadRole(str, Enum):
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    I1 = "I1"
    I2 = "I2"


class FastqLayout(str, Enum):
    A = "tenx-atac-r1-r2-r3.v1"
    B = "tenx-atac-r1-i2-r2.v1"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported-role-set"


class ReadMeaning(str, Enum):
    GENOMIC_1 = "genomic-read-1"
    GENOMIC_2 = "genomic-read-2"
    BARCODE = "raw-cell-barcode-i5"
    SAMPLE_INDEX = "sample-index"


LAYOUT_ROLES = MappingProxyType({
    FastqLayout.A: MappingProxyType({ReadRole.R1: ReadMeaning.GENOMIC_1,
        ReadRole.R2: ReadMeaning.BARCODE, ReadRole.R3: ReadMeaning.GENOMIC_2,
        ReadRole.I1: ReadMeaning.SAMPLE_INDEX}),
    FastqLayout.B: MappingProxyType({ReadRole.R1: ReadMeaning.GENOMIC_1,
        ReadRole.I2: ReadMeaning.BARCODE, ReadRole.R2: ReadMeaning.GENOMIC_2,
        ReadRole.I1: ReadMeaning.SAMPLE_INDEX}),
})




@dataclass(frozen=True)
class FastqName:
    sample: str
    sample_number: str
    lane: str | None
    role: ReadRole
    chunk: str






def _layout(names: list[FastqName | None], assay: FastqAssay | None,
            declared_layout: FastqLayout | None = None):
    roles = [n.role for n in names if n is not None]
    duplicate = len(roles) != len(set(roles))
    if duplicate or any(n is None for n in names) or assay is None:
        return FastqLayout.UNRESOLVED, duplicate
    required = set(roles) - {ReadRole.I1}
    a = set(LAYOUT_ROLES[FastqLayout.A]) - {ReadRole.I1}
    b = set(LAYOUT_ROLES[FastqLayout.B]) - {ReadRole.I1}
    if required == a and declared_layout in (None, FastqLayout.A):
        return FastqLayout.A, False
    if required == b and declared_layout in (None, FastqLayout.B):
        return FastqLayout.B, False
    if ((required < a and declared_layout in (None, FastqLayout.A))
            or (required < b and declared_layout in (None, FastqLayout.B))):
        return FastqLayout.UNRESOLVED, False
    return FastqLayout.UNSUPPORTED, False


def _missing_roles(names: list[FastqName | None], assay: FastqAssay | None,
                   declared_layout: FastqLayout | None):
    """Exact absent roles, conditional on each admissible incomplete layout.

    R1/R2 alone cannot distinguish missing R3 (A) from missing I2 (B).
    Preserve both alternatives unless a declaration or unique role set decides.
    """
    if assay is None or any(n is None for n in names):
        return ()
    roles = [n.role for n in names]
    if len(roles) != len(set(roles)):
        return ()
    observed = set(roles) - {ReadRole.I1}
    candidates = (declared_layout,) if declared_layout else (FastqLayout.A, FastqLayout.B)
    return tuple((candidate, tuple(sorted(required - observed, key=lambda r: r.value)))
                 for candidate in candidates
                 if observed < (required := set(LAYOUT_ROLES[candidate]) - {ReadRole.I1}))
