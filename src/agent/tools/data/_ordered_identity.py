"""Small order-sensitive SHA-256 utility, distinct from M10 set serialization."""

from collections.abc import Iterable
import hashlib


ORDERED_IDENTITY_ALGORITHM = "sha256-utf8-lf.v1"


def ordered_identity_sha256(values: Iterable[str]) -> str:
    """Hash each exact UTF-8 string followed by LF, in supplied order.

    No sorting, trimming, Unicode normalization, or domain prefix is applied.
    Empty collections, empty strings, duplicates, CR/LF and invalid Unicode
    fail. Duplicate detection requires memory proportional to unique IDs.
    The caller's artifact schema supplies the semantic domain of the digest.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("Ordered identities require a collection of strings.")
    seen: set[str] = set()
    digest = hashlib.sha256()
    for value in values:
        if type(value) is not str or not value or "\n" in value or "\r" in value:
            raise ValueError("Invalid ordered identity record.")
        if value in seen:
            raise ValueError("Duplicate ordered identity record.")
        seen.add(value)
        digest.update(value.encode("utf-8", errors="strict"))
        digest.update(b"\n")
    if not seen:
        raise ValueError("Ordered identities must not be empty.")
    return digest.hexdigest()
