"""Read an accepted historical publication, without recomputing science."""
from .raw_history import accepted
from ._barcode_qc_contract import load_manifest

def verify_barcode_qc(path, *, expected_sha256, **unused):
    accepted('qc', path, expected_sha256)
    return load_manifest(path, expected_sha256)
