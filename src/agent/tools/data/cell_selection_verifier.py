"""Read an accepted historical publication, without recomputing science."""
from .raw_history import accepted
from ._cell_selection_contract import load_manifest

def verify_cell_selection(path, *, expected_sha256, **unused):
    accepted('selection', path, expected_sha256)
    return load_manifest(path, expected_sha256)
