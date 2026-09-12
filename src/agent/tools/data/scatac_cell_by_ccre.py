"""M11.5b data-layer matrix publication; no registered tool or Agent lifecycle."""
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import tempfile
import time

from . import scatac_matrix_contract as m, _matrix_bedtools as bed
from . import _cell_by_ccre_io as io, _cell_by_ccre_production as production
from .cell_by_ccre_verifier import verify_cell_by_ccre
from .scatac_qc_reference import _fsync_dir

MatrixLimits = io.MatrixLimits


def build_cell_by_ccre(*, fragments_manifest_path, fragments_manifest_sha256,
                       selection_manifest_path, selection_manifest_sha256,
                       reference_manifest_path, reference_manifest_sha256,
                       output_dir, bedtools_path, limits=MatrixLimits()):
    """Publish one immutable H5AD after complete independent reconstruction.

    Explicit qualified executable and runtime/QC resource configuration are
    operator inputs, not planner parameters. Existing destinations fail closed.
    """
    started = time.monotonic()
    destination = Path(output_dir); m.absolute_path(str(destination))
    if destination != destination.resolve() or destination.exists() or destination.is_symlink(): m.fail('MATRIX_OUTPUT_CONFLICT')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pointers = {name:dict(manifest_path=str(path),manifest_sha256=digest) for name,path,digest in (
        ('fragments',fragments_manifest_path,fragments_manifest_sha256),
        ('selection',selection_manifest_path,selection_manifest_sha256),
        ('reference',reference_manifest_path,reference_manifest_sha256))}
    backend = bed.qualify_runtime(bedtools_path)
    bound = io.bind(pointers,limits)
    # An exclusive sibling lock protects cooperating publishers. No overwrite.
    lock = destination.parent/('.'+destination.name+'.matrix-lock')
    try: fd = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError: m.fail('MATRIX_OUTPUT_CONFLICT')
    os.close(fd)
    try:
        with tempfile.TemporaryDirectory(prefix='.matrix-stage-',dir=destination.parent) as directory:
            root = Path(directory); publication = root/'artifact'; publication.mkdir()
            budget = io.Budget(root,limits)
            with io.database(root/'production.sqlite',budget) as db:
                io.load_axes(db,bound,budget)
                diagnostic = production.construct_counts(db,bound,bedtools_path,root,budget)
                summary = io.write_h5(publication/'matrix.h5ad',db,bound,
                                      production.rows(db,bound.selection['selected_count']),budget)
            matrix = publication/'matrix.h5ad'
            value = dict(artifact_type=m.ARTIFACT,schema_version=1,contract_version=m.CONTRACT,
                         profile=asdict(m.PROFILE),profile_sha256=m.PROFILE_SHA256,
                         species=bound.reference.species,assembly=bound.reference.target_assembly,
                         upstream=bound.upstream,ordered_selected_sha256=bound.selection['ordered_selected_sha256'],
                         ordered_feature_sha256=bound.reference.ccre.ordered_feature_sha256,
                         shape=[bound.selection['selected_count'],bound.reference.ccre.feature_count],**summary,
                         matrix=dict(path='matrix.h5ad',sha256=bed.file_sha(matrix),size_bytes=matrix.stat().st_size,format='csr',dtype='int64'),
                         backend=backend,diagnostic=diagnostic,
                         readiness='matrix_available' if bound.selection['selected_count'] else 'no_selected_cells')
            value['identity_sha256'] = m.manifest_identity(value)
            raw = m.canonical(m.validate_manifest(value)); digest = hashlib.sha256(raw).hexdigest()
            (publication/'manifest.json').write_bytes(raw)
            construction_seconds = time.monotonic()-started
            budget.check()
            existing_bytes = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            remaining = limits.max_scratch_bytes-existing_bytes
            seconds = limits.max_seconds-int(time.monotonic()-started)
            if remaining <= 0 or seconds <= 0: m.fail('MATRIX_RESOURCE_LIMIT')
            verification_limits = replace(limits,max_scratch_bytes=remaining,max_seconds=seconds)
            verified = verify_cell_by_ccre(publication/'manifest.json',expected_sha256=digest,
                                           bedtools_path=bedtools_path,limits=verification_limits,scratch_parent=root)
            budget.peak_bytes = max(budget.peak_bytes,existing_bytes + verified['verification_scratch_peak_bytes'])
            if budget.peak_bytes > limits.max_scratch_bytes: m.fail('MATRIX_SCRATCH_LIMIT')
            bound.unchanged(); budget.check()
            if bed.qualify_runtime(bedtools_path) != backend: m.fail('MATRIX_BACKEND_INVALID')
            for path in (matrix,publication/'manifest.json'):
                with path.open('rb') as f: os.fsync(f.fileno())
            _fsync_dir(publication)
            if destination.exists() or destination.is_symlink(): m.fail('MATRIX_OUTPUT_CONFLICT')
            os.rename(publication,destination); _fsync_dir(destination.parent)
            from .authority_context import publication_moved
            publication_moved(publication, destination)
        return dict(manifest_path=str(destination/'manifest.json'),manifest_sha256=digest,
                    identity_sha256=value['identity_sha256'],**summary,diagnostic=diagnostic,
                    construction_seconds=construction_seconds,verification_seconds=verified['verification_seconds'],
                    scratch_peak_bytes=budget.peak_bytes)
    finally:
        lock.unlink()
