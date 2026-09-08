from pathlib import Path
import json

import pytest

from agent.tools.data import raw_scatac as raw
from agent.tools.data import raw_scatac_manifest as m
from agent.orchestration import PlanStep, build_default_tool_registry, verify_step


def verify(args, result):
    return verify_step(PlanStep('raw', 'inspect_raw_scATAC', args), args, result, build_default_tool_registry())


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('state', ['READY', 'NEEDS_USER_INPUT', 'INVALID', 'UNSUPPORTED'])
def test_fresh_verification_of_truthful_science(raw_factory, monkeypatch, kind, state):
    args = raw_factory(kind, malformed=state == 'INVALID', species='rat' if state == 'UNSUPPORTED' else 'human')
    if state == 'NEEDS_USER_INPUT':
        args['raw_assay'] = None
    result = raw.inspect_raw_scATAC(**args)
    before = sorted(Path(args['output_dir']).iterdir())
    monkeypatch.setattr(raw, 'inspect_raw_scATAC', lambda *a, **k: pytest.fail('verifier called public tool'))
    monkeypatch.setattr(m, 'publish_raw_intake_manifest', lambda *a, **k: pytest.fail('verifier published'))
    assert result['readiness'] == state
    verification = verify(args, result)
    assert verification.passed, verification
    assert 'raw_intake_source_reconstruction' in {c.name for c in verification.checks}
    assert sorted(Path(args['output_dir']).iterdir()) == before


@pytest.mark.parametrize('field,value', [('readiness', 'INVALID'), ('n_issues', 1),
    ('n_required_information', 1), ('input_kind', 'bam'), ('manifest_sha256', 'f' * 64)])
def test_summary_tampering(raw_factory, field, value):
    args = raw_factory()
    result = raw.inspect_raw_scATAC(**args)
    assert not verify(args, {**result, field: value}).passed


@pytest.mark.parametrize('mutation', ['bytes', 'readiness', 'extra', 'canonical_forgery', 'noncanonical'])
def test_manifest_tampering(raw_factory, mutation):
    args = raw_factory()
    result = raw.inspect_raw_scATAC(**args)
    path = Path(result['manifest_path'])
    content = json.loads(path.read_text())
    if mutation == 'bytes':
        path.write_bytes(b'broken')
    elif mutation == 'noncanonical':
        import hashlib
        payload = json.dumps(content, indent=2).encode()
        digest = hashlib.sha256(payload).hexdigest()
        path = path.parent / f'raw-scatac-intake-{digest}.json'
        path.write_bytes(payload)
        result.update(manifest_path=str(path), manifest_sha256=digest)
    elif mutation == 'canonical_forgery':
        # Valid canonical manifest and self-consistent digest/summary still need
        # fresh source/declaration verification.
        forged = raw._reconstruct(args['raw_input_paths'], species='mouse', raw_assay='TENX_ATAC')
        import hashlib
        digest = hashlib.sha256(m.canonical_manifest_bytes(forged)).hexdigest()
        path = path.parent / f'raw-scatac-intake-{digest}.json'
        m.publish_raw_intake_manifest(forged, path)
        result = raw._summary(forged, path, digest)
    else:
        content['readiness' if mutation == 'readiness' else 'extra'] = 'INVALID'
        path.write_text(json.dumps(content))
    assert not verify(args, result).passed


def test_fastq_observed_content_drift_even_with_original_size_and_mtime(raw_factory):
    import os
    args = raw_factory()
    result = raw.inspect_raw_scATAC(**args)
    path = next(Path(args['raw_input_paths']).iterdir())
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b'ACGT', b'TGCA'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not verify(args, result).passed


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
def test_source_drift_detected(raw_factory, kind):
    args = raw_factory(kind)
    result = raw.inspect_raw_scATAC(**args)
    path = next(Path(args['raw_input_paths']).iterdir())
    path.write_bytes(path.read_bytes() + b'changed')
    assert not verify(args, result).passed


def test_bam_index_observation_drift(raw_factory):
    args = raw_factory('bam')
    result = raw.inspect_raw_scATAC(**args)
    Path(args['raw_input_paths'], 'input.bam.bai').write_bytes(b'corrupt index')
    assert not verify(args, result).passed


@pytest.mark.parametrize('change', ['output', 'species', 'assay', 'layout', 'inventory'])
def test_argument_binding_and_inventory(raw_factory, change, tmp_path):
    args = raw_factory()
    result = raw.inspect_raw_scATAC(**args)
    if change == 'output':
        (tmp_path / 'elsewhere').mkdir()
        args['output_dir'] = str(tmp_path / 'elsewhere')
    elif change == 'inventory':
        args['raw_input_paths'] = raw_factory(root='another')['raw_input_paths']
    else:
        args[{'species': 'species', 'assay': 'raw_assay', 'layout': 'fastq_layout'}[change]] = {
            'species': 'mouse', 'assay': None, 'layout': 'tenx-atac-r1-i2-r2.v1'}[change]
    assert not verify(args, result).passed
