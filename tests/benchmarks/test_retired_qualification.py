"""Closed experiments cannot dispatch providers or rewrite accepted evidence."""
import json
import pytest
from benchmarks.interactive import run_breadth, run_stability, run_frontier, run_frontier_stability

@pytest.mark.parametrize('entry',[run_breadth.run_breadth,run_breadth.resume_breadth,
    run_stability.run_stability,run_frontier.run,run_frontier_stability.run])
def test_retired_entry_rejects_before_factory_or_output(entry,tmp_path):
    def forbidden(*args,**kwargs):pytest.fail('Retired runner constructed a provider')
    target=tmp_path/'output'
    with pytest.raises(RuntimeError,match='retired'):
        entry(output=target,live=True,factory=forbidden)
    assert not target.exists()

def test_q2_reader_preserves_original_json_and_digest(tmp_path):
    path=tmp_path/'results.json'
    original=b'{"historical_answer":"original prose","calls":3}\n'
    path.write_bytes(original)
    value,identity=run_breadth._original(path)
    assert value==json.loads(original)
    assert identity['sha256'] and path.read_bytes()==original

def test_private_diagnostic_values_remain_redacted():
    private=('private-api-credential','private project')
    value={'text':'private-api-credential private%20project private+project'}
    result=run_breadth._safe_json(value,private)
    assert all(secret not in json.dumps(result) for secret in private)
    assert 'private%20project' not in json.dumps(result)
