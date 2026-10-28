"""Simulated owner descriptions: exact model parent roles, no native science."""
from types import SimpleNamespace
import pytest
from agent.orchestration import prior_outputs as p, verification_authority as v, build_default_tool_registry
from agent.schemas.verification_authority import AuthorityError


@pytest.mark.parametrize('selected,parents,accepted', [
    (('trained',), {'trained': 'initialized'}, True),
    (('trained', 'initialized'), {'trained': 'initialized'}, True),
    (('initialized', 'trained'), {'trained': 'initialized'}, True),
    (('trained', 'unrelated'), {'trained': 'initialized'}, False),
    (('trained', 'sibling'), {'trained': 'initialized', 'sibling': 'initialized'}, False),
    (('trained', 'initialized', 'sibling'), {'trained': 'initialized', 'sibling': 'initialized'}, False),
])
def test_only_exact_initialized_parent_can_accompany_trained_child(monkeypatch, selected, parents, accepted):
    from agent.tools.data.authority_context import VerificationContext
    from agent.tools import scientific_authority
    monkeypatch.setattr(p, 'optional_binding_for_locator', lambda name, *args: SimpleNamespace(
        run_id=name, step_id=name, manifest_sha256=name, tool_name='posttrain_epizoo_model'))
    monkeypatch.setattr(p, 'merge_context', lambda *args: None)
    monkeypatch.setattr(v, '_load_context', lambda *args: None)
    monkeypatch.setattr(v, '_accepted', lambda store, run, step: (None, SimpleNamespace(
        record={'publication_path': '/'+run}), None, None))
    monkeypatch.setattr(scientific_authority, 'tool_kind', lambda *args: 'epizoo_adaptation')
    def describe(self, kind, path, sha):
        assert str(path) == '/'+sha
        parent = parents.get(sha)
        return {'upstream': {} if parent is None else {'initialized_model': {
            'kind': kind, 'manifest_path': '/'+parent, 'manifest_sha256': parent}}}, None
    monkeypatch.setattr(VerificationContext, 'describe', describe)
    if accepted:
        p.validate_active_outputs(selected, None, build_default_tool_registry())
    else:
        with pytest.raises(AuthorityError, match='distinct scientific roles'):
            p.validate_active_outputs(selected, None, build_default_tool_registry())
