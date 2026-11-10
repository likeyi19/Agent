"""Historical matrix cooutputs retain exact output identities without an execution recipe."""
from dataclasses import asdict, replace
import pytest
from agent.application.turn_context import snapshot
from test_dialogue_evidence import app, accepted
from test_scientific_matrix_artifacts import matrix_accepted


@pytest.mark.parametrize('reverse', [False, True])
def test_matrix_cooutputs_capture_one_exact_source(app, monkeypatch, reverse):
    revision_id, _, _ = matrix_accepted(app, monkeypatch)
    state = app.sessions.load('session')
    revision = state.revisions[0]
    source = revision.outputs[0]
    outputs = (source, replace(source, name='cells', output_key='n_cells'),
               replace(source, name='features', output_key='n_features'))
    if reverse:
        outputs = tuple(reversed(outputs))
    turn = state.turn('initial')
    selections = tuple(replace(turn.selections[0], name=o.name, output_key=o.output_key) for o in outputs)
    state = replace(state, revisions=(replace(revision, outputs=outputs),),
                    turns=(replace(turn, selections=selections),))
    tools = {}
    captured = snapshot(app.sessions, state, tool_names=tools)
    assert set(captured['bases'][revision_id]) == {'revision_id', 'outputs'}
    assert captured['bases'][revision_id]['outputs'] == [asdict(o) for o in outputs]
    assert len(tools) == 3


def test_independent_historical_matrix_sources_are_preserved_without_arbitrary_selection(app):
    accepted(app, 'build_scATAC_cell_by_ccre', copies=2)
    captured = snapshot(app.sessions, app.sessions.load('session'))
    base, = captured['bases'].values()
    assert len(base['outputs']) == 2
    assert {o['step_id'] for o in base['outputs']} == {'step0', 'step1'}
    assert 'matrix_source' not in base
