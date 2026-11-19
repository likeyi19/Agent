"""Exercise the production editor projection and serializer, including source scopes."""
import json

import pytest

from agent.application.turn_decisions import ExplicitArgument
from agent.application.structured_parameters import catalog, declarations
from agent.orchestration import build_default_tool_registry
from test_static import STATIC, _javascript_value


def editor_declaration(option, value, *, code=None, revision_steps=None):
    code = code or (STATIC / 'app.js').read_text()
    functions = code[code.index('async function openResultParameters('):code.index('async function refreshRequestOptions(')]
    metadata = (dict(generation=0, steps=revision_steps) if revision_steps is not None else
                dict(tools=catalog(build_default_tool_registry(), option['tool'], option.get('native_function'))))
    script = '''
    const state={session:{session_id:'session',generation:0}},nodes={};
    class Node {
      constructor(){this.children=[];this.value='';this.listeners={};} append(...items){this.children.push(...items);}
      replaceChildren(...items){this.children=items;} setAttribute(){}
      addEventListener(event,fn){this.listeners[event]=fn;} showModal(){}
      get childElementCount(){return this.children.length;}
    }
    const document={createElement:()=>new Node()},element=id=>nodes[id]||(nodes[id]=new Node());
    function selectOptions(){} function showError(error){throw error;}
    const api=async()=>(METADATA);const sessionPath=id=>'/sessions/'+id;
    '''.replace('METADATA', json.dumps(metadata)) + functions + '''
    const result={};
    OPEN.then(()=>{
      const expand=node=>{if(node.listeners.toggle){node.open=true;node.listeners.toggle();}node.children.forEach(expand);};
      expand(element('parameter-steps'));
      const row=state.parameterEditor.rows.find(r=>STEP===null || r.stepId===STEP);row.input.value=VALUE;
      result.overrides=JSON.parse(JSON.stringify(collectedParameters()));
      result.question=state.parameterEditor.question;
    });
    result;
    '''.replace('OPEN', "openResultParameters('revision')" if revision_steps is not None else
        "openRequestedParameter(" + json.dumps(option) + ",'question')").replace(
            'STEP', json.dumps(option.get('step_id'))).replace('VALUE', json.dumps(value))
    return _javascript_value(script)


@pytest.mark.parametrize('tool,function,argument,member,value,expected', [
    ('run_snapatac2', 'pp.make_fragment_file', 'kwargs', 'barcode_tag', 'RG', {'barcode_tag': 'RG'}),
    ('run_snapatac2', 'pp.make_fragment_file', 'kwargs', 'barcode_regex', '(?P<barcode>[ACGT]+)', {'barcode_regex': '(?P<barcode>[ACGT]+)'}),
    ('run_scanpy', 'pp.scale', 'count_slot', None, 'layers/counts', 'layers/counts'),
    ('annotate_scATAC_cell_types', None, 'minimum_peak_features', None, '100', 100),
    ('run_scanpy', 'pp.scale', 'gene_activity', None, 'false', False),
    ('run_scanpy', 'pp.scale', 'kwargs', 'zero_center', 'false', {'zero_center': False}),
])
def test_canonical_operand_port_is_not_explicit_source_scope(tool, function, argument, member, value, expected):
    option = dict(tool=tool, step_id='selected', argument=argument, port=argument,
        requirement_id='exact-requirement', task_id='exact-task', handle='spp.exact-question.exact-requirement')
    if function:
        option['native_function'] = function
    if member:
        option['member'] = member
    result = editor_declaration(option, value)
    declaration, = result['overrides']
    expected_declaration = dict(tool=tool, step_id='selected', argument=argument, value=expected)
    if function:
        expected_declaration['native_function'] = function
    assert declaration == expected_declaration
    assert result['question'] == dict(question_turn_id='question', handle=option['handle'])
    declarations([declaration], build_default_tool_registry())


@pytest.mark.parametrize('source', ['source.0', None])
def test_noncanonical_editor_retains_source_scope_and_strict_validation(source):
    option = dict(tool='run_scanpy', native_function='pp.scale', argument='kwargs', member='max_value',
        port='dataset', handle='parameter.value_exact')
    if source:
        option['source'] = source
    declaration, = editor_declaration(option, '4')['overrides']
    assert declaration['port'] == 'dataset'
    if source:
        assert declaration['source'] == source
        assert ExplicitArgument(**declaration).source == source
    else:
        assert 'source' not in declaration
        with pytest.raises(ValueError, match='Incomplete explicit source scope'):
            ExplicitArgument(**declaration)


def test_resource_qualified_declarations_keep_existing_strict_contract():
    resource_id = 'local-' + 'a' * 64
    declaration = dict(tool='run_scanpy', native_function='pp.scale', argument='kwargs',
        value={'max_value': 4}, port='dataset', resource_id=resource_id)
    assert ExplicitArgument(**declaration).resource_id == resource_id
    del declaration['port']
    with pytest.raises(ValueError, match='Incomplete explicit source scope'):
        ExplicitArgument(**declaration)


def test_result_editor_keeps_exact_selected_step_for_repeated_native_function():
    steps = [dict(step_id=step, tool='run_snapatac2', native_function='tl.leiden', parameters=[
        dict(argument='resolution', kwarg=True, recorded=True, value=1., description='Recorded setting')])
        for step in ('first', 'other')]
    result = editor_declaration(dict(argument='kwargs', step_id='other'), '.7', revision_steps=steps)
    assert result['overrides'] == [dict(tool='run_snapatac2', native_function='tl.leiden',
        step_id='other', argument='kwargs', value={'resolution': .7})]
    assert 'question' not in result
