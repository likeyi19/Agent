import json
from agent.orchestration._catalog_compaction import share_catalog_values
from catalog_expansion import expand_catalog


def test_generic_sharing_preserves_nested_values_and_reserved_looking_keys():
    original={'$':3,'@0':'literal','items':[{'long_parameter_name':{'nested':'long exact repeated scientific guidance'}}]*5,
        'options':{'$':7},'trailing':[False,[],None,None]}
    tools,values,keys,marker=share_catalog_values(original)
    payload=json.loads(json.dumps(dict(tools=tools,catalog_values=values,catalog_keys=keys,catalog_ref_key=marker)))
    assert expand_catalog(payload)==original
    assert marker!='$'


def test_phrase_sharing_is_exact_and_collision_free():
    from agent.orchestration._catalog_compaction import share_catalog_text
    text='Exact canonical fragment-record counts retain full ordered reference features.'
    original={'literal ~0~': ['prefix '+text+' suffix', text+' more']*8,
              'nested': {'fact':text}, 'untouched':[False,[],None]}
    tools,values,keys,marker=share_catalog_values(original)
    payload=dict(tools=tools,catalog_values=values,catalog_keys=keys,catalog_ref_key=marker,catalog_format={},
        user_request='PRIVATE exact request text')
    share_catalog_text(payload)
    assert payload['user_request']=='PRIVATE exact request text'
    assert payload['catalog_phrase_marker']!='~'
    payload=json.loads(json.dumps(payload))
    assert expand_catalog(payload)==original


def test_real_v3_phrase_expansion_preserves_all_catalog_meanings():
    import re
    from agent.orchestration import AgentRequest,build_default_tool_registry
    from agent.orchestration.llm_planner import _build_prompt,_prompt_catalog
    r=build_default_tool_registry();q=AgentRequest('test','Build a matrix.',{'input_path':'/PRIVATE/data.h5ad'})
    payload=json.loads(_build_prompt(q,r));expanded=expand_catalog(payload)
    marker=payload['catalog_phrase_marker']
    for tool in expanded.values():
        for section in (tool[2],tool[3]):
            for metadata in section.values():
                if isinstance(metadata[0],dict):
                    raw=payload['meanings'][metadata[0]['m']]
                    metadata[0]=re.sub(re.escape(marker)+r'(\d+)'+re.escape(marker),lambda m:payload['catalog_phrases'][int(m[1])],raw)
    assert expanded==json.loads(json.dumps(_prompt_catalog(r)))
