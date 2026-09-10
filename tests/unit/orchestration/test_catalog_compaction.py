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
