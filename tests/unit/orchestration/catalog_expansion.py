"""Independent test decoder for lossless provider-prompt projections."""
def expand_catalog(payload):
    import re
    phrase_marker=payload.get('catalog_phrase_marker')
    def expand_text(value):
        if isinstance(value,str) and phrase_marker:
            return re.sub(re.escape(phrase_marker)+r'(\d+)'+re.escape(phrase_marker),
                lambda match:payload['catalog_phrases'][int(match[1])],value)
        return value
    values=payload['catalog_values'];keys=payload['catalog_keys'];marker=payload['catalog_ref_key']
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={marker}:return expand(values[value[marker]])
            return {expand_text(keys.get(expand_text(k),expand_text(k))):expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return expand_text(value)
    result=expand(payload['tools'])
    if 'input_lineage' in payload.get('catalog_format', {}):
        for tool in result.values():
            for port in tool[1].values():
                if len(port)==5: port.insert(4,None)
    return result
