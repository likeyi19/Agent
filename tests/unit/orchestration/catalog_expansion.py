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
            return {keys.get(k,k):expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return expand_text(value)
    return expand(payload['tools'])
