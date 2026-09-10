"""Independent test decoder for lossless provider-prompt projections."""
def expand_catalog(payload):
    values=payload['catalog_values'];keys=payload['catalog_keys'];marker=payload['catalog_ref_key']
    def expand(value):
        if isinstance(value,dict):
            if set(value)=={marker}:return expand(values[value[marker]])
            return {keys.get(k,k):expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        return value
    return expand(payload['tools'])
