"""Lossless prompt-only sharing of repeated registry catalog values."""
from collections import Counter
import json


def share_catalog_values(catalog):
    """Return a catalog projection and a legend; executable wire schemas are untouched."""
    encode=lambda value:json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))
    counts=Counter(); originals={};keys=Counter()
    def collect(value):
        key=encode(value)
        if len(key)>15:
            counts[key]+=1;originals[key]=value
        if isinstance(value,dict):
            keys.update(value.keys())
            for child in value.values():collect(child)
        elif isinstance(value,(list,tuple)):
            for child in value:collect(child)
    collect(catalog)
    marker='$'
    while marker in keys:marker+='$'
    # Only values with an actual net saving become candidates. Definitions hold
    # original values, so expansion is a single unambiguous substitution.
    candidates={k for k,n in counts.items() if n>1 and (len(k)-10)*n>len(k)+2}
    definitions=[];indices={}
    def compact(value):
        key=encode(value)
        if key in candidates:
            if key not in indices:
                indices[key]=len(definitions);definitions.append(originals[key])
            return {marker:indices[key]}
        if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return tuple(compact(v) for v in value)
        return value
    result=compact(catalog)
    names=sorted(k for k,n in keys.items() if len(k)>8 and (len(k)-5)*n>len(k)+8)
    prefix='@'
    while any(k.startswith(prefix) for k in keys):prefix+='@'
    aliases={name:prefix+str(i) for i,name in enumerate(names)}
    def rename(value):
        if isinstance(value,dict):return {aliases.get(k,k):rename(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return tuple(rename(v) for v in value)
        return value
    return rename(result),rename(definitions),{aliases[k]:k for k in names},marker
