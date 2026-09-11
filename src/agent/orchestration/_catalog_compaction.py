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


def share_catalog_text(payload):
    """Losslessly share repeated phrases in catalog text, never request values.

    Phrase definitions contain original text, without references. A collision-free
    marker makes expansion one substitution pass; keys and wire enums are untouched.
    This keeps the complete eighteen-tool catalog within existing byte ceilings.
    """
    fields=('tools','catalog_values','meanings')
    strings=[]
    def collect(value):
        if isinstance(value,str): strings.append(value)
        elif isinstance(value,dict):
            strings.extend(value.keys())
            for child in value.values(): collect(child)
        elif isinstance(value,(list,tuple)):
            for child in value: collect(child)
    for field in fields: collect(payload.get(field,()))
    marker='~'
    while any(marker in s for s in strings): marker+='~'
    texts=[s for s in strings if ' ' in s]
    phrases=[]
    while len(phrases)<100:
        counts=Counter()
        for text in texts:
            words=text.split(' ')
            for start in range(len(words)):
                for n in range(2,min(16,len(words)-start)+1):
                    phrase=' '.join(words[start:start+n])
                    if len(phrase)>15 and marker not in phrase: counts[phrase]+=1
        tag=marker+str(len(phrases))+marker
        candidates=[((len(s)-len(tag))*n-len(s)-3,s) for s,n in counts.items() if n>1]
        if not candidates: break
        saving,phrase=max(candidates)
        if saving<=0: break
        phrases.append(phrase)
        texts=[s.replace(phrase,tag) for s in texts]
    def compact(value):
        if isinstance(value,str):
            for i,phrase in enumerate(phrases): value=value.replace(phrase,marker+str(i)+marker)
            return value
        if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return tuple(compact(v) for v in value)
        return value
    if phrases:
        for field in fields:
            if field in payload: payload[field]=compact(payload[field])
        payload['catalog_phrases']=phrases
        payload['catalog_phrase_marker']=marker
        payload['catalog_format']['phrase_reference']='In catalog strings, <catalog_phrase_marker>N<catalog_phrase_marker> expands once to catalog_phrases[N].'
