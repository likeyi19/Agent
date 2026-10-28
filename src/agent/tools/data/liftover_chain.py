"""Exact assembly-pair chain qualification and bounded UCSC acquisition.

This owner checks the chain container; EpiZoo invokes native liftOver and
bedtools. No alignment or coordinate-conversion algorithm is implemented here.
"""
from pathlib import Path
import gzip
import hashlib
import re
from agent.tools.models.epizoo_adaptation import contract as c


def qualify(resource):
    c.checks.shape(resource,('input_species','input_assembly','destination_species','destination_assembly',
        'input_fai','destination_fai','chain','source','usage'))
    for key in ('input_fai','destination_fai','chain'): c.check_file(resource[key])
    from .regulatory_feature_reference import validate_species
    validate_species(resource['input_species']);validate_species(resource['destination_species'])
    for k in ('input_assembly','destination_assembly','source','usage'):
        if not isinstance(resource[k],str) or not resource[k].strip(): raise ValueError('Chain identity/provenance required.')
    def lengths(record):
        result={}
        with Path(record['path']).open() as stream:
            for line in stream:
                f=line.split('\t')
                if len(f)!=5 or f[0] in result or int(f[1])<1: raise ValueError('Invalid qualified FAI.')
                result[f[0]]=int(f[1])
        return result
    source=lengths(resource['input_fai']); destination=lengths(resource['destination_fai'])
    path=Path(resource['chain']['path'])
    with path.open('rb') as stream: compressed=stream.read(2)==b'\x1f\x8b'
    opener=gzip.open if compressed else open
    active=None; count=0; ids=set()
    with opener(path,'rt') as stream:
        for line in stream:
            if len(line)>65536: raise ValueError('Oversized chain line.')
            f=line.split()
            if not f or f[0].startswith('#'): continue
            if f[0]=='chain':
                if active is not None or len(f)!=13: raise ValueError('Incomplete chain block/header.')
                _,score,tname,tsize,tstrand,tstart,tend,qname,qsize,qstrand,qstart,qend,identity=f
                tsize,tstart,tend,qsize,qstart,qend=map(int,(tsize,tstart,tend,qsize,qstart,qend))
                if (int(score)<0 or identity in ids or tstrand!='+' or qstrand not in ('+','-')
                        or source.get(tname)!=tsize or destination.get(qname)!=qsize
                        or not 0<=tstart<tend<=tsize or not 0<=qstart<qend<=qsize):
                    raise ValueError('Chain direction, coordinates or contig lengths do not match the exact assembly pair.')
                ids.add(identity); count+=1
                active=[tstart,qstart,tend,qend]
            else:
                if active is None or len(f) not in (1,3): raise ValueError('Invalid chain alignment block.')
                nums=list(map(int,f))
                if nums[0]<1 or any(v<0 for v in nums): raise ValueError('Invalid chain block size.')
                active[0]+=nums[0]; active[1]+=nums[0]
                if len(nums)==3:
                    active[0]+=nums[1];active[1]+=nums[2]
                    if active[0]>=active[2] or active[1]>=active[3]: raise ValueError('Chain block exceeds header bounds.')
                else:
                    if active[:2]!=active[2:]: raise ValueError('Chain block totals differ from header bounds.')
                    active=None
    if active is not None or not count: raise ValueError('Empty/incomplete chain.')
    return dict(chain_count=count,direction='input-to-destination',format='UCSC-chain',identity=c.digest(resource))


def acquire_ucsc(input_assembly,destination_assembly,root, *, max_bytes=250*1024**2):
    """Resolve a listed authoritative URL first; never guess a chain download.

    Returns None when the exact file is absent. Network/integrity errors remain
    errors; an unavailable assessment is never recorded as zero correspondence.
    """
    import urllib.request
    import urllib.error
    from urllib.parse import urljoin,urlparse
    from html.parser import HTMLParser
    import shutil
    from datetime import datetime,timezone
    for value in (input_assembly,destination_assembly):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,63}',value): raise ValueError('Invalid assembly identifier.')
    directory='https://hgdownload.soe.ucsc.edu/goldenPath/'+input_assembly+'/liftOver/'
    expected=input_assembly+'To'+destination_assembly[0].upper()+destination_assembly[1:]+'.over.chain.gz'
    class Links(HTMLParser):
        links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a': self.links.append(dict(attrs).get('href',''))
    def open_url(url):
        response=urllib.request.urlopen(url,timeout=45)
        parsed=urlparse(response.url)
        if parsed.scheme!='https' or parsed.netloc!='hgdownload.soe.ucsc.edu':
            response.close();raise ValueError('Unapproved chain download redirect.')
        return response
    try:
        with open_url(directory) as response: listing=response.read(2*1024**2+1)
    except urllib.error.HTTPError as exc:
        if exc.code==404: return None
        raise
    if len(listing)>2*1024**2: raise ValueError('UCSC chain listing exceeds bound.')
    parser=Links();parser.links=[];parser.feed(listing.decode('utf-8'))
    matches=[urljoin(directory,link) for link in parser.links if link==expected]
    if len(matches)!=1: return None
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/expected;metadata=root/(expected+'.acquisition.json')
    if path.exists():
        value=c.read_json(metadata);c.check_file(value['file']);return value
    if shutil.disk_usage(root).free<max_bytes+64*1024**2: raise ValueError('Insufficient download storage.')
    md5=None
    if 'md5sum.txt' in parser.links:
        with open_url(urljoin(directory,'md5sum.txt')) as response: checks=response.read(1024**2).decode()
        for line in checks.splitlines():
            f=line.split()
            if len(f)==2 and f[1].lstrip('*')==expected: md5=f[0]
    temporary=path.with_suffix(path.suffix+'.part')
    try:
        h=hashlib.md5();size=0
        with open_url(matches[0]) as response, temporary.open('xb') as stream:
            while chunk:=response.read(1024**2):
                size+=len(chunk)
                if size>max_bytes: raise ValueError('Chain download exceeds byte budget.')
                stream.write(chunk);h.update(chunk)
        if md5 is not None and h.hexdigest()!=md5: raise ValueError('Published chain MD5 mismatch.')
        temporary.rename(path)
        value=dict(url=matches[0],publisher='UCSC Genome Browser',release=input_assembly+' to '+destination_assembly,
            acquired_at=datetime.now(timezone.utc).isoformat(),file=c.file_record(path),published_md5=md5,
            input_assembly=input_assembly,destination_assembly=destination_assembly,
            usage='Research use; UCSC chain commercial licensing applies. See https://genome.ucsc.edu/store.html')
        c.write_json(metadata,value);return value
    finally:
        temporary.unlink(missing_ok=True)
