"""Pin an already installed isolated MACS3 3.1.0 environment; no installation."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',required=True);a=p.parse_args()
    env=Path(a.environment).resolve();root=env.parent;python=env/'bin/python';exe=env/'bin/macs3'
    version=subprocess.check_output([str(exe),'--version'],text=True).strip()
    if version!='macs3 3.1.0':raise ValueError('Expected independently qualified MACS3 3.1.0')
    lock=subprocess.check_output([str(python),'-c','import importlib.metadata as m;print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))'],text=True)
    paths=sorted({p.resolve() for p in env.rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in str(p)},key=str)
    files=[]
    for p in paths:
        sha=hashlib.sha256()
        with p.open('rb') as stream:
            while data:=stream.read(1024**2):sha.update(data)
        files.append(dict(path=str(p),sha256=sha.hexdigest(),size_bytes=p.stat().st_size))
    value=dict(version='3.1.0',python=subprocess.check_output([str(python),'--version'],text=True).strip(),files=files,
        provenance='PyPI MACS3 3.1.0; resolved dependency lock alongside runtime identity')
    for name,data in [('requirements.lock.txt',lock),('runtime.json',json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')]:
        with (root/name).open('x') as f:f.write(data)
    print(len(files),'runtime files pinned')

if __name__=='__main__':main()
