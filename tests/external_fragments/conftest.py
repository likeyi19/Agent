"""Small structural inspection fixtures; no scientific ingestion/canonicalization."""
import hashlib
from pathlib import Path
import pysam
import pytest

@pytest.fixture
def source_factory(tmp_path):
    number=0
    def make(*,encoding='bgzf',indexed=False,rows=None):
        nonlocal number
        p=tmp_path/('fragments-'+str(number)+'.gz');number+=1
        with pysam.BGZFile(str(p),'wb') as stream:
            stream.write(b''.join(rows or [b'chr2\t10\t30\tA-1\t3\n']))
        args=dict(source_path=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if indexed:
            pysam.tabix_index(str(p),preset='bed',force=True)
            index=Path(str(p)+'.tbi');args.update(source_index_path=str(index),source_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
        return args
    return make
