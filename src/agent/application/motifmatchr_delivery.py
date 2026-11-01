"""Native match matrices reuse accepted publication and private snapshot delivery."""
from agent.tools.analysis import motifmatchr as owner
from . import chromvar_delivery as native

FILES = {
    'matches.mtx': ('Native peak-by-motif matches (Matrix Market)', 'application/octet-stream'),
    'motif-matches.npz': ('Native peak-by-motif matches (SciPy sparse)', 'application/octet-stream'),
    'peaks.tsv': ('All original ordered genomic peaks', 'text/tab-separated-values; charset=utf-8'),
    'motifs.tsv': ('Native ordered motif names and versioned annotations', 'text/tab-separated-values; charset=utf-8'),
    'metadata.json': ('Native motif matching source, parameters and resources', 'application/json'),
    'native-receipt.json': ('Native matchMotifs completion and software versions', 'application/json'),
}


def accepted_artifacts(*args,**kwargs):
    return native.accepted_artifacts(*args,**kwargs,owner=owner,files=FILES,artifact_type='scientific_motif_matches')


def prepare(stream,role):
    if role not in FILES:
        raise ValueError('Unknown native motif artifact')
    if role.endswith(('.json','.tsv')):
        return native.prepare(stream,role)
    return stream.read()
