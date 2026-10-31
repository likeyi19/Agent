"""Native peak artifacts share accepted publication/snapshot delivery with motifs."""
from agent.tools.analysis import macs3 as owner
from . import chromvar_delivery as native
FILES = {
    'peaks_peaks.narrowPeak': ('Native MACS3 narrowPeak', 'text/plain; charset=utf-8'),
    'peaks_summits.bed': ('Native peak summits', 'text/plain; charset=utf-8'),
    'peaks_peaks.xls': ('Native peak statistics (private command paths omitted)', 'text/tab-separated-values; charset=utf-8'),
    'features.bed': ('Exact ordered experimental peak features', 'text/plain; charset=utf-8'),
    'distributions.png': ('Peak length and enrichment distributions', 'image/png'),
    'metadata.json': ('Peak parameters, population and provenance', 'application/json'),
    'native-receipt.json': ('Native MACS3 completion and runtime identity', 'application/json'),
}

def accepted_artifacts(*args,**kwargs):
    return native.accepted_artifacts(*args,**kwargs,owner=owner,files=FILES,artifact_type='scientific_peaks')

def accepted_figures(*args):
    return native.accepted_figures(*args,owner=owner,figure_name='distributions.png',figure_kind='peak_distributions')

def prepare(stream,role):
    if role not in FILES:raise ValueError('Unknown peak artifact')
    if role.endswith('.json'):return native.prepare(stream,role)
    data=stream.read()
    if role.endswith('.xls'):
        # Native statistical rows stay exact. Command/input/output paths are
        # private provenance, retained in the accepted native original.
        data=b''.join(line for line in data.splitlines(keepends=True) if not line.startswith(b'#'))
    return data
