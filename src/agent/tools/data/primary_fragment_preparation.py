"""Read-only dependencies of accepted historical primary-fragment preparations."""
from . import primary_contigs as scope
from . import scatac_fragments_v2 as v2
from ._fragments_common import PACKAGING_POLICY

CONTRACT = 'primary-nuclear-fragment-preparation.v1'


def dependencies(value):
    """All original and derived bytes remain live dependencies for authority reuse."""
    scoped = scope.verify_scope(value['scope']['path'],value['scope']['sha256'])
    return [value['scope'],scoped['reference'],scoped['classification_source'],value['source'],
            *([value['source_index']] if value['source_index'] else []),value['prepared'],value['prepared_index']]


def load(path,sha):
    value=scope.read(path,sha)
    v2.shape(value,('contract_version','profile_sha256','scope','source','source_index','prepared',
                    'prepared_index','packaging','summary'))
    if value['contract_version']!=CONTRACT or value['profile_sha256']!=scope.PROFILE_SHA256:scope.fail()
    if value['packaging'] != PACKAGING_POLICY:scope.fail()
    for k in ('scope','source','prepared','prepared_index'):
        v2.resource(value[k])
    if value['source_index'] is not None:v2.resource(value['source_index'])
    v2.shape(value['summary'],('source_records','retained_records','excluded_records','excluded_by_contig',
                              'retained_by_contig','retained_stream_sha256'))
    return value
