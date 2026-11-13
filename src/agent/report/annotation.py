"""Bounded hierarchical annotation facts from the single publication owner."""
from agent.tools.analysis.hierarchical_annotation import RESULT_FIELDS
CONTRACT_FIELDS = frozenset(RESULT_FIELDS)
FACT_FIELDS = tuple(k for k in RESULT_FIELDS if k != 'status' and not k.endswith(('_path','_sha256')))
REPORT_FIELDS = FACT_FIELDS
METHOD_FIELDS = ('native_methods', 'native_methods_omitted', 'interpretation_parameters', 'cellmarker_identity', 'initial_grouping')


def method_facts(result):
    """Read the accepted hierarchy's actual methods; never reconstruct science."""
    from agent.tools.analysis.sequence_prediction import read
    h = read(result['hierarchy_path'], result['hierarchy_sha256'])
    methods = []
    for op in h['operations'][:100]:
        native = op['native_result']
        methods.append(dict(branch_id=op['branch_id'], operation_id=op['operation_id'],
            function=native['function_name'], parameters=native['parameters'],
            software_versions=native.get('software_versions',{}),
            producing_run_id=op['binding']['run_id'], accepted_step_sha256=op['binding']['accepted_step_sha256'],
            manifest_sha256=native['manifest_sha256']))
    grouping = h['grouping']
    initial = ({k: grouping[k] for k in ('origin', 'groupby', 'qualification')}
        if grouping.get('origin') == 'user_supplied' else dict(origin='accepted_native'))
    initial['source_sha256'] = h['source']['sha256']
    return dict(initial_grouping=initial, native_methods=methods,native_methods_omitted=max(0,len(h['operations'])-100),
        interpretation_parameters=h['interpretation'],cellmarker_identity=h['cellmarker'])
