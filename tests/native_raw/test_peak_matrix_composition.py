"""Current native peak handoffs; no retired raw producer or counting execution."""
from pathlib import Path
import os

import anndata as ad
import pytest

from agent.orchestration.registry import build_default_tool_registry
from agent.orchestration.semantic_compiler import build_semantic_compiler_contract
from agent.tools.analysis import macs3


def test_peak_matrix_keeps_direct_bed_requests_and_accepts_native_caller_regions():
    registry=build_default_tool_registry();contract=build_semantic_compiler_contract(registry)
    channels=contract.step_output_channels
    forward=next(c for c in channels if c.producer_tool_name=='run_snapatac2'
        and c.consumer_tool_name=='call_scATAC_peaks' and c.target_port=='dataset')
    back=next(c for c in channels if c.producer_tool_name=='call_scATAC_peaks'
        and c.consumer_tool_name=='run_snapatac2' and c.target_port=='peaks')
    assert [(m.output_key,m.argument_name) for m in forward.members]==[('matrix_path','input_path')]
    assert [(m.output_key,m.argument_name) for m in back.members]==[('peaks_path','peaks_path'),('peaks_sha256','peaks_sha256')]
    # Library components and uploaded BEDs already use these same request rules.
    native=registry.get('run_snapatac2');port=next(p for p in native.semantic_planning.consumer_ports if p.name=='peaks')
    assert port.request_sources[0].selector=='peaks_path'
    assert {m.input_name for m in port.request_sources[0].members}=={'peaks_path','peaks_sha256'}
    assert not port.required
    assert not any(registry.contains(n) for n in ('build_scATAC_cell_by_features','import_scATAC_fragments'))
    peak_inputs=registry.get('call_scATAC_peaks').required_arguments
    assert 'input_path' in peak_inputs and 'fragments_manifest_path' not in peak_inputs


def test_native_peak_source_change_is_rejected_before_export(tmp_path,monkeypatch):
    source=tmp_path/'source.h5ad';source.write_bytes(b'changed')
    from agent.tools.analysis import native_callable
    monkeypatch.setattr(native_callable,'executable',lambda *a:pytest.fail('Changed source reached native export'))
    with pytest.raises(Exception,match='changed'):
        macs3.prepare_fragments(dict(input_path=str(source),input_sha256='0'*64),tmp_path)


def test_native_macs_to_peak_matrix_and_supplied_bed(native_runtime,tmp_path,monkeypatch):
    """Real small native MACS3 output feeds the sole native count implementation."""
    import numpy as np
    from agent.tools.analysis.native_matrix import SNAPATAC2
    fragments=tmp_path/'fragments.tsv'
    with fragments.open('w') as f:
        for cell in ('opaque_a','opaque_b','opaque_c'):
            for region in (5000,15000):
                for offset in range(120):
                    f.write(f'chr1\t{region+offset}\t{region+offset+80}\t{cell}\t1\n')
    sizes=tmp_path/'sizes';sizes.write_text('chr1\t100000\n')
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep,StepOutputRef,AgentRuntime,FileRunStore
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.orchestration import _serialize
    registry=build_default_tool_registry();store=FileRunStore(tmp_path/'run_state')
    candidate=AgentPlan('native-peaks','native-peaks','Simulated native scientific integration',(
        PlanStep('import','run_snapatac2',dict(function_name='pp.import_fragments',input_path=str(fragments),
            output_dir=str(tmp_path/'import'),chrom_sizes_path=str(sizes),species='human',assembly='simulated',
            kwargs=dict(sorted_by_barcode=False,min_num_fragments=0))),
        PlanStep('peaks','call_scATAC_peaks',dict(input_path=StepOutputRef('import','matrix_path'),
            output_dir=str(tmp_path/'peaks'),effective_genome_size=100000,
            genome_size_basis='Explicit simulated single-contig qualification genome; not human biology'),('import',)),
        PlanStep('matrix','run_snapatac2',dict(function_name='pp.make_peak_matrix',
            input_path=StepOutputRef('import','matrix_path'),output_dir=str(tmp_path/'called-matrix'),
            peaks_path=StepOutputRef('peaks','peaks_path'),peaks_sha256=StepOutputRef('peaks','peaks_sha256'),
            kwargs=dict(counting_strategy='fragment')),('import','peaks'))))
    state=AgentRuntime(registry=registry,run_store=store).run_prepared(AgentRequest('native-peaks',None,{}),candidate)
    assert state.status.value=='SUCCEEDED',state.errors
    assert all(step.verification.passed for step in state.steps)
    imported,peaks,called=[_serialize(step.result) for step in state.steps]
    args=_serialize(state.steps[1].resolved_arguments)
    assert bind_output(store,registry,state.run_id,'peaks','peaks').semantic_type=='scatac_peaks.v2'
    assert peaks['n_peaks']>0 and peaks['n_barcodes']==3
    assert peaks['total_source_support'] is None and peaks['assembly']=='simulated'
    assert peaks['source_identity']['sha256']==imported['matrix_sha256']
    # The identical ordinary supplied BED must have exactly identical native counts.
    # Existing regions must remain usable when the optional caller is unavailable.
    monkeypatch.setenv('AGENT_MACS3_EXECUTABLE',str(tmp_path/'unavailable-macs3'))
    monkeypatch.setattr(macs3,'native',lambda *_:pytest.fail('Existing regions invoked MACS3'))
    supplied=tmp_path/'researcher.bed';supplied.write_bytes(Path(peaks['peaks_path']).read_bytes())
    direct=SNAPATAC2.execute(dict(function_name='pp.make_peak_matrix',input_path=imported['matrix_path'],
        output_dir=str(tmp_path/'supplied-matrix'),peaks_path=str(supplied),
        kwargs=dict(counting_strategy='fragment')),'supplied-matrix')
    from agent.application.shared_resources import ScientificResourceLibrary
    from agent.application.native_raw_handoff import bind_plan,resource_evidence
    from agent.orchestration import AgentPlan,PlanStep
    library=ScientificResourceLibrary(tmp_path/'library')
    library.install('regions',dict(species='human',assembly='simulated',coordinate_system='zero_based_half_open',
        file=dict(path=str(supplied),sha256=peaks['peaks_sha256'])),label='Installed reference regions',
        version='simulated',provenance='Explicit technical fixture',usage='test')
    candidate=AgentPlan('reference','reference','Count against existing reference regions',(
        PlanStep('matrix','run_snapatac2',dict(function_name='pp.make_peak_matrix',input_path=imported['matrix_path'],
            output_dir=str(tmp_path/'reference-matrix'),kwargs=dict(counting_strategy='fragment'))),))
    bound=bind_plan(candidate,dict(catalog=resource_evidence(library.choices()),selected=None,custom={},species='human',assembly='simulated'))
    reference=SNAPATAC2.execute(_serialize(bound.steps[0].arguments),'reference-matrix')
    a=ad.read_h5ad(called['matrix_path']);b=ad.read_h5ad(direct['matrix_path'])
    assert a.obs_names.equals(b.obs_names) and a.var_names.equals(b.var_names)
    assert (a.X!=b.X).nnz==0 and np.issubdtype(a.X.dtype,np.integer)
    c=ad.read_h5ad(reference['matrix_path'])
    assert a.obs_names.equals(c.obs_names) and a.var_names.equals(c.var_names) and (a.X!=c.X).nnz==0
    rows=[line.split('\t') for line in Path(peaks['peaks_path']).read_text().splitlines()]
    assert a.var_names.tolist()==[f'{c}:{s}-{e}' for c,s,e,*_ in rows]
    # This source has 240 retained fragments in each cell, independent of labels.
    assert a.n_obs==3 and np.asarray(a.X.sum(axis=1)).ravel().tolist()==[240]*3
    macs3.verify_result(args,peaks)
    publication,_=macs3.load(peaks['manifest_path'],peaks['manifest_sha256'])
    assert macs3.recover(args,publication['execution_identity'])==peaks


@pytest.fixture
def native_runtime():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1':pytest.skip('Explicit native runtime qualification')
