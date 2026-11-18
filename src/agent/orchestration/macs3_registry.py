"""Native peak calling on an accepted SnapATAC2 fragment population."""
from pathlib import Path
from agent.tools.analysis import macs3 as owner


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception,AuthorityError):return ErrorClassification(ErrorCategory.VERIFICATION_ERROR,'PEAK_PUBLICATION_INVALID')
    if isinstance(exception,(ValueError,FileNotFoundError)):return ErrorClassification(ErrorCategory.USER_INPUT_ERROR,'PEAK_INPUT_OR_RESOURCE_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR,'PEAK_NATIVE_EXECUTION_FAILED')


def tool_spec():
    from .registry import (ToolSpec,ResultContract,DurableToolHooks,ScientificAuthorityHooks,
        ArtifactSemanticKind as A,PlanningToolRole,PlanningSourceEligibility as S,SemanticToolSpec,
        SemanticConsumerPortSpec as Consumer,SemanticProducerPortSpec as Producer,_planning_argument as arg,
        _planning_result,_tool_planning,_semantic_member as member,_semantic_request as request,
        _semantic_request_member as rm,_semantic_argument_port as scalar)
    required=dict(input_path=arg((str,Path),'Native fragment-backed H5AD defining the exact peak-calling population; use the accepted QC/filter result when requested.',
        artifacts=(A.RAW_SCATAC,),source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        output_dir=arg((str,Path),'Managed native peak publication directory.'),
        effective_genome_size=arg((int,float),'Explicit effectively mappable genome size for the actual assembly and scope; never substitute total FASTA length.',scientific_parameter=True),
        genome_size_basis=arg((str,),'Researcher-stated scientific basis and applicability of effective genome size.',scientific_parameter=True,conversational_exact_string=True))
    optional=dict(input_sha256=arg((str,),'Captured exact fragment-backed H5AD identity.'),
        qvalue=arg((int,float),'Native MACS3 peak-calling q-value cutoff, default 0.05; not differential accessibility significance.',scientific_parameter=True),
        exclude_contigs=arg((list,),'Explicit exact reference contig names to exclude; default none. No automatic aliases or mitochondrial filtering.',scientific_parameter=True))
    ports=[Consumer('dataset',(member('dataset','input_path'),),True,
        request_sources=(request('input_path',rm('dataset','input_path')),),
        accepted_upstream_types=('native_matrix_h5ad.v1',))]
    ports.extend(scalar(k,required=k in required) for k in ('output_dir','effective_genome_size','genome_size_basis','qvalue','exclude_contigs'))
    from .native_delivery import delivery_spec
    return ToolSpec(source_encodings={'dataset':('h5ad',)}, source_input_types={'dataset':('h5ad','external_fragments')}, scientific_delivery=delivery_spec(owner,'scientific_peaks',figure=('distributions.png','peak_distributions')),
        name=owner.TOOL,function=owner.call_scATAC_peaks,required_arguments=required,optional_arguments=optional,
        preparation_owner=owner.prepare_operation, parameter_prerequisites=owner.parameter_prerequisites,
        result_contract=ResultContract('ScATACPeaksResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result(
            'Exact experimental peak publication, native BED regions and ordered source cells. Empty calls contain no peaks.',bindable=True,artifact=A.SCATAC_PEAKS)
            for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('peaks',),describe=owner.describe,
            publication_record=owner.publication_record,reuse_result=owner.reuse_result,proof_metadata=owner.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Discover experimental accessible peaks with native MACS3 from the selected SnapATAC2 fragment-backed H5AD.',
            'BAM first uses native pp.make_fragment_file with its actual barcode tag/regex, then pp.import_fragments; native QC/filtering precedes peak calling when requested. The selected H5AD cell population is pooled without cell-type labels or clustering prerequisites.',
            'SnapATAC2 exports its native paired fragments. MACS3 uses FRAG --max-count 1 with no second Tn5 shift. Explicit effective genome size and its scientific basis are required; peak q-values are not differential accessibility.',
            'The peaks port supplies exact native BED regions to run_snapatac2 pp.make_peak_matrix. Its dataset must be the selected fragment-backed H5AD, not the BED. Retain that source for Gene Activity. Installed reference regions (including compatible cCREs) and researcher-supplied BED use the same peak-matrix path directly; neither requires peak calling.',capability_ids=('raw_preprocessing',)),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            Producer('peaks','scatac_peaks.v2',(member('path','peaks_path'),member('sha256','peaks_sha256')),lineage_from_port='dataset'),)))
