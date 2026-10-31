"""One native peak caller and ordinary derived-feature/population ports."""
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
    required={};optional={};ports=[]
    for port,prefix,types,mandatory in (
        ('fragments','fragments_manifest',('scatac_fragments.v2',),True),
        ('selected_cells','selected_cells_manifest',('scatac_cell_selection.v1',),False),
        ('explicit_cells','explicit_cells_manifest',('scatac_explicit_cells.v1',),False),
        ('barcodes','barcodes',(),False)):
        target=required if mandatory else optional
        for suffix in ('path','sha256'):
            target[prefix+'_'+suffix]=arg((str,Path) if suffix=='path' else (str,),
                'Exact source-bound fragments or population. Choose at most one population input; omission means all eligible records in one library.',
                artifacts=(A.ORDERED_CELL_IDS,) if port=='barcodes' else (),
                source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT if types else S.REQUEST_INPUT_ONLY)
        names=('manifest_path','manifest_sha256') if prefix.endswith('manifest') else ('path','sha256')
        members=tuple(member(n,prefix+'_'+s) for n,s in zip(names,('path','sha256')))
        ports.append(Consumer(port,members,mandatory,request_sources=(request(prefix+'_path',*(rm(n,prefix+'_'+s) for n,s in zip(names,('path','sha256')))),),accepted_upstream_types=types))
    for key in ('matrix_manifest_path','matrix_manifest_sha256','group_analysis_path','group_analysis_sha256'):
        optional[key]=arg((str,Path) if key.endswith('_path') else (str,),
            'Exact accepted count matrix and clustering; original-file identity and ordered namespace/barcode cells must agree with the fragments.',source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)
    ports.extend((Consumer('matrix',(member('manifest_path','matrix_manifest_path'),member('manifest_sha256','matrix_manifest_sha256')),False,
        accepted_upstream_types=('scatac_cell_by_features.v1','scatac_cell_by_ccre.v1')),
        Consumer('clustering',(member('analysis','group_analysis_path'),member('sha256','group_analysis_sha256')),False,
        accepted_upstream_types=('clustered_analysis.v1',))))
    optional['group_label']=arg((str,),'Exact accepted clustering label selecting one population; requires its exact original fragment-derived count matrix.',scientific_parameter=True,conversational_exact_string=True)
    ports.append(scalar('group_label',required=False))
    required['output_dir']=arg((str,Path),'Managed native peak publication directory.')
    required['effective_genome_size']=arg((int,float),'Explicit effectively mappable genome size for the actual assembly and scope; never substitute total FASTA length.',scientific_parameter=True)
    required['genome_size_basis']=arg((str,),'Researcher-stated scientific basis and applicability of effective genome size.',scientific_parameter=True,conversational_exact_string=True)
    optional['library_namespace']=arg((str,),'Select one existing library namespace in a multi-library artifact. Omit for an upstream single-library import; import namespace assigns identity, this optional selector only disambiguates existing libraries. Never pool biological libraries.',scientific_parameter=True,conversational_exact_string=True)
    optional['qvalue']=arg((int,float),'Native MACS3 peak-calling q-value cutoff, default 0.05; not differential accessibility significance.',scientific_parameter=True)
    optional['exclude_contigs']=arg((list,),'Explicit exact reference contig names to exclude; default none. No automatic aliases or mitochondrial filtering.',scientific_parameter=True)
    ports.extend(scalar(k,required=k in required) for k in ('output_dir','effective_genome_size','genome_size_basis','library_namespace','qvalue','exclude_contigs'))
    return ToolSpec(name=owner.TOOL,function=owner.call_scATAC_peaks,required_arguments=required,optional_arguments=optional,
        parameter_prerequisites=owner.parameter_prerequisites,
        result_contract=ResultContract('ScATACPeaksResult',owner.RESULT_FIELDS,planning_fields={k:_planning_result(
            'Exact experimental peak publication, derived feature reference and ordered population. Empty calls have no usable reference.',bindable=True,artifact=A.SCATAC_PEAKS)
            for k in owner.RESULT_FIELDS if k.endswith(('_path','_sha256'))}),
        durable_hooks=DurableToolHooks(owner.execute,owner.recover),exception_classifier=classify,recovery_policy_version=owner.POLICY,
        scientific_authority=ScientificAuthorityHooks(kind=owner.KIND,producer_ports=('peaks','reference','cells'),describe=owner.describe,
            publication_record=owner.publication_record,reuse_result=owner.reuse_result,proof_metadata=owner.proof_metadata,independent_active_publications=True),
        planning=_tool_planning(PlanningToolRole.OPERATION,'Discover experiment-specific accessible genomic peaks using native MACS3 on one scATAC fragment population.',
            'Consumes qualified canonical fragments from ordinary external import, FASTQ or BAM. External files need actual source history, assembly/reference and namespace via existing import; no historical Agent receipt is required. Whole library, QC-selected or supplied exact barcodes; no clustering prerequisite.',
            'FRAG paired-fragment mode, one canonical record per count (--max-count 1), no second Tn5 shift. Explicit effective genome size and scientific basis are required. Native q-values concern peak calling, never replicate-level differential accessibility.',
            'Publishes native narrowPeak, summits, XLS, figures and a regulatory-feature-reference.v1 peak_set plus ordered cells. Pass reference and cells to build_scATAC_cell_by_features with the exact source fragments to construct fragment-overlap count H5AD; choose downstream analysis independently.',capability_ids=('raw_preprocessing',)),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=(
            Producer('peaks','scatac_peaks.v1',(member('manifest_path'),member('manifest_sha256'))),
            Producer('reference','regulatory_feature_reference.v1',(member('manifest_path','reference_manifest_path'),member('manifest_sha256','reference_manifest_sha256'))),
            Producer('cells','scatac_explicit_cells.v1',(member('manifest_path','cells_manifest_path'),member('manifest_sha256','cells_manifest_sha256'))))))
