"""Sequence records and DNase predictions are distinct scientific semantic ports."""
from pathlib import Path
from agent.tools.analysis import sequence_prediction as owner


def classify(exception):
    from .registry import ErrorClassification
    from agent.schemas import ErrorCategory
    from agent.schemas.verification_authority import AuthorityError
    if isinstance(exception, AuthorityError):
        return ErrorClassification(ErrorCategory.VERIFICATION_ERROR, 'SEQUENCE_PUBLICATION_INVALID')
    if isinstance(exception, (ValueError, FileNotFoundError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, 'SEQUENCE_INPUT_INVALID')
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'SEQUENCE_EXECUTION_FAILED')


def tool_specs():
    from .registry import (ToolSpec, ResultContract, DurableToolHooks, ScientificAuthorityHooks,
        ArtifactSemanticKind as A, PlanningToolRole, PlanningSourceEligibility as S,
        SemanticToolSpec, SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer,
        _planning_argument as argument, _planning_result, _tool_planning,
        _semantic_member as member, _semantic_request as request, _semantic_request_member as rm,
        _semantic_argument_port as scalar)

    def bundle(port, prefix, artifact, required=False, upstream=()):
        args = {prefix+'_path':argument((str, Path), 'Exact captured scientific source or explicitly selected accepted output.',
                                      artifacts=(artifact,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
                prefix+'_sha256':argument((str,), 'Exact source digest, bound by Agent.', source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT)}
        source = Consumer(port, (member('path', prefix+'_path'),member('sha256',prefix+'_sha256')), required,
            request_sources=(request(prefix+'_path',rm('path',prefix+'_path'),rm('sha256',prefix+'_sha256')),),
            accepted_upstream_types=upstream)
        return args, source

    sequences, source = bundle('sequences','sequence',A.DNA_SEQUENCES,upstream=('dna_sequences.v1',))
    optional = dict(sequences)
    for name, description in (
        ('sequence', 'One DNA sequence quoted exactly from the researcher utterance. Never invent, truncate or rewrite bases; no upload is needed. Model-specific normalization belongs to the scientific owner.'),
        ('condition', 'Exact DHS64 biosample name from the qualified resource. Omit for all 64 conditions. Never substitute a nearest cell type.'),
        ('assembly', 'Explicit assembly of BED coordinates, matching a qualified library genome. Not required for supplied DNA/FASTA.')):
        optional[name] = argument((str,), description, scientific_parameter=True, conversational_exact_string=True)
    optional['head'] = argument((str,), 'Omit for continuous transformed DNase signal. Binary is the distinct sigmoid peak-call task, not enhancer functionality.',
        choices=('continuous','binary'), scientific_parameter=True, conversational_choice=True)
    output = argument((str,Path), 'Managed result destination.')

    def spec(name, function, required, optional, fields, ports, producers, kind, execute, description, notes, policy=owner.POLICY):
        return ToolSpec(name=name, function=function, required_arguments=required, optional_arguments=optional,
            result_contract=ResultContract('RegulatorySequenceResult', fields,
                planning_fields={k:_planning_result('Exact sequence publication and model-specific scientific result.',
                    bindable=True, artifact=A.DNA_SEQUENCES if k.startswith('sequences_') else A.SEQUENCE_PREDICTIONS)
                    for k in fields if k.endswith(('_path','_sha256'))}),
            durable_hooks=DurableToolHooks(execute,owner.recover), exception_classifier=classify,
            recovery_policy_version=policy,
            scientific_authority=ScientificAuthorityHooks(kind=kind, producer_ports=tuple(p.name for p in producers),
                describe=owner.describe, publication_record=owner.publication_record,
                reuse_result=owner.reuse_result, proof_metadata=owner.proof_metadata,
                independent_active_publications=True),
            planning=_tool_planning(PlanningToolRole.OPERATION,description,*notes),
            semantic_planning=SemanticToolSpec(consumer_ports=tuple(ports),producer_ports=tuple(producers)))

    predict = spec(owner.TOOL, owner.predict_regulatory_sequences, {'output_dir':output}, optional,
        owner.RESULT_FIELDS, [source, scalar('output_dir',required=True),
            *(scalar(k,required=False) for k in ('sequence','condition','head','assembly'))],
        [Producer('sequences','dna_sequences.v1',(member('path','sequences_path'),member('sha256','sequences_sha256'))),
         Producer('predictions','regulatory_predictions.v1',(member('path','manifest_path'),member('sha256','manifest_sha256')))],
        owner.KIND,owner.execute,'Predict human DNase accessibility from DNA sequences using released DHS64 folds 0, 1 and 3.',
        ('Supply exactly one sequence string or sequences source: ordinary FASTA or BED3–6 with explicit qualified assembly. Accepted sequences are reusable; no H5AD, cells or embeddings are involved.',
         'Default continuous-head arithmetic fold mean; retain all fold values and disagreement. ACGT only, 1–500 bases, native trailing zero padding, no clipping or invented genomic context. This is not enhancer activity, expression, MPRA or ATAC prediction.',
         'Prediction needs no observed truth. Use evaluate_regulatory_predictions separately only for supplied compatible observed DNase measurements.'))
    prediction, prior = bundle('predictions','prediction_manifest',A.SEQUENCE_PREDICTIONS,True,('regulatory_predictions.v1',))
    observed, truth = bundle('observations','observations',A.SEQUENCE_OBSERVATIONS,True)
    evaluate = spec(owner.EVALUATE, owner.evaluate_regulatory_predictions,
        dict(prediction,**observed,output_dir=output),{},owner.EVAL_FIELDS,
        [prior,truth,scalar('output_dir',required=True)],
        [Producer('evaluation','regulatory_prediction_evaluation.v1',(member('path','manifest_path'),member('sha256','manifest_sha256')))],
        owner.EVAL_KIND,owner.execute_evaluation,'Evaluate accepted DHS64 predictions against explicitly supplied compatible observed DNase measurements.',
        ('Never rerun prediction for evaluation. TSV columns: sequence_id, condition, measurement, value, provenance (optional sequence_sha256); exact identity, condition and native transform are required. Correspondence is never inferred from row count.',
         'Continuous: RMSE, MAE, Pearson and Spearman; binary peak calls: AUROC, average precision, Brier and log loss. MPRA, ATAC and gene expression are not DNase ground truth. Undefined metrics are null.'))
    from agent.tools.analysis import sequence_optimization as design
    options = dict(sequences)
    for key, description in (
        ('sequence', 'Exact researcher-supplied DNA, 2,114 real A/C/G/T bases. No fabricated flanks, truncation or padding.'),
        ('assembly', 'Explicit assembly for BED coordinates and qualified human genome; not required for FASTA or direct DNA.'),
        ('comparison_conditions', 'Comma-separated exact DeepPBMC condition names quoted from user intent. Omit to penalize increases in all six non-target conditions. This does not bound absolute changes.'),
        ('protected_regions', 'User-specified protected 0-based half-open sequence intervals in start:end,start:end notation. Unchanged positions are verified in the returned DNA.')):
        options[key] = argument((str,),description,scientific_parameter=True,conversational_exact_string=True)
    for key, types, description in (
        ('max_changes',(int,), 'Maximum final changed bases versus the starting sequence (1–100; default 5). Actual Hamming distance is verified, not inferred from iterations.'),
        ('iterations',(int,), 'Native substitution steps, at most max_changes; default max_changes. Native search can revisit positions and can worsen the objective.'),
        ('edit_start',(int,), 'First editable position, 0-based in supplied oriented DNA, default 0.'),
        ('edit_end',(int,), 'Exclusive end of editable interval, default 2114. No edits outside the interval.'),
        ('comparison_weight',(int,float), 'Nonnegative penalty for comparison-cell score increases, default 1. Native objective is target score minus one seventh weighted comparison scores; decreases improve it. Zero maximizes target alone.')):
        options[key] = argument(types,description,scientific_parameter=True)
    target = argument((str,), 'Exact supported DeepPBMC target condition from the installed model, required. Never substitute unsupported cell types.',scientific_parameter=True,conversational_exact_string=True)
    optimize = spec(design.TOOL,design.optimize_regulatory_sequences,dict(output_dir=output,condition=target),options,
        design.RESULT_FIELDS,[source,scalar('output_dir',required=True),scalar('condition',required=True),
            *(scalar(k,required=False) for k in options if k not in sequences)],
        [Producer('sequences','dna_sequences.v1',(member('path','sequences_path'),member('sha256','sequences_sha256'))),
         Producer('design','regulatory_sequence_design.v1',(member('path','manifest_path'),member('sha256','manifest_sha256')))],
        design.KIND,design.execute,'Optimize existing human regulatory DNA for a supported PBMC accessibility objective using native CREsted DeepPBMC in silico evolution.',
        ('No DHS64 prerequisite or score reuse. Supply direct DNA, ordinary FASTA, exact 2114-base BED3–6 with qualified human assembly, or explicitly selected accepted DNA sequences. No arbitrary cell conditions or gene-expression/enhancer-activity claims.',
         'One final candidate per starting sequence (up to 32 inputs); deterministic native search does not promise distinct alternatives. No de novo generation. Native weighted difference penalizes comparison increases, not absolute changes or hard score bounds. Clarify incompatible objectives.',
         'Return original and candidate scores under the same model, actual substitutions, constraints, native trajectory and reusable candidate sequences. Preserve explicit edit/protection coordinates and budget; do not invent user preferences. Biological improvement is unvalidated.'),policy=design.POLICY)
    return predict,evaluate,optimize
