"""scATAC region preparation through existing source and DNA semantic ports."""
from pathlib import Path
from agent.tools.analysis import scatac_regulatory_sequences as owner


def tool_spec(spec, output):
    from .registry import (ArtifactSemanticKind as A, PlanningSourceEligibility as S,
        SemanticConsumerPortSpec as Consumer, SemanticProducerPortSpec as Producer,
        _planning_argument as argument, _semantic_member as member, _semantic_request as request,
        _semantic_request_member as rm, _semantic_argument_port as scalar)
    optional = {
        'input_path': argument((str,Path), 'Original scATAC H5AD with genomic features.', artifacts=(A.RAW_SCATAC,), source=S.REQUEST_INPUT_OR_UPSTREAM_RESULT),
        'input_sha256': argument((str,), 'Exact captured input identity when supplied.', source=S.REQUEST_INPUT_ONLY),
        'da_path': argument((str,Path), 'Selected accepted replicate DA artifact; retain its exact feature-space lineage and contrast.', source=S.UPSTREAM_RESULT_ONLY),
        'annotation_manifest_path': argument((str,Path), 'Exact accepted annotation manifest for the original dataset.', source=S.UPSTREAM_RESULT_ONLY),
        'annotation_manifest_sha256': argument((str,), 'Exact accepted annotation checksum.', source=S.UPSTREAM_RESULT_ONLY),
        'candidate_manifest_path': argument((str,Path), 'Exact accepted region-selection publication for continuation without reselecting cells/features.', source=S.UPSTREAM_RESULT_ONLY),
        'candidate_manifest_sha256': argument((str,), 'Accepted region publication checksum.', source=S.UPSTREAM_RESULT_ONLY),
    }
    ports = [Consumer('dataset',(member('dataset','input_path'),),False,
        request_sources=(request('input_path',rm('dataset','input_path')),), accepted_upstream_types=('scatac_matrix_h5ad.v1',)),
        Consumer('differential_accessibility',(member('artifact','da_path'),),False,accepted_upstream_types=('differential_accessibility.v1',)),
        Consumer('annotation',(member('manifest_path','annotation_manifest_path'),member('manifest_sha256','annotation_manifest_sha256')),False,
            accepted_upstream_types=('scatac_cell_type_annotation.v1',)), scalar('input_sha256',required=False)]
    ports.append(Consumer('regions',(member('path','candidate_manifest_path'),member('sha256','candidate_manifest_sha256')),False,accepted_upstream_types=('scatac_regulatory_sequences.v1',)))
    for key, choices, description in (
        ('selection_method',('explicit','descriptive','differential_accessibility','accepted_candidates'),'Source modes are exclusive: explicit/descriptive (default) consume original H5AD dataset; differential_accessibility consumes accepted DA and its exact contrast; accepted_candidates consumes only accepted regions, optionally candidate_index/window_size, preserving their source declarations without H5AD scanning or prior selection settings.'),
        ('coordinate_source',('var_columns','feature_names'),'Explicit var columns (default names chrom/start/end) or chrom:start-end feature IDs. Neither establishes the coordinate convention.'),
        ('coordinate_system',('zero_based_half_open','one_based_closed'),'Actual declared source coordinate convention, required for an ordinary H5AD. Never infer from labels.'),
        ('matrix_semantics',('peak_counts','fragment_counts','insertion_counts','binary_accessibility'),'Actual selected matrix measurement, required only for descriptive ranking. No transformed values or inferred replicates.'),
        ('direction',('numerator','denominator'),'Direction of accepted DA effect; default numerator. Never reinterpret a condition contrast as a cell-type comparison.'),
    ):
        optional[key] = argument((str,), description, choices=choices, scientific_parameter=True, conversational_choice=True)
    for key, description in (
        ('species','Declared biological species, matching the qualified genome. Not inferred from cell names.'),
        ('assembly','Exact declared genome assembly, matching the installed reference. DA reuses its accepted lineage.'),
        ('feature_chrom_key','Exact chromosome var column; omitted var_columns uses chrom/start/end.'),
        ('feature_start_key','Exact start var column.'),('feature_end_key','Exact end var column.'),
        ('strand_key','Optional exact var strand column (+, -, .). Omit for unstranded reference-plus DNA.'),
        ('feature_ids','Comma-separated exact distinct source feature IDs for explicit selection, preserving requested order.'),
        ('count_slot','Selected matrix slot X (default) or layers/exact_name; measurement applies only to this slot.'),
        ('group_value','Exact observed or accepted annotation group label. No automatic cell-type validation or model-target mapping.'),
        ('comparison_group','Optional distinct exact group for descriptive detection-fraction difference, not DA.'),
        ('numerator_condition','Required exact accepted DA numerator condition.'),
        ('denominator_condition','Required exact accepted DA denominator condition.'),
    ):
        optional[key] = argument((str,),description,scientific_parameter=True,conversational_exact_string=True)
    optional['group_key'] = argument((str,), 'Exact obs grouping column in the selected dataset.', scientific_parameter=True, observation_source='dataset')
    for key, types, description in (
        ('top_n',(int,),'Number of candidates, default 10, maximum 1000; ties use original feature order.'),
        ('candidate_index',(int,),'1-based candidate in the explicitly selected accepted regions publication. Only with accepted_candidates; preserve its source region and observed evidence.'),
        ('max_peak_length',(int,),'Optional maximum original peak length before ranking/truncating the candidate count. No peak cropping; useful for a requested downstream context limit.'),
        ('min_fraction',(int,float),'Descriptive minimum group detection fraction, default 0; require at least one accessible cell. Not significance.'),
        ('window_size',(int,),'Optional real genomic context length. Expand around floor peak midpoint, never summit, clipping, invented flanks or silent cropping. Use 2114 only when requested for DeepPBMC; omit for exact peaks.'),
        ('max_fdr',(int,float),'Accepted DA maximum FDR, default 0.05, only for tested features in the actual contrast.'),
        ('min_abs_logfc',(int,float),'Accepted DA strictly positive directional effect threshold, default 0. Rank by FDR then directional effect.'),
    ):
        optional[key] = argument(types,description,scientific_parameter=True)
    ports += [scalar(k,required=False) for k in optional if k not in ('input_path','input_sha256','da_path','annotation_manifest_path','annotation_manifest_sha256','candidate_manifest_path','candidate_manifest_sha256')]
    ports.append(scalar('output_dir',required=True))
    return spec(owner.TOOL,owner.prepare_scatac_regulatory_sequences,{'output_dir':output},optional,owner.RESULT_FIELDS,ports,
        [Producer('sequences','dna_sequences.v1',(member('path','sequences_path'),member('sha256','sequences_sha256'))),
         Producer('regions','scatac_regulatory_sequences.v1',(member('path','manifest_path'),member('sha256','manifest_sha256')))],
        owner.KIND,owner.execute,'Select observed scATAC genomic regions and extract actual reference DNA for independent downstream sequence research.',
        ('Consumes ordinary H5AD directly and validates its own selected matrix and genomic features. No preliminary validate_scATAC_feature_space, pseudobulk, clustering or annotation step is required for supplied features/groups. Descriptive selection needs raw counts/binary accessibility and optional exact groups, not biological replication. DA selection reuses accepted replicate-aware statistics and exact contrast without edgeR rerun.',
         'Species, assembly and coordinate convention must be established. Preserve original peaks separately from real expanded windows. Unstranded regions use reference plus orientation. No invented DNA, clipping or summit; incompatible boundaries fail explicitly.',
         'Publish FASTA, peak/window BED, observed-evidence TSV and metadata. Downstream DHS64 prediction or DeepPBMC optimization is separate and only when requested, using exact DNA ports and model-specific human conditions and lengths. No ATAC-as-DNase truth or enhancer-function claim.'),policy=owner.POLICY)
