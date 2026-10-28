"""EpiZoo declarations in the existing attachment/prerequisite handoff.

Language interpretation remains typed provider output. This module binds exact
files and existing resources; it neither plans science nor implements EpiZoo.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import re
from agent.schemas.orchestration import _serialize
from .session_state import digest

TOOLS = ('adopt_scATAC_cell_by_features', 'adapt_epizoo_species', 'posttrain_epizoo_model', 'epizoo_embed_cells',
         'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap')


class EpiZooResourcePrerequisite(ValueError):
    """Resolvable scientific-resource prerequisite; never an execution retry."""


@dataclass(frozen=True)
class EpiZooDeclaration:
    operation: str | None = None
    source: str | None = None
    scientific_name: str | None = None
    taxonomy_id: int | None = None
    assembly: str | None = None
    matrix_semantics: str | None = None
    strategy: str | None = None
    feature_category: str | None = None
    resource: str | None = None
    genome_resource: str | None = None
    genome_source: str | None = None
    genome_index_source: str | None = None
    checkpoint_source: str | None = None
    chain_source: str | None = None
    reference_species: str | None = None

    def __post_init__(self):
        for key, choices in dict(operation=('inference', 'posttraining', 'initialization', 'posttraining_initialized'),
                matrix_semantics=('fragment_counts', 'binary_accessibility'),
                strategy=('de_novo', 'mapped_reference'), reference_species=('human','mouse'), feature_category=('peak_set', 'regulatory_regions')).items():
            if getattr(self, key) not in (None, *choices):
                raise ValueError('Unsupported EpiZoo scientific declaration.')
        for key, value in asdict(self).items():
            if value is None: continue
            if key == 'taxonomy_id':
                if type(value) is not int or value < 1: raise ValueError('Invalid taxonomy identity.')
            elif type(value) is not str or not value.strip() or len(value) > 256 or not value.isprintable():
                raise ValueError('Invalid EpiZoo declaration.')
            elif (key.endswith('_source') or key == 'source') and re.fullmatch(r'source\.(?:[0-9]|1[0-5])', value) is None:
                raise ValueError('Invalid scientific attachment handle.')
            elif key.endswith('resource') and re.fullmatch(r'resource\.[a-z][a-z0-9_-]{0,63}', value) is None:
                raise ValueError('Invalid scientific resource choice.')
        handles = sources(self)
        if len(handles) != len(set(handles)): raise ValueError('Scientific file roles must be distinct.')


def target(declared):
    if declared.operation=='posttraining_initialized': return 'posttrain_epizoo_model'
    return 'adapt_epizoo_species' if declared.operation in ('posttraining','initialization') else 'epizoo_embed_cells'


def sources(declared):
    return [getattr(declared, key) for key in ('source', 'genome_source', 'genome_index_source', 'checkpoint_source','chain_source')
            if getattr(declared, key) is not None]


def resources(input_sets):
    return tuple(s for s in input_sets if any(k in s.inputs() for k in
        ('target_model_manifest_path', 'adaptation_spec_path', 'qualified_genome', 'library_genome', 'library_epizoo_transfer')))


def captured_models(sessions, state):
    """Project exact captured outputs; never discover another Session's models."""
    from .input_sets import ScientificInputSet
    from agent.orchestration.verification_authority import _accepted_step
    if state.active_revision_id is None:
        return ()
    revisions={r.revision_id:r for r in state.revisions}
    choices={};revision=revisions[state.active_revision_id];distance=0
    while revision is not None:
        sessions._validate_revision(revision)
        for output in revision.outputs:
            run=sessions._application.run_store.load(output.run_id)
            steps={s.step_id:s for s in run.steps}
            planned={s.step_id:s for s in run.plan.steps}
            pending=[output.step_id];seen=set()
            while pending:
                step_id=pending.pop()
                if step_id in seen: continue
                seen.add(step_id);step=steps[step_id]
                pending.extend(planned[step_id].depends_on)
                if step.tool_name in ('adapt_epizoo_species','posttrain_epizoo_model'):
                    _accepted_step(sessions._application.run_store,output.run_id,step_id)
                    values=dict(target_model_manifest_path=step.result['manifest_path'],
                        target_model_manifest_sha256=step.result['manifest_sha256'],
                        target_model_source_run_id=output.run_id,target_model_source_step_id=step_id)
                elif step.tool_name=='epizoo_embed_cells' and all(step.resolved_arguments.get(k) for k in
                        ('target_model_manifest_path','target_model_manifest_sha256','target_model_source_run_id','target_model_source_step_id')):
                    _accepted_step(sessions._application.run_store,output.run_id,step_id)
                    values={k:step.resolved_arguments[k] for k in ('target_model_manifest_path',
                        'target_model_manifest_sha256','target_model_source_run_id','target_model_source_step_id')}
                else: continue
                identity='captured-model-'+digest(values)[:40]
                if identity in choices: continue
                values.update(captured_revision_id=revision.revision_id,captured_model_distance=distance)
                choices[identity]=ScientificInputSet(identity,
                    'EpiZoo model from accepted result '+str(revision.revision_id[:12])+' ('+step.tool_name+')',values,h5ad_companion=True)
        revision=revisions.get(revision.parent_revision_id);distance+=1
    return tuple(choices.values())


def evidence(input_sets):
    from agent.tools.models.epizoo_inputs import observe_resource
    from .local_resources import LocalResourceRecord
    from .shared_genome import check_content
    result = {}
    for choice in resources(input_sets):
        try:
            values = choice.inputs()
            if 'library_genome' in values:
                resource = values['library_genome']['resource']
                result[choice.input_set_id] = dict(operation='genome', species=resource['species'], assembly=resource['assembly'])
            elif 'qualified_genome' in values:
                record = LocalResourceRecord(**values['qualified_genome'])
                check_content(record)
                result[choice.input_set_id] = dict(operation='genome', **_serialize(record.reference_qualification['requirement']))
            else:
                result[choice.input_set_id] = observe_resource(values)
        except (ValueError, OSError, TypeError, KeyError):
            result[choice.input_set_id] = None
    return result


def schema(source_rows, resource_rows):
    from .turn_decisions import _object
    def optional(value): return {'anyOf': [value, {'type': 'null'}]}
    fields = {key: optional({'type': 'string', 'maxLength': 256}) for key in EpiZooDeclaration.__dataclass_fields__}
    for key, choices in dict(operation=['inference', 'initialization', 'posttraining_initialized'], strategy=['de_novo', 'mapped_reference'],
            matrix_semantics=['fragment_counts', 'binary_accessibility'], feature_category=['peak_set', 'regulatory_regions'],reference_species=['human','mouse'],
            **{k: [s['source'] for s in source_rows] for k in ('source', 'genome_source', 'genome_index_source', 'checkpoint_source','chain_source')},
            **{k: [r['handle'] for r in resource_rows] for k in ('resource', 'genome_resource')}).items():
        fields[key] = optional({'type': 'string', 'enum': choices}) if choices else {'type': 'null'}
    fields['taxonomy_id'] = optional({'type': 'integer', 'minimum': 1})
    fields['scientific_name']['anyOf'][0]['description'] = (
        'Explicit scientific species name for the target identity. Common names '
        'are language descriptions, not a qualified scientific_name. Unknown is null.')
    return _object(fields)


def _species_identity(declared):
    """Use the reference owner's exact scientific identity, without alias guesses."""
    from agent.tools.data.regulatory_feature_reference import validate_species
    if declared.scientific_name is None or declared.taxonomy_id is None:
        return None
    try:
        return asdict(validate_species(dict(scientific_name=declared.scientific_name,
                                           taxonomy_id=declared.taxonomy_id)))
    except ValueError:
        return None


def resolve(declared, choices, observations, *, requested=None, operation=None, source=None, checkpoint=None):
    """Unique compatible qualified choice; explicit invalid choices never fall back."""
    compatible = []
    identity = _species_identity(declared)
    for choice in choices:
        info = observations[choice.input_set_id]
        if info is None: continue
        wanted=operation or declared.operation
        if wanted=='posttraining_initialized':
            if info['operation']!='inference' or info.get('model_state')!='initialization': continue
        elif info['operation']!=wanted: continue
        if wanted=='initialization' and info.get('universal_target'):
            compatible.append(choice);continue
        if any(v is None for v in (declared.scientific_name, declared.taxonomy_id, declared.assembly)): continue
        if (identity is None or info['species'] != identity
                or info['assembly'] != declared.assembly): continue
        if info['operation'] == 'inference' and (info['matrix_semantics'] != declared.matrix_semantics
                or source is None or info['feature_membership_sha256'] != source['feature_membership_sha256']): continue
        if checkpoint is not None and info.get('checkpoint_sha256') != checkpoint: continue
        if info['operation'] == 'posttraining':
            if declared.strategy == 'mapped_reference' and info['mapping'] is None: continue
            if declared.strategy == 'de_novo' and info['mapping'] is not None: continue
            if declared.matrix_semantics == 'binary_accessibility' and declared.strategy == 'mapped_reference': continue
        compatible.append(choice)
    if requested is not None:
        selected = [c for c in compatible if 'resource.' + c.input_set_id == requested]
        return (selected[0], []) if len(selected) == 1 else (None, ['The selected scientific resource is unavailable or incompatible; supply its compatible resources or explicitly choose another.'])
    # A resource cannot establish declarations about the uploaded experiment.
    if (any(v is None for v in (declared.scientific_name, declared.taxonomy_id, declared.assembly))
            or operation != 'genome' and (declared.matrix_semantics is None or declared.operation is None
                or declared.operation == 'inference' and source is None)):
        return None, []
    aliases = {}
    captured = [c for c in compatible if 'captured_revision_id' in c.inputs()]
    if captured and requested is None:
        nearest=min(c.inputs().get('captured_model_distance',0) for c in captured)
        captured=[c for c in captured if c.inputs().get('captured_model_distance',0)==nearest]
    if captured:
        compatible = captured
    for choice in compatible:
        values = choice.inputs()
        # Only exact definitions are aliases; distinct vocabularies/weights clarify.
        key = digest(values)
        aliases[key] = min(choice, aliases.get(key, choice), key=lambda c: c.input_set_id)
    compatible = list(aliases.values())
    if len(compatible) == 1: return compatible[0], []
    label = 'genome' if operation == 'genome' else 'EpiZoo checkpoint and its qualified vocabulary/resources'
    if not compatible:
        message = 'Supply a compatible qualified ' + label + '.'
        if declared.strategy == 'mapped_reference' and operation != 'genome':
            message += ' mapped_reference requires qualified target-to-source mapping, retained source vocabulary, chain and pinned mapping tools.'
        return None, [message]
    return None, ['Choose a compatible ' + label + ': ' + '; '.join(c.label for c in compatible) + '.']


def requirements(binding, declared, choices, observations, source, pinned):
    missing = [text for key, text in (
        ('operation', 'Specify embeddings/inference or post-training.'), ('source', 'Identify the scATAC H5AD.'),
        ('scientific_name', 'Declare the scientific species name.'), ('taxonomy_id', 'Declare its taxonomy ID.'),
        ('assembly', 'Declare the genome assembly.'), ('matrix_semantics', 'Declare fragment counts or binary accessibility from processing history.'))
        if getattr(declared, key) is None]
    if source is not None and source.get('transfer_resource_problem'):
        missing.append(source['transfer_resource_problem'])
    # Old observations retain their exact historical prerequisite text. The
    # current source observation adds this owner check without reinterpreting
    # an old common-name declaration or inventing a scientific name from it.
    if (source is not None and source.get('interface_version') == 2
            and declared.scientific_name is not None and declared.taxonomy_id is not None
            and _species_identity(declared) is None):
        missing.append('Provide the scientific species name and its taxonomy ID; a common name alone does not establish this target-model identity.')
    if any(int(h.split('.')[1]) >= len(binding.members) for h in sources(declared)):
        raise ValueError('EpiZoo file role is outside captured attachments.')
    if source is not None and source['source_sha256'] != binding.members[int(declared.source.split('.')[1])]['source_sha256']:
        raise ValueError('Source observation identity changed.')
    checkpoint = (binding.members[int(declared.checkpoint_source.split('.')[1])]['source_sha256']
                  if declared.checkpoint_source else None)
    incomplete = (source is not None and source.get('interface_version') == 2
        and (_species_identity(declared) is None or any(getattr(declared, key) is None
             for key in ('operation', 'assembly', 'matrix_semantics'))))
    # Unknown experimental history cannot establish compatibility or a conflict.
    # Keep the explicit preference for later qualification; never replace it.
    selected, questions = ((None, []) if incomplete else resolve(
        declared, choices, observations, requested=declared.resource or pinned,
        source=source, checkpoint=checkpoint))
    missing += questions
    genome = None
    if declared.operation == 'inference':
        if any(getattr(declared, k) is not None for k in ('strategy', 'feature_category', 'genome_resource', 'genome_source', 'genome_index_source')):
            missing.append('Target inference uses the accepted model vocabulary; remove unrelated training/reference declarations by starting a separate task.')
    elif declared.operation in ('posttraining','initialization'):
        if declared.operation=='initialization' and declared.strategy=='mapped_reference' and declared.reference_species is None:
            missing.append('Choose the human hg38 or mouse mm10 pretrained reference for explicit mapped transfer; automatic assessment can evaluate both when no strategy is requested.')
        if declared.operation=='posttraining' and declared.strategy is None: missing.append('Choose de_novo or mapped_reference post-training.')
        if declared.feature_category is None: missing.append('Declare whether the exact H5AD regions are peaks or regulatory regions.')
        if declared.genome_source is None:
            if declared.genome_index_source is not None: missing.append('Identify the FASTA paired with this FAI.')
            genome, questions = resolve(declared, choices, observations, requested=declared.genome_resource, operation='genome')
            missing += questions
            if genome is None and not declared.genome_resource:
                missing.append('You may instead attach the matching plain genome FASTA and optional FAI to this pending task.')
        elif declared.genome_resource is not None:
            missing.append('Choose either the uploaded FASTA or the qualified genome explicitly; both cannot supply one role.')
    if genome is not None: declared = replace(declared, genome_resource='resource.' + genome.input_set_id)
    return declared, missing, selected


def binding(candidates, declared, selected, arguments, reference=None):
    from .local_resources import RegisteredInputGroup, group_role_inputs
    from agent.orchestration.registry import build_default_tool_registry
    values = selected.inputs()
    allowed = ({'target_model_manifest_path', 'target_model_manifest_sha256', 'target_model_source_run_id', 'target_model_source_step_id'}
               if declared.operation in ('inference','posttraining_initialized') else {'adaptation_spec_path', 'adaptation_spec_sha256'})
    # The catalog supplies model resources only. It cannot override task settings.
    values = {k: v for k, v in values.items() if k in allowed}
    values.update(arguments)
    declarations = dict(species=dict(scientific_name=declared.scientific_name, taxonomy_id=declared.taxonomy_id),
                        assembly=declared.assembly, matrix_semantics=declared.matrix_semantics)
    if declared.operation in ('posttraining','initialization'):
        if declared.strategy is not None: declarations['strategy'] = declared.strategy
    if any(k in arguments and arguments[k] != v for k, v in declarations.items()):
        raise ValueError('Explicit scientific arguments conflict with the captured declarations.')
    values.update(declarations)
    role = dict(tool='epizoo_embed_cells', port='dataset', source=declared.source)
    if declared.operation in ('posttraining','initialization'):
        values.update(reference_manifest_path=reference['manifest_path'],
                      reference_manifest_sha256=reference['manifest_sha256'])
        role = dict(tool='adopt_scATAC_cell_by_features', port='source', source=declared.source)
    if declared.operation=='posttraining_initialized':
        from agent.tools.models.epizoo_adaptation import publication
        model=publication.load_manifest(values['target_model_manifest_path'],values['target_model_manifest_sha256'])
        from agent.tools.data import scatac_matrix_contract
        from agent.tools.data.neutral_matrix_binding import reference_binding
        pin=model['arguments']['matrices'][0]
        matrix=scatac_matrix_contract.load_manifest_bytes(Path(pin['manifest_path']).read_bytes())
        ref=reference_binding(matrix)
        values.update(reference_manifest_path=ref['manifest_path'],reference_manifest_sha256=ref['manifest_sha256'])
        role=dict(tool='adopt_scATAC_cell_by_features',port='source',source=declared.source)
    if declared.operation=='initialization' and reference and 'adaptation_spec_path' in reference:
        values.update({k:reference[k] for k in ('adaptation_spec_path','adaptation_spec_sha256')})
    result = RegisteredInputGroup(candidates.members, values, [role],
        resource_choice=selected.resource_choice(), attachment_candidates=True)
    return replace(result, execution_inputs=group_role_inputs(result, [role], build_default_tool_registry()))


def prepare(owner, session_id, candidates, declared, selected, arguments, *, input_sets, resource_owner, origin_turn_id):
    from agent.tools.models.epizoo_inputs import qualify_resource
    from . import reference_supplementation as refs
    from .local_resources import LocalResourceRecord
    from types import SimpleNamespace
    from .raw_handoff import declaration_dict
    identity = digest(dict(sources=candidates.attribution(), declaration=declaration_dict(declared), resource=selected.resource_choice()))
    source = owner.load(candidates.members[int(declared.source.split('.')[1])]['resource_id'])
    roots = [p for p in resource_owner._approved if Path(source.source_path).is_relative_to(p)
             and p.is_relative_to(owner._workspace.root)]
    if not roots: raise ValueError('An approved private upload root is required.')
    directory = owner._workspace._ensure_directory(max(roots, key=lambda p: len(p.parts)) / 'epizoo_handoffs')
    directory = owner._workspace._ensure_directory(directory / digest(session_id))
    root = owner._workspace._ensure_directory(directory / identity)
    qualify_resource(selected.inputs())
    reference = None
    if declared.operation in ('posttraining','initialization'):
        required = dict(species=dict(scientific_name=declared.scientific_name, taxonomy_id=declared.taxonomy_id),
                        assembly=declared.assembly, feature_category=declared.feature_category)
        if declared.genome_resource:
            from .shared_genome import bind
            choices = [s for s in input_sets if 'resource.' + s.input_set_id == declared.genome_resource]
            if len(choices) != 1: raise ValueError('Captured genome choice unavailable.')
            values = choices[0].inputs()
            record = values['library_genome'] if 'library_genome' in values else LocalResourceRecord(**values['qualified_genome'])
            bound = bind(resource_owner, record, source, session_id,
                SimpleNamespace(turn_id=origin_turn_id, admitted={'reference_requirement': required}))
            reference = dict(manifest_path=bound.source_path, manifest_sha256=bound.source_sha256)
        else:
            fasta = owner.load(candidates.members[int(declared.genome_source.split('.')[1])]['resource_id'])
            index = owner.load(candidates.members[int(declared.genome_index_source.split('.')[1])]['resource_id']) if declared.genome_index_source else None
            fasta = owner.bind_raw_source('epizoo-genome:' + identity, fasta, session_id=session_id, input_type='fasta', source_index=index)
            from agent.tools.data import regulatory_feature_reference as scientific, scatac_reference as r
            bed = root / 'features.bed'
            if not bed.exists(): refs._write_bed(bed, refs._feature_names(source.source_path))
            fai = refs.prepare_fai(owner, fasta, root)
            value = scientific.build_regulatory_feature_reference(species=required['species'],
                target_assembly=declared.assembly, fasta_path=fasta.source_path, fai_path=fai,
                feature_bed_path=bed, feature_category=declared.feature_category,
                genome_provenance=r.SourceProvenance('caller_supplied', source='Task-declared genome; publisher and assembly origin not independently established.'),
                feature_provenance=r.SourceProvenance('caller_supplied', source='Exact ordered uploaded H5AD feature axis.'))
            path = root / 'reference.json'
            if path.exists():
                if scientific.load_regulatory_feature_reference(path)[1] != value: raise ValueError('Reference binding changed.')
                reference = dict(manifest_path=str(path), manifest_sha256=scientific.load_regulatory_feature_reference(path)[2])
            else:
                reference = scientific.publish_regulatory_feature_reference(value, path)
            reference = {k: reference[k] for k in ('manifest_path', 'manifest_sha256')}
            from agent.tools.data import chromap_reference_index
            from agent.tools.data.authority_context import VerificationContext
            from .shared_genome import persist_private
            files = VerificationContext().files([value.genome.fasta.path, value.genome.fai.path, value.features.bed.path])
            record = resource_owner.register('epizoo-reference:' + identity, path, input_type='regulatory_reference',
                label='Private EpiZoo feature reference', attribution='Owner-qualified exact H5AD regions and declared genome.',
                session_id=session_id, reference_qualification=dict(requirement=required,
                    reference_identity_sha256=value.reference_identity_sha256,
                    ordered_feature_sha256=value.features.ordered_feature_sha256, files=files,
                    session_id=session_id, origin_turn_id=origin_turn_id, source_record_sha256=source.record_sha256,
                    scope='session', review=None))
            persist_private(resource_owner, record.resource_id)
    if declared.operation=='initialization':
        chain=(owner.load(candidates.members[int(declared.chain_source.split('.')[1])]['resource_id'])
               if declared.chain_source else None)
        reference.update(prepare_transfer_spec(root,reference,declared,selected,resource_owner,chain=chain))
    result = binding(candidates, declared, selected, arguments, reference)
    owner.validate_binding(result)
    return result, dict(identity=identity, reference=reference)


def validate_completion(admitted, candidates, declared, selected, arguments):
    from .session_state import sha
    from .raw_handoff import declaration_dict
    receipt = admitted['epizoo_context']
    if (set(receipt) != {'identity', 'reference'} or receipt['identity'] != digest(dict(
            sources=candidates.attribution(), declaration=declaration_dict(declared), resource=selected.resource_choice()))
            or admitted['kind'] != 'execute' or admitted['operation'] != 'plan'):
        raise ValueError('EpiZoo handoff identity changed.')
    if declared.operation in ('posttraining','initialization'):
        expected={'manifest_path','manifest_sha256'}
        if declared.operation=='initialization': expected.update(('adaptation_spec_path','adaptation_spec_sha256'))
        if set(receipt['reference']) != expected: raise ValueError('Invalid reference binding.')
        sha(receipt['reference']['manifest_sha256'])
        if not Path(receipt['reference']['manifest_path']).is_absolute(): raise ValueError('Invalid reference locator.')
    elif receipt['reference'] is not None: raise ValueError('Inference cannot provision training references.')
    result = binding(candidates, declared, selected, arguments, receipt['reference'])
    if admitted['inputs'] != _serialize(result.execution_inputs) or admitted['registered_input'] != result.attribution():
        raise ValueError('Captured EpiZoo inputs changed.')


def prepare_transfer_spec(root, reference, declared, selected, resource_owner, *, chain=None):
    """Prepare task-local bindings from the one scientific resource library."""
    from dataclasses import asdict
    from agent.tools.models.epizoo_adaptation import contract as c, backend
    from agent.tools.data import regulatory_feature_reference as target_owner, scatac_reference as source_owner
    from agent.tools.data import liftover_chain
    entry=selected.inputs()['library_epizoo_transfer']
    library=resource_owner.library
    if library is None: raise ValueError('Qualified transfer resource library unavailable.')
    resource=library.check(entry)
    target_ref=target_owner.load_regulatory_feature_reference(reference['manifest_path'],expected_sha256=reference['manifest_sha256'])[1]
    mappings=[]
    if declared.strategy!='de_novo' and resource['liftover'] and resource['bedtools']:
        for pin in resource['source_references']:
            ref=source_owner.load_scatac_reference_bundle(pin['path'],expected_sha256=pin['sha256'])[1]
            if declared.reference_species is not None and ref.species!=declared.reference_species: continue
            destination=dict(scientific_name='Homo sapiens' if ref.species=='human' else 'Mus musculus',
                             taxonomy_id=9606 if ref.species=='human' else 10090)
            matches=[row for row in library.entries() if row['kind']=='liftover_chain'
                and row['resource']['input_species']==asdict(target_ref.species)
                and row['resource']['input_assembly']==target_ref.target_assembly
                and row['resource']['destination_species']==destination
                and row['resource']['destination_assembly']==ref.target_assembly]
            if len(matches)>1: raise ValueError('Multiple chain versions require an explicit compatible resource selection.')
            if chain is not None:
                qualified_chain=dict(input_species=asdict(target_ref.species),input_assembly=target_ref.target_assembly,
                    destination_species=destination,destination_assembly=ref.target_assembly,
                    input_fai=c.file_record(target_ref.genome.fai.path),destination_fai=c.file_record(ref.genome.fai.path),
                    chain=c.file_record(chain.source_path),source=chain.attribution,usage='Private Session-supplied chain; no shared promotion')
                try: liftover_chain.qualify(qualified_chain)
                except ValueError as exc:
                    raise EpiZooResourcePrerequisite('The supplied chain is incompatible with '+target_ref.target_assembly+' to '+ref.target_assembly+': '+str(exc)) from exc
            elif matches:
                qualified_chain=library.check(matches[0])
            else:
                acquisition=liftover_chain.acquire_ucsc(target_ref.target_assembly,ref.target_assembly,root.parent/'chains')
                if acquisition is None: continue
                qualified_chain=dict(input_species=asdict(target_ref.species),input_assembly=target_ref.target_assembly,
                    destination_species=destination,destination_assembly=ref.target_assembly,
                    input_fai=c.file_record(target_ref.genome.fai.path),destination_fai=c.file_record(ref.genome.fai.path),
                    chain=acquisition['file'],source=acquisition['url'],usage=acquisition['usage'])
                # Downloads remain private in the Session. Installation for
                # shared reuse is an explicit library action, never implicit.
                liftover_chain.qualify(qualified_chain)
                chain_record=root.parent/'chains'/(c.digest(qualified_chain)+'.qualified.json')
                if not chain_record.exists(): c.write_json(chain_record,qualified_chain)
            mappings.append(dict(source_species=ref.species,source_reference=pin,
                target_reference_identity=target_ref.reference_identity_sha256,chain=qualified_chain['chain'],
                liftover=resource['liftover'],bedtools=resource['bedtools'],min_overlap=1,
                direction='target-to-source',tool_versions=backend.mapping_tool_versions(
                    resource['liftover']['path'],resource['bedtools']['path'])))
    if declared.strategy=='mapped_reference' and len(mappings)!=1:
        raise EpiZooResourcePrerequisite('No compatible chain is available for explicit mapped-reference transfer from '+
            target_ref.target_assembly+' to '+('hg38' if declared.reference_species=='human' else 'mm10')+
            '. Supply the exact assembly-pair chain; the requested mapped strategy remains pending.')
    spec=dict(contract_version='epizoo-adaptation-inputs.v3',execution_profile='lifecycle.v3',
        reference_identity_sha256=target_ref.reference_identity_sha256,species=asdict(target_ref.species),
        assembly=target_ref.target_assembly,source_bundle=resource['source_bundle'],seam_bundle=resource['seam_bundle'],mapping=mappings)
    path=root/'initialization-spec.json'
    if path.exists():
        if c.read_json(path)!=spec: raise ValueError('Captured initialization resources changed.')
        pin=c.file_record(path)
    else: pin=c.write_json(path,spec)
    return dict(adaptation_spec_path=pin['path'],adaptation_spec_sha256=pin['sha256'])
