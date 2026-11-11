"""Project captured raw files and existing reference resources into native ports.

This adapter selects no scientific functions and executes no scientific work.
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

from agent.schemas.orchestration import _serialize
from .session_state import digest

# Reviewed external names for one exact qualified FASTA/FAI/ordered-axis triple.
# This is not an hg38/GRCh38 synonym or a substitute for component qualification.
_REFERENCE_NAMES = {
    ('fb7421217e7058120cd60a5277445198e8deef2dff1edc46cd1e98b31fe64cbb',
     '24bf2065650006f4d8e08897dcae433d50addcd7f342e566bbc6fd9145f64798',
     'd525ee20551f34768f4017c7a779a3f3c7b947dacdea27838a5776508834b306'):
        ('refdata-cellranger-arc-GRCh38-2020-A', 'refdata-cellranger-arc-GRCh38-2020-A-2.0.0'),
}


def _mapping_inputs(receipt):
    """Every registered Chromap function consumes a FASTA; mapping also uses
    the explicitly declared barcode-correction input. No QC/BED bundle is needed.
    """
    values = {}
    for port in ('reference', 'whitelist'):
        component, _, _ = _component(receipt, port)
        values[port + '_path'] = component['path']
        values[port + '_sha256'] = component['sha256']
    return values


def _mapping_index(values):
    # Reuse a recorded exact FASTA/index association; native indexing remains
    # an ordinary planned operation when no unique index is installed.
    from agent.tools._file_identity import file_closure
    root = os.environ.get('AGENT_CHROMAP_INDEX_ROOT')
    candidates = []
    for number, manifest in enumerate(Path(root).glob('*/manifest.json') if root else ()):
        if number >= 128: break
        try:
            row = json.loads(manifest.read_bytes())
            if row.get('fasta_sha256') == values['reference_sha256'] and row.get('index_file') == 'reference.chromap':
                index = manifest.parent / row['index_file']
                if file_closure([index])[0]['sha256'] == row['index_sha256']:
                    candidates.append(dict(index_path=str(index),index_sha256=row['index_sha256']))
        except (ValueError, OSError, KeyError):
            continue
    return candidates[0] if len(candidates) == 1 else {}


def identity(candidates, declared, selected, *, adapter='native-files.v1'):
    return digest(dict(sources=candidates.attribution(), declaration={k:v for k,v in asdict(declared).items() if k != 'resource_preference' or v is not None},
        resource=None if selected is None else selected.resource_choice(), adapter=adapter))


def prepare(owner, session_id, candidates, declared, selected, arguments, *, resource_evidence=None):
    if resource_evidence is None:
        raise ValueError('New native tasks require captured function-input resources')
    return prepare_inputs(owner, session_id, candidates, declared, selected, arguments, resource_evidence)


def validate(admitted, candidates, declared, selected, arguments, field):
    """Pure replay of captured identity; no native computation or source opening."""
    from .local_resources import LocalResourceAdmission, RegisteredInput, RegisteredInputCollection
    from .raw_handoff import FastqDeclaration
    receipt = admitted[field + '_context']
    if receipt.get('adapter') == 'native-inputs.v1':
        return validate_inputs(admitted, candidates, declared, selected, arguments, field)
    token = identity(candidates, declared, selected)
    if (set(receipt) != {'adapter', 'identity', 'resources', 'feature_reference'} or receipt['adapter'] != 'native-files.v1'
            or receipt['identity'] != token or admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'):
        raise ValueError('Native handoff identity changed')
    bound = LocalResourceAdmission._submission_binding(dict(execution_inputs=admitted['inputs'], registered_input=admitted['registered_input']))
    resources = receipt['resources']
    if not isinstance(resources, dict): raise ValueError('Invalid native resource projection')
    if _serialize(bound.resource_choice) != (None if selected is None else selected.resource_choice()):
        raise ValueError('Selected native resource changed')
    values = dict(resources, **arguments)
    if isinstance(declared, FastqDeclaration):
        if type(bound) is not RegisteredInputCollection: raise ValueError('Native source collection changed')
        if sorted(m['resource_id'] for m in bound.members) != sorted(LocalResourceAdmission.registration_id(
                'native-fastq:' + token + ':' + m.source) for m in declared.members):
            raise ValueError('Native FASTQ source membership changed')
        roles = {'read1_path': 'R1', 'read2_path': 'R3' if declared.layout == 'tenx-atac-r1-r2-r3.v1' else 'R2',
                 'barcode_path': 'R2' if declared.layout == 'tenx-atac-r1-r2-r3.v1' else 'I2'}
        for key, role in roles.items():
            values[key] = [candidates.members[int(m.source.split('.')[1])]['source_path']
                for m in sorted(declared.members, key=lambda m:(m.lane,m.chunk,m.role)) if m.role == role]
    else:
        original = candidates.members[int(declared.source.split('.')[1])]
        if type(bound) is not RegisteredInput or bound.resource_id != LocalResourceAdmission.registration_id('native-source:' + token):
            raise ValueError('Native raw source changed')
        values.update(input_path=original['source_path'], input_sha256=original['source_sha256'])
    if bound.composition != 'native-science.v1' or values != admitted['inputs']:
        raise ValueError('Native source/reference/parameter binding changed')


def historical_binding(owner, records, binding):
    """Read captured legacy source associations; never authorize new consumption.

    Session replay separately checks exact typed intent and accepted execution.
    There are no legacy input builders, workflow guards or scientific file reads.
    """
    from .local_resources import ResourceAdmissionError
    values = _serialize(binding.execution_inputs)
    def invalid(): raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
    if binding.composition is None and binding.tool_name == 'prepare_scATAC_fragments':
        spec=owner._group_registry().get(binding.tool_name)
        known=dict(spec.required_arguments) | dict(spec.optional_arguments)
        if set(values)-set(known) or set(spec.required_arguments)-{'output_dir'}-set(values): invalid()
        for name,value in values.items():
            try: known[name].validate(name,value)
            except (TypeError,ValueError): invalid()
        return values
    if binding.composition == 'fastq-science.v1':
        source = dict(raw_input_paths=owner._fastq_sources(records),raw_assay='TENX_ATAC',
            fastq_layout=records[0].fastq_member['fastq_layout'])
    else:
        if len(records) != 1: invalid()
        record=records[0]
        source = dict(source_path=record.source_path,source_sha256=record.source_sha256)
        if binding.composition == 'bam-science.v1':
            source['raw_input_paths'] = ([record.source_path] if record.source_path.endswith('.bam')
                else [dict(path=record.source_path,input_kind='BAM')])
        if record.source_index is not None:
            source.update(source_index_path=record.source_index['path'],source_index_sha256=record.source_index['sha256'])
    if any(values.get(k) != v for k,v in source.items()): invalid()
    return values


# Native task resource projection. This is a view of ScientificResourceLibrary
# and captured InputSets, not another catalog or scientific qualification pass.
def _species(value):
    from agent.tools.data.scatac_reference import species_identity
    return species_identity(value) if value in ('human', 'mouse') else value


def resource_evidence(input_sets, *, field=None):
    from agent.tools.data._fragment_reference import load
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    result = {}
    for choice in input_sets:
        values = choice.inputs()
        components = {}
        species = assembly = None
        parent = None
        try:
            if 'library_genome' in values:
                raw = values['library_genome']['resource']
                species, assembly = raw['species'], raw['assembly']
                components['reference'] = dict(raw['genome']['fasta'])
                components['chrom_sizes'] = dict(raw['genome']['fai'], ordered_sizes=raw['genome']['ordered_contig_sha256'], fasta_sha256=raw['genome']['fasta']['sha256'],
                    contig_count=raw['genome']['contig_count'])
            elif 'qualified_genome' in values:
                qualified = values['qualified_genome']['reference_qualification']
                raw = dict(**qualified['requirement'], genome=qualified['genome'])
                species, assembly = raw['species'], raw['assembly']
                components['reference'] = dict(raw['genome']['fasta'])
                components['chrom_sizes'] = dict(raw['genome']['fai'], ordered_sizes=raw['genome']['ordered_contig_sha256'], fasta_sha256=raw['genome']['fasta']['sha256'],
                    contig_count=raw['genome']['contig_count'])
            if values.get('qc_reference_manifest_path'):
                _, qc, _ = load_scatac_qc_reference_bundle(values['qc_reference_manifest_path'],
                    expected_sha256=values['qc_reference_manifest_sha256'])
                from .shared_resources import _qc_parent
                parent = _qc_parent(qc)
                if values.get('reference_bundle_sha256', qc.parent_manifest.sha256) != qc.parent_manifest.sha256:
                    raise ValueError('Reference grouping differs from its exact QC parent')
                components['gene_anno'] = asdict(qc.annotation.resource)
                path, sha = qc.parent_manifest.path, qc.parent_manifest.sha256
            else:
                path, sha = values.get('reference_bundle_path'), values.get('reference_bundle_sha256')
            if path:
                ref = parent if parent is not None else load(path, sha)
                species, assembly = ref.species, ref.target_assembly
                components.update(reference=asdict(ref.genome.fasta), chrom_sizes=dict(asdict(ref.genome.fai), ordered_sizes=ref.genome.ordered_contig_sha256, fasta_sha256=ref.genome.fasta.sha256,
                    contig_count=ref.genome.contig_count), peaks=asdict(ref.ccre.bed))
            if choice.fastq_whitelist is not None:
                components['whitelist'] = dict(choice.fastq_whitelist['resource'])
            for port in ('reference', 'chrom_sizes', 'gene_anno', 'peaks', 'whitelist'):
                if values.get(port + '_path') and values.get(port + '_sha256'):
                    species, assembly = values.get('species', species), values.get('assembly', assembly)
                    components[port] = dict(path=values[port + '_path'], sha256=values[port + '_sha256'])
            if components:
                # Qualified component provenance supplies exact external names.
                # An assembly string or approximate label never supplies aliases.
                genome = components.get('reference', {})
                sizes = components.get('chrom_sizes', {})
                aliases = _REFERENCE_NAMES.get((genome.get('sha256'), sizes.get('sha256'), sizes.get('ordered_sizes')), ())
                result[choice.input_set_id] = dict(species=_species(species), assembly=assembly,
                    components={k:{name:v[name] for name in ('path','sha256','ordered_sizes','contig_count','fasta_sha256') if name in v} for k,v in components.items()},
                    component_identity_version=2, aliases=sorted(aliases), label=choice.label, default=(getattr(choice, field + '_default') if field is not None else choice.fragments_default or choice.bam_default))
        except (ValueError, OSError, TypeError, KeyError):
            # Do not hide independently available references. Explicit selection
            # of this invalid entry remains a visible missing-resource question.
            result[choice.input_set_id] = None
    return result


def configured_inputs(choice, tool):
    """Project an explicitly selected Web InputSet into actual native ports.

    Ordinary automatic selection belongs to task/function binding. This advanced
    structured route captures file offers; it never preselects a default bundle.
    """
    if choice is None: return {}
    from agent.tools.analysis.snapatac_raw import PORTS
    from agent.tools.data.chromap_native import PORTS as CHROMAP_PORTS
    row = resource_evidence((choice,)).get(choice.input_set_id)
    values = {}
    if row is not None:
        for port,component in row['components'].items():
            if tool == 'run_snapatac2' and port in PORTS:
                path,sha,_ = PORTS[port]
                values.update({path:component['path'],sha:component['sha256']})
            elif tool == 'run_chromap' and port in ('reference','whitelist'):
                values[CHROMAP_PORTS[port]] = component['path']
                values[port + '_sha256'] = component['sha256']
    # Explicit native configuration remains an ordinary Registry argument.
    values.update({key:choice.inputs()[key] for key in ('function_name','kwargs') if key in choice.inputs()})
    return values


def requirements(binding, declared, choices, evidence):
    from .raw_handoff import source_handles
    from .resource_continuation import match_preference, preference_matches
    if any(int(h.split('.')[1]) >= len(binding.members) for h in source_handles(declared)):
        raise ValueError('Native source role is outside captured attachments')
    from .raw_handoff import FastqDeclaration
    missing = []
    if isinstance(declared, FastqDeclaration):
        missing = _fastq_requirements(binding, declared)
    elif declared.source is None:
        missing = ['Identify the raw input attachment.']
    context = dict(species=declared.species,assembly=getattr(declared,'assembly',None), catalog=evidence)
    aliases = lambda choice: (evidence.get(choice.input_set_id) or {}).get('aliases', ())
    matches = preference_matches(declared.resource_preference, choices, aliases=aliases)
    if len(matches) > 1:
        # A user-facing genome name may identify a genome entry and a QC
        # grouping of those same bytes. Resolve their actual function ports
        # later rather than asking the user to distinguish configuration aliases.
        selected = None
        questions = []
        if any(evidence.get(c.input_set_id) is None or not _context_matches(evidence[c.input_set_id],context) for c in matches):
            questions = ['The requested reference name identifies unavailable or conflicting references. Please identify the intended reference.']
    else:
        questions, selected = match_preference(declared.resource_preference, choices, aliases=aliases)
    if declared.resource is not None:
        questions = []  # An explicit offered selection supersedes textual name lookup.
        selected = next((s for s in choices if 'resource.' + s.input_set_id == declared.resource), None)
        if selected is None: questions = ['The selected reference is unavailable in this Session. Please identify the reference you want.']
    if selected is not None:
        raw = evidence.get(selected.input_set_id)
        if raw is None:
            questions = [f'{selected.label} has no available native reference inputs. Please supply the requested component or select another reference.']
        elif not _context_matches(raw,context):
            questions = [f'{selected.label} conflicts with the declared source species or assembly. Please correct the declaration or explicitly choose another reference.']
    if isinstance(declared, FastqDeclaration) and not missing and not questions:
        try:
            _mapping_inputs(_receipt(declared, evidence, selected))
        except ResourcePrerequisite as exc:
            questions = list(exc.questions)
    return missing + questions, selected


def _fastq_requirements(binding, declared):
    from types import SimpleNamespace
    from .local_resources import LocalResourceAdmission, ResourceAdmissionError, _fastq_member
    missing = [message for key,message in (
        ('layout','Specify the FASTQ read layout.'),
        ('namespace','Identify the one library whose selected lanes/chunks belong together.'),
        ('barcode_policy','Identify whether the supplied barcode reads require whitelist correction.'))
        if getattr(declared,key) is None]
    if not declared.members:
        missing.append('Associate the uploaded reads with their roles, lanes and chunks.')
    if declared.layout and declared.namespace and declared.members:
        projected = [SimpleNamespace(source_path='/declared/' + m.source,
            fastq_member=_fastq_member(dict(library_id=declared.namespace,fastq_layout=declared.layout,
                role=m.role,lane=m.lane,chunk=m.chunk,compression='plain'))) for m in declared.members]
        try:
            LocalResourceAdmission._fastq_groups(projected)
        except ResourceAdmissionError as exc:
            if exc.code != 'FASTQ_COLLECTION_INCOMPLETE': raise
            missing.append(exc.message)
    return missing


def _receipt(declared, evidence, selected):
    return dict(catalog=evidence,selected=_preferred_ids(declared,evidence,selected),custom={},
        species=declared.species,assembly=getattr(declared,'assembly',None))


def _source_inputs(declared, arguments):
    from .raw_handoff import BamDeclaration
    from agent.orchestration.native_parameters import input_name
    values = {input_name('run_snapatac2','pp.import_fragments',key):getattr(declared,key)
        for key in ('species','assembly') if getattr(declared,key,None) is not None}
    if not isinstance(declared,BamDeclaration):
        values[input_name('run_snapatac2','pp.import_fragments','fragment_order')] = 'unknown'
    return values | arguments


def _preferred_ids(declared, evidence, selected):
    if selected is not None:
        return selected.input_set_id
    from .resource_continuation import preference_matches
    matches = preference_matches(declared.resource_preference,
        [(key,row) for key,row in evidence.items() if row is not None],
        label=lambda pair:pair[1]['label'], identity=lambda pair:'resource.' + pair[0],
        aliases=lambda pair:pair[1].get('aliases', ()))
    return sorted(key for key,_ in matches) or None


def prepare_inputs(owner, session_id, candidates, declared, selected, arguments, evidence):
    from .raw_handoff import FragmentsDeclaration, FastqDeclaration, source_handles
    token = identity(candidates, declared, selected, adapter='native-inputs.v1')
    records = {h:owner.load(candidates.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)}
    receipt = dict(adapter='native-inputs.v1',identity=token,**_receipt(declared,evidence,selected))
    inputs = _source_inputs(declared, arguments)
    if isinstance(declared,FastqDeclaration):
        resources = _mapping_inputs(receipt)
        resources.update(_mapping_index(resources))
        if any(k in arguments and arguments[k] != v for k,v in resources.items()):
            raise ValueError('Scientific parameters cannot replace selected native reference files')
        inputs.update(resources)
        typed = [owner.bind_raw_source('native-fastq:' + token + ':' + m.source, records[m.source],
            session_id=session_id,input_type='fastq',declaration=dict(library_id=declared.namespace,
                fastq_layout=declared.layout,role=m.role,lane=m.lane,chunk=m.chunk)) for m in declared.members]
        record = owner.register_fastq_collection('native-fastq:' + token,[r.resource_id for r in typed],
            label='Declared native FASTQ library',attribution='Explicit source roles and library membership')
    else:
        record = owner.bind_raw_source('native-source:' + token,records[declared.source],session_id=session_id,
            input_type='external_fragments' if isinstance(declared,FragmentsDeclaration) else 'bam')
    for attr,port in (('feature_source','peaks'),('genome_index_source','chrom_sizes'),('genome_source','genome')):
        handle = getattr(declared,attr,None)
        if handle:
            supplied = records[handle]
            if supplied.session_id not in (None,session_id): raise ValueError('Private native resource is outside this Session')
            receipt['custom'][port] = dict(path=supplied.source_path,sha256=supplied.source_sha256)
    receipt['resources'] = inputs
    binding = owner.compose_native(record.resource_id,inputs,resource_choice=None if selected is None else selected.resource_choice())
    return binding, receipt


def validate_inputs(admitted, candidates, declared, selected, arguments, field):
    from .local_resources import LocalResourceAdmission
    from .raw_handoff import FastqDeclaration
    receipt = admitted[field + '_context']
    if (set(receipt) != {'adapter','identity','resources','catalog','selected','custom','species','assembly'}
            or receipt['identity'] != identity(candidates,declared,selected,adapter='native-inputs.v1')
            or receipt['catalog'] != admitted['raw_resource_evidence']
            or receipt['selected'] != _preferred_ids(declared,receipt['catalog'],selected)
            or any(receipt[k] != getattr(declared,k,None) for k in ('species','assembly'))):
        raise ValueError('Captured native resource inputs changed')
    values = _source_inputs(declared,arguments)
    if isinstance(declared,FastqDeclaration):
        values.update(_mapping_inputs(receipt))
        # Index discovery is a captured locator, not repeated during replay.
        for key in ('index_path','index_sha256'):
            if key in receipt['resources']: values[key] = receipt['resources'][key]
    if values != receipt['resources']: raise ValueError('Native input declarations changed')
    custom = {}
    for attr,port in (('feature_source','peaks'),('genome_index_source','chrom_sizes'),('genome_source','genome')):
        handle = getattr(declared,attr,None)
        if handle:
            member = candidates.members[int(handle.split('.')[1])]
            custom[port] = dict(path=member['source_path'],sha256=member['source_sha256'])
    if custom != receipt['custom']: raise ValueError('Native custom resource changed')
    if 'native_plan' in admitted:
        from agent.schemas.run_state import _decode_plan
        plan = _decode_plan(admitted['native_plan'])
        bindings = []
        if bind_plan(plan,receipt,bindings=bindings) != plan or bindings != admitted['native_resource_bindings']:
            raise ValueError('Captured native function inputs changed')
    if 'native_resource_plan' in admitted:
        from agent.schemas.run_state import _decode_plan
        try:
            bind_plan(_decode_plan(admitted['native_resource_plan']), receipt)
        except ResourcePrerequisite:
            # Saved prose is immutable presentation, never historical admission
            # authority. Replay verifies the actual missing function input.
            if (not isinstance(admitted['choices'], list) or not admitted['choices']
                    or any(type(text) is not str or not text.strip() for text in admitted['choices'])):
                raise ValueError('Invalid saved native resource question')
        else:
            raise ValueError('Captured native resource question has no missing input')
    bound = LocalResourceAdmission._submission_binding(dict(execution_inputs=admitted['inputs'],registered_input=admitted['registered_input']))
    if isinstance(declared,FastqDeclaration):
        if sorted(m['resource_id'] for m in bound.members) != sorted(LocalResourceAdmission.registration_id(
                'native-fastq:' + receipt['identity'] + ':' + m.source) for m in declared.members):
            raise ValueError('Native FASTQ source membership changed')
        roles = {'read1_path':'R1','read2_path':'R3' if declared.layout == 'tenx-atac-r1-r2-r3.v1' else 'R2',
            'barcode_path':'R2' if declared.layout == 'tenx-atac-r1-r2-r3.v1' else 'I2'}
        for key,role in roles.items():
            values[key] = [candidates.members[int(m.source.split('.')[1])]['source_path']
                for m in sorted(declared.members,key=lambda m:(m.lane,m.chunk,m.role)) if m.role == role]
    else:
        source = candidates.members[int(declared.source.split('.')[1])]
        values.update(input_path=source['source_path'],input_sha256=source['source_sha256'])
        if bound.resource_id != LocalResourceAdmission.registration_id('native-source:' + receipt['identity']):
            raise ValueError('Native raw source changed')
    if (admitted['kind'] != 'execute' or admitted.get('operation') != 'plan'
            or bound.composition != 'native-science.v1'
            or _serialize(bound.resource_choice) != (None if selected is None else selected.resource_choice())
            or values != admitted['inputs']):
        raise ValueError('Native task binding changed')


class ResourcePrerequisite(ValueError):
    def __init__(self, questions, options=()):
        self.questions = tuple(questions)
        self.options = tuple(options)
        super().__init__(' '.join(questions))


def public_inputs(receipt):
    """Explain automatic inputs without delegating file identity selection."""
    from agent.tools.analysis.snapatac_raw import APPLIES
    return dict(binding='Agent resolves exact reference inputs after native function selection',
        species=receipt['species'],assembly=receipt['assembly'],
        available=sorted({port for row in receipt['catalog'].values() if row is not None
            and _context_matches(row,receipt) for port in row['components']} | (set(receipt['custom']) - {'genome'})),
        functions={fn:list(ports) for fn,ports in APPLIES.items()})


def _context_matches(row, receipt):
    # Unknown declarations are not biological mismatches; known declarations
    # remain authoritative. Exact component identity owns alias equivalence.
    for key in ('species','assembly'):
        declared = receipt.get(key)
        if declared is None: continue
        expected = _species(declared) if key == 'species' else declared
        actual = row.get(key)
        if key == 'species' and isinstance(actual,dict) and isinstance(expected,str):
            actual = actual.get('scientific_name')
        if actual != expected:
            if key != 'assembly' or row.get('component_identity_version', 1) < 2:
                return False
            def genome_identity(candidate):
                components = candidate.get('components', {})
                fasta, sizes = components.get('reference', {}), components.get('chrom_sizes', {})
                return (fasta.get('sha256'), sizes.get('sha256'), sizes.get('ordered_sizes'))
            identity = genome_identity(row)
            if (not all(identity) or not any(other is not None and other.get('component_identity_version', 1) >= 2 and other.get('assembly') == expected
                    and other.get('species') == row.get('species') and genome_identity(other) == identity
                    for other in receipt.get('catalog', {}).values())):
                return False
    return True


def _component(receipt, port, description='The exact reference file for this native input.'):
    from .resource_continuation import distinct_choices
    requested = receipt['selected']
    selected = {requested} if isinstance(requested,str) else set(requested or ())
    candidates = [(key,row) for key,row in receipt['catalog'].items() if row is not None
        and _context_matches(row,receipt) and port in row['components']]
    preferred = [(k,r) for k,r in candidates if k in selected]
    if preferred: candidates = preferred
    reason = 'explicit' if preferred else 'automatic'
    mapped_fasta = receipt.get('resources',{}).get('reference_sha256')
    if port == 'chrom_sizes' and mapped_fasta is not None:
        candidates = [(k,r) for k,r in candidates if r['components'][port].get('fasta_sha256') == mapped_fasta]
    if port == 'chrom_sizes' and 'genome' in receipt['custom']:
        candidates = [(k,r) for k,r in candidates if r['components'][port].get('fasta_sha256') == receipt['custom']['genome']['sha256']]
    if preferred and not candidates and port not in receipt['custom']:
        raise ResourcePrerequisite(['The requested chromosome-size reference has no recorded association with the supplied or mapped genome. Please correct the reference choice or genome input.'])
    chosen = None
    if port in receipt['custom']:
        component = receipt['custom'][port]
        reason = 'supplied_file'
    else:
        recorded_sizes = {row['components'][port]['sha256']:row['components'][port]['ordered_sizes']
            for _,row in candidates if 'ordered_sizes' in row['components'][port]}
        options = distinct_choices(candidates, default=lambda pair:pair[1]['default'],
            identity=lambda pair:digest([pair[1]['species'],
                receipt['assembly'] if receipt.get('assembly') is not None else pair[1]['assembly'],
                recorded_sizes.get(pair[1]['components'][port]['sha256'],pair[1]['components'][port]['sha256'])]),
            order=lambda pair:pair[0])
        if len(options) != 1:
            label = {'chrom_sizes':'chromosome-size','gene_anno':'gene annotation','peaks':'region feature','reference':'genome FASTA','whitelist':'barcode whitelist'}[port]
            context = ' '.join(str(receipt[key]) for key in ('species', 'assembly') if receipt.get(key) is not None) or 'the supplied source'
            if not options:
                raise ResourcePrerequisite([f'This operation requires {description} No applicable {label} resource is available for {context}.'])
            descriptions = [row['label'] + (f" ({row['components'][port]['contig_count']} contigs; names, lengths or ordering differ)"
                if 'contig_count' in row['components'][port] else ' (distinct reference component)') for _,row in options]
            raise ResourcePrerequisite([f'There are distinct {label} references for {context}. Please identify the reference matching the source data or the requested feature set: ' + ' / '.join(descriptions)],
                options=[dict(handle='resource.' + key, label=row['label'], description=description)
                         for (key, row), description in zip(options, descriptions)])
        chosen,row = options[0]
        component = row['components'][port]
        if reason != 'explicit' and sum(bool(r['default']) for _,r in candidates) == 1: reason = 'configured_default'
    return component, chosen, reason


def bind_plan(plan, receipt, *, bindings=None):
    """Bind only selected native functions, before execution or publication.

    Equivalence is exact component bytes plus declared context, never assembly
    alone. A full bundle is merely one source of components. No files are opened.
    """
    from agent.tools.analysis.snapatac_raw import APPLIES, PORTS
    steps = []
    for step in plan.steps:
        args = dict(step.arguments)
        if step.tool_name != 'run_snapatac2':
            steps.append(step); continue
        function = args.get('function_name')
        for port in APPLIES.get(function, ()):
            if port == 'whitelist': continue  # Optional native input; never auto cell calling.
            path_key,sha_key,_ = PORTS[port]
            # An explicit native dictionary is already the function's input.
            if port in args.get('kwargs',{}):
                requested = receipt['selected']
                ids = {requested} if isinstance(requested,str) else set(requested or ())
                if port in receipt['custom'] or any(key in ids and row is not None and port in row['components']
                        for key,row in receipt['catalog'].items()):
                    raise ResourcePrerequisite(['Both a native value and an explicit reference supply the same input. Please choose which source should be used.'])
                continue
            component,chosen,reason = _component(receipt,port,PORTS[port][2])
            if bindings is not None:
                bindings.append(dict(step_id=step.step_id,tool=step.tool_name,function=function,port=port,
                    resource=chosen,reason=reason,path=component['path'],sha256=component['sha256']))
            if path_key in args and (args[path_key] != component['path'] or args.get(sha_key) != component['sha256']):
                raise ValueError('Plan cannot substitute a captured native reference input')
            args[path_key],args[sha_key] = component['path'],component['sha256']
        steps.append(replace(step,arguments=args))
    return replace(plan, steps=tuple(steps))
