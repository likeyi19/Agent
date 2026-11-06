"""Project captured raw files and existing reference resources into native ports.

This adapter selects no scientific functions and executes no scientific work.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path

from agent.schemas.orchestration import _serialize
from .session_state import digest


def reference_inputs(selected, declared, *, reference=None):
    from .raw_handoff import BamDeclaration
    from agent.orchestration.native_parameters import input_name
    from agent.tools.data._fragment_reference import load
    from agent.tools.data.scatac_qc_reference import load_scatac_qc_reference_bundle
    base = {} if selected is None else selected.inputs()
    path = base.get('reference_bundle_path') if reference is None else reference['manifest_path']
    sha = base.get('reference_bundle_sha256') if reference is None else reference['manifest_sha256']
    ref = load(path, sha)
    values = dict(chrom_sizes_path=ref.genome.fai.path, chrom_sizes_sha256=ref.genome.fai.sha256,
        peaks_path=ref.ccre.bed.path, peaks_sha256=ref.ccre.bed.sha256)
    if base.get('qc_reference_manifest_path'):
        _, qc, _ = load_scatac_qc_reference_bundle(base['qc_reference_manifest_path'],
            expected_sha256=base['qc_reference_manifest_sha256'])
        values.update(gene_anno_path=qc.annotation.resource.path, gene_anno_sha256=qc.annotation.resource.sha256)
    if getattr(declared, 'members', None):
        values = dict(reference_path=ref.genome.fasta.path)
        whitelist = _serialize(selected.fastq_whitelist)
        values['whitelist_path'] = whitelist['resource']['path']
        # Reuse only an exact existing reference-index association. No index is
        # built during interpretation; otherwise the Planner may select index.
        index_root = os.environ.get('AGENT_CHROMAP_INDEX_ROOT')
        candidates = []
        if index_root:
            for number, manifest in enumerate(Path(index_root).glob('*/manifest.json')):
                if number >= 128: break
                try:
                    value = json.loads(manifest.read_bytes())
                    if value.get('fasta_sha256') == ref.genome.fasta.sha256 and value.get('index_file') == 'reference.chromap':
                        from agent.tools._file_identity import file_closure
                        index = manifest.parent / value['index_file']
                        if file_closure([index])[0]['sha256'] == value['index_sha256']: candidates.append(str(index))
                except (ValueError, OSError, KeyError):
                    continue
        if len(candidates) == 1: values['index_path'] = candidates[0]
    else:
        values.update(species=declared.species, assembly=declared.assembly)
        from agent.tools.analysis.snapatac_raw import APPLIES, PORTS
        # Bind reference files to their actual native consumer. In particular,
        # an available GTF must not become a prerequisite for import/matrix-only tasks.
        scoped = {}
        for function, ports in APPLIES.items():
            for port in ports:
                for key in PORTS[port][:2]:
                    if key in values:
                        scoped[input_name('run_snapatac2', function, key)] = values[key]
        scoped.update({input_name('run_snapatac2', 'pp.import_fragments', key): values[key]
            for key in ('species', 'assembly')})
        values = scoped
    if not isinstance(declared, BamDeclaration):
        # External fragments and Chromap BED have no barcode-order guarantee.
        # This task-only source knowledge is not a user kwargs declaration.
        values[input_name('run_snapatac2', 'pp.import_fragments', 'fragment_order')] = 'unknown'
    return values


def identity(candidates, declared, selected):
    return digest(dict(sources=candidates.attribution(), declaration=asdict(declared),
        resource=None if selected is None else selected.resource_choice(), adapter='native-files.v1'))


def prepare(owner, session_id, candidates, declared, selected, arguments):
    from .raw_handoff import FastqDeclaration, FragmentsDeclaration, source_handles
    token = identity(candidates, declared, selected)
    records = {h: owner.load(candidates.members[int(h.split('.')[1])]['resource_id']) for h in source_handles(declared)}
    reference = None
    if isinstance(declared, FragmentsDeclaration) and declared.feature_source is not None:
        from .reference_supplementation import qualify_bed
        from agent.tools.data.scatac_reference import species_identity
        directory = owner._workspace._ensure_directory(owner._workspace.root / 'fragments_handoffs' / token)
        base = {} if selected is None else selected.inputs()
        fasta = None if declared.genome_source is None else owner.bind_raw_source('native-genome:' + token,
            records[declared.genome_source], session_id=session_id, input_type='fasta', source_index=records.get(declared.genome_index_source))
        reference = qualify_bed(owner, records[declared.feature_source], parent_path=base.get('reference_bundle_path'),
            parent_sha256=base.get('reference_bundle_sha256'), category=declared.feature_category, directory=directory,
            fasta=fasta, species=species_identity(declared.species), assembly=declared.assembly)
    resources = reference_inputs(selected, declared, reference=reference)
    from agent.orchestration.native_parameters import input_name
    order = input_name('run_snapatac2', 'pp.import_fragments', 'fragment_order')
    if order in arguments:
        # Grounded user source knowledge prevails over the task's unknown order.
        # Reference-file identities below remain protected.
        resources.pop(order, None)
    if set(resources).intersection(arguments) and any(resources[k] != arguments[k] for k in set(resources).intersection(arguments)):
        raise ValueError('Scientific parameters cannot replace selected native reference files')
    values = resources | arguments
    if isinstance(declared, FastqDeclaration):
        typed = []
        for member in declared.members:
            typed.append(owner.bind_raw_source('native-fastq:' + token + ':' + member.source,
                records[member.source], session_id=session_id, input_type='fastq', declaration=dict(
                    library_id=declared.namespace, fastq_layout=declared.layout, role=member.role, lane=member.lane, chunk=member.chunk)))
        record = owner.register_fastq_collection('native-fastq:' + token, [r.resource_id for r in typed],
            label='Declared native FASTQ library', attribution='Explicit source roles and library membership')
    else:
        record = owner.bind_raw_source('native-source:' + token, records[declared.source], session_id=session_id,
            input_type='external_fragments' if isinstance(declared, FragmentsDeclaration) else 'bam')
    owner._verify_records(list(records.values()))
    binding = owner.compose_native(record.resource_id, values, resource_choice=None if selected is None else selected.resource_choice())
    return binding, dict(adapter='native-files.v1', identity=token, resources=resources, feature_reference=reference)


def validate(admitted, candidates, declared, selected, arguments, field):
    """Pure replay of captured identity; no native computation or source opening."""
    from .local_resources import LocalResourceAdmission, RegisteredInput, RegisteredInputCollection
    from .raw_handoff import FastqDeclaration
    receipt = admitted[field + '_context']
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
        spec=owner._group_registry().accepted(binding.tool_name)
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
