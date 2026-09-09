"""Artifact authority and source reconstruction shared by execution/verification.

No discovery rules here: reconstruction delegates to M10, and role selection
projects its reviewed evidence. No complete FASTQ scan occurs in this module.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from . import _chromap as c, _raw_fastq as f, raw_scatac_manifest as m
from . import scatac_library_context as lc, scatac_reference as ref
from . import chromap_reference_index as ci
from .raw_scatac import _reconstruct
from ._fragments_common import TBI_LIMIT, digest, fail, snapshots, unchanged


@dataclass(frozen=True)
class FragmentInputs:
    intake_path: str
    intake_sha256: str
    context_path: str
    context_sha256: str
    reference_path: str
    reference_sha256: str
    index_path: str
    index_sha256: str


def _facts(intake, gid=None, fact=None):
    result = []
    for e in intake.evidence:
        if gid is not None and e.group_id != gid:
            continue
        try:
            value = json.loads(e.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get('contract') == f.CONTRACT and value.get('fact') == fact:
            result.append((e, value))
    return result


def fresh_intake(intake):
    roots = sorted({x.selection_root if x.selection is m.SelectionBasis.DIRECTORY_DISCOVERY
                    else x.path for x in intake.files})
    species = {g.species.value for g in intake.groups}
    assays = {v['value'] for _, v in _facts(intake, fact='assay')}
    layouts = {v['value'] for _, v in _facts(intake, fact='layout-declaration')}
    if len(species) != 1 or len(assays) != 1 or len(layouts) > 1:
        fail('FRAGMENTS_INTAKE_MISMATCH')
    try:
        current = _reconstruct(roots, species=next(iter(species)), raw_assay=next(iter(assays)),
                               fastq_layout=next(iter(layouts)) if layouts else None)
        if m.canonical_manifest_bytes(current) != m.canonical_manifest_bytes(intake):
            fail('FRAGMENTS_SOURCE_CHANGED_BEFORE_EXECUTION')
    except (OSError, ValueError) as exc:
        if getattr(exc, 'code', '').startswith('FRAGMENTS_'):
            raise
        fail('FRAGMENTS_SOURCE_CHANGED_BEFORE_EXECUTION')


def preflight(inputs, backend, *, fresh=True):
    """Lightweight relationships first, then bounded/raw and resource reinspection."""
    if type(inputs) is not FragmentInputs:
        fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
    for key, value in asdict(inputs).items():
        if key.endswith('_path'):
            if (type(value) is not str or not Path(value).is_absolute()
                    or str(Path(value)) != value or '..' in Path(value).parts
                    or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
                fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
        else:
            try:
                c.digest_text(value)
            except c.ChromapError:
                fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
    try:
        _, intake, _ = m.load_raw_intake_manifest(inputs.intake_path, expected_sha256=inputs.intake_sha256)
    except (OSError, ValueError):
        fail('FRAGMENTS_INTAKE_MISMATCH')
    try:
        _, context, _ = lc.load_scatac_library_processing_context(inputs.context_path,
                                                        expected_sha256=inputs.context_sha256)
        if context.intake.manifest_sha256 != inputs.intake_sha256:
            fail('FRAGMENTS_CONTEXT_MISMATCH')
        lc.validate_library_context_intake_binding(context, intake_manifest_path=inputs.intake_path)
    except (OSError, ValueError):
        fail('FRAGMENTS_CONTEXT_MISMATCH')
    try:
        _, bundle, _ = ref.load_scatac_reference_bundle(inputs.reference_path,
                                                       expected_sha256=inputs.reference_sha256)
        index = ci.load_chromap_reference_index(inputs.index_path, expected_sha256=inputs.index_sha256)
        binding = ci._binding(bundle, inputs.reference_sha256, backend)
        if any(getattr(index, k) != v for k, v in binding.items()):
            fail('FRAGMENTS_REFERENCE_MISMATCH')
    except (OSError, ValueError):
        fail('FRAGMENTS_REFERENCE_MISMATCH')
    groups = {g.id: g for g in intake.groups}; files = {x.id: x for x in intake.files}
    libraries = []
    for library in context.libraries:
        if (library.input_kind is not m.InputKind.FASTQ
                or library.barcode_interpretation is not lc.BarcodeInterpretation.RAW_SEQUENCE
                or library.correction_policy is not lc.CorrectionPolicy.WHITELIST_REQUIRED):
            fail('FRAGMENTS_BARCODE_POLICY_UNRESOLVED')
        if library.whitelist is None:
            fail('FRAGMENTS_WHITELIST_MISMATCH')
        if not 1 <= library.whitelist.barcode_length <= 32:
            fail('FRAGMENTS_BARCODE_LENGTH_UNSUPPORTED')
        selected = []
        for source in library.groups:
            g = groups[source.group_id]
            if g.species.value != bundle.species or g.target_genome_assembly.value != bundle.target_assembly:
                fail('FRAGMENTS_SPECIES_MISMATCH')
            if g.structure is not m.StructureState.SUPPORTED:
                fail('FRAGMENTS_LAYOUT_UNSUPPORTED')
            layouts = {v['value'] for _, v in _facts(intake, g.id, 'layout')}
            if len(layouts) != 1:
                fail('FRAGMENTS_LAYOUT_UNSUPPORTED')
            layout = f.FastqLayout(next(iter(layouts)))
            if layout not in f.LAYOUT_ROLES:
                fail('FRAGMENTS_LAYOUT_UNSUPPORTED')
            roles = {}
            for e, value in _facts(intake, g.id, 'read-meaning'):
                role = f.ReadRole(value['role'])
                if (e.file_id not in g.file_ids or f.LAYOUT_ROLES[layout].get(role).value != value['meaning']
                        or role in roles):
                    fail('FRAGMENTS_ROLE_MISSING')
                roles[role] = files[e.file_id].path
            if set(roles) != set(f.LAYOUT_ROLES[layout]) and set(roles) != set(f.LAYOUT_ROLES[layout]) - {f.ReadRole.I1}:
                fail('FRAGMENTS_ROLE_MISSING')
            # M10 group:<sha256> -> qualification adapter's digest-only ID.
            # The full authoritative ID is retained in manifest/library bindings.
            selected.append(c.QualificationGroup(g.id.removeprefix('group:'), layout,
                tuple(sorted(roles.items(), key=lambda item: item[0].value))))
        if len({g.layout for g in selected}) != 1:
            fail('FRAGMENTS_LAYOUT_UNSUPPORTED')
        try:
            for group in selected:
                for role, path in group.files:
                    if role is not f.ReadRole.I1:
                        c.input_path(path)
            for path in (bundle.genome.fasta.path, library.whitelist.resource.path,
                         Path(inputs.index_path).parent / index.index_file):
                c.input_path(path)
            for meaning in (f.ReadMeaning.GENOMIC_1, f.ReadMeaning.GENOMIC_2, f.ReadMeaning.BARCODE):
                paths_for_role = [path for group in selected for role, path in group.files
                                  if f.LAYOUT_ROLES[group.layout][role] is meaning]
                if len(','.join(paths_for_role).encode()) > 120000:
                    fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
        except c.ChromapError:
            fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
        libraries.append((library, tuple(sorted(selected, key=lambda g: g.group_id))))
    paths = [x.path for x in intake.files] + [getattr(inputs, n) for n in
        ('intake_path', 'context_path', 'reference_path', 'index_path')]
    paths += [bundle.genome.fasta.path, bundle.genome.fai.path, bundle.ccre.bed.path,
              str(Path(inputs.index_path).parent / index.index_file)]
    if bundle.annotation:
        paths.append(bundle.annotation.resource.path)
    paths += [lib.whitelist.resource.path for lib, _ in libraries]
    try:
        before = snapshots(paths)
    except OSError:
        fail('FRAGMENTS_SOURCE_CHANGED_BEFORE_EXECUTION')
    if fresh:
        fresh_intake(intake)
        try:
            lc.reinspect_barcode_resources(context)
        except (OSError, ValueError):
            fail('FRAGMENTS_WHITELIST_MISMATCH')
        try:
            ref.reinspect_scatac_reference_bundle_sources(bundle)
        except (OSError, ValueError):
            fail('FRAGMENTS_REFERENCE_MISMATCH')
        try:
            ci.verify_chromap_reference_index(inputs.index_path, reference_manifest_path=inputs.reference_path,
                expected_reference_sha256=inputs.reference_sha256, backend=backend,
                expected_manifest_sha256=inputs.index_sha256)
        except (OSError, ValueError):
            fail('CHROMAP_INDEX_INVALID')
    contigs = ci.verify_fasta_fai(bundle)
    if any(length > TBI_LIMIT for _, length in contigs):
        fail('FRAGMENTS_INDEX_POLICY_UNSUPPORTED')
    unchanged(before)
    lineage = {'intake_contract': intake.intake_contract_version,
        'context_identity': context.context_identity_sha256, 'reference_identity': bundle.reference_identity_sha256,
        'species': bundle.species, 'assembly': bundle.target_assembly,
        'ordered_contig_sha256': bundle.genome.ordered_contig_sha256,
        'index_identity': index.index_identity_sha256, 'index_file_sha256': index.index_sha256,
        'index_policy': index.build_policy}
    return dict(intake=intake, context=context, bundle=bundle, index=index, libraries=libraries,
                contigs=contigs, snapshots=before, lineage=lineage)


def library_binding(library, groups):
    portable = asdict(library)
    portable['whitelist']['resource'].pop('path')
    portable['whitelist']['resource'].pop('provenance')
    return {'namespace': library.namespace, 'group_ids': ['group:' + g.group_id for g in groups],
        'library_identity': digest(portable),
        'role_binding_sha256': digest([asdict(g) for g in groups]),
        'whitelist_sha256': library.whitelist.resource.sha256,
        'whitelist_set_sha256': library.whitelist.barcode_set_sha256,
        'barcode_length': library.whitelist.barcode_length}
