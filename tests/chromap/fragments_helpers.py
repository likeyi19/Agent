"""Generated M10 -> M11.1 -> M11.2 inputs for fragments tests."""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import zlib
import struct

from agent.tools.data import _chromap as c, _raw_fastq as f
from agent.tools.data import raw_scatac_manifest as m, scatac_library_context as lc
from agent.tools.data import chromap_reference_index as ci
from agent.tools.data._fragments_binding import FragmentInputs
from agent.tools.data._fragments_common import FragmentsRuntime

BC = 'ACGTACGTACGTACGT'


def backend():
    return c.BackendIdentity(c.QUALIFIED_EXECUTABLE_SHA256, 'x86_64', 'Linux', 'little', 64, c.policy_sha256())


def inputs_for(tiny, groups, white, *, shared=False, executable=None):
    intake = f.inspect_fastq_inputs([path for g in groups for _, path in g.files],
        species='human', assay=f.FastqAssay.TENX_ATAC, declared_layout=groups[0].layout)
    ip = m.publish_raw_intake_manifest(intake, tiny['root'] / 'intake.json')
    declarations = []
    memberships = [tuple(g.id for g in intake.groups)] if shared else [(g.id,) for g in intake.groups]
    for i, ids in enumerate(memberships):
        declarations.append(lc.LibraryDeclaration(namespace=f'library_{i}', group_ids=ids,
            membership_basis=lc.MembershipBasis.CALLER_SHARED_LIBRARY if shared else lc.MembershipBasis.SINGLE_GROUP,
            barcode_interpretation=lc.BarcodeInterpretation.RAW_SEQUENCE,
            correction_policy=lc.CorrectionPolicy.WHITELIST_REQUIRED, whitelist=white))
    context = lc.build_scatac_library_processing_context(intake_manifest_path=ip['manifest_path'],
        expected_intake_sha256=ip['manifest_sha256'], libraries=tuple(declarations),
        selection_mode=lc.SelectionMode.ALL_GROUPS)
    cp = lc.publish_scatac_library_processing_context(context, tiny['root'] / 'context.json')
    rp = tiny['pointer']; directory = tiny['root'] / 'index'
    if executable:
        ci.prepare_chromap_reference_index(reference_manifest_path=rp['manifest_path'],
            expected_reference_sha256=rp['manifest_sha256'], executable=executable,
            backend=backend(), output_dir=directory)
    else:
        directory.mkdir(); (directory / 'reference.chromap').write_bytes(b'opaque fake index')
        index = ci.ChromapReferenceIndex(**ci._binding(tiny['bundle'], rp['manifest_sha256'], backend()),
            index_sha256=c.sha256(directory / 'reference.chromap'), index_size_bytes=17)
        index = replace(index, index_identity_sha256=ci._identity(index))
        (directory / 'manifest.json').write_bytes(ci.canonical_chromap_reference_index_bytes(index))
    return FragmentInputs(ip['manifest_path'], ip['manifest_sha256'], cp['manifest_path'], cp['manifest_sha256'],
        rp['manifest_path'], rp['manifest_sha256'], str(directory / 'manifest.json'),
        c.sha256(directory / 'manifest.json'))


def bgzf_bytes(payload):
    """Tiny generated BGZF fixtures, including multiple blocks and EOF."""
    from agent.tools.data._fragment_io import BGZF_EOF
    result = bytearray()
    for offset in range(0, len(payload), 32000):
        block = payload[offset:offset+32000]
        encoder = zlib.compressobj(wbits=-15)
        packed = encoder.compress(block) + encoder.flush()
        header = bytes.fromhex('1f8b08040000000000ff060042430200') + struct.pack('<H', len(packed)+25)
        result.extend(header + packed + struct.pack('<II', zlib.crc32(block), len(block)))
    return bytes(result) + BGZF_EOF
