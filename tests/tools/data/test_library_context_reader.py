"""Retained accepted-fragment context integrity without raw-intake implementation."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from agent.tools.data import scatac_library_context as context
from agent.tools.data.scatac_reference import ResourceIdentity, SourceProvenance


def historical_context():
    whitelist=context.BarcodeWhitelistIdentity(ResourceIdentity('/original/whitelist', 'a'*64,
        SourceProvenance('unknown')),2,16,'b'*64)
    library=context.ProcessingLibrary('library',(context.GroupBarcodeBinding('group:'+'c'*64,
        'fastq_read','R2'),),'fastq',None,context.MembershipBasis.SINGLE_GROUP,
        context.BarcodeInterpretation.RAW_SEQUENCE,context.CorrectionPolicy.WHITELIST_REQUIRED,whitelist)
    value=context.LibraryProcessingContext(context.IntakeBinding('/original/intake.json','d'*64),
        (library,),context.SelectionMode.ALL_GROUPS,'0'*64)
    return replace(value,context_identity_sha256=context._context_digest(value))


def test_existing_context_reader_retains_digest_and_nested_whitelist(tmp_path):
    value=historical_context();path=tmp_path/'context.json';path.write_text(json.dumps(asdict(value)))
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    _,loaded,pinned=context.load_scatac_library_processing_context(path,expected_sha256=checksum)
    assert loaded==value and pinned==checksum
    assert loaded.libraries[0].whitelist.resource.path=='/original/whitelist'
    # Loading context never opens the original intake, reads, or whitelist.
    assert not __import__('pathlib').Path('/original').exists()
    path.write_text(path.read_text().replace('library','changed'))
    with pytest.raises(context.LibraryContextError):
        context.load_scatac_library_processing_context(path,expected_sha256=checksum)


@pytest.mark.parametrize('change',[dict(input_kind='rna'),dict(input_kind='bam'),
    dict(groups=[dict(group_id='group:'+'c'*64,source='unknown',locator='R2')]),
    dict(groups=[dict(group_id='group:'+'c'*64,source='bam_cell_identifier',locator='CB')]),
    dict(whitelist=None)])
def test_reader_keeps_closed_barcode_and_input_relationships(change):
    value=asdict(historical_context());value['libraries'][0].update(change)
    with pytest.raises((context.LibraryContextError,ValueError)):
        context.validate_scatac_library_processing_context(value)
