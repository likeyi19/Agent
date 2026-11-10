"""Raw tasks retain explicit downstream intent without supplying derived sources."""
from types import SimpleNamespace

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.raw_handoff import validate_arguments


@pytest.fixture(params=['fastq', 'bam', 'fragments'])
def raw_binding(request, tmp_path):
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    kind = request.param
    if kind == 'fastq':
        records = []
        for role in ('R1', 'R2', 'R3'):
            path = tmp_path / (role + '.fastq')
            path.write_bytes(b'@synthetic\nACGT\n+\nIIII\n')
            records.append(owner.register(role, path, input_type='fastq', label=role,
                attribution='Synthetic binding test only', fastq_member=dict(library_id='test',
                fastq_layout='tenx-atac-r1-r2-r3.v1', role=role, lane='001', chunk='001', compression='plain')))
        record = owner.register_fastq_collection('test', [r.resource_id for r in records],
            label='Synthetic collection', attribution='Binding test only')
        compose = owner.compose_native
    else:
        path = tmp_path / ('source.bam' if kind == 'bam' else 'source.fragments')
        path.write_bytes(b'Registered bytes are not scientific qualification')
        record = owner.register('source', path, input_type='bam' if kind == 'bam' else 'external_fragments',
            label='Synthetic source', attribution='Binding test only')
        compose = owner.compose_native
    return kind, lambda values: compose(record.resource_id, values)


def test_raw_task_accepts_explicit_analysis_parameters(raw_binding):
    kind, compose = raw_binding
    values = dict(run_snapatac2__tl_spectral__kwargs={'n_comps':30},
        run_snapatac2__pp_knn__kwargs={'n_neighbors':15}, run_snapatac2__tl_leiden__kwargs={'resolution':0.6})
    binding = compose(values)
    assert all(binding.execution_inputs[k] == v for k, v in values.items())
    validate_arguments(tuple(SimpleNamespace(tool=tool) for tool in (
        'run_snapatac2', 'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap')), kind)


@pytest.mark.parametrize('values', [dict(input_path='/foreign.h5ad'), dict(embedding_path='/foreign.npy'),
    dict(analysis_path='/foreign.json'), dict(cell_ids_path='/foreign.tsv'), dict(overwrite=True)])
def test_raw_task_cannot_supply_derived_sources_or_overwrite(raw_binding, values):
    _, compose = raw_binding
    with pytest.raises(ResourceAdmissionError):
        compose(values)
