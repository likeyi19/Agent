"""Project a selected accepted Leiden result to the native ordered group TSV.

This is data handoff, never clustering, source alignment or label inference.
"""
from pathlib import Path
import io

import pandas as pd

from agent.orchestration.prior_outputs import accepted_step_digest
from agent.orchestration.verification_authority import _accepted_step
from agent.schemas import StepOutputRef
from agent.schemas.prior_output import PriorOutputRef
from .session_state import digest
from .result_reading import dependency as _dependency, original_source


class GroupAssociationError(ValueError):
    """A valid accepted result lacks the required association with these counts."""


def choices(sessions, interaction):
    """Offer only clusters on explicit captured-result ancestry, without file IO."""
    state = sessions.load(sessions._interaction_session_id)
    if interaction.base_revision_id is None:
        return []
    revision = next(r for r in state.revisions if r.revision_id == interaction.base_revision_id)
    store = sessions._application.run_store
    found = {}
    for output in revision.outputs:
        stored, planned, _ = _accepted_step(store, output.run_id, output.step_id)
        if accepted_step_digest(stored) != output.accepted_step_sha256:
            raise ValueError('Captured grouping result anchor changed')
        run_id, step_id = output.run_id, output.step_id
        if stored.tool_name == 'compute_cell_umap' and stored.result.get('clustering_present', True):
            if not isinstance(planned.arguments.get('analysis_path'), (StepOutputRef, PriorOutputRef)):
                # A standalone/historical layout can answer questions without
                # qualifying a clustering source for a different analysis.
                # Recorded but corrupt dependencies still fail below.
                continue
            run_id, step_id = _dependency(store, run_id, step_id, 'analysis_path')
            stored, _, _ = _accepted_step(store, run_id, step_id)
        if stored.tool_name != 'cluster_cells':
            continue
        identity = dict(run_id=run_id, step_id=step_id, accepted_step_sha256=accepted_step_digest(stored))
        handle = 'groups.' + digest(identity)
        found[handle] = dict(handle=handle, **identity, n_cells=stored.result['n_cells'],
            n_groups=stored.result['n_clusters'], resolution=stored.result['resolution'])
    return list(found.values())


def public_choices(sessions, interaction):
    return [{k: row[k] for k in ('handle', 'n_cells', 'n_groups', 'resolution')}
            for row in choices(sessions, interaction)]


def project(sessions, interaction, selected, source):
    rows = [r for r in choices(sessions, interaction) if r['handle'] == selected]
    if len(rows) != 1:
        raise ValueError('Select one offered accepted clustering result')
    row = rows[0]
    store = sessions._application.run_store
    clustered, _, _ = _accepted_step(store, row['run_id'], row['step_id'])
    original = original_source(store, row['run_id'], row['step_id'])
    if original is None or original['sha256'] != source['source_sha256']:
        raise GroupAssociationError('The selected clustering does not retain an exact source identity for these counts. Select its original count matrix or supply an independently derived ordered grouping.')
    from agent.tools.analysis.categorical import observations, axis, require_same_cells
    ids, labels = observations(clustered.result['analysis_path'], clustered.result['analysis_sha256'], 'leiden')
    try:
        require_same_cells(ids, axis(source['source_path'], source['source_sha256']))
    except ValueError as exc:
        raise GroupAssociationError('The selected clustering and counts have different ordered cells. Cell alignment or relabeling is not automatic.') from exc
    buffer = io.StringIO()
    pd.DataFrame(dict(cell_id=ids, group=labels.astype(str))).to_csv(
        buffer, sep='\t', index=False)
    content = buffer.getvalue().encode()
    identity = digest(dict(selected=row, source_sha256=source['source_sha256']))
    workspace = sessions._application._workspace
    directory = workspace._ensure_directory(workspace.root/'annotation_groups')
    root = workspace._ensure_directory(directory/identity)
    path = root/'groups.tsv'
    from .local_resources import LocalResourceAdmission
    # The ordinary immutable local-resource publication/registration owns the
    # projection too. It remains Session-private and never enters the library.
    import os, tempfile
    fd, temporary = tempfile.mkstemp(dir=root, prefix='.groups-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError('Accepted grouping projection changed')
    finally:
        Path(temporary).unlink(missing_ok=True)
    owner = LocalResourceAdmission(workspace, approved_source_roots=(root,))
    record = owner.register('annotation-groups-' + identity, path, input_type='groups',
        label='Accepted Leiden groups', attribution='Exact ordered projection of ' + selected,
        session_id=sessions._interaction_session_id)
    member = owner.compose_group([record.resource_id], attachment_candidates=True).members[0]
    return dict(selection=row, member=dict(member))
