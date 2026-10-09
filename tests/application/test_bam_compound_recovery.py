"""Accepted BAM production keeps its source lifetime across explicit intake edges."""
import json
import os
from pathlib import Path

import pytest

from agent.application.sessions import AnalysisSessions
from agent.tools.data import bam_fragments, bam_fragments_verifier

import test_registered_bam_integration as registered_bam
from test_registered_bam_integration import bam_factory


class CompoundModel(registered_bam.Model):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if ('turn_schema_version' not in value and 'output_selection_schema_version' not in value
                and 'selection_schema_version' not in response_schema.get('properties', {})):
            self.calls.append(value)
            def source(port, input_name):
                return dict(target=port, source=dict(kind='input', input=input_name))
            return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
                dict(step_id='inspect', tool=registered_bam.INSPECT, control_dependencies=[],
                     sources=[source('raw_input', 'raw_input_paths')]),
                dict(step_id='prepare', tool=registered_bam.PREPARE, control_dependencies=[], sources=[
                    source('source', 'source_path'), dict(target='intake', source=dict(kind='step', step='inspect')),
                    source('library_context', 'library_context_path'), source('reference', 'reference_bundle_path'),
                    source('source_profile', 'source_profile')]),
            ])))
        return super().complete(prompt=prompt, response_schema=response_schema)


@pytest.mark.parametrize('compound', [False, True], ids=['direct', 'explicit-intake-edge'])
@pytest.mark.parametrize('mutation', ['changed', 'deleted'])
def test_verified_production_keeps_history_while_raw_presentation_requirements_remain_owned(
        bam_factory, tmp_path, monkeypatch, compound, mutation):
    arguments = bam_factory()
    registry, calls = registered_bam.instrument(monkeypatch)
    if compound:
        monkeypatch.setattr(registered_bam, 'Model', CompoundModel)
    service, models = registered_bam.application(tmp_path, registry=registry)
    resource = registered_bam.register(service, arguments)
    values = {key: value for key, value in arguments.items()
              if key not in {'source_path', 'source_sha256', 'output_dir'}
              and not (compound and key in {'intake_manifest_path', 'intake_manifest_sha256'})}
    values.update(registered_bam.DECLARATIONS)
    binding = service.resources.compose_bam(resource.resource_id, values)
    service.create_session('session')
    with monkeypatch.context() as stopped:
        def interrupt(*args, **kwargs):
            raise registered_bam.Interrupted()
        stopped.setattr(AnalysisSessions, '_record_result', interrupt)
        with pytest.raises(registered_bam.Interrupted):
            registered_bam.submit(service, binding)
    state = service._application.sessions.load('session')
    assert not state.revisions and state.generation == 0 and state.turn('analysis').status == 'linked'
    run = service._application.run_store.load(state.turn('analysis').run_id)
    assert run.lifecycle_status.value == 'SUCCEEDED'
    producer = next(step for step in run.steps if step.tool_name == registered_bam.PREPARE)
    authority = producer.verification.artifact_authority
    assert authority['schema_version'] == 2
    assert authority['source_policy'] == 'historical_verified_sources.v1'
    assert dict(path=resource.source_path, sha256=resource.source_sha256,
                size_bytes=resource.size_bytes) in authority['historical_sources']
    expected = dict(intake=1 if compound else 0, production=1, verification=1)
    assert calls == expected and len(models) == 1
    source = Path(resource.source_path)
    original_bytes, original_stat = source.read_bytes(), source.stat()
    if mutation == 'deleted':
        source.unlink()
    else:
        source.write_bytes(source.read_bytes() + b'changed after complete scientific acceptance')
    def forbidden(*args, **kwargs):
        raise AssertionError('Historical completion reran BAM production or independent reconstruction.')
    monkeypatch.setattr(bam_fragments, 'prepare_in_stage', forbidden)
    monkeypatch.setattr(bam_fragments_verifier, 'reconstruct', forbidden)
    recovered = service.recover_turn('session', 'analysis', complete_presentation=True)
    if compound:
        # The unchanged raw evidence owner requires its captured source size and
        # mtime for bounded reinspection. Complete producer authority must not
        # turn that presentation prerequisite into an irreversible Session failure.
        assert recovered.status == 'finalizing' and recovered.revision_id is None
        pending = service._application.sessions.load('session')
        assert pending.generation == 0 and not pending.revisions
        assert pending.turn('analysis').status == 'run_succeeded'
        assert calls == expected and len(models) == 1
        source.write_bytes(original_bytes)
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        recovered = service.recover_turn('session', 'analysis', complete_presentation=True)
    assert recovered.status == 'succeeded' and recovered.revision_id
    assert service._application.sessions.load('session').generation == 1
    assert service.evidence('session', recovered.revision_id, 'fragments').status == 'available'
    assert registered_bam.submit(service, binding) == recovered
    assert calls == expected and len(models) == 1
