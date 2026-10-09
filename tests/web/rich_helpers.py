"""Actual tiny publications and normal scoped Planner for rich web acceptance."""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from agent.application import InteractiveAgentApplication, OutputSelection, ResearchAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepOutputRef
from agent.web.config import ScientificInputSet


PROFILES = tuple(PlanningModelProfile(key, 'scripted', 'model/' + key) for key in ('alpha', 'beta'))


class ScientificWork:
    """Count production and independent owner bodies in main/server workers."""
    def __init__(self):
        from agent.tools.data import (
            _barcode_qc_production as qc, _cell_selection_production as selection,
            _cell_by_ccre_production as matrix, external_fragments as fragments,
            barcode_qc_verifier as qv, cell_selection_verifier as sv,
            cell_by_ccre_verifier as mv, external_fragments_verifier as fv,
            scatac_fragments_v2_verifier as generic,
        )
        functions = {
            'production.qc': qc.produce, 'production.selection': selection.produce,
            'production.matrix': matrix.construct_counts, 'production.fragments': fragments.prepare_in_stage,
            'owner.qc': qv.verify_barcode_qc.__wrapped__,
            'owner.selection': sv.verify_cell_selection.__wrapped__,
            'owner.matrix': mv.verify_cell_by_ccre.__wrapped__,
            'owner.fragments': fv.verify_external_fragments.__wrapped__,
            'owner.generic': generic.verify_fragments_v2.__wrapped__,
        }
        self.codes = {f.__code__: name for name, f in functions.items()}
        self.counts = Counter()
        self.lock = threading.Lock()
        self.previous_main, self.previous_thread = sys.getprofile(), threading.getprofile()
        sys.setprofile(self.observe)
        threading.setprofile(self.observe)

    def observe(self, frame, event, arg):
        if event == 'call' and frame.f_code in self.codes:
            with self.lock:
                self.counts[self.codes[frame.f_code]] += 1

    def snapshot(self):
        with self.lock:
            return dict(self.counts)

    def close(self):
        sys.setprofile(self.previous_main)
        threading.setprofile(self.previous_thread)


class RichModel:
    def __init__(self, profile):
        self.model_id = profile.model_id
        self.calls = []

    def complete(self, *, prompt, response_schema):
        p = json.loads(prompt)
        self.calls.append(p)
        if 'turn_schema_version' in p:
            utterance = p['utterance']
            if utterance == 'Set minimum depth to 1.':
                operation = p['bases']['current']['operations'][0]['handle']
                choice = dict(kind='execute', base='current', operation=operation, target='selection',
                    delta=dict(parameter='min_qc_fragment_records', operation='set', literal='1', evidence=utterance.rstrip('.')))
            elif utterance.startswith('Run option'):
                offered = p['dialogue']['execution_candidates']
                candidate = offered[0]['candidate'] if offered else 'unavailable-candidate'
                choice = dict(kind='execute_candidate', candidate=candidate, evidence=utterance)
            elif 'analyze next' in utterance:
                qc = next(o for o in p['dialogue']['outputs'] if o['output_name'] == 'qc' and o['is_active'])
                choice = dict(kind='answer_guidance', targets=[dict(output=qc['handle'], subject=None)], candidate=None)
            else:
                current = next(o for o in p['dialogue']['outputs'] if o['is_active'] and o['output_name'] in ('qc', 'selection'))
                choice = dict(kind='answer_scientific', target=dict(output=current['handle'], subject=None), comparison=None, focus='question')
            return json.dumps(dict(turn_schema_version=1, decision=choice))
        if 'guidance_schema_version' in p:
            return json.dumps(dict(candidates=[dict(capability='select_scATAC_cells', explanation=dict(
                support='insufficient_evidence', paragraphs=[dict(parts=[dict(kind='text', text=t)]) for t in (
                    'Selection could help address the stated objective.',
                    'The selection criteria require explicit input.',
                    'Further validation is required before execution.')]))]))
        if 'dialogue_schema_version' in p:
            claim = next(c['claim_id'] for c in p['evidence']['claims'] if c['field'] in ('n_observed_barcodes', 'n_selected'))
            return json.dumps(dict(support='supported', paragraphs=[dict(parts=[
                dict(kind='text', text='The accepted evidence records:'), dict(kind='claim', id=claim)])]))
        if 'output_selection_schema_version' in p:
            return json.dumps(dict(outputs=[dict(name='selection', step_id='new_selection', output_key='manifest_path')]))
        if 'selection_schema_version' in p:
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['raw_preprocessing'])))
        qc = next(o['handle'] for o in p['active_outputs'] if o['producer_tool'] == 'compute_scATAC_qc')
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=[
            dict(step_id='new_selection', tool='select_scATAC_cells',
                 sources=[dict(target='barcode_qc', source=dict(kind='context', handle=qc))],
                 control_dependencies=[])])))


@dataclass
class RichHarness:
    service: InteractiveAgentApplication
    application: ResearchAgentApplication
    models: list
    work: ScientificWork
    initial_revision: str
    inputs: tuple
    environment: dict

    def close(self):
        self.work.close()
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @property
    def input_sets(self):
        return self.inputs


def rich_harness(tmp_path, *, managed_matrix=False, managed_tables=False,
                 selection_names=('fragments', 'qc', 'selection', 'matrix')):
    """Seed normal accepted science, then expose only the interactive facade."""
    import pysam
    from agent.tools.data import scatac_reference as reference, scatac_qc_reference as qc_reference
    from agent.tools.data.external_fragment_manifest import PROFILE_ID
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment = {key: os.environ.get(key) for key in (
        'AGENT_QC_BEDTOOLS', 'AGENT_MATRIX_BEDTOOLS', 'AGENT_QC_ALLOW_SYNTHETIC', 'AGENT_QC_RESOURCE_CATALOG')}
    os.environ.update(AGENT_QC_BEDTOOLS='/usr/bin/bedtools', AGENT_MATRIX_BEDTOOLS='/usr/bin/bedtools', AGENT_QC_ALLOW_SYNTHETIC='1')
    os.environ.pop('AGENT_QC_RESOURCE_CATALOG', None)
    work = ScientificWork()
    try:
        fasta = tmp_path / 'reference.fa'
        fasta.write_text('>chr2\n' + 'A' * 6001 + '\n>chr1\n' + 'C' * 6001 + '\n')
        pysam.faidx(str(fasta))
        bed = tmp_path / 'features.bed'
        bed.write_text('chr2\t0\t6001\nchr1\t0\t6001\n')
        parent = reference.build_scatac_reference_bundle(species='human', target_assembly='hg38',
            fasta_path=fasta, fai_path=Path(str(fasta) + '.fai'), ccre_bed_path=bed)
        pointer = reference.publish_scatac_reference_bundle(parent, tmp_path / 'reference.json')
        annotation = tmp_path / 'annotation.tsv'
        annotation.write_text('chr2\t3000\t3100\t+\tg1\tt1\tprotein_coding\n')
        qc_reference.build_scatac_qc_reference_bundle(parent_manifest_path=pointer['manifest_path'],
            parent_manifest_sha256=pointer['manifest_sha256'], annotation_path=annotation,
            annotation_source='synthetic', annotation_release='1',
            classifications=tuple(qc_reference.QCContig(n, 6001, 'primary_nuclear_qc') for n in ('chr2', 'chr1')),
            classification_source='explicit-test', output_dir=tmp_path / 'qc-reference')
        qc_pointer = tmp_path / 'qc-reference' / 'manifest.json'
        source = tmp_path / 'external-fragments.tsv'
        source.write_text('chr2\t2950\t3050\tA\t1\nchr2\t1000\t1001\tA\t2\nchr1\t0\t100\tB\t3\n')
        fragment_inputs = dict(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_profile=PROFILE_ID, reference_bundle_path=pointer['manifest_path'], reference_bundle_sha256=pointer['manifest_sha256'],
            namespace='explicit_library', source_selection='unknown', output_dir=str(tmp_path / 'fragments'))
        pair = lambda step, prefix: {prefix + suffix: StepOutputRef(step, 'manifest_' + suffix) for suffix in ('path', 'sha256')}
        steps = (
            PlanStep('fragments', 'import_scATAC_fragments', fragment_inputs),
            PlanStep('qc', 'compute_scATAC_qc', pair('fragments', 'fragments_manifest_') | dict(
                qc_reference_manifest_path=str(qc_pointer), qc_reference_manifest_sha256=hashlib.sha256(qc_pointer.read_bytes()).hexdigest(),
                output_dir=str(tmp_path / 'qc')), ('fragments',)),
            PlanStep('selection', 'select_scATAC_cells', pair('qc', 'barcode_qc_manifest_') | dict(
                min_qc_fragment_records=0, min_tss_enrichment='0', output_dir=str(tmp_path / 'selection')), ('qc',)),
            PlanStep('matrix', 'build_scATAC_cell_by_ccre', pair('fragments', 'fragments_manifest_') | pair('selection', 'selected_cells_manifest_') | dict(
                reference_manifest_path=pointer['manifest_path'], reference_manifest_sha256=pointer['manifest_sha256'],
                output_dir=str(tmp_path / 'matrix')), ('fragments', 'selection')),
        )
        plan = AgentPlan('seed-plan', 'seed-request', 'Tiny accepted preprocessing.', steps)
        class SeedPlanner:
            def plan(self, request, registry):
                return plan
        application = ResearchAgentApplication(tmp_path / 'workspace', planner=SeedPlanner())
        managed = ({'matrix'} if managed_matrix else set()) | ({'qc', 'selection'} if managed_tables else set())
        if managed:
            plan = replace(plan, steps=tuple(replace(step, arguments={
                **step.arguments,
                'output_dir': str(application._workspace.run_paths('seed-request:run').scientific / step.step_id),
            }) if step.step_id in managed else step for step in steps))
        application.sessions.create('analysis')
        state = application.sessions.run('analysis', 'seed', AgentRequest('seed-request', 'Tiny accepted preprocessing.', {}),
            tuple(OutputSelection(name, name, 'manifest_path') for name in selection_names), expected_generation=0)
        assert state.turn('seed').status == 'activated', state
        models = []
        def factory(profile):
            model = RichModel(profile)
            models.append(model)
            return model
        service = InteractiveAgentApplication(application.workspace_root, model_profiles=PROFILES,
            default_profile_id='alpha', planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
        inputs = (ScientificInputSet('thresholds', 'Explicit permissive selection', dict(min_qc_fragment_records=1, min_tss_enrichment='0')),)
        return RichHarness(service, application, models, work, state.active_revision_id, inputs, environment)
    except BaseException:
        work.close()
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        raise
