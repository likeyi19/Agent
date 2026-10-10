"""One bounded live human upload/analysis through the ordinary native Web UI.

Requires --live after separate deterministic/compute readiness checks. Frozen
UA3.5.4a transport owns paid bounds and raw capture; every model response is real.
The existing application, executor, scientific owners and browser remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from agent.application import InteractiveAgentApplication
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import ToolRegistry, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry, build_default_planning_model_factory_registry
from agent.web.app import create_app
from agent.web.config import ScientificInputSet, load_web_configuration

from . import run_ua354a as base
from .run_ua354b import RECOVERY_POLICY
from .diagnostics import sanitize_provider_error


UTTERANCE = ('Analyze these human scATAC-seq cells with EpiZoo, construct their '
             'neighbor graph, perform Leiden clustering, and compute UMAP.')
COMPANION_ID = 'human-ua355-declaration'
DEFAULT_BROWSER_HELPER = '/home/likeyi/agent-m18.5-k22bsed1/browser_common.py'


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def workspace_bytes(root):
    return {str(p.relative_to(root)): file_sha(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


class Runner(base.Runner):
    def __init__(self, output):
        super().__init__(output)
        self.output.chmod(0o700)
        self.case = 'A1'  # Existing guard identity, restricted to one browser turn.
        self.constructions = []
        self.science_calls = []
        self.observations = Counter()
        self.service = None
        self.server = self.server_thread = None
        self.http_port = 0
        self.report.update(kind='UA3.5.5-native-human-workflow', schedule=['A1'],
            utterance=UTTERANCE, planning_recovery_policy=RECOVERY_POLICY.to_dict(),
            discovery_policy='fresh_exact_catalog_once', scientific_executor='ordinary_default',
            foundation_owner_call_ceiling=1, scientific_cell_ceiling=32,
            script_sha256=file_sha(__file__), base_runner_sha256=file_sha(base.__file__),
            ua354b_runner_sha256=file_sha(Path(__file__).with_name('run_ua354b.py')),
            checks=[], scientific_calls=self.science_calls, model_constructions=self.constructions)
        self.save()

    def factory(self, profile):
        model = super().factory(profile)
        self.constructions.append(profile.profile_id)
        self.save()
        return model

    def counted_registry(self):
        registry = build_default_tool_registry()
        def counted(spec):
            def invoke(**arguments):
                if spec.name == 'epizoo_embed_cells' and spec.name in self.science_calls:
                    raise RuntimeError('Evaluation foundation-owner call bound exceeded.')
                self.science_calls.append(spec.name)
                print(json.dumps({'scientific_owner_entered': spec.name}), flush=True)
                self.save()
                return spec.function(**arguments)
            return replace(spec, function=invoke)
        return ToolRegistry(tuple(counted(registry.get(name)) for name in registry.names()))

    def application(self, config, workspace, registry):
        ordinary = build_default_planning_model_factory_registry()
        # Preserve configured choices/default while closing this evaluation's
        # paid dispatch allowlist before any unexpected provider construction.
        factories = {provider: self.factory for provider in ordinary.provider_ids}
        factories['openrouter'] = self.factory
        profiles = config.model_profiles
        if base.CANDIDATE.profile_id in {p.profile_id for p in profiles}:
            if next(p for p in profiles if p.profile_id == base.CANDIDATE.profile_id) != base.CANDIDATE.profile():
                raise ValueError('Configured F1 profile differs from the accepted exact profile.')
        else:
            profiles = (*profiles, base.CANDIDATE.profile())
        uploads = workspace / 'uploads'
        uploads.mkdir(parents=True, exist_ok=True, mode=0o700)
        return InteractiveAgentApplication(workspace, model_profiles=profiles,
            default_profile_id=config.default_profile_id, display_labels=dict(config.display_labels),
            planning_model_factory_registry=PlanningModelFactoryRegistry(factories),
            registry=registry, approved_source_roots=(uploads,),
            planning_recovery_policy=RECOVERY_POLICY, epizoo_resources=config.epizoo_resources)

    def start_server(self, input_sets, config, workspace):
        import uvicorn
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', self.http_port))
        self.http_port = listener.getsockname()[1]
        listener.listen(128)
        web = create_app(self.service, input_sets=input_sets, max_workers=1,
            uploads=H5ADUploadAdmission(self.service.resources, workspace / 'uploads',
                max_bytes=config.upload_max_bytes, max_concurrent=1),
            epizoo_resources=config.epizoo_resources)
        self.server = uvicorn.Server(uvicorn.Config(web, log_level='warning', access_log=False))
        self.server_thread = threading.Thread(target=lambda: self.server.run(sockets=[listener]), daemon=True)
        self.server_thread.start()
        deadline = time.monotonic() + 20
        while not self.server.started:
            if time.monotonic() >= deadline:
                raise RuntimeError('Evaluation Web server did not start.')
            time.sleep(.05)

    def stop_server(self):
        if self.server is not None:
            self.server.should_exit = True
            self.server_thread.join(30)
            if self.server_thread.is_alive():
                raise RuntimeError('Evaluation Web server did not drain.')
            self.server = self.server_thread = None

    def check(self, name, passed, **details):
        row = dict(check=name, passed=bool(passed), **details)
        self.report['checks'].append(row)
        self.save()
        print(json.dumps(row), flush=True)
        return bool(passed)

    def profile_observer(self, frame, event, arg):
        if event != 'call':
            return
        path, name = frame.f_code.co_filename, frame.f_code.co_name
        observed = {
            ('/agent/orchestration/verifier.py', 'verify_step'),
            ('/agent/orchestration/verifier.py', 'verify_run'),
            ('/agent/tools/models/epizoo.py', 'embed_cells'),
            ('/agent/tools/models/epizoo.py', 'load_model'),
            ('/agent/tools/models/epizoo.py', '_load_resources'),
            ('/agent/tools/analysis/epizoo_embedding.py', '_validate_backend_result'),
        }
        for suffix, function in observed:
            if path.endswith(suffix) and name == function:
                self.observations[suffix.split('/agent/', 1)[1] + ':' + name] += 1

    def run(self, args):
        import anndata as ad
        import torch
        source = Path(args.source_h5ad).absolute()
        workspace = Path(args.workspace).absolute()
        if workspace.exists():
            raise ValueError('Acceptance requires a fresh private workspace.')
        # This checks the evaluation workload bound, not scientific qualification.
        matrix = ad.read_h5ad(source, backed='r')
        try:
            shape = list(matrix.shape)
            if not 16 <= shape[0] <= 32 or shape[1] != 1_355_445:
                raise ValueError('Bounded human input must have 16–32 cells and the full human feature axis.')
        finally:
            matrix.file.close()
        if not torch.cuda.is_available():
            raise RuntimeError('Required qualified CUDA device is unavailable.')
        properties = torch.cuda.get_device_properties(0)
        if properties.total_memory > 24 * 1024 ** 3:
            raise RuntimeError('Acceptance hardware differs from the qualified 24 GiB class.')
        config = load_web_configuration(args.operator_config)
        defaults = [r for r in config.epizoo_resources if r.species == 'human' and r.default]
        if len(defaults) != 1:
            raise ValueError('Exactly one operator-admitted human resource default is required.')
        if COMPANION_ID in {s.input_set_id for s in config.input_sets}:
            raise ValueError('Evaluation companion identity conflicts with operator inputs.')
        input_sets = (*config.input_sets, ScientificInputSet(COMPANION_ID,
            'Human species declaration for bounded acceptance', {'species': 'human'}, h5ad_companion=True))
        self.report.update(input=dict(source_sha256=file_sha(source), shape=shape),
            operator_default_profile=config.default_profile_id,
            operator_profile_ids=[p.profile_id for p in config.model_profiles],
            operator_input_set_ids=[s.input_set_id for s in config.input_sets],
            evaluation_profile_admission=asdict(base.CANDIDATE.profile()),
            evaluation_companion_admission=dict(input_set_id=COMPANION_ID,
                h5ad_companion=True, execution_inputs={'species': 'human'}),
            resource_default=defaults[0].choice(),
            browser_helper_sha256=file_sha(args.browser_helper),
            compute=dict(device='cuda:0', name=properties.name, total_memory_bytes=properties.total_memory))
        self.save()
        registry = self.counted_registry()
        self.service = self.application(config, workspace, registry)
        helpers = {'__name__': 'ua355_existing_native_helpers'}
        exec(compile(Path(args.browser_helper).read_text(), 'ua355-existing-browser-helper', 'exec'), helpers)
        Browser = helpers['Browser']
        visible = Path(tempfile.mkdtemp(prefix='agent-ua355-native-', dir='/home/likeyi'))
        visible.chmod(0o700)
        profile = visible / 'firefox-profile'
        profile.mkdir(mode=0o700)
        visible_source = visible / source.name
        shutil.copy2(source, visible_source)
        port_socket = socket.socket()
        port_socket.bind(('127.0.0.1', 0))
        marionette_port = port_socket.getsockname()[1]
        port_socket.close()
        (profile / 'user.js').write_text('\n'.join((
            f'user_pref("marionette.port", {marionette_port});',
            'user_pref("browser.shell.checkDefaultBrowser", false);',
            'user_pref("browser.startup.homepage_override.mstone", "ignore");')))
        log = (self.output / 'firefox.log').open('w')
        process = browser = None
        old_profile, old_thread_profile = sys.getprofile(), threading.getprofile()
        try:
            sys.setprofile(self.profile_observer)
            threading.setprofile(self.profile_observer)
            self.start_server(input_sets, config, workspace)
            process = subprocess.Popen([args.firefox, '--headless', '--no-remote', '--profile', str(profile),
                '--marionette', '--remote-allow-system-access'], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'MOZ_HEADLESS': '1'})
            browser = Browser(self.output, url=f'http://127.0.0.1:{self.http_port}', port=marionette_port)
            browser.start(clear_storage=True)
            sid = browser.create_session()
            browser.set_value('#model-choice', base.CANDIDATE.profile_id)
            browser.set_value('#input-choice', COMPANION_ID)
            element = browser.command('WebDriver:FindElement', {'using': 'css selector', 'value': '#upload-file'})
            identifier = element.get('element-6066-11e4-a52e-4f735466cecf') or element.get('ELEMENT')
            browser.command('WebDriver:ElementSendKeys', {'id': identifier, 'text': str(visible_source)})
            browser.wait('return !document.getElementById("upload-file-button").disabled;')
            browser.click('#upload-file-button')
            browser.wait('return document.getElementById("resource-choice").value && !document.getElementById("submit-turn").disabled;')
            resource_id = browser.js('return document.getElementById("resource-choice").value;')
            registered = self.service.resources.load(resource_id)
            self.report['registered_input'] = registered.to_dict()
            if not self.check('Native file upload preserves exact bytes; upload alone performs no models or science',
                registered.source_sha256 == self.report['input']['source_sha256']
                and not self.constructions and not self.science_calls,
                resource_id=resource_id, snapshot=browser.snapshot()):
                raise ValueError('Upload identity or zero-execution prerequisite failed.')
            choices = browser.js('return {profile:document.getElementById("model-choice").value,input_set:document.getElementById("input-choice").value,resource:document.getElementById("resource-choice").value,epizoo:document.getElementById("epizoo-resource-choice").value};')
            if choices != dict(profile=base.CANDIDATE.profile_id, input_set=COMPANION_ID,
                               resource=resource_id, epizoo=''):
                raise ValueError('Browser profile, explicit declaration or default resource choice changed.')
            self.report['browser_selected_inputs'] = choices
            torch.cuda.reset_peak_memory_stats(0)
            browser.submit(UTTERANCE, timeout=900)
            state = self.service._application.sessions.load(sid)
            interaction = state.interactions[-1]
            view = self.service.turn(sid, interaction.turn_id).to_dict()
            self.artifact('workflow-session.json', state.to_dict())
            self.artifact('workflow-view.json', view)
            self.report['results'] = [dict(case='A1', session_id=sid, turn_id=interaction.turn_id,
                utterance=UTTERANCE, view=view)]
            run = None
            if view.get('run_id'):
                run = self.service._application.run_store.load(view['run_id'])
                self.artifact('workflow-run.json', run.to_dict())
            self.check('Real interpreted task has normal scientific execution and durable accepted Revision',
                view.get('status') == 'succeeded' and bool(view.get('revision_id'))
                and all(self.science_calls.count(name) == 1 for name in (
                    'epizoo_embed_cells', 'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap'))
                and run is not None and run.preflight_verification is not None
                and run.preflight_verification.passed
                and all(step.verification and step.verification.passed for step in run.steps),
                snapshot=browser.snapshot(), scientific_calls=list(self.science_calls))
            browser.screenshot('workflow-result.png')
            if view.get('revision_id'):
                browser.wait_revision()
                available = browser.js('return Array.from(document.getElementById("evidence-output").options).map(n=>n.value);')
                evidence = []
                for index, name in enumerate(available):
                    text = browser.show_evidence(name)
                    evidence.append(dict(output=name, text=text))
                self.artifact('browser-evidence.json', evidence)
                artifacts = browser.fetch(f'/sessions/{sid}/revisions/{view["revision_id"]}/artifacts')
                self.artifact('browser-artifacts.json', artifacts)
                reports = [a for a in artifacts.get('data', {}).get('artifacts', []) if a.get('artifact_type') == 'analysis_report']
                report_text = browser.view_report(reports[0]['handle']) if reports else None
                self.artifact('browser-report.json', dict(text=report_text))
                self.check('User can inspect accepted Revision, scientific evidence and existing safe report',
                    bool(available) and bool(report_text), evidence_outputs=available)
                browser.screenshot('workflow-evidence.png')
                figures = [a for a in artifacts.get('data', {}).get('artifacts', []) if a.get('artifact_type') == 'analysis_figure']
                if figures:
                    browser.js('const node=Array.from(document.querySelectorAll("button[data-artifact-handle]")).find(n=>n.dataset.artifactHandle===arguments[0]);if(!node)throw new Error("Accepted figure control missing");node.click();', [figures[0]['handle']])
                    browser.wait('const node=document.getElementById("figure-content");return !node.hidden&&node.complete&&node.naturalWidth>0;')
                    browser.screenshot('workflow-figure.png')
                self.check('User can view the existing accepted scientific PNG figure', bool(figures))
            else:
                self.check('User can inspect accepted Revision, scientific evidence and existing safe report', False,
                    reason='No accepted Revision was created; original failure/clarification preserved.')
            def replay_counts():
                return dict(calls=len(self.report['calls']), constructions=len(self.constructions),
                    science=len(self.science_calls), observations=dict(self.observations))
            before = replay_counts()
            saved = state.to_dict()
            persisted = workspace_bytes(workspace)
            browser.refresh()
            after_refresh = replay_counts()
            self.check('Refresh restores the accepted history without models or science',
                self.service._application.sessions.load(sid).to_dict() == saved
                and before == after_refresh, before_counts=before, after_counts=after_refresh)
            self.stop_server()
            self.service = self.application(config, workspace, registry)
            self.start_server(input_sets, config, workspace)
            browser.refresh()
            after_restart = replay_counts()
            self.check('Fresh server reopens the same persisted result without semantic/scientific replay',
                self.service._application.sessions.load(sid).to_dict() == saved
                and workspace_bytes(workspace) == persisted
                and before == after_restart, before_counts=before, after_counts=after_restart,
                snapshot=browser.snapshot())
            browser.screenshot('workflow-restored.png')
            browser.command('Marionette:SetContext', {'value': 'chrome'})
            errors = browser.js('return Services.console.getMessageArray().filter(m=>m instanceof Ci.nsIScriptError && m.sourceName.includes(arguments[0])).map(m=>m.errorMessage);', [browser.url])
            browser.command('Marionette:SetContext', {'value': 'content'})
            self.check('Native browser has no application JavaScript errors', not errors, errors=errors)
            self.report['compute']['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            self.report['normal_verifier_and_backend_call_observations'] = dict(self.observations)
            self.report['source_bytes_preserved'] = file_sha(source) == self.report['input']['source_sha256']
            self.check('Workload and original source remain within the admitted bounds',
                self.report['source_bytes_preserved']
                and self.report['compute']['peak_allocated_bytes'] < 24 * 1024 ** 3
                and self.observations['tools/models/epizoo.py:embed_cells'] <= 1)
            self.report['status'] = 'PASS' if all(row['passed'] for row in self.report['checks']) else 'FAIL'
            self.save()
        finally:
            if browser:
                try:
                    browser.close()
                except Exception:
                    pass
            if process and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            log.close()
            self.stop_server()
            sys.setprofile(old_profile)
            threading.setprofile(old_thread_profile)
            shutil.rmtree(visible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Explicitly enable the single bounded provider/browser/science acceptance.')
    for name in ('output', 'workspace', 'source-h5ad', 'operator-config'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--browser-helper', default=DEFAULT_BROWSER_HELPER)
    parser.add_argument('--firefox', default='/snap/bin/firefox')
    parser.add_argument('--discovery-json', help='Reuse this evaluation\'s exact catalog evidence from a zero-completion attempt.')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; no browser, provider or science is started by offline use.')
    runner = Runner(args.output)
    try:
        if args.discovery_json:
            path = Path(args.discovery_json).absolute()
            prior_report = json.loads(path.read_text())
            prior = prior_report['discovery']
            row, = prior['candidates']
            if (prior_report['kind'] != 'UA3.5.5-native-human-workflow'
                    or prior_report['candidate'] != base.CANDIDATE.manifest()
                    or prior_report['calls'] or prior_report['scientific_calls']
                    or prior['catalog_requests'] != 1
                    or row['candidate'] != base.CANDIDATE.manifest()
                    or row['discovery_result'] != 'EXACT_MODEL_AVAILABLE'
                    or base.Decimal(row['pricing']['prompt']) != base.PROMPT_PRICE
                    or base.Decimal(row['pricing']['completion']) != base.COMPLETION_PRICE
                    or base.Decimal(row['pricing'].get('request', '0')) != 0
                    or not {'response_format', 'structured_outputs', 'max_completion_tokens'} <= set(row['supported_parameters'] or ())):
                raise base.Stop('Reused same-evaluation catalog identity or policy changed.')
            runner.report['discovery'] = prior
            runner.report['discovery_policy'] = 'reuse_exact_same_evaluation_catalog; no additional lookup'
            runner.report['discovery_reused_from'] = dict(path=str(path), sha256=file_sha(path))
            runner.availability = base.Availability(**row['availability'])
            runner.save()
        else:
            runner.discovery()
        runner.run(args)
    except BaseException as exc:
        runner.report['hard_stop'] = sanitize_provider_error(exc, secrets=runner.private)
        runner.report['status'] = 'FAIL'
        runner.report['normal_verifier_and_backend_call_observations'] = dict(runner.observations)
        runner.save()
        raise
    print(json.dumps(dict(status=runner.report['status'], calls=len(runner.report['calls']),
        science=runner.science_calls, output=str(runner.output))), flush=True)
    return 0 if runner.report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
