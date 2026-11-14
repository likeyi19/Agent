"""Private systemd worker bootstrap. No caller-supplied Python or shell code."""
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    # Verify the kernel boundary before importing scientific packages or data.
    group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    effective = {key: (group/key).read_text().strip() for key in
                 ('memory.max', 'memory.swap.max', 'memory.oom.group')}
    if (group.name != job['unit'] or effective != {
            'memory.max': str(job['memory_bytes']), 'memory.swap.max': '0', 'memory.oom.group': '1'}
            or str(os.getpid()) not in (group/'cgroup.procs').read_text().split()):
        raise RuntimeError('Kernel native execution boundary is not effective')
    (job_path.parent/'membership.json').write_text(json.dumps(dict(
        pid=os.getpid(), cgroup=str(group), effective=effective,
        cpu_max=(group/'cpu.max').read_text().strip() if (group/'cpu.max').is_file() else None)))
    os.environ.clear()
    os.environ.update(job['environment'])  # Activated operator runtime, never printed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from agent.tools import _native_process as process
    from agent.tools.data.authority_context import authority_operation, restore_worker_context, snapshot_worker_context
    from agent.schemas.orchestration import _serialize
    context = restore_worker_context(job.get('context'))
    from agent.orchestration.prior_outputs import prior_output_store
    from agent.orchestration.run_store import FileRunStore
    store = FileRunStore(job['store_root']) if job.get('store_root') else None
    token = process._CONTROLLED.set(True)
    try:
        with authority_operation(context), prior_output_store(store):
            result = dispatch(job['operation'], job['arguments'])
            response = dict(result=_serialize(result), context=snapshot_worker_context(context))
    except Exception as exc:
        traceback.print_exc()
        from agent.orchestration.registry import _classify_tool_exception
        classification = _classify_tool_exception(exc)
        response = dict(error=dict(type=type(exc).__name__, message=str(exc),
            category=classification.category.value, code=getattr(exc, 'code', classification.code)))
    finally:
        process._CONTROLLED.reset(token)
    import resource
    usage = {}
    for key, who in (('worker', resource.RUSAGE_SELF), ('descendants', resource.RUSAGE_CHILDREN)):
        value = resource.getrusage(who)
        usage[key] = dict(user_seconds=value.ru_utime, system_seconds=value.ru_stime,
                          max_rss_kib=value.ru_maxrss)
    response['resource_usage'] = usage
    temporary = job_path.parent/'response.tmp'
    temporary.write_text(json.dumps(response, allow_nan=False))
    temporary.replace(job_path.parent/'response.json')


def dispatch(operation, args):
    if operation == 'native_callable':
        from agent.tools.analysis.native_callable import execute
        return execute(**args)
    if operation == 'command':
        from agent.tools._native_process import run
        args['log_path'] = Path(args['log_path'])
        return run(**args)
    if operation == 'annotation_observe':
        from agent.tools.analysis.annotation_input import observe
        return observe(**args)
    if operation == 'matrix':
        from agent.tools.analysis import native_matrix
        owner = {p.package: p for p in (native_matrix.SCANPY, native_matrix.EPISCANPY,
            native_matrix.SNAPATAC2, native_matrix.SIGNAC)}[args['package']]
        return owner.execute(args['values'], args['execution_identity'])
    if operation in ('chromvar', 'chromvar_native', 'cicero_cds'):
        from importlib import import_module
        return import_module('agent.tools.analysis.' + operation).execute(**args)
    raise ValueError('Unregistered internal native operation')


if __name__ == '__main__':
    main()
