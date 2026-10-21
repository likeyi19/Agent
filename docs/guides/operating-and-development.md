# Operating and development guide

The root [README](../../README.md) owns usage examples and current boundaries.
[AGENTS.md](../../AGENTS.md) owns engineering rules, scientific contracts and
accepted limitations. The [documentation index](../README.md) separates current
owner records from historical acceptance evidence.

Use the existing `agent` Python environment and run commands from the repository
root. The implementation is under `src/agent`; normal module entry points use
`PYTHONPATH=src`. Packaging and an installed console script remain deferred.
The tracked empty `pyproject.toml` retains its existing project-marker behavior.

Optional dependencies are grouped under `requirements/`:

| File | Purpose |
| --- | --- |
| [bam.txt](../../requirements/bam.txt) | Optional BAM input support |
| [web.txt](../../requirements/web.txt) | FastAPI and Uvicorn for the browser application |
| [web-test.txt](../../requirements/web-test.txt) | Web dependencies plus HTTPX for HTTP tests |

For the Web application:

```bash
python -m pip install -r requirements/web.txt
PYTHONPATH=src python -m agent.web --config /path/to/operator-web.json \
  --host 127.0.0.1 --port 8000
```

The [README usage section](../../README.md#usage) covers CLI planning, execution,
continuation and cancellation. The [M18.3 configuration record](../history/m18.3-basic-web-application.md)
describes the operator JSON, admitted model profiles and named input sets;
[R7](../architecture/ua3.6.5-r7-agent-internal-state-lifecycle-simplification.md)
owns current interruption and recovery behavior.

Repository ownership follows the existing layout:

| Location | Purpose |
| --- | --- |
| `src/agent/` | Planning, application, orchestration, scientific tools and packaged Web assets |
| `tests/` | Automated checks, scientific qualification and technical fixture definitions |
| `benchmarks/` | Planner and interactive-provider qualification harnesses |
| `third_party/epizoo/`, `third_party/epiagent/` | Pinned scientific backend source, packaging and original notices |
| `third_party/patches/chromap/` | Reviewed upstream patch and qualification record |
| `requirements/` | Optional dependency groups |
| `docs/` | Guidance, architecture and historical engineering records |

Benchmark results and acceptance execution state use explicit destinations outside
the repository. FS-3 retires the old root `evals/` and `outputs/` trees after
retaining the necessary verified evidence under
`/home/likeyi/program/agent-data/acceptance/historical/`. Current benchmark readers
take an explicit evidence location; retired command examples remain labelled as
historical records. Use task-owned external temporary directories for new runs.

External file placement follows these existing responsibilities:

| Location under `agent-data/` | Purpose |
| --- | --- |
| `resources/` | Qualified scientific references, models, indexes and runtimes |
| `datasets/` | Reusable original inputs and scientifically meaningful derived datasets |
| `workspaces/` | Managed application Sessions, Runs and private uploads |
| `acceptance/` | Compact accepted scientific and engineering evidence |
| `config/` | Private operator configuration |
| `scratch/<task>/` | Disposable development output, removed when the task completes |

An isolated system temporary directory is also suitable when its lifecycle and
cleanup are explicit. Avoid persistent random-named Agent directories under the
home directory. Retain only useful final evidence after a task, with source
identity and limitations. At the end of each development or acceptance work unit,
review newly created files and directories, retain necessary evidence and remove
disposable artifacts. At significant milestones, review the repository root and
Agent-owned storage within a bounded scope for obsolete files, duplicate assets
and misplaced outputs. These are development conventions, not runtime path
restrictions on researchers' files.

The opt-in [real EpiZoo/SEAM qualification helper](../../tests/epizoo_adaptation/qualify_real_backend.py)
lives beside its maintained owner tests. It exercises supplied pretrained weights
and native adaptation on an artificial four-cell/32-feature fixture. It is a
technical compatibility check, not a biological performance test or ordinary UAT
prerequisite. Native training requires the existing explicit scientific settings;
the ordinary regression suite does not invoke it.

The configured `workspace_root` owns Sessions, RunStore state, run artifacts,
uploads and private registrations together. The configured
`scientific_resource_root` owns the shared ScientificResourceLibrary.
Operator configuration, reusable biological inputs, scientific resources and
model checkpoints remain outside Git. [R4](../architecture/ua3.6.5-r4-unified-scientific-resource-library.md)
and [R7](../architecture/ua3.6.5-r7-agent-internal-state-lifecycle-simplification.md)
describe these owners and supported reuse. Moving repository documentation does
not relocate any scientific or runtime data.

For development, choose tests relevant to the changed boundary and use small
fixtures before native scientific execution. Keep temporary workspaces outside
the repository. When checking filesystem cleanup, use Python `-B` (or
`PYTHONDONTWRITEBYTECODE=1`) and pytest `-p no:cacheprovider` to avoid regenerating
the removed caches. Scientific and provider qualification gates retain their
existing explicit requirements; consult [AGENTS.md](../../AGENTS.md).

The [scientific storage guide](scientific-storage.md) describes the consolidated
external resources, configured environment and fresh ordinary UAT deployment.
The [backend installation guide](backends.md) explains the two optional managed
packages. Their source belongs in the repository; checkpoints, references and
ordinary biological inputs retain the external storage roles above.
