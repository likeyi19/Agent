# Operating and development guide

The root [README](../../README.md) owns usage examples and current boundaries.
[AGENTS.md](../../AGENTS.md) owns engineering rules, scientific contracts and
accepted limitations. The [documentation index](../README.md) separates current
owner records from historical acceptance evidence.

Activate the existing Conda environment with `conda activate agent`, or launch
noninteractive commands with `conda run --no-capture-output -n agent`. Invoking
`agent/bin/python` directly selects an interpreter but does not activate Conda's
configured environment variables. This distinction applies to Web acceptance
servers and their subprocesses, including provider credentials and native tools.
Run commands from the repository root. The implementation is under `src/agent`; normal module entry points use
`PYTHONPATH=src`. Packaging and an installed console script remain deferred.
The tracked empty `pyproject.toml` retains its existing project-marker behavior.

Optional dependencies are grouped under `requirements/`:

| File | Purpose |
| --- | --- |
| [bam.txt](../../requirements/bam.txt) | Optional BAM input support |
| [web.txt](../../requirements/web.txt) | FastAPI and Uvicorn for the browser application |
| [web-test.txt](../../requirements/web-test.txt) | Web dependencies plus HTTPX for HTTP tests |

For the Web application:

The [manual Web testing guide](manual-web-testing.md) provides the prepared,
separate researcher workspace, complete host commands, finite upload/download
settings, a runtime-derived native memory budget, SSH forwarding and shutdown.

```bash
python -m pip install -r requirements/web.txt
PYTHONPATH=src conda run --no-capture-output -n agent python -m agent.web --config /path/to/operator-web.json \
  --host 127.0.0.1 --port 8000
```

The local deployment's existing `agent-data/config/launch-uat.sh` already uses
`conda run` and accepts the same Web arguments, including an explicit `--config`
override for an isolated acceptance workspace. Use it for Firefox acceptance.
Before any live submission, verify the service process's interpreter, Conda prefix
and environment name, and only the boolean presence of `OPENAI_API_KEY`. Never
print variable values or substitute another credential after a launch failure.
The server inherits activated variables; native subprocesses inherit them in turn.

The [README usage section](../../README.md#usage) covers CLI planning, execution,
continuation and cancellation. The [M18.3 configuration record](../history/m18.3-basic-web-application.md)
describes the operator JSON, admitted model profiles and named input sets;
[R7](../architecture/ua3.6.5-r7-agent-internal-state-lifecycle-simplification.md)
owns current interruption and recovery behavior.

The Web client keeps the selected conversation, draft, attachment references and
result preview in tab-scoped session storage. Refresh and a normal server restart
reopen that tab's conversation without resubmitting work. Recent conversations
remains browser-local shared discovery; it does not enumerate server Sessions.
Closing a tab may discard its unsent draft/display choices. When upgrading from
the older browser-global convenience storage, reopen the desired conversation
once using Recent conversations or its Session ID. Old scientific history remains
on the server. No entire Session history or scientific authority is stored in the
browser.

Latest requested outputs in the result selector uses exact published figure and
export source references. Selecting an earlier presentation preserves its original
wording and source attribution; it does not change the conversation's active
scientific result. Downloads report a request only. Verify the saved file through
the browser's own download interface when local delivery matters.

Scientific results show a workflow overview with repeated counts combined; exact
per-output records, original explanations and evidence qualifications remain in
collapsed native details. Figure previews open the existing full-size viewer;
each preview has its own download. Primary scientific data and the report appear
together. Supporting evidence and all native/intermediate artifacts remain
expandable. The conversation shows dialogue and existing result-navigation buttons,
without a second artifact inventory. Distinct annotation, compact analysis and
integrated H5ADs retain their actual contents and exact source identities; display
ranking follows source relationships rather than filenames.

Authoritative finite reference choices appear as keyboard-accessible Web buttons.
A click submits the captured question and choice identities as typed intent through
the existing pending continuation and planning flow, without a language-model call
to interpret the click. No numbered-text adapter is maintained. Open-ended replies
continue through ordinary language interpretation. Language guidance applies to all
languages supported by the configured model and respects explicit instructions;
short value-only answers retain their conversation context. Scientific identifiers,
historical displayed text and precise error codes remain unchanged. Static technical
UI/status labels are not translated through additional model calls.

Execution progress has one segment per planned step, its stored state, the
completed count and the actual running function where supplied. Symbols and
accessible labels accompany colors. Expand the details for errors and all steps;
Cancel turn retains its existing behavior. Counts measure completed steps, not
elapsed runtime. Conversation generation follows the user's language and explicit
language requests while preserving scientific names. Natural reference selections
bind offered identities; ambiguous descriptions still require clarification.

RunStore records timezone-aware wall-clock observations exactly, including clock
corrections during execution or between processes. `created_at`, `updated_at`,
step start/finish and trace timestamps are audit observations; their numerical
order does not establish execution causality. Revision CAS, allowed lifecycle
transitions, contiguous trace sequence and immutable trace prefixes retain that
responsibility. Timestamp syntax, required fields, scientific verification and
artifact identities remain checked. A clock correction grants no retry authority.

The executor measures elapsed step durations with its existing monotonic clock;
raw process-relative clock readings are not stored as cross-restart authority.
Historical timestamps and accepted outputs are not rewritten. The existing
record format is unchanged; reading new records containing a clock rollback
requires the corrected reader. Accurate absolute UTC remains a host-operator
responsibility. The [OPT-3B clock closeout](../../../agent-data/acceptance/ua3.6.5/uat-02/post-round2-opt-3b/clock-closeout/REPORT.md)
records the host observations, software qualification and remaining limitations.

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
Reusable references and model resources normally register in the single active
`/home/likeyi/program/agent-data/resources/library` through its existing installation
owner. Isolate experiments with Sessions/workspaces and private uploads. Historical
acceptance library indexes remain where needed for reproducibility; they are not
preferred alternative catalogs and do not need copying for each new experiment.
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
