# Native execution resource safety

`agent.tools._native_process` is the shared execution-resource owner. Executor,
RunStore, Session, Revision and each scientific publication owner retain their
existing responsibilities. No task scheduler or automatic retry is added.

On Linux, the owner starts an existing user-systemd transient service. A fresh
Python interpreter verifies its own cgroup membership, `memory.max`,
`memory.swap.max=0` and `memory.oom.group=1` before importing scientific packages or
loading data. Native R/Python descendants inherit the same aggregate memory bound.
`OOMPolicy=kill`, `KillMode=control-group`, `Restart=no` and a finite service runtime
ensure termination belongs to the OS. The supervisor is outside that group.
A process group or RSS sample is not used as a substitute for memory enforcement.

The operator configures the activated Agent environment:

| Variable | Meaning |
| --- | --- |
| `AGENT_NATIVE_MEMORY_BYTES` | Required positive integer aggregate task memory budget; no default. |
| `AGENT_NATIVE_TIMEOUT_SECONDS` | Positive finite whole-task timeout; default 7200 seconds. |
| `AGENT_NATIVE_ARCHIVE_BYTES` | Cicero exchange expansion temporary-storage budget; default 2 GiB. This protects disk usage, not scientific applicability. |

Budgets are operator settings, never Planner arguments or researcher prerequisites.
Choose memory in the deployment's resource plan, considering other services and
parent cgroup limits. Do not infer a dataset's scientific validity from its size.
No estimator, automatic downsampling, parameter reduction or alternative method
is used. A task may fail honestly within its configured budget.

The deployment needs cgroup v2 memory/swap controllers and a user-systemd manager
that can enforce the listed properties (including OOMPolicy). Unsupported or
missing facilities fail closed. Agent does not edit system settings, install a
new daemon, elevate permissions or enable delegation. The activated environment,
including configured R paths/libraries, is privately transferred to the worker;
it is not printed or passed as command-line environment values.

There is one active native task per Unix service user. A user runtime lock and the
single `agent-native-task.service` unit prevent concurrent Agent processes from
multiplying the configured budget. A still-present unit after supervisor failure
blocks a new task; the operator inspects it before cleanup. No queue or retry is
created. The finite systemd runtime still applies after supervisor death.

Cancellation and timeout stop the owned service subtree. Diagnostic-read failures
cannot skip termination, and a timed-out launcher still triggers owned cleanup. The supervisor waits for
OS termination before deleting its private request/environment files and only
new unpublished staging directories. Completed immutable publications remain
subject to ordinary verification and acceptance. An OOM/timeout never becomes a
successful Run or accepted Revision. Terminal failed Runs remain failed on resume.
Existing exact-receipt recovery for an interrupted publication never recomputes
science. Read-only accepted evidence uses its existing authority and need not
launch a worker.

Private execution diagnostics are adjacent to the scientific output directory as
`.OUTPUT-execution.json`; direct command diagnostics use `LOG.execution.json`.
They record the configured budget, verified kernel values, PID/cgroup, systemd
result and bounded native diagnostics. They are operational records, not a second
scientific receipt or authority. `oom-kill` is distinguished from a native Python
allocation exception and from scientific input rejection. Worker logs retain
native errors without inventing scientific conclusions.

The current whole-owner boundary covers annotation input observation, uploaded-reference
qualification, iterative annotation native children, MatrixPackage (SnapATAC2, Scanpy, EpiScanpy,
Signac), chromVAR and Cicero CDS. The existing command runner also applies the
same boundary to repository-controlled native commands. This is not a claim that
every Python-only tool or GPU allocation in Agent is isolated. CUDA device memory
is not governed by cgroup host-memory controls.

Retained constraints have separate purposes: exact hashes/axes/group provenance;
valid sparse encoding and numeric round trips; HDF5 link/filter and bounded
metadata layouts; bounded JSON/control messages; native R seed representation;
fixed supported object ports; no automatic whole-sparse-matrix densification;
and finite upload, private delivery and archive disk budgets. Matrix decoded-byte
and HDF5-chunk caps are not applied inside the controlled scientific invocation.
The ordinary service-side inspection and download owners retain their independent
bounded-read/delivery policies.

See the [SRV-2C execution-safety report](../../../agent-data/acceptance/scientific-results/srv-2c/execution-safety-2026-11-07/REPORT.md)
for actual Ubuntu/systemd verification, tiny native scientific acceptance,
controlled kernel OOM, failure/restart tests and deployment limitations. The
artificial test budgets are evidence values, not universal recommendations.
