# Post-M11.7b.3a — Application verification audit and minimal authority reuse

Starting baseline: clean `main`, `40a0a0d436b54bb4a6ba50b294447dcb2222e51a`.
This bounded change consumes the accepted
[Post-M11.7b.2 authority architecture](post-m11.7b.2-scientific-authority-propagation.md).
It does not qualify legacy runs or establish real-data acceptance.

## Audit and implementation

Before this change, new durable scientific execution already reused upstream
scientific authority correctly. Application `_complete` ran after that executor
context had ended, so every reporting call took the standalone scientific path:

```text
ResearchAgentApplication.run / resume
  → AgentRuntime (execution or recovery)
  → _complete, under composition lease
    → evidence build / verification
      → verify_run (structural consistency only)
      → topological verify_step → scientific owner verifiers
      → explicit fact projections → verify_public_result → owner verifiers again
    → visualization capability / build / verification
      → verify_analysis_evidence → same scientific paths
    → report build / verification, including final application verification
      → _source_snapshots → verified evidence / visualization → same paths
      → deterministic report / manifest / copied-figure validation
```

The duplicated QC metrics, Selection decisions, Matrix counts/axes and producer
reconstruction were redundant when the application store held compatible accepted
authority. Evidence identity, resolved arguments, dependency bindings, registry
compatibility, snapshot checks and deterministic presentation validation are distinct
facts and remain necessary. `verify_run` never reconstructed science.

After this change, the same call graph executes inside one explicit
`accepted_authorities(application.run_store, run_id)` scope, opened under the
composition lease only after successful runtime completion:

```text
successful runtime result + trusted accepted RunStore
  → load compatible schema-2 authority, validate physical closure and execution
  → unchanged evidence / visualization / report calls
    → unchanged public-result and owner wrappers
      → validate authority, integrity, lineage, resources and runtime compatibility
      → reuse covered scientific proof
    → independently validate application / presentation facts
  → accepted-anchor validation and scope exit
  → application success
```

No consumer bypasses its existing verifier. No authority, receipt, cache, registry,
scientific parameter, schema family or lifecycle state is added. No scientific
owner implementation, Planner or compiler changes. Authority entry/exit failures
produce sanitized `APP_EVIDENCE_FAILED`; a final scope failure returns no report
reference. Interruptions still propagate and the operation context is reset.

## Boundaries retained

- The loader accepts authority only through the trusted local RunStore and exact
  successful execution anchors. Result objects or manifest bytes alone grant none.
- Existing .2 checks bind manifest bytes, artifact payloads, sidecars, publication
  receipts, arguments, dependencies, reference/resources, profiles, verifier versions
  and producer qualification scopes. Generic fragment integrity never grants FASTQ,
  BAM or external qualification.
- Missing authority takes independent deep verification within the operation.
  Successful fallback proof is local; it is never persisted as a legacy upgrade.
  Incompatible or corrupt recorded authority fails rather than falling back.
- Standalone evidence/visualization/report APIs have no implicit store or context.
  Their default scientific verification remains deep. Operator code can explicitly
  use the existing trusted loader with those APIs.
- Runtime resume, completed-step revalidation and receipt recovery finish before
  application authority reuse starts. Their strict behavior is unchanged. Terminal
  application resume validates compatible authority and presentation again without
  mutating accepted runtime state or rerunning production.
- PLAN_ONLY, unsuccessful and cancelled runtime results never enter composition.
  Publication ordering and cooperative scientific cancellation are unchanged;
  reporting-stage cancellation remains deferred.
- Historical raw-source identity follows the .2 policy. Ordinary scientific reuse
  does not claim current raw-source freshness; archived sources remain historical
  evidence. Explicit `current_source_freshness.v1` remains an operator audit.
  An explicit `inspect_raw_scATAC` plan step still performs its own existing bounded
  source-aware verification: it has no .2 scientific authority and is not skipped.
  Other processed-H5AD/DA verification is likewise unchanged.

The six affected scientific evidence bases now say `independent_*` rather than
`fresh_independent_*`. They describe the independent proof underlying the artifact,
which can be reconstructed now or consumed through validated authority. Fresh
step/evidence/presentation checks still run. This deterministic wording is identical
for strict and reused verification; it does not introduce a mode-dependent payload.
Historical source digests describe verified source bytes, not a new freshness audit.
Existing evidence with the old basis strings fails exact projection comparison;
there is no silent rewriting or migration. The preserved legacy run is untouched.

## Validation

Focused tests use tiny BAM/external DAG fixtures and the existing synthetic FASTQ
backend fixture, including a nonempty FASTQ matrix. Instrumentation counts actual
owner bodies for strict and reused verification, and covers all three producers,
QC, Selection, Matrix and canonical fragment integrity. Normal application execution
requires one pass per owner; terminal resume adds zero. Standalone report verification
still enters all deep owners. Archived raw sources remain usable only through the
trusted historical-authority path. Tests also cover missing-authority fallback,
corrupt payload/receipt/reference/authority/evidence/report, failed scope completion,
interruption cleanup, and immutable terminal persistence.

All commands run from the repository root with:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
export NUMBA_CACHE_DIR=/tmp/agent-numba-cache
export MPLCONFIGDIR=/tmp/agent-matplotlib-cache
```

No `RUN_*` gates were enabled for the focused commands:

```bash
conda run --no-capture-output -n agent python -m pytest -q \
  tests/authority_dag/test_application.py -x
# 20 passed in 252.45 seconds; exit 0.

conda run --no-capture-output -n agent python -m pytest -q \
  tests/chromap/test_authority_dag.py -k fastq_application -x
# 1 passed, 2 deselected in 7.78 seconds; exit 0.

conda run --no-capture-output -n agent python -m pytest -q \
  tests/bam_fragments/test_integration.py \
  tests/external_fragments/test_integration.py \
  -k 'application_execute_report_resume_and_corruption and source'
# 2 passed, 69 deselected in 6.87 seconds; exit 0.
```

The first new-test attempt exposed a misspelled instrumentation target and stopped
before application execution. The corrected focused run above is the acceptance
result. The FASTQ fixture simulates the backend; it is not a real producer workload.

The bounded domain regression explicitly removes all `RUN_*` gates in-process:

```bash
conda run --no-capture-output -n agent python -c \
  'import os, sys, pytest; [os.environ.pop(k) for k in tuple(os.environ) if k.startswith("RUN_")]; sys.exit(pytest.main(sys.argv[1:]))' \
  -q tests/application tests/report tests/unit/orchestration \
  tests/integration/test_milestone7_application.py \
  tests/integration/test_milestone7_analysis_evidence.py \
  tests/integration/test_milestone7_analysis_report.py \
  tests/integration/test_milestone7_analysis_visualization.py \
  tests/raw_intake/reporting \
  tests/chromap/test_fragments_evidence.py \
  tests/chromap/test_fragments_report.py \
  tests/chromap/test_fragments_application.py \
  tests/chromap/test_fragments_orchestration.py \
  tests/bam_fragments/test_integration.py \
  tests/external_fragments/test_integration.py \
  tests/barcode_qc/test_integration.py \
  tests/cell_selection/test_integration.py tests/matrix_integration \
  > /tmp/agent-post-m117b3a-domain.log 2>&1
```

Domain result: **1,295 passed, 3 skipped, 3 existing warnings in 517.60 seconds
(8m 37s), exit 0.** Warnings are the existing louvain/pkg_resources deprecations
and Numba/TBB version warning. The three real FASTQ backend cases stayed gated.
Source and test files were unchanged during and after this domain run. Only
acceptance documentation was finalized afterward. No full repository suite ran.

## Review file inventory

| File | Reason |
| --- | --- |
| `src/agent/application/service.py` | Open existing trusted authority scope after runtime completion; sanitize scope failures and withhold a report on failure. |
| `src/agent/report/evidence.py` | Describe independent scientific proof without asserting reconstruction on each consumption. |
| `src/agent/report/fragments.py` | Apply that wording to FASTQ artifact projections. |
| `src/agent/report/bam_fragments.py` | Apply that wording to BAM artifact projections. |
| `src/agent/report/external_fragments.py` | Apply that wording to external artifact projections. |
| `src/agent/report/barcode_qc.py` | Apply that wording to QC sidecars. |
| `src/agent/report/cell_selection.py` | Apply that wording to Selection sidecars. |
| `tests/authority_dag/test_application.py` | Instrument normal/standalone paths, fallback, corruption and interrupted/failed completion with tiny BAM/external DAGs. |
| `tests/chromap/test_authority_dag.py` | Add synthetic FASTQ application/terminal-resume coverage with a nonempty matrix. |
| `tests/bam_fragments/test_integration.py` | Distinguish historical application reuse from explicit current-source freshness. |
| `tests/external_fragments/test_integration.py` | Cover the same boundary for external adoption. |
| `README.md` | Link the new current application behavior. |
| `AGENTS.md` | Record its engineering/compatibility boundary. |
| `docs/post-m11.7b.3a-application-authority-reuse.md` | Preserve this audit, call graph, scope and acceptance evidence. |

## Deferred scope

Legacy authority qualification and preserved-run application closeout, real FASTQ/BAM
workloads, biological validation, broad reporting/evidence redesign, visualization
expansion, new preprocessing and performance work unrelated to scientific reuse
remain deferred. Existing .2 follow-ups (dependency-subset loading, further resource
qualification deduplication, stronger filesystem snapshot isolation) remain deferred.
This is not an arbitrary verification-call-count target or the final .3 closeout;
no full repository regression, commit or push is part of this step.


## Final repository state

`git status` retains `main` at the starting HEAD, aligned with `origin/main`, with
12 modified tracked files and two new untracked files (this record and the new
application authority test). No files are staged. `git diff --check` passes.
`git diff --stat` describes tracked changes; the two new files are listed separately
by status and in the inventory above. No commit or push was performed.
