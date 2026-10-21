# M16.1–M16.3 milestone regression closeout

## A. Baseline and repository state

Agent HEAD and local `origin/main` remain
`c94135deaf23029b93cb530d7cbce3f5d397d83d` on `main`. Nothing was staged at entry
or exit. No fetch, commit or push was performed. Registry remains 23. EpiZoo
remains clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.

The accepted uncommitted M16.3 work at entry was exactly:

- `src/agent/application/__init__.py`
- `src/agent/application/dialogue_evidence.py`
- `src/agent/application/scientific_dialogue.py`
- `src/agent/application/sessions.py`
- `src/agent/application/turn_decisions.py`
- `tests/application/test_dialogue_detail.py` (untracked)
- `tests/authority_dag/test_dialogue_detail_authority.py` (untracked)
- `docs/m16.3-grounded-detail-comparison.md` (untracked)

Existing untracked `evals/` was preserved. All 127 files match the earlier M16.3
preservation inventory. Both regression runs preserved all 569 snapshotted files:
187 source files, 188 test files, 55 documents, 127 eval files and 12 prior M16.3
diagnostic files. This includes all 364 source/test Python files. Other scientific
source, orchestration, reporting and authority-schema files have no M16 diff
against the pre-M16 checkpoint `431a875b2b8f987a5e3272b22d4db206d2d3f4d8`.
Preservation claims are scoped to that inventory and the inspected Git trees,
not an exhaustive hash audit of every external dataset/checkpoint.

## B. Regression strategy

First reran the accepted M16 focused gate. After it passed, ran the established
full lightweight repository suite exactly once, with every `RUN_*` variable
absent, no exclusions and no deselection. The full suite covers all requested
application, session, authority, evidence, reporting, Planner/compiler and
representative scientific execution areas. Separate overlapping broad subsets
were unnecessary. No expensive opt-in real-data acceptance or live API call ran.

## C–H. Existing system compatibility

| Requested area | Result and evidence |
| --- | --- |
| C. M15 sessions | Pass. Session tests: 35; real prior-output tests: 38; intent-turn tests: 18; response tests: 8. Persistence, immutable revisions, generations, navigation, rollback/branch/continuation, stale descendants, active/historical references and fresh-process recovery remain valid. |
| D. Turn admission/routing | Pass. Turn-decision tests: 18; response-contract tests: 20; application suite: 240 total. Execute, Navigate, Clarify, operational Answer and scientific Answer preserve their distinct paths. Explicit new science executes through the existing Planner/Application; interpretation cannot silently execute it. |
| E. Planner/compiler | Pass. All 894 orchestration unit tests, 145 provider-adapter unit tests and 115 benchmark tests pass. Catalog, semantic compiler, planning, prior-output binding, lifecycle and provider contracts remain compatible. These are deterministic tests, not live planning benchmarks. |
| F. Scientific tools | Pass. Raw intake 172; BAM fragments 226; external fragments 113; QC 112; selection 121; canonical matrix 25 plus matrix integration 45; annotation integration 24; adaptation 86 plus public species-adaptation integration 31. Additional scientific tests are included in the full totals below. No science was reimplemented or requalified biologically. |
| G. Authority/verification/lineage | Pass. All 208 authority-DAG tests, plus relevant verifier/lifecycle tests, pass. Accepted evidence, retained cross-run results, publication/source identity, prior-output reuse, closure and corruption rejection remain intact. |
| H. Evidence/reports | Pass. All 84 report tests, plus integration evidence/report/visualization and application tests, pass. Deterministic evidence/report generation remains independent of dialogue; interpretation does not rebuild presentation. |

Counts in the evidence descriptions overlap by design; they are not added to the
repository total.

## I–K. M16 deterministic acceptance

- **I. M16.1:** 31 application evidence tests and 2 real-owner authority tests pass.
  Exact source/output access, multiple capabilities, active/historical and retained
  references, missing/corrupt evidence, unsupported contracts and zero reconstruction
  remain covered.
- **J. M16.2:** 55 dedicated dialogue/closeout/intent-context/provider-contract and
  authority tests pass, alongside shared turn/response/session tests. Subject
  resolution, canonicalization, grounded claims, reviewed meanings, ambiguity,
  restart/predecessor/concurrency, stale evidence and execution routing remain valid.
- **K. M16.3:** 21 detail/comparison application tests and 2 real QC authority tests
  pass. Exact publication binding, digests, size limits, explicit omissions,
  annotation rationale, non-annotation detail, compatible operands, unsupported
  correspondence and mutation rejection remain valid.

These 111 dedicated M16 tests plus 73 shared session/turn/response tests form the
184-test focused gate. They also all occur in the full regression.

## L. Scientific no-work accounting

Evidence/dialogue fixtures prohibit registry execution, runtime run/resume,
executor entry, owner verification and evidence/report construction. Real owner
DAG tests instrument production, owner reconstruction, integrity and presentation;
interpretation/detail/comparison and navigation retain their zero-work assertions.
The existing multi-turn M15 test interleaves explicit selection/matrix execution
with operational answers and restart, distinguishing expected execution calls
from read-only answers. M16 tests separately verify that new-science requests use
the existing execution path and cannot be accepted as already-computed evidence.

No new reannotation, marker discovery, statistical testing or unintended execution
occurs in interpretation. The full scientific-tool suite intentionally executes
small scientific fixtures; zero-work claims apply to interpretation operations,
not to those explicit execution tests.

## M. Architecture and complexity

The architecture remains the existing M15 session/revision/interaction path,
M16 accepted evidence access and generic dialogue, with optional bounded published
detail and compatible comparisons. Scientific execution remains Planner/Application
→ registered owner → verified result.

The three M16 core modules retain distinct responsibilities: `dialogue_evidence`
for accepted read-only access; `scientific_dialogue` for exact discussion references,
context and claim rendering; `dialogue_execution` for admission and delegation to
the existing execution path. M16.3 adds no core module. Scientific result-shape
adapters and reviewed compatibility rules are not capability-specific conversational
routers. Provider names/models do not branch generic dialogue semantics.

There is no second session engine, dialogue engine, authority, persisted evidence
or interpretation store, or duplicate scientific state. Exact locators and digests
in different boundary views refer to the same accepted result rather than create
parallel identity authorities. No correctness or unnecessary-complexity defect
justified refactoring. No refactor was performed.

## N. Commands and exact results

```bash
/home/likeyi/anaconda3/bin/conda run --no-capture-output -n agent \
  python outputs/m16-regression-closeout/regression.py focused
/home/likeyi/anaconda3/bin/conda run --no-capture-output -n agent \
  python outputs/m16-regression-closeout/regression.py full
```

The child command is `/home/likeyi/anaconda3/envs/agent/bin/python -m pytest -q`
followed by the focused files or `tests`, and a phase-specific `--junitxml` path.
The focused file list is the accepted 12-file list in the M16.3 report and is
recorded verbatim in `focused-start.json`.

Both runs set `PYTHONPATH=src`, `NUMBA_CACHE_DIR=/tmp/agent-numba-cache`,
`MPLCONFIGDIR=/tmp/agent-matplotlib-cache`, and `PYTHONDONTWRITEBYTECODE=1`.
All `RUN_*` variables were absent at entry and remained absent in pytest.

| Run | Passed | Skipped | Failed/errors | Warnings | Pytest duration | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Focused M16/shared gate | 184 | 0 | 0 | 0 | 70.76 s | 0 |
| Full lightweight repository | 4,591 | 83 | 0 | 7 | 2,974.29 s | 0 |

Wrapper wall times were 73.055 s and 2,976.750 s respectively. The focused tests
are contained in the full run; these totals must not be added.

Full-suite area counts (all failures/errors zero):

| Test directory | Passed | Skipped |
| --- | ---: | ---: |
| annotation_integration | 24 | 0 |
| application | 240 | 0 |
| authority_dag | 208 | 0 |
| bam_fragments | 226 | 0 |
| barcode_qc | 112 | 0 |
| benchmarks | 115 | 0 |
| cell_by_ccre | 25 | 1 |
| cell_selection | 121 | 0 |
| chromap | 289 | 27 |
| epizoo_adaptation | 86 | 0 |
| external_fragments | 113 | 0 |
| external_matrix | 40 | 0 |
| fragment_feature_integration | 35 | 0 |
| fragment_features | 74 | 0 |
| integration | 38 | 3 |
| matrix_integration | 45 | 0 |
| primary_contigs | 12 | 0 |
| raw_intake | 172 | 0 |
| regulatory_features | 63 | 0 |
| report | 84 | 0 |
| species_adaptation_integration | 31 | 0 |
| tools | 1,399 | 52 |
| unit/orchestration | 894 | 0 |
| unit/providers | 145 | 0 |

The 83 skips are explicit opt-ins: matrix stress 1, Chromap qualification 27,
live providers 3, edgeR backend/verifier 48, full mouse reference 1, and EpiZoo
integration/checkpoint/parity 3. No additional exclusion was introduced.
The seven warnings are the established dependency deprecations, TBB compatibility
warning and intentional duplicate-observation-name negative fixtures.
`git diff --check` passes.

Logs, JUnit XML, phase commands, preservation hashes and area counts are retained
under ignored `outputs/m16-regression-closeout/`.

## O–R. Failures, changes, provider status and limits

**O. Failures:** None. No genuine M16 regression, unrelated failing test or stale
expectation required classification or repair. Warnings/skips are accounted above.

**P. Closeout changes:** This new report only, plus ignored local regression
scripts/logs/XML. All accepted production files, tests and prior M16.3 documentation
remain byte-identical to the start of closeout. Nothing is staged.

**Q. Provider status:** Zero new live-provider calls. No provider contract changed,
so another smoke was unnecessary. The recorded M16.2/M16.3
`PROVIDER_LANGUAGE_QUALITY_LIMITATION` remains separate from passing deterministic
Agent correctness. This does not newly qualify live end-user prose quality.

**R. Remaining limits:** The accepted M16.3 bounded detail adapters, subject/record
bounds and constrained comparison rules are unchanged. Unsupported biological
inference, inferred cluster correspondence and new statistics remain unsupported.
Model prose remains non-authoritative. Skipped expensive workflows were not newly
qualified; no preserved real-data acceptance was rerun. No M16.4 or later work began.

## S. Commit-readiness recommendation

The requested deterministic and repository-level gates pass. Recommend committing
and pushing the reviewed M16.3 changes and this closeout record when authorized,
without including existing local `evals/` or ignored diagnostics. This task did not
stage, commit or push anything.

Explicit answers:

1. M15 session/revision regression introduced? **No regression found.**
2. Scientific execution semantics changed? **No.**
3. Scientific owner boundaries changed? **No.**
4. Authority/verification/lineage semantics changed? **No.**
5. Second session/dialogue architecture introduced? **No.**
6. Capability-specific conversational routing introduced? **No.**
7. Provider-specific workaround in generic Agent code? **No.**
8. Interpretation/detail/comparison read-only with respect to scientific truth? **Yes.**
9. New-science requests use the existing scientific execution path? **Yes.**
10. Architecture remains minimal and clear? **Yes.**
11. Recorded live-provider issues are non-blocking provider-quality limitations? **Yes, for this deterministic architecture closeout; live prose quality remains unqualified.**
12. Ready to commit and push M16.3 closeout? **Yes; neither action was performed.**

**M16.1–M16.3 regression passed; M16.3 is ready to commit and push.**
