# Work Unit A.1 — UMAP / Leiden decoupling

2026-10-14. **Complete within the bounded software scope.** The complexity gate
passed: the existing semantic ports and accepted-neighbors authority support
the corrected dependency without a new engine, schema migration or Planner
redesign. UA3.6.4a/b/c/d product acceptance remains **PARTIAL**.

## Baseline and preservation

Starting and ending HEAD: `10ff1844146c6371e38d54473be34401f1646b87`.
The initial working tree contained the unstaged Boolean-admission change in
`src/agent/tools/models/epizoo.py`, its tests in
`tests/tools/models/test_epizoo.py`, the untracked
[Work Unit A record](ua3.6.4a-scientific-integration-simplification.md), and
pre-existing untracked `evals/`. All three Work Unit A files retain their exact
starting SHA-256 values. All 1,274 eval files retain their names, sizes and
modification times (metadata comparison, not a fresh content-hash audit).
Historical results and operator configuration were not modified. Native EpiZoo
remains clean. Nothing was staged, committed or pushed.

## Dependency and compatible representation

The old owner, Registry, verifier and presentation path required
`embeddings → neighbors → Leiden → UMAP`. Native Scanpy UMAP consumes the
neighbor graph and representation, not Leiden labels. The supported graph is now:

```text
EpiZoo embeddings → neighbors ┬→ Leiden
                            └→ UMAP
```

The existing `neighbors → Leiden → UMAP` route remains supported when the
selected source includes clustering. Both routes call the same `sc.tl.umap`
with unchanged seed, min_dist, spread, dimensionality and initialization.
Neighbors are not recomputed. A small real Scanpy test produces exactly equal
coordinates with and without labels on the same graph.

Only new label-free outputs add `clustering_present=False` to the lightweight
result and existing artifact provenance; their safe export records the same
marker. Absence of this marker retains the historical clustered contract and
still requires labels. Missing labels therefore do not silently convert a
damaged historical artifact into an unclustered one. Contradictory labels or
clustering provenance fail closed. Required result fields, semantic output type
`umap_analysis.v1`, recovery policy `compute-cell-umap-v1`, authority formats and
the existing safe-export contract are retained. No historical data is migrated.

## Exact changes

| File | Local change |
| --- | --- |
| `src/agent/tools/analysis/embedding_analysis.py` | UMAP admits `neighbors` and `clustering` stages; validates the explicit label-free variant and preserves the existing clustered result. |
| `src/agent/orchestration/registry.py` | UMAP's input admits both `cell_neighbors.v1` and `clustered_analysis.v1`; corresponding artifact metadata and descriptions expose the real dependency. Validates the optional false marker. |
| `src/agent/orchestration/verifier.py` | Checks the actual source stage, optional-label declaration, exact cell order/embeddings/labels, source graph alignment and parameter lineage. No upstream reconstruction or numerical UMAP reimplementation. |
| `src/agent/report/evidence.py` | Projects the new marker only when present, leaving historical fact projections unchanged. |
| `src/agent/application/dialogue_evidence.py` | Checks the marker against the accepted result; reports clustering as not performed for the new variant; exposes exact qualified neighbor identity/consumption under `umap_neighbors_*`. Existing clustering attribution remains intact. |
| `src/agent/report/visualization.py` | Adds `umap_coordinates`: single-color points, no cluster legend or invented labels. Existing `umap_leiden` rendering/digest behavior remains. |
| `src/agent/report/analysis_report.py` | Adds the coordinate-only caption and explicit clustering-presence fact; label-free reports use “Neighbors and UMAP” even without a figure. |
| `src/agent/application/analysis_delivery.py` | Exports exact cells, EpiZoo embeddings and UMAP coordinates with optional real Leiden labels; validates the variant against the accepted download metadata. |
| `src/agent/application/interactive_schemas.py` | Adds the exact reviewed “Embeddings and UMAP — safe export” descriptor. |
| `src/agent/web/static/app.js` | Admits that one additional download descriptor in the existing browser allowlist; no UI redesign. |

Tests added: `tests/tools/analysis/test_umap_variants.py` and
`tests/authority_dag/test_umap_independence.py`. Focused existing tests updated:
`tests/authority_dag/test_processed_prior_outputs.py` (new compatible consumer,
retaining an incompatible-consumer rejection), `tests/report/test_analysis_report.py`
(figureless label-free report), and `tests/web/test_static.py` (download descriptor).
This document is the only new documentation file for A.1.

The Registry change supplies normal LLM planning, compilation and preflight
with the new channel. Scripted semantic decisions exercise same-plan UMAP-only,
independent sibling Leiden/UMAP, and exact prior-neighbors continuation. No
language rules or automatic workflow completion were added. The legacy
`DeterministicPlanner` bundled-workflow template is unchanged; it is not the
profile-based application planning path exercised here.

## Verification and historical compatibility

Small simulated 32-cell, 512-dimensional embeddings drive real CPU Scanpy work.
No GPU inference, biological dataset workflow, provider completion, full repository
suite or historical browser session was rerun. Tests distinguish software
qualification from live-provider intent quality and biological acceptance.

| Executed selection | Result |
| --- | --- |
| Initial existing owner/export/semantic metadata checks | 69 passed, 3 warnings, 10.40 s. |
| New owner variant/mismatch checks | 13 passed, 3 warnings, 9.64 s. |
| Focused combined owner, prior-output, dialogue, compiler, verifier, visualization, report and static-browser selection | 272 passed, 2 failed, 3 warnings, 82.88 s. Failures classified below. |
| Exact starting-HEAD reproduction of the legacy recovery fixture | 1 failed, 3 warnings, 7.86 s; same pre-UMAP authority failure. |
| Final affected report/application/updated consumer-channel checks | 36 passed, 3 warnings, 31.36 s. |
| Final pinned-export and both application variants after adding the export marker | 8 passed, 3 warnings, 31.01 s. |

These overlapping counts are not summed. All passing invocations exited 0.
The combined run's obsolete test expecting neighbors → UMAP rejection was
updated and passes in the final affected run. Its other failure is
`test_terminal_resume_reporting_is_fresh_zero_execution_and_state_immutable`
in `tests/integration/test_milestone7_analysis_visualization.py`. That fixture
returns a standalone neighbors publication from a replacement callable inside
a new durable execution. The existing owner rejects binding its publication to
that execution (`VERIFICATION_AUTHORITY_INVALID`, step `neighbors`), before
clustering or UMAP. An isolated `git archive` of exact starting HEAD reproduces
the same failure. This pre-existing fixture remains unchanged; it is not counted
as a passing regression or repaired by weakening authority.

New application tests use ordinary production owners and cover successful
same-plan and prior-output compilation, verified publication, exact evidence,
real PNG generation, pinned safe downloads, Session/Revision restoration and
fresh-process terminal recovery for both variants. Recovery invokes zero
neighbors/Leiden/UMAP production. Original accepted neighbor artifacts and the
new clustered/unclustered output/report/PNG bytes remain unchanged across reads
and recovery. Existing Leiden-only and legacy-authority tests remain covered.
No new historical authority for clustering or UMAP outputs is invented; reuse
continues within the existing supported authority contracts.

Mismatch tests reject missing graphs, wrong stages, missing required labels,
invented labels/clustering provenance, altered representations, cell-order
substitution, changed graph weights and tampered accepted source publications.
Checks of graph alignment preserve native behavior: UMAP's
`simplicial_set_embedding` zeroes weak connectivity edges in place. An initial
exact-output-graph assertion exposed this; the verifier instead requires exact
distances and unchanged weights for retained connectivity edges. It does not
reimplement optimization or change coordinates/graph construction. This is
structural/source-alignment verification, not independent numerical reproduction.

Commands used the agent environment Python with `-B`, `PYTHONPATH=src:.`,
`PYTHONDONTWRITEBYTECODE=1`, `pytest -q -p no:cacheprovider` and isolated basetemps.
The combined run removed all `RUN_*` environment gates. Exact combined argv,
logs, baseline reproduction and preservation metadata are in the temporary
local directory `/tmp/agent-umap-decoupling-5MRU42/`. Warnings are the existing
louvain deprecations and unavailable TBB threading layer. `git diff --check`
passes. Final follow-up checks were limited to the subsequently affected paths.

**A.1 can close; Work Unit B can begin as a separately scoped task.** The known
baseline fixture failure is recorded above. Live-LLM/product acceptance and
cross-Revision scientific comparison remain UA3.6.5 work; all four UA3.6.4
scenarios remain PARTIAL. EpiZoo vocabulary/order/model requirements, arbitrary
peak compatibility, native EpiZoo and Scanpy algorithms are unchanged.
