# UA3.6.4d — Final repository review and checkpoint size correction

Date: 2026-10-14. The user explicitly authorized final focused review, one commit
and a normal push to `origin/main`. This supersedes the earlier reports' no-commit
instructions for this repository closeout only. **UA3.6.4d remains PARTIAL overall.**
The four preceding milestone reports are preserved verbatim as historical records.
No new training, SEAM, model inference or live-provider request was run.

## Review and baseline

`main`, HEAD, local `origin/main` and actual remote `main` all started at
`ce5b53d7f716b777fc5333ea84e4d868ccf0da71`. The index was empty. The complete tracked
diff and untracked inventory were reviewed; pre-existing `evals/` is excluded.
Process inspection found the current Codex app server and its code-mode host,
with no separate Codex developer or scientific execution process. File hashes
matched C's recorded final source/report hashes before this review's correction.
Subsequent changes were limited to the correction/tests below and this report.

Actual code review covered the registered target-model input/output bindings,
native inference and training adapters, accepted authority consumption, reference
supplementation and continuation, genome promotion/private feature binding,
Session artifact selection and streaming delivery. EpiZoo continues to own the
scientific algorithms. The existing Registry still has 23 tools; no trainer,
Planner, workflow engine or resource database was added. Neutral adaptation does
not require human/mouse cCRE vocabularies. Target inference retains exact feature
membership and records a lossless column permutation when needed; cell order is
preserved. Shared genome identity excludes dataset features and models. Neither
uploaded H5AD nor adapted checkpoint is automatically published globally.

Native omission still resolves through the pinned native dataclass, and explicit
overrides remain in resolved execution arguments, manifests, receipts and
accepted authority. Historical method/profile identities and strict training
recovery remain unchanged. New production training still requires operator
requalification of bundles that pin changed adapter bytes, as recorded in C.
This review neither migrates authority nor changes operator configuration.

## Full-vocabulary checkpoint finding

Both preserved accepted checkpoints exceed C's original 2 GiB ceiling:

| Accepted result | Features | Exact checkpoint bytes | Checkpoint SHA-256 |
| --- | ---: | ---: | --- |
| M14.4 macaque | 615,873 | 4,938,458,542 | `81c33e89e0fd79068139f604b097330ee9fae0704945111a2ffe34c3390dafe5` |
| M14.5 zebrafish | 466,307 | 4,018,926,698 | `1de0a9d7c391a8eba55b307f3318af0a96b65e7d2b01038af23d73f150f1d257` |

Their manifest SHA-256 values remain respectively
`9177a816503b18787a42d7e00e5ff258fd8924b5097621775a5b8dd1b8d82139` and
`fcb309b7d4e78bd8c0fc9ace8f7eb4e3db6b57653c0e200788ff43bb98517caa`.
Exact paths, stat sizes and transport observations are in the external review
record `genuine-transport.json`.

The checkpoint-only ceiling is now **5 GiB (5,368,709,120 bytes)**, the smallest
whole-GiB ceiling covering both publications. The corresponding safe error says
5 GiB. Matrix, annotation, graph, table and analysis ceilings remain 1 GiB;
metadata remains bounded to 8 MiB. The shared two-transfer limit, exact disk
reservations plus 64 MiB margin, no-symlink opening, source stability and SHA-256
checks, anonymous snapshots and cleanup are unchanged. No architecture change,
new file server or permanent model copy was needed.

Both complete genuine files passed the production snapshot and chunk iterator:
all bytes matched their accepted hashes, reads were at most 1 MiB, and reservations
and transfer counts returned to zero. M14.4 took 5.75 seconds and M14.5 4.71 seconds;
process peak RSS was 658,928 KiB. This is direct transport qualification with
existing accepted files, not a new Session/browser or historical authority
migration. C's native Firefox acceptance of the 1,262,255,722-byte checkpoint
remains the browser evidence. No checkpoint was deserialized for these checks.
The downloads still require exact vocabulary/corpus-DF companions and compatible
inference software; they are not a standalone inference package.

## Focused validation and preservation

External review evidence is at
`/home/likeyi/program/agent-acceptance/ua3.6.4d-commit-review`.

- Model-delivery selection: **25 passed, 2 warnings in 28.65 seconds**. Includes
  genuine-size and exact-5-GiB reservation admission, one-byte-over rejection,
  insufficient disk, source-open cleanup, preserved other-role limits, bounded
  reads, authority/lineage rejection and disconnect cleanup. Test model payloads
  are explicitly simulated; genuine transport is recorded separately above.
- Existing matrix/processed streaming and HTTP selection: **56 passed, 3 warnings
  in 132.42 seconds**. All 525 source/test/static files were unchanged during it.
- Added model-specific HTTP 413 coverage uses the existing safe-error test; its
  seven parameter cases pass. No attachment headers or private paths leak on error.
- All selections removed `RUN_*` gates, used no `-k` exclusions, and ran without
  concurrent source/test edits. Exact commands and logs are external. No full
  regression or long scientific workflow was launched.

The previous 958-test run predates the disclosed final B fixes. Its later 156,
15 and 110-test selections qualify those fixes; C's 119 tests and additional
lineage test remain recorded evidence. Final C source/report hashes matched at
review entry. Only model role limits and their tests changed here. The inherited
MAESTRO annotation fixture failure remains reproduced on committed baseline,
not repaired or excluded to manufacture an all-green milestone claim.

Preservation checks confirm all 981 inventoried eval files, all 35 Work A and
102 Work B workspace files, and 14 selected historical scientific/resource files
retain their byte hashes. All 2,874 inventoried files across M14.4, M14.5 and the
four prior UA3.6.4d acceptance trees retain size/mtime. The two large checkpoint
hashes were checked in full above. This is a bounded audit, not a new byte audit
of every historical RunStore. Provider ledgers, credentials, operator configs,
original pretrained and SEAM resources were not edited. One initial preservation
script used relative B paths without the workspace prefix; that read failed,
was corrected, and the complete preservation check passed.

EpiZoo remains clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`;
Agent's `tools/models/epizoo.py` is unchanged. Git whitespace checks pass. The
approved commit message is:

`Complete UA3.6.4d cross-species scientific workflow integration`

Only the exact files below belong to this commit. No eval, dataset, genome,
checkpoint, acceptance workspace, provider payload, configuration or credential
is included. Commit/push identity and final remote alignment are reported after
publication, outside this immutable pre-commit review record.

## Remaining scope

Bounded genuine science and software integration succeeded; the overall workflow
remains **PARTIAL**. Live-model planning/resource clarification, general H5AD
representation flexibility, broad biological convergence and product usability
remain carryover work. Short technical training does not establish biological
adaptation performance. No UA3.6.5 or integration-simplification work begins here.

## Exact committed file inventory

```text
docs/ua3.6.4d-commit-review.md
docs/ua3.6.4d-epizoo-cross-species-user-workflow.md
docs/ua3.6.4d-final-closeout.md
docs/ua3.6.4d-posttraining-workflow-acceptance.md
docs/ua3.6.4d-reference-supplementation-acceptance.md
src/agent/application/analysis_delivery.py
src/agent/application/dialogue_execution.py
src/agent/application/interactive.py
src/agent/application/interactive_schemas.py
src/agent/application/local_resources.py
src/agent/application/matrix_delivery.py
src/agent/application/model_delivery.py
src/agent/application/reference_supplementation.py
src/agent/application/resource_continuation.py
src/agent/application/responses.py
src/agent/application/sessions.py
src/agent/application/shared_genome.py
src/agent/application/turn_decisions.py
src/agent/application/turns.py
src/agent/application/uploads.py
src/agent/orchestration/error_policy.py
src/agent/orchestration/planning_scope.py
src/agent/orchestration/registry.py
src/agent/orchestration/species_adaptation_registry.py
src/agent/orchestration/target_model_binding.py
src/agent/orchestration/verifier.py
src/agent/report/analysis_report.py
src/agent/report/evidence.py
src/agent/tools/analysis/embedding_analysis.py
src/agent/tools/analysis/epizoo_embedding.py
src/agent/tools/analysis/neighbors_delivery.py
src/agent/tools/models/epizoo_adaptation/backend.py
src/agent/tools/models/epizoo_adaptation/contract.py
src/agent/tools/models/epizoo_adaptation/publication.py
src/agent/tools/models/epizoo_adaptation/resources.py
src/agent/tools/models/epizoo_target_inference.py
src/agent/tools/models/native_posttraining.py
src/agent/tools/models/species_adaptation.py
src/agent/web/app.py
src/agent/web/config.py
src/agent/web/static/app.js
src/agent/web/static/index.html
tests/application/test_h5ad_composition.py
tests/application/test_initial_scientific_declarations.py
tests/application/test_interactive_schemas.py
tests/application/test_new_target_composition.py
tests/application/test_turn_decisions.py
tests/epizoo_adaptation/test_native_configuration.py
tests/species_adaptation_integration/test_model_delivery.py
tests/species_adaptation_integration/test_public.py
tests/tools/analysis/test_compact_analysis_delivery.py
tests/tools/analysis/test_embedding_analysis.py
tests/tools/models/test_epizoo_target_inference.py
tests/tools/models/test_target_model_binding.py
tests/web/test_config.py
tests/web/test_reference_supplementation.py
tests/web/test_scientific_artifacts.py
tests/web/test_target_resource_continuation.py
tests/web/test_unified_scientific_result_delivery.py
```
