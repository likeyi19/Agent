# USI-3A — Qualified scientific resource reuse and natural continuation

2026-10-15 (Asia/Shanghai). **PASS within the bounded canonical FASTQ/BAM resource
resolution, derivation and continuation scope described below.**

## A. Baseline and scope

Initial HEAD: `355d380b8cb8c8df5fa583b2b7c75a157965185e`,
`Complete USI-2 unified scientific input and raw-data handoff`. Tracked tree and
index were clean; `evals/` was already untracked. No commit, push, configuration
activation, external download, checkpoint loading or GPU execution is performed.

Evidence root **R**: `/home/likeyi/agent-usi3a-9lgqwv4j/`, outside Git and historical
acceptance directories. `baseline.json` records tracked-file hashes and bounded
size/mtime inventories for existing evals and operator files. Test sources and
results are generated, explicitly simulated fixtures under R. They establish
software/scientific-contract behavior, not biological qualification.

The implementation covers the existing conversational canonical human/mouse
FASTQ/BAM handoff. Inspection alone still requires no preprocessing reference.
There is no general discovery service, new tool, resource inventory, dependency,
workflow planner, Web redesign or resource-download mechanism.

## B. Focused architectural findings

| Existing owner/component | Finding and reuse |
| --- | --- |
| `ScientificInputSet`, `resource_continuation.catalog` | Existing bounded, operator-admitted configurations and exact definition digests already describe available resources. Configuration is not scientific qualification. No second catalog is needed. |
| `raw_handoff`, typed `request_roles`/`answer_roles` | Already retain the original objective, exact uploaded registrations, library declarations, explicit scientific arguments and pending-turn identity. Default-only resource selection was the missing boundary. |
| `scatac_reference` | Owner-built pinned manifests carry species/assembly, ordered contigs/features and resource identities. Its loader checks manifest integrity and closed scientific contracts without rescanning genome data. Consuming owners independently establish current source eligibility. |
| `scatac_library_context` | Already derives the necessary source-specific library resource from a real verified intake and explicit declarations. Whitelist eligibility belongs here and to FASTQ consumption. Reference choice is independent of library/barcode context. |
| BAM/FASTQ owners | Already validate source/reference/context compatibility and fresh content before transformation. BAM's existing canonical reference bundle is genuinely consumed, including FASTA for applicable sequence-dictionary checks; this work adds no FASTQ, Chromap or BAM-index requirement. |
| `reference_supplementation`, `shared_genome` | Existing H5AD-specific supplementation can generate a managed FAI from an eligible supplied FASTA and derive an ordered private feature reference against an approved shared genome. These are retained, not repurposed as a canonical BAM resource builder. |
| Chromap index owner | Supports explicit qualified index construction; it is not an automatic, cost-free resource search. No speculative alignment/index construction is added. |
| Session/RunStore/Registry/compiler/delivery | Existing execution, authority, lineage, cancellation, source scope and downloadable output paths remain authoritative. |

Three unnecessary couplings are removed for new handoffs:

1. A compatible unique resource no longer needs an operator `*_default` flag,
   nor a redundant InputSet species declaration duplicating the owner manifest.
2. Pending tasks no longer require the entire resource catalog to remain frozen.
   New definitions may be added; every captured definition must remain exact.
3. The library-context preparation identity no longer includes reference choice,
   its label/default policy or unrelated downstream resource configuration. It
   retains exact captured source registrations, raw declarations and the genuine
   FASTQ whitelist dependency.

The existing module owns these changes. Persisted owner manifests are admission
observations in the existing Interaction, not another scientific authority schema.
A small read-only branch preserves USI-2's original default-only interpretation
and original context identities. It does not expose a second execution path.

## C. Implemented behavior

After typed raw-role interpretation, Agent reads pinned reference manifests through
the existing owner loader. Invalid/unavailable manifests and species/assembly
conflicts are ineligible. It does not infer sample species or BAM source assembly
from a filename, reference availability or a configuration label. Missing sample
history, read membership and barcode-processing declarations still clarify.

An explicit offered choice wins only when its contract is valid and compatible.
A recorded choice is retained through further questions; failure never silently
substitutes another candidate. An explicit reply may correct the resource choice
while keeping all scientific source declarations and parameters immutable.

Without an explicit choice, one compatible candidate binds automatically. An
existing unique configured default resolves a choice only among compatible
candidates. Multiple materially distinct candidates otherwise produce a question
with labels, species, assembly and feature counts. Exact equivalent mappings and
whitelists, including the same pinned manifest at another locator, use stable
InputSet-ID ordering; explicit alias selection is still preserved. This policy
never orders distinct scientific references to choose a winner.

Truly unavailable resources retain the existing raw prerequisite and explain that
a qualified compatible reference (and FASTQ whitelist) must be provisioned. A new
operator-admitted definition can then supplement the same pending task. It cannot
replace a captured definition, change uploaded bytes, consume unrelated staged
uploads or erase explicit thresholds. Raw arbitrary-reference upload qualification
is not added; uploaded bytes alone never become a qualified reference.

Library-resource derivation continues through actual intake, independent intake
verification and `build_scatac_library_processing_context`. There is no fabricated
intake and no added scientific execution engine. New `library.v1` preparation
identities permit exact previously admitted contexts from the same Session to be
reused across reference choices. Reuse loads the pinned context and intake,
validates their owner lineage, and freshly verifies the source intake; it does
not rerun preparation intake production or rebuild the context. Normal planned
scientific operations still execute/verify normally. Unaccepted files on disk
are not discovered as reusable proof. Corrupt accepted content fails closed.

## D. Scientific correctness and limits

Manifest selection establishes a valid pinned owner contract and task compatibility;
it is not a claim that every referenced byte, biological origin, whitelist or
runtime has already passed execution qualification. Deep source qualification
remains with the existing consuming owners. Application matching does not stream
whole genomes, reconstruct features, run barcode correction or duplicate those
scientific algorithms. New execution always retains existing integrity and
compatibility checks. Historical reads use captured observations without opening
current resources or re-running qualification.

The resource choice definition, admitted declaration, manifest observations,
source/context receipt and normal Run/Revision provenance remain separate. No
selection observation is emitted as schema-2 scientific authority. Accepted
fragments/QC/selection/matrix continue to receive their normal independently
verified authority and exact lineage. Existing scientific algorithms, Registry,
EpiZoo, MAESTRO, QC thresholds and selection defaults are unchanged.

The configured catalog retains its existing single-lab operator sharing boundary.
No private local registration, other Session, model checkpoint or result is scanned
to enlarge that catalog. Context reuse receives only the current Session's admitted
receipts. Existing source-registration Session checks still reject foreign uploads.
The feature-reference and genome-sharing mechanisms retain their separate explicit
sharing review; they are not silently promoted to raw reference candidates.

Supported derivation here is a source-bound library context, with the existing
H5AD FAI/feature derivation retained in regression scope. It does not invent an
assembly, reconstruct barcode-correction history, generate canonical cCRE resources
from BAM/FASTA, build general references or add automatic expensive Chromap index
preparation. Legacy contexts remain readable at their exact historical identities;
only new `library.v1` contexts participate in the new context-reuse policy.

## E. Acceptance evidence

Final commands, frozen hashes, results and preservation receipts are recorded
under R. All scoped lightweight runs disable `RUN_*` gates; the qualified FASTQ
positive is a separately enabled, bounded synthetic execution.

Development observations:

- Existing BAM chat regression: **8 passed in 23.53 s**.
- Initial new resource tests: **11 passed in 69.32 s**.
- Expanded combination: **27 passed, 2 failed, 1 skipped in 114.92 s**. Both
  failures were old FASTQ expectations for the missing-resource phrase; the
  corrected final message retains that phrase and explicitly identifies resource
  compatibility. These failures are not reported as passes.
- Exact baseline archive: **1 passed in 16.70 s**, producing a real accepted USI-2
  BAM Session. `check_legacy.py` then reads it with current code, an unavailable
  provider factory and forbidden current resource observation: exact persisted
  presentation and Session restoration pass. The first archive command used an
  incorrect working directory, and the first read harness omitted its approved
  upload root; both harness errors were corrected before these results.

The new scripted-provider HTTP vertical covers automatic unique reuse (without
`bam_default` or duplicate configured species), absent resources followed by
additive provisioning/restart, multiple choices followed by an explicit reply,
and an incompatible explicit choice followed by an explicit correction. Every
case follows the production application handoff through all five existing tools:
inspection, BAM fragments, QC, selection and canonical matrix. Each yields a
verified **1 × 2** int64 sparse matrix with total count **2** and the original
objective/explicit zero thresholds. These thresholds are simulated acceptance
parameters, not biological recommendations. All five scientific downloads match
their advertised SHA-256. Ordinary report delivery is covered by existing BAM
and qualified FASTQ positives.

A second complete raw task in the same Session succeeds with preparation intake
production and context building forbidden, proving reuse through admission rather
than only an isolated helper. New planned production and independent verification
still each execute once. Direct reuse across a different reference choice also
preserves exact context bytes; corrupted context reuse fails. A selected reference
whose pinned manifest becomes invalid clarifies without fallback or preparation.
Additional cases cover unknown/incompatible choices, missing essential sample
context, exact aliases, immutable declarations and additive-only catalog changes.
Existing USI-1/USI-2 tests cover unused/foreign uploads, mutation, retry, generation,
Session recovery, scientific handoff and downloads.

Final focused declaration/resource/FASTQ/BAM combination: **81 passed, 1 skipped
in 121.58 s**, exit 0 (`focused-final.log`). The separate qualified FASTQ positive
passes **1 test in 21.99 s**, exit 0 (`qualified-fastq.log`), with real pinned
Chromap and synthetic inputs. All **537 frozen source/test/resource files** remain
unchanged through that qualified run.

The expanded **52-module** scoped regression completes **2,236 passed, 7 skipped,
2 failed, 2 warnings in 986.61 s**, exit 1. All 537 frozen files are unchanged.
There are no case exclusions. The failures are:

- `test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`:
  the already recorded unknown-singleton 404 versus expected 422 assertion.
- `test_definition_identity_captures_every_operator_field_and_is_detached`:
  an additional baseline assertion includes absent `fastq_whitelist=None` in its
  expected digest, whereas the existing definition intentionally omits that absent
  extension to preserve legacy identity. Neither the InputSet implementation nor
  this test was changed by USI-3A.

Both independently reproduce together against the exact baseline archive:
**2 failed in 1.74 s** (`baseline-failures.log`). Neither is repaired or classified
as a regression. The broader suite is not claimed green.

After that frozen run, final review adds one bounded refinement: equivalent
resource aliases are also collapsed when a genuinely distinct candidate is present,
so the clarification lists each scientific choice once. Its explicit preference
and mixed-alias assertions are included in the final affected-domain check; a
scripted invalid-preference witness also now spells out that preference in the
user utterance. No other production behavior changes after the broad run.

The **final code** passes **82 tests in 140.24 s**, exit 0, across raw declaration
contracts, new resource resolution and existing FASTQ/BAM chat integration, with
the qualified FASTQ gate enabled. There are **no skips or exclusions** in this
final selection. All **537 final frozen files** remain unchanged throughout
(`final-frozen.json`, `final-affected-command.json`, `final-affected.xml`,
`final-affected.log`, `final-affected-result.json`). This directly exercises the
last refinement and both scientific pathways; the 52-module run was not repeated.
The baseline Session read is checked again with the final code.

Preservation: all **1,274** pre-existing eval files and the operator configuration
retain their recorded size/mtime, with no added/removed files in those inventories.
All initially tracked files outside the three declared production/test edits keep
their original SHA-256, including scientific owners and EpiZoo. The five-file
change inventory below includes the two new files. HEAD remains the initial
baseline and the index remains empty. These are bounded source-hash and protected
metadata checks, not a claim of rehashing every historical biological dataset.
`preservation-final.json` records the check.

## F. Final assessment and changed files

Assessment: **PASS for the defined bounded USI-3A implementation**. Criteria 1–5
are covered by compatible/invalid/ambiguous/explicit/missing-context cases;
6–8 by actual owner-generated library contexts, exact reuse, refusal to invent
missing references/history and preserved supplementation; 9–10 by original-task
continuation, source pins, independent scientific authority, restart and legacy
reads; 11 by current-Session-only receipt reuse and existing foreign-source/private
reference checks; 12–14 by the real simulated FASTQ/BAM executions, downloadable
outputs, the relevant regression and unchanged Registry/scientific owners.

The bounded implemented scope does not
qualify live-provider interpretation or biological product readiness. No live
provider calls are made in this work unit. **USI-3B.1 can proceed without
additional prerequisite architecture work at this resource/continuation boundary.**
Its external-fragments conversational role and qualification wiring remain its
own work; the existing task/prerequisite and scientific-owner pattern applies.

Changed files:

- `src/agent/application/raw_handoff.py`: qualified manifest matching, captured
  choices, additive continuation, independent context identity and exact reuse,
  bounded historical replay.
- `src/agent/application/turn_decisions.py`: tell the interpreter to express
  explicit resource preferences and leave deterministic resolution to Agent.
- `tests/application/test_fastq_declaration_contract.py`: retire the old blanket
  prohibition on an explicit resource correction; scientific declarations stay
  immutable.
- `tests/web/test_raw_resource_resolution.py`: focused contracts and genuine
  synthetic HTTP/scientific acceptance.
- `docs/usi-3a-qualified-scientific-resources.md`: this report.

Relevant commands (all run from the repository unless a recorded `cwd` says
otherwise):

- `git rev-parse HEAD`, `git status --short`, `git diff --check`, scoped `git diff`,
  and `git diff --cached --name-only` establish baseline, review and index state.
- Targeted `rg --files`, `rg -n`, `cat` and `sed` read README/AGENTS, USI-1/2,
  UA3.6.4d, and the input/resource/owner/execution modules listed above.
- `/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q` with
  `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1` runs the named development/focused
  modules. Commands use isolated `--basetemp` directories under R; corresponding
  logs are `first-tests.log`, `resource-tests.log`, `focused.log` and
  `focused-final.log`; the last focused argv is also in `focused-final-command.json`.
- `git -C /home/likeyi/program/Agent archive HEAD | tar -x -C R/baseline-source`,
  followed by the baseline BAM positive under that archive's `PYTHONPATH`, builds
  the historical compatibility witness. Current-code `python -B R/check_legacy.py`
  performs the read-only comparison (`legacy-build.log`, `legacy-read.log`).
- `python -B R/closeout.py` records and runs exact argv/environment in
  `qualified-fastq-command.json` and `scoped-command.json`. The former enables only
  `RUN_CHROMAP_QUALIFICATION=1` with the unchanged
  `/tmp/m116c-test-build-identity.json`; the latter disables all `RUN_*` gates.
  There are no exclusions within the selected modules. JUnit XML, log and result
  JSON accompany each command. `python -B R/final_affected.py` records/runs the
  final affected-domain command with the same qualified fixture, producing the
  final frozen hashes and result described above.
- The exact baseline archive runs the two named failing node IDs with `python -B
  -m pytest -q`, `PYTHONPATH=src` and isolated `--basetemp=R/baseline-failures`;
  `baseline-failures.log` retains their independent reproduction.
- Python inventory checks compare frozen source SHA-256 and the baseline's
  protected eval/operator size/mtime metadata; this is not a full rehash of
  historical biological datasets. `df -h` confirms ample space for tiny fixtures.

No files are staged, committed or pushed.
