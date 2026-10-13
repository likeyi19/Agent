# UA3.6.4b — Real raw scATAC-seq preprocessing user workflow acceptance

Assessment date: **2026-10-13, Asia/Shanghai**. Overall qualification is
**BLOCKED — resource/deployment and existing public-scope prerequisites**.
Real source transfer, fail-closed Web admission, and focused software checks pass.
No new real raw-to-fragments-to-QC-to-selection-to-matrix execution or live-model
semantic success is claimed. No production defect requiring a fix was demonstrated.

The task evidence directory is `/tmp/agent-ua364b-9i5vbt6k`. Historical M11 and
M14 acceptance remains distinct from these new observations. The user's follow-up
confirms that no additional independently qualified BAM or human/mouse external
fragments are available and directs bounded investigation of existing records.

## A. Baseline and preservation

HEAD, local `main`, local `origin/main`, and remote `main` are all
`8d7ecf8842834758b56df3893a17f696e929e889`, **Improve UA3.6.4a scientific dialogue
and evidence binding**. Remote main was read twice using `git ls-remote`; no fetch
was performed. The expected UA3.6.4a changes, including the polling fix, are
committed. Initial tracked worktree and index are clean; only existing `evals/`
is untracked. There are 636 tracked files and 1,274 existing evaluation files
(981 are listed by Git as nonignored untracked files).

`baseline.json`, `protected-before.json`, `additional-protected-before.json`,
`preservation-final.json`, and `file-set-final.json` retain the audit. Comparison
covers 84,335 primary snapshot entries and 1,787 additional source entries, with
zero changes. Small primary files are SHA-256 checked through 2 MiB; larger files
use size, mode and nanosecond mtime. Additional source files through 8 MiB are
hashed. This is a bounded preservation check, not a fresh byte audit of every
large dataset or checkpoint. Historical resource/acceptance/Web/evaluation roots
also have zero added or removed files. Uploaded copies receive separate exact
byte checks because those bytes are consumed by this task.

Production operator configuration, original Sessions/Runs/Revisions, scientific
payloads, authority, previous acceptance evidence and EpiZoo remain unchanged.
EpiZoo is clean at `029cd631d0f9806a646c4a3b42ce10b958f2b67f`.
Only this repository report is added. Nothing is staged, committed or pushed;
no reset, clean, fetch or pull is performed.

## B. Real resource readiness

The production operator file remains
`/home/likeyi/program/agent-web-config/operator-web.json`. It currently has two
InputSets, no raw-family companions, and a default per-file upload limit of
**268,435,456 bytes**. The existing supported maximum is **1,073,741,824 bytes**;
neither limit was increased. Missing configured companions alone do not imply
missing scientific algorithms.

| Family | Available evidence and feasible level | Missing prerequisite for the requested new-upload workflow |
| --- | --- | --- |
| FASTQ | Genuine M11.7 PBMC Next GEM library: eight gzip files, two complete A-layout groups, explicit R1/R2-barcode/R3/I1 roles; historical complete synchronization and qualification for 52,413,244 read pairs. Full human reference, whitelist/context, Chromap and index resources exist. Historical UA3.4.3 demonstrated complete upload and bounded inspection under an explicitly configured 1 GiB limit. | Six of eight files exceed today's configured 256 MiB limit. All fit the existing absolute 1 GiB maximum, but this task does not change deployment limits. A newly uploaded complete library also needs its own exact intake/group/location-bound library context. Historical context is not transferred. Current readiness: blocked upload/production; historically qualified intake/production at original locations. |
| BAM | Existing paired-ATAC scientific owners and software fixtures are available. M14.9 is synthetic/data-layer qualification, not real BAM acceptance. | No independently qualified real paired-ATAC source with exact reference, corrected-CB history and uploaded-source context was found in the reviewed locations. Current readiness: software-qualified, lacking real source admission. |
| External fragments | Genuine M14.8 macaque source, primary preparation, exact TBI and adoption, plus M14.11 generic QC/selection/matrix resources exist. The exact prepared pair fits current upload limits and was newly registered successfully. | The registered raw import retains human/hg38 and mouse/mm10 reference contracts. Neutral primary preparation/adoption is a data-layer capability, with no corresponding registered Web raw producer. Its old preparation pointer cannot silently bind a relocated upload. No separately qualified human/mouse external source declaration was found. Current readiness: real registration and accepted artifact inputs; complete new-upload production blocked by public scope/source binding. |

The PBMC source total is **4,693,218,895 bytes**; the largest member is
835,954,322 bytes. Historical raw sizes match, but complete FASTQ decoding and
hashing were not repeated. The historical library-context, human parent and QC
manifest hashes match their frozen request. The QC catalog remains
`9d9ea0340640fbab74f5f591831592ad3badf5973a09fcb988a7e2cba10e5eae`;
the pinned Chromap executable and full reference indexes remain present.
`readiness-evidence.json` records exact paths, identity checks and runtime evidence.
Approximately 47.6 GB disk was free at audit. No alignment or repeated large
producer execution was initiated to reproduce existing science.

The FASTQ size finding is a current deployment constraint, not an absolute
transport limitation: the previously reviewed 1 GiB configuration can accommodate
this complete library. Raising the isolated configuration alone would still not
qualify its new source-bound context or justify repeating full-scale alignment.

The wider bounded search also found 1,777 tiny BAMs under Descart's Dictys data
directory. The associated notebook explicitly creates them from an accessibility
matrix, with artificial 100,000-base reference lengths and synthetic alignments.
A bounded sample is readable, but these are not qualified raw paired-ATAC sources.
Other research-directory fragment files have no located Agent source-specific
qualification; five-column appearance does not establish assembly, corrected
barcodes, Tn5 adjustment or read-pair support semantics. They were not relabelled
as qualified inputs. No replacement dataset or reference was downloaded.

### Exact M14.8 and M14.11 distinction

Five read-only checks in `macaque_admission_audit.py` pass. Six historical
input/output manifest hashes match; all three M14.11 bounded downstream results
satisfy their current public result contracts. The existing `ScientificInputSet`
accepts the exact M14.11 artifact inputs without inventing a new resource registry.
These checks read bounded manifests and validate types, not scientific payloads.

| Existing accepted object | Exact manifest SHA-256 |
| --- | --- |
| M14.8 admitted primary fragments | `57a5a57caffc6ddaab8aa6f3308d8ec8d1400db4c2bd3b0d14d9b77569e04854` |
| Neutral feature reference | `4eaa2d7c513a82784836230667945e875faa913c939b805498d53cc385a6c2b2` |
| M14.11 qualified QC reference | `5a06cf01a41b884b23ca625fa027079c91997389435c7ba7c6fd269195cf3aab` |
| M14.11 QC output | `9ced021b00152058842a8f796f2654dd97d2a5e24ebf71a13ae8fa7edae62c22` |
| M14.11 selection output | `bf49b6e448cd6d76881ae8c6c9b6fb458014f095421d4b430500030476885bc5` |
| M14.11 matrix output | `d779e067b685b19071c762e8d7052cc49fef3f27aa933a39cc2b86da378efa15` |

The actual loader used by public raw import rejects the neutral reference with
`REFERENCE_CONTRACT_INVALID`, before resource reconstruction or production.
The public import's result validator also rejects the existing neutral result.
This matches the explicit scope in `neutral_fragment_import.py`; it is not an
inferred species restriction. The Web fragments companion supports the existing
legacy import/QC/selection/cCRE chain and cannot supply `primary_preparation`.

There is a legitimate distinct entry point: an operator can offer the original
accepted fragment/reference/QC manifest pairs as structured artifact inputs to
the registered downstream tools. M14.11 already proves that scientific route.
It does not establish new-upload adoption or a live conversational Session.
Re-running its approximately 1,048-second historical execution solely to reproduce
those facts would add no raw-upload integration evidence. No historical Run was
manufactured into a new Revision, and no executable prior-output authority was
inferred from paths or matching bytes.

## C. Ordinary user workflow

Native Firefox uploaded the existing research file
`GSM8598620_FT7_11_S200_001.fragments.tsv.gz` (**4,025,758 bytes**), retaining the
explicit registration attribution **scientific provenance unverified**. This
tests actual transport only; its filename supplies no scientific declarations.
The normal composer submitted:

> Please preprocess this scATAC-seq fragments dataset, perform quality control, select cells using specified qualified criteria, construct the cell-by-feature matrix, and summarize the accepted results. Ask for any missing scientific declarations.

Ordinary admission returned HTTP 400 / `FRAGMENTS_RESOURCE_REQUIRED`, with
visible guidance to select approved reference/source declarations or have the
operator configure them. The source remained selected and the request remained
visible in the composer. There was no interpreter call, accepted interaction,
pending scientific clarification, Run or Revision. This is a deployment gate,
not successful conversational clarification/continuation.

The final browser Session is `98e1eb92-1e40-46f9-b2b5-775909277717`, generation
zero. Registration is
`local-4444b5404e10499452e774bfe189ae2260c6a5fb66c1f2710b98388dce50c0e6`.
The selected real PBMC R1 file then returned `UPLOAD_TOO_LARGE` through the
normal upload action and HTTP size boundary; no FASTQ member was published.

After the historical macaque audit, a separate isolated server streamed the exact
accepted M14.8 **121,673,085-byte prepared BGZF plus 813,220-byte TBI** through
the same ordinary upload endpoint. HTTP 201 created
`local-a167762d582c008143d9c1bba75f3967238e51112e2e5483108966fe10a8df39`.
Both copies match the original preparation record's physical pins:

- BGZF: `bde857546f431bb0bb22fe3088f597dfce363bda45f3b7278f5cca56625cc76c`.
- TBI: `6ebcb3d004f10c54acda8f97d5701b26b19903c3ba94cbd6153582e7f0383342`.
- Original preparation record: `e064438cc413b3edc21270e8b17e5653654b485ae08861139f527c96857dd84a`.

`qualified-transport/result.json` and `source-pin-check.json` retain these
bindings. Upload does not independently verify TBI queries, issue scientific
authority, relocate the preparation record or qualify import. The original
source/index association comes from M14.8; this task verifies exact transfer.

## D. Real scientific execution

New real-data scientific execution is **not performed**: the complete requested
source-to-matrix path did not pass prerequisite admission. Browser/real-transfer
probes record zero provider constructions and zero scientific execution or
verification entries. No real Run, scientific result, authority or Revision is
created. Synthetic regression tests do execute their existing scientific owners;
their calls are not counted as real-data acceptance.

Historical evidence remains valid within its recorded scope:

- M11.7/M11.8: `m117b-pbmc1k-nextgem-r1:run`, 263,747 observed barcodes,
  1,119 selected candidates, 1,119 × 1,355,445 int64 CSR, 11,535,823 nonzeros.
  The frozen explicit criteria are depth ≥1,000, TSS ≥4 and flank evidence ≥20.
- M14.8: `m14.8-primary-real:run`, 18,107,459 admitted primary records and a
  331 × 615,873 matrix under its exact explicit cells and neutral reference.
- M14.11: `m14.11-macaque:run`, 3,220 observed/selected candidates and a
  3,220 × 615,873 int64 CSR with 6,913,545 nonzeros. Depth ≥1 and TSS ≥1 are its
  frozen permissive acceptance-only criteria, not calibrated biological defaults.

These are historical results from their original publications, not new UA3.6.4b
measurements. Canonical record counts, exact row/feature order and
`cell_call_method=none` / `cell_call_state=not_assessed` retain their meanings.

## E. Multi-turn analysis

A new accepted preprocessing Revision is unavailable, so a real threshold-change
turn, qualified upstream reuse and downstream matrix update are **not tested**.
No threshold is guessed and no historical artifact is silently promoted into a
new Session's executable prior output. Existing software continuation and binding
cases pass in the focused suite. They do not substitute for the requested live
multi-turn scientific demonstration.

## F. Scientific results and delivery

No new real QC/selection/matrix evidence, scientific report or scientific download
is eligible because no new scientific result was accepted. The blocked Web Session
has empty scientific history. Historical M11.8/M14.11 reports remain preserved;
reportability, cross-turn executable authority and download eligibility remain
separate. Existing synthetic Web tests exercise import/QC/selection/matrix results,
accepted authority, reports and the five already eligible downloads. No new export
route or eligibility exception is added.

## G. Provider qualification and spending

Conda `agent` reports OpenRouter and Groq credentials available using booleans
only. No credential value is printed, copied or persisted. The fresh task ledger
`budget.json` sets ceilings of 18 completions, six per turn, USD 5.00 conservative
reservations and 4,096 output tokens, with dispatch disabled pending readiness.
Actual completions, reservations and reported cost are all **zero**; there are
no unfinished reservations. Providers are forbidden at the real admission probes,
so no SDK request, retry, fallback, semantic repair or model substitution occurs.

The initial native checks clone the production Groq-only profile configuration;
the visible Groq selection is not a model invocation. The later isolated
`qualified-transport/operator-isolated.json` reuses both accepted profiles from
UA3.6.4a, with `openai/gpt-5.6-sol` on OpenRouter as default and Groq available.
Ordinary `/api/v1/models` confirms both. The real operator file is unchanged.
Live Interpreter, capability selection, Planner, output selection and Answerer
quality are all **not tested** at this blocked stage.

## H. Firefox and recovery

Final native Firefox acceptance passes **13 checks**: nine initial checks and
four after stopping and replacing the server. Actual file selection/upload,
missing-resource feedback, upload-size rejection, refresh, exact source identity,
Session reopen and zero provider/science activity are covered. This is transport
and admission recovery, not accepted-science recovery. Evidence, screenshots,
fresh profile and exact commands are in `native-final/`.

The first harness incorrectly expected size feedback before clicking Upload and
timed out. The current product correctly enforces that bound at HTTP admission.
The original failed log and `native-initial-failure.json` remain; the corrected
harness clicks Upload and checks the existing rejection. Production code was not
changed. Final screenshots were inspected. All task servers and Firefox processes
were stopped.

## I. Code, regression and architecture

No source or test file changes. The immutable Registry remains **23 tools**.
Reviewed ownership spans `local_resources.py`, `uploads.py`, Web configuration/
submission, Registry semantic ports, the existing compiler, public producer
wrappers and their independent owners. The neutral data-layer adapters are not
planner tools. No second Planner, registry, conversation parser, Session system,
scientific algorithm or authority is introduced.

Focused regression: **304 passed, six skipped in 131.08 seconds**, exit 0.
Command uses the `agent` Python with `PYTHONPATH=src:.`, bytecode disabled, all
`RUN_*` gates removed, private caches and isolated temporary output:

```bash
python -B -m pytest -q -p no:cacheprovider \
  tests/application/test_fastq_composition.py \
  tests/application/test_fastq_resources.py \
  tests/application/test_bam_composition.py \
  tests/application/test_fragments_composition.py \
  tests/application/test_resource_choice_identity.py \
  tests/web/test_fastq_execution.py \
  tests/web/test_bam_execution.py \
  tests/web/test_external_fragments_execution.py \
  tests/web/test_qualified_resource_handoff.py \
  tests/web/test_fragments_selection.py \
  --basetemp=/tmp/agent-ua364b-9i5vbt6k/pytest-tmp
```

Exact invocation and environment handling are in `run_focused.py` and
`focused-command.json`; result/log are retained separately. Six existing Chromap
opt-in cases remain skipped. No full repository suite or heavyweight opt-in
qualification was run. Source/index mismatch, exact library membership, missing
producer context, forged resource choice, owner qualification, real synthetic
publication and historical retry/reopen checks retain their existing boundaries.
The UA3.6.4a timestamp irregularity did not reproduce and was not investigated.

The shallow source-audit helper initially tried gzip decoding on three plain
TSV/BED files. Its `BadGzipFile` observations remain in `source-shallow-checks.json`;
`source-shallow-correction.json` records the corrected plain-file inspection.
These were audit-helper errors, not scientific-owner findings of corrupt sources.

## J. Final qualification

| Family | UA3.6.4b status | Exact demonstrated scope |
| --- | --- | --- |
| FASTQ | **BLOCKED** | Real original-source readiness audit and current native upload-limit rejection; focused software checks. No newly completed/upload-qualified library or new real preprocessing. |
| BAM | **BLOCKED** | Source provenance investigation and existing software integration checks. No independently qualified real BAM workflow. |
| External fragments | **PARTIAL** | Actual file registration, exact transfer of the historically qualified macaque prepared pair, native fail-closed missing-resource feedback, existing downstream contract checks. New-upload scientific execution remains blocked. |
| Overall | **BLOCKED** | No representative complete real uploaded-source workflow passes all admission gates. Software and transport passes do not make this full scientific acceptance. |

The shortest existing public raw route needs an exactly declared human/hg38 or
mouse/mm10 external source and its compatible qualified companion resources.
For the preserved FASTQ route, deployment must explicitly admit the full file
sizes and qualify the actual uploaded intake/library identities, then assess
alignment cost. Neither entails rewriting historical contexts. Using macaque
for complete raw-upload acceptance would require separately authorized public
neutral-producer integration preserving primary-preparation lineage; adding it
speculatively would exceed this task's fixed public tool/contracts scope.

## K. UA3.6.5 carryover

UA3.6.4a live cross-Revision scientific dialogue remains **PARTIAL**. Its final
`factual_binding / unbound_factual_literal` rejection is unresolved, and the
historical categorical rejection retains its recorded limitations. Raw-source
transport and synthetic preprocessing tests do not resolve dialogue compliance
or arbitrary prose entailment. Carry this item and the unfinished real UA3.6.4b
workflow into final integrated acceptance with their exact scopes.

## L. Recommendation

Independent UA3.6.4c scenario work may proceed on its own qualified resources;
no demonstrated scientific execution defect from this task requires blocking it.
Do not mark UA3.6.4b complete or imply three-family real acceptance. Preserve the
explicit resource/deployment and public-scope prerequisites above. No UA3.6.4c/d
work was started, and no further provider or scientific execution is scheduled.
