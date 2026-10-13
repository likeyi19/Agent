# UA3.6.4b — Real raw scATAC-seq preprocessing user workflow acceptance

This is a chronological record. The latest assessment is the **Work Unit B**
section below; preceding Work Unit A and blocked-stage findings remain historical.

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


## Work Unit A continuation — 2026-10-13

This section supplements, and does not replace, the historical conclusions above.
**Work Unit A is PARTIAL:** the shared continuation implementation and bounded
software acceptance are distinct from live-provider task completion. The sole
live OpenRouter response used an invalid character span for a researcher-supplied
threshold and correctly failed admission. It produced no scientific Run or
Revision. No favorable-answer retry or semantic repair was attempted. A separate,
visibly scripted-provider Session exercises the real-data scientific path below;
it does not qualify live language understanding. Work Unit B has not begun.

### A. Baseline and preservation

HEAD, local main, origin/main and remote main were aligned at
`4c485c183e19dead17b7c94cdc373f45e2a5d496` (`Simplify unified scientific admission
and raw resource selection`). HTTPS `git ls-remote` and the GitHub connector
confirmed remote main; an initial SSH connection failed. The tracked worktree
and index were clean. The 981 pre-existing nonignored untracked files were all
under `evals/`. No reset, clean, fetch, pull, stage, commit or push was performed.

New evidence, acquired resources, managed uploads, operator configurations and
scientific workspaces live outside Git at
`/home/likeyi/program/ua364b-fragments-aad_21w8` (abbreviated **R** below).
`baseline.json` records repository state. `protected-before.json` captures 86,124
existing files: size/mtime/mode for all and SHA-256 for files at most 2 MiB. This is
bounded preservation evidence, not a new full hash of large scientific assets.

### B. Shared implementation and permanent policy

The existing Session prerequisite gains a typed `resources` field and paired
`request_resources` / `answer_resources` interpreter decisions. The original
objective, registered source/index, explicit declarations, chosen InputSet
identity, generation and predecessor remain captured. Providers receive opaque
choice handles and labels; exact existing operator InputSet values remain private.
No resource registry, pending-task engine, scientific algorithm or Planner was
added. `ScientificInputSet` moved to the application layer with its established
web import preserved. Existing compiler, preflight and scientific owners admit
and verify execution. Existing exact-string/categorical declarations now expose
the external-fragments namespace, source profile and source selection.

A resource answer may complete an existing reviewed InputSet with explicit
spanned declarations. Ordinary clarification retains the pending objective.
Changed source/index, stale generation, altered catalog, forged choice and
conflicting captured declarations fail closed. Compiler-reported missing raw
resources now produce a durable clarification. Catalog changes during a pending
request are rejected rather than silently adopting a replacement definition.

The existing pending role boundary additionally permits explicit append of
registered H5AD members up to the captured required-role count. Captured members,
order, roles and resource choice remain fixed. Focused coverage uploads an
additional file through the HTTP endpoint after a Session objective is pending,
then binds its opaque position through the ordinary role continuation. This is
bounded admission coverage, not full arbitrary-file or scientific-role acceptance.

`AGENTS.md` now records the authorized public-resource acquisition and simulated
fixture policy and the permanent user-centered continuation principle. This does
not change production configuration or relax scientific qualification.

### C. Genuine source and bounded acquisition

The [official 500 PBMC Next GEM dataset](https://www.10xgenomics.com/datasets/500-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-next-gem-v-1-1-1-1-standard-2-0-0)
provides Cell Ranger ATAC 2.0.0 fragments, index and summary. The exact source
headers and summary identify `refdata-cellranger-arc-GRCh38-2020-A-2.0.0`.
The [fragments specification](https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/outputs/fragments-file)
documents zero-based half-open Tn5-adjusted coordinates, corrected barcode/GEM
suffix and read-pair support. Historical alignment and barcode correction remain
producer declarations, not independently reconstructed history.

`resources/acquisition.json` records all five downloads with URL, release,
size, SHA-256, role and date. Cumulative download: **116,976,405 bytes**. The
largest file is **110,797,499 bytes**; both task limits are respected. No published
checksum was found; the multipart ETag was not treated as a checksum. The source
index is 874,165 bytes and summary 2,590,764 bytes. A bounded urllib GET returned
403; a subsequent bounded curl transfer succeeded. Two small 1.2.0 metadata/index
files were also inspected, but that release's actual source assembly is hg19;
no 1.2.0 fragments were downloaded or used. Release versions were not conflated.

The original 2.0.0 fragment file includes 15 scaffold names absent from the
existing qualified UCSC-style reference. No alias repair was performed. Under the
existing supported `subset_export` contract, `prepare_subset.py` prepared a
primary-nuclear subset using the qualified QC reference's complete ordered
classification (24 primary nuclear contigs). Of **10,891,721** source records,
**10,889,126** were retained byte-for-byte and **2,595** scaffold records excluded.
A separate full source/output pass verified every retained record and order.
This is explicitly documented acquisition preparation, not fabricated Agent
producer history or neutral primary-preparation authority.

`resources/subset-provenance.json` binds source, classification and output:
subset BGZF 110,850,431 bytes,
SHA-256 `03a6d40d3a8a934c550fb4a4b86362585d11d8a99835ba73f96683b6fb23ac9a`;
index 879,539 bytes,
SHA-256 `b64a3a7a4fc024fbecaea83810ac081571e8523cd8098f750d47667af34282aa`.
Source namespace is explicitly `pbmc500`; profile is
`10x-atac-fragments.v1`; source selection is `subset_export`.

### D. Qualified reference and researcher criteria

The isolated operator InputSet uses the existing human hg38 parent reference
`agent-resources/parents/human-hg38/m116-r1/reference.json`, SHA-256
`20681078b31446df31c689992191d842d4110435004a5e547efea9696b59d7c4`, and QC
`agent-resources/qc/human-hg38/gencode-49-chr/m116-r1/manifest.json`, SHA-256
`e9037c7a80964495e5b3aaea4aa6d634035d05cf7aaf5a36d8cff27bb21afbd5`.
The existing operator QC catalog and `/usr/bin/bedtools` are used unchanged.
The full canonical ordered **1,355,445-feature** axis is retained. No reference,
QC, selection or matrix scientific algorithm changed; Registry remains 23.

The researcher explicitly authorized exploratory acceptance criteria:
QC fragment records ≥ **1,000**, TSS enrichment ≥ **4**, TSS flank evidence ≥
**20**, then TSS enrichment ≥ **5** with all other thresholds preserved. These
are not calibrated cell-calling, purity or biological-quality standards. No
threshold adjustment is authorized because of unexpectedly few selected cells.

Before execution, `execution-estimate.json` recorded disk/memory and stage costs,
including independent verification. Available disk was approximately 15 TiB under
`/home` and 44 GiB under `/tmp`; CPU execution needs no GPU or dense full matrix.

### E. Live-provider outcome and cumulative budget

The predecessor UA3.6.4b ledger had zero paid calls and zero reserved spending;
`provider/predecessor.json` pins that observation. The inherited task allowance
remains 18 calls, six per turn, USD 5 conservative reservation and 4,096 output
tokens, with SDK retries and fallback disabled. Offline guard tests pass 11 checks.

Native Firefox uploaded the real subset pair and submitted the full scientific
request in Session `1d22c0c2-6d85-4a55-bea6-41868ba2b76a`. The actual model
`openai/gpt-5.6-sol` returned `request_resources` with three explicit thresholds.
Its flank-evidence literal `20` had incorrect offsets `202:204` (the actual
span is `203:205`; the supplied span contains ` 2`); exact argument
binding returned `ungrounded_operand`. The user-visible response requested a
valid explicit value. No pending objective, Run, Revision or science was admitted
for this invalid decision. Exact prompt/schema and provider response are retained
in `provider/request-01.json` and `provider/response-01.json`.

**One paid completion**, **USD 0.132598 reserved**, provider-reported
**USD 0.0303085**; no second attempt, model substitution or deterministic span
repair. This is an observed live operand-grounding failure, not evidence that the full
live workflow passes. The scripted Session below cannot change that conclusion.

### F. Separate scripted-provider, genuine-data acceptance

The second native Firefox Session is
`893096c5-b397-4753-a930-ae40d8714db6`, using isolated
`operator-technical.json` / `technical-workspace` and the visible profile label
**SCRIPTED technical acceptance — not a live model**. The uploaded bytes are the
genuine PBMC subset pair above. Only interpreter/Planner decisions are scripted;
compiler, preflight, scientific production, independent owners, authority,
publication, evidence, reports and Web workers are the production implementations.
`technical_model.py` and `technical-provider.jsonl` preserve all eight scripted
responses. They are not paid calls or evidence of live semantic quality.

The initial resource request preserves the full original objective and the three
researcher thresholds. A subsequent ordinary reference/profile/namespace answer
resolves the offered InputSet, without reupload or internal path/hash entry. The
normal four-step plan imports fragments, computes QC, selects cells and builds
the matrix. QC covers **219,745 observed barcodes**. Every owner accepts schema-2
scientific authority; selection remains `none / not_assessed` for cell calling.

| Criterion / artifact | Initial result | Continued result |
| --- | --- | --- |
| Minimum QC fragment records | 1,000 | 1,000 |
| Minimum TSS enrichment | 4 | 5 |
| Minimum TSS flank evidence | 20 | 20 |
| Actual retained candidates | **568** | **567** |
| H5AD shape | **568 × 1,355,445** | **567 × 1,355,445** |
| CSR stored nonzeros | 6,469,402 | 6,469,014 |
| Total canonical fragment-overlap counts | 8,712,514 | 8,712,126 |
| Zero rows | 0 | 0 |

Initial Revision:
`c83cf34a1449b7dd455b80e245b5c519e740f7a7bff08b52d5b7b149bfeb3c22`.
New child Revision:
`023151006383f4873edecc7c8df8de659d76b3400d629ef883ac226fa39f7b94`.
The continuation is the ordinary utterance **“Set TSS to 5 and rebuild the
matrix”** with a captured, exact parameter delta. It plans only `new_selection`
and `new_matrix`, binding the accepted original fragments and QC through exact
prior-output references. No threshold was adjusted in response to population size.

`inspect_results.py` / `technical-result-checks.json` verify both actual H5ADs:
nonnegative int64 CSR, exact shape/nnz/sum, exact accepted file SHA-256, every
ordered feature against the original cCRE BED, every ordered cell against its
accepted selected-cell table, and matching manifest/axis identities. All **567
shared cells match every sparse value exactly**; no complete matrix was densified.

| Covered production / independent owner | First execution | Threshold continuation |
| --- | --- | --- |
| External fragments | 1 / 1 | 0 / 0 |
| Barcode QC | 1 / 1 | 0 / 0 |
| Cell selection | 1 / 1 | 1 / 1 |
| Cell-by-cCRE matrix | 1 / 1 | 1 / 1 |

`technical-science-observations.json` records code-frame entry/return observations
without replacing scientific functions. These counts describe the eight named
functions, not a claim that qualification and integrity IO were absent. In
particular, initial QC independently reinspects the 3.32 GB existing GENCODE
annotation through the unchanged owner boundary.

Actual first-step times were import **836.94 s**, QC **2,056.47 s**, selection
**19.25 s**, matrix **757.83 s**, including the step's existing qualification and
verification work. **The initial QC estimate of 3–12 minutes was too optimistic;
actual QC took 34.27 minutes under call profiling.** This overrun is retained as a
resource-estimation limitation; it was not hidden by weakening verification or
starting a second QC. Before the continuation, `followup-estimate.json` used the
observed first matrix to estimate selection <1 minute and matrix ≤15 minutes,
with no upstream production/reconstruction. The observed server peak resident set, including delivery, was **1,083,232 KiB**.
No GPU or model checkpoint was used.

Initial matrix SHA-256:
`b380f5adab9d5e480557472d8c00ac5def14985470233d24817739ed78740650`.
Continued matrix SHA-256:
`637525d4d4853fdcb6d3e320d1294982c2488a5b009b67dfff51c30d52375b11`.
Both retain ordered-feature SHA-256
`3d04a9b92096d3ee8fc2bd533d56744647c096a35211a53c662955704c737a60`.

The native observation helper reached its 60-minute wait limit while the first
run was still correctly displayed as “Analysis in progress”. Reopening the same
Session only resumed observation; no submission or scientific step was repeated.
A second helper snapshot captured follow-up completion text before asynchronous
Revision metadata refresh; generation-2 recapture corrected the observation
without resubmission. The early snapshots and helper exceptions remain preserved.
These are explicitly recorded harness limitations, not erased evidence.

### G. Native delivery, navigation and fresh-process recovery

`technical-browser-delivery.json` records five eligible scientific downloads per
Revision: barcode QC, length distribution, decisions, selected cells and the
actual H5AD. All **10 browser transfers returned 200** and matched their offered
accepted SHA-256 and exact byte size. H5AD sizes are **178,679,259** and
**178,682,287 bytes**, respectively. Delivery is original accepted matrix content,
not a placeholder or a locally fabricated download.

Both explicit report handles render readable, revision-specific safe projections
of accepted evidence. These carry the original accepted report digest and remain
distinct from the original Markdown report and from scientific authority.
Evidence, executable prior-output reuse and download eligibility were exercised
through their respective existing boundaries; one was not assumed from another.
The first large-H5AD browser observation hit Marionette's default 30-second script
timeout. A 180-second **observer** timeout allowed the complete transfers above;
no product timeout, verification or download policy was relaxed.

Native controls inspect both Revisions, activate the original, reactivate the new
child, and refresh. Generation progresses **2 → 3 → 4**. Six exact conversation
texts and both Revisions remain available. Covered scientific counts are unchanged
through reports, evidence, downloads, navigation and refresh.

The first server was then stopped cleanly. `server_readonly.py` starts the same
application/workspace/profile with a provider factory that raises if construction
is attempted. Native reopen, both Revision/evidence/artifact listings, the report
and refresh pass with **zero covered science calls**, unchanged scripted-provider
history and no provider construction. `technical-recovery.json` and
`technical-recovery.png` record generation 4, both Revisions and the same six
conversation texts.

`workspace-before-restart.json` / `workspace-after-restart.json` match all **57
files, 582,717,305 bytes** exactly. `technical-preservation.json` additionally
confirms the original scientific Run bytes and original Revision are unchanged,
and the new Revision's parent is exactly the original. Terminal recovery did not
rebuild science, invent authority or alter accepted artifacts.

The principal review artifacts in R are the two operator configurations,
`resources/acquisition.json`, `resources/subset-provenance.json`, provider request/
response/ledger files, `technical-provider.jsonl`, science observations, result
checks, browser delivery/recovery records, preservation snapshots and test logs.
Native screenshots include the resource question, accepted matrix views,
`technical-delivery.png` and `technical-recovery.png`.

### H. Focused validation and exact failures

All invocations use the `agent` Python, `PYTHONPATH=src:.`, disabled bytecode,
private caches/basetemps and no pytest cache. The recorded runner scripts remove
all `RUN_*` gates. No full-repository suite or opt-in scientific qualification ran.

- Provider budget guard: **11 passed in 1.28 s** (`provider-tests.log`).
- Intermediate shared continuation/role tests: **38 passed in 32.24 s**.
- Intermediate fragments/parameters/role scope: **129 passed in 242.83 s**
  after updating old missing-resource failure expectations.
- Broader 25-file application/web/Registry/prior-output/intent regression:
  **744 passed, 8 failed, 6 skipped, 3 warnings in 2,174.70 s**, exit 1
  (`regression.log`, `regression-command.json`). Seven failures were obsolete
  expectations: five missing-resource terminal failures and two tests excluding
  the now explicitly reviewed namespace declaration. They were updated while
  preserving malformed identity-pair and unreviewed-string rejection.
- The eighth failure was the gzip layout-A inspection fixture reporting
  `INTERACTIVE_APPLICATION_FAILED` with a planning projection and no scientific
  steps. Its exact isolated rerun passed **1 test in 3.18 s**, and the case also
  passed in the final affected-file suite. The initiating exception is not exposed
  by that fixture; its cause remains unqualified. Do not label it a proven timing
  defect or claim the initial broader run passed. Its state and failed log remain.
- Final seven affected-file suite: **280 passed, 6 skipped in 228.31 s**, exit 0
  (`final.log`, `final-command.json`, `final-result.json`). All **489** frozen
  source/test Python files retained their SHA-256 during this final run. This
  includes resource continuation, actual additional HTTP upload, scalar handoff,
  persistence tamper rejection, external-fragments execution, existing BAM/FASTQ
  compatibility, pending roles and scientific declaration contracts. The six
  existing opt-in Chromap cases remain skipped.

Two new test-harness errors were corrected without production workarounds: an
upload fixture initially chose a directory outside its managed workspace, and a
scripted Planner initially referred to an unavailable namespace input instead of
omitting it. The corrected tests passed and are included in the final suite.
These fixtures are explicitly simulated software acceptance, not real biology.

### I. Preservation and review state

`preservation-result.json` compares the protected baseline, excluding only the
explicit repository edits. Before final documentation, **86,106** protected files
were checked, including **85,484** small-file SHA-256 comparisons, with **zero
unexpected changes**. The final comparison checks **86,105** files with **85,483** small-file
hashes and again finds **zero unexpected changes**; the appended report is the
additional explicitly excluded repository edit. Final Git checks are recorded
alongside that result. Existing operator configuration, prior UA ledgers, historical evals,
scientific assets and EpiZoo remain preserved. `git diff --check` passes and the
Git index remains empty. Datasets and checkpoints were not added to Git.

### J. Remaining boundaries and qualification

| Boundary | Demonstrated scope / limitation |
| --- | --- |
| Registered fragments plus an existing reviewed resource choice | Typed pending request and exact InputSet answer can resume the captured objective through normal scientific execution; native scripted real-data scope is reported separately above. |
| Explicit scalar/categorical declarations | Existing spanned argument binding is reused. Invalid spans fail closed. An invalid initial decision does not establish a pending objective; the live failure exposed that remaining user-experience gap. |
| Multiple clarification turns | An ordinary uncertainty answer retains the resource prerequisite; reference/profile choice followed by a separate namespace answer is covered. Profile/namespace declarations in the resource answer are covered. Arbitrary staged categorical/resource completion after selecting an incomplete InputSet remains unqualified; do not claim a general prerequisite solver. |
| Additional uploaded file | HTTP upload during an existing Session and explicit append to a pending H5AD role group are covered. Initial groups still need at least two registered files; arbitrary zero/one-source missing-file continuation and arbitrary file types are not qualified. |
| Missing/unqualified operator resource | The objective is preserved with an understandable provisioning request. No scientific execution occurs. Newly changing a captured catalog does not automatically repair the pending task; catalog-refresh continuation is deferred. |
| User BED/cCRE | No new Web BED resource boundary was added. Follow-up needs bounded upload, explicit species/assembly and reference binding, existing reference-builder validation/qualification, and an opaque resulting resource choice. Default qualified canonical cCRE suffices here. |
| Source provenance | Exact admitted subset conservation and producer-profile checks do not reconstruct historical alignment, duplicate marking or barcode correction. |
| Biological claims | Exploratory thresholds, retained cell counts and verified count matrices do not establish cell purity, calibrated calling or general biological quality. |
| FASTQ / BAM | Existing shared-admission compatibility tests only. Work Unit B and new real-family acceptance were not started. |

**Do not mark UA3.6.4b complete.** Preserve the earlier FASTQ/BAM conclusions and
this live-provider failure when planning subsequent work. The shortest next
product gap is reliable typed clarification with exact operand grounding and
retention of an actionable objective when the initial decision is invalid; this
must not be addressed by hidden language repair or threshold guessing.

### K. UA3.6.5 carryover and deployment

UA3.6.4a cross-Revision live scientific dialogue remains **PARTIAL**, including
its recorded `factual_binding / unbound_factual_literal` rejection. This task does
not resolve factual prose entailment or its historical categorical limitation.
No unrelated timestamp investigation or 23-tool architecture audit was undertaken.

No new production default was installed. The exact existing qualified reference
identities above and the source-specific reviewed InputSet are sufficient for
this isolated acceptance; deploying an analogous choice requires an explicit
operator InputSet in the desired deployment, respecting each new source's
qualification. The acquired PBMC source is acceptance data, not a universal
default or authority template. All implementation and documentation changes are
unstaged; no commit or push was performed.

## Work Unit B — Shared literal grounding and bounded FASTQ/BAM workflows

Assessment date: **2026-10-13, Asia/Shanghai**. This section supplements, and does
not rewrite, Work Unit A or the earlier blocked acceptance. Evidence root **B** is
`/home/likeyi/program/ua364b-work-b-uu8i0zpu`. The bounded synthetic software
workflows pass; **overall UA3.6.4b remains PARTIAL**, with genuine FASTQ/BAM
production-data and complete live-provider execution qualification distinguished
below. No UA3.6.4c/d work, commit or push is performed.

### A. Exact baseline and preservation

HEAD, main, origin/main and remote main all begin at
`fdd3008cab76591fac8769e5f55507dc8c77006f`, **Add shared resource continuation and
fragments workflow evidence**. The preceding `4c485c1` changes and Work Unit A
are present. `git ls-remote` checks the remote without fetching. The initial
index and tracked worktree are clean; only historical `evals/` is untracked.
`baseline.json` and `protected-before.json` capture 102,967 existing files,
including the entire Work Unit A tree and the previously recorded protected
resource/operator/evaluation roots. Preservation checks use exact SHA-256 through
2 MiB and size/mode/nanosecond mtime above that bound; this is not a new full
byte audit of large biological inputs or checkpoints.

Production changes are limited to `turn_decisions.py`, `dialogue_execution.py`
and `turns.py`. Two new application test modules cover grounding and continuation.
This report is the sole documentation change. Registry, scientific owners,
EpiZoo, production operator configuration and historical scientific records are
not edited. Final preservation checks cover **102,963** protected files, including **102,325**
small-file SHA-256 checks, with **zero unexpected changes**. Work Unit A has no
added or removed files. Final refs still match the starting commit and the index
is empty. Exact checks are in `preservation-final.json` and `git-final.json`.

### B. Scientific parameter grounding

The retained original Work Unit A response is replayed **offline**, unchanged.
Its `20` declaration still fails the unchanged strict span admission at
`202:204`. The new provider-entry boundary locates its unique exact literal at
`203:205`; all three explicit values bind. `offline-original.json` records both
outcomes and the original response-file digest. This does not change the original
failed live turn or count replay as a new live success.

The provider schema now requests `literal` and a verbatim `evidence` clause,
not character arithmetic. Agent requires exactly one occurrence of the evidence
in the utterance and exactly one literal within that evidence. It then creates
the existing strict `ScopedArgument` span. Repeated values can be associated by
distinct exact clauses; unresolved repetitions clarify. Existing well-formed
legacy span responses remain readable. A mismatched legacy span is normalized
only at fresh provider ingestion and only when its exact literal is unique.
Persisted records are never relocated or repaired on loading.

All subsequent token-boundary, numeric grammar, exact ratio, string quotation,
Unicode, choice, Registry identity/type/range, input-conflict and unique-consumer
checks remain in force. New test cases cover negative signs, decimals, embedded
numbers, scientific notation, repeated values/clauses, explicit correction text,
Unicode strings, missing/forged evidence and incompatible wire fields. Existing
negative scientific/source/authority tests remain applicable.

This is textual grounding, not a deterministic natural-language entailment proof.
The LLM still owns whether a clause is a declaration about this parameter, whether
an explicit correction supersedes an earlier value, and whether an unrelated
mention is irrelevant. No parameter-name parser, keyword association, first-match
fallback or fabricated value is added. Conflicting repeated parameter identities
are rejected structurally; arbitrary unreported semantic conflict remains part
of provider qualification, not something claimed solved by string matching.

### C. Missing information and continuation

A structurally valid fresh decision with an unresolved declaration for one
registered consumer can create the existing **parameters** prerequisite without
planning or science. Valid declarations and original inputs are retained. The
rejected wire is stored as diagnostic Session data and replay-checked; it is not
scientific authority. The corrective turn supplies exact current-turn evidence,
then follows ordinary planning and scientific admission. Restart, invalid numeric
correction, duplicate-identity rejection and checksum-recomputed tampering are
covered. Existing pending answers remain atomic and keep their predecessor.

The new mechanism is bounded: it does not salvage arbitrary malformed decisions,
unregistered parameters, contradictory received inputs, or simultaneous rejected
declarations for several independent consumers. It is not a general solver for
arbitrary interleavings of resource, species, role and parameter questions. In
particular, completion of one prerequisite does not claim that an independently
missing resource catalog can be provisioned or refreshed automatically.

FASTQ and BAM both use Work Unit A's `request_resources` / `answer_resources`
contract. The new software demonstrations preserve the uploaded identities,
original full objective and all three explicit selection declarations while the
user selects the offered source-bound InputSet. Required parameters/species,
missing/ambiguous references, explicit choices, unique defaults, source-profile
requirements and incompatible contexts are exercised by the affected regression
and existing shared owners; unsupported provenance is not inferred from format.

For FASTQ, the initial collection lacks **R3 genomic read 2**. Completion returns
`FASTQ_COLLECTION_INCOMPLETE` with lane/chunk/role guidance; all uploaded members
remain registered. Uploading R3 and resubmitting those exact member IDs completes
the library. No replacement or inherited PBMC qualification occurs. This is
additional-file continuation of **library assembly**. A pre-completion collection
is not yet a selectable chat input, so this does **not** qualify a durable chat
objective submitted before any complete library exists. The normal supported
sequence remains upload/group/complete, then submit the analysis request. Existing
pending H5AD-role append behavior is separate; arbitrary raw-file append to an
already captured scientific objective is not introduced.

### D. FASTQ workflow

`workflows.py` generates five synchronized paired templates from the existing
seed-112, 12,000-base `chrTiny` fixture. Layout A explicitly supplies R1 genomic
read 1, R2 barcode, R3 genomic read 2 and I1 sample index. The two positions and
multiplicities are `(100,200,3)` and `(2100,2200,2)`, with an explicit matching
barcode whitelist. These are valid FASTQ records, not placeholder input. The
existing A/B, gzip/plain, lane/chunk and membership negative cases also run.

The source-bound library context is newly built against the **uploaded locators**
through the existing intake/library owners. A fresh tiny Chromap index uses the
existing qualified patched executable and build identity; no full genome index
is built or downloaded. The generated reference, feature BED and transcript-TSS
resource use explicit isolated **test** qualification. Their human/hg38 fields
are test contract values, not a claim that `chrTiny` is biological hg38.

The ordinary upload/completion and Session HTTP interfaces request resources,
accept the user's exact choice, then run inspection → qualified FASTQ fragments
→ QC → selection → cell-by-cCRE matrix. Provider decisions are scripted for this
software execution; the compiler, actual Chromap, all scientific owners,
independent verification, authority, reporting and delivery are real.

The explicit exploratory **test policy** is minimum depth **0**, minimum TSS
**0.0**, minimum flank evidence **1**. It comes from the original user utterance,
not a universal default or an operator-injected threshold. Two fragments and one
observed barcode yield one selected candidate and a **1 × 1 int64 CSR**, one
stored nonzero, count sum **1**. `matrix.h5ad` is **31,240 bytes**, SHA-256
`088225a654108ab98c71330a5136e957646738204ec0c85e756c1cca63e30ce8`.

### E. BAM workflow

The existing generator writes a valid BAM with two aligned paired templates
(central and flank), consistent mate flags/coordinates/CIGARs and generated CB
tags, against two 6,001-base synthetic contigs. The reference includes both
ordered features. Uploaded bytes receive their own intake and source-bound
library context. The test declares the existing corrected-CB paired-ATAC profile
within this simulated construction; it does not certify historical barcode
correction of real sequencing. BAM inspection is separately qualified and does
not confer producer eligibility. Missing history/profile/reference, incorrect
context and source substitution retain their existing rejection boundaries.

The same HTTP resource question/answer and five-step chain executes with the same
explicit test thresholds. Two templates yield two fragments, one observed/selected
barcode and a **1 × 2 int64 CSR**, one nonzero, count sum **2**. The complete
ordered feature axis retains the zero column. The H5AD is **31,240 bytes**, SHA-256
`480a827d8606ff4360e74991700d701a127f25da8ff528420632d5886210d959`.
No new genuine biological BAM with independently reviewed source history was
acquired; this is software qualification only.

For both families, `workflow-*.json`, `provider-scripted-*.json`, `config-*.json`
and `matrices.json` preserve exact source registrations, declarations, resolved
arguments, results, Session/Revision identity and sparse values/axes. Each isolated
workspace uses Session ID `analysis`, generation 1; identities must be interpreted
within their respective workspace, not merged because a textual ID matches.
Both generated Revision IDs are
`b2bdacc2c340d22c32034e50df3806edd47efe400d6533bd08e721188aaf0ab3` because these
isolated fixtures use the same logical turn/output identities. Their separate
workspaces, source pins, scientific authorities and matrix hashes remain distinct;
this is not evidence of scientific correspondence or interchangeable artifacts.

### F. Fragments carryover and real semantic observations

Work Unit A's genuine PBMC subset, both Revisions, 568/567 selected candidates,
full canonical H5ADs, authority, reports, downloads and provider ledger remain
historical evidence. No import/QC/matrix stage is repeated on those data.

Five new real GPT-5.6 Sol completions use the unchanged OpenRouter profile/adapter:

| Observation | Result | Scope |
| --- | --- | --- |
| Original fragments request | `request_resources`, matrix target; exact `1000 / 4 / 20`, with `20` at `203:205` | Fresh real interpreter success for the demonstrated grounding defect; no Planner/science |
| Ordinary simulated FASTQ request | `request_resources`, matrix target; exact `0 / 0.0 / 1` | Real intent/parameter interpretation on captured ordinary application context |
| Ordinary simulated BAM request | Same supported resource request and exact values | Same bounded interpretation scope |
| FASTQ resource answer | Exact offered resource handle | Real answer interpretation over a scripted-created pending application context |
| BAM resource answer | Exact offered resource handle | Same bounded answer scope |

The answer observations are **not** continuations of the preceding live initial
turns: their exact pending contexts come from the separately recorded software
workflows. The scripted workflows choose the producer target whereas the live
initial decisions choose the final matrix target. No response is rewritten to
hide that distinction. All raw prompts/schemas/responses and the five decisions
are retained. This does not qualify a complete live Interpreter → Planner →
science → conversational-result chain. There are no semantic retries or silent
fallbacks. The earlier live failure remains part of the qualification record.

### G. Resources and acquisition

**Zero new external downloads.** `fixture-inventory.json` lists every one of the
31 generated/derived non-workspace fixture files (143,450 bytes total), with exact
size/SHA-256, generator, test identity and qualification scope. The two accepted
workspaces additionally retain exact uploaded copies, scientific manifests,
authorities and outputs. Seed, roles, reference/coordinate and barcode semantics
are recorded above and in the inventory/generator. No simulated barcode or
alignment history is promoted to production qualification.

Existing resources reused read-only are the qualified Chromap executable/build
record, its stock qualification executable, BGZF/tabix tools and `/usr/bin/bedtools`.
`/tmp/m116c-test-build-identity.json` pins source commit, patch and executable
identities; candidate SHA-256 is
`7ba402d0d69d08fedd525a4de83a4dd749670e88d4de507c236b07f03a285649`.
Scientific compatibility is established by the existing fixture/reference/runtime
owners, not inferred from the filenames. The synthetic QC allowance is explicit
and confined to the isolated processes. No resource becomes a production default;
production deployment would require an independently qualified real reference
and a newly source-bound library InputSet under the existing operator contract.

### H. Regression, browser and exact failures

- Initial binding/contract check: **163 passed, 1 failed in 1.47 s**; the sole
  new-test fixture error is described below.
- Shared parameter/binding/safety/continuation regression: **247 passed in
  158.31 s**. Initial new repair-specific check: **2 passed in 7.84 s**.
- Explicitly enabled tiny qualified FASTQ plus BAM downstream production:
  **2 passed in 61.82 s**. The existing Chromap qualification build record is
  supplied; this is bounded technical execution, not a broad opt-in suite.
- Main directly affected 12-file regression: **392 passed, 6 skipped, 3 warnings
  in 670.11 s**, exit 0. All **491** frozen source/test Python files are unchanged
  during the run. All `RUN_*` gates are removed. The six skips are existing
  opt-in Chromap cases; the positive downstream execution above and the two
  separate workflow demonstrations genuinely run the qualified owner.
- Final targeted safety/contract/pending-role regression after adding the existing
  guidance-candidate bypass guard to the rejection-capture path: **261 passed
  in 64.43 s**, exit 0. It includes the additional exact-string continuation,
  categorical-answer and candidate-guard cases. No full repository suite ran.

Exact commands, logs and frozen hashes are retained in `regression.py`,
`regression-command.json`, `regression-result.json`, `final-safety.log` and
`final-safety-frozen.json`. The main regression includes scientific parameter
contracts/safety/continuation, both raw-family composition and Web execution,
resource continuation, scientific Web binding and conversational authority reuse.
The final safety command is recorded in `final-safety-result.json`.

Native Firefox in **fresh server processes** passes 22 checks, 11 per family:
reopen exact history/generation/Revision, read accepted matrix evidence, download
each of five artifacts with matching SHA-256/size, render the report, refresh,
and verify zero provider constructions/scientific body entries. This is native
read/delivery/restart acceptance; workflow creation/upload was through real HTTP
ASGI application endpoints, not native mouse-driven input. `native-results.json`,
`native-*.png` and the report texts retain evidence. All pre-restart workspace
files preserve exact bytes (**53 FASTQ, 38 BAM**). Both task servers and Firefox are stopped afterwards.

The newly generated FASTQ workflow completed science, downloads and result saving
before the helper attempted `dataclasses.asdict` on an immutable InputSet mapping.
That **helper export** failed with `mappingproxy` serialization; its original log
is retained. Configuration was reconstructed from the saved exact catalog without
repeating FASTQ science, and BAM then ran separately. The initial new Unicode/
choice unit test used a species parameter on `inspect_scATAC`, which has no such
parameter; correcting the test to `epizoo_embed_cells` resolved that fixture error.
Neither required weakening production validation. No recurrence of Work Unit A's
unexplained gzip-layout-A failure was observed; no timing-cause claim is made.

### I. Performance and spending

The scientific work is bounded to five FASTQ pairs, two BAM pairs and roughly
12 kb references, with independent verification enabled. Each isolated workflow
performs one intake, one producer and independent producer verification, and one
production/independent owner body each for QC, selection and matrix. FASTQ also
performs one alignment. HTTP duplicate retries, downloads and native recovery add
zero covered scientific bodies. Fixture context/index provisioning occurs before
these measured workflows and is separately retained; the counts do not claim
that qualification/integrity IO was absent.

Recorded question-through-result/download times are **36.28 s FASTQ** and
**32.14 s BAM**. The separate existing downstream pair regression takes **61.82 s**.
No stage approaches the 20-minute approval boundary. No GPU or checkpoint is
used and no complete sparse matrix is densified. Process peak memory was not
sampled for these new workflows; no unmeasured RSS estimate is presented as fact.
The resource file set is below 1 MiB before generated workspace publications;
free disk was checked before the new scientific runs. The historical 34-minute
real QC stage is neither rerun nor used to predict tiny-reference runtime.

The original ledger is checked before dispatch and copied to an additive B ledger;
its original row remains unchanged. `provider/ledger.json`, request/response files
and `budget-summary.json` show **5 new completions**, **USD 0.624854
new conservative reservations**, **USD 0.0984650 new reported cost**.
Cumulative UA3.6.4b accounting is **6 completions**, **USD 0.757452
reserved**, **USD 0.1287735 reported**. There are **zero unfinished
reservations**. The B guard enforces at most 8 additional calls and $2.50 in
additional reservations, in addition to the retained 18-call/$5.00 whole-task
policy. Output limit is 4,096, SDK retries zero, and provider fallback disabled.
No provider credentials are printed, copied or persisted.

### J. Architecture

One Interpreter/Planner path, one existing Session/pending mechanism, the same
immutable **23-tool Registry**, existing input/resource catalog, compiler and
scientific authority/verifier owners remain. No per-format language parser,
additional workflow engine, resource registry, model-specific scientific logic,
new authority framework, upload replacement, default threshold or new scientific
algorithm is introduced. Strict durable span validation remains authoritative.
Rejection diagnostics can retain a question, never a scientific result.

The UI already distinguishes request received, planning, ready for execution,
analysis in progress, clarification needed and failure, and states that Session
responses are saved. Native restart confirms terminal restoration. The reviewed
client does not expose measured percent completion; no percentage is invented.
A distinct current-scientific-step progress indicator was not added or qualified.
No UI redesign or speculative presentation fix was justified.

### K. Qualification

| Area | Qualification | Remaining limit |
| --- | --- | --- |
| Fragments | Genuine Work Unit A science retained; fresh live literal grounding succeeds | Original historical live failure retained; complete live end-to-end chain still unqualified |
| FASTQ | **PASS — bounded synthetic software workflow**, legal owner execution, resource continuation, matrix/report/download/restart; separate live initial/answer interpretation | No newly qualified genuine FASTQ source-to-matrix run; no native upload-driven or full live-provider execution claim |
| BAM | **PASS — bounded synthetic software workflow**, with inspection distinguished from producer eligibility; same delivery/restart and separate live semantics | Independent genuine BAM processing history still unavailable |
| Shared parameter behavior | **PASS within tested scope** — deterministic exact quote binding, strict provenance, actionable single-consumer correction | Semantic association still LLM-owned; arbitrary malformed/multi-consumer/staged prerequisites not automatically repaired |
| UA3.6.4b overall | **PARTIAL** | Synthetic and scripted-provider success does not establish overall genuine/live production acceptance |

### L. Recommendation

Work Unit B can close **within its demonstrated technical scope**, with no need
to repeat the genuine PBMC pipeline or build another framework to obtain a larger
biological dataset. The demonstrated off-by-one usability defect is corrected;
FASTQ/BAM have bounded ordinary-interface, verified software workflows. Proceeding
to independent UA3.6.4c work is reasonable after review, without marking overall
UA3.6.4b PASS. This task does not start that work.

Track genuine FASTQ/BAM production qualification and one complete live-provider
application chain separately. The concrete remaining upload usability boundary
is a chat request made **before** a FASTQ library can be completed: uploaded members
are retained and actionable completion guidance exists, but no durable scientific
objective spans that pre-collection state. Researchers following the supported
complete-library-first sequence are not blocked by that boundary. Do not describe
it as a solved general additional-file conversational workflow.

UA3.6.4a cross-Revision scientific dialogue remains PARTIAL and carries to
UA3.6.5 unchanged. All implementation/report changes remain unstaged. No reset,
clean, fetch, pull, stage, commit or push is performed.
