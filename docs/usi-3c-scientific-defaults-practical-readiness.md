# USI-3C — Scientific defaults and practical input readiness

2026-10-15. **PARTIAL.** Existing owner defaults and explicit-parameter continuation
are verified; no defensible automatic value for the two required selection
thresholds is established in this unit. Real file transfer, a bounded real
scientific workflow, reports, downloads and restart pass. Two demonstrated upload
storage-error defects are corrected without changing science or architecture.

## A. Baseline and preservation

Initial HEAD, local `origin/main` and independently queried remote `main`:
`11d24b892415486b49f78d1a51e8c2f77d01217f`. Tracked tree and index were clean;
`evals/` was already untracked. USI-3B is committed at this baseline. Its functional
acceptance, three established baseline failures and unresolved planning observation
remain unchanged. The stability investigation was not reopened.

New evidence root **R**: `/home/likeyi/agent-usi3c-ad7n0umr/`. `baseline.json`
records 695 tracked-file hashes, all 1,274 existing eval-file metadata entries and
49,376 protected file size/mtime/mode entries under the existing scientific
acceptance, resource and operator directories, UA3.6.4b evidence and USI-3B evidence
roots. These are bounded preservation inventories, not a full rehash of historical
biological files or model checkpoints. Consumed source files are additionally
hashed. No public datasets are downloaded; no GPU, model loading, training,
production configuration activation or historical Session write occurs.

The operator configuration and original scientific stores remain read-only.
Isolated test configuration alone temporarily raises the upload limit to 1 GiB.
Final preservation passes (`R/preservation.json`): all 1,274 eval entries and
49,376 protected entries retain size/mtime/mode, with no additions to those
protected roots; all unrelated tracked hashes are unchanged. Operator SHA-256
remains `f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
HEAD is unchanged and the index is empty. Nothing is staged, committed or pushed.

## B. Architectural audit

| Existing owner | Finding / action |
| --- | --- |
| `scatac_qc_profile`, QC reference qualification and verifier | Fixed metric definitions, exact rational sufficient statistics and qualified TSS resources already exist. No researcher-supplied QC window or smoothing choice is necessary. Preserve them. |
| `scatac_selection_profile`, selection Registry contract | Two thresholds are required; three optional criteria default to disabled. This is already one authoritative policy with independent verification. No duplicate default framework is needed. |
| `dialogue_execution`, typed scalar declarations and prerequisites | Exact utterance operands, owner validation, missing-parameter questions, scoped correction and restart continuation already exist. No new language parser or automatic workflow completion is needed. |
| Accepted manifests, resolved arguments and evidence/report projections | Effective thresholds, scientific profile, resource identity, lineage, explicit arguments and omitted-owner-default origins are already recorded. No parameter database/schema extension is needed. |
| Upload admission, local registrations and raw handoff | HTTP request streaming, SHA-256/length checks, temporary staging, same-filesystem publication and source-byte reuse already work. Complete sources are retained for identity and later consumption. |
| Existing executor and scientific owners | Bounded local workers, cooperative cancellation, disk-backed QC/counting, sparse output and independent verification are reused unchanged. |
| Scientific download preparer | A private, hashed snapshot pins delivered bytes. This deliberate full-file disk copy prevents later source mutation from changing an accepted download; it is retained. Transport and validation are chunked. |

The audit found no justified parameter-contract removal. Required numerical
thresholds reflect an unresolved scientific choice, not a redundant demand for
owner-established values. Qualified TSS resources remain necessary even when a
threshold is explicitly zero. Reference selection and threshold declarations
remain independent existing inputs.

The concrete engineering defects were confined to upload failure handling:

1. `ENOSPC` and `EDQUOT` became the generic unsafe/unavailable-storage error.
   They now return `UPLOAD_STORAGE_EXHAUSTED`, HTTP 507, explaining space/quota
   exhaustion and same-attachment retry. Permission failures retain their existing
   classification; private exception details are not exposed.
2. A real `io.BufferedWriter` over a failing raw device raises again on `close`
   after a failed flush. That second exception masked the classified transfer error
   or cancellation and skipped staging cleanup. Cleanup now tolerates this close
   error, removes staging and preserves the original outcome. Successful flush,
   fsync, publication, identity and registration checks are unchanged.

Fault injection demonstrates both defects without filling a shared disk. There is
no automatic retry, new scheduler/storage system, dependency or framework.

## C. Scientific default assessment

| Parameter / behavior | Meaning and applicability | Decision |
| --- | --- | --- |
| QC metric definitions | Weight-one canonical fragment records; QC depth on explicitly classified primary nuclear contigs; endpoints at start/end−1. TSS incidence uses a 101 bp center and two 100 bp flanks, exact `200*C/(101*(L+R))`. Requires qualified reference/TSS resources. | Existing fixed scientific profile; no new user parameter or numeric default. |
| `min_qc_fragment_records` | Required inclusive minimum canonical QC records per barcode, integer 0…10^18. Not read-pair support, total reads or counts in peaks. | No new default established. Preserve explicit/accepted task value or ask. |
| `min_tss_enrichment` | Required inclusive minimum dimensionless ratio under `agent-tss-incidence-101-200.v1`, exact nonnegative integer/decimal/rational representation. | No new default established for the current method/reference combination. Preserve explicit/accepted task value or ask. |
| `min_tss_flank_evidence` | Optional inclusive minimum L+R insertion/TSS incidence count, not unique fragments. | Existing omission/null = disabled. No arbitrary minimum flank count introduced. Explicit zero remains an explicit value. |
| `max_qc_fragment_records` | Optional inclusive maximum of the same QC-depth measurement. | Existing omission/null = disabled; no inferred doublet filter. |
| `max_nucleosome_signal` | Optional inclusive maximum mononucleosomal/nucleosome-free record ratio; lengths 147…293 / 1…146 bp. | Existing omission/null = disabled. If enabled, undefined ratio fails. No borrowed cross-method cutoff. |

The existing method is explicit candidate filtering, with statistical cell calling
`none / not_assessed`. Disabled optional filters do not establish purity, remove
doublets or supply a statistical cell-calling model. Operational limits, fixed
scientific computation and biological selection cutoffs are different categories.

Authoritative external methods were reviewed specifically to assess transferability:

- ArchR documents human-oriented defaults of 1,000 fragments and TSS 4, while
  explicitly noting dependence on the TSS set and possible need to change them.
  This is evidence for an ArchR context, not automatic qualification of Agent's
  current reference/profile. [ArchR per-cell QC documentation](https://www.archrproject.com/archive/1.0.1/bookdown/per-cell-quality-control.html).
- Its implementation also uses fragment-size/contig defaults, strand-collapsed
  TSS identities, a denominator floor and rounded TSS scores. Agent's exact
  incidence/null-background contract differs; numerical policy equivalence was
  not established. [ArchR implementation](https://raw.githubusercontent.com/GreenleafLab/ArchR/master/R/CreateArrow.R).
- Signac's PBMC tutorial combines peak-count, TSS, nucleosome, FRiP and blacklist
  filters. Its peak-count threshold is not Agent's canonical QC-record count.
  [Signac PBMC tutorial](https://stuartlab.org/signac/1.13.0/articles/pbmc_vignette).

Sources accessed 2026-10-15. The conclusion is bounded: this unit did not establish
a transferable owner-approved numerical profile. It does not assert that no such
profile could ever be qualified. Historical PBMC `1000 / 4 / 20`, later TSS 5, and
synthetic zero settings remain explicitly experimental inputs. They are not
promoted into defaults. No percentile, median or adaptive caller is introduced.

## D. Effective execution behavior

Existing behavior satisfies precedence without adding another policy layer:
explicit admitted operands take priority, scoped accepted task values survive
continuation, and omitted optional arguments use their existing owner defaults.
Missing required values trigger the existing prerequisite. Unrelated historical
values are not inherited. Conflicting or invalid declarations require correction;
an explicit value is never silently clamped or replaced. Ratios remain exact,
not float approximations.

The new real-data witness first asks for QC, selection and a matrix **without any
threshold numbers**. It asks only for minimum QC fragment depth and TSS enrichment,
performs no science, and captures the original task. A fresh server then accepts
scripted explicit operands `1000 / 4 / 20` and executes the same task. These are
technical acceptance settings carried from the historical experiment, not this
user's biological recommendation or a successful no-threshold default-policy test.

Existing focused tests exercise initial explicit operands, partial answers,
invalid negative values, correction, restart, unrelated-task isolation and retained
parameters. The USI-3B chat witness supplies explicit numeric zeros, executes actual
selection/matrix construction and rejects a barcode with undefined TSS enrichment.
Zero is neither omission nor undefined background. Missing qualified QC input is
tested independently from missing thresholds.

The delivered real report contains exact `effective_thresholds`,
`selection_threshold_origins`, QC method, qualification, profile identity and
upstream identities. Two omitted upper filters are labelled `existing_owner_default`;
the three supplied criteria are `explicit_execution_argument`. Existing interaction
records retain the exact typed utterance operands. No new provenance authority is
created; historical reports and scientific profiles are unchanged.

## E. Practical file and runtime evidence

### Limits and file lifecycle

| Boundary | Actual existing control / implication |
| --- | --- |
| Browser | Sends File/Blob through XHR; no JavaScript whole-file `arrayBuffer` preparation. Browser memory/throughput is not measured here. |
| HTTP upload | Default/operator deployment limit 256 MiB per file; configurable up to 8 GiB. Content-Length is checked early and actual streamed bytes are independently counted. A paired source/index has separate per-file bounds. |
| Upload owner | Writes/hash updates in at most 1 MiB slices; 128 staging/completed files; default two concurrent transfers, configurable one through four per instance. Per-upload filesystem leases preserve identity. No aggregate byte quota or new reservation service is added. |
| Registration/handoff | Full hash/integrity reads may occur more than once but do not materialize duplicate full-size source files. Completed uploads survive restart and are shared by task-local typed registration. |
| Execution | Default two local workers, configured one for this witness; supported one through eight. No durable job queue or global RSS governor. Existing owner cancellation/checkpoints remain authoritative. |
| QC | Disk-backed SQLite, 8 MiB configured cache; up to 10 million barcodes / 1 billion records, 16 GiB sidecars and a 64 GiB production SQLite check. These are protective ceilings, not validated throughput promises or a single global disk quota. |
| External import / matrix | External sort uses one worker and a 64 MiB sort buffer. Matrix defaults cap features at 2 million, selected cells at 10 million, scratch at 100 GiB and stage time at 24 hours. Sparse construction is unchanged. |
| Download | Scientific tables/matrices have a 1 GiB per-file ceiling and two concurrent snapshots. Disk admission includes snapshot size, current reservations and a 64 MiB margin. Stream chunks are 1 MiB; snapshots close after delivery/cancellation. A scientifically accepted larger matrix can still exceed the current delivery ceiling. |

Uploading within a byte limit does not imply scientific eligibility or execution
capacity. Production defaults were not raised; this unit does not qualify the
8 GiB upper setting, whole-library FASTQ processing or maximum-size downloads.

### Real inputs and bounded execution

Existing approved local sources were used without downloads or historical writes:

| Witness | Bytes | Observation |
| --- | ---: | --- |
| PBMC 1k Next GEM L001 R1 FASTQ.gz | 835,954,322 | Default 256 MiB setting returns 413 from length headers before body transfer. Isolated 1 GiB setting streams/registers the complete file in 6.52 s. Same-identity retry takes 1.44 s and preserves inode/mtime. Transfer only; no new FASTQ alignment or library qualification. |
| Existing PBMC500 primary subset fragments | 110,850,431 | Complete HTTP transfer/registration in 1.06 s. Transport witness only; no redundant full scientific reconstruction. |
| Four-barcode real fragments subset | 703,660 | All 80,808 original records for the first four barcodes in the historical accepted selection; complete genome-wide records and exact original order. Independent second source/output pass verifies byte conservation. Upload 0.08 s; actual new scientific execution below. |

`R/real-input.json` records exact paths, source/subset SHA-256, acquisition lineage,
barcodes, record counts and preparation rule; `prepare_real.py` is reproducible.
The full fragment source's pinned SHA-256 is
`03a6d40d3a8a934c550fb4a4b86362585d11d8a99835ba73f96683b6fb23ac9a`;
the four-barcode gzip SHA-256 is
`27a7208d2a59ecf53c7f88975fa7e76d4d4c4437da518a71ee7d0a31b1199c1d`.
This is a preselected technical sample, **not** an unbiased cell population,
biological threshold calibration or simulated data. It preserves the existing
10x source-profile declaration and `subset_export` provenance.

Before execution, `execution-budget.json` recorded the historical stage timings,
full-reference cost, available 15.57 TB disk and a bounded plan: one four-barcode
workflow, 30-minute science deadline, 10 GiB workspace and 4 GiB server RSS ceilings.
The external acceptance monitor samples workspace bytes and server RSS every
0.5 s; it is not production scheduling infrastructure or a process-tree memory
limit. Full reference qualification is retained despite the small fragment sample.

The normal HTTP/application path uses existing operator-qualified hg38 reference
and GENCODE 49 QC resources, with synthetic admission disabled. Scripted providers
choose roles, operands and the explicit plan; actual scientific tools, compiler,
authority, publication and downloads run normally. The four stages succeed with
schema-2 independent verification:

| Stage | Measured seconds |
| --- | ---: |
| External fragment import | 11.98 |
| QC, including existing resource qualification | 320.08 |
| Cell selection | 0.35 |
| Canonical matrix | 26.29 |

Submission-to-terminal observation takes 372.58 s. The accepted int64 CSR has
**4 × 1,355,445**, **56,205 nonzeros**, total count **74,778**. All four QC rows,
every selected-cell sparse matrix value and every ordered feature match the exact
historical accepted counterparts. `verify_real.py` performs chunked HDF5/sparse
comparison, without densifying a complete matrix or reconstructing old science.
All four selection candidates remain `cell_call_state=not_assessed`.

The server's observed peak RSS is **931,049,472 bytes**; during large upload it is
691,855,360 bytes versus a 689,901,568-byte startup observation. This supports bounded
transfer memory for this file, not a general maximum-memory guarantee. Peak visible
workspace size is **1,472,491,394 bytes**, including both engineering uploads and
scientific scratch/output. It excludes anonymous download snapshots and subprocess
RSS. Measurements are on a shared host, with some focused regression work concurrent;
they are not isolated performance benchmarks.

All five scientific downloads match their accepted SHA-256. `matrix.h5ad` is
**163,398,899 bytes**, delivered in **48.99 s**, including existing snapshot/privacy
preparation. QC/selection table sizes are 386, 2,856, 251 and 223 bytes. The report
also downloads successfully. There are exactly three completed upload payloads,
no staging remnants, and the consumed fragment source uses the original upload
path/hash. No handoff copy is created. Final server restart forbids provider
construction; historical views and exact completed retry match.

The real positive uploads occurred before the final buffered-failure cleanup
refinement. That refinement affects exceptional cleanup only; its final behavior
is covered by the 72 focused tests and final regression. Scientific source files
are unchanged throughout this unit, and accepted science was not rerun to test
an upload-error branch.

The large FASTQ is registered as a neutral attachment, not declared scientifically
ready. No native browser benchmark, complete sequencing library, large population
QC, BAM transformation, GPU or production deployment is qualified by this witness.

## F. Acceptance and regression

Commands use `/home/likeyi/anaconda3/envs/agent/bin/python -B`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, and isolated R basetemp/cache paths.
All `RUN_*` gates are unset for regression. Real science explicitly sets the
unchanged qualified QC catalog and BEDTools paths, with `AGENT_QC_ALLOW_SYNTHETIC`
unset. Exact scientific server argv/environment are in `server-command.jsonl`.
`focused-commands.json` records all three development pytest invocations;
`environment.json` records Python 3.11.16 and the installed package versions.

| Check | Result / evidence |
| --- | --- |
| Storage classification reproduction | 6 failed / 3 passed, 36 deselected in 1.61 s; new fault tests against baseline implementation, `storage-before.log`. |
| Initial focused upload correction | 68 passed in 7.62 s, `storage-fixed.log`. |
| Buffered device reproduction | Raw `OSError` escaped and a `.part` file remained; `probe_buffered_failure.py`, `buffered-failure-before.json`. |
| Final upload/application/HTTP checks | 72 passed in 8.80 s, `storage-final.log`; includes ENOSPC/EDQUOT/EACCES, stage/write/publication sync, buffered close, cancellation, cleanup and exact retry. |
| Early regression, intentionally stopped | 209 passed in 235.91 s, exit 2; stopped after the newly demonstrated buffered-close defect, before changing source. All 534 frozen files unchanged. Original log/XML/command retained; this is not a completed regression. |
| Final affected-domain regression | **579 passed, 3 warnings in 666.03 s**, exit 0. All **534 frozen source/test/static/resource files** unchanged; `final-regression-command.json`, `final-regression.log`, `final-regression.xml`, `final-regression-result.json`. |
| Real preparation/execution/delivery | `prepare_real.py`, `run_real.py`, `real_server.py`, `real-results.json`, `resource-observations.json`, `real-verification.json`; all checks described above pass. |

Reproduction commands:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q \
  tests/application/test_uploads.py tests/web/test_upload_api.py \
  --basetemp=/home/likeyi/agent-usi3c-ad7n0umr/storage-final \
  -o cache_dir=/home/likeyi/agent-usi3c-ad7n0umr/cache
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  /home/likeyi/agent-usi3c-ad7n0umr/run_final_regression.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  /home/likeyi/agent-usi3c-ad7n0umr/prepare_real.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  /home/likeyi/agent-usi3c-ad7n0umr/run_real.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  /home/likeyi/agent-usi3c-ad7n0umr/verify_real.py
```

These are the executed commands; use a fresh isolated evidence root for any new
execution instead of overwriting this acceptance. The first read-only comparison
script incorrectly expected the historical *upstream-only* summary to include the
matrix. Its preserved `KeyError` log is `real-verification-first.log`; the corrected
script resolves the exact historical Run by selection digest and passes without
rerunning science. No production fix was made for that diagnostic-script error.

The final regression selects 19 paths covering complete cell-selection tests,
QC metric science, typed parameter admission/evidence/continuation, uploads across
raw formats, chat fragments and missing-file continuation, matrix/table privacy,
streamed delivery, HTTP and restart. No individual cases are excluded within those
paths. The three warnings concern deprecated pkg_resources/louvain and the
installed TBB threading-layer version; no scientific setting was changed to
suppress them. This is the single completed affected-domain regression after the
upload correction stabilized, not a full repository test run.

The three independently demonstrated USI-3B baseline assertions (catalog size,
absent whitelist identity, singleton-resource status) are not modified or included
merely to improve counts. The non-reproducing planning anomaly remains unresolved;
the real server retained a temporary exception hook and observed no such failure.
No claim of a green full-repository suite or resolved USI-3B stability is made.

## G. Final disposition

**PARTIAL.** The requested positive no-threshold selection workflow remains
unqualified: numerical defaults require a scientifically justified owner profile
for the actual measurement/reference semantics. This limit is not filled with
historical exploratory numbers. Existing optional defaults, exact user overrides,
clarification/continuation and provenance remain working.

Practical acceptance passes for the measured real transfers and four-barcode
scientific workflow. Maximum upload settings, complete raw sequencing workloads,
large-population execution and browser memory/throughput remain untested here.
The two demonstrated upload failure defects are fixed at their existing owner.

The current interfaces are ready for subsequent application-handoff work **with
explicit selection criteria**; no new prerequisite architecture is demonstrated.
Scientific default qualification remains an explicit USI-3C limitation rather
than being silently reassigned. USI-4/5/6 retain their later raw/feature, MAESTRO
grouping/resources and EpiZoo/post-training application handoffs. USI-7 retains
deployment sizing, limit configuration and browser/UX qualification; UA3.6.5
retains ordinary-user live-provider, real-data and browser acceptance. None is
claimed completed or started by this unit.

Changed files:

- `src/agent/application/uploads.py`: precise storage exhaustion and cleanup of
  buffered failures without masking the original error/cancellation.
- `tests/application/test_uploads.py`: storage failure, cleanup and retry coverage.
- `tests/web/test_upload_api.py`: safe HTTP 507 and same-upload retry.
- `docs/usi-3c-scientific-defaults-practical-readiness.md`: this assessment and evidence.


## H. Final practical-default policy and closeout (USI-3C follow-up)

This section supersedes the initial PARTIAL disposition in section G within the
engineering and practical-default scope explicitly authorized by the final
USI-3C request. Sections A–G above are preserved verbatim as historical evidence;
the earlier scientific caution and the completed 836 MB upload/real PBMC
acceptance are not rewritten as default calibration.

### Adopted policy and scientific limits

Cell Selection now uses `min_qc_fragment_records=1000` and
`min_tss_enrichment=4` when those arguments are omitted. These are transparent,
adjustable starting thresholds chosen by the final task policy, not estimates
learned from the four preselected PBMC barcodes, universal biological optima,
statistical cell calling or ArchR certification.

The units and measurement profile are unchanged: depth counts canonical fragment
records in the qualified primary-nuclear QC scope, not read-pair support or
molecules. TSS enrichment remains the exact ratio `200*C / (101*(L+R))`, with the
existing center/flank windows, endpoint incidence and qualified transcript-TSS
reference. Zero flank background remains undefined and fails selection, even
when a user explicitly sets the TSS threshold to zero. ArchR's normalization,
annotation and aggregation choices differ; there is no numerical equivalence or
formula conversion claim. No QC, TSS, fragment-counting, nucleosome, selection
comparison or independent verification algorithm changes.

Validated explicit values take precedence over captured values for the same
continuing task; captured values take precedence over omission defaults. Zero is
an explicit value. Negative, boolean, float and explicit null minima remain
invalid, rather than being replaced with a default. The other three thresholds
remain absent/null = disabled. An empty selection succeeds with
`no_selected_cells`; thresholds are never lowered to force a positive result.
Qualified QC/TSS resources, exact source identity, species/assembly compatibility,
lineage and independent verification remain mandatory. Default thresholds supply
none of these resources or identities.

### Implementation and compatibility

- `src/agent/tools/data/scatac_selection_profile.py:DEFAULTS` is the single
  numeric policy definition. `thresholds()` defaults only missing keys;
  `_cell_selection_contract.arguments()` records explicit effective values in
  the ordinary manifest and receipt. The comparison profile and its hash remain
  unchanged. The independent verifier still requires complete effective values
  and independently reconstructs the same decisions.
- `scatac_cell_selection.select_scATAC_cells` uses those constants in its public
  signature. Registry optionality and semantic ports now reflect omission
  support; the existing Registry reads defaults from that signature. No new
  resolver, default registry, schema, dependency or policy framework is added.
  `output_dir` remains required and is now keyword-only so the signature can
  express the two optional minima. Repository callers and persisted plans use
  named arguments. Former calls supplying output_dir as a fifth positional
  argument now raise TypeError explicitly; they are not silently reinterpreted.
- The executable plan retains omitted versus supplied argument presence. The
  ordinary effective-threshold evidence and existing `selection_threshold_origins`
  distinguish `existing_owner_default` from `explicit_execution_argument`;
  typed utterance records separately establish `explicit_user_instruction`.
  Supplied values are never relabelled as defaults merely because they equal
  1000 or 4. Existing reports and safe downloaded report projections explain
  adjustability and empty-selection behavior.
- The existing selection recipe projection in `turn_context` exposes the two
  effective minima for editing and read-only parameter answers. Its explicit
  accepted arguments override the fixed owner defaults. Delta execution checks
  the same recipe before compiling a new plan. An independently requested
  selection does not inherit an earlier selection's parameters, even when it
  explicitly reuses that selection's upstream QC. Captured continuation records
  and exact prior-output bindings continue to identify retained values; an
  execution argument alone does not claim a fresh user declaration.
- A bounded reader in `dialogue_execution` accepts the former two-required-port
  pending records when validating/reopening their exact captured task and stored
  missing-source diagnostic. Such historical pending tasks retain their original
  outstanding declarations; new tasks do not create those routine questions.
  Rejected explicit declarations still require correction. This is a small
  compatibility reader, not a parallel execution engine. Legacy pending tests
  use a test-only former Registry port fixture; fresh-process reopening uses
  the current default Registry.
- Pre-default accepted selections already recorded both minima explicitly.
  Their effective values, manifest/receipt identities, accepted authority and
  scientific meanings remain unchanged. No historical artifact migration or
  data rewrite is performed. README and the current AGENTS contract now describe
  this override without erasing the historical milestone record.

### Acceptance evidence and scope

New evidence is isolated at
`/home/likeyi/agent-usi3c-defaults-hk8xy3fy/`. The original evidence at
`/home/likeyi/agent-usi3c-ad7n0umr/` is retained, including its final 579-pass
regression, real transfer timings, real scientific execution and recovery.
No large upload, real PBMC production, model loading or GPU run was repeated.

`tests/web/test_selection_defaults.py` sends the no-number request
“Select candidate cells from the supplied verified barcode QC.” through the
normal HTTP worker, interpreter, semantic Planner, deterministic binding,
scientific owner, independent verification, evidence and report download.
Only language decisions are scripted. A labelled synthetic fixture has 1,001
distinct canonical records for barcode A and one no-TSS-background record for
barcode B. Defaults select A and reject B; selection production and the
independent verifier each run once. The actual manifest records 1000 and exact
4/1; both origins are owner defaults, no numerical argument is fabricated in
the executable plan, and completed retry runs zero additional science. This is
bounded software acceptance, not biological calibration or live-provider quality.

Application cases cover omission of both, explicit 2000/5, depth-only 1500,
explicit TSS zero, exact decimal values, invalid inputs, disabled optional
criteria, empty selection, evidence and restart/retry. Authority-bound BAM and
external-fragment cases reuse accepted QC, independently execute defaults,
then change TSS to zero while preserving default or explicit depth. Their prior
Runs remain byte-equivalent as serialized records. Read-only threshold answers
report the effective default without science. Original pending tasks, corrections,
stale generations, changed inputs, ambiguity and string-parameter continuation
retain their existing protections.

The final regression command, frozen source/test/resource hashes, JUnit results,
log and process result are recorded under `final-regression-*`; the exact path
list is `regression-paths.json`. All `RUN_*` gates are unset. One unrelated
pre-existing string-column inventory assertion is explicitly outside that final
selection: `test_registry_metadata_offers_only_reviewed_free_text_declarations`
expects a list omitting `epizoo_embed_cells.assembly` and the existing annotation
`group_column`/`count_slot`. The unmodified starting commit independently fails
that same assertion (`baseline-metadata.log`, 1 failed in 1.60 s). The assertion
and those scientific capabilities are not changed. This is not a claim of an
all-green full repository suite.

Development observations are retained rather than discarded: the first trial
exposed the Registry/public-signature requiredness check, corrected by keeping
output_dir mandatory; the interrupted run is not an acceptance pass. The initial
owner/typed batch had 348 passed and 17 failures: 16 test lookups still expected
required minima, plus the independent baseline string inventory assertion.
The 16 lookups now use the optional contract. The old FASTQ “missing thresholds”
case is explicitly a historical pending-contract fixture; current default-driven
scientific execution has its separate positive HTTP acceptance. No unrelated
scientific behavior was changed to remove a baseline failure.


Focused command records are in `focused-commands.json` (exact argv, environment,
working directory, log and exit status). Counts overlap and must not be summed:

| Check | Result |
| --- | --- |
| Typed continuation and historical pending reopening | 33 passed, 130.43 s |
| New default execution, HTTP, and authority-bound delta cases | 17 passed, 18 other cases deselected by the focused selector, 243.93 s |
| Initial owner/typed batch | 348 passed, 17 failed, 173.97 s; 16 old required-argument test lookups corrected, one unrelated baseline failure retained |
| Corrected declaration contracts plus historical FASTQ pending case | 145 passed, one independently confirmed baseline case deselected, 3.59 s |
| Registry, semantic compiler, report and response contracts | 187 passed, 9.11 s |
| Initial raw-input/current-context batch | 40 passed, three old missing-threshold assertions failed, one guarded FASTQ execution skipped, 525.03 s |
| Current raw invalid-parameter/stale-generation cases after updating the fixture | 2 passed, 22.65 s |
| Repeated-operation ambiguity rejection after updating the expected compiler exit | 1 passed, 3.77 s |

The first affected-domain attempt was stopped after 548 passes and one obsolete
ambiguity-response assertion (397.55 s, pytest exit 2; 399.57 s process time).
All 536 frozen files were unchanged during that attempt. The former required
parameters reached the application declaration-scope guard; optional parameters
now hit the compiler's earlier `AMBIGUOUS_OPTIONAL_INPUT_SCOPE` rejection. Both
paths execute zero science and never choose a first consumer. The test now checks
the exact compiler rejection and idempotent retry. No production ambiguity
heuristic or error-mapping layer was introduced. `interrupted-regression-*`
retains this attempt; its scientific fixtures remain at `final-regression-temp`.
The final attempt uses a separate `complete-regression-temp`.

The executed final command is:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /home/likeyi/anaconda3/envs/agent/bin/python -B \
  /home/likeyi/agent-usi3c-defaults-hk8xy3fy/run_final_regression.py
```

The runner invokes pytest over the 23 paths in `regression-paths.json`, with
`-k 'not test_registry_metadata_offers_only_reviewed_free_text_declarations'`,
isolated pytest/Matplotlib/Numba paths and JUnit output. It removes all `RUN_*`
environment gates and verifies source/test/resource hashes on completion. No
other individual case is excluded. The skipped guarded Chromap/FASTQ execution
is not counted as scientific acceptance: this follow-up covers FASTQ typed
handoff and prerequisite compatibility, with actual small BAM/external-fragments
owner execution and the independent positive default-selection HTTP case.


### Final result, preservation and disposition

The completed affected-domain regression passes **795 tests, with 1 skipped,
1 explicitly deselected baseline assertion and 1 warning in 2,281.77 seconds**
(pytest exit 0; process time 2,284.05 seconds). All **536 frozen source, test,
static and resource files** remain unchanged during execution. The warning is
pytest's existing `record_property`/JUnit xunit2 compatibility warning in the BAM
HTTP fixture, not a scientific verification failure. No full-repository green
claim is made. No additional testing is needed for this unchanged implementation.

The pre-commit preservation check confirms all 1,274 pre-existing eval file
metadata entries and 49,376 protected file entries unchanged, with no protected
additions. The operator configuration retains SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
The three prior upload implementation/test files are byte-identical to the
preserved initial USI-3C working state. Sections A–G of this report remain an
exact byte prefix. QC production/profile, selection production, both independent
verifiers and EpiZoo match their starting file hashes (`scientific-preservation.json`).
No biological data, model weights, operator configuration, Session/Revision
records or evidence directories are staged. No new dependency is introduced.

**USI-3C — COMPLETE within its defined engineering and practical-default scope.**

The default-parameter usability gap is closed. This does not establish universal
biological optimality, ArchR numerical equivalence, cell purity, statistical cell
calling, full-scale raw-data execution or live-provider language qualification.
The original real-data and large-transfer results retain exactly their earlier
bounded scope. The unresolved USI-3B planning anomaly and its separately recorded
baseline failures remain historical observations; USI-3C does not claim to fix
them. USI-4 can proceed with the existing scientific-resource and authority
requirements intact.

Git delivery starts from verified `main` at
`11d24b892415486b49f78d1a51e8c2f77d01217f`, matching local `origin/main` and remote
`main`. The final request explicitly authorizes the 28-file USI-3C work unit,
including the preserved upload corrections, default behavior/tests, this report
and the README/AGENTS current-contract corrections. `git diff --check` passes.
The authorized commit message is:

```text
Complete USI-3C scientific defaults and practical input readiness
```

Commit/push and post-publication reference verification are recorded separately
in `git-final.json` in the isolated evidence root and in the delivery response;
this avoids embedding a self-referential commit hash in its own report. The Git
operation is a normal push of `main`, with no force push or history rewrite.
