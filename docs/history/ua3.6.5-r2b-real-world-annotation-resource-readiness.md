# UA3.6.5-R2B — Real-world annotation and resource readiness

2026-10-19 (Asia/Shanghai). **R2B PARTIAL.** The targeted closeout below adds shared
evidence-only answer recovery, verified native scope diagnostics, clearer H5AD
guidance and annotation summaries, and persistent normal-runtime activation.
Both original questions now return accepted facts in the live control journey.
The full external matrix remains outside the qualified input policy; common
annotation-reference deployment remains blocked by signature permission.
Original evidence root **R**:
`/home/likeyi/agent-ua365-r2b-6zkcjv2c`, outside Git. UA3.6.5 remains open.

## Baseline and reviewed boundaries

Initial branch `main`; HEAD, local `main`, `origin/main` and the read-only remote
`main` query all return `d5320bebb6aaa7e812cc3f3f173b4fa04b5cf862`.
R1 `d46e132b27fa0ad80c5859bedcf67fd6639dd47d` is its parent. The tracked tree
and index were clean; `?? evals/` was the sole initial status entry.
`R/baseline.json` records 720 tracked-file hashes and 123,109 protected entries.
Protection uses size/mtime/mode plus hashes for files ≤64 KiB and every eval file;
it is not a new full-byte audit of all historical biological datasets/models.

Reviewed README/AGENTS and the USI-4 handoff/resource closeout, UA3.6.4c native
peak acceptance and peak-contract review, R1 and R2A records; inspected current
annotation input, resource binding/discovery, scientific owner, Registry,
capability selection, Session continuation and error-policy code. Historical
scientific qualification is distinguished from new execution throughout.

The existing direct method remains `maestro-enhanced-count-weighted-peaks.v1`:
human hg38 raw Cell Ranger ATAC cut-site counts, qualified primary nuclear peaks,
explicit exact groups, compatible refgenes/signatures and tissue context. Native
Enhanced RP, marker statistics, scoring, conservative assignment, independent
owner replay and publication are reused. No new tool, grouping engine, reference
catalog, scientific method or UI layout is introduced.

## Authentic external input

New acquisition is the official [10x 1k PBMC Next GEM dataset](https://www.10xgenomics.com/datasets/1-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-next-gem-v-1-1-1-1-standard-2-0-0),
published 2021-05-03, Cell Ranger ATAC 2.0.0, human PBMC, CC BY 4.0. The publisher
reports 1,004 detected nuclei. This is a newly acquired publisher dataset,
separate from the 484-cell acceptance input; the publisher states the PBMC loads
share nuclei preparation, so no independent-donor replication is claimed.

All files come from
`https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_1k_nextgem/`:

| Suffix after `atac_pbmc_1k_nextgem` | Bytes | SHA-256 |
| --- | ---: | --- |
| `_filtered_peak_bc_matrix.h5` | 12,853,716 | `a1160985dd9dbec14e5c4ccb2f6baabaf12be4e2d7d914b6d4ad7f3543bd1992` |
| `_analysis.tar.gz` | 63,996,499 | `392d0796df58cb95e225059fea1307715d64ef5813334e159b98e0e7986d13b4` |
| `_web_summary.html` | 2,674,277 | `48de685573766fa677034db6c9a23fa18e8ba3b2ad1118b7474d4fbc0fd10850` |

`R/acquire.py` checks free disk (about 15.6 TB), streams with connection/read and
elapsed bounds, checks Content-Length, caps each file at 250 MiB and hashes exact
bytes. `R/acquisition.json` records URLs/dates/roles; no publisher checksum was
available in the inspected evidence. Total new scientific download: 79,524,492
bytes. No reference/model download was necessary.

Publisher feature metadata explicitly says GRCh38. The
[10x matrix contract](https://www.10xgenomics.com/support/software/cell-ranger-atac/latest/analysis/feature-barcode-matrices)
establishes features × barcodes and cut-site counts; peak IDs retain publisher
zero-based half-open coordinates and barcode suffixes. `R/prepare_data.py`
transposes sparse storage into ordinary cells × peaks AnnData and imports
`analysis/clustering/graphclust/clusters.csv` only after exact ordered barcode
comparison. It retains every cell, peak and count, without filtering, alias
repair, normalization, annotation-specific metadata or invented provenance.
This is documented standard-format preparation, not an original publisher H5AD.

Prepared file: 1,004 × 82,579 int32 CSR, 8,164,121 nonzeros, 22,279,127 total
counts, maximum 264; six publisher groups (359/178/132/115/114/106 cells).
Size 26,869,587; SHA-256
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`.
`R/preparation.json` records round-trip sparse equality and exact axes/groups.
`R/source-scope.json` identifies 31 peaks (6,062 counts) outside the existing
qualified primary scope. They are not silently classified by name or removed.
The full publisher matrix therefore does **not** complete the current direct
annotation method. No acceptance-tailored primary subset is manufactured.

## Resource readiness and deployment

| Resource | Physical / qualification | Normal Web visibility |
| --- | --- | --- |
| MAESTRO GRCh38 refgenes, revision `74f10babf4125b98ef005df4186b14ab02e13bfd` | Existing pinned source; SHA `9b18e56531155226450254aefd192bb7f49d03d74c5bbfdc53af5411a0d2fb37` | Not configured |
| Human immune CIBERSORT signature export, same revision | Existing documented header-only export; SHA `3c202410d4f46049a298784838e5f9d214fc67fcddf34536d1c3811e383b601a` | Not configured |
| Qualified hg38 GENCODE-49 chromosome/QC bundle | Existing owner catalog; manifest SHA `e9037c7a80964495e5b3aaea4aa6d634035d05cf7aaf5a36d8cff27bb21afbd5` | Not offered as an annotation dependency |
| Native MAESTRO / R / Presto runtime | Existing historical installation and pinned sources | Required annotation environment absent from the inspected normal conda launch |

The production JSON has two InputSets (accepted PBMC cCRE input and mouse species)
and EpiZoo resources. Its three local-resource records are private H5AD uploads,
not qualified annotation references. Its default conversational profile is Groq;
Firefox explicitly selects the configured OpenRouter GPT-5.6 Sol profile.
Physical presence, scientific qualification and deployment availability differ.

`R/resource_readiness.py` freshly validates exact existing gene/signature/scope
pins using the existing owner: 28,307 symbols, 78,362 primary transcripts and 28
candidates. This validates computational compatibility and declared applicability,
not biological truth or newly reconstructed genome qualification. Pinned native
source checks pass. A first diagnostic incorrectly assumed a `Runtime.check()`
method and failed; the corrected diagnostic uses the existing checks.

`R/operator-initial.json` preserves the production-equivalent isolated config.
`R/operator.json` adds the existing reviewed human PBMC reference through the
existing InputSet mechanism; `R/scientific-environment.json` supplies the existing
catalog/runtime only to the isolated server. No production file/default changes,
private-resource promotions or cross-user redistribution occur. The pending
external task discovers this added resource after restart without reupload.
Production deployment requires explicitly reviewing/enabling these existing
resource and runtime pointers; this acceptance configuration is not deployment.

## Real browser/model observations

All described live turns use native headless Firefox/Marionette, ordinary upload,
composer and HTTP application paths, and the configured
`openai/gpt-5.6-sol:floor` transport/routing/retry/price limits. Credentials are read
from the existing environment and never copied into evidence. Provider
prompt/schema/response audit and science-entry counters are observational;
no response or scientific implementation is replaced. Audit calls are not billed
cost measurements.

| Trace | Request / observed result |
| --- | --- |
| A1 | “Can you annotate the cell types in this dataset? These are human PBMCs.” Correct typed annotation request and targeted assembly/history/count-slot/grouping clarification. |
| A2 | Natural reply identifies 10x/Cell Ranger 2.0.0, hg38, original cut-site counts in X and publisher `cellranger_graphclust`. The model emits prose measurement history; Agent incorrectly reports incompatible declarations because it expects an undisclosed exact identifier. No science. |
| A3 | After the handoff repair and isolated resource activation, a natural correction continues the **same** task/source. The model selects the canonical measurement and exact offered `human PBMC` context/reference. Planning reaches the owner; full external peaks fail scope qualification before any native annotation. Initial UI gives only “The operation failed.” |
| B1–B2 | Historical genuine 484-cell positive control, ordinary annotation request and grouping reply. Interpreter correctly binds annotation. Capability selection incorrectly offers label transfer only; detailed planning returns unsupported. No native science. |
| C1–C2 | Unchanged original control request and grouping reply after the capability-description correction. The selector includes marker annotation, Planner binds the original counts/groups, and native production plus one independent owner verification succeed. |
| C3–C4 | “Which groups remain unresolved, and what evidence supports these annotations?” and “How many cells were assigned, and were any groups unresolved?” Both correctly select the accepted result but fail answer-generation factual binding. Zero additional science. |
| A4 | Reopen the original external Session and request annotation again with the same explicit scientific context. Correct canonical measurement/reference binding; Firefox now reports `ANNOTATION_PEAK_SCOPE_UNSUPPORTED` with the actual primary-scope limitation. No native backend, annotation result or Revision. |

Exact utterances, decisions, screenshots, timelines, Session/Run identities and
errors are retained in `R/*-page.txt`, named browser JSON files,
`provider-audit.jsonl`, `science-audit.jsonl`, and `workspace/`.
The historical control is explicitly separate from external-user acceptance.
Across ten live turns there are **23 actual provider calls**: ten interpreter,
four capability selection, four detailed planning, three output selection and
two answerer calls. All use the configured model; no transport errors or model
fallbacks occur (`R/audit-summary.json`). These bounded trials do not establish
provider reliability or a causal statistical estimate for the metadata change.

## Findings and owning repairs

| Severity | Finding / disposition |
| --- | --- |
| P1, repaired | `measurement` accepted free text while deterministic admission required an identifier absent from prompt/schema. Schema now exposes the owner constant and its supported meaning, retains unknown/null and unsupported text, and leaves language interpretation to the LLM. No regex repair or relabelling of binary/normalized counts. |
| P1, repaired | Pending annotation declarations could not correct a wrong measurement/context. Typed corrections now create a new record for species/assembly/measurement/context; captured source/group file handles remain fixed. Source-metadata merge stays strict. Original records and accepted evidence are not rewritten. |
| P1, repaired metadata | Capability selector still described marker annotation as canonical-only. Updated its existing description to reflect direct peak H5AD, explicit groups, native MAESTRO and gene/signature resources. No forced selection, auto-completed workflow or second planner. Live quality remains separately measured. |
| P2, repaired | Scope rejection was hidden by generic `ValueError` classification. The coordinate owner now raises a typed scope error for nonmember contigs; the existing classifier/error catalog supplies a safe actionable message. Coordinate bounds, scope membership and algorithm remain unchanged. |
| P2, unresolved | Both live evidence answers repeat the categorical status `assigned` in prose. All claim IDs are valid, but the existing whole-claim contract rejects that prose. Offline replay identifies `factual_binding/unbound_categorical_value` in both; no evidence corruption or wrong current-result selection. The report/detail/download routes remain usable. |
| P2, recorded | The visible summary prioritizes an artifact-schema number and omitted-record count, while labels require expanded details; failed read-only answers display “Analysis failed” beside a still-successful result. No UI redesign or patch is made. |
| Deployment gap | Qualified resources/runtime exist but are not available to ordinary production Sessions. Isolated activation succeeds at binding; production activation remains explicit. |
| Scientific/product limit | Full external publisher peak axis is outside current primary scope. This is an honest method-boundary failure. Supporting an explicit provenance-preserving preparation or broader qualified scope requires a separate reviewed capability decision; no silent filtering. |

No P0 scientific-integrity or privacy failure was observed. The repeated answer
failure was investigated at the shared renderer, not dismissed as an isolated
response. `R/answer_diagnostics.py` replays captured responses without providers
or science. The categorical guard treats even ordinary predicate wording as a
value repetition; the provider nevertheless violates its explicit current
contract. Permitting prose reuse safely requires reviewed subject/value
correspondence rules, not a special exemption for “assigned.” This unresolved
shared product limitation is part of the PARTIAL disposition.

Changed owners: `application/raw_handoff.py`, `application/turn_decisions.py`,
`orchestration/planning_scope.py`, `orchestration/annotation_registry.py`,
`orchestration/error_policy.py`, and the typed exception in
`tools/analysis/external_marker_annotation.py`. No native calculations, EpiZoo,
Registry membership, data or UI source changes. Focused tests are in the existing
annotation handoff/external-entry files. The initial-scientific-declarations
tests also replace obsolete fixed parameter totals with exact offered-pair
coverage and uniqueness checks. No new production module or compatibility
writer is added.

## Accepted control, downloads and recovery

The control is the previously prepared genuine 484-cell/65,887-primary-peak
H5AD (source SHA `72e581c3f027cc9330098f655335b6cc9406d39c136c882d4e2e95768c2efe38`),
with all 65,908 original publisher peaks preserved in raw. Its historical primary
selection is disclosed in UA3.6.4c; R2B does not recreate or tailor this source.
The new value is the live request/clarification/planning/native-result chain.

- Session: `ceccb6f2-7fee-46f9-8673-50fd0551a690`.
- Run: `turn-efaccb4900ddc3676237e76ec1c2bd50a7461a63c5191ef481233e3852d774e0:run`.
- Revision: `277a861fb29a3410fba4c3c405f5a4aa1443e9488dd78e5ac76c6003df56642f`.
- Manifest SHA: `7d5dc351f091ab96504aaae89e115f73abbf3b45b986e99b62b07bf3bd69a442`.

Exactly **two native backend calls**, one production and one within **one owner
verifier entry**, are observed. Four source-qualification entries cover the two
external failures plus control production and verification. No upstream
preprocessing, PCA, EpiZoo, implicit clustering or optional annotation validation
runs. The method labels all 484 cells: group 1 Monocytes (207), group 2 ActNK
(114), group 4 MemoryBcells (57), group 3 NaiveCD4Tcells (106); zero unassigned or
ambiguous. These match the historical control group summaries and remain
`validation_state=not_assessed`, not biological ground truth.

Firefox displays the completion summary, report, detail evidence and native
H5AD download. No scientific figure is produced by this annotation path.
The 22,903-character delivered report is explicitly a safe technical report.
The native download is 91,020,412 bytes, SHA-256
`6c694d33aacab55dd87276d32014e7e3e126dc8723c97ddc6af5fa7a1fea7f24`,
matching the browser response and `X-Artifact-Content-SHA256`. It intentionally
differs from source publication SHA
`accb3bcffc42e50943806898a86f0bdd32759039b2f8363c83b722ba545187c9`
because the established export omits the private source locator.
`R/science_check.py` reopens the actual download and checks exact axes, original
obs/var, sparse X/raw and every group-to-cell label/status. No backend rerun.

Refresh and a fresh Firefox/server restore all eight conversation texts,
generation 1, the same Revision and report. **All 71 workspace files remain
byte-identical, with zero provider/scientific calls** (`R/recovery.json`).
Earlier failed Sessions remain accessible; the original USI-4 accepted Session
also passes the separate read-only check below. No branch/comparison acceptance
or clustering-to-annotation grouping handoff is claimed in this round.

## Validation and closeout

The initial handoff run had 17 passes and one new-test failure because the test
assumed unavailable `jsonschema`. Removed that dependency; final handoff run:
**18 passed in 71.90 seconds**, exit 0. It covers pending correction after restart,
original record preservation, exact source identity, source substitution rejection,
strict metadata conflicts and unsupported measurements. These are scripted,
small/simulated software tests, not live biological qualification.

`R/historical_read.py` reopens the preserved USI-4 Session, Revision, artifact
inventory and accepted evidence: HTTP 200, zero provider/scientific calls and all
38 workspace files byte-identical. Source exception typing does not rewrite or
recompute historical results. New consumption of private qualified resources
continues to enforce their existing owner-version pins.

The first adjacent regression was intentionally interrupted to incorporate the
newly observed capability-description defect; it is not an acceptance pass.
Its log/result/source freeze remain under `R/regression-interrupted/`.
The final production snapshot's adjacent suite records **395 passed, three
failed, one warning in 704.07 seconds**, exit 1; no skips or test exclusions,
all `RUN_*` gates disabled, all 550 frozen source/test/static files unchanged.
It covers all annotation integration, capability-scoped planning, scientific
parameter admission, pending roles, FASTQ resources, qualified-resource discovery,
Web resource continuation and Session upload reuse. Exact argv, log and XML:
`R/regression-command.json`, `R/regression.log`, `R/regression.xml`.

All three failures reproduce on an untouched `git archive` of `d5320be` in
**1.69 seconds**. They are obsolete parameter totals in
`tests/application/test_initial_scientific_declarations.py`:
`test_reviewed_species_and_fragment_choices_are_offered_in_closed_schema`, and
`test_pending_scalar_schema_preserves_exact_pairs_and_scoped_choices` for
`species-answer_prerequisite` and `parameters-answer_parameters`. The actual
parameter sets are correct; totals are 29/26 rather than the historical 24/21,
and the numeric-only fixed total is also stale. Their baseline reproduction is
preserved in `R/baseline-failures*`.

Removed those fixed-size assumptions and retained/strengthened exact
projection coverage, uniqueness, offered finite choices and closed schemas.
The complete affected module then passes **56 tests in 26.56 seconds**, exit 0
(`R/parameter-final.log`, XML and command record). Only that test file changes
after the frozen adjacent run; production code is identical. All 398 selected
cases therefore have passing coverage across the adjacent and corrected-module
runs; there is **no claim of one final all-green 398-test invocation**, and counts
overlap. The warning is the intentional duplicate-cell-name negative fixture.
No full historical scientific suite or large biological recomputation is run.

Two harness limitations are retained in the evidence: the first control browser
launch raced server startup and failed before submission/provider dispatch; its
normal retry is logged separately. The initial resource-readiness diagnostic
failure and unavailable test-only `jsonschema` import are described above. Neither
is counted as a successful product trial.

### Preservation, Git state and remaining work

`R/preservation.json` finds **zero changes among 123,109 protected entries** under
the stated bounded policy, preserves all **1,274 eval files** and their inventory,
and checks exact hashes of acquired originals and the converted external H5AD.
The original production configuration, scientific references, historical Sessions,
R1/R2A evidence and checkpoints are not replaced. Native EpiZoo is unchanged.
Both isolated server and Firefox are stopped after acceptance.

Final branch is `main`; HEAD, local `main`, `origin/main` and freshly queried
remote `main` remain `d5320bebb6aaa7e812cc3f3f173b4fa04b5cf862`.
`git diff --check` passes. The index is empty. Nothing is staged, committed or
pushed. `R/final-git.json` records exact final hashes/status:

```text
 M src/agent/application/raw_handoff.py
 M src/agent/application/turn_decisions.py
 M src/agent/orchestration/annotation_registry.py
 M src/agent/orchestration/error_policy.py
 M src/agent/orchestration/planning_scope.py
 M src/agent/tools/analysis/external_marker_annotation.py
 M tests/annotation_integration/test_conversational_handoff.py
 M tests/annotation_integration/test_external_entry.py
 M tests/application/test_initial_scientific_declarations.py
?? docs/ua3.6.5-r2b-real-world-annotation-resource-readiness.md
?? evals/
```

**R2B is PARTIAL; UA3.6.5 remains ongoing.** The original external input cannot
complete the qualified primary-scope method, the normal production deployment
still lacks activated annotation resources/runtime, and both live scientific
answers fail their current generation contract. Control execution, accepted
results/downloads and recovery must not be presented as resolving those limits.
No universal H5AD/MAESTRO compatibility or independent biological accuracy is
claimed.

The highest-value next exploration is the shared scientific-answer boundary:
make ordinary supported explanations usable while preserving exact
subject/value attribution, using the captured two-response reproducer before
changing its factual checks. Separately review an explicit, provenance-preserving
primary-scope preparation capability for full publisher matrices and the concrete
resource/runtime deployment above. Accepted-clustering-to-original-count annotation
remains a valuable subsequent journey; this round does not establish or refute
that handoff.

## Authorized Part A closeout — 2026-10-19

The subsequent user request authorizes two separate commits and normal pushes.
Before Part A, HEAD/main/origin/main/remote main all matched `d5320be`; the exact
nine modified source/test files and this report matched `R/final-git.json`.
Review confirms the four repairs preserve scientific ownership and captured
source identities. The recorded regression remains applicable with its separate,
overlapping corrected-module run; it is not one all-green regression invocation.

New evidence root **F**: `/home/likeyi/agent-ua365-r2b-closeout-5np37837`.
The original R evidence is preserved. Final focused handoff and scope-error
checks pass **19 tests in 74.92 seconds**, exit 0 (`F/part-a.log`, `F/part-a.xml`).
`git diff --check` passes. Only the ten reviewed R2B source/test/report files are
included in the authorized Part A commit, titled
`Complete UA3.6.5-R2B annotation handoff corrections`. Evals and scientific data
are excluded. Product acceptance remains **PARTIAL**. The earlier unstaged Git
record above describes the original exploration checkpoint.

## Targeted follow-up — scientific answers, scope review and normal runtime

Part A was committed and pushed as
`b9a2819f4ef7480bf76a3f226c89e4ea08b50213`; HEAD, main, origin/main and remote main
were verified equal before Part B. All additional evidence is under **F** above;
R remains unchanged. This section supersedes the earlier prospective closeout
recommendations where the investigation below supplies new evidence.

### Shared answer boundary

`F/answer-diagnostics.json` independently reproduces both original responses as
`factual_binding/unbound_categorical_value`. Their reference IDs are valid. The
word `assigned` collides with status values inside detailed assignment claims.
The second response also calls “groups omitted” unresolved groups, although that
field measures summary truncation. Allowing its surrounding prose solely because
some referenced claims are valid would preserve an unsupported interpretation.

The existing Answerer now recovers from a factual-literal prose rejection by
using the same renderer to retain only complete selected claims, reviewed
meanings and qualified comparison pairs. It discards **all** model-authored prose
in that response, states that the explanation could not safely be used, and
marks the result `insufficient_evidence`. This is one provider call, not a retry,
new Answerer, word exemption or MAESTRO-specific rule. The strict renderer still
rejects unbound numerical/categorical literals. Invalid shapes, references,
substituted values/subjects, missing comparison operands or compatible pairs,
and changed accepted evidence remain failures. All later paragraphs are checked;
an earlier prose failure cannot conceal a later forged reference.

`F/answer-recovered.json` replays the exact two captured responses through the
new generation boundary without providers or science. Both return their selected
accepted facts. The unsupported equivalence between unresolved and omitted groups
is absent. Recovery preserves exact Revision/subject/value/source identities;
it does not certify that every question is completely answered or that arbitrary
model prose is entailed by evidence. Ordinary valid prose retains its existing
behavior. New tests cover annotation status terms, PCA/EpiZoo accepted metadata,
scientific numbers, malformed claims and cross-Revision discussions.

`F/accepted-answer-replay.json` additionally reloads the original Session's
actual accepted context, checks that its offered claims exactly match the
captured prompts, and replays both original responses with exact source and
Revision attribution. Its 71 workspace files remain unchanged.
`F/accepted-embedding-answers.json` independently uses preserved R2A PCA,
EpiZoo-derived Leiden and cross-Revision Leiden evidence. All three contexts
pass valid-answer, unsafe-prose recovery and forged-reference rejection checks;
72 historical workspace files remain unchanged. These are scripted answer
organization over real accepted evidence, with zero providers or science,
not additional live language-quality qualification.

### Native MAESTRO scope finding

The exact R2B input remains SHA
`c75bca64515710f6045a1c65d9f0dfb9fe9827102501856007b2bfcb58b33c81`,
1,004 cells × 82,579 peaks. No source bytes, axes, counts or identifiers change.
The 31 outside-primary peaks contain 6,062 of 22,279,127 counts. Their exact
coordinates and per-peak counts are in `F/native-scope.json`.

The pinned native Enhanced RP routine matches contig strings and reference gene
geometry; it does not consult Agent's QC classification. Instrumentation captures
the actual native transcript-by-peak weight matrix, independently of final
annotations. With original names, all 31 extra columns have zero native weights.
Full-input and primary-only diagnostic calculations give identical RP values and
exact gene/cell axes (28,307 × 1,004; 12,960,672 nonzeros). This comparison never
replaces the source H5AD or publishes an acceptance-tailored input.

That equality is **not** sufficient evidence for safe noncontribution. The
[official UCSC hg38 alias table](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/chromAlias.txt.gz)
provides exact accession/UCSC correspondences for all ten affected contigs. The
bounded diagnostic download is 13,393 bytes, SHA
`43aac315a93939c54d8b168ea8118fb7760b5e7ff8653f59dd712cfc7467be56`;
URL/date/publisher/role are in `F/alias-acquisition.json`. All coordinates fit the
corresponding lengths in the preserved hg38 FAI. Nine of these contigs have
transcripts under their UCSC names in the pinned gene resource.

A **diagnostic-only**, explicit alias substitution into native RP gives 55
nonzero transcript/peak weights across **26 of the 31 peaks**. Their weighted
transcript contribution sums to 11,244.555314815603 before gene aggregation and
normalization; this is not a cell count or a claim about label changes.
`F/alias-diagnostic.json` records each correspondence and weight count. No aliases
are inferred from spelling or repaired in production. This establishes that the
original zero contribution is a naming mismatch, not biological absence.

The existing qualified primary-only admission is therefore retained for this
round. It is an Agent qualification policy, **not a universal MAESTRO algorithm
requirement**. The safe error now says so explicitly. Admitting this full input
by quietly letting unmatched accessions contribute zero would be misleading;
using the alternative names changes scientific inputs, and a primary analysis
view must explicitly qualify and record its exclusions. Neither interpretation
is silently introduced. A broader owner contract needs explicit, qualified
coordinate correspondence or a reproducible declared analysis view. Native
algorithms, EpiZoo, original inputs and historical publication identities remain
unchanged. No new adapter or conversion framework is added.

### Format and interface corrections

The dual-pane interface is preserved. Composer guidance explicitly requires H5AD
for matrix analysis/annotation, explains that Cell Ranger `.h5` needs a scientific
conversion, and requires preservation of cells × features, counts and identifiers.
Renaming is not conversion. The existing H5AD owner and annotation error catalog
now explain the same boundary. No native H5 import or automatic conversion exists.

The annotation summary displays accepted cell/group counts, assigned/unassigned/
ambiguous counts and group labels, instead of prioritizing artifact-schema and
summary-truncation numbers. Full evidence remains available. Failed read-only
answers display “Answer unavailable · Accepted result unchanged”; execution
failures retain their execution status. These are presentation changes, not
scientific state transitions or new evidence.
Native Firefox exposed a narrow summary card that wrapped group records into
nearly vertical text. The existing group-summary card now spans the result
grid. `F/final-annotation-summary.png` and the computed layout check verify the
permanent CSS in a fresh browser/server, without injected preview styles.

### Deployment permission and runtime

`F/deployment-before.json` captures the normal operator and conda-environment
identities. Owner-only backups are under `F/secure-backup` (directory 0700,
files 0600); they contain sensitive environment state and are never committed.
No Agent Web service or active execution was running before activation. The
candidate configuration was first composed under an isolated workspace and the
native source/package pins were checked (`F/runtime-candidate.json`).

The normal `agent` conda environment now persistently sets the existing
`AGENT_QC_RESOURCE_CATALOG`, `AGENT_ANNOTATION_MAESTRO_ROOT`,
`AGENT_ANNOTATION_RSCRIPT` and `AGENT_ANNOTATION_R_LIBRARY` pointers. Only those
four environment entries change. Fresh activation verifies R 4.6.1, Presto 1.1.0,
Matrix 1.7.6 and dplyr 1.2.1, plus the pinned native source files
(`F/runtime-normal-environment.json`). No model, batch/device/dtype, credential,
resource binary or existing choice changes. The normal operator JSON remains SHA
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
The credential-bearing conda state file is restricted from its existing mode
0664 to 0600; existing credential values are preserved (`F/deployment-after.json`).

**Runtime availability is improved; a common annotation reference is not activated.**
The exact CIBERSORT-derived signature still lacks a verified deployment-sharing
permission basis. MAESTRO's GPL code license does not independently establish
permission for third-party signature bytes. The inspected
[Stanford CIBERSORTx terms](https://cibersortx.stanford.edu/runcibersortx.php)
restrict service use and supply no verified redistribution grant for this retained
file; this is not a finding that all local research use is prohibited.

Alternative review is recorded in `F/permission_review.json`. The inspected
[PanglaoDB publisher repository](https://github.com/oscar-franzen/PanglaoDB) does
not establish the required grant in its reviewed pages.
[CellMarkers](https://www.cellmarkers.com/docs/copyright) explicitly offers
CC BY 4.0 database content with exceptions, but its stated flow-cytometry/CyTOF/
microscopy marker purpose is not a qualified native MAESTRO PBMC gene-signature
replacement. No protein-panel conversion or biological qualification is invented.
No private resource is promoted, and no configured InputSet bypasses the trusted
sharing boundary. The smallest resolution is applicable permission for the exact
retained signature and intended deployment, or a qualified permitted replacement,
followed by the existing shared-resource promotion with pinned permission evidence.

The actual normal configuration was launched after ordinary conda activation and
used through Firefox/OpenRouter. Session
`5cb9ff19-83c8-4a15-8fb8-4212c7c99fc4` correctly captures the source, asks for count
slot/group provenance, accepts the reply, and reports the unavailable compatible
reference. Generation stays zero; no annotation runs. Restart restores the four
conversation texts and pending task exactly (`F/normal-restart-check.json`).
This is a verified deployment blocker, not a successful normal annotation trial.

### Live replay, exports and historical compatibility

A separate isolated Session uses the existing private reference InputSet; it
inherits the newly activated normal runtime environment. It is explicitly not
common-reference deployment acceptance. Real Firefox and the existing OpenRouter
GPT-5.6 Sol profile perform ordinary upload, clarification and grouping reply:

- Session `543f7e1b-8163-4efe-9fac-3dbb2a1f540f`.
- Run `turn-7baafbff9df3f06d90002edb3b0b702875d500e73f2909ced53ab726cc346bc2:run`.
- Revision `888b54d347f071b1102ed06f785e4d9fe8377f3172268e72dcc0b9bc3dfc696e`.
- Manifest SHA `1809cb83471bd9bfa067d8cb33af10be344181549fc58b755e249552fa5fd16a`.

The 484-cell historical control completes exactly two native annotation backend
calls: production and one independent verifier replay. Labels/counts match the
historical control, with validation still `not_assessed`. The two original
questions are then asked verbatim through the browser. Both display “Answer
available”, keep generation 1 and the same active Revision, and execute zero
additional science. The first response uses accepted prose and bound detail;
the second invokes the new evidence-only recovery. The latter presents 484
assigned cells, four groups and their exact assignment records, while explicitly
withholding the rejected explanation. These are useful accepted facts, not a
claim of fully conversational or concise presentation quality.

Firefox accesses the technical report (34,090 characters), assignment/marker
detail, and native annotated-H5AD download. The downloaded bytes have SHA
`5f0a255ab8e5efb2f4ff31d1ad4005b88c73612264d3e2a7faf62f657d76cad2`.
`F/control-science-check.json` independently reopens the actual download and
verifies exact cells/features, original obs/var, sparse X/raw values and dtypes,
and group-to-cell assignment propagation. The original source and delivered
content identities remain distinct under the established privacy-safe export.
No annotation biology is reconstructed by download verification.

The full external input is also replayed in Session
`530618f0-40bf-4c5b-b275-abee2d184d5d`. Its first real interpreter response wrongly
sets the H5AD as both matrix and separate grouping-file source; admission rejects
it. An ordinary follow-up explicitly states that the grouping is an obs column
inside the H5AD. The same uploaded source then reaches the owner and displays the
revised actionable scope error, with zero native annotation calls and no Revision.
No model-specific repair or forced planning follows the first semantic error.

`F/audit-summary.json` records **16 provider calls across eight live turns**:
eight interpreter, two capability selection, two Planner, two output selection
and two answerer calls. All use the configured model, without transport failures
or model fallback. These counts exclude offline captured-response replay and
are not billing measurements or a provider-reliability estimate. Source
qualification runs three times (control production/replay and external rejection);
the separately recorded native scope diagnostics are not Agent tool executions.

Read-only historical compatibility checks reopen the preserved USI-4 and original
R2B Sessions, Revisions, scientific artifact inventory and accepted evidence.
They preserve all **38** and **71** respective workspace files byte-for-byte,
with zero provider construction or scientific calls (`F/historical-read.json`,
`F/r2b-historical-read.json`). Protected-entry checks find zero changes among
**123,109** historical entries and **14,501** original R evidence files. The
normal smoke test adds a new Session/upload through ordinary APIs; it does not
rewrite any previously existing production record.

Fresh Firefox and server processes restore the control's eight conversation
texts, generation 1 and exact Revision, reopen the report, and preserve all
55 isolated workspace files (`F/recovery.json`). Provider and science audit
files remain byte-identical. The restored fallback text remains its originally
persisted wording; the final rendering copy clarifies that stronger conclusions
are unestablished without denying the accepted count facts. Existing history
is never rewritten. The browser also verifies the H5AD guidance and distinct
answer-failure versus execution-failure labels. Initial harness interruptions
and retries are retained in F; they are not counted as successful product turns.

### Targeted validation and disposition

The adjacent regression covers scientific dialogue, annotation integration,
rich Web/API/static behavior, uploads, scientific artifacts and planning scope:
**473 passed, seven failed, one warning in 905.59 seconds**, exit 1.
All 550 frozen source/test/static/R files remain unchanged during that run;
no `RUN_*` gates or test exclusions are used (`F/regression-command.json`,
`F/regression-result.json`, `F/regression.xml`). This is not an all-green run.

Five failures reproduce on untouched `d5320be` in
`tests/application/test_dialogue_detail.py` (**17 passed, five failed, 12.57 s**).
The old fixture omitted the manifest arguments that establish method identity.
Its default now supplies the explicit legacy-method arguments already used by
the module's current-contract test. Scientific readers and evidence admission
are not weakened. One static assertion is updated for the intentional
response-kind argument to `statusLabel`.

The remaining `test_first_threshold_execution_retains_exact_qc_for_followup_guidance`
failure reproduces unchanged on `d5320be` (**one failed, 43.13 s**): the existing
20-second polling window expires with the selection step `SUCCEEDED`, presentation
`finalizing`, and no error. Both current and baseline persisted turns subsequently
reach `activated` at generation 2 (`F/rich-timeout-completion.json`). No timeout
increase, application workaround or unrelated refactor is introduced. The
timing limitation is retained as an unresolved baseline test failure.

The final affected-module run passes **227 tests**, with that **one reproduced
baseline timeout**, one warning, in **265.83 seconds** (exit 1). It includes
dialogue detail, the shared answerer, categorical binding, cross-Revision
comparison, Web static/rich/upload presentation, rich API and external-entry
annotation tests (`F/final-focused.xml`, `F/final-focused.log`). All five
fixture failures and the stale static assertion are corrected. The separate
final answer-focused run passes **55 tests in 11.84 seconds**. These scopes
overlap; they are not summed into an all-green regression total.

Final preservation checks again find zero changes in 123,109 protected entries,
14,501 original evidence files and the 1,274-file eval inventory. As at intake,
large-file protection uses recorded size/mtime; stored hashes are checked where
present. This is not a fresh full-byte audit of all historical data. Native
EpiZoo source is unchanged. Both temporary Web and Firefox processes are stopped;
the normal deployment was initially stopped and remains so, with its reviewed
runtime settings persistent for the next ordinary launch.

**R2B remains PARTIAL.** The independently verified answer, format, error and
presentation corrections can close out separately from external-input and
shared-reference readiness. The highest-value next step is permission for the
exact signature and intended deployment, or qualification of a permitted
replacement through the existing resource system. Full external peak admission
additionally needs qualified, explicit coordinate correspondence or an explicitly
declared analysis view. No next scientific work unit is begun.

The reviewed Part B implementation, tests and this record are the sole payload
of the authorized second commit, titled
`Improve UA3.6.5 scientific answers and annotation readiness`. Pre-commit
HEAD/main/origin/main/remote main all equal Part A `b9a2819`. `git diff --check`
passes. No eval, dataset, resource binary, checkpoint, operator file or credential
is staged. `F/final-git.json` records the resulting commit and post-push branch
agreement; the commit identity is also returned in the closeout response.
