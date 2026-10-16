# USI-4 — MAESTRO conversational application handoff

Sections A–I retain the original acceptance record. Section J records the later
resource-access simplification and shared-resource closeout and supersedes the
original requirement for an operator-admitted complete annotation specification.

2026-10-16 (Asia/Shanghai). All new acceptance evidence is outside Git at
`/home/likeyi/agent-usi4-_sz1791j` (**R**). No commit or push is authorized or performed.

## A. Baseline and preservation

Initial HEAD, local main and origin/main:
`e2c477d4c5848ddce7a818513ca579f6ee5732ca`,
`Complete USI-3C scientific defaults and practical input readiness`.
A read-only remote-main query also returns this commit. The index and tracked
tree were clean; `?? evals/` was the sole initial status entry.

`R/baseline.json` records tracked-file SHA-256 and bounded size/mtime/mode
inventories for existing evals, UA3.6.4c direct-peak acceptance, scientific
resources and historical scientific acceptance. These inventories are not a
fresh full-byte audit of every biological dataset or checkpoint. Consumed real
source and resource identities are separately checked. New fixtures, derived
resources, uploads, Runs and Sessions remain under R. No production operator
configuration, existing InputSet, scientific reference or historical result is
replaced. No scientific download, package installation, GPU use or model training
is needed; approximately 15 TiB free disk was observed before acceptance.

## B. Existing capabilities reused

The existing public `annotate_scATAC_cell_types` remains the only annotation tool.
Registry size remains 23. The direct method and resource contracts are
`maestro-enhanced-count-weighted-peaks.v1` and
`scatac-annotation-peak-resources.v1`. The frozen historical canonical/binary
method remains separately recoverable. USI-4 does not remove another cCRE gate:
UA3.6.4c already removed that obsolete requirement for direct peaks.

The existing scientific boundary accepts a pinned source H5AD, reviewed
specification, count slot, exact obs group column or ordered groups TSV, and an
output directory. `external_marker_annotation` qualifies counts/coordinates/groups;
`annotation_contract` validates pinned resources and compatibility;
`scatac_annotation` owns publication, independent backend replay, exact cell
propagation, derivative conservation and authority. Native MAESTRO Enhanced RP,
the pinned R/Presto marker adapter, signed signature scoring and conservative
null/tie assignment policy are unchanged. Historically bound Python/R files and
EpiZoo are unchanged.

The application reuses immutable upload registrations, `ScientificInputSet`,
the USI-3A catalog/compatibility/unique-choice/explicit-choice/additive-continuation
mechanisms in `raw_handoff`, shared scalar declarations, existing role
prerequisites, Planner/compiler/preflight, ordinary RunStore execution,
Session/Revision persistence and generation checks. Annotation delivery and
report delivery retain their existing pinned HTTP endpoints. No second registry,
reference cache, workflow engine, scientific implementation or download subsystem
is introduced.

## C. Reproduced gaps and minimal changes

1. The existing direct scientific route required manual reviewed-InputSet selection
   and explicit grouping inputs. The single-H5AD browser wire offered neither the
   annotation task declaration nor source-bound grouping observations. Existing
   USI-3A matching only covered raw preprocessing references. The new annotation
   variant uses the same `request_roles`/`answer_roles` admission, matching and
   durable continuation. It binds the selected H5AD explicitly, preserves source
   identity, and asks only for unresolved scientific declarations/choices.
2. Categorical obs columns could not establish clustering semantics, and no safe
   owner observation supported a grouping question. A small scientific-owner
   helper observes bounded obs structure and declared metadata from the exact
   registered bytes. It does not inspect integer values to infer count history.
   It reports eligible column names without their cell values. Unqualified or
   competing group roles require a targeted choice; it does not invent clusters.
3. A whole-result question could not obtain group marker rationale: detail
   projection required a named group, while the interpreter correctly rejected
   an ungrounded `subject=3` for “Which groups were annotated as T cells?”. The
   existing evidence owner now projects a bounded set of accepted group records
   and their published primary-candidate support. Whole-result answering can use
   those exact records without predicting a subject before seeing evidence.
4. Initial annotation completion used generic execution text. It now explains
   group-derived marker inference, the derived H5AD and report downloads, and the
   lack of independent biological validation. No additional answerer/provider
   call or scientific claim generation is introduced.

The single-H5AD submission retains its existing execution binding. A captured
candidate view of that exact registration lets typed annotation admission use
existing roles without changing ordinary inspection/EpiZoo routing. Legacy
completed retries retain their original submission shape. The browser change
only recognizes annotation prerequisites when attaching a requested supplement.
The new declaration parser enforces the schema's 128-character single-line text
bound. A selected source must exist in the captured attachment set before the
scientific observer can run. Neither check qualifies scientific input values.

## D. Resolution, parameters and scientific correctness

Source species, assembly, measurement and tissue context may come from explicit
typed declarations or recognized, source-bound H5AD metadata. Conflicts fail
closed. Metadata describes declared processing history; it does not independently
reconstruct that history or prove biological suitability. Missing context remains
a question and is never inferred from resource availability or a filename.

The current direct owner supports human/hg38 raw
`cellranger-atac-cut-site-counts.v1`, sparse CSR/CSC, exact finite nonnegative
integral counts within its float64 exactness bound, and column-aligned
zero-based half-open `chrom:start-end` features in qualified primary nuclear
contigs. Fragment-overlap counts, binary accessibility, normalized/TF-IDF values,
other species/assemblies and arbitrary coordinate conversions remain outside
this qualified method. No canonical feature vocabulary or historical fragment
Run is required. Source X, raw, layers, identities and admitted metadata retain
the existing conservation contract.

`count_slot` and `group_column` stay on the existing exact-string parameter path.
Valid explicit values override captured scalar choices; otherwise captured values
are retained, followed by recognized source or reviewed-resource declarations.
No new numeric/default policy is introduced. Explicit invalid values are retained
for rejection or correction, never silently replaced. Decay, normalization,
marker thresholds and assignment policy remain fixed method constants; an
unregistered `decay` operand is rejected. No numeric zero-valued annotation
parameter is exposed by this owner.

An explicit obs column or TSV must cover exact ordered cells and contain at least
two valid groups. Optional H5AD `uns['grouping']` metadata can associate a column
with `{role: 'clustering', provenance: <derivation description>}`. Only structurally
valid columns with that explicit role are eligible for automatic unique selection;
this is declared grouping provenance, not experimental validation. Scalar historic
`uns['grouping']` descriptions and arbitrary categoricals do not establish a
column association. Distinct eligible choices clarify. Sample/batch/condition
columns are never automatically classified as clusters. A changed pending group
choice requires matching provenance; an absent grouping asks for a column or TSV.
No implicit EpiZoo/native clustering is added.

Resource matching uses the same operator InputSet catalog and the annotation
owner's pinned specification. Method, species, assembly, measurement, tissue
context, namespace and gene/signature/scope contracts remain authoritative.
A unique compatible resource binds automatically; explicit/captured choices are
retained and incompatible choices do not fall back. Distinct compatible resources
clarify. Additive provisioning can complete the same pending task without changing
captured definitions or sources. Exact aliases use the established deterministic
alias policy. The provider receives safe resource context, not paths or digests.

Agent binds the selected reviewed specification to the task's grouping provenance
in an immutable private specification. This is an application resource binding,
not scientific authority. Normal owner execution establishes the result authority.
It preserves the scientific resources' exact pins and method identity; no resource
is promoted into a production default. An operator deploying this capability
must admit an applicable qualified specification through the existing InputSet
configuration. The accepted real example uses UA3.6.4c specification SHA-256
`9c34ca0c5d475ed0a02c608cd1f3b21283f1e332702c435b4894f59d89388b88`.
The isolated acceptance configuration is not an automatic production activation.

This was the initial USI-4 resource-access boundary. Section J records its
subsequent correction for ordinary uploaded gene/signature resources; the earlier
acceptance evidence and limitations remain recorded here as history.

## E. Fresh real-data application evidence

The preserved official 10x PBMC 500 Next GEM / Cell Ranger ATAC 2.0.0 material
is verified against the [UA3.6.4c acquisition and preparation record](ua3.6.4c-native-peak-h5ad-acceptance.md).
The original prepared H5AD is 91,072,184 bytes, SHA-256
`72e581c3f027cc9330098f655335b6cc9406d39c136c882d4e2e95768c2efe38`.
It contains 484 cells and 65,887 explicitly selected primary nuclear peaks in X;
all 65,908 publisher features remain in raw. Publisher graph clusters are retained
in `obs['cellranger_graphclust']`. Historical data acquisition and preparation are
reused evidence; USI-4 freshly executes annotation on those exact bytes.

`R/real_workflow.py` uploads the real file through the session-scoped HTTP upload
and normal single-H5AD submission. The natural request names human PBMC context,
MAESTRO, suitable grouping, obs labels, report and downloads. The user supplies
no InputSet, resource ID, digest, manifest or internal scientific declaration.
Transport handles are managed by the ordinary application/harness. A focused
question retains the original task and asks for count slot, grouping column and
derivation provenance. The reply names X and the publisher clustering. The sole
compatible marker/reference resource resolves automatically.

Only interpreter/Planner/output-selection decisions are scripted. Actual sparse
native RP, R/Presto markers, signature annotation and verification execute.
Instrumentation records **two backend calls: one production and one inside one
annotation-owner verifier entry**, with no historical binary backend. There is no
matrix adoption, upstream fragment reconstruction, EpiZoo or optional validator.

| Publisher group | Cells | Primary label |
| --- | ---: | --- |
| 1 | 207 | Monocytes |
| 2 | 114 | ActNK |
| 4 | 57 | MemoryBcells |
| 3 | 106 | NaiveCD4Tcells |

All 484 cells are assigned; unresolved and ambiguous counts are zero in this
particular dataset. These are group-derived method/reference inferences with
`validation_state=not_assessed`, not per-cell independent predictions or biological
ground truth. Synthetic owner tests cover null/tie behavior separately.

Session `usi4`, Run
`turn-d0d0cdc6fba0a640bc26dad14f892f10b2cb28217ae7b5296a5402bf0ec34cca:run`,
Revision `0e8a3f267e51e600a5e15b8cfa42d03940e0da1758943cc740b551d949d32733`.
Authority SHA-256:
`815470deaa46b64179728876540354f504b3e21c7a27d59c70c9d1e4e90a3878`,
241 file bindings, no canonical matrix upstream authority.

Reopened output checks exact cell/feature order, original obs/var, X and complete
raw. The existing owner derivative checker validates every group-to-cell
assignment and source conservation. HTTP returns the actual pinned annotated
export: 91,020,412 bytes, SHA-256
`c4e02461c33f8e82c123ad2885f19a0207a552fca1126711cd9e41c8b9072d0b`.
The output H5AD reopens and agrees with publication; source bytes remain exact.
Report delivery and annotation-rationale evidence return HTTP 200. The export
retains the existing omission of the owner's private source locator and its
separate delivery provenance. No original file is overwritten.

The whole-result follow-up now cites exact accepted group 3 assignment and marker
records: 106 cells, NaiveCD4Tcells, positive BCL11B/FLT3LG/IL7R/ITK/LEF1/WNT7A.
It performs zero additional science and creates no Revision. New whole-result
detail retains group/candidate identity, signed evidence and coverage, limits
records in publication order, and explicitly limits support to primary candidates
of displayed groups. Other candidates require the existing group-specific detail.
Omissions never establish biological absence or calibrated confidence.

Fresh-process recovery preserves **38 workspace files** byte-for-byte, with
provider construction and scientific entries forbidden. Active result, evidence,
artifact handles and terminal resume remain available. The new synthetic HTTP
reannotation witness changes X to `layers/raw_counts` using the same registered
source, creates a child Revision, preserves the prior Run/artifact, navigates back,
downloads the second result and reopens without providers/science. This witness
qualifies revision/parameter software behavior, not additional real biology.

A separate synthetic HTTP check supplements a pending task with an ordinary
uploaded `attachment` containing the ordered groups TSV. It preserves the original
H5AD and request, binds `groups_path` without an obs-column binding, and completes
the existing annotation owner. The upload representation is neutral; the typed
interpreter declares its scientific groups role.

Read-only fresh-process checks also reopen both preserved UA3.6.4c Sessions:
the historical canonical/binary result (30 workspace files) and direct-peak
result (46 files). All bytes remain unchanged with provider construction and
scientific execution forbidden. `R/historical-reads.json` records their exact
Session, Run and Revision identities. These are historical accessibility checks,
not new executions or requalification of historical biology.

The first real audit incorrectly expected an empty upstream mapping rather than
absence of a matrix authority; science had already succeeded. It was corrected
and auditing resumed over the accepted result without rerunning science. An
initial scripted subject-3 answer was rejected because the question did not name
that subject. A subsequent whole-result scripted answer initially selected too
many unrelated claims and omitted the T-cell marker claim; the final scripted
answer selects the exact T-cell records. All attempts remain in R. Neither harness
correction is presented as a new biological execution or live-provider success.

## F. Live-model smoke and acceptance limits

One initial interpreter call uses the previously configured OpenRouter
`openai/gpt-5.6-sol` profile after activating the existing agent environment.
Credentials are never printed or copied. The prior acceptance-only reservation
ledger is copied to R and tightened to one total call, one call per turn and
USD 1 maximum, with zero SDK retries and no model fallback. Its conservative
reservation is USD 0.169968, not a claim of actual charged cost.

The provider returns **HTTP 404 / PLANNING_PROVIDER_ERROR**, with no completion.
`live-request.json`, `live-ledger.json`, `live-result.json` and `live-smoke.log`
retain the bounded attempt. This establishes neither language interpretation
quality nor a live request-to-result chain. No second provider/model is tried.
Comprehensive native-browser/live-provider product acceptance remains UA3.6.5.
USI-4's successful real science/application path uses scripted language decisions.

## G. Regression and final assessment

Focused handoff, reannotation and new whole-result detail checks pass **13 tests
in 56.67 seconds**. A subsequent explicit-over-captured count-slot refinement
passes its **two resource-continuation tests in 12.40 seconds**. Earlier development
failures and fixes remain in R; counts overlap and must not be summed.

The initial combined detail/dialogue/handoff run records 89 passed and six failed
in 103.42 seconds. Five failures independently reproduce on the exact baseline
archive (16 passed / five failed in 13.04 seconds). The sixth was the new test
using that same incomplete legacy fixture; its current-contract variant includes
the existing required method arguments. Historical test expectations are retained.
The five baseline failure names and final affected-domain results are recorded
at closeout below.

`R/regression.py` records exact argv, disables all RUN_* gates, uses isolated
pytest/Matplotlib/Numba paths and runs the selected annotation, input, resource,
continuation, conversation, HTTP, delivery and static-browser tests. It hashes
all source/test files before and after. No individual test is excluded and no
historical eval collection or full repository suite is run.

The affected-domain run finishes with **822 passed, one skipped, nine failed,
two warnings in 915.66 seconds**, exit 1. All **543 frozen source/test files**
remain unchanged during this run. The skip is the explicitly gated
`test_chat_missing_member_real_context_producer_matrix_download_restart`
(`RUN_CHROMAP_QUALIFICATION=1` required). Warnings concern intentionally duplicated
obs names in a negative fixture and pytest's JUnit `record_property` format.
There are **no exclusions**. This is not reported as an all-green regression.

All nine failures reproduce on an unmodified `git archive` of the exact baseline:

- `test_dialogue_detail.py`: `test_annotation_exact_detail_and_no_work`,
  `test_oversized_record_is_explicitly_omitted`,
  `test_detail_restart_subject_followup_and_comparison`,
  `test_no_file_discovery_and_exact_candidate_selection`, and
  `test_changed_detail_during_generation_is_rejected`.
- `test_interactive_rich.py`:
  `test_existing_supported_detail_owner_is_reused_without_science[False-annotation_rationale-3]`.
- `test_initial_scientific_declarations.py`:
  `test_reviewed_species_and_fragment_choices_are_offered_in_closed_schema`, and
  `test_pending_scalar_schema_preserves_exact_pairs_and_scoped_choices` for
  `species-answer_prerequisite` and `parameters-answer_parameters`.

The first six use an annotation manifest fixture without the already-required
`arguments` key. The remaining three have obsolete fixed parameter totals:
the baseline exposes 27 rather than 24 in the general/parameters schema and
24 rather than 21 in the species continuation schema. Baseline detail tests
record 16 passed / five failed; a separate exact four-case baseline reproduction
records four failed in 2.10 seconds. USI-4 changes no historical expectation and
does not repair these unrelated baseline fixtures.

After the frozen run, only the new handoff source-membership/text-bound checks
and three focused tests are added: grouping-file supplementation, oversized
typed text, and unoffered-source rejection before scientific observation. The
final focused result is recorded below. Fresh-process real-result recovery is
also repeated after these changes and again preserves 38 files with zero
providers/science.

Reproduction commands use the existing agent environment and isolated evidence
root (the scripts themselves record scientific runtime configuration and exact
test arguments):

```bash
R=/home/likeyi/agent-usi4-_sz1791j
PY=/home/likeyi/anaconda3/envs/agent/bin/python
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$R/regression.py"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m pytest -q \
  tests/annotation_integration/test_conversational_handoff.py \
  tests/application/test_dialogue_detail.py::test_whole_annotation_rationale_preserves_group_marker_correspondence \
  --basetemp="$R/final-focus-corrected-temp" -o cache_dir="$R/final-focus-cache"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B -m pytest -q \
  tests/annotation_integration/test_conversational_handoff.py::test_unoffered_source_fails_before_observation \
  --basetemp="$R/unoffered-final-temp" -o cache_dir="$R/final-focus-cache"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$R/real_workflow.py"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$R/real_workflow.py" --audit
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$R/real_workflow.py" --reopen
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$R/historical_reads.py"
git diff --check
```

`regression-command.json` contains all 28 selected test paths and the exact argv;
`regression.xml` and `regression.log` retain individual outcomes. Baseline commands
run from `R/baseline-source` with its own `PYTHONPATH=src`, selecting the named
original tests; logs are `baseline-details.log` and `baseline-extra.log`.
The first final-focused command misspelled the new test node and collected no
tests; `final-focus.log` retains that exit-4 invocation. The corrected node-selection
run is `final-focus-corrected.log`. It records **15 passed and one failed in 66.81
seconds**: the new unoffered-source test expected a terminal failure, whereas
the existing invalid-decision boundary correctly returns a clarification before
any scientific observation. Only this new expected status is corrected, also
asserting `invalid_decision`, zero science and zero revisions. Its focused rerun
passes **one test in 3.87 seconds**, exit 0 (`unoffered-final.log`). Thus all 16
new/affected focused cases have passing results on the final implementation;
there is no claim of a single final 16-test green run. These harness errors are
not omitted or counted as passing execution.
`live_smoke.py` is the separate single-call provider attempt;
it is not part of the scientific execution commands above.

## H. Changes, preservation and remaining limits

| Files | Owning change |
| --- | --- |
| `src/agent/application/raw_handoff.py` | Annotation declarations, shared qualified-resource resolution, exact grouping/source binding and durable replay |
| `src/agent/tools/analysis/annotation_input.py` (new) | Bounded source-bound metadata/group observations under the scientific owner |
| `src/agent/application/turn_decisions.py`, `turns.py` | Typed interpreter schema, prompt context and ordinary role continuation |
| `src/agent/application/interactive.py`, `session_state.py` | Exact single-H5AD candidate projection and persisted annotation prerequisite |
| `src/agent/application/dialogue_execution.py`, `responses.py` | Existing execution handoff and truthful completion/clarification text |
| `src/agent/application/dialogue_evidence.py` | Whole-result accepted primary-marker support with bounded correspondence |
| `src/agent/web/static/app.js` | Existing pending-task attachment continuation recognizes annotation |
| `tests/annotation_integration/test_conversational_handoff.py` (new), `tests/application/test_dialogue_detail.py` | Focused handoff and whole-result evidence contracts; original expectations retained |
| This document (new) | Audit, scientific acceptance and limitations |

Final preservation uses the same file enumeration and `stat` metadata semantics
as the initial inventory. All **11,369 protected entries**, including **1,274
eval files**, remain unchanged with no additions in the inventoried roots.
The first closeout script accidentally used `lstat` and included directory
symlinks; its apparent differences were enumeration differences, confirmed against
the original `stat` values, and the corrected matching audit passes. Both records
remain under R. The production operator configuration retains SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`, matching the
previous acceptance record. Scientific owner implementation, EpiZoo, Registry,
Planner/compiler and all unrelated tracked files retain their baseline bytes.
No index changes, commits or pushes occur.

Concrete limits remain: the qualified direct method is human hg38 with declared
Cell Ranger ATAC cut-site counts and compatible curated gene/signature resources;
no arbitrary normalized matrix, new species/assembly, automatic clustering,
threshold tuning or biological validation is added. A resource is still an
operator-admitted qualified specification; acceptance does not modify deployment
defaults. Reannotation acceptance reselects the same registered source, without
uploading or adopting it again. It does not demonstrate implicit source inheritance.
The whole-result rationale is bounded and can require a specific-group follow-up.
Live interpretation remains unqualified after the provider's HTTP 404; no full
native-browser acceptance is claimed. The nine independently reproduced baseline
test failures remain visible for their owning follow-up work.

## I. Final disposition

**USI-4 COMPLETE** within the qualified direct-peak scientific scope and the
explicit scripted-language/real-science acceptance boundary above. The ordinary
upload and task handoff, focused grouping question, automatic qualified resource
binding, actual MAESTRO annotations, evidence follow-up, preserved H5AD download,
reannotation/navigation and durable recovery are demonstrated. Completion does
not assert live-provider reliability, production deployment or biological ground
truth. No new scientific algorithm or parallel application engine is introduced.

**USI-5 can proceed** after review of these unstaged changes, retaining the nine
baseline test failures as separately identified debt and UA3.6.5 as the owner of
comprehensive live/browser qualification. No USI-5 implementation has begun.

Final `git diff --check` passes. HEAD/main/origin/main remain at the accepted
baseline; the index is empty. Final `git status --short`:

```text
 M src/agent/application/dialogue_evidence.py
 M src/agent/application/dialogue_execution.py
 M src/agent/application/interactive.py
 M src/agent/application/raw_handoff.py
 M src/agent/application/responses.py
 M src/agent/application/session_state.py
 M src/agent/application/turn_decisions.py
 M src/agent/application/turns.py
 M src/agent/web/static/app.js
 M tests/application/test_dialogue_detail.py
?? docs/usi-4-maestro-conversational-handoff.md
?? evals/
?? src/agent/tools/analysis/annotation_input.py
?? tests/annotation_integration/test_conversational_handoff.py
```

`evals/` is pre-existing and unchanged. All other listed changes belong to this
handoff and its tests/documentation. Nothing is staged, committed or pushed.

## J. Resource-access and contract-simplification closeout

### J.1 Initial state and scope

This follow-up starts at the same HEAD
`e2c477d4c5848ddce7a818513ca579f6ee5732ca`, preserving every uncommitted USI-4
change listed above. New evidence is isolated at
`/home/likeyi/agent-usi4-resources-azazd0kq` (**S**). `S/start/` preserves the
starting tracked files and all three initial USI-4 additions. `S/baseline.json`
records their hashes and a protected-file inventory that additionally includes
the complete original USI-4 evidence root R. Nothing in R is used as a write
destination. No biological dataset download, installation, production
configuration change, commit or push is needed. The later permission audit
downloads only two bounded publisher metadata pages (39,758 bytes total), recorded
in J.5. Approximately 15 TiB free disk is available.

### J.2 Contract audit and decisions

The actual obstruction was `raw_handoff.resources/_resolve_resource` plus
`_prepare_annotation`: annotation could bind only the gene/signature files already
inside an operator InputSet's specification. The scientific `Resource` contract
itself does **not** require operator registration. It requires explicit declared
origin/release, species/assembly, context, namespace, semantics and pinned bytes.
The native backend accepts paths from that same contract. Existing attachment
registrations already provide immutable identity and Session isolation.

| Requirement | Classification, owner and action |
| --- | --- |
| Human hg38 raw Cell Ranger ATAC cut-site peak counts, exact sparse axes and integral values | A: existing direct scientific owner; retained unchanged. Integer appearance is not processing-history proof. |
| Exact grouping and independent derivation declaration | A/B: existing grouping owner and typed task binding; retained. No labels or clusters are invented. |
| Native MAESTRO refgenes transcript/exon input | A: annotation owner; retained. Uploaded files receive bounded structural validation of the exact supported table, strands, transcript/exon coordinates, exon counts/order and primary-chromosome correspondence. No arbitrary GTF/BED converter is added. |
| Candidate/gene signatures and namespace | A: existing `read_signatures` and scoring owner; reused. At least two entries per candidate remain necessary for the existing denominator. Duplicate entries retain their scientific weighting. Matching `gene_symbol` declarations and nonempty gene correspondence are checked; coverage is not biological validation. |
| Resource origin/release and tissue applicability | A/B: typed user declarations, persisted in the existing resource specification; retained. Missing facts clarify. Supplied provenance is labelled declared, not independently reconstructed or experimentally validated. |
| Exact genome chromosome lengths and primary-nuclear classification | A: existing qualified genome-reference owner; retained. No chromosome-name inference, alias repair or coordinate conversion. |
| QC bundle used as the existing chromosome-scope carrier | Historical representation, not a requirement to run QC. The already-qualified shared genome dependency is now exposed independently from gene/signature resources. No TSS, QC, cCRE or fragments computation is introduced. See the concrete scope limit below. |
| Operator-managed complete annotation InputSet | C: retained as an optional convenient complete resource choice. |
| Requiring every user's new gene/signature files to be manually admitted into that InputSet | D: removed at the existing handoff boundary. Uploaded roles can supply either component, or both, and Agent constructs the same `scatac-annotation-peak-resources.v1` specification. |
| Resource digests, internal paths and manifests as manual user prerequisites | D: removed from the uploaded-resource path. Trusted registrations supply exact paths/hashes; typed language declarations supply meaningful context/provenance. |
| Re-parsing uploaded gene/signature content during Session history reads | E: avoided. Owner admission produces a source-bound observation; deterministic replay checks its exact specification identity and shape without scientific IO. New execution retains ordinary owner production and independent result verification. |
| Private file access, source conservation, authority and lineage | B: existing upload registrations, Session ownership, exact-prefix supplementation, immutable publication and scientific authority; retained. Shared availability extends the existing reference catalog and operator promotion boundary, as described in J.5. |

The shared genome dependency is not optional science. An existing qualified hg38
scope must be available through existing configured references or the qualified
resource catalog, either as part of an annotation specification or through
existing QC-reference inputs. Catalog discovery constructs the internal InputSet
projection automatically; the researcher need not construct one. Other
arguments in a QC InputSet are not forwarded into annotation. It supplies only
the exact chromosome reference. This is independent of admitting a user's new
marker/gene resource, and requires no per-user or per-task operator registration.
If that genome dependency is absent, gene and marker tables cannot establish it;
the task stays pending with that exact missing dependency. This closeout does not
implement user replacement of genome-classification resources or claim that the
historical QC serialization is intrinsically necessary. Its replacement would
require source-backed chromosome qualification beyond these gene/signature files.

### J.3 Minimal implementation

`AnnotationUploads` is an optional member of the existing annotation declaration.
The interpreter binds explicit gene/signature file roles and their declared
species, assembly, biological context, gene namespace, publisher/derivation and
release. Unknown declarations remain questions. Uploads use the existing neutral
attachment endpoint and existing `request_roles`/`answer_roles` continuation;
there is no new upload route or resource store.

`annotation_input` owns the supported resource admission and specification
construction. Catalog resources and uploaded resources feed the **same** native
specification, backend, verifier and publication contract. Existing scientific
backend and contract files remain byte-identical. Uploaded components replace
only their explicit roles. Catalog entries may supply the remaining gene,
signature or shared chromosome dependency. Scientifically distinct retained
components clarify; exact identical remaining dependencies may use the existing
deterministic alias policy. Invalid explicit uploads never fall back to catalog
markers. A changed uploaded file role requires fresh origin/release declarations.

The native MAESTRO gene table has patch transcripts outside the supplied primary
chromosome dictionary. An initially overstrict development check rejected one
such transcript (`chr1_KN196472v1_fix`). That check was corrected before real
execution: retain every row unchanged; verify all transcript/exon structure and
the lengths of known chromosomes; require primary-chromosome gene coverage.
Unknown patch chromosomes cannot receive input peaks because the unchanged peak
owner requires qualified primary chromosomes. There is no silent row filtering,
chromosome relabelling or inference about patch classification.

Uploaded resource admission is bounded to 64 MiB per component and 200,000
transcript rows. These are computational admission bounds, not biological
qualification rules or general upload limits. The accepted native reference is
within both. The new owner observation distinguishes computational compatibility
and declared applicability from independent biological validation.

Private references retain their original Session-bound upload records. Derived
task specifications remain private. Successful qualification also registers a
reusable private reference through the existing local resource catalog; promotion
requires the separate operator review in J.5. Both original upload digests appear in the scientific specification,
result manifest and accepted authority file bindings. The optional declaration
extension is omitted from serialization when absent, preserving the exact
original USI-4 completed-record identities.

### J.4 Fresh real acceptance

Both fresh executions reuse the original verified 484-cell PBMC source bytes
identified in Section E. Language interpretation, planning and answer selection
are scripted; MAESTRO scientific computation is real. No live-model call is
made in this follow-up and no new live-provider qualification is claimed.

| Acceptance | Fresh evidence |
| --- | --- |
| A: existing compatible catalog resource | `S/catalog/real_workflow.py`: ordinary H5AD upload, grouping clarification, automatic resource selection and actual annotation; no manual per-task InputSet selection. |
| B/C: uploaded references and missing-resource continuation | `S/uploaded/real_workflow.py`: configuration contains **no annotation specification**, only the existing qualified genome scope. The first turn asks for gene/signature files and missing grouping declarations. The reply uploads both actual resources through Session attachment endpoints and continues the original source and task. |
| D: explicit uploaded resource over catalog choice | Focused HTTP test binds uploaded genes/signatures while a complete compatible catalog choice exists; accepted specification paths point to the uploads and preserve their declared provenance. Synthetic backend scope is explicit. |
| E: negative/integrity paths | Focused tests cover wrong species/assembly, missing context/provenance, unsupported namespace, malformed genes/exons, noncorresponding markers, distinct retained signatures, unavailable explicit choice, changed resource bytes during continuation, and another Session's private upload. |
| F: original acceptance preservation | Fresh-process read-only reopening of R's accepted result preserves all 38 workspace files, with provider construction and scientific calls forbidden (`S/original-result-reopen.json`). |

The uploaded gene table is the legitimate MAESTRO
`annotations/GRCh38_refgenes.txt` at revision
`74f10babf4125b98ef005df4186b14ab02e13bfd`, SHA-256
`9b18e56531155226450254aefd192bb7f49d03d74c5bbfdc53af5411a0d2fb37`.
The uploaded signatures are the preserved MAESTRO human immune CIBERSORT export
with column-header replacement only, at the same revision, SHA-256
`3c202410d4f46049a298784838e5f9d214fc67fcddf34536d1c3811e383b601a`.
Acquisition and transformation provenance is inherited from the accepted
UA3.6.4c/M13 evidence. Upload identity and current structural compatibility are
freshly checked, not inferred from filenames. The new observation records
28,307 gene symbols, 78,362 primary-chromosome transcripts and 28 candidates.

Each fresh execution records **one production backend call plus one backend
call inside one independent annotation-verifier entry**. Catalog execution takes
76.08 seconds; uploaded-resource continuation takes 78.05 seconds. Both assign
the same four groups and all 484 cells as the original acceptance. Full group
records, candidate scores/positive and negative marker evidence, and signature
coverage agree exactly across all three results (`S/scientific-equivalence.json`).
These remain group-derived inferences with validation not assessed.

Both outputs reopen with exact source cell/feature order, original observation
metadata, X and raw preserved. The original derivative verifier confirms exact
group-to-cell propagation. H5AD and report endpoints return HTTP 200, and the
whole-result T-cell marker question is answered from exact accepted claims with
zero additional science. Download identities:

| Run workspace | Export bytes | Export SHA-256 | Authority SHA-256 |
| --- | ---: | --- | --- |
| `S/catalog/real-workspace` | 91,020,412 | `7863248d97631e033707f9f99c74131d14b80a85847caa2125e4318f15d0f7e3` | `e76f326018a4c252470e3954fde76311d6db2351a5159776dca706f21338652f` |
| `S/uploaded/real-workspace` | 91,020,412 | `282687741512483cf5c2accaa6d733ea8c48053bb185c4cc4d552e9cd8869625` | `cf18d01ea7418c15cfceec99398f42410ed068440c10fcb6984f7702a1fd0ac8` |

Each authority has 241 file bindings. These isolated workspaces intentionally
use the same Session/turn names, so their Run/Revision identifiers match Section
E; workspace plus authority identifies the distinct executions. Paths/provenance
make export bytes differ without changing scientific results. `real-result.json`
in each directory records the complete arguments and identities.

Fresh-process reads and terminal resume preserve **36 catalog-workspace files**
and **44 uploaded-workspace files**, with zero provider construction and zero
science. The original R remains untouched. A synthetic pending-resource restart
also accepts missing provenance without reuploading any file, and verifies that
the original request continues. Resource admission is instrumented once per
successful uploaded-resource task; later reads do not repeat it.

After shared persistence was integrated, `S/final-uploaded/real_workflow.py`
freshly repeats the actual uploaded-reference path in another isolated workspace.
It completes in **80.90 seconds**, again with exactly two backend calls (production
and independent verification) inside one verifier entry, no legacy backend, and
241 authority bindings. Its export is 91,020,412 bytes, SHA-256
`2aa258ed7f098f0bdca89005b8d796971fd780262898e63de14785ffd29f3f4e`;
authority SHA-256 is
`efb230d51ed6f597900644c83b39471fdb202ac08b1bea057c8ee7579f3eeb7c`.
All group, cell, candidate-evidence and signature-coverage records agree exactly
with the original and both earlier follow-up runs
(`S/scientific-equivalence-final.json`). Its source/derivative, report, download
and evidence checks pass. A fresh process preserves **48 workspace files**,
discovers the private qualified reference for `usi4` but not another scope, and
checks its pinned identity with providers and resource/scientific qualification
forbidden. The exact execution script is retained as `executed-real_workflow.py`;
the later script adds only these read-only assertions. The real resources remain
private. Final-code recovery also preserves the original R's 38 files, the catalog
witness's 36 files, and both historical UA3.6.4c Sessions' 30 and 46 files.

The final implementation is additionally executed in
`S/closeout-uploaded/real_workflow.py`, after the last publication-location and
upload-picker corrections. This is a distinct fresh scientific run, not a retry
of an accepted Run. It passes in **80.74 seconds**, with one production backend
call and one backend call inside one independent verifier entry. Export size is
91,020,412 bytes, SHA-256
`726e9adc36cb07481fbc61547dd6d9caaebf24654762f92c01730b441900a4c3`;
authority SHA-256 is
`c0213041edc944cfb0acfb5cd3c8aa3f642fbab182349be7839e1e7983e3324c`,
again with 241 bindings. `scientific-equivalence-closeout.json` records exact
group/cell/candidate/coverage agreement with all four earlier real results,
including original R. H5AD preservation, both delivery endpoints and the evidence
follow-up pass. Fresh-process recovery again preserves 48 files, with providers,
annotation and resource qualification forbidden. All follow-up real executions
are explicitly separate; their call counts are not presented as one execution.

### J.5 Shared persistence, access and cross-task audit

The baseline already contains `LocalResourceAdmission`, immutable registrations,
private qualified regulatory references, `reference_supplementation.promote`, and
`shared_genome` genome-only approval and per-task feature binding. The existing
genome path reuses exact FASTA/FAI bytes for another compatible task without
copying the genome or inheriting a previous dataset's feature vocabulary. It
already rejects incompatible species, assembly, chromosomes and feature bounds.
No second catalog, cache, download manager or scientific framework is needed.

Two concrete gaps were corrected at those owners. Annotation references were
absent from persistent resource discovery, and new deployment approvals recorded
only free-text operator/provenance/reuse statements, without a separate pinned
redistribution record. `shared_resources` implements two closed qualification
variants inside the **same** local catalog: the existing direct annotation
specification and the existing human/mouse QC-reference bundle. It is not another
resource store or a new scientific format. Genome FASTA/FAI retain their existing
`shared_genome` registration and binding implementation.

| Resource | Qualification and reuse boundary |
| --- | --- |
| Genome FASTA/FAI | Existing reference-owner sequence/index qualification and exact genome identity; existing genome-only promotion, species/assembly compatibility, and private feature derivation for each new task. |
| Gene annotations and marker/signature resources | Existing annotation-owner qualification and exact component pins. A reusable specification replaces task grouping text with an explicit per-task requirement; current grouping is always bound again from the consuming task. Uploaded references persist privately without a separate operator InputSet. Gene and signature roles can still be replaced separately using the typed continuation. |
| Existing QC-reference bundles | Existing QC owner independently checks the supported bundle's source derivation before private registration. The same qualified pointers can supply QC inputs and annotation's chromosome-scope dependency. No new GTF parser, new species support or arbitrary reference format is introduced. |
| Deployment sharing | Existing trusted operator provisioning only. New promotions require exact scientific-file hashes, publisher-source and permission URLs, a local permission-evidence file and digest, and an explicit reviewed reuse basis. Evidence must cover every shared scientific file; only exact known implementation paths are excluded. A `.py` filename does not grant an exception. |

Discovery filters private registrations by Session scope before presenting any
resource choice. Available compatible shared references enter the existing
InputSet matching and unique-choice path automatically. The interpreter sees
safe labels and scientific context, never private paths or permission records.
Private references are reusable by later tasks in the same Session; another
Session cannot select them by guessing their resource handle or attachment ID.
Public promotion creates a new immutable deployment record without changing the
private upload or original registration. Existing approved references remain
readable under their original explicit review; they are not silently upgraded
to the new evidence format. Every **new** promotion uses the strengthened review.

Permission review is an operator responsibility at the existing trusted local
deployment boundary. The software checks completeness, exact identity and
integrity of that review; it does not decide a resource's license from a URL,
filename or user assertion. There is no public HTTP promotion route and no
interpreter decision that can call this API. This closeout does not establish
redistribution permissions for the real MAESTRO/CIBERSORT resources and does not
promote them or alter production defaults. Positive annotation-sharing fixtures
are clearly labelled simulated author-permission/software tests, not public
biological qualification. The real genome witness below has a separate publisher
review. Deploying a real shared resource requires operator review of its
actual source/release, scientific suitability and redistribution terms, followed
by `reference_supplementation.promote` (or existing genome-only promotion) with
the pinned permission evidence. No researcher-side InputSet is then required.

Catalog listing is metadata-only. A selected reference is checked against exact
manifest/component/owner-code pins and permission evidence before new scientific
consumption. Unsupported contract versions, changed content or changed owner
implementation fail closed; a new immutable qualification is required. Ordinary
history reads and terminal retries do not repeat qualification. The annotation
registration pins only its consumed scope manifest, genes, signatures and owner
implementation: it does **not** reintroduce an upstream FASTA/cCRE availability
gate removed by UA3.6.4c. QC and genome registrations retain their own existing
source qualification and conservation requirements.

The HTTP sharing witness uploads and qualifies simulated references in one
Session, reopens the application without any configured InputSets, reuses the
references in a later task, and executes annotation in a second independent
Session after explicit simulated promotion. The second Session uploads only its
own H5AD. Normal production and independent verification each run once per task.
Separate checks cover private rejection, missing/incomplete permission evidence,
permission tampering, version and scientific incompatibility, QC-to-annotation
scope reuse, mutated QC genome dependencies, configurable attachment roots, and
annotation reuse without unconsumed upstream FASTA/cCRE files. Existing genome
tests retain distinct feature binding, restart and negative bounds checks; their
promotion calls now supply explicit simulated permission evidence. These are
tests of the existing Session access boundary, not a claim of a new authenticated
multi-tenant account or cross-deployment sharing service.

The shared-resource audit also reproduces an existing picker defect: a promoted
genome whose original FASTA lives in the completed-upload directory was included
in the ordinary upload list, then rejected by the upload owner. The existing
`uploads.choices` now projects only its supported ordinary input types. Shared
references remain discoverable through the reference catalog. The regression
asserts HTTP 200 from `/api/v1/resources` after promotion, retains the H5AD choice,
and excludes the genome catalog record. Publication also uses the approved root
containing the explicitly bound uploaded component, including when several roots
are approved. Internal qualified-resource handles require their exact generated
shape, so a normal operator label such as `qualified-human` does not accidentally
claim a catalog identity. None of these corrections changes scientific algorithms.

`S/public-genome/accept.py` additionally qualifies and shares the **real preserved
UCSC hg38 genome** inside an isolated acceptance catalog. The M11.6b source review
already establishes exact full decompressed identity with the official UCSC
distribution and its published archive MD5. Current publisher review uses
[UCSC's data policy](https://genome.ucsc.edu/license/) and the
[hg38 distribution README](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/README.txt):
UCSC permits use of downloaded data subject to source restrictions; the assembly
README records NCBI's no-restriction distribution policy and its general
third-party-rights caveat. The recorded decision is limited to this assembly
sequence and its mechanical index, not annotation tracks, software, liftOver
chains or marker signatures. This is an explicit source-specific review, not a
conclusion inferred merely from public download access.

The exact qualified identity is GRCh38 initial / `GCA_000001405.15`, 455 total
contigs. FASTA is **3,273,481,150 bytes**, SHA-256
`5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51`;
FAI is **19,381 bytes**, SHA-256
`3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21`.
The existing reference owner freshly checks full sequence/index correspondence
and exact ordered features. The existing genome-only promotion creates shared
identity `a770b6e1f2c256af19f86ddf6585b21a670480a858113051306f6e26e2939a2e`.
Two independent scopes discover the sole compatible genome, bind different
two-feature sets and preserve identical FASTA/FAI identities without copying or
reuploading those files. Their counts and cells are **simulated**, using real
compatible feature coordinates; this qualifies genome/reference reuse, not new
experimental biology or a live application-language flow. Wrong assembly is
rejected. The complete witness takes **192.38 seconds**, with zero annotation,
model training or provider calls. A fresh process preserves **20 workspace
files** with reference qualification forbidden.

The two metadata downloads use 30-second timeouts and 64-KiB per-file bounds;
their URLs, dates, exact sizes and hashes are in `S/public-genome/result.json`.
The pinned permission review SHA-256 is
`aee47f699f3a11c32d12d56887e3447d0edfd5545fc00366aa79105e1a55e067`.
The source genome, FAI and original source-review file remain unchanged. No
production catalog or default is changed. Activating this exact shared genome in
a production deployment remains an explicit operator provisioning step using the
same reviewed identity and evidence; subsequent users do not construct InputSets.

### J.6 Validation and disposition

The final complete focused invocation passes **54 tests, two warnings in 205.81
seconds**, exit 0. All **546 frozen source/test/static/resource files** remain
unchanged. This is one green invocation covering the complete conversational
handoff, uploaded references, shared resources, existing genome/reference
supplementation and whole-result marker correspondence. The warnings are existing
`pkg_resources` and Louvain deprecations. Exact command, JUnit outcomes, hashes
and log are `S/final-focused-command.json`, `final-focused.xml`,
`final-focused-frozen.json` and `final-focused.log`.

The final affected-domain invocation finishes with **863 passed, one skipped,
nine failed, four warnings in 1,117.68 seconds**, exit 1. All **546 frozen files**
remain unchanged. The 29 selected test paths cover annotation, input/resource
admission, conversation, continuation, shared reference/genome handling, HTTP,
delivery, upload and static-browser behavior. Every `RUN_*` gate is disabled;
there are no individual exclusions, no historical eval collection and no full
repository-suite claim. `S/regression-command.json`, `regression.xml`,
`regression.log`, `regression-frozen.json` and `regression-result.json` contain
the exact invocation and individual outcomes.

The failure-name set is exactly the same nine failures independently reproduced
on baseline HEAD in Section G (`S/regression-baseline-comparison.json`). Six
fixtures lack the already-required annotation `arguments` key; three expect
obsolete scalar totals. The skip remains
`test_chat_missing_member_real_context_producer_matrix_download_restart`, gated
by `RUN_CHROMAP_QUALIFICATION=1`. Warnings concern the intentional duplicated-obs
negative fixture, JUnit `record_property` formatting and the two existing
dependency deprecations above. This is **not** an all-green regression.

Development evidence is retained separately and counts overlap. The initial
handoff run has 14 passing tests and one old wording assertion expecting
operator provisioning; it was updated to the requested upload instruction. The
pre-sharing full focused run passes 31 tests in 132.35 seconds. When the user
added sharing requirements, the first affected-domain attempt was interrupted
after 280.07 seconds, before editing; it is not a completed regression result
(`S/interrupted-regression/`). All 544 files frozen for that attempt were unchanged.
An initial sharing test run has 17 passes and one harness failure from expecting
a nonexistent `turn` HTTP response wrapper; the assertion was corrected. The
later pre-picker complete focused run passes 53 tests in 158.17 seconds.

The pre-picker affected-domain run records **862 passed, one skipped, nine failed,
four warnings in 1,063.59 seconds**, exit 1, with 546 frozen files unchanged.
It is preserved in `S/pre-picker-regression/`. All nine failures are the exact
baseline failures listed in Section G. The separate
`sharing-choices-repro.log` demonstrates the real upload-picker defect, with
`LOCAL_RESOURCE_BINDING_INVALID`; it is not dismissed as baseline debt. Its fix,
publication-root selection and handle matching pass five targeted checks in
23.10 seconds, followed by the complete final focused invocation above. No test
failure is hidden, no individual test is excluded, and overlapping counts are
not summed. The nine unrelated baseline expectations are not rewritten.

Reproduction uses the existing agent environment and isolated evidence root:

```bash
S=/home/likeyi/agent-usi4-resources-azazd0kq
PY=/home/likeyi/anaconda3/envs/agent/bin/python
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/final_focused.py"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/regression.py"
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/closeout-uploaded/real_workflow.py" --reopen
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/public-genome/accept.py" --reopen
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/historical_reads.py"
PYTHONDONTWRITEBYTECODE=1 "$PY" -B "$S/final_audit.py"
git diff --check
```

The real execution and public-genome scripts without `--reopen` are one-shot
acceptance drivers: run copies from **new empty evidence directories** for a
fresh execution. Do not rerun their creation branches over accepted workspaces.
The scripts record the existing scientific runtime configuration; no installation
or biological download is needed. Public-genome creation acquires only the two
bounded publisher metadata pages.

Relative to the preserved original USI-4 working tree, this follow-up changes:

| Files | Purpose |
| --- | --- |
| `application/raw_handoff.py`, `tools/analysis/annotation_input.py` | Typed uploaded components, existing-owner resource qualification, exact task/resource binding and automatic private persistence. |
| `application/turn_decisions.py` | Safe interpreter instruction for explicit uploaded reference roles and meaningful declarations. |
| `application/local_resources.py`, `application/shared_resources.py` (new), `application/reference_supplementation.py`, `application/shared_genome.py` | Existing catalog qualification variants, scope-aware discovery, selected-resource integrity and pinned permission review for new promotions. |
| `application/turns.py`, `application/resource_continuation.py`, `application/uploads.py` | Discover qualified resources through ordinary InputSets, check selected registrations, and retain a valid ordinary upload list after promotion. |
| `tests/annotation_integration/test_uploaded_resources.py` and `test_shared_resources.py` (new), existing handoff/reference-supplementation tests | Real interface software contracts, private/sharing boundaries, permission/integrity negatives, reuse and persistence. |
| This document | Audit, acceptance, exact scopes and remaining limits. |

Application/tool paths in this table are relative to `src/agent/`. All other
original USI-4 changes listed in Section I remain intact. Scientific backend and
authority implementations, EpiZoo, Registry and Planner/compiler remain unchanged.

Final preservation checks pass for **26,810 protected file entries**, including
**1,274 eval files**, with no changes or additions in the inventoried roots.
This is the same bounded size/mtime/mode inventory used at follow-up start,
not a fresh complete byte audit of every protected dataset. Consumed real
source/reference identities and the reopened acceptance workspaces have the
separate exact hash checks described above. `S/preservation-final.json` records
the final audit. The production operator configuration remains SHA-256
`f59312491c2719ae93bc3366724335550a713e5c76cf6bd84ddbc01a0558477e`.
HEAD, local main and origin/main remain
`e2c477d4c5848ddce7a818513ca579f6ee5732ca`. The index is empty and
`git diff --check` passes. No staging, commit or push occurs.

**USI-4 RESOURCE CLOSEOUT COMPLETE** within the qualified direct-peak scope.
Ordinary users can upload compatible gene/signature resources, answer meaningful
scientific questions, and execute the existing MAESTRO owner without manually
creating an InputSet or reviewed specification. References persist privately;
reviewed shared resources reuse the existing deployment catalog and compatibility
path. A real genome sharing witness and independent-Session annotation witnesses
exercise the common sharing boundary. No parallel resource framework, scientific
algorithm, workflow engine or production default is introduced.

Concrete limits remain: the annotation method is human hg38 with declared raw
Cell Ranger ATAC cut-site counts; arbitrary GTF conversion, new assemblies,
automatic clustering and threshold tuning are unsupported. A qualified chromosome
scope is still needed because gene/marker files alone cannot establish chromosome
lengths or primary-nuclear classification. Source processing history and resource
applicability are declared and checked for compatibility, not independently
reconstructed biological facts. Sharing requires reviewed permission and
qualification; real MAESTRO/CIBERSORT references remain private here. Access
isolation uses the existing Session/trusted-deployment boundary. Live-provider
quality and comprehensive native-browser acceptance remain unqualified and owned
by UA3.6.5; successful language decisions here are scripted. The nine baseline
test failures remain explicit debt. **USI-5 is ready to proceed after review** of
these unstaged changes within those limits; no USI-5 implementation has begun.

Final `git status --short`:

```text
 M src/agent/application/dialogue_evidence.py
 M src/agent/application/dialogue_execution.py
 M src/agent/application/interactive.py
 M src/agent/application/local_resources.py
 M src/agent/application/raw_handoff.py
 M src/agent/application/reference_supplementation.py
 M src/agent/application/resource_continuation.py
 M src/agent/application/responses.py
 M src/agent/application/session_state.py
 M src/agent/application/shared_genome.py
 M src/agent/application/turn_decisions.py
 M src/agent/application/turns.py
 M src/agent/application/uploads.py
 M src/agent/web/static/app.js
 M tests/application/test_dialogue_detail.py
 M tests/web/test_reference_supplementation.py
?? docs/usi-4-maestro-conversational-handoff.md
?? evals/
?? src/agent/application/shared_resources.py
?? src/agent/tools/analysis/annotation_input.py
?? tests/annotation_integration/test_conversational_handoff.py
?? tests/annotation_integration/test_shared_resources.py
?? tests/annotation_integration/test_uploaded_resources.py
```

`evals/` is pre-existing and unchanged. All other entries are original USI-4
work preserved from the starting tree or the bounded closeout changes recorded
here. No accepted original resource, Session, scientific output or evidence root
is replaced.
