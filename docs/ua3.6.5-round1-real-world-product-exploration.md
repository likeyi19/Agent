# UA3.6.5-R1 — Authentic scientific inputs and product exploration

Date: 2026-10-19. Status: **PARTIAL product acceptance**. The bounded exploration
and two confirmed continuation fixes are documented here; this is not a claim
that ordinary external peak matrices can complete clustering or annotation, or
that UA3.6.5 as a whole is complete. No P0 was observed. The new external data
completed inspection and evidence-grounded conversation. Successful downstream
analysis used a separately labelled historical control.

Evidence root **R** is `/home/likeyi/agent-ua365-r1-ntdyu1a4`. Input acquisition,
native Firefox screenshots/timeline, persisted Sessions/Runs, actual provider
diagnostics, native downloads and test logs remain there, outside Git. No
staging, commit or push was performed. Existing production configuration was
not changed.

## A. Repository baseline

Actual HEAD, local `main`, `origin/main`, and the safely queried remote `main`
all resolved to `961c41516be0cc3838fa85dc369f5e72fcf4f545`. The branch was
`main`. The tracked worktree and index were initially clean; the only initial
status entry was `?? evals/`. The preceding `1e3cded...` was not treated as HEAD.
The committed dual-pane UI and the USI-7 closeout were inspected before work.
USI-4/5/6, native-peak UA3.6.4c and relevant dialogue/authority contracts informed
the exploration; their historical successes are not counted as new acceptance.

`R/baseline.json` records 712 tracked files and 107,531 protected file entries
from the inherited historical scope, USI-7 evidence, operator configuration,
existing web state and `evals/`. The post-work audit checks size, mtime and mode
for all entries and SHA-256 for 107,453 small files. All remain unchanged;
all 1,274 protected `evals/` files remain present, without additions. This is a
bounded preservation audit, not a full rehash of every large historical file.
The two consumed input files also receive exact SHA-256 checks.

The existing `agent` Conda environment and RTX 4090 were used. About 15 TB disk
space was available before acquisition. The isolated copy `R/operator.json`
changes only workspace/upload destinations. Browser access used
`127.0.0.1:8865`; production port 8000 and the normal operator file were untouched.
Model profile `openrouter-gpt-5-6-sol` retains the operator's actual routing ID
`openai/gpt-5.6-sol:floor`, timeout, retry policy, 4,096 output-token setting and
input/output price caps 2/10. No model-specific scientific code was introduced.

## B. Authentic input inventory

### Newly acquired external input

The unmodified author-distributed
[Chen-2019-ATAC.h5ad](https://download.gao-lab.org/GLUE/tutorial/Chen-2019-ATAC.h5ad)
is the mouse cortex SNARE-seq accessibility dataset used by the
[GLUE preprocessing tutorial](https://scglue.readthedocs.io/en/latest/preprocessing.html).
It was not prepared for Agent, converted, supplemented with Agent metadata,
prequalified through a developer path, or registered as a shared reference.

| Property | Observed identity or limit |
| --- | --- |
| Publisher | Gao lab GLUE tutorial distribution of Chen et al. 2019 |
| Download date | 2026-10-19, Asia/Shanghai |
| Server release indicator | HTTP Last-Modified: 15 August 2021 03:41:53 GMT |
| Exact bytes | 60,786,319 |
| SHA-256 | `cc0ac62fd36a0be400e7d3bf2f7e32d5d03a191c332cd829e590e437048019f3` |
| Published checksum | None established; locally calculated digest is not a publisher checksum |
| Matrix | 9,190 cells × 241,757 features; float32 CSR; 22,575,391 nonzeros |
| Density | 0.010161099678490371, approximately 1.0161% |
| Metadata | Unique cell/feature axes; `obs.domain`, `obs.cell_type`; no `var` columns, `obsm` or `uns` content |
| Species | Mus musculus, from publisher context and explicit user declaration |
| Assembly | Not independently established; not inferred from feature names |
| Processing semantics | Publisher's scATAC peak accessibility representation; not qualified as canonical cCRE or Cell Ranger cut-site counts |
| Barcode/coordinate interpretation | Preserve supplied identifiers/order and feature strings; no correction, aliasing or coordinate conversion |
| Storage/sharing | `R/inputs/Chen-2019-ATAC.h5ad`, private acceptance upload; no dataset-specific redistribution license established |

Acquisition streamed with timeouts, a 250 MiB file bound and integrity recording.
Only this approximately 58 MiB file was downloaded. Source and uploaded bytes
were retained. Sparse inspection and the subsequent diagnostic never densified
the complete matrix. `R/input-acquisition.json` and `input-observations.json`
contain the acquisition and bounded observations.

Native Firefox file selection uploaded it through the ordinary UI. Inspection
was outcome **A: correct direct acceptance**. Initial follow-on input loss was
**D: incorrect application blockage**, shared with the eligible control and
fixed below. After source reuse was corrected, EpiZoo rejected the incompatible
feature space: **C: correct capability rejection**, with inadequate explanation.
The annotation attempt did not reach a usable targeted clarification or result;
it is not counted as successful acceptance.

### Historical control, kept separate

The real 32-cell human PBMC canonical matrix from USI-6 was uploaded again
through the same ordinary browser path. It is an existing developer-sized
control, not a newly representative external dataset. Its originally recorded
temporary location no longer existed; the harness was corrected to use the
preserved completed-upload copy, with no scientific content change:

`/home/likeyi/agent-usi6-07lldlo1/final/workspace/uploads/completed/94069c98c0afe1561f02d01f32177995796261965db85bc1b5131332fc6c5692.h5ad`

It has 32 × 1,355,445 int64 CSR entries, 336,129 nonzeros, 80,541,640 bytes and
SHA-256 `0c601bf017053f82fa34273156f647148c3ccddecf6646fcd9fcb4cc4aabe264`.
New inference/clustering on these bytes is new execution on historical input;
it does not extend the biological acceptance of the external Chen matrix.

## C. User journey evidence

All turns below were submitted through the actual static Web application in
native headless Firefox, with the real OpenRouter profile selected. They were
not HTTP-only substitutes or scripted provider responses. Browser automation
sets control values and triggers ordinary DOM events; upload, submission and
download use the actual browser controls. Scripted models appear only in the
separate software tests in section F.

### Journey A: ordinary external data

Session `a2fa9fff-bc30-4974-9eff-bc0bffcc5200`:

| Turn/evidence prefix | Actual request | Outcome |
| --- | --- | --- |
| `a1` | “Help me understand this dataset.” | Real interpreter/planner selected inspection. `inspect_scATAC` ran once, accepted 9,190 cells and 241,757 features; the left workspace showed the accepted result/report. |
| `a2` | “How many cells and features are there, and what does the sparsity tell me?” | Accepted-evidence answer gave the dimensions and sparsity. No new scientific Run. Refresh preserved the dialogue. |
| `a3` | “Cluster these cells and show me a UMAP.” | Before correction, prior upload was not offered, while inspection supplied no qualified execution source. Failed before scientific production; the UI merely said unsupported. |
| `a4` | “Cluster these mouse cells and show me a UMAP.” | After correction, the actual model explicitly selected the original uploaded source and the existing mouse resource. Native EpiZoo validation failed; downstream neighbors/Leiden/UMAP were skipped. No new accepted revision. |

The original inspection remains the sole accepted revision,
`c3e58f048fd4ff2c413a7b8e4385821dd80f43a84771f01a7af93ae5b7e10be8`.
The unchanged native validator was separately called on the original sparse
file and reproduced: “mouse input must have 1,341,077 raw cCRE features; got
241,757.” This diagnostic does not run inference or convert the peak matrix.
Its log is `R/native-input-diagnostic.json`. The failed browser Run itself stores
a sanitized `TOOL_EXCEPTION`; the detailed cause is established by this separate
diagnostic, not inferred from the generic UI message alone.

### Journey B: supported multi-turn analysis and comparison

Session `4f7a3276-edb1-47a5-b382-6cf31da44b0a`:

| Turn/evidence prefix | Actual request | Outcome |
| --- | --- | --- |
| `b1` | “This is a human PBMC scATAC-seq dataset. Help me understand this dataset.” | Ordinary upload and accepted inspection. |
| `b2` | “Cluster these human cells and show me a UMAP.” | Same original input-loss failure as A; this scientifically eligible control established a genuine application P1. |
| `b2-replay` | Exact same request | After input-offer correction: real EpiZoo inference → neighbors → Leiden → UMAP, each once, all accepted. |
| `b3` | “Use resolution 1.0 instead.” | Failed before science because selected cluster/UMAP display outputs did not offer their recorded qualified graph dependency. |
| `b3-replay` | Exact same request | After lineage-context correction: only Leiden ran, explicitly reusing the exact accepted graph. Four clusters. |
| `b4` | “Use resolution 2.0 instead.” | Only Leiden ran against that same graph. Five clusters. |
| `b5` | “Compare this result with the previous one.” | Real accepted-evidence answer compared resolution 2.0/1.0 and five/four clusters for the same 32-cell graph, with no new scientific Run. |

The initial scientific chain used owner defaults: 15 neighbors, resolution 1.0,
seed 0 and UMAP `min_dist=0.5`. Thus the first explicit 1.0 request did **not**
increase resolution; the later 2.0 request supplied the genuine change. Timings
were 52.69 s embedding, 10.08 s neighbors, 0.10 s initial Leiden and 0.42 s UMAP.
GPU memory observation was 13,153 MiB total device use, not an isolated peak
allocation measurement. There was no post-training or full-data densification.

The four accepted revisions, in order, are:

1. Inspection: `fa7439a3fed4ae89cb2c4549e612fcfc7bf79a500dfc3f42e80bab86db08b645`.
2. Initial cluster/UMAP: `047aa84bf134a95e36a0563c60c20063696c7c94fdf93f82bb6c62d1e8868a6d`.
3. Explicit resolution 1.0: `9ee6af970f91880ad3ceefbc73bb39d906c257d808f2ff80ba602d8243300d17`.
4. Explicit resolution 2.0: `3f1af109df4dd9d59f12ef207386dce091c0bb9e4f35bc63e9be21e067575113`.

Recorded `PriorOutputRef` bindings establish exact graph reuse, independently of
text labels. No embedding or neighbor production step occurs in the two later
Runs. This is a statement about recorded production, not a blanket measurement
of every owner verification/read. Comparison explicitly avoids cluster matching,
cell-type equivalence, biological superiority or statistical significance.

### Journey C: an appropriate annotation reference

In the external-data Session, the actual request was:

> These are mouse cortex scATAC-seq peaks from the Chen 2019 SNARE-seq dataset.
> Can you use an appropriate existing reference to annotate the cell types?

The initial `c1` failed through the missing-source problem. `c1-replay` and
`c1-diagnostic` repeated the original request after correction. The diagnostic
captured the real provider's typed `request_roles`: annotation target, exact
`source.0`, mouse, cortex context and unspecified assembly. Agent's scientific
input observer then raised `Unsupported H5AD group encoding`. The old author
file is readable by AnnData and by the accepted generic inspection, but lacks
group encoding metadata required by the current peak-input reader.

This is an Agent scientific input-layout/admission issue, not an OpenRouter
network failure or evidence that the model selected the wrong file. Separately,
the current direct peak-count annotation contract is qualified for human/hg38
Cell Ranger ATAC cut-site counts; this mouse SNARE-seq matrix does not establish
that measurement/species contract. Removing the layout check alone would not
scientifically qualify this task. No assembly, cut-site semantics, grouping or
reference compatibility was fabricated. No human marker reference was silently
substituted. Annotation and public-reference acquisition are **not passed**.

`R/provider-audit.jsonl` contains five later diagnostic-stage real provider
completions, including this decision and the external clustering replay. It is
not a complete call count for earlier browser turns. The logging wrapper calls
the original provider unchanged; the exception observer records admission
exceptions without replacing responses. No provider/network failure or current
OpenRouter 404 was observed in the executed browser journeys.

### Results, downloads and recovery

The browser displayed the accepted control summary, technical report and
original UMAP PNG. Selecting revision 2 for preview preserved active revision 4.
Native download clicks saved all four offered H5AD artifacts; each native file
matched the browser HTTP response size, Content-Disposition and SHA-256:

| File | Bytes | Delivered SHA-256 |
| --- | ---: | --- |
| `clusters.h5ad` | 90,784 | `784dc2014dafbcf2adb0872cb335d8d27b205e506e55675ab8ecc0881c697d27` |
| `analysis.h5ad` | 91,048 | `2afe519d53ea1c28639f93ad9c25f917c9ea33726d9a302fe2e948e1beb9af4d` |
| `embeddings.h5ad` | 84,504 | `9af327d5fbdcfa980dd08716991a8b8ecb66a3a6090db9337434b728d502edf8` |
| `neighbors.h5ad` | 121,712 | `06e57ac2713b9ce3cb7d7236b6b15960992b90715e2ceb46fc48e6fd74763e4c` |

These are the existing privacy-reviewed delivery projections, not claims of
byte equality to the underlying scientific source H5AD. The original report
and delivered report identities likewise remain distinct. Independent readback
of all four downloaded H5AD files matches the pinned source digest and preserves
every offered scientific array, graph, label and ordered cell ID exactly;
`R/download-scientific-equality.json` records these comparisons.

Refresh and a fresh server process restored the exact 14 conversation texts,
four revisions, generation 4, active revision, historical preview and artifact
identities for B. All four delivery hashes remained equal. All 90 files in the
isolated workspace were unchanged; the provider audit did not grow. Recovery
therefore created no Runs, invoked no logged provider and changed no persisted
scientific state. Evidence: `browser-results.log`, `native-downloads.json`,
`recovery-result.json`, screenshots and the browser timeline.

Across the two Sessions there are 14 user turns, five accepted revisions and
ten Runs. Five Runs succeeded, four failed before scientific production, and
one attempted external embedding and failed. Successful new production totals
are two inspections, one embedding, one neighbor graph, three Leiden executions
and one UMAP. These totals exclude scripted tests and the read-only validator
diagnostic.

## D. Scientific resource readiness

The existing operator catalog contains qualified original human and mouse
EpiZoo defaults. The ordinary control replay selected the human default without
manual InputSet construction or checkpoint reupload. The external mouse replay
selected the mouse default, but that does not make its peak space compatible.
The shared pretrained checkpoint is pinned at
`6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`.
Human auxiliary pins are
`8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b`
and `994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875`;
mouse pins are
`c4c63aaae8a6f841812189bf59930014162c3f61c7a463d744be71595359171d`
and `96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393`.
Qualification remains the existing owner-reviewed catalog record; R1 creates
no replacement model qualification.

The local inventory also contains the preserved human hg38 GENCODE 49 and mouse
mm10 GENCODE M25 QC/TSS resources and catalog under
`/home/likeyi/program/agent-resources`, plus the historically qualified
annotation/reference resources described in USI-4. Physical presence is not
automatic availability in every deployment: the normal operator configuration
does not activate the full historical isolated annotation setup. No reference
catalog or production default was silently changed.

For C, the immediate blocker is not a missing downloadable mouse marker file:
the native input layout and independently limited annotation measurement/species
contract both matter. Buying acceptance by downloading an incompatible human
reference would be incorrect. Thus no new scientific reference was downloaded,
qualified or globally registered. The only acquired file is the external private
test input in B. This round verifies real qualified EpiZoo resource reuse, and
examines annotation readiness, but does not qualify a new public reference or
demonstrate real missing-reference upload continuation. A compatible human
peak-count journey should next distinguish deployment activation from genuine
resource absence using the existing owners/catalog.

## E. Friction and architectural review

| Priority / finding | Evidence and responsibility | Change or disposition |
| --- | --- | --- |
| P1: inspection followed by analysis loses the uploaded file | A3 and eligible B2 both failed. Session uploads existed, but only the new turn's selected files were offered. An inspection scalar is correctly not executable matrix authority. | Offer all bounded previously submitted Session attachments as explicit candidate handles; let the LLM select roles. No automatic file binding or authority upgrade. Exact B2 replay succeeds. |
| P1: continuation depends on output-display selection | B3 failed although the accepted chain had an exact qualified neighbor graph. The output selector had retained clusters/UMAP, not the graph. | Resolve exact recorded scientific argument dependencies in the existing prior-output owner; offer qualified sources for explicit planning. No hidden step-order inference or automatic workflow completion. Exact B3 replay and B4 succeed with Leiden only. |
| P2: invalid admission falls back to obsolete QC instructions | C replay exposed generic advice about thresholds/versions for an annotation input-layout failure. | Replace the misleading generic fallback with neutral interpretation/admission wording. Root-cause-specific scientific guidance remains incomplete. |
| P2: generic runtime failure hides feature-space incompatibility | A4 correctly fails, but the UI's generic error is not a useful explanation of required cCRE space. | Recorded; no speculative scientific parser or unsafe exception-text exposure added. |
| P2 / unqualified capability: annotation input encoding and mouse SNARE semantics | Actual owner exception reproduced; independent supported-species/measurement limit remains. | No fixture rewrite or relaxed scientific contract. Valid legacy H5AD support requires owner-level work on an otherwise eligible input, with independent verification. |
| P3: source label in the inspection answer is awkward | A2 cites a cooutput labelled “Cell metadata columns” for facts from the same accepted inspection source. | Exact attribution remains; no cosmetic evidence-model refactor. |
| Bounded offer inconsistency | Existing attachments permit 16 candidates, while typed `InputRole.source` accepted only positions 0–7. | Align source handles to 0–15, retain role-count limits and reject position 16. Test selection of position 15. |

The remaining restrictions on source integrity, private Session scope, ordered
scientific identity, qualified resource pins and native feature/measurement
compatibility are necessary. There is no evidence here justifying removal of
those checks. No tool-specific natural-language matcher, independent planner,
new upload system, resource framework or scientific algorithm was added.

## F. Implementation and verification

### Changes and boundaries

`interactive.py` captures up to 16 exact previously submitted attachment
registrations for a no-input turn. The immutable offer records registration
digests and checks Session ownership. Explicit current inputs take precedence;
pending prerequisites keep their existing continuation path. More than 16
candidates produces an overflow offer, never a chosen subset. Completed retries
reuse their original captured offer, unaffected by later uploads.

`local_resources.py`, `turns.py`, `dialogue_execution.py` and `raw_handoff.py`
reuse existing grouped role binding for these offers. Candidate availability is
distinct from executable input: only an explicit typed role/declaration activates
the existing binding path. Old declarations are not implicitly inherited.
Metadata offers do not read source bytes; selected execution still verifies the
source. Answers do not need the original upload to remain present. Prompts
expose safe existing descriptors, never paths, authority payloads or callables.

`prior_outputs.py` follows only accepted `StepOutputRef`/`PriorOutputRef` argument
edges, validates exact anchors/resolved identities, stops at reusable qualified
publications, and rejects cycles or changed lineage. It does not search paths,
follow control-only dependencies, discover sibling results, reconstruct science
or grant authority to clusters/inspection. Results with no reference arguments
have no dependencies to resolve. This last boundary was corrected after two
historical answer tests exposed unnecessary argument revalidation.

`sessions.py` offers these exact dependencies in captured planning context.
Display selection no longer gates availability of a known scientific source.
Derived sources have no implicit retention locator: the revision inventory is
unchanged, and actual use still requires the existing explicit plan/compiler and
integrity checks. The output-selection schema includes `retain` only when there
are real retainable locators. Metadata reads remain separate from execution
integrity verification. Native EpiZoo/MAESTRO source, Registry inventory,
scientific algorithms and scientific authority schemas are unchanged.

### Verification record

Focused software tests use tiny inputs and scripted providers; they do not
count as real LLM or biological acceptance. New cases cover exact prior upload
selection, multiple sources, Session isolation, explicit-input precedence,
immutable retries, source mutation, missing-source answers, overflow and all 16
candidate positions. Authority tests cover retained/omitted qualified sources,
cluster/UMAP lineage, changed resolved identities, metadata versus execution
integrity, and preservation of captured revision identity.

- Initial corrected focus: 29 passed in 18.31 s.
- Lineage/upload focus after the metadata-integrity correction: 45 passed,
  3 warnings in 98.42 s.
- Final candidate/role focus: 70 passed in 6.68 s.
- Historical interactive boundary recheck after excluding reference-free
  argument revalidation: 30 passed in 11.51 s.
- Final relevant regression: **790 passed, 3 dependency warnings in 506.62 s**,
  exit 0. All **553 frozen source/test/resource files remain unchanged** during
  execution; no `RUN_*` gate is enabled. The wall-clock driver records 510.56 s.
  Exact results are in `R/verified-final-regression.log`, its JUnit XML and
  `verified-final-regression-result.json`.

An intermediate run of that relevant scope completed with 788 passed and the
two historical answer/guidance failures in 505.00 s. It had loaded the pre-fix
lineage code. Both failures exposed the same unnecessary reference-free argument
validation, were corrected at the dependency resolver, and passed the 30-test
boundary recheck. The final rerun uses a frozen corrected source snapshot;
the intermediate result is retained as `R/final-regression.log`, not represented
as a clean final run.

The broader exploratory application/web run was interrupted after 482 passed,
7 skipped and 3 failures in 544.55 s. Two failures were reproduced unchanged in
an isolated `git archive HEAD` checkout: the resolution/declaration test
`test_invalid_explicit_declaration_never_reaches_scientific_production[0.7-changes8]`
and `test_selection_transport_rejects_unbounded_ambiguous_or_path_inputs[changes1]`
(expected 422, baseline returns 404). The third was a 20-second rich-API
finalization timeout under the concurrent run; it passed in the baseline
standalone reproduction. That baseline reproduction was 2 failed, 1 passed in
68.67 s. The first test supplies the actual unique literal `0.7` with incorrect
legacy offsets; the existing `_ground_declaration` intentionally normalizes that
unique literal before admission. It does not substitute a different numerical
value. The second expects a singleton selection to fail at transport rather
than ordinary unavailable-resource resolution. These are baseline expectation
conflicts, not newly demonstrated scientific corruption; their tests were not
rewritten in this round. They are not silently excluded new failures or claimed
passes.

The chosen final scope starts with all 24 modules from the accepted USI-7
584-test application/delivery/web scope and adds 11 changed or adjacent
role/continuation/authority/annotation/EpiZoo-handoff modules. Exact arguments
are saved in `R/final-regression-command.json`; all `RUN_*` gates are disabled,
no in-module exclusions are used, caches and test outputs are outside the repo.
The corrected final command/log use the `verified-final-regression` prefix.
This is relevant software regression, not a full new biological qualification
or a rerun of every expensive scientific pipeline.

Native browser replays, new real production and fresh-server recovery are
separate evidence described in C. No scientific owner implementation changed,
so no new training/whole-genome scientific regression was introduced. Baseline
protection results are in `R/preservation.json`; input source bytes, operator
settings, prior web Sessions, historical evidence and `evals/` are preserved.

## G. Remaining limitations and next steps

R1 is **PARTIAL** for representative product readiness. It establishes an
unprepared external upload, meaningful inspection and follow-up, actual
OpenRouter/browser use, two reproduced and repaired generic continuation
barriers, qualified resource reuse, real control analysis, comparison, downloads
and recovery. It does not establish external peak-to-EpiZoo conversion,
mouse SNARE-seq annotation, general legacy H5AD qualification or a newly
acquired public-reference workflow. No invalid scientific output was activated.

The most useful next exploration is a modest, independently sourced real
human/hg38 peak-count dataset within the existing direct-annotation measurement
contract, uploaded through the same UI. First establish its published assembly
and count semantics; then exercise natural grouping/resource clarification,
existing qualified genome reuse, and gene/marker resource continuation. Repair
legacy H5AD admission at the scientific owner if that otherwise eligible file
exposes the same layout restriction. Improve typed user-facing capability/error
reasons so researchers can distinguish missing information from unsupported
science without learning internal contracts. Do not transform the current
Chen file just to make this round look successful.

The implementation remains unstaged and reviewable. Applying it to the normal
service requires the reviewed code and a normal server restart; the isolated
acceptance workspace and private uploaded files are not production defaults.
Final worktree changes comprise nine production files, three existing test
files, one new upload-reuse test file and this report. The original untracked
`evals/` remains untouched. `git diff --check` passes; HEAD and the index remain
unchanged. The isolated Firefox and acceptance server were shut down after
recovery checks; the production service was not restarted.

## H. Bounded pre-commit review and correction

This addendum supersedes the software-review status and file count above; the
original R1 browser/scientific evidence and PARTIAL product acceptance remain
unchanged. Review evidence is **Q** =
`/home/likeyi/agent-ua365-r1-review-CHBXOWYY`. HEAD and the separately queried
remote main remain `961c41516be0cc3838fa85dc369f5e72fcf4f545`. Q captures the
initial R1 diff, 714 file hashes and all 1,274 `evals/` hashes. No new live-provider,
browser, large biological, PCA, model-training or production deployment run was
performed in this review.

### Attachment reuse: one confirmed omission repaired

The original no-current-input condition incorrectly hid historical files when
a new companion was attached. Two simulated annotation integration cases failed
on the original R1 implementation: old H5AD plus new gene/signature files, and
old gene/signature files plus a new H5AD. This was a real P1 continuation gap,
not a scientific restriction or an LLM decision failure.

The correction stays in existing submission capture and role binding:

- Current candidate files retain their exact positions; same-Session historical
  candidates are appended once by exact registration identity. The prompt
  distinguishes current-turn candidates from historical candidates.
- The ordinary browser's single-H5AD wire still has its explicit primary
  `resource_id`. Its existing `candidate_input` view can now carry companion
  candidates; source 0 remains the selected H5AD. Typed annotation/EpiZoo
  declarations cannot replace it with a historical H5AD.
- Explicit execution inputs and configured resource choices retain their
  existing binding. Candidate additions neither transfer old scientific
  declarations nor replace a current role/resource decision. Selected
  deployment-visible resources retain their existing access rules; automatically
  offered historical attachments must belong to the same Session.
- Current attachment attribution remains the actual current message. Historical
  candidates are not relabelled as new uploads. Their roles, scientific
  eligibility and resource qualification still require the existing owners.
- The combined automatic offer is bounded at 16. On historical overflow, current
  explicit files remain available, the overflow is exposed, and no historical
  subset is silently selected. An over-bound current group is rejected.
- Pending clarification retains its original candidate handles and uses the
  existing exact-prefix/additive continuation. Completed retries retain their
  original immutable offers even after later submissions or missing source
  bytes. Source integrity is still enforced when the source is consumed.

No new permanent module, catalog, upload mechanism, semantic matcher or
scientific validator was introduced. The existing `available_input_view` remains
a pure view into the existing grouped binder. Its additional check ensures that
an expanded offer preserves the captured current prefix; it does not create a
second binding/qualification implementation. The old single-source candidate
guard was narrowed to preserve the selected primary while permitting companions.

### Scientific dependency reuse: no further production correction

The prior-output owner follows only recorded accepted argument references,
checks source anchors and resolved identities, and stops at qualified
publications. Presentation locators choose the requested result roots; scientific
dependency discovery does not infer sources from display labels, selected
cooutputs, sibling steps or control-only dependencies. Inspection output is never
promoted to executable matrix authority. Derived dependencies have no implicit
retention locator and do not mutate a Revision.

Metadata reads and execution integrity remain distinct. The helper reuses the
existing resolver and authority checks rather than rebuilding scientific results
or qualifying them in the application layer. Repeated metadata anchor checks
inside these shared owners are not a second scientific qualification pass; no
cache or additional authority abstraction is justified by this review. Existing
tests cover captured-base navigation, omitted/retained graphs, control edges,
changed resolved identities, and corrupt graph rejection. No further defect was
found within this bounded review.

### Focused verification and remaining risks

The two original mixed-input cases failed in 8.31 s, then passed after the
correction. An intermediate upload/resource scope passed 25 tests in 108.09 s.
The expanded new-case scope passed **14 tests in 56.98 s**, including the actual
single-H5AD browser request shape, explicit-primary substitution rejection,
current-plus-historical execution, missing-source completed retries, overflow,
and clarification that adds a third file while retaining original handles.
These use scripted models and small/simulated science; the annotation fixture
runs its normal production/verifier boundaries, not a live MAESTRO biological
qualification. The existing resource tests also assert one qualification rather
than repeated qualification.

The final bounded adjacent check and preservation results are recorded at
review closeout below. This is not a rerun of the original 790-test scope; that
result remains evidence for the pre-review R1 snapshot. The remaining product
limits in G are unchanged. Automatic file offers beyond 16 require explicit
selection. New combined-file semantics have deterministic software coverage,
not a new live-provider acceptance claim.

## I. Read-only PCA capability audit and minimum R2A scope

### What exists

The current default Registry has **23 tools** and no PCA, TruncatedSVD or LSI
producer. Searches covered current tracked code, executable history under
`src/tests/scripts`, and Python/R/notebook files in `evals/` without modifying
them. No project-owned historical PCA embedding implementation was found in
that scope. This does not contradict the team's prior use in external notebooks
or another repository; that exact method/preprocessing recipe is not established
by this checkout.

| Layer / location | Actual capability and restriction |
| --- | --- |
| `tools/analysis/epizoo_embedding.py`; `tools/models/epizoo.py` | EpiZoo produces `.npy`, exact ordered cell IDs and a compact H5AD. Native vocabulary/resource validation is mandatory and unchanged. There is no PCA path. |
| `tools/models/epizoo.py::embed_cells`; `tools/models/epizoo_adaptation/preprocessing.py::preprocess` | Existing sparse TF-IDF and cell-sentence preparation delegate to `epizoo.data.processing`; these are model-specific preprocessing, not a decomposition or an LSI producer. Do not reinterpret their feature frequencies/token ordering as a general PCA profile. |
| `orchestration/registry.py` | `epizoo_embedding_bundle.v1` is the sole accepted upstream embedding type for `build_cell_neighbors`. Its request fields can point to files, but no PCA tool/result contract or producer channel exists. |
| `tools/analysis/embedding_analysis.py::_load_embeddings` | Memory-mapped `.npy`, finite float32, exactly 512 columns. The function does not independently prove which algorithm produced an arbitrary 512-column array. Passing such an array is not supported PCA integration and would label it `X_epizoo`. |
| `tools/analysis/embedding_analysis.py::_validate_neighbors_artifact`, `cluster_cells`, `compute_cell_umap` | Shared Scanpy graph/Leiden/UMAP implementations are reusable, but validation requires compact H5AD, `X_epizoo`, 512 columns and the existing provenance. UMAP already supports graph-only input without requiring clustering. |
| `tools/analysis/neighbors_publication.py` | Qualified graph profile and exact-source verifier bind `X_epizoo`, dimension 512, dtype float32 and representation parameters. Merely relaxing the initial dimension check is insufficient. |
| `orchestration/verifier.py`; `tools/analysis/neighbors_delivery.py`; `application/analysis_delivery.py` | Verification and safe downloads also require EpiZoo identity/dimensions. They must evolve coherently with the scientific contract, not bypass it. |
| `orchestration/planning_scope.py`, `semantic_compiler.py`, `executor.py::preflight` | Scope text explicitly describes EpiZoo embedding analysis; channels derive from Registry ports; preflight rejects unknown tools/invalid contracts. Ordinary Web dialogue cannot currently compile and execute a supported PCA chain. No live model was called to test wording during this audit. |
| Installed libraries | Scanpy 1.11.5, scikit-learn 1.9.0, SciPy 1.17.1, NumPy 2.4.6 and AnnData 0.12.19 are available. A callable library implementation is not a registered Agent capability. |

Paths in the table are relative to `src/agent/`. Q's `pca-inventory.json` records
the actual Registry names, selected semantic ports, authority hooks and installed
versions. EpiZoo embedding, clustering and UMAP ToolSpecs have no registered
scientific-authority hooks; the neighbor graph has the existing `cell_neighbors`
owner. A generic reusable PCA embedding publication is not already supplied by
these hooks.

The earlier native MAESTRO review identifies a real external
[`ATACRunSeurat` implementation](https://raw.githubusercontent.com/liulab-dfci/MAESTRO/74f10babf4125b98ef005df4186b14ab02e13bfd/R/ATACRunSeurat.R).
Its separate branches use TF-IDF/SVD for LSI, or normalization/scaling followed
by `RunPCA` for PCA. It also filters cells/features and runs its own neighbors,
clustering, UMAP and marker tests. It is neither a registered Agent embedding
tool nor an isolated replacement for the requested shared downstream path.
Adopting that whole workflow would add unnecessary orchestration and potentially
change the accepted cell population. No R dependency installation or execution
was performed.

### PCA semantics and memory

True PCA centers features before decomposition. Sparse input can use an
implicit centered operator with an explicitly selected ARPACK solver, rather
than materializing a centered dense matrix. The installed scikit-learn
`_fit_truncated` implementation uses `_implicit_column_offset` for sparse input
and `svds`; its installed Scanpy wrapper can select this route. Component count
must be strictly below the smaller matrix dimension. These observations agree
with the [official PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

`TruncatedSVD` does not center input and must not be labelled PCA.
TF-IDF followed by SVD is the distinct LSI path relevant here; it is not obtained
merely by renaming PCA. Scanpy's `zero_center=False` selects uncentered SVD;
`chunked=True` selects incremental PCA and can densify chunks. Therefore neither
flag is a silent memory-recovery substitution. See the
[TruncatedSVD contract](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
and [Scanpy PCA contract](https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html).
The online sklearn documentation is 1.9.1; the audited installed implementation
is 1.9.0. R2A should pin its actual solver/version rather than rely on `auto`.

For the external 9,190 × 241,757 matrix, a full dense copy alone needs about
8.277 GiB at float32 or 16.553 GiB at float64. A feature covariance matrix at
float64 needs about **435.46 GiB**. Fifty float64 loading vectors need about
92.22 MiB and the scores about 3.51 MiB, before solver/input/workspace overhead.
These are arithmetic estimates, not measured runtime acceptance. Avoid explicit
full centering/scaling and a feature-covariance solver; bound sparse copies,
component count, iteration workspace and output arrays. No PCA was run on the
external data in this audit.

### Minimum scientifically correct R2A

**Option 3: a genuinely missing owner-level PCA capability**, plus a bounded
generalization of existing embedding/downstream contracts. It is not merely
composition or an enhancement of an already registered PCA tool.

1. Add one scientific embedding owner/tool using the installed decomposition
   implementation. Specify the accepted input/count slot and preprocessing
   profile, centering, scaling/whitening, component count, solver, seed, dtype
   and resource bounds. Preserve exact cell/feature order and input identity;
   never silently filter cells, normalize already processed values again or
   infer count semantics from a filename. Genome/species resources should not
   become prerequisites for numerical PCA unless an actual scientific step
   requires them. The team's prior recipe should be reused if located and
   qualified, not presumed from the word PCA.
2. Extend the existing embedding bundle/compact-analysis boundary with explicit
   method, variable dimension, ordered axes, parameters and source provenance.
   Keep legacy EpiZoo artifacts and pins exact through an explicit version or
   small reader. Do not relabel PCA as `X_epizoo`, pad to 512 dimensions, change
   native EpiZoo validation, or build parallel neighbors/Leiden/UMAP engines.
3. Generalize the existing graph owner, verifier, qualified publication profile
   and safe delivery to consume the method-aware representation. Retain graph
   integrity and lineage for parameter continuation. Record enough PCA fit
   state/variance and verification evidence for reproducibility, using existing
   publication/authority mechanisms rather than a new trust system.
4. Register the tool, semantic producer/consumer channels and capability scope;
   extend the existing input-role and evidence/report projections where needed.
   Keep current Planner/compiler/preflight and Session mechanisms. Do not
   automatically generalize EpiZoo-specific label transfer: independently fitted
   PCA reference/query coordinates are not interchangeable.
5. Verify sparse versus small dense-reference numerical behavior, parameter and
   axis preservation, failure/recovery bounds, exact graph reuse and delivery;
   then perform the bounded real external Web journey in R2A. A generic request
   may select an available compatible method through typed intent and owner
   validation. An explicit EpiZoo request must never silently fall back to PCA.
   Reports must state the method actually executed.

No R2A capability or scientific-method change is implemented by this addendum.

## J. Pre-commit review closeout

The final adjacent suite passed **138 tests, 3 warnings in 283.26 s**, exit 0
(286.69 s including process overhead). It covers web attachment admission,
annotation and EpiZoo handoff, interactive admission/retries, captured scientific
dependencies and multi-input roles. All `RUN_*` gates were absent; this was a
bounded software check, not broad scientific regression. All **553 frozen
source/test/resource files** remained unchanged during the run. The warnings
concern deprecated `pkg_resources`/Louvain and an unavailable TBB threading layer;
no test failed. Exact commands, log, XML and hashes are in Q's
`focused-command.json`, `focused-final.log`, `focused-final.xml`,
`focused-result.json` and `source-before.json`.

The read-only preservation check found no change among the **107,531 protected
entries** using the original bounded stat/small-file-hash policy. All **90 R1
workspace files** and **1,274 eval files** match their recorded hashes; the native
EpiZoo implementation is unchanged. See Q's `preservation.json`. Historical R1
evidence was read without rewriting it. No provider, large scientific run,
resource acquisition or model change was part of this review.

**Recommendation: R1 is ready for commit and push after user authorization.**
Both P1 fixes respect existing owners; the confirmed mixed-current/historical
attachment defect is corrected with focused coverage. Product acceptance remains
**PARTIAL**, and R2A remains future work. This recommendation does not qualify
new live-provider behavior or external-data PCA/annotation. The commit scope is
the 15 R1 source/test/report files below, excluding protected `evals/`.

Final branch is `main`; HEAD, local and queried remote `main` remain
`961c41516be0cc3838fa85dc369f5e72fcf4f545`. `git diff --check` passes. The index
is empty; nothing was staged, committed or pushed. Exact `git status --short`:

```text
 M src/agent/application/dialogue_execution.py
 M src/agent/application/interactive.py
 M src/agent/application/local_resources.py
 M src/agent/application/raw_handoff.py
 M src/agent/application/responses.py
 M src/agent/application/sessions.py
 M src/agent/application/turn_decisions.py
 M src/agent/application/turns.py
 M src/agent/orchestration/prior_outputs.py
 M tests/annotation_integration/test_uploaded_resources.py
 M tests/application/test_multi_input_role_decisions.py
 M tests/application/test_pending_role_contract.py
 M tests/authority_dag/test_captured_planning_context.py
?? docs/ua3.6.5-round1-real-world-product-exploration.md
?? evals/
?? tests/web/test_session_upload_reuse.py
```
