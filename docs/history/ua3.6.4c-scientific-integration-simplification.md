# Work Unit C — MAESTRO integration simplification

2026-10-14 (Asia/Shanghai). **Bounded review with one clarification-text fix.**
The existing direct peak-count H5AD route remains the supported scientific path;
no annotation method or input-contract change is needed. All four UA3.6.4 product
scenarios remain **PARTIAL**. A complete live-model request-to-result chain is
not established by this unit.

## Baseline and preservation

HEAD, `main` and local `origin/main` are
`bc9963840394a5df802857d7ce09794d3e990379`. The index and tracked worktree start
clean. Untracked Work Unit B's
[report](ua3.6.4b-scientific-integration-simplification.md) and `evals/` are
preserved. No remote synchronization, stage, commit or push occurs.
Evidence is in `/tmp/agent-maestro-simplification-qe9kgoij` (**R**).
The initial snapshot hashes 676 tracked files, the B report and operator config;
it records names/sizes/nanosecond mtimes for 1,274 eval files. Eval checking is
metadata preservation, not a fresh content-hash audit.

The latest scientific baseline is
[direct peak-H5AD implementation and acceptance](ua3.6.4c-native-peak-h5ad-acceptance.md).
The [native review](ua3.6.4c-maestro-native-simplification-review.md) is its earlier
design record, not evidence that the implementation is still pending.
[M13.5](m13.5-agent-annotation-integration.md) and
[M13.4b](m13.4b-annotation-validation-semantics.md) describe the retained historical
binary/replay and annotation-versus-validation contracts.

## Ordinary researcher workflow and necessary inputs

The researcher selects a registered H5AD as the annotation source and an existing
reviewed resource InputSet, then supplies an exact grouping TSV or names an obs
group column. Existing typed file roles, parameter declarations and resource
questions feed the normal Interpreter → Planner → compiler/preflight →
`annotate_scATAC_cell_types` path. Agent binds source/resource hashes, constructs
the resolved specification and publishes scientific authority internally. There
is no canonical adoption Run or trusted historical matrix store prerequisite for
the direct route.

| Requirement | Current owner behavior and scientific reason |
| --- | --- |
| Counts | Explicit `X` or `layers/name`; canonical sparse CSR/CSC with finite nonnegative integral values ≤ `2^53-1`, exactly representable in the private float64 working copy. Selected counts are not binarized or normalized before RP. Other admitted source slots are preserved. |
| Measurement and species | New direct method qualifies declared human/hg38 `cellranger-atac-cut-site-counts.v1`. Integer values do not prove upstream counting history. Declared fragment-overlap, binary-accessibility or TF-IDF semantics do not silently enter this profile. Historical binary processing stays a distinct method. |
| Features and cells | Unique exact ordered identities; feature names are zero-based half-open `chrom:start-end`, within the qualified primary nuclear scope. No cCRE vocabulary, automatic aliasing, feature reordering, merging or silent scaffold removal. |
| Gene resource | Pinned MAESTRO transcript/exon refgenes, matching species/assembly and gene namespace; supplies Enhanced RP geometry and gene identities. |
| Marker resource/context | Pinned candidate–gene signatures with compatible namespace, explicit tissue/context and recorded normalization. Signature context must match the requested context; database membership is not ground truth. |
| Contig scope/runtime | Existing qualified hg38 QC-reference classification/catalog is reused to establish contig eligibility; annotation runs no barcode QC. Pinned MAESTRO source slice, Rscript and R packages supply execution. A new peak collection does not require rebuilding these resources. |
| Grouping | Explicit obs string/integer column or exact ordered `cell_id\tgroup` TSV, every cell once, at least two groups, recorded derivation provenance. Native marker/Presto comparisons consume these groups; exact propagation needs their correspondence to the matrix. No groups, correspondence or biological cell labels are guessed. |
| Storage | Existing bounded ordinary AnnData layout, ≤1 GiB source, ≤2 GiB decoded values, ≤100,000 cells, ≤2,000,000 features and ≤100,000,000 stored count entries. Unsupported/private-locator metadata fails explicitly. These are current engineering bounds, not universal biological requirements. |

`external_marker_annotation.py:weighted_rp` consumes counts, peaks, cells and gene
annotations; **RP itself does not require groups**. The registered operation is
the complete group-marker annotation, not a standalone gene-score tool, so its
group prerequisite is appropriate. Exposing RP alone would be a new capability,
not a reason to remove grouping from the current annotation operation.

`count_slot` and `group_column` use existing shared exact-literal admission and
are honored without silently substituting another column/layer. Decay, RP method,
normalization and marker thresholds are fixed in the accepted scientific profile;
arbitrary numerical retuning is not a currently exposed parameter interface.

Resources can be reused through the same reviewed specification for different
compatible peak collections: `annotation_contract.py:resolved_specification`
adds the selected grouping to reusable resource declarations, while
`external_marker_annotation.py:coordinates` checks scope rather than equality to
an old BED. There is **no general automatic annotation-resource selector** from
species/tissue alone. Existing InputSets offer exact compatible supplementation;
an already captured choice is not replaced. The isolated PBMC menu in historical
acceptance is not proof of an installed universal production default.

## Scientific ownership, output and compatibility

The direct adapter invokes the hash-pinned native `calculate_RP_score`/
`ExtractGeneInfo`/`RP_AddExonRemovePromoter` bodies with count-preserving sparse
input, Enhanced RP and decay 10,000. The unchanged R adapter applies sparse
LogNormalize and calls native `FindAllMarkersPresto` and Presto. It retains raw
p < .01, |logFC| > .25, rounded detection fraction > .1 and adjusted p < 1e-5.
There is no local RP or Wilcoxon reimplementation to remove.

The small normalization/signature-scoring adapter preserves the accepted policy:
signed summation and duplicate signature weighting, coverage evidence, unique
positive maximum assignment, unresolved nonpositive scores and ambiguous exact
positive ties. Replacing it with the full Seurat wrapper would introduce extra
processing and different assignment behavior; this unit makes no such change.

`scatac_annotation.verify_annotation` independently reloads pinned inputs and
replays the same backend once, then compares RP, markers, scores, axes, labels and
the conserved derivative. This is computational replay, not independent proof
of biological identity. Accepted authority/proofs avoid replay during supported
terminal reads and recovery; source/resource integrity checks remain necessary.
No new meaningful duplicate scientific execution was demonstrated.

The output preserves admitted source X/layers, axes, obs/var, raw, ancillary arrays
and metadata, and adds annotation columns and provenance. Evidence/report/rationale
show assignments, unresolved/tied cases and support. `validation_state=not_assessed`
and absence of accuracy/confidence calibration remain explicit. Optional CellMarker
or EpiAgent corroboration is not required and cannot fill nulls or change labels.
The safe downloadable annotated H5AD removes the owner's private source locator
and records export provenance; it is a derivative with its own digest, not a claim
of identical publication bytes or new scientific authority.

The historical real 10x PBMC 500 acceptance already established 484 cells,
count-weighted RP, one owner replay, annotated H5AD delivery, source conservation
and fresh-process Session recovery with scripted language decisions. It also
qualified a tiny weighted-versus-binary native RP counterexample. Those results
are reused as historical evidence; the expensive analysis is not repeated here.
Its observed continuous live-model attempt chose an empty resource question when
the reviewed resources were selected and a group file was missing. It did not
execute science and remains a failed live-entry observation.

## Findings and exact correction

| Finding | Classification / disposition |
| --- | --- |
| Canonical cCRE adoption, historical Run identity or binary conversion imposed on ordinary peaks | **KEEP existing correction**: UA3.6.4c already removed these requirements. Direct method and source qualification are explicit; no reopening or new adapter. |
| Exact counts/measurement, resource compatibility, primary scope and groups for full annotation | **KEEP**: required by the qualified computation and lineage. Removing them would change meaning or permit unsupported inputs. |
| Group file/obs column and count layer interaction | **KEEP**: typed `request_roles`/`answer_roles` continuation and scalar literal binding already work; source and original objective survive supplementation. Bad provider decisions do not establish that these mechanisms are absent. |
| Empty resource-choice guidance asserts a missing reference | **SIMPLIFY**: local presentation error reproduced through the ordinary MAESTRO HTTP submission path; corrected below. |
| Provider asks for resources instead of missing groups | **DEFER**: selecting the proper prerequisite is LLM semantic work. No automatic resource→role conversion, hidden workflow completion or claim of repaired live planning. |
| General resource discovery, arbitrary uploads/tissues/measurements, automatic clustering/correspondence | **DEFER**: unified attachment/application handoff or additional scientific qualification. No new platform, marker database, method or Registry redesign. |

The reproduced request has a selected peak H5AD and reviewed resource InputSet,
but a scripted Interpreter chooses `request_resources`. Existing exact-catalog
logic correctly offers no **additional** InputSet when the selected one adds no
new values. The response previously asserted that “an operator must provision the
required reference and source declarations”. Empty offers establish no such fact:
the resources may already be supplied, unavailable, or incompatible with captured
values. Scientific sufficiency remains owned by typed contracts and science owners.

Only `application/responses.py:clarification_text` changes in production. Empty
resource guidance now says:

> Your original request and file are preserved. No additional reviewed resource choice is available for this request. Check the selected inputs and declarations; an operator can help review resource availability.

The correction removes an unsupported provisioning instruction. It **does not
unblock the erroneous provider decision** or assert that selected resources are
qualified. The pending resource prerequisite, catalog checks, source identity,
planning/science admission and nonempty-choice text are unchanged. Existing saved
display records and historical scientific artifacts are not migrated.

One regression in `tests/annotation_integration/test_external_entry.py` exercises
this case through upload/selection and HTTP submission, with exact source/resource
retention, zero annotation/planning calls, no Revision and idempotent retry.
This report is the only new repository file for C.

## Focused verification and preservation

Commands use the agent environment Python with `-B`, `PYTHONPATH=src:.`, disabled
bytecode, `pytest -q -p no:cacheprovider`, isolated basetemps/caches and all `RUN_*`
gates removed. R contains `run_checks.py`, exact command JSON, logs and exit records.

| Run | Observed result |
| --- | --- |
| New HTTP regression before the correction | **1 failed in 2.70 s**, exit 1, at the misleading-text assertion; the intended failure reproduces the defect |
| Five focused modules: annotation external entry, user workflow, historical annotation integration, marker owner helpers and Web resource continuation | **106 passed, 1 setup error, 1 warning in 184.88 s**, exit 1; the new regression passes |
| Isolated rerun of `test_direct_input_rejects[nonfinite]` | **1 passed in 2.17 s**, exit 0 |

The setup error occurred in the unchanged `authority_dag.test_propagation.completed`
fixture while creating its upstream tiny Run: `PersistedRunState` rejected
`updated_at` preceding `created_at`. It occurred before annotation qualification
or the changed clarification renderer. The root cause of that timestamp ordering
is not qualified here; no clock/runtime fix, retry policy or validation exception
was introduced. The full focused batch is not described as a clean pass. Its
one warning is the intentional duplicate-cell input fixture.

These are **synthetic/software and scripted-provider checks**. Integration fixtures
run their small upstream owners but stub the MAESTRO annotation backend; they
exercise real source qualification, scoring/propagation, publication, replay
comparison, authority, reports, HTTP downloads and recovery. The historical binary
working-copy test uses a pinned test function to assert binary input and source
immutability. None of these is a new native weighted-RP/Presto or biological
annotation execution. Previously accepted native numerical and real-PBMC evidence
remains the basis for unchanged scientific behavior.

Covered cases include counts/semantics/coordinates/assembly and resource rejection,
exact groups and nulls/ties, count-layer/obs-column binding, missing-group upload
continuation, resource choice/tamper/stale rejection, annotated-H5AD preservation
and safe export, no matrix authority on the direct route, frozen historical source
authority requirements, terminal zero-backend reuse and Session reopening.
No full repository suite, expensive PBMC rerun, live provider, native browser,
external resource acquisition, package installation or scientific parameter change
occurs.

Final preservation checks confirm all **674 unaffected tracked files** retain
their initial hashes, with exactly the two intended tracked changes. The B report
and operator configuration hashes and all 1,274 eval names/sizes/mtimes are unchanged.
The historically authority-bound files retain the exact accepted hashes:

- `marker_annotation.py`: `e69739983c6acc99177b66c63a84ab641031e87250afdbe5cf1cb59517135348`.
- `r/maestro_markers_v1.R`: `2b62353381ed385d7f6198e168dc3a9f68b610e1e0ffc2f56a3c0d40817f5ba4`.

`preservation-final.json` records the comparison. No historical scientific artifacts,
authority records, native source or accepted workspaces are edited. HEAD/local
`origin/main` remain at the baseline and the index stays empty. `git diff --check`
and a separate whitespace check of the new report pass. The textual clarification
change has no scientific authority, replay, count-weighting or binary-compatibility
consequence.

## Closeout

Work Unit C can close after the bounded text correction and focused validation.
The useful scientific simplification was already implemented in UA3.6.4c;
this unit preserves it and corrects one misleading user instruction. Complete
live resource/group clarification, resource provisioning/selection and unified
attachment interaction remain subsequent application/UA3.6.5 work. No scenario
is promoted from PARTIAL.

Work Unit D may begin as a separately scoped task; its specification was not
provided here and no D work is started. C does not justify expanding annotation
science or rerunning PBMC acceptance.
