# UA3.4.3 — FASTQ library upload and UA3.4 closeout

Implemented 2026-10-09, continuing the existing UA3.4.1/UA3.4.2 work.
This record separates Web software acceptance from scientific qualification.

## Baseline and implementation

Started on `main` at `c885a77ed26e720e93cb41f0acba9ddb355688c3`; local `main`,
local `origin/main` and independently read remote `main` matched. Twelve tracked
files were modified and eleven UA3.4 files were new, all unstaged, plus existing
`evals/`. All 23 existing files and the tracked diff were captured outside Git.
No reset, clean, fetch, pull or competing editing task. Earlier stage records
describe their original checkpoints; this closeout supersedes their Git status.

FASTQ extends the same upload owner, approved root, private staging, locks, limits
and local-resource catalog. No extra upload service, collection store, scientific
tool or workflow engine. A narrow collection-record variant in the same catalog
persists existing UA2.4 member references and its unchanged collection fingerprint.
Legacy record serialization and inline collection attribution remain exact.

Incremental changed files:

- `src/agent/application/uploads.py`, `local_resources.py`, `dialogue_execution.py`
- `src/agent/web/app.py`, `config.py`
- `src/agent/web/static/app.js`, `index.html`, `styles.css`
- `tests/web/test_upload_static.py`
- `README.md`
- Earlier UA3.4.1/UA3.4.2 records: deployment paths removed for commit privacy;
  original scientific acceptance facts retained.

New files: `tests/application/test_fastq_uploads.py`, `test_fastq_composition.py`,
`tests/web/test_fastq_selection.py`, `test_fastq_execution.py`, and this record.

## Exact FASTQ contract and interaction

One explicitly named library may contain multiple complete lane/chunk groups.
The existing owner defines both supported layouts:

| Layout | Required genomic roles | Required barcode role | Optional role |
| --- | --- | --- | --- |
| `tenx-atac-r1-r2-r3.v1` (A) | R1, R3 | R2 | I1 sample index |
| `tenx-atac-r1-i2-r2.v1` (B) | R1, R2 | I2 | I1 sample index |

Before upload the researcher declares library name, layout, role, three-digit
lane/chunk and plain/gzip encoding. These are explicit grouping declarations,
not reconstructed sequencing history or biological namespace. The ordinary file
picker uploads one member at a time. Original names are safe display labels,
with no role/layout inference. Generated names mechanically encode the declared
roles for the unchanged scientific filename contract; files are never renamed
after registration. Scientific owners check actual FASTQ/gzip data and lockstep
read synchronization. Their existing plain/gzip FASTQ semantics are unchanged.

Select completed members and choose **Complete FASTQ library**. The existing
owner's layout tables check every group, including optional I1, and supply exact
missing-role/meaning feedback. Conflicting library/layout, duplicate roles,
incomplete groups and changed sources fail closed. Member records remain distinct
from complete selectable collections. Completion creates no Session/Revision,
provider call, raw inspection or production.

The existing PUT adds `input_type=fastq` with the six declaration fields
`library_id`, `fastq_layout`, `role`, `lane`, `chunk`, `compression`.
`POST /api/v1/uploads/fastq-collections` accepts `collection_id`, display `label`
and opaque `member_ids`; callers cannot submit paths or digests. `/resources`
projects completed members separately as `fastq_members`; scientific choices
contain complete collections plus the existing three families.

Select a complete collection and submit an ordinary inspection request. Inspection
needs no producer resources; captured layout also supplies its explicit TENX_ATAC
assay. Preparation remains ordinary interpreted planning through the existing
`prepare_scATAC_fragments` and exact intake/context/reference semantic ports.
Application builds no scientific steps or downstream defaults. Unsupported intent
remains unsupported; existing eligible results appear on the corresponding chat
message and download without science or another Revision.

## Qualified resources and missing prerequisites

Public production remains human/hg38 or mouse/mm10, with existing exact library
membership, namespace, raw-barcode policy and applicable whitelist, compatible
reference bundle and qualified Chromap/index/packaging runtime. Original-location
contexts are not silently rebound to uploaded paths. Existing explicit producer
subset semantics remain under the library/producer owners; no silent selection.

An operator InputSet uses `fastq_companion: true`; explicit selection wins, and
only one explicitly marked `fastq_default: true` can supply a default. Missing or
ambiguous context does not obstruct inspection. Captured assay/layout and source
membership cannot be overridden. The operator configuration has no activated
FASTQ contexts and was not edited. EpiZoo resources are unrelated.

Valid typed configuration example, using actual qualified resource pins:

```python
from agent.web.config import ScientificInputSet

context = ScientificInputSet(
    "reviewed-fastq", "Reviewed human hg38 FASTQ library",
    {
        "species": "human",
        "library_context_path": qualified_context_path,
        "library_context_sha256": qualified_context_sha256,
        "reference_bundle_path": qualified_reference_path,
        "reference_bundle_sha256": qualified_reference_sha256,
    },
    fastq_companion=True,
)
```

Use the same fields in an operator JSON `input_sets` entry. The existing library
context owner must qualify actual intake group IDs, membership, namespace and
whitelist policy for those stored sources. A prepare-only plan needs an explicit
pinned intake; an explicit same-plan inspection edge can supply it instead.
`AGENT_CHROMAP_BIN` and `AGENT_CHROMAP_INDEX_ROOT` retain their existing qualified
runtime/reference-index requirements. No missing resources are downloaded or
manufactured by Web. Numerical parameters still require typed selections;
conversational prerequisite collection remains UA3.5 work.

Admission and exact compiler missing-port facts supply reviewed context/reference/
intake errors. Owner findings remain authoritative for corrupt reads, mismatched
whitelists/resources and runtime qualification. No exception-prose parser, keyword
router, hidden repair or generic prerequisite solver.

## Identity, atomicity and historical lifetime

Each member captures its complete SHA/size, source locator and immutable six-field
attribution in the existing source record. Generated paths use only a library/layout
hash and declared role/group/encoding. Per-file limits remain 256 MiB by default,
at most 1 GiB; concurrency and shared 128-file bounds remain. Copy/hash writes are
at most 1 MiB. Interrupted or oversized members never become registered sources.

Upload-ID retries preserve declarations and bytes. Another upload ID cannot occupy
the same role slot, including alternate compression. Failed publication leaves no
complete collection; complete crash-orphan bytes retain the existing retry policy.
Collection completion rechecks every member's pins/snapshots and publishes one
immutable record atomically. Canonical member ordering and UA2.4 fingerprints are
unchanged. Discovery filters before its returned-choice bound, so 128 members
plus their collection remain discoverable without increasing scan/result limits.

Actual compiled inspections consume the complete selected paths and captured
assay/layout. Preparation uses the exact selected intake and qualified context/
reference; accepted producer provenance checks each consumed source's path/SHA/size.
New consumption checks every member before providers and after planning.

Completed historical result reads/retries remain source-independent. Unfinished
compound raw-inspection/production presentation retains verified production while
the existing raw-evidence owner requires original sources and captured mtime;
exact restoration permits completion. BAM's same distinction is covered in the
combined regression. Scientific authority and Session/Revision were not expanded.

## Validation and real acceptance

Set `AGENT_PYTHON` to the installed agent interpreter. Commands use existing
dependencies/resources, scripted providers, no GPU or full repository suite:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest tests/application/test_fastq_uploads.py tests/application/test_uploads.py tests/application/test_fragments_uploads.py tests/application/test_bam_uploads.py tests/web/test_upload_api.py tests/web/test_fastq_selection.py -q -p no:cacheprovider
# 149 passed in 3.20 s

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 RUN_CHROMAP_QUALIFICATION=1 "$AGENT_PYTHON" -m pytest tests/web/test_fastq_execution.py -q -p no:cacheprovider
# 21 passed in 56.87 s; AGENT_CHROMAP_QUALIFICATION_RECORD supplied externally

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest tests/application/test_fastq_composition.py tests/web/test_fastq_selection.py tests/application/test_registered_fastq_integration.py tests/application/test_local_resources.py -q -p no:cacheprovider
# 306 passed in 26.43 s
```

Static boundaries: 39 passed. Final-code native Firefox: 13 checks passed,
one ordinary inspection; uploads/group completion/refresh/fresh-server reads add
no providers/science. Existing H5AD, fragments/TBI and BAM controls also pass.
Native script, screenshots and `native-checks.json` are retained outside Git.

Actual tiny qualified Chromap tests cover both layouts, optional I1, plain/gzip,
multi-lane/chunk membership, fresh intake, independently verified authority,
missing/conflicting context, mutation, retry and historical recovery. A tiny
downstream plan executes existing QC/selection/matrix owners and verifies all five
existing downloads by accepted size/SHA. Synthetic resources and permissive test
criteria qualify software, not biological quality or species defaults.

One real HTTP acceptance streams the complete preserved reviewed PBMC library:
eight files, **4,693,218,895 bytes**, both lanes, declared A/I1 roles taken from
accepted intake facts. All original qualification SHA/size pins match. An isolated
service explicitly uses the existing 1 GiB per-file limit and 1 MiB chunks.
Upload/completion run no providers/science. Ordinary scripted inspection executes
once, independently verifies **READY / eight files / two groups**, creates one
Revision and offers evidence/report. Exact retry runs no additional science.
The check completes in **44.542 s**; original source metadata/qualification bytes
and historical producer context are unchanged. Its script and `acceptance.json`
are retained outside Git. No alignment, QC, matrix or live-provider rerun.

This is real transport/bounded-intake acceptance, not new uploaded-source producer
biological qualification. External fragments/BAM real production qualification
and deployment-specific FASTQ contexts remain separate provisioning requirements.

## UA3.4 closeout

**ACCEPTED — UA3.4 multi-format Web input integration complete.**

The final affected Application/Web regression passes **1,217 tests, zero failures
or skips, three existing louvain/Numba warnings in 470.88 s**, exit 0. No production
edits followed that passing run. Exact scope command, with the qualified runtime
record supplied externally:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 RUN_CHROMAP_QUALIFICATION=1 "$AGENT_PYTHON" -m pytest \
  tests/web tests/application/test_local_resources.py tests/application/test_uploads.py \
  tests/application/test_fragments_uploads.py tests/application/test_bam_uploads.py \
  tests/application/test_fastq_uploads.py tests/application/test_h5ad_composition.py \
  tests/application/test_fragments_composition.py tests/application/test_bam_composition.py \
  tests/application/test_fastq_composition.py tests/application/test_bam_compound_recovery.py \
  tests/application/test_fastq_resources.py tests/application/test_registered_resource_integration.py \
  tests/application/test_registered_bam_integration.py tests/application/test_registered_fastq_integration.py \
  tests/application/test_registered_fastq_production.py tests/application/test_scientific_matrix_artifacts.py \
  tests/application/test_matrix_delivery_privacy.py tests/application/test_matrix_delivery_stream.py \
  tests/application/test_scientific_table_artifacts.py tests/application/test_table_delivery_privacy.py \
  -q -p no:cacheprovider
```

The scope covers all four upload families,
typed companions, ordinary planning, incomplete/incompatible inputs, multi-turn
and restart, evidence/report/figures and the five eligible scientific downloads.
No new scientific download classes or neutral public producers were added.

Registry remains 23. All 161 checked protected source files match the UA3.3 baseline:
scientific owners, orchestration/Planner/compiler/providers/schemas, interactive
facade, Session state and delivery owners. Small shared changes admit existing
collection bindings in the Web worker and split discovery before existing bounds;
they select no workflow and weaken no scientific contract.

Preservation checks cover all 705 evaluations, 198 scientific outputs, existing
208 application/Web state files, operator configuration and the original EpiZoo
checkpoint. No data, credentials, machine-private paths or temporary outputs are
included in the intended commit. The combined diff includes only implementation,
focused tests and documentation. Git finalization uses the requested coherent
message and normal compatible push, with no forced update or automatic divergence repair.

Next UA3.5 slice: one bounded typed conversational handoff for missing declarations
on these existing inputs. Let users supply a reviewed missing prerequisite, reuse
their qualified selection and continue the ordinary plan; preserve source/Revision
lineage and existing result downloads. Do not infer biological history or thresholds.
UA3.5 is not implemented here.
