# UA3.4.2 — BAM Web upload and qualified context selection

Implemented on 2026-10-09, continuing the unstaged UA3.4.1 work. No commit/push.
This records the original stage checkpoint. The [UA3.4.3 closeout](ua3.4.3-fastq-web-upload.md)
supersedes its repository and Git status.

## A. Actual baseline

Branch `main`; HEAD, local `main` and local `origin/main` all identify
`c885a77ed26e720e93cb41f0acba9ddb355688c3`. Initially 11 tracked files were modified,
five new UA3.4.1 files and existing `evals/` were untracked, and the index was empty.
The 16 UA3.4.1 files and complete tracked diff were saved outside Git before editing.
No competing task beyond this delegated team
was found. No reset, clean, fetch, pull or discarded work.

Checks against the prior accepted inventories preserve 705 evaluations, 198 output
files and 208 files across the existing PBMC application and Web workspace. Full
SHA-256, size, mtime and mode checks pass. The external operator configuration
retains SHA-256 `dd86afd663fc2345b542bd78fa4bf97a5ebbfaf6d2557e84736bcddb4a6d4389`;
the 5,231,645,507-byte EpiZoo checkpoint retains SHA-256
`6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`.
Existing scientific state, input bytes and configuration were not modified.

## B. Incremental implementation

Modified beyond UA3.4.1:

- `src/agent/application/uploads.py`
- `src/agent/application/local_resources.py`
- `src/agent/application/dialogue_execution.py`
- `src/agent/web/app.py`
- `src/agent/web/config.py`
- `src/agent/web/static/app.js`
- `src/agent/web/static/index.html`
- `tests/web/test_upload_static.py`
- `tests/web/test_fragments_selection.py` (its formerly unsupported BAM case now uses FASTQ)
- `README.md`

Added:

- `tests/application/test_bam_uploads.py`
- `tests/application/test_bam_composition.py`
- `tests/application/test_bam_compound_recovery.py`
- `tests/web/test_bam_selection.py`
- `tests/web/test_bam_execution.py`
- This record.

The same upload owner/root/staging/capacity/catalog now accepts one `input_type=bam`
source, stores a generated `.bam` name and registers the exact bytes through UA2.3.
H5AD and fragments storage identities and optional TBI pairing remain unchanged.
There is no additional service, source schema/store, authority or workflow engine.

`bam-science.v1` supplies the registered `raw_input_paths`, `source_path` and
`source_sha256` plus explicit Registry-typed inspection/producer/downstream inputs.
It chooses no operation. Exact consumption guards check actual compiled and
accepted resolved arguments. Only presentation of compiler-authored missing BAM
prerequisite facts was added to `dialogue_execution.py`; intent/plan parsing and
scientific execution are unchanged.

## C. Supported BAM and qualified context

Transport registers bytes without claiming BAM readability. The unchanged raw
owner inspects BAM through pysam/htslib. Inspection needs only the selected source
and managed output directory; species, assay and source assembly are optional
explicit declarations. Its bounded observations and `CB` tags do not establish
corrected-barcode history or authorize production.

Public production remains `agent-cb-paired-atac.v1`: one BAM/group/library/namespace;
human/hg38 or mouse/mm10; exact compatible FAI names/lengths; paired primary ATAC
alignments; explicit `CB` corrected-identifier / already-corrected library policy;
declared unshifted coordinates and retained duplicate pairs. No inference, barcode
correction, alignment, alias repair or neutral producer exposure was added.
Historical source processing remains declared, not independently reconstructed.

Required producer inputs are the selected BAM path/SHA, pinned intake manifest
path/SHA, pinned library context path/SHA, pinned reference bundle path/SHA,
source profile and managed output directory. Namespace/group/barcode policy come
from the existing library context, rather than redundant Web fields. Intake may
be an explicit qualified pair or the exact semantic output of an inspection of
this selected source. Scientific owners validate the complete context/reference.
No BAI/CSI is required, uploaded or implicitly selected in this integration.

An old library context for the original file location is not applicable merely
because the uploaded BAM has identical content: the stored source and its intake
must have the exact existing identity relationship. Operator qualification for
that completed source is required; Application never rewrites the context.

## D. UI and operator provisioning

Choose **BAM**, attach one source, upload, select its completed label and submit
an ordinary inspection request. To request qualified preparation, choose an
approved BAM context in the existing composer selector. Source, context and
model selections are captured per turn and rechecked after refresh/restart.
Natural language selects tools; typed InputSets supply scientific declarations
and numerical parameters. Conversational collection remains UA3.5 work.

Existing InputSets use `bam_companion: true`. Explicit selection wins; only exactly
one explicitly marked `bam_default: true` may supply a default. Zero defaults
leaves inspection usable; ambiguity is reported only when actual preparation
requires context. EpiZoo and fragments contexts cannot substitute for BAM.
The browser receives labels/opaque IDs and flags, not paths, digests or payloads.

The external operator JSON contains no activated BAM contexts and is unchanged.
The following valid typed example shows the corresponding InputSet entry. Its
variables must be the operator's actual qualified resource pins for the uploaded
BAM; it creates no qualification and does not edit configuration:

```python
from agent.web.config import ScientificInputSet

bam_context = ScientificInputSet(
    input_set_id="reviewed-bam-library",
    label="Reviewed human hg38 paired ATAC library",
    bam_companion=True,
    execution_inputs={
        "species": "human", "raw_assay": "SCATAC",
        "source_genome_assembly": "hg38",
        "source_profile": "agent-cb-paired-atac.v1",
        "library_context_path": qualified_context_path,
        "library_context_sha256": qualified_context_sha256,
        "reference_bundle_path": qualified_reference_path,
        "reference_bundle_sha256": qualified_reference_sha256,
    },
)
```

For JSON, use the same fields in an `input_sets` entry. A prepare-only plan also
needs `intake_manifest_path`/`intake_manifest_sha256`; an explicit inspection→prepare
edge may supply them instead. Operators use existing library-context construction
and publication, with actual source history, selection/group/namespace and barcode
declarations. No automatic admission or default insertion occurs here.

## E. Integrity, feedback and results

Existing private streaming writes/hash chunks of at most 1 MiB, exclusive publication,
single-record atomic discovery, safe names, retries and integrity checks remain.
Configured per-file bounds remain 256 MiB by default, at most 1 GiB. Oversize errors
report actual bytes and ask the operator to review `upload_max_bytes`; limits are
never removed. Upload itself invokes no providers/science or scientific Revision.

New consumption checks full source bytes before providers and again after planning.
Plans must inspect this singleton source or prepare its exact path/SHA, selected
profile/context/reference and explicit intake. Source overrides and substitutions
fail closed. Independent owners retain qualification and scientific authority.

Missing source/context/reference/profile/intake have reviewed safe BAM errors when
admission or the compiler establishes that precise fact. Partial isolated request
digests can retain the compiler's safe `UNAUTHORIZED_REQUEST_INPUT` finding; no
exception prose, language heuristics or plan repair manufactures prerequisites.
Wrong correction policy, transplanted context and reference/source mismatch retain
owner error codes. Unsupported interpreter intent remains clarification.

Accepted evidence/reports and existing UA3.3 matrix/QC/selection downloads remain
bound to the correct accepted chat result. Downloading runs no science and creates
no Revision. Uploaded BAM and fragments/TBI have no new download class.

Completed historical reads/retries do not require the source. Direct accepted BAM
production presentation can also complete after source deletion/change. For an
unfinished compound inspection→production presentation, the unchanged raw-evidence
owner requires original bytes and the intake's captured mtime. That state remains
`finalizing` with verified production retained until the exact source is restored;
restoration allows completion without providers, producer recomputation or independent
BAM reconstruction. Raw-evidence reconstruction remains under its existing owner.
No historical scientific-authority expansion was introduced to bypass this rule.

## F. Validation

Commands use `AGENT_PYTHON`, set to the existing environment's interpreter; no
installs, references, live providers or GPU inference. Test-only synthetic qualification and explicit permissive downstream
thresholds do not establish biological quality or default thresholds.

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest tests/application/test_bam_uploads.py tests/application/test_uploads.py tests/application/test_fragments_uploads.py tests/web/test_upload_api.py -q -p no:cacheprovider
# 87 passed in 2.65 s

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest tests/web/test_bam_execution.py -q -p no:cacheprovider
# 18 passed in 31.03 s

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest tests/application/test_bam_compound_recovery.py tests/application/test_bam_composition.py tests/application/test_registered_bam_integration.py -q -p no:cacheprovider
# 69 passed in 44.17 s
```

Additional focused BAM selectors/config/composition: 121 passed in 1.73 s.
Preserved local-resource/fragments/selection/UA2.3 boundaries: 277 passed in 37.00 s.
Affected static tests: 35 passed. Native Firefox: 14 passed using one tiny explicit
synthetic source, two planned inspections, one production and one independent
verification. Upload/refresh/restart/result reads add no providers/science.
The native `native_check.py`, `native-checks.json` and screenshot evidence are
retained outside Git; the script uses the same installed interpreter.

Actual owners exercise human/mouse inspection/preparation, fresh semantic intake,
missing/invalid contexts, exact selected source, source drift, retries/restart,
unsupported requests and direct/compound historical completion. One synthetic
BAM-derived QC→selection→matrix plan verifies all five eligible downloads against
accepted byte size/SHA, with no science on delivery or retry.

The final affected-domain regression passes **956 tests, zero failures/skips,
three existing louvain/Numba warnings, 358.21 s**, exit 0. It includes the entire
existing Web domain and only affected Application registration/composition/recovery
and UA3.3 delivery boundaries. No production edits followed this passing run.
No full repository suite ran. Exact command:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 "$AGENT_PYTHON" -m pytest \
  tests/web tests/application/test_local_resources.py tests/application/test_uploads.py \
  tests/application/test_fragments_uploads.py tests/application/test_bam_uploads.py \
  tests/application/test_h5ad_composition.py tests/application/test_fragments_composition.py \
  tests/application/test_bam_composition.py tests/application/test_bam_compound_recovery.py \
  tests/application/test_registered_resource_integration.py tests/application/test_registered_bam_integration.py \
  tests/application/test_scientific_matrix_artifacts.py tests/application/test_matrix_delivery_privacy.py \
  tests/application/test_matrix_delivery_stream.py tests/application/test_scientific_table_artifacts.py \
  tests/application/test_table_delivery_privacy.py -q -p no:cacheprovider
```

Real biological acceptance is separately unavailable. Previously accepted BAMs are
tiny synthetic fixtures. Available Descart/Dictys BAMs have no reviewed Agent
paired-ATAC history/library/reference qualification. No biological source was
relabeled, trimmed, downloaded or declared qualified from tags.

## G–I. Architecture, repository status and next stage

All 161 checked protected source files match the UA3.3 baseline, including the
Registry/scientific owners, Planner/compiler/providers/schemas, interactive facade,
Session state and delivery owners. Registry remains 23 with unchanged semantics.
RunStore and scientific authority/Revision schemas remain intact.
UA3.4.1 files remain present; its record and core execution/composition/transport
tests remain byte-identical. H5AD/fragments and existing rich result/download
behavior pass the affected checks. No external operator configuration activation.

UA3.4.1 and UA3.4.2 remain unstaged together: 12 tracked files modified and 11 new
files, plus the preserved untracked `evals/`. All 16 captured UA3.4.1 files remain;
seven are byte-identical and nine received the listed narrow extensions. The index
is empty and baseline refs remain unchanged. No commit or push. Acceptance logs
and preservation records are retained outside Git.

The smallest UA3.4.3 slice is one explicitly grouped supported FASTQ library using
the existing UA2.4 registered collection and the same bounded upload transport.
First establish exact required read roles/layout and optional supported index role;
publish one discoverable complete collection, retain exact membership/order and
operator-qualified library/reference declarations, then ordinary inspection and
existing producer binding. Avoid grouping from first matches or guessed history.
Preserve all three existing upload roles and close UA3.4 with focused multi-file
atomicity, scripted execution and bounded browser checks. No FASTQ work started.
