# Scientific storage and the fresh acceptance workspace

The FS-2 deployment uses `/home/likeyi/program/agent-data/` outside Git.
The existing ScientificResourceLibrary remains the shared discovery owner.
FS-3 retires the qualified legacy resource roots and obsolete development
execution trees. Selected byte-preserved final evidence lives under
`acceptance/historical/`; historical accepted records have not been rewritten.
FS-3B groups this evidence by scientific/engineering theme and milestone, removes
the remaining obsolete browser workspaces, and consolidates cleanup reports under
`acceptance/filesystem-cleanup/2026-10-21/`. Its `EVIDENCE.txt` and FS-3B mappings
resolve original recorded paths to retained current bytes; original report paths
remain historical facts.
The independent-project exceptions below retain their original ownership.

| Location | Owner and purpose |
| --- | --- |
| `resources/library/` | The active ScientificResourceLibrary index and installed specifications |
| `resources/models/epizoo/` | Original pretrained EpiZoo and optional SEAM checkpoint bytes |
| `resources/references/` | Exact vocabularies, genomes, annotations, TSS references and reference manifests |
| `resources/indexes/chromap/` | Existing qualified human/mouse native indexes with new manifest bindings |
| `resources/runtimes/` | Relocated pinned Chromap and MAESTRO source/R package closures |
| `datasets/original/` | Acquired biological inputs, including the retained PBMC FASTQs |
| `datasets/derived/` | Selected useful matrices, fragments, prepared features and annotation inputs |
| `workspaces/uat/` | Fresh ordinary ManagedWorkspace; no copied development Sessions or datasets |
| `config/` | Private operator configuration and the UAT launch command |
| `acceptance/` | Compact resource acquisition, qualification and deployment evidence |

`datasets/DATASETS.txt` records each retained dataset's meaning, provenance and
identity. `resources/RESOURCES.txt` records the FS-2 capability dependencies;
`resources/FS3-ADDITIONS.txt` records the optional reference additions and retirement.
These are human documentation; storing a file there grants no input qualification.
Future UAT data must be selected and uploaded through the normal Web interface.
`datasets/FS3-ADDITIONS.txt` documents the additional unique public inputs found
during legacy retirement, with their source and measurement limitations.

## Configuration and launch

The existing `agent` Conda environment holds the reviewed resource locations.
Reactivate it after an operator changes its variables, or use `conda run`.
The private `config/operator-uat.json` selects the new workspace, its upload
directory and the new library. It preserves the existing model profiles and
default profile but has an empty `input_sets` list.

Launch the normal application through the private deployment command:

```bash
/home/likeyi/program/agent-data/config/launch-uat.sh
```

The launcher activates the existing environment and invokes `python -m agent.web`
with the ordinary operator configuration. It introduces no separate admission or
execution route. The fresh workspace is prepared without starting a persistent
Web server or running comprehensive scientific UAT.

The unchanged default upload limit is 256 MiB. The retained canonical PBMC H5AD
fits; larger FASTQs require an explicit `upload_max_bytes` configuration decision
within the existing supported bounds before raw FASTQ UAT.

## Optional scientific dependencies

| Capability | Additional scientific resources |
| --- | --- |
| H5AD inspection, generic PCA, neighbors, Leiden and UMAP | Existing Python numerical stack and ordinary compatible inputs; no pretrained checkpoint required |
| Human/mouse EpiZoo | Qualified checkpoint plus exact species vocabulary/frequency files and the managed `third_party/epizoo/` backend/runtime |
| MAESTRO annotation | Qualified gene/signature references, pinned MAESTRO code, relocated R packages and the existing `agent-edger` R runtime |
| FASTQ preprocessing | Qualified Chromap binary/index, matching reference and explicit assay/library/barcode declarations |
| QC and feature matrices | Method-appropriate qualified references, exact scientific inputs and the pinned BEDTools runtime where required |
| Species adaptation | Explicit source/SEAM bundles, backend configuration, target genome/features and method-specific mapping resources |
| Differential accessibility | Explicit qualified edgeR runtime and the repository-controlled R scripts |

The older CellMarker source/table closure remains independent research evidence;
its former post-annotation corroboration operation stays retired. ICTA-3A adds a
separate internal CellMarker 3.0 static evidence resource from
[Zenodo 22808257](https://zenodo.org/records/22808257). Native MAESTRO continues
to use its own compatible gene/signature resources. Cell Ontology resources and
optional mapping dependencies are not required by Agent.

The five verified original ZIPs and derived `cellmarker3.sqlite` live under
`resources/cellmarker3/zenodo-22808257/`. Only `all_cell_marker.txt` supplies index
rows; the other archives are overlapping provenance collections. The
[`cellmarker` owner](../../src/agent/tools/data/cellmarker.py) streams the All ZIP,
preserves its 23 strings and one-based data-record positions, and creates three
SQLite indexes on `(species, marker)`, `(species, symbol)` and `(species, gene_id)`.
No TXT extraction, normalization, biological weighting or annotation is involved.

Provision with `build_index(all_zip, sqlite_path, expected_sha256=verified_sha)`.
It verifies every field and source position before publishing the database
atomically; an existing index is verified and reused, never overwritten. Install
through `ScientificResourceLibrary.install('cellmarker', resource, ...)`. The
resource has `schema='cellmarker-static-resource.v1'`, the selected `record_url`,
`index_format='cellmarker-evidence.sqlite.v1'`, an `archives` mapping keyed by all
five original ZIP names, and `index`. Every file descriptor contains its canonical
`path`, `sha256` and `size_bytes`. The existing library owns version coexistence,
immutable file checks and atomic installation. This kind creates no Planner or
Native Functions choice and grants no annotation-result authority.

An internal consumer explicitly selects a resource ID and resolves it once per
operation, then streams technical pages:

```python
from agent.tools.data.cellmarker import CellMarkerEvidence

evidence = CellMarkerEvidence(library, selected_resource_id)
for page in evidence.query(["Gpr116", "Adgrf5"], species="Mouse", page_size=128):
    for item in page["items"]:
        fields = item["fields"]          # all 23 original strings
        matches = item["matches"]        # input position, value and matched field
        source_rows = item["source_rows"]
        multiplicity = item["multiplicity"]
```

Names match `marker` and `symbol` exactly and case-sensitively; that does not
establish aliases. `gene_ids=["224792.0"]` matches original `gene_id` strings;
`"224792"` is a different query, with no floating-point or namespace conversion.
Species may be omitted to search all stored species. `filters` accepts exact
string values for other source columns, including empty strings when explicitly
querying missing fields. Optional context is never required. Empty marker/ID
strings match nothing; unmatched queries yield a completed page with no items.

Identical full records are grouped with their actual source positions and
multiplicity. Differing tissue, cell, publication, source, dataset and method
contexts remain distinct. Pages are transport boundaries: consume them all for
complete evidence, or close the iterator when stopping retrieval. Batch inputs use
a connection-local TEMP table, not an arbitrary marker-count limit. Each page
includes the query identity and static source/index identity. The main database
is opened read-only. Resolution verifies the library's file closure once;
subsequent queries/pages use existing exact-file snapshot checks. A fresh process
resolves the installed resource offline without rebuilding it. These internal
records are not an ordinary user-facing scientific result or model-context dump.

The [Gate 0 inspection](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-3a/gate0/REPORT.md)
and [ICTA-3A implementation evidence](../../../agent-data/acceptance/iterative-cell-type-annotation/icta-3a/REPORT.md)
record source semantics, actual counts, query limitations and verification.

The EpiZoo owner resolves `AGENT_EPIZOO_CHECKPOINT` and
`AGENT_EPIZOO_RESOURCES_DIR` at scientific consumption. An omitted Python
checkpoint/directory argument means the configured owner location; an explicit
Python argument takes precedence. Missing configuration fails at EpiZoo use,
without an old-directory fallback. Imports and generic PCA remain usable without
these variables. Checkpoint shapes, token order, filtering, inference settings
and scientific algorithms are unchanged.

The other existing operator bindings remain
`AGENT_ANNOTATION_{MAESTRO_ROOT,RSCRIPT,R_LIBRARY}`,
`AGENT_CHROMAP_{BIN,INDEX_ROOT}`, `AGENT_QC_BEDTOOLS` and
`AGENT_MATRIX_BEDTOOLS`. OS executables and the existing Conda R installation
remain in their qualified installation locations. Relocating that entire
environment is outside this deployment.

Ordinary Web QC uses the configured ScientificResourceLibrary installation and
exact file closure. It needs no separate qualification catalog.
`AGENT_QC_RESOURCE_CATALOG` remains the legacy standalone-owner route outside
a configured library; accepted historical catalog closures retain their meaning.
See the [peak calling guide](macs3-peak-calling.md) for the installation example.

Qualified macaque normalized TSS/QC assets are optional and use the existing
explicit neutral-reference owner/catalog workflow. Shared-library discovery of
generic QC is not implemented. Relocated scope/QC manifests acquire new bound
identities; old accepted fragment/preparation records must not be relabelled to
claim correspondence with them. Future processing establishes its own exact
scope/preparation binding through the ordinary scientific owners.

FS-3C integrates EpiZoo and EpiAgent source as code-only repository snapshots;
see the [backend installation guide](backends.md). Independent research checkouts
are not Agent import dependencies. Their scientific source, data, checkpoint and
Git index/ref bytes remain preserved. FS-3C records one metadata exception: the
independent EpiAgent `.git` container directory's timestamps changed; the cause
is unproven and no timestamps were restored. Their original notebook data paths
retain independent ownership.

Optional EpiAgent assets now live under `resources/models/epiagent/` and
`resources/references/epiagent/`; the model's `resources.json` records exact source
and canonical identities. The full cCRE BED reuses the existing canonical human
reference only after complete byte equality; its separate document-frequency array
is retained unchanged. These assets do not become generic startup dependencies,
a new Registry tool or a new entry in the existing nine-entry scientific library.

## Verification and retention

FS-2 checks relocated source/destination hashes, existing owner qualification,
native runtime compatibility and bounded tests. A small native annotation fixture
is explicitly simulated and establishes relocation behavior only. No new model
training, paid provider calls, biological evaluation or general UAT qualification
is implied.

The maintained M14.2 qualification capability now lives at
[`tests/epizoo_adaptation/qualify_real_backend.py`](../../tests/epizoo_adaptation/qualify_real_backend.py).
It takes an explicit `--epizoo` source directory (normally `third_party/epizoo`) and configured or explicit
checkpoint/resource locations. Its artificial training fixture remains an opt-in
native technical check. It uniquely checks the real pretrained EpiZoo/SEAM and
full native CUDA model path; mocked adaptation tests do not establish that
compatibility. FS-3B preserves the helper byte-for-byte without running training.
FS-3 retires the historical M14.3/M14.7 utilities, six UA3.5 runner/fixture files
and four obsolete two-stage experiment assets after preserving native and cold
recovery evidence and resolving callers. Maintained tests cover the relevant
fresh-process zero-science recovery obligations. Current direct-v4 planning,
production `PlanningScope` and independent scientific checks remain maintained.

Current benchmark evidence readers take an explicit consolidated evidence path.
New benchmark outputs and disposable workspaces belong outside the repository,
in task-owned locations such as `agent-data/scratch/`. The old repository
`evals/` and `outputs/` trees are retired. Source/Session history is never edited
to create authority for a relocated scientific result.

The seven FS-3 external-project exceptions and the loose human FASTA retain their
independent ownership. FS-3C checks complete loose/canonical reference identity and
FASTA/FAI geometry; matching bytes alone do not establish exclusive Agent ownership.
The independent backend/data consumers remain untouched. Separate research projects,
Conda/R installations, shared build directories and personal browser state are
outside Agent cleanup. Exact findings and remaining exceptions are recorded in the
FS-3C report under `acceptance/filesystem-cleanup/2026-10-21/fs3c/`.
The [development guide](operating-and-development.md) records future placement
and disposable-output cleanup conventions.
