# Native scATAC peak calling

`call_scATAC_peaks` discovers experimental peaks with optional isolated MACS3 3.1.0. It is one Registry operation; import, QC, cell selection, feature counting, PCA, neighbors, Leiden, UMAP and chromVAR remain independent operations.

Supply ordinary eligible 10x-style fragments with their actual processing history, exact assembly/reference and one processing-library namespace. Existing external import qualifies them without requiring an earlier Agent receipt. Accepted canonical FASTQ/BAM/external fragments are also usable. Source intervals must already be Tn5-adjusted; no second shift or contig renaming occurs. The existing Web raw-input declaration supports human/mouse reference routes; other species can use the species-neutral scientific source/reference owners. An installed reference is not proof of an external file's alignment history.

State an effective genome size and its scientific basis/applicability. The owner never substitutes total FASTA length. The native q-value cutoff defaults to 0.05. Exact contig exclusions are optional; none are inferred. Every call selects one namespace. Omit `library_namespace` for a unique library; this selector is distinct from the namespace assigned by import.

Choose all eligible library records, an existing QC-selected population, an explicit ordered-cell artifact, or an ordinary one-barcode-per-line file with a captured checksum. Whole-library observed barcodes are not QC-called cells. An accepted clustering can select a group only together with its exact fragment-derived count matrix: original-file identity, ordered cells and namespace/barcode correspondence must agree. Cell-count equality never establishes correspondence.

Native FRAG paired-fragment calling uses a deterministic unit-record projection and `--max-count 1`. Original support totals and producer meanings are retained separately. Read-pair/PCR support is not independent molecule count or cell-by-peak count. Weighted calling is deliberately not exposed. Native `--barcodes` derives some defaults before subsetting in 3.1.0; Agent prepares the complete selected population first so other barcodes do not influence those defaults. No native mathematics is replaced.

Results retain original narrowPeak, summit BED and XLS, an exact coordinate-ID feature BED in native order, distributions, parameters and lineage. BED/narrowPeak coordinates are zero-based half-open; native XLS start/summit positions are one-based. Native probability fields are minus-log10 p/q; fold enrichment, pileup and display scores are distinct quantities. Peak-calling q-values are not differential accessibility between experimental conditions.

Nonempty calls publish `regulatory-feature-reference.v1` with category `peak_set`, plus exact ordered population ports. Pass these and the original fragments to `build_scATAC_cell_by_features`. This reuses the existing fragment-record overlap engine and preserves feature/cell order and integer counts. Experimental peaks are not automatically installed as shared reference resources. Empty calls retain honest native outputs but supply no usable downstream feature reference; thresholds are never relaxed to force peaks.

The resulting H5AD can supply existing PCA/neighbors/Leiden/UMAP and chromVAR. Explicit TF-IDF/L2 preprocessing is a PCA choice, not a mandatory peak pipeline. ChromVAR still requires compatible measurement, genome, coordinates and motif resources. Genomes with the same display assembly but different names or bytes must not be substituted. Preserve the matrix's recorded assembly declaration: changing `hg38` to `GRCh38` is not a way to select a differently labeled resource. When several installed genomes match the declared assembly, `genome_resource` selects an exact installed label or ID. The selected entry must still match that assembly. Operators may install an existing qualified reference's genome and explicit declaration in place, without copying its FASTA/FAI or changing another entry. This is not an assembly alias. Otherwise the task FASTA input remains available.

For ordinary Web use, install qualified QC resources once through `ScientificResourceLibrary.install('qc', ...)` in the configured `scientific_resource_root`. Installation reconstructs the native source/annotation relationship and pins its complete file closure. New QC checks that installed closure; it does not require a second `AGENT_QC_RESOURCE_CATALOG` or an acceptance-specific configuration. The legacy catalog remains an explicit standalone-owner route outside a configured library. A later request can choose a matching resource through ordinary clarification while retaining its explicitly selected accepted-result context, even with no new upload. Choosing a resource does not qualify source correspondence; the consuming scientific owner still checks the exact reference and fragments.

Accepted QC publication identity uses its saved qualification, backend and original catalog closure after restart. Changing the current operator catalog does not redefine that history. Original pinned files remain integrity-checked. New library-qualified QC records the immutable installed resource ID; accepted QC reuse does not acquire new-execution authority or depend on the current resource selection.

## Optional runtime

Use an isolated Python 3.12.16 environment at `agent-data/resources/runtimes/macs3/env`; never install this backend into Agent's Python 3.11 environment. Install `MACS3==3.1.0` using the resolved `requirements.lock.txt` retained with the qualified runtime. Upstream distribution/license files remain installed. Run `tests/macs3/qualify_runtime.py --environment /absolute/path/to/macs3/env` once to record a new runtime identity and dependency lock. Existing identities are not overwritten. `AGENT_MACS3_EXECUTABLE` may point to another independently qualified environment with the same parent layout. Backend absence does not affect unrelated tools or Registry construction.

Native execution uses structured subprocess arguments, two CPU threads, cancellation and a two-hour execution ceiling. The input preparation bounds the barcode population to two million, and the existing feature reference bounds peaks to two million. No universal scalability claim is made. Native diagnostics remain private; download preparation removes private command paths from XLS headers and metadata while preserving statistical rows. Accepted reads/recovery do not repeat native science.

See the [STE-2A acceptance report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-2/REPORT.md) for actual scientific/browser qualification and current readiness. Implementation availability alone does not establish complete researcher acceptance.

## Normal deployment and finalization

The configured library is the common resource owner. For an already qualified QC
bundle, an operator can install its existing bytes without changing the launcher:

```python
from agent.application.shared_resources import ScientificResourceLibrary

library = ScientificResourceLibrary(configured_scientific_resource_root)
library.install(
    "qc",
    {"qc_reference_manifest_path": absolute_manifest_path,
     "qc_reference_manifest_sha256": exact_manifest_sha256},
    label="Scientifically descriptive reference label", version="source release",
    provenance="Exact original source and qualification basis", usage="Authorized research use",
)
```

The [finalization record](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-2/finalization/REPORT.md)
contains the actual in-place installation, ordinary launcher/browser qualification,
parameter audit, preservation evidence and Git closeout. Legacy standalone
`AGENT_QC_RESOURCE_CATALOG` examples are not an additional Web deployment step.
The optional pinned MACS3 runtime remains isolated.

The accepted demonstration explicitly requested 1,000 fragment records and TSS
enrichment ≥2; the selection defaults remain 1,000 and ≥4. Its effective genome
size 2,913,022,398 was the explicitly requested MACS3 GRCh38 constant.
[MACS3 documentation](https://macs3-project.github.io/MACS/docs/callpeak.html) and
[deepTools](https://deeptools.readthedocs.io/en/develop/content/feature/effectiveGenomeSize.html)
identify its conventional non-N basis. It is not FASTA byte length or a new
read-length/MAPQ-specific mappability estimate; filtered-read applications may
justify a different explicitly declared estimate. Original results are retained.
