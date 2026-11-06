# Raw scATAC processing with native scientific tools

Agent plans and invokes supported native functions through the existing Registry,
Executor, scientific publication, RunStore, Revision, Session and Web owners.
Scientific task requests and explicit native-function requests use this same route.
The native-first replacement has completed engineering closeout with
**CONDITIONAL GO within bounded scope** in the
[replacement report](../../../agent-data/acceptance/raw-intake-simplification/native-first/REPORT.md).
Native execution, saved-result delivery and historical compatibility pass bounded
checks. The single final genuine Firefox/OpenAI request stopped at resource-selection
clarification; successful post-correction live planning and presentation remain
unqualified. This is not unrestricted input, provider or biological qualification.
The paused SRV-2C PBMC10k evaluation has not resumed.

## Scientific operations

| Input / goal | Registered interface and native function | Output |
| --- | --- | --- |
| Paired FASTQ lanes, barcode reads and reference | `run_chromap`, `index` / `map` | Native Chromap index or BED fragments |
| BAM to fragments | `run_snapatac2`, `pp.make_fragment_file` | Native plain, gzip or Zstandard fragments |
| Original fragments to native data | `run_snapatac2`, `pp.import_fragments` | Native fragment-storage H5AD with barcode observations |
| TSS enrichment | `run_snapatac2`, `metrics.tsse` | Native H5AD with QC observations and library TSS profile |
| Fragment-size distribution | `run_snapatac2`, `metrics.frag_size_distr` | Native library distribution in H5AD |
| Barcode selection | `run_snapatac2`, `pp.filter_cells` | Native selected H5AD, or the independent return when `inplace=false` |
| Peak, cCRE or supplied region matrix | `run_snapatac2`, `pp.make_peak_matrix` | Native cell-by-region H5AD |

These are individual operations selected by planning, not a mandatory fixed
pipeline. Native fragment import does not contain an accessibility matrix in `X`.
Build a matrix before passing data to PCA, LSI or an annotation method that needs
counts. Existing downstream native algorithms are unchanged. A compatible matrix
uses the ordinary published H5AD dataset port; no separate raw result catalog exists.

## Requests and exact inputs

A researcher can describe fragments, species, assembly, the intended reference,
QC/selection goal and matrix representation in ordinary language. For example:

> These are human PBMC scATAC fragments on hg38. Use the installed reference,
> inspect fragment QC, select suitable barcodes and build a cCRE matrix.

Missing references or ambiguous scientific choices may require clarification.
Internal source-profile identifiers and Agent processing namespaces are no longer
prerequisites for native raw processing. Barcodes are identifiers; import or QC
alone does not establish biological cells.

An explicit native request can name, for example, `snapatac2.pp.import_fragments`
with `sorted_by_barcode=false` and `min_num_fragments=1`, followed by
`pp.make_peak_matrix` with `counting_strategy=fragment`. Function-scoped kwargs
are preserved. Unspecified values use the installed native default. Agent forwards
invalid scientific values to the native function and surfaces its diagnostic;
it does not silently repair parameters or retry with a different method. Chromap
presence-only switches accept `true`; `false` cannot disable a native preset and is
rejected as unrepresentable, rather than being silently omitted.

Ordinary fragment tasks carry unknown barcode ordering through the existing input
handoff. Agent selects `sorted_by_barcode=false` and SnapATAC2 sorts the original
source itself. Researchers need no sorting flag or order detector. Explicit native
kwargs and explicitly declared barcode grouping take precedence; order is never inferred from a filename. This task choice is recorded separately from user kwargs
and native defaults; direct native-function requests omit task order knowledge and
retain `sorted_by_barcode=true` when the argument is omitted. Native BAM extraction
guarantees barcode-sorted fragments, so its import can retain that native default.
Chromap BED has no such guarantee. Broad planning remains model dependent; the
earlier ordering failure and explicit follow-up remain historical evidence in the
report, with bounded final alignment recorded separately.

File inputs use captured source roles and exact checksums. Native chromosome
sizes, annotation, peaks, whitelist, reference and index inputs have separate ports.
Plain fragments, gzip/BGZF and Zstandard are passed directly to native ingestion.
There is no compulsory Agent sorting, canonicalization, recompression or indexing.
A supplied TBI remains provenance; SnapATAC2 import does not consume it, so its
coordinate convention is not an import eligibility gate. The separate bounded
`inspect_scATAC_fragments` operation may describe a TBI header but establishes no
source/index correspondence or downstream scientific authority.

The existing ScientificResourceLibrary projects selected installed resources into
these ports. Task handoff binds FAI, GTF and regions to their particular native
functions; an available unused GTF does not require TSS QC for an import/matrix-only
goal. Source species/assembly bind to import, then follow the native H5AD metadata.
Supplied custom regions retain their declared assembly and exact
coordinates. Explicit native-function mode may instead bind the exact supported
reference files directly. There is no inferred chromosome aliasing, coordinate
conversion or species correspondence.

FASTQ lanes belong to the explicitly declared library. Matching R1/R2/barcode
lists are passed together to one Chromap map call; native software owns barcode
correction, mapping, duplicate handling and Tn5 shifts. Native BAM extraction uses
its requested barcode tag or regex and native paired-read rules. It does not inherit
the old Agent corrected-CB BAM policy by implication.

## Measurement and default differences

SnapATAC2 2.9.0 imports paired-fragment positions and lengths. Its `n_fragment`
excludes mitochondrial records. Column-five duplicate support contributes to the
native duplicate diagnostic; it is not a weight for the accessibility matrix.
Native import may reorder barcodes. Subsequent QC and matrix operations preserve
the imported cell order; filtering preserves an ordered subset.

| Native choice | Meaning |
| --- | --- |
| `counting_strategy=fragment` | Each fragment-body overlap contributes once to a region |
| `counting_strategy=insertion` | Tn5 endpoint incidence; both endpoints may contribute |
| `counting_strategy=paired-insertion` | Native paired-insertion counting, the default |
| `use_x` / single-end / native base values | Separate native aggregation meanings, recorded separately |

Do not label these representations interchangeably. The native matrix retains
its actual sparse count values and ordered feature names. Agent projects native
coordinate names into `var.chrom/start/end` for cross-tool use, without recounting
or changing feature order. Normalized accessibility and binary accessibility are
separate measurements.

Native import defaults to `min_num_fragments=200` and `sorted_by_barcode=true`.
Only ordinary task input-order adaptation changes the sorting choice; it does not
lower import retention or change counting. Explicit native-function calls preserve
omitted defaults. A supplied `sorted_by_barcode` value always prevails.
Native cell filtering defaults to `min_counts=1000`, `min_tsse=5`. The old Agent
selection used a different TSS calculation and threshold 4; these are not equivalent
methods. The native fragment-size distribution is library-wide, not the old
per-barcode nucleosome signal. No replacement per-cell nucleosome ratio is invented.
New import results record the retained barcode population and effective native
import threshold. They do not invent a pre-import barcode count or scan the source
again to obtain one. Later native filtering separately records input, selected and
output populations, including whether an independent selection was applied.
Native defaults remain in force unless the user requests alternatives. Explicit
minimum depth, QC thresholds and counting strategy are forwarded through their
native kwargs. An empty native selection remains empty; no threshold is lowered
to make a small fixture succeed. Native TSSe is distinct from the retired Agent
TSS algorithm. QC values remain in the downloadable native H5AD; compact summaries
expose recorded parameters and populations rather than every QC distribution.

Chromap stock 0.3.2 caps the tested BED duplicate support at 255. The historical
qualified patch widens that representation to uint64; the 300-read comparison
produces support 255 versus 300 with identical tested coordinates and corrected
barcode. The configured patched binary remains available. Ordinary builds are also
admitted with exact executable provenance, without an assertion of uncapped support.
Support-sensitive comparisons and duplicate diagnostics must respect this difference.
SnapATAC2 matrix counts do not use this support as a weight.

Native BAM processing has its own duplicate-flag and MAPQ filtering, pairing,
endpoints and shifts (+4/-5 by default). It excludes duplicate-flagged reads and
uses native MAPQ 30 by default. It is not retroactively labeled as historical Agent
BAM policy v1/v2. Refer to the report for actual small executions and limitations.

## Installation and deployment

Use the Agent environment for orchestration. Install its explicit native storage
readers with `pip install -r requirements/raw-native.txt`. Provision SnapATAC2 2.9.0
in an isolated compatible Python runtime with AnnData, NumPy, SciPy and hdf5plugin.
Set `AGENT_SNAPATAC2_PYTHON` to that runtime's Python, or use the existing resource
library runtime layout `runtimes/native-matrix-1/env/bin/python`.

Set `AGENT_CHROMAP_BIN` to an installed Chromap executable. The current local
installation is under `runtimes/chromap/chromap`; it still requires this explicit setting. `AGENT_CHROMAP_INDEX_ROOT` optionally
allows discovery of an exact reference-associated pre-existing index. New indexes
are ordinary explicit native publications. No installation or download occurs during
scientific execution. Native wrappers use CPU tools and contain no GPU/device-specific
implementation assumptions. Deployment must provision the selected runtime; installed
Signac/ArchR alternatives do not silently replace a selected SnapATAC2 call.

The native H5AD can use Blosc or Zstandard filters. Agent registers the installed
hdf5plugin decoder and retains bounded storage/link checks. Privacy downloads use
the shared exporter, copy exact safe values and axes, and use gzip when copying a
native plugin-compressed dataset. Accepted native files retain their original bytes.
An export is a derivative download, not a new scientific authority. Upload size and
streaming limits remain the configurable Web limits in the
[operating guide](operating-and-development.md).

## Integrity, reuse and history

Agent pins actual input/resource/implementation identities, records effective
native parameters, observes native completion, checks required output availability
and ordered axes, and publishes through the existing NativePublication owner.
Cancellation terminates the native process and never publishes a success manifest
for incomplete work. Accepted recovery and restart validate saved publications;
they do not repeat native science or require the original raw file to remain present.

The former seven raw preprocessing/QC/selection/counting tools are absent from the
executable Registry. Their custom algorithms and complete scientific reconstruction
verifiers are removed. Small read-only metadata and accepted-authority readers retain
old Sessions, figures, downloads and exact method identities. An unaccepted old
publication cannot acquire authority through those readers, and an interrupted
legacy raw execution cannot resume the retired scientific methods.

The previous [simplification report](../../../agent-data/acceptance/raw-intake-simplification/REPORT.md)
and historical milestone documents remain unchanged evidence of the earlier methods.
They do not define current raw execution. Biological validity, arbitrary assays,
full PBMC10k scale and unrestricted provider reliability are not qualified by this
bounded replacement.
