# Native motif-associated accessibility

`compute_motif_accessibility` uses the original chromVAR and motifmatchr R packages
to calculate per-cell motif accessibility deviations with GC/accessibility-matched
background peaks. It is independent of EpiZoo, PCA, clustering, annotation and
peak-set motif enrichment. Ask for TF-associated motif accessibility and supply
an ordinary cell-by-peak AnnData file; optional existing groups support descriptive
summaries. No prior Agent analysis or dataset certification is required.

## Scientific inputs

Dense, CSR and CSC count matrices are supported in `X` (default) or
an exact layer name such as `raw_counts` (`layers/raw_counts` also works). Declare the selected slot's actual measurement: raw peak
counts, fragment-overlap counts, insertion/cut-site counts or binary accessibility.
Cell Ranger ATAC cut-site counts remain insertion counts. Finite nonnegative
integers establish numeric eligibility, not upstream biology. Normalized,
log-transformed, TF-IDF and motif deviation matrices are not count inputs.
A separately selected layer needs its own measurement declaration.

Peaks may use `var` chrom/start/end columns (or explicitly selected columns), or
`chrom:start-end` / `chrom_start_end` feature IDs. Establish the actual coordinate
convention, zero-based half-open or one-based closed. Standard `uns.assembly`,
`uns.coordinate_system`, `uns.matrix_semantics` and a unique `var.genome` value
are reused by the owner. Recorded X semantics do not apply to another layer.
Read-only source observations expose missing declarations to conversation without
creating execution authority. Once this method and an ordinary source are bound,
the Registry calls its scientific owner to identify unresolved declarations before
planning. A model's empty missing-parameter list cannot qualify transformed X.
The same owner declaration logic governs execution; numeric and resource checks
remain there. The existing typed parameter question preserves the objective,
source and prior choices without a Run. A natural answer supplies the selected
layer and actual measurement; no layer is chosen from its name. Planned upstream
matrices remain subject to their compiler and execution checks. Other free-form
metadata is not silently converted into a scientific declaration.

A matching genome and taxon-appropriate motif collection are required. Assembly
names alone do not establish compatible contig dictionaries: UCSC hg38 scaffolds
and the official GRCh38-2020-A reference use different names. Select compatible
reference bytes; the operation never renames, clips or drops unmatched intervals.
A task-provided FASTA can enter through the existing `genome` source role with
assembly and species taxonomy ID; its index is generated privately, without
changing the upload. Otherwise the unique exact-assembly genome in the configured
ScientificResourceLibrary is used. When several entries match, `genome_resource`
selects the exact installed label or ID through the ordinary scientific parameter
interface. Its declared assembly must still match; matching bytes do not silently
alias `hg38` and `GRCh38`. Operators may install a qualified reference's explicit
assembly declaration and exact genome in place, preserving all existing entries.

Native `filterPeaks(min_fragments_per_peak=1, non_overlapping=FALSE)` excludes
zero-total peaks and records their original positions. Overlap pruning and sample
QC are not silently applied. `filterSamples` would require total-read observations
that a count matrix need not contain. Zero-total cells retain native undefined
values. Native singular GC/accessibility backgrounds fail explicitly, without
adding artificial jitter or substituting another method. All-N retained reference
sequences have undefined GC and cannot support this native background.

The sparse exchange has a 2 GiB matrix allocation budget; dense inputs are read
in bounded chunks. This is a runtime bound, not a biological eligibility rule.
Counts beyond exact native double precision are rejected rather than rounded.
Ordinary shared storage-safety and upload quotas still apply.

## Results and interpretation

The result retains original ordered cells and versioned motif accessions:

- `motif-accessibility.h5ad`: X is bias-corrected deviation; `layers/z_scores`
  contains the separate native background-standardized z-score matrix.
- `peaks.tsv`, `cells.tsv`, `motifs.tsv`: original-to-retained mappings, exclusions,
  GC, sequence hashes, motif accessions/names/families and native match fractions.
- `motif-matches.npz`: SciPy sparse retained-peak × motif boolean matrix.
- `background-peaks.npy`: zero-based indices into the retained peak order;
  `peaks.tsv` maps these to original features. Columns are background iterations.
- `motif-summary.tsv`, `group-summary.tsv`: descriptive z-score variability and
  supplied-group means, with finite-cell counts; no replicate-level significance.
- `patterns.png`, `metadata.json`, `native-receipt.json`: descriptive figure,
  source/resource identities, parameters and actual native software completion.

The figure displays the 20 most variable motifs, retains all cells and clips only
the color scale at ±3. Downloads retain full values. Optional group labels order
cells, without establishing cell types. NaN remains undefined, never zero.
Related TFs may have similar motifs; these results do not identify unique TF
expression, physical binding, regulatory causality or experimental truth.
Peak-set motif enrichment is a different analysis and is not integrated here.

Downloads use accepted artifact authority and the existing private snapshot/
redaction path. Reading, downloading, restarting or recovering a completed result
does not repeat motif scanning or chromVAR. The initial STE-1 full acceptance
record has a known JSON precision defect: its recorded zero motif cutoff is not
the actual native cutoff. Current evidence explains this discrepancy without
changing historical values or inferring corrections from current defaults. The
[closeout report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-1/closeout/REPORT.md)
records the final interaction validation status. Interrupted native work without a
completed publication does not gain retry authority.

## Optional operator installation

The tested isolated runtime is R 4.5.3, chromVAR 1.32.0, motifmatchr 1.32.0 and
TFBSTools 1.48.0. Complete Linux-64 package pins and actual qualification are in the
[STE-1 report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-1/REPORT.md).
Do not upgrade the qualified edgeR/MAESTRO runtimes or Agent's Python environment.
The native runner checks the tested chromVAR/motifmatchr versions. A later runtime
requires its own native qualification.

The default executable is `runtimes/chromvar/env/bin/Rscript` beside the resource
library directory; operators may explicitly set `AGENT_CHROMVAR_RSCRIPT`.
Unrelated Agent startup and tools do not invoke R. Scientific computation uses a
single native BiocParallel serial worker with recorded seed (default 20261030),
50 background iterations, and motifmatchr `p.cutoff=5e-5`, `bg='subject'`.
These are native method settings, not inferred researcher instructions.

Install a `motif_collection` through `ScientificResourceLibrary.install`, using
schema `jaspar-motif-collection.v1`, exact `pfm` and `metadata` path/SHA identities,
release, CORE collection, taxonomic group, declared applicable species taxonomy
IDs, license and citation. These describe reusable reference applicability, not
approved experimental datasets. Installation checks accession/version membership
and position-frequency validity; exact file integrity owns later reuse.
The current collection is JASPAR 2026 CORE vertebrates, 1,019 nonredundant matrices.
Genome and motif entries are independent. No network database lookup occurs during
scientific execution.

chromVAR is MIT licensed, motifmatchr GPL-3 with its bundled notice, and TFBSTools
GPL-2. JASPAR data require CC BY 4.0 attribution. Keep the upstream notices and
reference usage conditions; the integration does not redistribute a bundled R
runtime or genome.

Method and resource references: [chromVAR](https://doi.org/10.1038/nmeth.4401),
[native deviations](https://greenleaflab.github.io/chromVAR/articles/Articles/Deviations.html),
[motifmatchr](https://bioconductor.org/packages/motifmatchr/),
[JASPAR 2026](https://doi.org/10.1093/nar/gkaf1209).
