# Native research objects and sequence-bin enrichment

STE-NFI-1H adds focused ArchR and monaLisa functions through the existing native
function interface. The Registry, fixed Interpreter, Planner, Compiler, Executor,
scientific authority, Session/Revision and download owners remain shared.
The [acceptance report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1h/REPORT.md)
separates native parity, scripted integration and genuine Firefox/OpenAI evidence.

## ArchR 1.0.3

`run_archr(function_name, input_path, input_sha256, output_dir, ...)` supports:

| Function | Actual input | Native result |
| --- | --- | --- |
| `createArrowFiles` | Sorted five-column fragments, exact sample names and declared genome/coordinate/shift semantics | Genuine ArrowFiles, including native default gene scores |
| `ArchRProject` | Exact accepted Arrow publication | Saved native project with copied companion ArrowFiles |
| `getMatrixFromProject` | Exact accepted project publication | Existing native matrix as SummarizedExperiment, sparse Matrix Market and ordered axes |
| `getAvailableMatrices` | Exact accepted project publication | Native available-matrix names |

The qualified entrance is human hg38/GRCh38, zero-based half-open fragments whose
endpoints are already Tn5 adjusted (`fragment_endpoints=tn5_adjusted`).
The explicit `unadjusted` declaration is representable but refused by this
entrance; the adapter never adds a shift. An assembly name does not prove experimental
processing history. The existing ScientificResourceLibrary supplies the uniquely
qualified GRCh38 genome for contig/bounds checks. ArchR's versioned bundled Hg38
gene/genome annotations supply native TSS and gene-score semantics; the exact
installed annotation bytes are pinned alongside the implementation. No annotation
is fabricated. No automatic package installation occurs during execution.

`sample_name` maps to the first native `sampleNames` entry. Optional `second_path`
and `second_sample_name` add a distinct second sample. These are source-specific
sample identities, not inferred biological replicates. A private BGZF/index copy
preserves every decoded record; the adapter does not sort, shift, rename barcodes,
collapse duplicates or select cells. ArchR performs its own documented filtering.

One or two independently identified fragment files are supported per creation.
This is an Agent integration limit, not a native ArchR limit: upstream accepts
ordered file and sample-name vectors. Current source binding resolves fixed
`dataset` and `second` ports; multi-attachment capture does not construct a
variable ordered file/sample collection. The existing collection mechanism owns
raw/FASTQ intake, not independent ArchR samples. Removing the limit would require
broader source-binding, per-sample declaration and managed-consumption changes;
adding more fixed slots would only move the cap. Final closeout retains the
qualified boundary without concatenating samples or introducing a manifest format.

`kwargs` preserves reviewed native parameters. Creation admits minTSS, minFrags,
maxFrags, minFragSize, maxFragSize, nucLength, promoterRegion, excludeChr, nChunk,
addTileMat, addGeneScoreMat, threads, subThreading and verbose. Project construction
admits showLogo and threads. Matrix retrieval admits useMatrix, useSeqnames,
excludeChr, binarize, threads and verbose. Inspection has no kwargs.
Omitted parameters retain native defaults. Thread requests above 16 and nChunk
above 100 exceed execution limits; an excessive omitted thread default is refused,
not silently changed. A fresh immutable destination controls output paths and
copying; users cannot replace gene annotations, evaluate R expressions or overwrite
accepted Arrows through kwargs.

`createArrowFiles` normally computes GeneScoreMatrix itself. Project construction
and matrix retrieval do not call `addGeneScoreMatrix`. A missing matrix produces
the native failure. Gene scores estimate accessibility-related activity; they are
not measured RNA expression. `addPeak2GeneLinks` remains unsupported: a genuine
project with the appropriate peak, reduced-dimension and expression/integration
matrix prerequisites is a separate scientific integration task. Cicero is never
substituted for an explicitly requested ArchR function.

The snapshot ZIP retains the native Arrow or saved project structure. Saved
projects reopen with `ArchR::loadArchRProject` after extracting the ZIP; the native
loader resolves its companion Arrow paths. Native binary objects may retain
package provenance/path fields. Downloading an object does not authorize executing
an uploaded copy. Inside Agent, reopening requires the exact publication and
checksum already accepted by RunStore, with matching version/profile and complete
artifact integrity. Arbitrary RDS, ZIP and H5AD inputs are not native projects.
This is a focused project snapshot, not a universal R serializer or another object
database. Independent native publications and their displayed scalar summaries may
retain exact historical parents without making every parent a current result.
Their accepted bindings and artifact integrity remain checked; coupled scientific
owners still require the matching active source. Gene-score exports keep native feature-by-cell orientation and ordered
feature/cell metadata. Creation's private QC diagnostics are not presented as
downloadable figures; no automatic PNG rendering is advertised.

Example research request:

> Build an ArchR project from these human GRCh38 fragments and retrieve gene
> scores. They are zero-based half-open, already Tn5 adjusted. Name this sample
> PBMC. For Arrow creation use minTSS=0, minFrags=1000, addTileMat=false,
> nChunk=1 and threads=2. Keep the other native defaults.

Those explicit thresholds are an example, not Agent's scientific defaults or
a general QC recommendation. Later ask to inspect the accepted project or retrieve
its existing matrix; no fragment reprocessing is needed.

## monaLisa 1.16.0

`run_monalisa(function_name="calcBinnedMotifEnrR", ...)` consumes ordered genomic
regions from ordinary H5AD `var`, qualified matching reference sequences and a
taxon-compatible motif collection from ScientificResourceLibrary. Counts and cell
groups are not its scientific inputs. Supply either an exact `bin_key` in `var`,
or a two-column TSV `feature_id`/`bin` in exactly the same region order. Equal row
counts do not establish correspondence. Categorical field levels are preserved;
plain labels use explicitly sorted string levels. The adapter does not invent bins,
differential-accessibility results, a genome or a motif collection.

Real DNAStringSet sequences and accession-keyed PWMatrixList motifs feed the native
function. JASPAR PFMs use native TFBSTools `toPWM`; motif accessions and TF display
names remain separate. Reviewed kwargs are background, test, maxFracN,
maxKmerSize, min.score, matchMethod, GCbreaks, pseudocount.log2enr,
p.adjust.method and verbose. Genome-background sampling is unsupported and never
replaced by another background. Native sequence exclusions, undefined values and
ordinary scientific errors remain upstream behavior. Input execution limits are
100,000 sequences, 100 million bases, 128 bin levels and maxKmerSize at most four.

The exact native SummarizedExperiment is downloadable, along with all seven
motif-by-bin assays in a table, sequences, bins, motif identities and provenance.
`negLog10Padj` is negative log10 adjusted P, not a probability; `log2enr` is the
native enrichment statistic with its recorded pseudocount/background assumptions.
The R object preserves native row/column metadata and weighting information. It
is output only, never an uploaded executable input. No native figure is generated.
Enrichment is distinct from motif matching, chromVAR deviations, TF footprinting,
TF binding, gene expression and regulatory topic models.

Example:

> Run monaLisa calcBinnedMotifEnrR on this GRCh38 peak H5AD using var field
> accessibility_bin. Coordinates are zero-based half-open. Use the unique
> installed matching genome and human motif collection, leaving native kwargs
> omitted. Give me the enrichment statistics and native result download.

## pycisTopic and other candidates

pycisTopic remains uninstalled and unregistered. Its current upstream
[license](https://github.com/aertslab/pycisTopic/blob/main/LICENSE.txt) limits the
grant to internal academic use and restricts transfer. That grant does not establish
authorization for general hosted researcher access, even when a service is
noncommercial. Commercial use and redistribution need their own authorization.
The file names SCENIC+ while pycisTopic metadata expressly points to that file;
the inconsistency is not treated as permission. No restrictions were bypassed.
Topic modeling and recoverable CistopicObject/LDA state remain missing capabilities.
monaLisa does not close that gap. scvi-tools, SCENIC+, TOBIAS and other mature
packages remain equal-status, demand-driven candidates, with their actual object,
measurement, dependency and license requirements.

## Reproducible runtime

The isolated runtime is `agent-data/resources/runtimes/archr-1/env`. Existing
environments and production configuration are unchanged. `AGENT_ARCHR_RSCRIPT`
and `AGENT_MONALISA_RSCRIPT` select an operator-provisioned equivalent runtime.
The [Conda lock](../../requirements/native/archr-1.0.3-monalisa-1.16.0-linux-64.txt)
pins the Linux dependencies including monaLisa. ArchR uses upstream commit
`6feec354ad6c8052ddbc4626a2ca2d858ed465bf` (1.0.3), installed without source changes.
The lock's comments also pin the additional ArchR, chromVARmotifs and presto
source commits, retrieval URLs and archive SHA-256 hashes; the Conda package list
alone does not install them. The retained archives live in the runtime's
`sources/` directory, outside Git. Verify those hashes before installing into a
new environment; do not fall back to an unpinned branch if retrieval differs.
After creating and activating the environment from the lock, use its `R`:

```bash
R CMD INSTALL /path/to/verified/chromVARmotifs.tar.gz
R CMD INSTALL /path/to/verified/presto.tar.gz
R CMD INSTALL /path/to/verified/ArchR.tar.gz
Rscript --vanilla -e 'stopifnot(as.character(packageVersion("ArchR")) == "1.0.3", as.character(packageVersion("monaLisa")) == "1.16.0"); library(ArchR); addArchRGenome("hg38", install=FALSE)'
```

The locked BSgenome dependency and bundled ArchR annotations must be present;
the matching ScientificResourceLibrary genome and motif resources remain separate
operator prerequisites. Runtime recreation is qualified for the recorded Linux
environment, not every platform or future upstream archive service.
The report's `audit/source-archives.json` retains the original acquisition record.
Original license notices remain in the retained source archives and installed
packages where installed by upstream.
ArchR declares MIT; monaLisa declares GPL >=3. The complete dependency/version/
license metadata is recorded externally; this is not a blanket redistribution or
biological qualification. Scientific references retain their existing licenses.
ArchR redistribution must retain its MIT copyright/permission notice. Conveying
monaLisa or presto requires compliance with the applicable GPL terms, including
corresponding-source obligations; this runtime record does not grant additional
rights to bundled references or clear a particular hosted deployment.

The focused [research-object tests](../../tests/native_functions/test_research_objects.py)
exercise dispatch, exact inputs, safe archive authority and source-role admission.
Independent native and real browser acceptance helpers/evidence live under the
milestone directory. They are not production execution entry points.
