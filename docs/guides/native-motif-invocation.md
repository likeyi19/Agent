# Native motif function invocation

`run_motifmatchr` invokes the installed `motifmatchr::matchMotifs()` function on
genomic peaks from an ordinary H5AD. It uses the existing conversation, Planner,
Registry, execution, scientific authority, Revision and download mechanisms.
No R code or scripts can be submitted.

The bounded pilot passed genuine Firefox execution, six verified downloads and
restart retrieval using the configured Qwen profile. Research use is conditional
on this documented scope. Earlier provider failures, occasional unavailable prose
and the earlier overbroad missing-file question remain recorded in the acceptance report.
The final review also qualifies a non-JSON continuation, a count-free H5AD upload
and a corrected dataset-only missing-input question; wording can still require clarification.

Attach the H5AD in the normal Web composer and ask, for example:

> Run motifmatchr::matchMotifs() on these peaks. The assembly is GRCh38 and the
> coordinates use zero_based_half_open (zero-based half-open).
> Use p.cutoff=0.0001, bg='even', and out='matches'.
> Use compatible installed genome and motif resources and return the matches.

Use the actual assembly and coordinate convention of your data; the example
values describe the qualified PBMC input. The explicit convention token avoided
an observed model quotation error; the shared binder also supports the correctly
quoted natural phrase. Exact literal grounding remains mandatory.

Native arguments can be ordinary assignments such as the example above, or
natural language such as “set p cutoff to 0.0001, use background even, and return
the matches matrix.” The Interpreter selects reviewed member names and quotes
each value from the message. The existing parameter binder constructs the native
mapping; it does not evaluate R/Python expressions or invent omitted parameters.
Explicit JSON, for example `{"p.cutoff": 0.0001, "bg": "even"}`, remains supported.
Numeric, string and explicitly supplied JSON values retain their types, including
values the native package may reject. Ambiguous or conflicting declarations need
clarification. Source and parameter grounding remain strict, so an incorrectly
quoted provider response may also require clarification.

Missing coordinate conventions, assemblies or reference choices use ordinary
clarification. Required datasets and optional user resources are distinguished
in Registry offers. A request without a dataset can ask for that dataset alone;
compatible installed resources need not be uploaded. No manifest path, checksum
or R object is requested from the researcher.

## Scientific input and native scope

The H5AD supplies ordered `var` coordinates: chrom/start/end columns (including
explicitly selected alternative column names), or genomic feature identifiers.
Coordinate convention and assembly must be declared or present in source
metadata. Standard `uns.assembly`, `uns.coordinate_system` and unique `var.genome`
values are reused. Exact source bytes and qualified genome/motif resources remain
pinned. Contigs are not renamed and intervals are not clipped.

Only peak coordinates are used. Counts, a count-slot declaration, positive-count
peak filtering, clustering and chromVAR backgrounds/deviations are unnecessary.
Transformed or absent X does not disqualify valid genomic peaks. All supplied
peaks remain in their original order, including zero-count peaks. HDF5 storage
safety and bounded label exchange still apply.

The reviewed dispatch supports only `matchMotifs`. Supported kwargs are
`p.cutoff`, `bg` and `out`. Values are bounded JSON data; `bg` can also be a
numeric array or a named A/C/G/T frequency object. The scientific package owns
parameter validity. Native errors are retained without modifying values or
retrying. Input objects, genome paths, scripts and commands cannot be passed in
kwargs. Unspecified optional arguments are omitted from the R call so the
installed native method supplies its defaults. The original pilot published
`out="matches"`; STE-NFI-1G additionally supports `out="scores"` and `out="positions"`.

The existing ScientificResourceLibrary supplies the unique assembly-matched
genome and taxon-compatible motif collection, or an explicit resource selection.
The optional already-supported FASTA source port requires its actual assembly
and taxonomy. Missing installations require operator provisioning through that
library. The isolated STE-1 runtime is reused (motifmatchr 1.32.0); this operation
does not load chromVAR. No new runtime or resource system is installed.

Ordinary BED upload is currently routed to sequence-specific consumers and has
no general peak-source port. This pilot stays with H5AD rather than introducing
a second genomic input representation.

## Results and limits

For `out="matches"`, the result is a native boolean peak-by-motif matrix. Downloads include Matrix
Market and SciPy sparse NPZ representations, ordered peaks, native motif names
and accession/version annotations, metadata and a completion receipt. Peak
tables use zero-based half-open coordinates; R receives start + 1 and unchanged
end. Matrix rows and columns follow the supplied peaks and native motif order.
Privacy preparation applies to metadata and tables; private locator-like labels
may be omitted from delivered tables while ordered indices remain available.

Accepted reads, downloads and reopening do not rerun motif matching. Completion
and artifact integrity verification do not repeat the native algorithm. Native
matching establishes sequence matches, not enrichment, TF binding, expression,
regulatory causality or biological validation. No other analysis or figure is
automatically produced.

The [STE-NFI-0 acceptance report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-0/REPORT.md)
owns measured native parity, Firefox acceptance status, provider limitations and
the readiness decision. Implementation availability alone does not establish
ordinary researcher acceptance.

Native reference: [motifmatchr vignette](https://bioconductor.org/packages/release/bioc/vignettes/motifmatchr/inst/doc/motifmatchr.html).


STE-NFI-1G scores output preserves native `motifMatches`, `motifScores` and
`motifCounts` assays as separate matrices. Positions output preserves the native
GRangesList: `positions.tsv` contains motif index/name, chromosome, 1-based closed
start/end, width, strand and score. The original ordered peaks remain separate;
Agent does not invent a hit-to-peak association absent from the native result.
Both modes also deliver the exact `native-result.rds`, retaining native attributes.
RDS is an output download, not a newly admitted executable R input. Native binary
snapshots retain native labels; projected metadata/TSVs follow existing privacy rules.

`n_matches` counts positive peak/motif incidences for matrix modes. It is null for
positions; `n_positions` counts native genomic hits. Neither is a cell count or a
biological validation metric. There is no fabricated boolean output for positions.
The same native publication owner selects the exact mode-specific file closure;
accepted historical matches retain their original fields and profile.

Independent native R object equality and simulated acceptance/download/restart
checks pass. The initial genuine 1G Firefox/OpenAI positions request failed at
interpretation with an incomplete provider response, before science. The integrated
closeout subsequently qualifies genuine Firefox/official OpenAI positions execution:
2,000 GRCh38 peaks, 1,019 motifs and 684,091 native genomic hits. All six downloads
are checked: exact RDS bytes, unchanged TSV contents with export line endings, and
JSON retaining scientific fields while removing private paths/session details.
Initial erroneous coordinate questions are preserved. Their shared prerequisite
and parameter-continuation scope omission is corrected; the genuine continuation
executes once. A separate absent-file request asks for the H5AD dataset without a
Run. This is bounded delivery qualification, not TF binding or universal provider
qualification. See section 14 of the [1G report](../../../agent-data/acceptance/scientific-tool-ecosystem/ste-nfi-1g/REPORT.md).
