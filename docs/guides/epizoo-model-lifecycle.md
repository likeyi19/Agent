# Target-species EpiZoo model lifecycle

SWC-3 separates native parameter transfer, target inference and post-training.
The existing adaptation owner, ScientificResourceLibrary, Registry, accepted
results and download service remain responsible for those operations. EpiZoo
implements parameter transfer, sequence embeddings, preprocessing, cell
inference and training. No PCA substitute or independent mapping engine is used.

## Researcher interaction

Supply a sparse H5AD with exact genomic-region identifiers and declare the
species, assembly and actual X measurement (fragment counts or binary
accessibility). Ask to transfer EpiZoo without training, optionally followed by
zero-shot embedding, Leiden and UMAP. Agent prepares the exact ordered target
reference and resolves installed EpiZoo/SEAM and genome resources. Missing
experimental declarations remain questions. An installed genome cannot establish
which assembly an experiment used.

Omit the transfer strategy for automatic assessment. Explicit `de_novo` needs no
chain. Explicit `mapped_reference` names a qualified human/hg38 or mouse/mm10
source and uses its actual usable correspondence even below automatic preference
criteria. A missing compatible chain remains a resource prerequisite; it does
not silently change that request to de novo. A private supplied chain remains
private. Researchers do not prepare indexed BED files or internal specifications.

An accepted initialized model can supply native EpiZooX zero-shot embeddings.
Neighbors, Leiden and UMAP use the ordinary shared tools. Later ask to train the
initialized model on the compatible data, then recompute embeddings from that
trained checkpoint. Model selection follows the active accepted research branch
and its explicit dependencies. Historical model choices remain available;
unresolved alternatives require clarification. A browser historical preview
does not activate a different scientific result.

## Mapping evidence and selection

Native `epizoo.data.ccre.build_ccre_map()` runs UCSC liftOver and bedtools overlap
against the **retained checkpoint vocabulary**, preserving original target and
retained-source indices. For each target it selects the eligible reference with
the largest overlap, retaining native stable tie handling. SWC-3 adds an optional
observation callback before native temporary files are removed; the mapping
algorithm and its numerical settings are unchanged. The same returned mapping
initializes the model. Verification checks persisted mapping observations and
integrity without executing liftOver again.

`epizoo-transfer-preference.v1` uses:

- Usable mapping fraction = mapped target features / all target features.
- Native minimum individual overlap: one base pair.
- Native liftOver minimum match: its pinned default, 0.95. This is a different
  denominator and purpose from usable mapping fraction.
- Automatic mapped preference: usable mapping fraction at least 0.20 and median
  selected overlap / lifted target interval length at least 0.50.
- If both retained references qualify, choose larger usable fraction, then larger
  median overlap fraction. An exact tie uses the resource identity for a stable
  result, without a species-name priority.
- Otherwise choose applicable de novo. Distinguish evaluated low correspondence
  from absence of a qualified mapping assessment. Never invent zero coverage for
  an assessment that was not attempted.

Both strategies reuse pretrained EpiZoo parameters. Mapped transfer additionally
uses correspondence to inherit retained reference cCRE identity and decoder rows;
de novo does not imply training the whole model from scratch.

These are provisional, reproducible operational preferences, not experimentally
established biological performance thresholds. The coverage criterion requires
more than isolated matches; the overlap criterion prevents predominantly marginal
intersections from driving automatic mapped preference. Neither establishes
orthology or superior cell embeddings. Explicit mapped requests with usable
correspondence are not vetoed by these thresholds. An empty correspondence cannot
fulfil mapped inheritance.

Evidence records total/lifted/usable target counts, retained reference size,
coverage, selected overlap base-pair and interval-fraction summaries, multiple
eligible overlaps, tied best overlaps, repeated reference assignments, exact
chain bytes, tool identities and effective mapping settings. The initialized
publication retains those records with its transfer decision.

## Resources and acquisition

`epizoo_transfer` entries in ScientificResourceLibrary bind qualified source and
SEAM bundles, exact human/mouse references and mapping executables.
`liftover_chain` entries bind the input/destination species and assemblies, chain
bytes, qualified FAI bytes, source and usage basis. Qualification checks header
coordinates, conversion direction, contig lengths and complete block totals.
Task features and mapping results are outside the shared chain identity.

Discovery checks the library first. Bounded acquisition reads the exact UCSC
assembly directory and follows an actually listed exact-pair chain link. It
checks available storage, enforces download size bounds and per-request timeouts and verifies a
published MD5 when available, while recording SHA-256 and acquisition provenance.
A download is not itself scientific qualification. No assembly substitution,
chain renaming reversal or synthesized alignment is supported.

Downloads made while preparing a Session stay in its private preparation area.
Explicit operator installation enables shared reuse through the existing library;
private uploads are not automatically promoted. UCSC chain licensing and
commercial-use conditions apply ([UCSC licensing](https://genome.ucsc.edu/store.html)).
Actual chain availability is assembly-specific; see the
[danRer11 release directory](https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/liftOver/).

## Initialization and training

New lifecycle publications use `epizoo-target-model.v2`. Initialization performs
strict native pretrained transfer with SEAM embeddings and records exactly zero
attempts and optimizer updates. Computing the input-corpus document-frequency
vector or extracting DNA/SEAM embeddings is not post-training. SR/CCA requirements
for both accessible and inaccessible features per training cell do not gate
initialization.

`posttrain_epizoo_model` consumes an exact accepted initialized checkpoint, the
same ordered feature/reference identity and compatible measurement semantics.
It does not transfer source weights or run SEAM again. Native EpiZooXPostTrainer
owns SR/CCA losses, AdamW, AMP and scheduler execution. Each publication pins its
parent, data, effective configuration, attempted/applied/skipped steps, finite
loss observations and resulting checkpoint. Warm-starting weights uses a new
optimizer and scheduler; it is not exact optimizer-state resume.

The new operational default is 1,000 attempts, learning rate 5e-5 and batch 1.
Warmup is 10% of attempts, capped at 1,000; logging and final saving fit the
requested schedule. This takes the native tutorial's bounded 1,000-step example
as an operational starting scale, not a convergence prescription. The native
class's 500,000-step default is not silently launched for a new casual request.
Explicit positive step/learning-rate overrides and qualified batches 1–4 remain
available. An explicit larger schedule is the researcher's compute request;
GPU state and disk lower bounds are checked, with no silent reduction or fallback.
Activation memory and runtime are additional, so capacity checks are not a
promise that every vocabulary will fit. Interrupted jobs do not gain automatic
retry or optimizer-resume authority.

Accepted v1 adapted checkpoints retain their original training semantics and
readers. Old native-v2 schedules are not reinterpreted. A single explicit
before/after native source compatibility pair allows historical inference across
the additive mapping-observation callback; arbitrary native code changes,
missing modules or altered checkpoint bytes remain rejected.

## Interpretation and downloads

Model results distinguish initialization and post-training, source and parent
identities, actual strategy, observed mapping evidence and zero/applied training
updates. New model downloads include checkpoint, configuration, exact ordered
feature vocabulary and document-frequency vector. Native EpiZoo code and the
recorded numerical environment remain necessary for independent reuse; a
checkpoint download is not exported Agent execution authority.

Native target inference preserves exact cell identities and records any explicit
lossless feature permutation into the accepted vocabulary. Missing/extra features,
incompatible measurements or assembly conflicts are rejected. Both zero-shot and
post-trained results remain available. Comparisons require compatible cell axes
and scientific inputs. Independent UMAP layouts do not establish biological
improvement, and finite losses or successful execution do not establish
convergence or cell-type accuracy.

The consolidated [SWC-3 report](../../../agent-data/acceptance/scientific-workflow-completion/swc-3/REPORT.md)
records the actual validation scope, failures and remaining limitations.
The [bounded finalization](../../../agent-data/acceptance/scientific-workflow-completion/swc-3/finalization/REPORT.md)
closes under Outcome B: scientific completion is retained, with live explanations
and short continuations still unqualified after empty provider output-limit responses.
Established declarations can be retained from exact accepted context without user
repetition; different or ambiguous inputs still require their own valid bindings.
