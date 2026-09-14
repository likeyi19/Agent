# M12 — Matrix-Level Input Support: preserved audit

Status: **AUDIT ONLY; implementation and projection semantics not approved.**
This preserves the preceding read-only audit of clean `main` at
`24facb31a300d65170173d87f649578983c403a7`. M11.8 retains its original
Biological Validation & Raw Preprocessing Closeout designation. No M12 code,
schema, tool, test or scientific transformation was implemented.

## Accepted architectural facts

The current `scatac-cell-by-ccre.v1` contract is more than an H5AD layout.
`scatac_matrix_contract.py`, `_cell_by_ccre_io.py`, `cell_by_ccre_verifier.py`
and their tests require exact fragment/selection/reference lineage; the frozen
`canonical-fragment-record-overlap-counts.v1` profile; full reference columns;
exact selected rows; canonical int64 CSR; qualified runtime; and independently
reconstructed counts and fragment diagnostics. Production human/hg38 and
mouse/mm10 vocabularies contain 1,355,445 and 1,341,077 columns respectively.
Tiny reference fixtures remain legal; matching shape does not authenticate a
production vocabulary. Zero rows/columns and empty selection are preserved.

External full-cCRE adoption is feasible, but compatibility checks cannot create
fragment/QC lineage, canonical-record counting history, or QC-selected cell
claims. The closed v1 manifest cannot truthfully describe such adoption unchanged.
A narrow versioned extension of the existing matrix artifact family is a design
candidate, not an approved schema or universal producer framework.

Reuse the existing matrix owner, bounded sparse IO, reference identities,
durable publication/receipt/recovery patterns, schema-2 scientific authority,
trusted RunStore loading, and application authority scope. Add only the necessary
reviewed producer/profile adapters and bounded report projections. Do not create
a second matrix verification framework or new authority hierarchy.
`qualify_legacy_authorities` applies to supported successful persisted Agent runs
with execution/receipt anchors; it is not an arbitrary external-H5AD import API.
External-fragment adoption provides the useful precedent: independently verify
source conservation, while retaining upstream history as declared/unreconstructed.

## Proposed minimal input and adoption

Start with one constrained H5AD representation, cells × regions in `.X`, exact
cell identifiers/order, and canonical `chrom:start-end` feature names. Bind
explicit source path/SHA, species/assembly, semantics, and reference path/SHA.
Exact adoption should require the complete ordered reference vocabulary; reject
partial, extra, reordered, and retained-only vocabularies initially. Preserve
external IDs without fabricating barcode namespaces or selection history.

Validate all sparse values/indices, dimensions, identity uniqueness and coordinate
bounds. Canonical CSR and checked int64 output can reuse existing mechanics;
exactly integer-valued finite floats can be losslessly converted, without rounding.
Reject malformed offsets, duplicate sparse entries, explicit zeros, negative,
fractional, nonfinite or overflowing values. Boolean input needs binary semantics.
M8 already distinguishes declared `fragment_counts`, `insertion_counts`, and
`binary_accessibility`; this vocabulary does not prove historical counting science.
Its duplicate-summing normalization is not a substitute for strict adoption.
The M11.5 logical digest hardcodes fragment-count semantics and must not be reused
unchanged for a different value meaning. Preserve existing v1 identities.

Independently verify source/output conservation within the matrix owner before
publication. Persist normal authority only after successful proof and durable
execution binding. Reuse that authority downstream; standalone/recovery retain
their strict behavior. No fabricated fragments, QC, selection or FRiP diagnostic.

## Peak/region projection remains a scientific decision

No accepted aggregated-peak-to-cCRE count rule was found in the inspected Agent
code, tests, milestone history, or local EpiZoo/EpiAgent preprocessing.
EpiAgent's `construct_cell_by_ccre_matrix` consumes fragment-intersection rows,
using supplied support or unit contributions. EpiZoo's `build_ccre_map` performs
cross-species liftOver/maximum-overlap index mapping; it does not authorize
redistributing aggregated peak counts.

Peak totals discard within-region locations. Even one peak overlapping one cCRE
does not establish equality unless coordinates and counting definitions match.
One-to-many copying multiplies counts; splitting requires assumptions. Many-to-one
summing can double-count shared fragments, even across disjoint source regions.
Partial-overlap fractions do not reveal signal distribution. Binary input loses
multiplicity; integer input does not distinguish cut sites from fragment counts.
Arbitrary aggregated peaks cannot generally reconstruct exact M11.5 counts.

Two initial options were identified: exact full-cCRE adoption only; or an explicit
weaker binary overlap-evidence transformation. The latter was recommended for
consideration in the audit but **is not approved for implementation**:

```text
Y[cell,cCRE] = 1 iff a source region with X[cell,region] > 0
              overlaps that cCRE by at least one base.
```

This would mean active-region overlap evidence, not localized observed fragment
counts or established cCRE accessibility. Multiple source hits combine with OR;
count magnitude is explicitly discarded; touching boundaries do not overlap.
Reject duplicate/malformed/out-of-reference coordinates, preserve input identities,
allow unsorted coordinates through indexed private projections, retain full zero
columns, and disclose unmapped evidence. Zero means no projected evidence, not
demonstrated biological inaccessibility. Profile, semantics, report wording and
downstream eligibility require a decision before implementation. Model feature
shape compatibility is not biological model-readiness.

## Future work boundary

A possible three-step sequence is exact adoption; one approved projection profile;
then thin Agent integration and real acceptance. Focus tests on conservation,
axis/CSR failures, hand-computed overlaps, mutation, lifecycle, and authority reuse.
Full regression belongs at final integration acceptance when justified, not every
edit. Likely integration points are the existing matrix contract/IO/owner,
`scientific_authority.py`, registry metadata, step verification and report facts.

Defer MEX/CSC/dense/layer adapters, format inference, partial-vocabulary repair,
aliases/liftOver, weighted/fractional/nearest-region projections, statistical cell
calling, doublets, FRiP, annotations, clustering and model execution. No new generic
adapter framework, planner heuristics, qualification subsystem or legacy migration
is justified by this audit. Implementation must first agree on truthful external
artifact semantics and whether any approximate projection is wanted at all.
