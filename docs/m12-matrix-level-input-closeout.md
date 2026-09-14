# M12 — Matrix-Level Input Support closeout

**M12 MATRIX-LEVEL INPUT SUPPORT CLOSED**

This documentation/evidence closeout starts from clean `main` at
`f6b221833542f5da412821915366d26aa6d487d2`. It changes no production behavior.
This record supersedes the open-status/proposal language in the historical
[M12 audit](m12-matrix-level-input-audit.md) and [M12.1 record](m12.1-exact-external-cell-by-ccre-adoption.md).
M11 remains closed.

## Supported: exact external canonical cell-by-cCRE adoption

M12.1's `adopt_scATAC_cell_by_ccre` is the production-supported matrix-level path:
constrained int64 CSR H5AD with the exact complete species-specific canonical
reference vocabulary. It retains truthful external declarations, independently
proves source/output conservation, and reuses the matrix owner, scientific
authority and normal durable lifecycle. It fabricates no fragment, QC or selection
lineage and grants no biological model-readiness claim. Its implementation and
[accepted input contract](m12.1-exact-external-cell-by-ccre-adoption.md) are unchanged.

The supported routes remain raw FASTQ/BAM/fragments → canonical cell-by-cCRE
through the existing explicit preprocessing stages, and already canonical external
cell-by-cCRE → exact adoption. Arbitrary matrix formats are not supported.

## Unsupported by design: generic ordinary cell-by-peak projection

Aggregated peaks discard within-region fragment locations, so their counts cannot
generally reconstruct canonical fragment-derived cCRE counts. EpiZoo consumes
ordered cCRE tokens: matrix magnitude affects TF-IDF ordering and rank embeddings
([Agent preprocessing](../src/agent/tools/models/epizoo.py)). Binary active-peak
overlap evidence was the only truthful general candidate identified for empirical
evaluation; it claims overlap evidence rather than localized target counts.

M12.2a reused preserved human/hg38 PBMC Next GEM run
`m117b-pbmc1k-nextgem-r1:run` and selected 32 cells deterministically without labels.
C was an exact subset of its accepted canonical count matrix; B binarized C;
P counted the same fragments directly into one existing PBMC consensus vocabulary
and projected active peaks by positive-base overlap and OR. C→B isolated count/rank
loss; B→P exposed additional geometry-induced support, rank and sampling changes.
The deployed preprocessing/settings were retained, with two additional canonical
seed controls through the existing lower-level API. No raw pipeline was rerun.

The vocabulary had 412,490 unique 500 bp peaks from a separate PBMC collection.
This tests a real alternative vocabulary, not same-dataset peak calling or extreme
broad peaks. All model input IDs matched the recorded tokenizer outputs.

| Bounded finding | Result |
| --- | --- |
| C/B canonical support | Identical; ranking changed |
| B/P full-support Jaccard | Median 0.743; range 0.242–0.763 |
| P identities absent from B | Median 17.64% per cell |
| B identities lost in P | Median 11.41%; range 8.84–74.10% |
| Normalized embedding L2, C/B vs B/P | Medians 0.101 vs 0.255 |
| C/P displacement relative to each long cell's mean canonical seed-control displacement | Median 3.62×; range 1.85–21.94×, 16 cells |
| Severe support-loss cases | 74.10% and 50.25%; C/P L2 1.127 and 0.942 |
| Canonical between-cell normalized L2 | Median 0.955 across 496 pairs |
| Projected embedding nearest to its own canonical embedding | 23/32 cells; a distance diagnostic, not annotation accuracy |

High median C/P cosine (0.968) alone was insufficient evidence of fidelity.
No arbitrary acceptance threshold or post hoc cell exclusion was introduced.
The tested PBMC case does not justify exposing a general projection capability;
it does not establish that every possible vocabulary must fail.

**KEEP CELL-BY-PEAK PROJECTION UNSUPPORTED.** This is the final scientific/product
boundary, not unfinished implementation. Binary overlap projection is not approved;
no production projected-artifact profile is to be created. There is no automatic
fallback using binary OR, largest overlap, fractional weights or count copying.
The experiment will not be expanded merely to seek a favorable result.

## Durable evidence and closeout checks

The completed evidence is retained outside Git at
`/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1`:
[report](/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1/REPORT.md),
[per-cell results](/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1/per_cell.csv),
[original audit manifest](/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1/audit_manifest.json),
and [retention index](/home/likeyi/program/agent-acceptance/m12.2a/pbmc-paired-32-r1/retention-index.json).
Preparation/resource/checkpoint identities, exact cells, geometry/token diagnostics,
five embedding arrays, token sequences, compact peak counts/mapping, the peak
vocabulary, experimental scripts and logs are retained byte-for-byte. Seven large
reproducible matrix/BED/incidence intermediates are omitted; their original hashes
and retention rationale remain indexed. Original reports/scripts retain historical
temporary paths; the retention index documents their durable location.

Closeout checks cover source/archive SHA-256 equality, file presence, documentation
links, the documentation-only change inventory and `git diff --check`. No experiment,
inference, production tests, full regression or scientific reconstruction is rerun.
M11 artifacts, production source, tests, registry, Planner, Application, authority
and artifact contracts are unchanged. No commit or push is part of this closeout.

With raw preprocessing and matrix-level input support closed, the next major
user-facing capability should return to **Direct Cell-Type Annotation**. Begin
with one focused capability/interface audit: define the direct input-to-annotation
contract, assess reuse of existing embedding and label-transfer tools, and identify
the minimum reference/model requirements and unsupported cases. Do not implement
it during M12 closeout.
