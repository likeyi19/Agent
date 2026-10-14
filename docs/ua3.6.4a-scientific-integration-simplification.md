# Scientific Integration Simplification — Work Unit A

2026-10-14. **Localized simplification completed:** ordinary EpiZoo admission
now accepts sparse Boolean storage of already-binary values, equivalent to the
previously accepted integer encoding. No scientific transformation changes.
UA3.6.4a/b/c/d remain **PARTIAL** at product acceptance.

## Baseline and scope

HEAD was exactly `10ff1844146c6371e38d54473be34401f1646b87`.
Initial `git status --short` contained only `?? evals/`; tracked files were clean.
The native EpiZoo checkout was clean at
`029cd631d0f9806a646c4a3b42ce10b958f2b67f` and remains so.
No historical record, operator configuration, dataset or checkpoint was edited.
The 1,274 pre-existing eval files retain their names, sizes and modification
times; this is a metadata comparison, not a fresh content-hash audit.
No downloads, provider calls, pretrained-checkpoint loads, commits or pushes occurred.

Historical evidence comes from the [interface audit](epizoo_interface.md),
[real human workflow](ua3.5.5-human-epizoo-user-workflow.md), and
[dialogue closeout](ua3.6.4a-scientific-dialogue-closeout.md). These establish the
previous real embedding → neighbors → Leiden → UMAP execution, parameter
continuation, historical reuse, delivery/recovery and current-result explanation.
They do not establish the deferred live cross-Revision comparison. Those records
were read, not rewritten or reenacted.

## Native requirements and ownership

The registered `epizoo_embed_cells` path is
`orchestration/registry.py` → `tools/analysis/epizoo_embedding.py` →
`tools/models/epizoo_cache.py` / `tools/models/epizoo.py` → native EpiZoo.
`tools/data/scatac.py:inspect_scATAC` reads H5AD structure in backed read-only
mode; the embedding adapter materializes sparse X for the native pipeline.
External files need no canonical-matrix adoption, matrix authority sidecar,
fragment history, FASTA/FAI or Agent reference-bundle manifest for this route.
The adoption tools' stricter int64/fragment-count contracts are not embedding
admission prerequisites.

The local native implementation was inspected directly:
`/home/likeyi/program/EpiZoo/epizoo/data/processing.py`, `data/datasets.py`,
`inference/embeddings.py`, `visualization/umap.py` and `metrics/cluster.py`.

| Requirement | Concrete reason and limit |
| --- | --- |
| Correct checkpoint and vocabulary | Native inference consumes ordered token IDs, not arbitrary peak coordinates. The current joint checkpoint has 1,514,484 tokens, 512-dimensional embeddings and 8,192 rank positions. Agent checks configuration/tensor shapes and strictly loads state; compatible shapes alone do not qualify biological identity. Interactive resources additionally pin the qualified checkpoint and auxiliaries. |
| Current raw feature layout | The qualified human profile uses 1,355,445 raw columns and 700,460 retained features; mouse uses 1,341,077 and 814,020. The frequency vector and positional filter belong to this profile. Native `filter_cCREs` fixes retained sizes. These are model/preprocessing requirements, not a universal definition of legitimate scATAC data. |
| Exact retained identities/order | `_validate_retained_feature_order` compares names at every filter position with the resource CSV. No aliases, coordinate conversion, missing-feature filling or silent reordering occurs. It does **not** verify every discarded feature name or reconstruct source genome assembly/alignment. |
| Raw count-like values | The owner requires finite, nonnegative, integer-like sparse values; integer-valued floating storage was already accepted. Already-normalized fractional values remain rejected. Binary 0/1 values were already admissible as numeric counts. Boolean storage carries those same values. |
| Informative, identifiable rows | Unique nonempty ordered cell IDs, positive row counts and at least one retained token remain required. Native primitives can mechanically process empty rows, but this does not qualify an empty-sentence embedding as an informative cell result. |
| Qualified preprocessing | Native TF-IDF runs before filtering, using the species frequency vector, cell number (8.2 million human / 12.5 million mouse), float32 and scale 10,000. Native sentence ranking applies offset 4, plus 700,460 for mouse. Dataset sampling/collation and extraction remain native. |
| Explicit execution profile | The public owner retains batch size 4, length 8,192, seeded random truncation (seed 0), zero workers, float32 model parameters and requested AMP. These are recorded implementation/reproducibility choices, not universal biological requirements. Dense input rejection protects memory; backed-file inspection followed by sparse loading already supports external H5AD. |

`application/local_resources.py:QualifiedEpiZooResource` binds species and model
auxiliary hashes, not an individual H5AD peak list. `select_epizoo_resource`
uses an explicit choice or unique declared-species default and preserves missing
or ambiguous prerequisites. A different dataset peak collection does not require
reidentifying a reusable genome; it does require demonstrating compatibility
with the chosen model. No new genome/resource abstraction is needed here.

## Findings

**SIMPLIFY — Boolean storage rejection.** Before the change,
`_validate_adata` rejected `np.bool_` because it is not a NumPy number subtype.
A two-cell simulated matrix encoded as int8 passed; its Boolean equivalent
raised `Sparse scATAC values must be real numeric counts.` Native
`compute_tfidf` explicitly casts both sparse encodings to float32, producing
exactly equal sparse values and `generate_cell_sentences` tokens. This proves
a representation-only restriction, not a need for new science. The fix admits
Boolean dtype without converting input counts to binary, inferring provenance,
relaxing feature compatibility or changing previously accepted numeric behavior.

**KEEP — Current model compatibility and value/identity checks.** A generic
peak H5AD is not automatically valid for this positional pretrained vocabulary.
Native `compute_tfidf` is a general utility; its ability to operate on other
matrices does not qualify their tokens for this checkpoint. No evidence here
justifies arbitrary peak overlap/projection or a reduced-vocabulary shortcut.

**KEEP — Mature science and authority boundaries.** Agent calls native EpiZoo
preprocessing/inference and Scanpy `pp.neighbors`, `tl.leiden`, `tl.umap`.
EpiZoo's convenience UMAP recomputes neighbors with different seed handling;
its Louvain helper optimizes against labels. Replacing the accepted separate
fixed-Leiden path with those helpers would change behavior. There is composition
overlap, not duplicated numerical algorithms to remove.
`orchestration/verifier.py:_verify_embedding` checks owner provenance and result
artifacts without rerunning inference. `neighbors_publication.py:_verify`
checks graph structure and exact source alignment; `describe` and accepted
authority preserve historical consumption without reopening original sources.
Resource hashing/stable-read checks protect consumed bytes and warm-cache
identity; they are not repeated scientific reconstruction.

**DEFER — UMAP's clustering prerequisite.** Native UMAP requires the neighbor
graph, not Leiden labels. Agent's narrower contract is real: `compute_cell_umap`
requires stage `clustering`; Registry admits only `clustered_analysis.v1`;
`_verify_umap` validates labels; `application/analysis_delivery.py` promises and
exports Leiden labels; `dialogue_evidence.py:_clustering_source` attributes
clustering through the exact accepted source. Thus this is a workflow restriction,
not a native mathematical necessity. Removing only the owner check would break
the shared artifact/delivery contract. General label-optional UMAP requires a
separate coordinated contract change; it is deferred under this unit's boundary
against shared-layer/semantic expansion. Existing accepted behavior is preserved.

**DEFER — Other feature vocabularies, adaptation and product acceptance.**
New peak-to-model mappings need independent scientific qualification, not relaxed
identity checks. Cross-species/post-training and unified input/UI work are outside
this unit. UA3.6.5 retains live cross-Revision comparisons and evidence-bound
answer qualification, plus the remaining real-data/live-provider gaps of all
four PARTIAL scenarios.

## Exact changes and verification

Only these files change:

- `src/agent/tools/models/epizoo.py`: allow Boolean alongside real numeric sparse
  storage in the dtype guard. This is a reproduced admission bug; no native
  EpiZoo algorithm, preprocessing order or scientific parameter changes.
- `tests/tools/models/test_epizoo.py`: ten parameterized cases covering Boolean
  CSR/CSC H5AD round-trip admission and nonmutation, invalid dimensions/rows/cell
  identities/feature order, the existing full mouse vocabulary, and exact native
  Boolean/int8 preprocessing parity.
- This closeout record.

Newly executed verification (exit 0 for both runs):

| Run | Selection and result |
| --- | --- |
| Before the fix | **155 passed, 3 skipped, 3 warnings, 30.32 s.** Existing EpiZoo wrapper/cache/embedding, embedding-analysis, neighbor-publication, H5AD inspection and human-resource-mapping modules; five selected processed-prior-output test functions (six cases) for separate-run execution, historical/current-source distinction, fresh-process terminal recovery, accepted same-plan recovery and legacy reads. |
| After the fix | **81 passed, 2 skipped, 2 warnings, 11.40 s.** Entire `tests/tools/models/test_epizoo.py` and `tests/tools/analysis/test_epizoo_embedding.py`, enabling `RUN_EPIZOO_INTEGRATION=1`; checkpoint-load and CUDA-parity gates remain disabled. All ten new cases pass, plus the existing native small-subset preprocessing test with stubbed extraction. |

Both used `/home/likeyi/anaconda3/envs/agent/bin/python -B -m pytest -q
-p no:cacheprovider`, `PYTHONPATH=src:.`, `PYTHONDONTWRITEBYTECODE=1` and isolated
temporary test directories. The baseline removed all `RUN_*` environment gates.
Exact baseline argv, logs and the preservation metadata comparison are retained
in `/tmp/agent-scientific-integration-a-DaKZxa/` (temporary local evidence).
The post-change command selected the two modules above with
`RUN_EPIZOO_CHECKPOINT_TEST=0 RUN_EPIZOO_PARITY_TEST=0` and basetemp
`/tmp/agent-scientific-integration-a-DaKZxa/changed-pytest-tmp`.
`git diff --check` passes. Warnings concern existing louvain deprecations and,
in the baseline, the unavailable TBB threading layer.

Scope is deliberately separated: simulated storage/native-primitive parity;
a binary test derivative of two existing mouse cells for full-vocabulary
validation; scripted-provider/synthetic-embedding software recovery checks;
and previously recorded real-human acceptance. No new biological qualification
or actual model inference is claimed. The full repository suite, large GPU
workflows, live providers and historical browser sessions were not rerun.

Public signatures, result schemas, recovery versions, authority profiles,
lineage and historical artifacts are unchanged. The baseline reuse/recovery
checks remain applicable because their code and numerical inputs did not change;
only the affected embedding-owner tests were repeated after the fix.
**Work Unit A can close with this localized simplification and explicit deferred
boundary. Work Unit B is the recommended next separately scoped unit; it has not
begun. This does not promote any UA3.6.4 product scenario from PARTIAL to PASS.**
