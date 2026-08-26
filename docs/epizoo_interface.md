# EpiZoo cell-embedding interface audit

Audit date: 2026-08-26

Scope: the first Agent milestone only—raw scATAC-seq `AnnData` to EpiZoo cell embeddings. This is a read-only audit of the EpiZoo/EpiAgent backends and checkpoint; it is not an implementation.

Local sources of truth:

- `/home/likeyi/program/EpiZoo/examples/01_extract_cell_embeddings.ipynb`
- `/home/likeyi/program/EpiZoo/epizoo/data/processing.py`
- `/home/likeyi/program/EpiZoo/epizoo/data/datasets.py`
- `/home/likeyi/program/EpiZoo/epizoo/inference/embeddings.py`
- `/home/likeyi/program/EpiZoo/epizoo/inference/utils.py`
- `/home/likeyi/program/EpiZoo/epizoo/models/epizoo.py`
- `/home/likeyi/program/EpiZoo/epizoo/models/moe_transformer.py` (needed to interpret checkpoint keys)
- `/home/likeyi/program/EpiZoo/examples/06_compute_loa_score.ipynb` (human preprocessing parameters)
- `/home/likeyi/program/EpiAgent/data/cCRE.bed` (historical human raw-feature order check only)
- `/home/likeyi/program/model_checkpoints/EpiZoo/pretrained_EpiZoo.pth`

## 1. Validated end-to-end data flow

The validated order is important: TF-IDF is computed in the full raw species feature space, and filtering happens afterward.

| Step | Existing API | Exact input | Exact output | Important parameters and assumptions | Sparse? | Cell order? |
|---|---|---|---|---|---|---|
| Load raw h5ad | Tutorial uses `scanpy.read_h5ad(path)` | Cells by raw species cCREs in `.X`; feature positions must match the species reference order | In-memory `AnnData` | The mouse tutorial file is `2000 x 1,341,077`, with CSR `.X` | Yes for the audited mouse h5ad | Yes |
| TF-IDF | `epizoo.data.processing.compute_tfidf()` | Raw `AnnData`; full-length species document-frequency vector | A copy of the input `AnnData`; by default TF-IDF CSR replaces `.X` | `cell_number` is species-specific; `scale_factor=10_000`, `dtype=float32`, `store="X"`. Formula: `TF=count/row_sum`; `IDF=log(1 + cell_number/(df+1))`; output is `TF*IDF*10_000`. Zero-sum rows remain zero | Yes when `.X` is a SciPy sparse matrix; output is forced to CSR | Yes |
| cCRE filtering | `epizoo.data.processing.filter_cCREs()` | TF-IDF `AnnData`, positional integer `filter_idx`, species ID | `adata[:, filter_idx].copy()` with retained cCREs | Must be after TF-IDF because the supplied `df` is in the raw feature space. Current function checks only species and retained-list length, not raw dimension/order/index validity | Yes when input is sparse | Yes; feature order becomes `filter_idx` order |
| Cell sentences | `epizoo.data.processing.generate_cell_sentences()` | Filtered TF-IDF `AnnData` | Another copy; per-cell Python list in `.obs["cell_indices"]`; scalar species in `.obs["species"]` | For each row, nonzero local column indices are sorted by descending TF-IDF, then offset. Default `base_offset=4`; mouse additionally uses `species_offset=700_460` | Sparse row path uses CSR indices/data and never densifies a row | Yes; loops through rows `0..n_obs-1` and assigns with `obs_names` |
| Inference dataset | `epizoo.data.datasets.InferenceCellDataset` | Cell-sentence sequence and equally long species sequence | One `(input_ids, species)` item per cell | `max_length=8192`; maximum cCRE tokens is 8190 because `[CLS]` and `[SEP]` are added. Default `random_sample=True`; if truncating, sampled positions are sorted so retained tokens keep their original rank order | Matrix no longer involved | Yes by dataset index |
| Batch collation | `epizoo.data.datasets.inference_collate_fn` | List of dataset items | `(input_ids, list(species))`; `input_ids` is `LongTensor[batch, batch_max_length]` | Adds right padding token `0` to the longest sequence in each batch | Not applicable | Yes within a batch |
| DataLoader | `torch.utils.data.DataLoader` | `InferenceCellDataset` | Sequential batches | Validated tutorial: `batch_size=32`, `shuffle=False`, `collate_fn=inference_collate_fn`, default `num_workers=0` | Not applicable | Yes only if `shuffle=False` |
| EpiZoo forward | `epizoo.models.epizoo.EpiZoo.forward()` | `input_ids` | Mapping containing `cell_emb`; optionally full transformer output | Token embedding is `ccre_emb + seq_emb + rank_emb`; padding mask is `input_ids != 0`; cell embedding is encoder output at sequence position 0 (`[CLS]`) | Not applicable | Yes by tensor batch row |
| Extraction | `epizoo.inference.embeddings.extract_cell_embeddings()` | Model and DataLoader | By default NumPy array `[n_cells, emb_dim]` | Moves model to resolved device, calls `eval()`, uses `torch.no_grad()`, CUDA autocast by default, calls model with `return_transformer_out=False`, moves each batch output to CPU, and concatenates in iteration order | Not applicable | Yes for a sequential DataLoader; this function does not carry or reorder by an explicit cell index |

Additional ordering details:

- The audited filter-index arrays are unique and strictly increasing, so retained local cCRE index `j` is the `j`th row of the corresponding filter CSV.
- `numpy.argsort(-values)` is used without an explicit stable sort. Descending TF-IDF order is validated, but the ordering of exactly tied scores is not specified by the code.
- `extract_cell_embeddings()` ignores the species field returned by the inference collator; species affects inference through token offsets, not a separate model argument.
- `return_transformer_out=False` prevents returning the full encoder tensor but does not make the encoder compute only `[CLS]`.

## 2. Human configuration

| Property | Validated value | Evidence |
|---|---:|---|
| Species ID | `0` | Dataset/model conventions and human tutorial |
| Expected raw input feature dimension | `1,355,445` | Shape of `cCRE_frequencies_human.npy`; also exactly the row count of EpiAgent's historical `data/cCRE.bed` |
| Document-frequency file | `/home/likeyi/program/EpiZoo/data/cCRE_frequencies_human.npy` | `float64`, shape `(1,355,445,)` |
| Filter-index file | `/home/likeyi/program/EpiZoo/data/cCRE_filter_idx_human.csv` | Columns `cCRE`, `idx`; 700,460 rows |
| Retained cCREs | `700,460` | Filter CSV row count and `filter_cCREs()` validation |
| Filter-index range | `0..1,355,442` | Actual CSV values; unique and strictly increasing |
| TF-IDF `cell_number` | `8,200,000` | Human pipeline in `06_compute_loa_score.ipynb` |
| `base_offset` | `4` | `generate_cell_sentences()` default |
| Species offset | `0` | Human call uses `species=0`; extra offset is applied only for species 1 |
| Human token IDs | `4..700,463` inclusive | 700,460 retained local indices plus offset 4 |

The human filter CSV's `cCRE` values match `EpiAgent/data/cCRE.bed` at every selected `idx`; all 700,460 selected positions were checked. The wrapper should not otherwise depend on EpiAgent.

Resource SHA-256:

- frequencies: `8b576fa4fc60a1e2fc77607ffacff2883441def3d8cb8225776b71ad6c00e80b`
- filter index: `994d9c3e87208074e695c4c418b28d9587dd8991ad033cf33e62f96ceebc7875`

## 3. Mouse configuration

| Property | Validated value | Evidence |
|---|---:|---|
| Species ID | `1` | Dataset/model conventions and embedding tutorial |
| Expected raw input feature dimension | `1,341,077` | Shape of `cCRE_frequencies_mouse.npy` and tutorial h5ad |
| Document-frequency file | `/home/likeyi/program/EpiZoo/data/cCRE_frequencies_mouse.npy` | `float64`, shape `(1,341,077,)` |
| Filter-index file | `/home/likeyi/program/EpiZoo/data/cCRE_filter_idx_mouse.csv` | Columns `cCRE`, `idx`; 814,020 rows |
| Retained cCREs | `814,020` | Filter CSV row count and `filter_cCREs()` validation |
| Filter-index range | `3..1,341,062` | Actual CSV values; unique and strictly increasing |
| TF-IDF `cell_number` | `12,500,000` | Embedding tutorial and all other audited mouse tutorials |
| `base_offset` | `4` | `generate_cell_sentences()` default |
| Species offset | `700,460` | Embedding tutorial; equals retained human vocabulary size |
| Mouse token IDs | `700,464..1,514,483` inclusive | 814,020 retained local indices plus `4 + 700,460` |

For the supplied Fang2021 h5ad, all 814,020 retained `adata.var_names[filter_idx]` exactly match the mouse filter CSV's `cCRE` column. The raw feature names are unique.

Resource SHA-256:

- frequencies: `c4c63aaae8a6f841812189bf59930014162c3f61c7a463d744be71595359171d`
- filter index: `96a80287ae085d7e9e10d05f0dd7d5b266ad86d08b91b01ccc07af6b9b01e393`

### Final joint vocabulary layout

| Token ID range | Meaning | Count |
|---|---|---:|
| `0` | `[PAD]` | 1 |
| `1` | `[CLS]` | 1 |
| `2` | `[SEP]` | 1 |
| `3` | Reserved by the four-token offset; unnamed and unused by the audited cell APIs | 1 |
| `4..700,463` | Filtered human cCREs in filter-CSV order | 700,460 |
| `700,464..1,514,483` | Filtered mouse cCREs in filter-CSV order | 814,020 |

Therefore `vocab_size=1,514,484`, representing valid IDs `0..1,514,483`. No local EpiZoo cell API assigns a meaning to token 3, so its semantics must not be guessed.

## 4. Checkpoint-derived model configuration

Checkpoint: `/home/likeyi/program/model_checkpoints/EpiZoo/pretrained_EpiZoo.pth`

- SHA-256: `6b2d13fdbd54a9b0d56efa5afa81bc4832b813f4eac8662e4d93cc08c9d9b39a`
- Format: unwrapped `collections.OrderedDict`, 793 tensors, no saved config/metadata wrapper
- All checkpoint tensors: `torch.float16`
- Parameter count: `2,615,692,905`
- Tensor storage: 5,231,385,810 bytes = 4.872 GiB in FP16; the same parameters occupy 9.744 GiB in FP32, excluding activations and allocator overhead

| Configuration | Checkpoint-compatible value | Tensor/key evidence |
|---|---:|---|
| `vocab_size` | `1,514,484` | `ccre_emb.weight` and `seq_emb.weight`: `(1,514,484, 512)` |
| `human_vocab_size` | `700,460` | Human decoder weight `(700,460, 512)` and bias `(700,460,)` |
| `mouse_vocab_size` | `814,020` | Mouse decoder weight `(814,020, 512)` and bias `(814,020,)` |
| `emb_dim` | `512` | All embedding widths and encoder hidden dimensions |
| `max_rank` | `8,192` | `rank_emb.weight`: `(8,192, 512)` |
| `num_layers` | `30` | Contiguous checkpoint key prefixes `encoder.layers.0` through `.29` |
| `num_heads` | `8` for the repository-compatible construction; not independently inferable from checkpoint shapes | Tutorial leaves the `EpiZooConfig` default of 8 unchanged. QKV `(1,536,512)` proves `3 * hidden_size`, but does not encode how 512 is partitioned into heads |
| `use_moe` | `True` | Every layer has `mlp.gate` and `mlp.experts.*` keys |
| `num_experts` | `4` | Every layer has expert IDs 0, 1, 2, and 3; gate weight is `(4,512)` |
| `top_k` | Repository/tutorial default `2`; not inferable from weights | Routing `top_k` creates no differently shaped parameter |
| FFN/intermediate dimension | `2,048` | Expert weights `(2,048,512)` and `(512,2,048)` |
| CCA hidden dimension | `128` | CCA first layer `(128,1,024)` and last layer `(1,128)` |
| Signal decoder dimensions | Human `512 -> 700,460`; mouse `512 -> 814,020` | Species decoder tensors above |
| FlashAttention setting | Tutorial/config default `True`; not inferable from weights | Runtime choice does not alter these checkpoint shapes |

FP16 storage by component is approximately: `ccre_emb` 1.444 GiB, `seq_emb` 1.444 GiB, `rank_emb` 0.008 GiB, encoder 0.528 GiB, CCA head 0.00025 GiB, and signal decoders 1.447 GiB.

### Default-config discrepancy

The current `EpiZooConfig` defaults describe the unfiltered raw cross-species space and an 18-layer model:

- `vocab_size=2,696,526 = 1,355,445 + 1,341,077 + 4`
- `human_vocab_size=1,355,445`
- `mouse_vocab_size=1,341,077`
- `num_layers=18`

Those defaults are incompatible with this checkpoint. They cause shape mismatches for both token embedding matrices and both signal decoders, and omit checkpoint encoder layers 18 through 29.

The embedding tutorial supplies the correct overrides:

```python
EpiZooConfig(
    vocab_size=700_460 + 814_020 + 4,
    human_vocab_size=700_460,
    mouse_vocab_size=814_020,
    num_layers=30,
)
```

The other unchanged defaults match checkpoint shapes where inferable: `emb_dim=512`, `max_rank=8192`, `num_experts=4`, FFN dimension `4 * 512=2048`, and `cca_hidden_dim=128`. The tutorial's `strict=False` is unsafe and unnecessary for the checkpoint-compatible shape configuration. The wrapper must use `strict=True`, must not expose a non-strict escape hatch, and must fail on any missing, unexpected, or shape-mismatched key.

## 5. Exact EpiZoo APIs to reuse

The wrapper should orchestrate, not reimplement, these validated APIs:

1. `compute_tfidf(adata, df, cell_number=..., scale_factor=10_000, dtype=np.float32, store="X", verbose=...)`
2. `filter_cCREs(adata, filter_idx=..., species=..., verbose=...)`
3. `generate_cell_sentences(adata, matrix_key="X", obs_key="cell_indices", species=..., base_offset=4, species_offset=...)`
4. `InferenceCellDataset(cell_sentences=..., species=..., max_length=8192, random_sample=...)`
5. `DataLoader(..., shuffle=False, collate_fn=inference_collate_fn)`
6. `EpiZoo(cfg=checkpoint_compatible_config)`
7. `extract_cell_embeddings(model, dataloader, device=..., use_amp=..., return_numpy=True, show_progress=...)`

The wrapper must add validation, path/resource selection, deterministic seeding, strict checkpoint loading, order metadata, and reproducibility metadata around these calls. It should not copy EpiZoo's scientific formulas or model code into Agent.

## 6. Proposed `load_model()` interface

```python
def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EpiZoo:
    ...
```

Required behavior:

- Construct the fixed checkpoint-compatible config from section 4. Do not accept arbitrary architecture overrides in v1.
- Load the checkpoint with `map_location="cpu"` and the safest weights-only mode supported by the pinned PyTorch version.
- Require an unwrapped state dict or explicitly unwrap only a documented `state_dict` field; validate that all values are tensors.
- Load with `strict=True`. There should be no public `strict` argument.
- Preserve/choose the requested compute dtype deliberately. The tutorial constructs the module in PyTorch's default FP32, loads the FP16 state into those FP32 parameters, and then uses CUDA autocast; FP32 is therefore the validated runtime default even though it expands the checkpoint to 9.744 GiB. An FP16-parameter mode may be exposed, but it needs a numerical-parity smoke test and must be recorded.
- Move to `device`, call `eval()`, and attach or return enough immutable metadata to record checkpoint path/hash and resolved config.
- Default to CPU placement so loading does not unexpectedly consume GPU memory. CPU placement is for loading/inspection; actual inference should normally use CUDA with the supported FlashAttention environment.

## 7. Proposed `embed_cells()` interface

```python
def embed_cells(
    model: EpiZoo,
    adata: anndata.AnnData,
    *,
    species: Literal["human", "mouse", 0, 1],
    resources_dir: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = 4,
    max_length: int = 8192,
    random_sample: bool = True,
    random_seed: int = 0,
    use_amp: bool = True,
    num_workers: int = 0,
    show_progress: bool = True,
) -> EpiZooEmbeddingResult:
    ...
```

`EpiZooEmbeddingResult` is a structured result, not another scientific capability. It should contain:

- `embeddings`: C-contiguous `numpy.ndarray`, shape `(adata.n_obs, 512)`, normalized to `float32` for downstream stability
- `obs_names`: immutable copy of input cell identifiers in exact output-row order
- `metadata`: species ID/name, raw and retained dimensions, TF-IDF parameters, resource paths and hashes, checkpoint path/hash, checkpoint-compatible model config, device, compute dtype, AMP flag, batch size, max length, truncation mode, seed, and EpiZoo/package versions

Required behavior:

- Normalize species name/ID, select the exact resources and offsets in sections 2–3, and do not permit caller overrides of scientific constants in v1.
- Validate before preprocessing, then call the APIs in section 5 in order.
- Use `shuffle=False` and, initially, `num_workers=0` to make sampling/order behavior reproducible.
- Seed NumPy deterministically when `random_sample=True` because `InferenceCellDataset` calls global `numpy.random.choice` during iteration. Ideally preserve and restore caller RNG state.
- Verify output row count, embedding width, finiteness, and exact `obs_names` equality before returning.
- Do not mutate the caller's `AnnData`. EpiZoo's preprocessing functions already return copies; release intermediate copies as soon as possible.

The validated dataset default is `random_sample=True`. Exposing the flag is necessary: `False` deterministically keeps the top 8190 TF-IDF-ranked cCREs, while `True` follows the backend's random subsampling behavior. A seed must be recorded either way.

## 8. Input validation requirements

Validation must happen before expensive copies/model movement and must use sparse-safe operations.

1. `adata` is an `AnnData` with `n_obs > 0`, nonempty unique `obs_names`, and cells in rows.
2. `.X` is a SciPy sparse matrix (CSR preferred; CSC can be converted to CSR without densifying). Reject a full dense scATAC matrix in v1.
3. Reject backed `anndata` sparse datasets in v1 unless the implementation explicitly materializes them as a SciPy sparse matrix without densification. EpiZoo's current `sp.issparse()` checks do not recognize the audited backed `_CSRDataset` object.
4. Raw feature dimension is exactly 1,355,445 for human or 1,341,077 for mouse.
5. `df` has exactly the raw feature dimension, is one-dimensional, finite, and nonnegative.
6. Filter CSV has exactly the expected retained row count; `idx` is integer, unique, strictly increasing, in bounds; `cCRE` is present and unique.
7. At every retained position, `adata.var_names[filter_idx]` exactly equals the filter CSV's `cCRE` value. Dimension alone is insufficient because TF-IDF and tokens are positional.
8. Sparse stored values are finite, nonnegative raw counts and integer-valued within a small numerical tolerance. This prevents accidental double-TF-IDF processing.
9. No explicit stored zeros after cleanup; call sparse `eliminate_zeros()` on a copy if required.
10. Handle zero-count cells explicitly. Backend code would create `[CLS, SEP]`; the wrapper should reject them with cell IDs unless this behavior is consciously accepted and documented later.
11. `1 <= max_length <= 8192` is not sufficient because two special tokens are required; require `2 <= max_length <= 8192`. For useful biological input, a stricter minimum may be chosen.
12. `batch_size > 0`, `num_workers >= 0`, valid device, and CUDA availability when requested.
13. The model's stored config and tensor shapes match section 4 before preprocessing/inference.
14. Generated token IDs stay inside the correct species interval and below `vocab_size`; generated sentence/species lengths equal `n_obs`.

Do not reject the validated human frequency resource merely because 16 document-frequency values exceed the tutorial's `cell_number=8,200,000`; this inconsistency needs provenance clarification but is part of the currently validated local pipeline.

## 9. Expected output structure

For `n` input cells, return:

```text
EpiZooEmbeddingResult
├── embeddings: ndarray[n, 512], float32, finite
├── obs_names: tuple[str, ...], length n, exactly input order
└── metadata: mapping
    ├── species: {id, name}
    ├── preprocessing: {cell_number, scale_factor, base_offset,
    │                   species_offset, max_length, random_sample, seed}
    ├── resources: paths + SHA-256
    ├── checkpoint: path + SHA-256
    ├── model_config: checkpoint-compatible values
    └── execution: device, dtype, AMP, batch_size, package versions
```

The result must not contain or densify the raw scATAC matrix. If an `AnnData` representation is wanted by a downstream caller, it can build a small embedding-only `AnnData` from the `(n,512)` result and `obs_names`.

## 10. Failure cases

- Wrong species, raw dimension, feature order, resource pair, or token offset: fail before TF-IDF.
- Dense or unsupported backed `.X`: fail rather than risking a full-matrix conversion via `np.asarray()`.
- Negative, nonfinite, already normalized, empty, or all-zero data: fail with affected cell/feature information where practical.
- Bad filter indices: current EpiZoo code may raise `AssertionError` or fail during slicing; wrapper validation must raise informative `ValueError`/tool-specific errors first.
- All-zero matrices can also make `compute_tfidf(verbose=True)` fail when it calls min/max on an empty data array.
- Missing/corrupted checkpoint or resources, hash drift, non-state-dict checkpoint, non-tensor entries, or architecture mismatch: fail; never retry with `strict=False`.
- `max_length > 8192`: rank-embedding index error. Long sequences at `max_length=8192` are truncated to 8190 cCRE tokens.
- Empty DataLoader: `extract_cell_embeddings()` fails at `torch.cat([])`; prevalidate `n_obs > 0`.
- `shuffle=True` or a custom sampler: embeddings no longer necessarily match input order; the wrapper must own DataLoader construction.
- Unseeded random truncation: nondeterministic embeddings for cells with more than 8190 retained accessible cCREs.
- Missing/incompatible `flash_attn`, CUDA, PyTorch, Transformers, NumPy/Scanpy/AnnData stack: fail during import/model construction with an environment diagnostic.
- CUDA OOM: report requested batch size, maximum batch sequence length, dtype, and free/total VRAM; advise retry with a smaller batch rather than silently changing scientific parameters.

## 11. RTX 4090 / 24 GB memory considerations

- Checkpoint parameter storage alone is 4.872 GiB in FP16 or 9.744 GiB after expansion to FP32. CUDA allocator state, activations, attention workspaces, and input batches are additional.
- Even though embedding inference does not call the signal decoders, their strict checkpoint-compatible weights occupy about 1.447 GiB FP16. Removing them would change the model/state-dict contract and is outside v1.
- The validated tutorial behavior is FP32 model parameters on GPU plus CUDA autocast. Start with that behavior at `batch_size=1`, then test 2 and 4; do not adopt the tutorial's batch size 32 as a 24 GB default. A deliberate FP16-parameter option lowers weight storage to 4.872 GiB but must pass wrapper-versus-tutorial numerical tests before becoming the default.
- Dynamic batch padding means memory is driven by the longest sentence in each batch. Length-aware batching could save memory later, but it would need explicit output-index tracking to restore cell order and is not part of v1.
- FlashAttention with unpadding is important for sequences up to 8192. Without the validated FlashAttention path, quadratic padded attention can exceed 24 GB; do not silently fall back for the full smoke set without measuring.
- The encoder still creates per-token outputs for every layer before taking `[CLS]`. `return_transformer_out=False` does not eliminate those intermediate activations.
- Never form a dense raw or filtered matrix. For the 2,000-cell mouse smoke data alone, a dense filtered float32 matrix would be about 6.07 GiB; full-scale datasets would be far larger.
- TF-IDF, filtering, and sentence generation each make an `AnnData` copy. Ensure old intermediates are released promptly. Python lists of token IDs and the dataset's parsed copies can also consume substantial host RAM at scale.
- Accumulating final embeddings on CPU is inexpensive relative to the model: 2,000 x 512 x float32 is about 3.9 MiB.
- `torch.cuda.empty_cache()` every ten batches is part of the existing extraction API. It may reduce cached-memory pressure but is not a substitute for a safe batch size.

## 12. Smoke-test plan

Dataset: `/home/likeyi/program/EpiZoo/data/Fang2021_downsampled_2000_cells.h5ad`

Audited facts, without running model inference:

- Shape is `2000 x 1,341,077`; `.X` loads as SciPy CSR float32 with unique feature names.
- Tutorial raw matrix has 6,703,562 nonzeros.
- After positional filtering, sparse shape is `2000 x 814,020` with 6,144,312 nonzeros.
- Retained accessible cCREs per cell: min 388, median 2,247, mean 3,072.156, max 21,938; no zero cells.
- 107 cells have more than the allowed 8190 cCRE tokens, so truncation mode and seed materially affect this smoke test.

Planned tests, in order:

1. Environment preflight: use EpiZoo's required Python 3.11 environment; pin compatible PyTorch/CUDA/FlashAttention and EpiZoo's declared package versions. Confirm direct imports before allocating the model.
2. Resource/config unit test: verify resource hashes, shapes, index properties, selected feature-name equality, vocabulary intervals, checkpoint hash, and checkpoint-derived config.
3. Sparse preprocessing test on 8 cells while retaining all raw columns: run the exact TF-IDF -> filter -> sentence chain; assert every intermediate matrix remains CSR, cells/`obs_names` remain ordered, and token IDs fall in the mouse interval.
4. Truncation test: include the cell with 21,938 retained cCREs. Check output length 8192 including special tokens, deterministic repetition for a fixed seed, and correct behavior for both `random_sample=True` and `False`.
5. Strict-load test on CPU: construct the section-4 config, load with `strict=True`, assert zero missing/unexpected keys, tensor/config shapes, eval mode, and recorded checkpoint hash. Do not perform CPU inference with the CUDA/FlashAttention configuration.
6. Minimal CUDA inference: move the validated FP32-parameter model to the RTX 4090, enable AMP, and embed 1–2 cells with `batch_size=1`; assert `(n,512)`, finite values, expected output dtype conversion, and GPU peak memory below 24 GB. Separately test FP16 parameters only as an explicitly recorded memory optimization.
7. Wrapper-versus-manual test: on the same small ordered subset, compare the wrapper output with the explicit validated EpiZoo call chain under identical truncation seed/settings. Require matching rows and tight numerical tolerance appropriate to FP16/AMP.
8. Order test: use deliberately distinctive/reversed `obs_names`; assert returned IDs and rows remain in that exact order.
9. Failure tests: wrong species dimension, shuffled `var_names`, dense `.X`, backed `.X`, negative/nonfinite values, zero cell, missing resources, `max_length=8193`, and checkpoint/default-config mismatch.
10. Only after the above passes, embed all 2,000 downsampled cells with a measured safe batch size (initially 2–4), monitor `torch.cuda.max_memory_allocated()`, and assert final shape `(2000,512)`, finiteness, repeatability, and exact cell order. This is still a smoke dataset, not full-scale inference.

## 13. Remaining uncertainties

1. Token ID 3 is reserved by the four-token offset but has no named constant or use in the audited cell pipeline.
2. The checkpoint contains no serialized config. Eight attention heads, top-k 2, and FlashAttention enabled are the repository/tutorial construction values, but they cannot be proven from tensor shapes alone. Only the head count affects computation without changing parameter shapes.
3. The human tutorial specifies `cell_number=8,200,000`, but 16 values in `cCRE_frequencies_human.npy` exceed 8,200,000 (maximum 9,984,077). The local code and tutorial still unambiguously prescribe 8,200,000; the frequency provenance should be confirmed with EpiZoo maintainers before declaring the human normalization metadata internally consistent.
4. The current `filter_cCREs()` docstring calls 700,460/814,020 the expected vocabulary sizes and asserts `len(filter_idx)`, but does not validate the raw input feature dimension, feature order, or bounds. Those checks belong in the wrapper.
5. `filter_cCREs()` documents boolean or integer `filter_idx` input, but its species assertions require `len(filter_idx)` to equal the retained count. A normal full-raw-length boolean mask therefore cannot pass the assertion. The supplied CSV integer arrays are the validated path and should be used.
6. `compute_tfidf()` documentation says a supplied `df` with omitted `cell_number` falls back to `adata.n_obs`; the implementation actually raises `ValueError`. The wrapper must always pass the audited species-specific value explicitly.
7. The current active Agent shell is not the EpiZoo runtime declared by `pyproject.toml`: it is Python 3.10.13, PyTorch 2.0.0, Transformers 5.9.0, NumPy 1.26.3, SciPy 1.15.3, and AnnData 0.11.4; `flash_attn` is missing. EpiZoo declares Python `>=3.11,<3.12`, Transformers 4.57.6, NumPy 2.1.2, SciPy 1.17.1, and AnnData 0.12.10. Direct EpiZoo import currently fails. The model tool cannot be runtime-tested until a compatible environment is selected.
8. EpiZoo does not declare PyTorch or FlashAttention versions in its `pyproject.toml`/`requirements.txt`, even though the model imports FlashAttention unconditionally. A reproducible compatible pair must be pinned for Agent. FlashAttention's `Block` module also determines encoder state-dict key names, so the version must match the current EpiZoo source/checkpoint contract.
9. Runtime strict loading could not be executed in the active shell because EpiZoo cannot be imported. Static checkpoint evidence is complete: its 793 keys exactly follow the current source's expected three embeddings, 30 repeated four-expert layers, CCA head, and two species decoders. A real `strict=True` construction/load is the first environment smoke test.
10. Exact tie ordering in cell sentences is unspecified because `np.argsort` is not requested to be stable.

## Readiness decision

The scientific interface is sufficiently resolved to implement the minimal wrapper: species resources, preprocessing order, offsets, vocabulary, checkpoint architecture, sparsity behavior, and order behavior are all explicit. Implementation is safe only if it uses the checkpoint-derived config with strict loading, adds the validation above, preserves sparse SciPy matrices, and begins with the small smoke tests. Runtime acceptance is currently blocked by environment compatibility—not by an unresolved model or data interface.
