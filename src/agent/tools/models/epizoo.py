"""Validated EpiZoo cell-embedding wrapper.

This module orchestrates EpiZoo's existing preprocessing and inference APIs. It
does not reimplement the backend's scientific transformations or model logic.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module, metadata as importlib_metadata
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterator, Literal, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader


DEFAULT_CHECKPOINT_PATH = Path(
    "/home/likeyi/program/model_checkpoints/EpiZoo/pretrained_EpiZoo.pth"
)
DEFAULT_RESOURCES_DIR = Path("/home/likeyi/program/EpiZoo/data")

MODEL_CONFIG: Mapping[str, Any] = MappingProxyType({
    "vocab_size": 1_514_484,
    "human_vocab_size": 700_460,
    "mouse_vocab_size": 814_020,
    "emb_dim": 512,
    "max_rank": 8192,
    "num_layers": 30,
    "num_heads": 8,
    "use_moe": True,
    "num_experts": 4,
    "top_k": 2,
})


@dataclass(frozen=True)
class _SpeciesConfig:
    name: str
    species_id: int
    raw_dimension: int
    retained_ccres: int
    cell_number: int
    species_offset: int
    frequency_filename: str
    filter_filename: str

    @property
    def token_min(self) -> int:
        return 4 + self.species_offset

    @property
    def token_max(self) -> int:
        return self.token_min + self.retained_ccres - 1


HUMAN_CONFIG = _SpeciesConfig(
    name="human",
    species_id=0,
    raw_dimension=1_355_445,
    retained_ccres=700_460,
    cell_number=8_200_000,
    species_offset=0,
    frequency_filename="cCRE_frequencies_human.npy",
    filter_filename="cCRE_filter_idx_human.csv",
)
MOUSE_CONFIG = _SpeciesConfig(
    name="mouse",
    species_id=1,
    raw_dimension=1_341_077,
    retained_ccres=814_020,
    cell_number=12_500_000,
    species_offset=700_460,
    frequency_filename="cCRE_frequencies_mouse.npy",
    filter_filename="cCRE_filter_idx_mouse.csv",
)

_SPECIES_BY_NAME = {"human": HUMAN_CONFIG, "mouse": MOUSE_CONFIG}
_SPECIES_BY_ID = {0: HUMAN_CONFIG, 1: MOUSE_CONFIG}


@dataclass(frozen=True)
class EpiZooEmbeddingResult:
    """Cell embeddings and the information required to reproduce them."""

    embeddings: np.ndarray
    obs_names: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _Resources:
    frequencies: np.ndarray
    filter_indices: np.ndarray
    retained_names: np.ndarray
    frequency_path: Path
    filter_path: Path
    frequency_sha256: str
    filter_sha256: str


@dataclass(frozen=True)
class _EpiZooBackend:
    compute_tfidf: Any
    filter_cCREs: Any
    generate_cell_sentences: Any
    InferenceCellDataset: Any
    inference_collate_fn: Any
    extract_cell_embeddings: Any
    EpiZoo: Any
    EpiZooConfig: Any


@lru_cache(maxsize=1)
def _get_backend() -> _EpiZooBackend:
    """Import EpiZoo lazily so validation-only callers do not initialize CUDA."""

    try:
        processing = import_module("epizoo.data.processing")
        datasets = import_module("epizoo.data.datasets")
        embeddings = import_module("epizoo.inference.embeddings")
        models = import_module("epizoo.models.epizoo")
    except Exception as exc:  # pragma: no cover - exact backend error is environment-specific
        raise RuntimeError(
            "Unable to import the EpiZoo core modules. Run this operation in the "
            "validated EpiZoo environment with a visible CUDA device and compatible "
            "PyTorch/FlashAttention versions."
        ) from exc

    return _EpiZooBackend(
        compute_tfidf=processing.compute_tfidf,
        filter_cCREs=processing.filter_cCREs,
        generate_cell_sentences=processing.generate_cell_sentences,
        InferenceCellDataset=datasets.InferenceCellDataset,
        inference_collate_fn=datasets.inference_collate_fn,
        extract_cell_embeddings=embeddings.extract_cell_embeddings,
        EpiZoo=models.EpiZoo,
        EpiZooConfig=models.EpiZooConfig,
    )


def _build_model_config(backend: _EpiZooBackend) -> Any:
    return backend.EpiZooConfig(**dict(MODEL_CONFIG))


def _normalize_species(species: Literal["human", "mouse", 0, 1]) -> _SpeciesConfig:
    if isinstance(species, str):
        normalized = species.strip().lower()
        if normalized in _SPECIES_BY_NAME:
            return _SPECIES_BY_NAME[normalized]
    elif not isinstance(species, (bool, np.bool_)) and isinstance(
        species, (int, np.integer)
    ):
        species_id = int(species)
        if species_id in _SPECIES_BY_ID:
            return _SPECIES_BY_ID[species_id]

    raise ValueError("`species` must be 'human', 'mouse', 0, or 1.")


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid PyTorch device: {device!r}.") from exc

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved} was requested but CUDA is not available in this runtime."
        )
    return resolved


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unwrap_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "EpiZoo checkpoint must be a state-dict mapping or contain a `state_dict` mapping."
        )

    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in checkpoint.items():
        if not isinstance(key, str):
            raise TypeError("Every checkpoint state-dict key must be a string.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint entry {key!r} is not a tensor.")
        state_dict[key] = value
    return state_dict


def _expect_shape(
    state_dict: Mapping[str, torch.Tensor], key: str, expected: tuple[int, ...]
) -> None:
    if key not in state_dict:
        raise ValueError(f"Checkpoint is missing required tensor {key!r}.")
    actual = tuple(state_dict[key].shape)
    if actual != expected:
        raise ValueError(
            f"Checkpoint tensor {key!r} has shape {actual}; expected {expected}."
        )


def _validate_checkpoint_state_dict(state_dict: Mapping[str, torch.Tensor]) -> None:
    _expect_shape(state_dict, "ccre_emb.weight", (1_514_484, 512))
    _expect_shape(state_dict, "seq_emb.weight", (1_514_484, 512))
    _expect_shape(state_dict, "rank_emb.weight", (8192, 512))
    _expect_shape(state_dict, "cca_head.net.0.weight", (128, 1024))
    _expect_shape(state_dict, "cca_head.net.4.weight", (1, 128))
    _expect_shape(
        state_dict, "signal_decoder.decoders.human.weight", (700_460, 512)
    )
    _expect_shape(
        state_dict, "signal_decoder.decoders.mouse.weight", (814_020, 512)
    )

    layer_ids = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"encoder\.layers\.(\d+)\.", key))
    }
    if layer_ids != set(range(30)):
        raise ValueError(
            "Checkpoint encoder layers do not match the required contiguous range 0..29."
        )

    for layer_id in range(30):
        _expect_shape(
            state_dict,
            f"encoder.layers.{layer_id}.mixer.Wqkv.weight",
            (1536, 512),
        )
        _expect_shape(
            state_dict,
            f"encoder.layers.{layer_id}.mlp.gate.weight",
            (4, 512),
        )
        expert_ids = {
            int(match.group(1))
            for key in state_dict
            if (
                match := re.match(
                    rf"encoder\.layers\.{layer_id}\.mlp\.experts\.(\d+)\.", key
                )
            )
        }
        if expert_ids != {0, 1, 2, 3}:
            raise ValueError(
                f"Checkpoint layer {layer_id} experts are {sorted(expert_ids)}; "
                "expected [0, 1, 2, 3]."
            )
        for expert_id in range(4):
            _expect_shape(
                state_dict,
                f"encoder.layers.{layer_id}.mlp.experts.{expert_id}.0.weight",
                (2048, 512),
            )
            _expect_shape(
                state_dict,
                f"encoder.layers.{layer_id}.mlp.experts.{expert_id}.2.weight",
                (512, 2048),
            )


def _add_fixed_loss_buffers(
    state_dict: OrderedDict[str, torch.Tensor], cfg: Any
) -> tuple[str, ...]:
    """Add only deterministic loss buffers absent from the inference checkpoint.

    PyTorch 2.7 includes BCE ``pos_weight`` values in module state dicts, while
    the validated EpiZoo checkpoint contains only learned model tensors. These
    buffers are fixed by EpiZooConfig and are not learned parameters.
    """

    expected = {
        "cca_loss_fn.pos_weight": float(cfg.cca_pos_weight),
        "signal_loss_fn.pos_weight": float(cfg.signal_pos_weight),
    }
    added: list[str] = []
    for key, value in expected.items():
        if key in state_dict:
            tensor = state_dict[key]
            if tensor.numel() != 1 or float(tensor.detach().cpu().item()) != value:
                raise ValueError(
                    f"Checkpoint buffer {key!r} does not match EpiZooConfig value {value}."
                )
        else:
            state_dict[key] = torch.tensor(value, dtype=torch.float32)
            added.append(key)
    return tuple(added)


def _validate_config_values(cfg: Any) -> None:
    mismatches = {
        key: (getattr(cfg, key, None), expected)
        for key, expected in MODEL_CONFIG.items()
        if getattr(cfg, key, None) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"EpiZoo model config is checkpoint-incompatible: {details}.")


def _validate_model_structure(model: Any) -> None:
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise TypeError("`model` must be an EpiZoo model with a `cfg` attribute.")
    _validate_config_values(cfg)

    expected_shapes = {
        "ccre_emb.weight": (1_514_484, 512),
        "seq_emb.weight": (1_514_484, 512),
        "rank_emb.weight": (8192, 512),
        "signal_decoder.decoders.human.weight": (700_460, 512),
        "signal_decoder.decoders.mouse.weight": (814_020, 512),
    }
    state = model.state_dict()
    for key, expected in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != expected:
            actual = tuple(state[key].shape) if key in state else None
            raise ValueError(
                f"Model tensor {key!r} has shape {actual}; expected {expected}."
            )
    if len(model.encoder.layers) != 30:
        raise ValueError("EpiZoo model must contain exactly 30 encoder layers.")


def _validate_loaded_model(model: Any) -> None:
    if not getattr(model, "_agent_checkpoint_validated", False):
        raise ValueError(
            "`model` must be created by agent.tools.models.epizoo.load_model() "
            "so strict checkpoint compatibility is guaranteed."
        )
    _validate_model_structure(model)


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Load the fixed pretrained EpiZoo architecture with strict validation."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"EpiZoo checkpoint not found: {checkpoint_path}")
    if dtype not in {torch.float16, torch.float32}:
        raise ValueError("`dtype` must be torch.float16 or torch.float32.")
    resolved_device = _resolve_device(device)

    backend = _get_backend()
    cfg = _build_model_config(backend)
    _validate_config_values(cfg)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state_dict = _unwrap_state_dict(checkpoint)
    _validate_checkpoint_state_dict(state_dict)
    synthesized_buffers = _add_fixed_loss_buffers(state_dict, cfg)

    # Meta construction avoids randomly initializing 2.6B parameters that are
    # immediately replaced by checkpoint tensors. `assign=True` materializes
    # the strictly matched CPU state before the deliberate dtype/device move.
    with torch.device("meta"):
        model = backend.EpiZoo(cfg=cfg)
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict EpiZoo checkpoint loading returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )

    model = model.to(device=resolved_device, dtype=dtype)
    model.eval()
    _validate_model_structure(model)
    if model.training:
        raise RuntimeError("EpiZoo model remained in training mode after eval().")

    model._agent_checkpoint_validated = True
    model._agent_checkpoint_path = str(checkpoint_path)
    model._agent_checkpoint_sha256 = _sha256_file(checkpoint_path)
    model._agent_model_config = dict(MODEL_CONFIG)
    model._agent_checkpoint_missing_keys = tuple(incompatible.missing_keys)
    model._agent_checkpoint_unexpected_keys = tuple(incompatible.unexpected_keys)
    model._agent_synthesized_checkpoint_buffers = synthesized_buffers
    return model


def _validate_adata(adata: Any, species_cfg: _SpeciesConfig) -> ad.AnnData:
    if not isinstance(adata, ad.AnnData):
        raise TypeError("`adata` must be an anndata.AnnData object.")
    if adata.n_obs <= 0:
        raise ValueError("`adata` must contain at least one cell.")
    if not adata.obs_names.is_unique:
        raise ValueError("`adata.obs_names` must be unique to preserve cell identity.")
    empty_names = [str(name) for name in adata.obs_names if not str(name)]
    if empty_names:
        raise ValueError("`adata.obs_names` must not contain empty cell identifiers.")
    if getattr(adata, "isbacked", False):
        raise TypeError(
            "Backed AnnData matrices are not supported in v1; load a SciPy sparse "
            "matrix into memory without densifying it."
        )
    if not sp.issparse(adata.X):
        raise TypeError(
            "`adata.X` must be a SciPy sparse matrix; dense scATAC matrices are rejected."
        )
    if adata.n_vars != species_cfg.raw_dimension:
        raise ValueError(
            f"{species_cfg.name} input must have {species_cfg.raw_dimension:,} raw "
            f"cCRE features; got {adata.n_vars:,}."
        )

    matrix = adata.X
    data = matrix.data
    if np.issubdtype(data.dtype, np.complexfloating) or not np.issubdtype(
        data.dtype, np.number
    ):
        raise TypeError("Sparse scATAC values must be real numeric counts.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Sparse scATAC values must all be finite.")
    if np.any(data < 0):
        raise ValueError("Sparse scATAC values must be nonnegative.")
    if data.size and not np.allclose(data, np.rint(data), rtol=0.0, atol=1e-6):
        raise ValueError(
            "Sparse scATAC values must be count-like integers; the input may already "
            "be normalized."
        )

    row_sums = np.asarray(matrix.sum(axis=1)).reshape(-1)
    zero_rows = np.flatnonzero(row_sums <= 0)
    if zero_rows.size:
        names = [str(adata.obs_names[i]) for i in zero_rows[:10]]
        suffix = "" if zero_rows.size <= 10 else f" (and {zero_rows.size - 10} more)"
        raise ValueError(f"Zero-count cells are not supported: {names}{suffix}.")

    return adata


def _prepare_sparse_adata(adata: ad.AnnData) -> ad.AnnData:
    """Copy validated input and normalize sparse storage without densifying it."""

    prepared = adata.copy()
    prepared_x = prepared.X.tocsr(copy=False)
    prepared_x.eliminate_zeros()
    prepared.X = prepared_x
    return prepared


def _load_resources(
    species_cfg: _SpeciesConfig, resources_dir: str | Path
) -> _Resources:
    resources_dir = Path(resources_dir).expanduser().resolve()
    frequency_path = resources_dir / species_cfg.frequency_filename
    filter_path = resources_dir / species_cfg.filter_filename
    for label, path in (("frequency", frequency_path), ("filter-index", filter_path)):
        if not path.is_file():
            raise FileNotFoundError(f"EpiZoo {label} resource not found: {path}")

    frequencies = np.load(frequency_path, allow_pickle=False)
    if frequencies.ndim != 1 or frequencies.shape[0] != species_cfg.raw_dimension:
        raise ValueError(
            f"{species_cfg.name} frequency resource must have shape "
            f"({species_cfg.raw_dimension},); got {frequencies.shape}."
        )
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("EpiZoo frequency resource must be finite and nonnegative.")

    frame = pd.read_csv(filter_path, index_col=0)
    required_columns = {"cCRE", "idx"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(
            f"EpiZoo filter resource must contain columns {sorted(required_columns)}."
        )
    if len(frame) != species_cfg.retained_ccres:
        raise ValueError(
            f"{species_cfg.name} filter resource must contain "
            f"{species_cfg.retained_ccres:,} rows; got {len(frame):,}."
        )
    if frame["idx"].isna().any() or not pd.api.types.is_integer_dtype(
        frame["idx"].dtype
    ):
        raise ValueError("EpiZoo filter `idx` values must be non-null integers.")

    filter_indices = frame["idx"].to_numpy(dtype=np.int64, copy=True)
    if np.unique(filter_indices).size != filter_indices.size:
        raise ValueError("EpiZoo filter indices must be unique.")
    if filter_indices.size > 1 and not np.all(np.diff(filter_indices) > 0):
        raise ValueError("EpiZoo filter indices must be strictly increasing.")
    if filter_indices[0] < 0 or filter_indices[-1] >= species_cfg.raw_dimension:
        raise ValueError("EpiZoo filter indices are outside the raw feature dimension.")

    if frame["cCRE"].isna().any() or not frame["cCRE"].is_unique:
        raise ValueError("EpiZoo retained cCRE names must be non-null and unique.")
    retained_names = frame["cCRE"].astype(str).to_numpy(copy=True)

    return _Resources(
        frequencies=np.asarray(frequencies),
        filter_indices=filter_indices,
        retained_names=retained_names,
        frequency_path=frequency_path,
        filter_path=filter_path,
        frequency_sha256=_sha256_file(frequency_path),
        filter_sha256=_sha256_file(filter_path),
    )


def _validate_retained_feature_order(
    adata: ad.AnnData, resources: _Resources
) -> None:
    actual = np.asarray(adata.var_names.take(resources.filter_indices), dtype=str)
    expected = resources.retained_names
    matches = actual == expected
    if not np.all(matches):
        first = int(np.flatnonzero(~matches)[0])
        raw_idx = int(resources.filter_indices[first])
        raise ValueError(
            "Input cCRE feature order does not match the EpiZoo reference at "
            f"raw position {raw_idx}: got {actual[first]!r}, expected {expected[first]!r}."
        )


def _obs_names_tuple(adata: ad.AnnData) -> tuple[str, ...]:
    return tuple(str(name) for name in adata.obs_names)


def _assert_cell_order(adata: ad.AnnData, expected: tuple[str, ...], stage: str) -> None:
    if _obs_names_tuple(adata) != expected:
        raise RuntimeError(f"Cell order changed during EpiZoo {stage}.")


def _validate_cell_sentences(
    cell_sentences: Any,
    obs_names: tuple[str, ...],
    species_cfg: _SpeciesConfig,
) -> None:
    if len(cell_sentences) != len(obs_names):
        raise RuntimeError(
            "Generated cell-sentence count does not match the number of input cells."
        )
    empty: list[str] = []
    for index, sentence in enumerate(cell_sentences):
        tokens = np.asarray(sentence)
        if tokens.size == 0:
            empty.append(obs_names[index])
            continue
        if not np.issubdtype(tokens.dtype, np.integer):
            raise RuntimeError(f"Generated tokens for cell {obs_names[index]!r} are not integers.")
        if tokens.min() < species_cfg.token_min or tokens.max() > species_cfg.token_max:
            raise RuntimeError(
                f"Generated tokens for cell {obs_names[index]!r} are outside the "
                f"{species_cfg.name} interval "
                f"[{species_cfg.token_min}, {species_cfg.token_max}]."
            )
    if empty:
        shown = empty[:10]
        suffix = "" if len(empty) <= 10 else f" (and {len(empty) - 10} more)"
        raise ValueError(
            f"Cells have no retained EpiZoo cCREs after filtering: {shown}{suffix}."
        )


@contextmanager
def _numpy_random_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("EpiZoo model has no parameters.") from exc


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def embed_cells(
    model: Any,
    adata: ad.AnnData,
    *,
    species: Literal["human", "mouse", 0, 1],
    resources_dir: str | Path = DEFAULT_RESOURCES_DIR,
    device: str | torch.device | None = None,
    batch_size: int = 4,
    max_length: int = 8192,
    random_sample: bool = True,
    random_seed: int = 0,
    use_amp: bool = True,
    num_workers: int = 0,
    show_progress: bool = True,
) -> EpiZooEmbeddingResult:
    """Embed raw sparse scATAC cells with the validated EpiZoo pipeline."""

    species_cfg = _normalize_species(species)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("`batch_size` must be an integer.")
    if int(batch_size) <= 0:
        raise ValueError("`batch_size` must be positive.")
    batch_size = int(batch_size)
    if isinstance(max_length, bool) or not isinstance(max_length, (int, np.integer)):
        raise TypeError("`max_length` must be an integer.")
    if not 2 <= int(max_length) <= 8192:
        raise ValueError("`max_length` must be between 2 and 8192 inclusive.")
    max_length = int(max_length)
    if num_workers != 0:
        raise ValueError("EpiZoo wrapper v1 requires `num_workers=0`.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)):
        raise TypeError("`random_seed` must be an integer.")
    random_seed = int(random_seed)
    if not 0 <= random_seed <= 2**32 - 1:
        raise ValueError("`random_seed` must be between 0 and 2**32 - 1.")
    if not isinstance(random_sample, (bool, np.bool_)):
        raise TypeError("`random_sample` must be a boolean.")
    if not isinstance(use_amp, (bool, np.bool_)):
        raise TypeError("`use_amp` must be a boolean.")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("`show_progress` must be a boolean.")

    validated_adata = _validate_adata(adata, species_cfg)
    input_obs_names = _obs_names_tuple(validated_adata)
    _validate_loaded_model(model)
    resources = _load_resources(species_cfg, resources_dir)
    _validate_retained_feature_order(validated_adata, resources)
    prepared = _prepare_sparse_adata(validated_adata)

    backend = _get_backend()
    tfidf = backend.compute_tfidf(
        prepared,
        resources.frequencies,
        cell_number=species_cfg.cell_number,
        scale_factor=10_000.0,
        dtype=np.float32,
        store="X",
        verbose=False,
    )
    if not sp.isspmatrix_csr(tfidf.X):
        raise RuntimeError("EpiZoo TF-IDF output must remain a SciPy CSR matrix.")
    _assert_cell_order(tfidf, input_obs_names, "TF-IDF preprocessing")
    del prepared

    filtered = backend.filter_cCREs(
        tfidf,
        filter_idx=resources.filter_indices,
        species=species_cfg.species_id,
        verbose=False,
    )
    if filtered.n_vars != species_cfg.retained_ccres or not sp.issparse(filtered.X):
        raise RuntimeError("EpiZoo cCRE filtering produced an invalid sparse feature matrix.")
    _assert_cell_order(filtered, input_obs_names, "cCRE filtering")
    del tfidf

    sentenced = backend.generate_cell_sentences(
        filtered,
        matrix_key="X",
        obs_key="cell_indices",
        species=species_cfg.species_id,
        base_offset=4,
        species_offset=species_cfg.species_offset,
    )
    _assert_cell_order(sentenced, input_obs_names, "cell-sentence generation")
    cell_sentences = sentenced.obs["cell_indices"].to_numpy(copy=True)
    _validate_cell_sentences(cell_sentences, input_obs_names, species_cfg)
    del filtered, sentenced

    dataset = backend.InferenceCellDataset(
        cell_sentences=cell_sentences,
        species=[species_cfg.species_id] * len(input_obs_names),
        max_length=max_length,
        random_sample=bool(random_sample),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=backend.inference_collate_fn,
    )

    resolved_device = (
        _resolve_device(device) if device is not None else _model_device(model)
    )
    model.eval()
    with _numpy_random_seed(random_seed), torch.no_grad():
        embeddings = backend.extract_cell_embeddings(
            model,
            dataloader,
            device=str(resolved_device),
            use_amp=bool(use_amp),
            return_numpy=True,
            show_progress=bool(show_progress),
        )

    embeddings = np.ascontiguousarray(np.asarray(embeddings), dtype=np.float32)
    expected_shape = (len(input_obs_names), int(MODEL_CONFIG["emb_dim"]))
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"EpiZoo returned embeddings with shape {embeddings.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError("EpiZoo returned nonfinite cell embeddings.")

    model_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    metadata: dict[str, Any] = {
        "species": {"id": species_cfg.species_id, "name": species_cfg.name},
        "checkpoint": {
            "path": model._agent_checkpoint_path,
            "sha256": model._agent_checkpoint_sha256,
            "synthesized_fixed_buffers": list(
                model._agent_synthesized_checkpoint_buffers
            ),
        },
        "model_config": dict(MODEL_CONFIG),
        "resources": {
            "frequencies": {
                "path": str(resources.frequency_path),
                "sha256": resources.frequency_sha256,
            },
            "filter_indices": {
                "path": str(resources.filter_path),
                "sha256": resources.filter_sha256,
            },
        },
        "preprocessing": {
            "raw_dimension": species_cfg.raw_dimension,
            "retained_ccres": species_cfg.retained_ccres,
            "cell_number": species_cfg.cell_number,
            "scale_factor": 10_000.0,
            "base_offset": 4,
            "species_offset": species_cfg.species_offset,
        },
        "batch_size": batch_size,
        "max_length": max_length,
        "random_sample": bool(random_sample),
        "random_seed": random_seed,
        "num_workers": 0,
        "device": str(resolved_device),
        "dtype": model_dtype,
        "amp": {
            "requested": bool(use_amp),
            "enabled": bool(use_amp and resolved_device.type == "cuda"),
        },
        "versions": {
            "torch": torch.__version__,
            "epizoo": _package_version("epizoo"),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "anndata": _package_version("anndata"),
            "transformers": _package_version("transformers"),
            "flash_attn": _package_version("flash-attn"),
        },
    }
    return EpiZooEmbeddingResult(
        embeddings=embeddings,
        obs_names=input_obs_names,
        metadata=metadata,
    )


__all__ = ["EpiZooEmbeddingResult", "embed_cells", "load_model"]
