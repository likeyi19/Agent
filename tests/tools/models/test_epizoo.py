from __future__ import annotations

from dataclasses import replace
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from agent.tools.models import epizoo as wrapper


FANG_H5AD = Path(
    "/home/likeyi/program/EpiZoo/data/Fang2021_downsampled_2000_cells.h5ad"
)
CHECKPOINT = Path(
    "/home/likeyi/program/model_checkpoints/EpiZoo/pretrained_EpiZoo.pth"
)


def test_fixed_checkpoint_configuration() -> None:
    assert dict(wrapper.MODEL_CONFIG) == {
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
    }
    signature = inspect.signature(wrapper.load_model)
    assert "strict" not in signature.parameters
    assert signature.parameters["checkpoint_path"].default == wrapper.DEFAULT_CHECKPOINT_PATH


@pytest.mark.parametrize(
    ("value", "name", "species_id"),
    [("human", "human", 0), ("MOUSE", "mouse", 1), (0, "human", 0), (1, "mouse", 1)],
)
def test_species_normalization(value, name: str, species_id: int) -> None:
    config = wrapper._normalize_species(value)
    assert config.name == name
    assert config.species_id == species_id


@pytest.mark.parametrize("value", [True, False, "rat", 2, None])
def test_species_normalization_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="species"):
        wrapper._normalize_species(value)


@pytest.mark.parametrize(
    ("config", "frequency_shape", "retained", "first_idx", "last_idx"),
    [
        (wrapper.HUMAN_CONFIG, (1_355_445,), 700_460, 0, 1_355_442),
        (wrapper.MOUSE_CONFIG, (1_341_077,), 814_020, 3, 1_341_062),
    ],
)
def test_species_resources(
    config,
    frequency_shape: tuple[int, ...],
    retained: int,
    first_idx: int,
    last_idx: int,
) -> None:
    resources = wrapper._load_resources(config, wrapper.DEFAULT_RESOURCES_DIR)
    assert resources.frequencies.shape == frequency_shape
    assert resources.filter_indices.shape == (retained,)
    assert resources.retained_names.shape == (retained,)
    assert resources.filter_indices[0] == first_idx
    assert resources.filter_indices[-1] == last_idx
    assert np.all(np.diff(resources.filter_indices) > 0)
    assert len(resources.frequency_sha256) == 64
    assert len(resources.filter_sha256) == 64


def test_config_validation_rejects_mismatch() -> None:
    valid = SimpleNamespace(**dict(wrapper.MODEL_CONFIG))
    wrapper._validate_config_values(valid)

    invalid_values = dict(wrapper.MODEL_CONFIG)
    invalid_values["num_layers"] = 18
    with pytest.raises(ValueError, match="num_layers=18"):
        wrapper._validate_config_values(SimpleNamespace(**invalid_values))


def _synthetic_compatible_state_dict() -> dict[str, torch.Tensor]:
    def meta(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device="meta", dtype=torch.float16)

    state: dict[str, torch.Tensor] = {
        "ccre_emb.weight": meta((1_514_484, 512)),
        "seq_emb.weight": meta((1_514_484, 512)),
        "rank_emb.weight": meta((8192, 512)),
        "cca_head.net.0.weight": meta((128, 1024)),
        "cca_head.net.4.weight": meta((1, 128)),
        "signal_decoder.decoders.human.weight": meta((700_460, 512)),
        "signal_decoder.decoders.mouse.weight": meta((814_020, 512)),
    }
    for layer in range(30):
        state[f"encoder.layers.{layer}.mixer.Wqkv.weight"] = meta((1536, 512))
        state[f"encoder.layers.{layer}.mlp.gate.weight"] = meta((4, 512))
        for expert in range(4):
            state[f"encoder.layers.{layer}.mlp.experts.{expert}.0.weight"] = meta(
                (2048, 512)
            )
            state[f"encoder.layers.{layer}.mlp.experts.{expert}.2.weight"] = meta(
                (512, 2048)
            )
    return state


def test_checkpoint_shape_validation_without_allocating_weights() -> None:
    state = _synthetic_compatible_state_dict()
    wrapper._validate_checkpoint_state_dict(state)

    state["ccre_emb.weight"] = torch.empty((10, 512), device="meta")
    with pytest.raises(ValueError, match="ccre_emb.weight"):
        wrapper._validate_checkpoint_state_dict(state)


def test_fixed_loss_buffers_are_explicitly_synthesized() -> None:
    state = _synthetic_compatible_state_dict()
    cfg = SimpleNamespace(cca_pos_weight=1.0, signal_pos_weight=100.0)
    added = wrapper._add_fixed_loss_buffers(state, cfg)
    assert added == ("cca_loss_fn.pos_weight", "signal_loss_fn.pos_weight")
    assert state["cca_loss_fn.pos_weight"].item() == 1.0
    assert state["signal_loss_fn.pos_weight"].item() == 100.0
    assert wrapper._add_fixed_loss_buffers(state, cfg) == ()


def test_input_validation_rejects_non_anndata() -> None:
    with pytest.raises(TypeError, match="AnnData"):
        wrapper._validate_adata(object(), wrapper.MOUSE_CONFIG)


def test_input_validation_rejects_empty_anndata() -> None:
    empty = ad.AnnData(sp.csr_matrix((0, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="at least one cell"):
        wrapper._validate_adata(empty, wrapper.MOUSE_CONFIG)


def test_input_validation_rejects_duplicate_cell_names() -> None:
    duplicate = ad.AnnData(
        sp.csr_matrix([[1], [1]], dtype=np.float32),
        obs={"cell": ["a", "b"]},
    )
    duplicate.obs_names = ["same", "same"]
    with pytest.raises(ValueError, match="unique"):
        wrapper._validate_adata(duplicate, wrapper.MOUSE_CONFIG)


def test_input_validation_rejects_dense_matrix() -> None:
    dense = ad.AnnData(np.ones((1, 1), dtype=np.float32))
    with pytest.raises(TypeError, match="sparse"):
        wrapper._validate_adata(dense, wrapper.MOUSE_CONFIG)


def test_input_validation_rejects_wrong_species_dimension() -> None:
    wrong = ad.AnnData(sp.csr_matrix([[1]], dtype=np.float32))
    with pytest.raises(ValueError, match="1,341,077"):
        wrapper._validate_adata(wrong, wrapper.MOUSE_CONFIG)


@pytest.fixture(scope="module")
def mouse_subset() -> ad.AnnData:
    if not FANG_H5AD.is_file():
        pytest.skip(f"Smoke dataset not found: {FANG_H5AD}")
    full = ad.read_h5ad(FANG_H5AD)
    subset = full[:2].copy()
    assert sp.isspmatrix_csr(subset.X)
    return subset


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("negative", "nonnegative"),
        ("fractional", "count-like"),
        ("zero_cell", "Zero-count"),
    ],
)
def test_sparse_value_validation(mouse_subset, mutation: str, message: str) -> None:
    invalid = mouse_subset.copy()
    if mutation == "negative":
        invalid.X.data[0] = -1
    elif mutation == "fractional":
        invalid.X.data[0] = 0.5
    else:
        invalid.X = sp.vstack(
            [sp.csr_matrix((1, invalid.n_vars), dtype=np.float32), invalid.X[1]],
            format="csr",
        )

    with pytest.raises(ValueError, match=message):
        wrapper._validate_adata(invalid, wrapper.MOUSE_CONFIG)


def test_retained_feature_order_validation_rejects_mismatch(mouse_subset) -> None:
    resources = wrapper._load_resources(
        wrapper.MOUSE_CONFIG, wrapper.DEFAULT_RESOURCES_DIR
    )
    invalid = mouse_subset.copy()
    names = invalid.var_names.to_numpy(copy=True)
    names[resources.filter_indices[0]] = "not-the-reference-ccre"
    invalid.var_names = names
    with pytest.raises(ValueError, match="feature order"):
        wrapper._validate_retained_feature_order(invalid, resources)


def test_generated_token_range_validation() -> None:
    wrapper._validate_cell_sentences(
        [[wrapper.MOUSE_CONFIG.token_min, wrapper.MOUSE_CONFIG.token_max]],
        ("cell",),
        wrapper.MOUSE_CONFIG,
    )
    with pytest.raises(RuntimeError, match="outside"):
        wrapper._validate_cell_sentences(
            [[wrapper.MOUSE_CONFIG.token_min - 1]],
            ("cell",),
            wrapper.MOUSE_CONFIG,
        )
    with pytest.raises(ValueError, match="no retained"):
        wrapper._validate_cell_sentences([[]], ("cell",), wrapper.MOUSE_CONFIG)


class _FakeInferenceDataset(Dataset):
    def __init__(self, cell_sentences, species, max_length=8192, random_sample=True):
        self.sentences = [np.asarray(value, dtype=np.int64) for value in cell_sentences]
        self.species = list(species)
        self.max_length = max_length
        self.random_sample = random_sample

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int):
        tokens = self.sentences[index]
        limit = self.max_length - 2
        if len(tokens) > limit:
            if self.random_sample:
                chosen = np.sort(np.random.choice(len(tokens), limit, replace=False))
                tokens = tokens[chosen]
            else:
                tokens = tokens[:limit]
        input_ids = torch.tensor([1, *tokens.tolist(), 2], dtype=torch.long)
        return input_ids, self.species[index]


def _fake_collate(batch):
    ids, species = zip(*batch)
    return pad_sequence(ids, batch_first=True, padding_value=0), list(species)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self._agent_checkpoint_path = str(CHECKPOINT)
        self._agent_checkpoint_sha256 = "fake-checkpoint-hash"
        self._agent_synthesized_checkpoint_buffers = (
            "cca_loss_fn.pos_weight",
            "signal_loss_fn.pos_weight",
        )


def _fake_backend(
    call_order: list[str], output_mode: str = "valid"
) -> wrapper._EpiZooBackend:
    def compute_tfidf(adata, df, **kwargs):
        call_order.append("compute_tfidf")
        assert sp.isspmatrix_csr(adata.X)
        assert kwargs["cell_number"] == 12_500_000
        return adata.copy()

    def filter_ccres(adata, filter_idx, species, verbose):
        call_order.append("filter_cCREs")
        assert species == 1
        return adata[:, filter_idx].copy()

    def generate_sentences(
        adata, matrix_key, obs_key, species, base_offset, species_offset
    ):
        call_order.append("generate_cell_sentences")
        result = adata.copy()
        matrix = result.X.tocsr()
        sentences = []
        for row_index in range(matrix.shape[0]):
            row = matrix.getrow(row_index)
            order = row.indices[np.argsort(-row.data)]
            sentences.append((order + base_offset + species_offset).tolist())
        result.obs[obs_key] = sentences
        result.obs["species"] = species
        return result

    def extract(model, dataloader, **kwargs):
        call_order.append("extract_cell_embeddings")
        rows = []
        for input_ids, _species in dataloader:
            values = np.zeros((input_ids.shape[0], 512), dtype=np.float32)
            width = min(512, input_ids.shape[1])
            values[:, :width] = input_ids[:, :width].numpy()
            rows.append(values)
        output = np.concatenate(rows, axis=0)
        if output_mode == "bad_shape":
            return output[:, :511]
        if output_mode == "nonfinite":
            output[0, 0] = np.nan
        return output

    return wrapper._EpiZooBackend(
        compute_tfidf=compute_tfidf,
        filter_cCREs=filter_ccres,
        generate_cell_sentences=generate_sentences,
        InferenceCellDataset=_FakeInferenceDataset,
        inference_collate_fn=_fake_collate,
        extract_cell_embeddings=extract,
        EpiZoo=None,
        EpiZooConfig=None,
    )


def test_small_sparse_pipeline_preserves_order_and_is_deterministic(
    monkeypatch, mouse_subset
) -> None:
    call_order: list[str] = []
    monkeypatch.setattr(wrapper, "_validate_loaded_model", lambda model: None)
    monkeypatch.setattr(wrapper, "_get_backend", lambda: _fake_backend(call_order))
    model = _FakeModel()
    original_names = tuple(mouse_subset.obs_names)
    original_nnz = mouse_subset.X.nnz

    first = wrapper.embed_cells(
        model,
        mouse_subset,
        species="mouse",
        batch_size=1,
        max_length=16,
        random_sample=True,
        random_seed=0,
        device="cpu",
        show_progress=False,
    )
    second = wrapper.embed_cells(
        model,
        mouse_subset,
        species=1,
        batch_size=1,
        max_length=16,
        random_sample=True,
        random_seed=0,
        device="cpu",
        show_progress=False,
    )
    different_seed = wrapper.embed_cells(
        model,
        mouse_subset,
        species="mouse",
        batch_size=1,
        max_length=16,
        random_sample=True,
        random_seed=1,
        device="cpu",
        show_progress=False,
    )

    assert first.embeddings.shape == (2, 512)
    assert first.embeddings.dtype == np.float32
    assert first.embeddings.flags.c_contiguous
    assert first.obs_names == original_names
    np.testing.assert_array_equal(first.embeddings, second.embeddings)
    assert not np.array_equal(first.embeddings, different_seed.embeddings)
    assert tuple(mouse_subset.obs_names) == original_names
    assert mouse_subset.X.nnz == original_nnz
    assert "cell_indices" not in mouse_subset.obs
    assert first.metadata["species"] == {"id": 1, "name": "mouse"}
    assert first.metadata["random_sample"] is True
    assert first.metadata["random_seed"] == 0
    assert first.metadata["amp"]["enabled"] is False
    assert call_order[:4] == [
        "compute_tfidf",
        "filter_cCREs",
        "generate_cell_sentences",
        "extract_cell_embeddings",
    ]


@pytest.mark.parametrize(
    ("output_mode", "message"),
    [("bad_shape", "shape"), ("nonfinite", "nonfinite")],
)
def test_embedding_output_validation(
    monkeypatch, mouse_subset, output_mode: str, message: str
) -> None:
    monkeypatch.setattr(wrapper, "_validate_loaded_model", lambda model: None)
    monkeypatch.setattr(
        wrapper, "_get_backend", lambda: _fake_backend([], output_mode=output_mode)
    )
    with pytest.raises(RuntimeError, match=message):
        wrapper.embed_cells(
            _FakeModel(),
            mouse_subset,
            species="mouse",
            batch_size=1,
            max_length=16,
            random_seed=0,
            device="cpu",
            show_progress=False,
        )


@pytest.mark.skipif(
    os.environ.get("RUN_EPIZOO_INTEGRATION") != "1",
    reason="Set RUN_EPIZOO_INTEGRATION=1 in the validated host runtime.",
)
def test_actual_epizoo_sparse_preprocessing_on_small_subset(
    monkeypatch, mouse_subset
) -> None:
    backend = wrapper._get_backend()

    def fake_extract(model, dataloader, **kwargs):
        count = sum(batch[0].shape[0] for batch in dataloader)
        return np.zeros((count, 512), dtype=np.float32)

    monkeypatch.setattr(wrapper, "_validate_loaded_model", lambda model: None)
    monkeypatch.setattr(
        wrapper,
        "_get_backend",
        lambda: replace(backend, extract_cell_embeddings=fake_extract),
    )
    result = wrapper.embed_cells(
        _FakeModel(),
        mouse_subset,
        species="mouse",
        batch_size=1,
        max_length=64,
        random_seed=0,
        device="cpu",
        show_progress=False,
    )
    assert result.embeddings.shape == (2, 512)
    assert result.obs_names == tuple(mouse_subset.obs_names)


@pytest.mark.skipif(
    os.environ.get("RUN_EPIZOO_CHECKPOINT_TEST") != "1",
    reason="Set RUN_EPIZOO_CHECKPOINT_TEST=1 for the 2.6B-parameter CPU load test.",
)
def test_strict_checkpoint_loading_on_cpu() -> None:
    model = wrapper.load_model(CHECKPOINT, device="cpu", dtype=torch.float16)
    assert model._agent_checkpoint_missing_keys == ()
    assert model._agent_checkpoint_unexpected_keys == ()
    assert model._agent_checkpoint_validated is True
    assert model.training is False
    wrapper._validate_config_values(model.cfg)
    assert model.ccre_emb.weight.shape == (1_514_484, 512)
    assert len(model.encoder.layers) == 30


@pytest.mark.skipif(
    os.environ.get("RUN_EPIZOO_PARITY_TEST") != "1",
    reason="Set RUN_EPIZOO_PARITY_TEST=1 for the CUDA scientific parity test.",
)
def test_cuda_wrapper_matches_manual_epizoo_pipeline(monkeypatch) -> None:
    subset_indices = [1055, 22, 667, 145]
    expected_retained_counts = [21_938, 21_563, 19_041, 18_610]
    batch_size = 4
    max_length = 8192
    random_seed = 0
    device = "cuda:0"
    atol = 1e-5
    rtol = 1e-5

    full = ad.read_h5ad(FANG_H5AD)
    subset = full[subset_indices].copy()
    expected_obs_names = tuple(str(name) for name in subset.obs_names)

    backend = wrapper._get_backend()
    wrapper_input_ids: list[torch.Tensor] = []

    def capturing_wrapper_collate(batch):
        collated = backend.inference_collate_fn(batch)
        wrapper_input_ids.append(collated[0].detach().cpu().clone())
        return collated

    instrumented_backend = replace(
        backend,
        inference_collate_fn=capturing_wrapper_collate,
    )
    monkeypatch.setattr(wrapper, "_get_backend", lambda: instrumented_backend)

    model = wrapper.load_model(
        CHECKPOINT,
        device=device,
        dtype=torch.float32,
    )
    wrapper_first = wrapper.embed_cells(
        model,
        subset,
        species="mouse",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        random_sample=True,
        random_seed=random_seed,
        use_amp=True,
        show_progress=False,
    )
    wrapper_second = wrapper.embed_cells(
        model,
        subset,
        species="mouse",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        random_sample=True,
        random_seed=random_seed,
        use_amp=True,
        show_progress=False,
    )

    frequencies = np.load(
        wrapper.DEFAULT_RESOURCES_DIR / "cCRE_frequencies_mouse.npy",
        allow_pickle=False,
    )
    filter_indices = pd.read_csv(
        wrapper.DEFAULT_RESOURCES_DIR / "cCRE_filter_idx_mouse.csv",
        index_col=0,
    )["idx"].to_numpy()
    manual = backend.compute_tfidf(
        subset,
        frequencies,
        cell_number=12_500_000,
        scale_factor=10_000.0,
        dtype=np.float32,
        store="X",
        verbose=False,
    )
    assert tuple(str(name) for name in manual.obs_names) == expected_obs_names
    manual = backend.filter_cCREs(
        manual,
        filter_idx=filter_indices,
        species=1,
        verbose=False,
    )
    retained_counts = manual.X.getnnz(axis=1).tolist()
    assert retained_counts == expected_retained_counts
    assert tuple(str(name) for name in manual.obs_names) == expected_obs_names
    manual = backend.generate_cell_sentences(
        manual,
        matrix_key="X",
        obs_key="cell_indices",
        species=1,
        base_offset=4,
        species_offset=700_460,
    )
    assert tuple(str(name) for name in manual.obs_names) == expected_obs_names

    manual_dataset = backend.InferenceCellDataset(
        cell_sentences=manual.obs["cell_indices"].to_numpy(),
        species=[1] * manual.n_obs,
        max_length=max_length,
        random_sample=True,
    )
    manual_input_ids: list[torch.Tensor] = []

    def capturing_manual_collate(batch):
        collated = backend.inference_collate_fn(batch)
        manual_input_ids.append(collated[0].detach().cpu().clone())
        return collated

    manual_dataloader = DataLoader(
        manual_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=capturing_manual_collate,
    )
    random_state = np.random.get_state()
    np.random.seed(random_seed)
    try:
        manual_embeddings = backend.extract_cell_embeddings(
            model,
            manual_dataloader,
            device=device,
            use_amp=True,
            return_numpy=True,
            show_progress=False,
        )
    finally:
        np.random.set_state(random_state)

    manual_embeddings = np.ascontiguousarray(manual_embeddings, dtype=np.float32)
    assert len(wrapper_input_ids) == 2
    assert len(manual_input_ids) == 1
    assert torch.equal(wrapper_input_ids[0], wrapper_input_ids[1])
    assert torch.equal(wrapper_input_ids[0], manual_input_ids[0])

    difference = np.abs(wrapper_first.embeddings - manual_embeddings)
    max_absolute_difference = float(difference.max())
    mean_absolute_difference = float(difference.mean())
    allclose = bool(
        np.allclose(
            wrapper_first.embeddings,
            manual_embeddings,
            atol=atol,
            rtol=rtol,
        )
    )
    wrapper_reproducible = bool(
        np.array_equal(wrapper_first.embeddings, wrapper_second.embeddings)
    )

    print("parity_subset_indices", subset_indices)
    print("parity_subset_obs_names", expected_obs_names)
    print("parity_retained_counts", retained_counts)
    print("wrapper_shape", wrapper_first.embeddings.shape)
    print("manual_shape", manual_embeddings.shape)
    print("wrapper_dtype", wrapper_first.embeddings.dtype)
    print("manual_dtype", manual_embeddings.dtype)
    print("wrapper_finite", bool(np.all(np.isfinite(wrapper_first.embeddings))))
    print("manual_finite", bool(np.all(np.isfinite(manual_embeddings))))
    print("cell_order_preserved", wrapper_first.obs_names == expected_obs_names)
    print("max_absolute_difference", max_absolute_difference)
    print("mean_absolute_difference", mean_absolute_difference)
    print("allclose", allclose, "atol", atol, "rtol", rtol)
    print("wrapper_reproducible", wrapper_reproducible)

    assert wrapper_first.embeddings.shape == (4, 512)
    assert manual_embeddings.shape == (4, 512)
    assert wrapper_first.embeddings.dtype == np.float32
    assert manual_embeddings.dtype == np.float32
    assert np.all(np.isfinite(wrapper_first.embeddings))
    assert np.all(np.isfinite(manual_embeddings))
    assert wrapper_first.obs_names == expected_obs_names
    assert allclose
    assert wrapper_reproducible
