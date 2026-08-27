from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from agent.tools.analysis import epizoo_embedding as tool
from agent.tools.models import epizoo_cache


@pytest.fixture
def input_h5ad(tmp_path: Path) -> Path:
    adata = ad.AnnData(
        sp.csr_matrix(
            np.array(
                [
                    [1, 0, 2, 0],
                    [0, 3, 0, 1],
                    [4, 0, 5, 6],
                ],
                dtype=np.float32,
            )
        )
    )
    adata.obs_names = ["cell-a", "cell-b", "cell-c"]
    adata.var_names = ["peak-1", "peak-2", "peak-3", "peak-4"]
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture
def mocked_backend(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}
    embeddings = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)

    def fake_get_cached_model(*, checkpoint_path, device):
        calls["load_model"] = {
            "checkpoint_path": checkpoint_path,
            "device": device,
        }
        return object()

    def fake_embed_cells(model, adata, **kwargs):
        calls["embed_cells"] = kwargs
        assert sp.issparse(adata.X)
        return SimpleNamespace(
            embeddings=embeddings,
            obs_names=tuple(str(name) for name in adata.obs_names),
            metadata={
                "species": {"id": 1, "name": "mouse"},
                "checkpoint": {"path": "/models/validated_epizoo.pth"},
                "device": "cpu",
            },
        )

    monkeypatch.setattr(tool, "get_cached_epizoo_model", fake_get_cached_model)
    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", fake_embed_cells)
    return calls, embeddings


def test_success_creates_ordered_artifacts_and_lightweight_json_result(
    input_h5ad: Path, tmp_path: Path, mocked_backend
) -> None:
    calls, expected_embeddings = mocked_backend
    output_dir = tmp_path / "artifacts"

    result = tool.epizoo_embed_cells(
        input_h5ad,
        output_dir,
        species="mouse",
        checkpoint_path="/models/validated_epizoo.pth",
        device="cpu",
    )

    assert json.loads(json.dumps(result)) == result
    assert "embeddings" not in result
    assert all(not isinstance(value, np.ndarray) for value in result.values())
    assert result["status"] == "success"
    assert result["n_cells"] == 3
    assert result["embedding_dim"] == 512
    assert result["embedding_dtype"] == "float32"
    assert result["finite"] is True
    assert result["cell_order_preserved"] is True
    assert result["backend"] == "EpiZoo"
    assert result["species"] == "mouse"
    assert result["checkpoint_path"] == "/models/validated_epizoo.pth"
    assert result["device"] == "cpu"

    saved = np.load(result["embedding_path"], mmap_mode="r", allow_pickle=False)
    assert saved.shape == (3, 512)
    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, expected_embeddings)
    assert Path(result["cell_ids_path"]).read_text(encoding="utf-8").splitlines() == [
        "cell-a",
        "cell-b",
        "cell-c",
    ]
    assert calls["embed_cells"] == {
        "species": "mouse",
        "device": "cpu",
        "batch_size": 4,
        "max_length": 8192,
        "random_sample": True,
        "random_seed": 0,
        "use_amp": True,
        "num_workers": 0,
        "show_progress": False,
    }


@pytest.mark.parametrize(
    ("bad_embeddings", "message"),
    [
        (np.full((3, 512), np.nan, dtype=np.float32), "non-finite"),
        (np.zeros((2, 512), dtype=np.float32), "2 embedding rows"),
    ],
)
def test_invalid_backend_output_is_rejected_without_artifacts(
    input_h5ad: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_embeddings: np.ndarray,
    message: str,
) -> None:
    monkeypatch.setattr(tool, "get_cached_epizoo_model", lambda **kwargs: object())
    monkeypatch.setattr(
        tool.epizoo_backend,
        "embed_cells",
        lambda model, adata, **kwargs: SimpleNamespace(
            embeddings=bad_embeddings,
            obs_names=tuple(adata.obs_names),
            metadata={},
        ),
    )
    output_dir = tmp_path / "failed"

    with pytest.raises(RuntimeError, match=message):
        tool.epizoo_embed_cells(
            input_h5ad, output_dir, species="mouse", device="cpu"
        )

    assert list(output_dir.glob("*.npy")) == []
    assert list(output_dir.glob("*.txt")) == []


def test_cell_order_mismatch_is_rejected_without_artifacts(
    input_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "get_cached_epizoo_model", lambda **kwargs: object())
    monkeypatch.setattr(
        tool.epizoo_backend,
        "embed_cells",
        lambda model, adata, **kwargs: SimpleNamespace(
            embeddings=np.zeros((3, 512), dtype=np.float32),
            obs_names=tuple(reversed(adata.obs_names)),
            metadata={},
        ),
    )
    output_dir = tmp_path / "wrong-order"

    with pytest.raises(RuntimeError, match="cell identifiers"):
        tool.epizoo_embed_cells(
            input_h5ad, output_dir, species="mouse", device="cpu"
        )

    assert list(output_dir.iterdir()) == []


def test_invalid_input_path_is_rejected_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_loaded = False

    def fake_load_model(**kwargs):
        nonlocal model_loaded
        model_loaded = True

    monkeypatch.setattr(tool, "get_cached_epizoo_model", fake_load_model)

    with pytest.raises(FileNotFoundError, match="AnnData file not found"):
        tool.epizoo_embed_cells(
            tmp_path / "missing.h5ad",
            tmp_path / "output",
            species="mouse",
            device="cpu",
        )

    assert model_loaded is False
    assert not (tmp_path / "output").exists()


def test_existing_outputs_require_explicit_overwrite(
    input_h5ad: Path, tmp_path: Path, mocked_backend
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    embedding_path, cell_ids_path = tool._artifact_paths(input_h5ad, output_dir)
    embedding_path.write_bytes(b"old embedding")
    cell_ids_path.write_text("old cell\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite=True"):
        tool.epizoo_embed_cells(
            input_h5ad, output_dir, species="mouse", device="cpu"
        )

    assert embedding_path.read_bytes() == b"old embedding"
    assert cell_ids_path.read_text(encoding="utf-8") == "old cell\n"

    result = tool.epizoo_embed_cells(
        input_h5ad,
        output_dir,
        species="mouse",
        device="cpu",
        overwrite=True,
    )
    assert np.load(result["embedding_path"], allow_pickle=False).shape == (3, 512)
    assert Path(result["cell_ids_path"]).read_text(encoding="utf-8").splitlines() == [
        "cell-a",
        "cell-b",
        "cell-c",
    ]


def test_failed_inference_leaves_no_completed_artifacts(
    input_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool, "get_cached_epizoo_model", lambda **kwargs: object())

    def fail_inference(*args, **kwargs):
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", fail_inference)
    output_dir = tmp_path / "failed-inference"

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        tool.epizoo_embed_cells(
            input_h5ad, output_dir, species="mouse", device="cpu"
        )

    assert list(output_dir.iterdir()) == []


def test_tool_reuses_cached_model_across_request_paths(
    input_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second_input = tmp_path / "other-cells.h5ad"
    copyfile(input_h5ad, second_input)
    checkpoint = tmp_path / "epizoo.pth"
    cached_model = object()
    load_count = 0
    inference_model_ids: list[int] = []

    def fake_load_model(**kwargs):
        nonlocal load_count
        load_count += 1
        return cached_model

    def fake_embed_cells(model, adata, **kwargs):
        inference_model_ids.append(id(model))
        return SimpleNamespace(
            embeddings=np.zeros((adata.n_obs, 512), dtype=np.float32),
            obs_names=tuple(adata.obs_names),
            metadata={
                "species": {"id": 1, "name": "mouse"},
                "checkpoint": {"path": str(checkpoint.resolve())},
                "device": "cpu",
            },
        )

    epizoo_cache.clear_epizoo_backend_cache()
    monkeypatch.setattr(
        epizoo_cache.epizoo_backend, "load_model", fake_load_model
    )
    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", fake_embed_cells)
    try:
        first = tool.epizoo_embed_cells(
            input_h5ad,
            tmp_path / "first-output",
            species="mouse",
            checkpoint_path=checkpoint,
            device="cpu",
        )
        second = tool.epizoo_embed_cells(
            second_input,
            tmp_path / "second-output",
            species="mouse",
            checkpoint_path=checkpoint,
            device="cpu",
        )
    finally:
        epizoo_cache.clear_epizoo_backend_cache()

    assert first["status"] == second["status"] == "success"
    assert load_count == 1
    assert inference_model_ids == [id(cached_model), id(cached_model)]
