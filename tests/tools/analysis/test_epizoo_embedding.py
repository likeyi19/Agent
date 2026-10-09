from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
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


@pytest.fixture
def pinned_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "pinned.pth"
    checkpoint.write_bytes(b"reviewed checkpoint fixture")
    directory = tmp_path / "fixed-resources"
    directory.mkdir()
    frequencies = directory / tool.epizoo_backend.MOUSE_CONFIG.frequency_filename
    frequencies.write_bytes(b"reviewed frequencies fixture")
    filters = directory / tool.epizoo_backend.MOUSE_CONFIG.filter_filename
    filters.write_bytes(b"reviewed filter fixture")
    # Reviewed files precede consumption. Avoid same-clock-tick creation and
    # mutation on filesystems whose timestamps update less often than writes.
    for source in (checkpoint, frequencies, filters):
        original_stat = source.stat()
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns - 1_000_000_000))
    monkeypatch.setattr(tool.epizoo_backend, "DEFAULT_RESOURCES_DIR", directory)
    pins = {"resource_id": "reviewed-mouse",
            "checkpoint_sha256": tool.epizoo_backend._sha256_file(checkpoint),
            "frequencies_sha256": tool.epizoo_backend._sha256_file(frequencies),
            "filter_indices_sha256": tool.epizoo_backend._sha256_file(filters)}
    model = SimpleNamespace(
        _agent_checkpoint_sha256=pins["checkpoint_sha256"],
        _agent_checkpoint_source_snapshot=tool.epizoo_backend._resource_file_snapshot(checkpoint),
    )
    metadata = {"checkpoint": {"path": str(checkpoint), "sha256": pins["checkpoint_sha256"]},
                "resources": {"frequencies": {"path": str(frequencies), "sha256": pins["frequencies_sha256"]},
                              "filter_indices": {"path": str(filters), "sha256": pins["filter_indices_sha256"]}},
                "species": {"name": "mouse"}, "device": "cpu", "dtype": "float32",
                "batch_size": 4, "max_length": 8192, "random_sample": True,
                "random_seed": 0, "num_workers": 0, "amp": {"requested": True, "enabled": False}}
    calls = []

    def infer(actual_model, adata, **kwargs):
        calls.append(kwargs)
        assert actual_model is model
        return SimpleNamespace(embeddings=np.zeros((adata.n_obs, 512), dtype=np.float32),
                               obs_names=tuple(adata.obs_names), metadata=metadata)

    monkeypatch.setattr(tool, "get_cached_epizoo_model", lambda **kwargs: model)
    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", infer)
    return checkpoint, frequencies, filters, pins, model, metadata, calls


def test_pinned_execution_publishes_bounded_owner_provenance(
    input_h5ad: Path, tmp_path: Path, pinned_backend,
) -> None:
    checkpoint, _, _, pins, _, _, calls = pinned_backend
    result = tool.epizoo_embed_cells(input_h5ad, tmp_path / "pinned-output", species="mouse",
                                    checkpoint_path=checkpoint, device="cpu",
                                    expected_resource_identity=pins)
    provenance = result["resource_provenance"]
    assert json.loads(json.dumps(provenance)) == provenance
    assert provenance["expected_resource_identity"] == provenance["actual_resource_identity"] == pins
    assert provenance["inference_settings"] == {
        "batch_size": 4, "max_length": 8192, "random_sample": True, "random_seed": 0,
        "num_workers": 0, "device": "cpu", "dtype": "float32", "amp_requested": True,
        "amp_enabled": False, "show_progress": False, "overwrite": False,
    }
    assert calls[0]["resources_dir"] == tool.epizoo_backend.DEFAULT_RESOURCES_DIR
    assert "path" not in json.dumps(provenance)


def test_pinned_execution_preserves_explicit_cpu_device_index(
    input_h5ad: Path, tmp_path: Path, pinned_backend,
) -> None:
    checkpoint, _, _, pins, _, metadata, _ = pinned_backend
    metadata["device"] = "cpu:0"
    result = tool.epizoo_embed_cells(input_h5ad, tmp_path / "indexed-cpu", species="mouse",
                                    checkpoint_path=checkpoint, device="cpu:0",
                                    expected_resource_identity=pins)
    assert result["device"] == "cpu:0"
    assert result["resource_provenance"]["inference_settings"]["device"] == "cpu:0"


@pytest.mark.parametrize("field", ["checkpoint_sha256", "frequencies_sha256", "filter_indices_sha256"])
def test_changed_pinned_files_fail_before_inference(
    input_h5ad: Path, tmp_path: Path, pinned_backend, field: str,
) -> None:
    checkpoint, frequencies, filters, pins, _, _, calls = pinned_backend
    {"checkpoint_sha256": checkpoint, "frequencies_sha256": frequencies,
     "filter_indices_sha256": filters}[field].write_bytes(b"changed")
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "changed-output", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert calls == []
    assert not (tmp_path / "changed-output").exists()


def test_incompatible_warm_cache_fails_before_inference(
    input_h5ad: Path, tmp_path: Path, pinned_backend,
) -> None:
    checkpoint, _, _, pins, model, _, calls = pinned_backend
    model._agent_checkpoint_sha256 = "0" * 64
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "warm-cache", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert calls == []
    assert list((tmp_path / "warm-cache").iterdir()) == []


@pytest.mark.parametrize("proof", [None, ()])
def test_warm_cache_without_matching_stable_load_proof_fails_before_inference(
    input_h5ad: Path, tmp_path: Path, pinned_backend, proof,
) -> None:
    checkpoint, _, _, pins, model, _, calls = pinned_backend
    if proof is None:
        del model._agent_checkpoint_source_snapshot
    else:
        model._agent_checkpoint_source_snapshot = proof
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "unsupported-cache", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert calls == []
    assert list((tmp_path / "unsupported-cache").iterdir()) == []


def test_checkpoint_mutated_and_restored_since_cache_load_fails_before_inference(
    input_h5ad: Path, tmp_path: Path, pinned_backend,
) -> None:
    checkpoint, _, _, pins, _, _, calls = pinned_backend
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"different cached checkpoint bytes")
    checkpoint.write_bytes(original)
    assert tool.epizoo_backend._sha256_file(checkpoint) == pins["checkpoint_sha256"]
    with pytest.raises(tool.EpiZooResourceIdentityError):
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "restored-cache", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert calls == []
    assert list((tmp_path / "restored-cache").iterdir()) == []


@pytest.mark.parametrize("mutation", ["checkpoint", "frequencies", "filter_indices", "device", "dtype", "batch_size"])
def test_actual_backend_provenance_must_match_before_publication(
    input_h5ad: Path, tmp_path: Path, pinned_backend, mutation: str,
) -> None:
    checkpoint, _, _, pins, _, metadata, calls = pinned_backend
    if mutation == "checkpoint":
        metadata["checkpoint"]["sha256"] = "0" * 64
    elif mutation in {"frequencies", "filter_indices"}:
        metadata["resources"][mutation]["sha256"] = "0" * 64
    else:
        metadata[mutation] = {"device": "cuda:0", "dtype": "float16", "batch_size": 8}[mutation]
    with pytest.raises(tool.EpiZooResourceIdentityError):
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "wrong-metadata", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert len(calls) == 1
    assert list((tmp_path / "wrong-metadata").iterdir()) == []


def test_resources_changed_during_inference_are_not_published(
    input_h5ad: Path, tmp_path: Path, pinned_backend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, frequencies, _, pins, _, metadata, _ = pinned_backend

    def changed_inference(model, adata, **kwargs):
        frequencies.write_bytes(b"changed during inference")
        return SimpleNamespace(embeddings=np.zeros((adata.n_obs, 512), dtype=np.float32),
                               obs_names=tuple(adata.obs_names), metadata=metadata)

    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", changed_inference)
    with pytest.raises(tool.EpiZooResourceIdentityError):
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "changed-during", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert list((tmp_path / "changed-during").iterdir()) == []


@pytest.mark.parametrize("field", ["checkpoint_sha256", "frequencies_sha256", "filter_indices_sha256"])
def test_resources_mutated_and_restored_during_inference_are_not_published(
    input_h5ad: Path, tmp_path: Path, pinned_backend, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    checkpoint, frequencies, filters, pins, _, metadata, _ = pinned_backend
    path = {"checkpoint_sha256": checkpoint, "frequencies_sha256": frequencies,
            "filter_indices_sha256": filters}[field]
    original = path.read_bytes()

    def restored_inference(model, adata, **kwargs):
        path.write_bytes(b"different resource bytes consumed during inference")
        path.write_bytes(original)
        return SimpleNamespace(embeddings=np.zeros((adata.n_obs, 512), dtype=np.float32),
                               obs_names=tuple(adata.obs_names), metadata=metadata)

    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", restored_inference)
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "restored-during", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert tool.epizoo_backend._sha256_file(path) == pins[field]
    assert list((tmp_path / "restored-during").iterdir()) == []


def test_owner_resource_read_failure_has_existing_safe_pin_code(
    input_h5ad: Path, tmp_path: Path, pinned_backend, monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, _, pins, _, _, _ = pinned_backend

    def changed_owner(*args, **kwargs):
        raise tool.epizoo_backend.EpiZooResourceChangedError()

    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", changed_owner)
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "owner-changed", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert list((tmp_path / "owner-changed").iterdir()) == []


@pytest.mark.parametrize("resource", ["frequencies", "filter_indices"])
def test_actual_auxiliary_read_digest_rejects_same_stat_restoration(
    input_h5ad: Path, tmp_path: Path, pinned_backend, monkeypatch: pytest.MonkeyPatch, resource: str,
) -> None:
    checkpoint, frequency, filters, pins, _, metadata, _ = pinned_backend
    config = replace(tool.epizoo_backend.MOUSE_CONFIG, raw_dimension=4, retained_ccres=2)
    np.save(frequency, np.array([1.0, 2.0, 3.0, 4.0]))
    pd.DataFrame({"cCRE": ["peak-1", "peak-2"], "idx": [0, 1]}).to_csv(filters)
    pins["frequencies_sha256"] = tool.epizoo_backend._sha256_file(frequency)
    pins["filter_indices_sha256"] = tool.epizoo_backend._sha256_file(filters)
    path = frequency if resource == "frequencies" else filters
    field = "frequencies_sha256" if resource == "frequencies" else "filter_indices_sha256"
    original_bytes = path.read_bytes()
    if resource == "frequencies":
        alternate = BytesIO()
        np.save(alternate, np.array([40.0, 30.0, 20.0, 10.0]))
        alternate_bytes = alternate.getvalue()
    else:
        alternate_bytes = pd.DataFrame({"cCRE": ["peak-3", "peak-4"], "idx": [0, 1]}).to_csv().encode()

    original_snapshot = tool.epizoo_backend._resource_file_snapshot
    frozen_snapshot = original_snapshot(path)
    original_read = Path.read_bytes
    consumed = {}

    def same_tick_snapshot(source):
        # This host can report identical stat timestamps for immediate writes.
        # Model that observed granularity deterministically; consumed-byte
        # digests must enforce the pins even when metadata changes are hidden.
        return frozen_snapshot if source == path else original_snapshot(source)

    def swapped_read(source):
        if source != path:
            return original_read(source)
        path.write_bytes(alternate_bytes)
        values = original_read(source)
        path.write_bytes(original_bytes)
        return values

    def actual_resource_read(model, adata, **kwargs):
        resources = tool.epizoo_backend._load_resources(config, kwargs["resources_dir"])
        consumed["digest"] = (resources.frequency_sha256 if resource == "frequencies"
                              else resources.filter_sha256)
        consumed["values"] = (resources.frequencies.tolist() if resource == "frequencies"
                              else resources.retained_names.tolist())
        actual_metadata = dict(metadata, resources={
            "frequencies": {"path": str(frequency), "sha256": resources.frequency_sha256},
            "filter_indices": {"path": str(filters), "sha256": resources.filter_sha256},
        })
        return SimpleNamespace(embeddings=np.zeros((adata.n_obs, 512), dtype=np.float32),
                               obs_names=tuple(adata.obs_names), metadata=actual_metadata)

    monkeypatch.setattr(tool.epizoo_backend, "_resource_file_snapshot", same_tick_snapshot)
    monkeypatch.setattr(Path, "read_bytes", swapped_read)
    monkeypatch.setattr(tool.epizoo_backend, "embed_cells", actual_resource_read)
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.epizoo_embed_cells(input_h5ad, tmp_path / "same-stat-restored", species="mouse",
                                checkpoint_path=checkpoint, device="cpu", expected_resource_identity=pins)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_MISMATCH"
    assert consumed["digest"] == sha256(alternate_bytes).hexdigest()
    assert consumed["digest"] != pins[field]
    assert consumed["values"] == ([40.0, 30.0, 20.0, 10.0] if resource == "frequencies"
                                  else ["peak-3", "peak-4"])
    assert original_read(path) == original_bytes
    assert list((tmp_path / "same-stat-restored").iterdir()) == []


@pytest.mark.parametrize("bad", [None, {}, {"resource_id": "unsafe/path"},
    {"resource_id": "valid", "checkpoint_sha256": "A" * 64, "frequencies_sha256": "b" * 64,
     "filter_indices_sha256": "c" * 64},
    {"resource_id": "valid", "checkpoint_sha256": "a" * 64, "frequencies_sha256": "b" * 64,
     "filter_indices_sha256": "c" * 64, "path": "/private"}])
def test_expected_resource_identity_has_exact_bounded_shape(bad: object) -> None:
    with pytest.raises(tool.EpiZooResourceIdentityError) as caught:
        tool.validate_expected_resource_identity(bad)
    assert caught.value.code == "EPIZOO_RESOURCE_IDENTITY_INVALID"
