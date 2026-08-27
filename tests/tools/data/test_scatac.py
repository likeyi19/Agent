from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from agent.tools.data import inspect_scATAC


@pytest.fixture
def sparse_h5ad(tmp_path: Path) -> Path:
    matrix = sp.csr_matrix(
        np.array(
            [
                [1, 0, 2, 0],
                [0, 3, 0, 0],
                [4, 0, 5, 6],
            ],
            dtype=np.int32,
        )
    )
    adata = ad.AnnData(
        X=matrix,
        obs={"batch": ["a", "a", "b"], "quality": [0.9, 0.8, 0.7]},
        var={"chromosome": ["chr1", "chr1", "chr2", "chr2"]},
    )
    adata.obs_names = ["cell-a", "cell-b", "cell-c"]
    adata.var_names = ["peak-1", "peak-2", "peak-3", "peak-4"]
    path = tmp_path / "sparse.h5ad"
    adata.write_h5ad(path)
    return path


def test_inspects_sparse_anndata_shape_and_metadata(sparse_h5ad: Path) -> None:
    result = inspect_scATAC(sparse_h5ad)

    assert result["input_path"] == str(sparse_h5ad.resolve())
    assert result["n_cells"] == 3
    assert result["n_features"] == 4
    assert result["x_is_sparse"] is True
    assert result["x_storage_type"].endswith("._CSRDataset")
    assert result["x_dtype"] == "int32"
    assert result["nnz"] == 6
    assert result["density"] == pytest.approx(0.5)
    assert result["obs_columns"] == ["batch", "quality"]
    assert result["var_columns"] == ["chromosome"]
    assert result["obs_names_sample"] == ["cell-a", "cell-b", "cell-c"]
    assert result["var_names_sample"] == [
        "peak-1",
        "peak-2",
        "peak-3",
        "peak-4",
    ]


def test_result_is_json_serializable(sparse_h5ad: Path) -> None:
    result = inspect_scATAC(sparse_h5ad)
    assert json.loads(json.dumps(result)) == result


def test_inspection_uses_backed_read_and_never_densifies(
    sparse_h5ad: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent.tools.data.scatac as module

    original_read_h5ad = module.ad.read_h5ad
    backed_modes: list[str | None] = []

    def tracked_read_h5ad(path, *, backed=None):
        backed_modes.append(backed)
        return original_read_h5ad(path, backed=backed)

    def fail_dense_conversion(*args, **kwargs):
        raise AssertionError("Inspection attempted to densify a sparse matrix.")

    monkeypatch.setattr(module.ad, "read_h5ad", tracked_read_h5ad)
    monkeypatch.setattr(sp.csr_matrix, "toarray", fail_dense_conversion)
    monkeypatch.setattr(sp.csc_matrix, "toarray", fail_dense_conversion)

    result = inspect_scATAC(sparse_h5ad)

    assert backed_modes == ["r"]
    assert result["x_is_sparse"] is True
    assert result["nnz"] == 6


def test_dense_x_is_not_scanned_for_nnz_or_density(tmp_path: Path) -> None:
    path = tmp_path / "dense.h5ad"
    ad.AnnData(np.ones((2, 3), dtype=np.float32)).write_h5ad(path)

    result = inspect_scATAC(path)

    assert result["x_is_sparse"] is False
    assert result["x_dtype"] == "float32"
    assert result["nnz"] is None
    assert result["density"] is None


def test_nonexistent_path_raises_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5ad"
    with pytest.raises(FileNotFoundError, match="AnnData file not found"):
        inspect_scATAC(missing)


def test_invalid_path_and_malformed_file_raise_actionable_errors(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="path.*string or pathlib.Path"):
        inspect_scATAC(123)  # type: ignore[arg-type]

    wrong_extension = tmp_path / "matrix.txt"
    wrong_extension.write_text("not an h5ad", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected a \.h5ad"):
        inspect_scATAC(wrong_extension)

    malformed = tmp_path / "malformed.h5ad"
    malformed.write_text("not an HDF5 file", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read scATAC AnnData file"):
        inspect_scATAC(malformed)
