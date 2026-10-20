"""Cross-tool input/continuation contracts, with explicitly simulated counts."""
from pathlib import Path
import hashlib
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.data.scatac import inspect_scATAC
from agent.tools.analysis import replicate_pseudobulk as pb
from agent.tools.analysis import differential_accessibility_backend as da
from agent.tools.analysis.pca_embedding import pca_embed_cells
from agent.orchestration import PlanStep, build_default_tool_registry, verify_step
from test_replicate_pseudobulk import _source, _feature
from test_differential_accessibility_verifier import _source_and_pseudobulk, _arguments


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_native_h5ad_without_filename_gate_and_exact_aggregation(tmp_path, storage):
    original = _source(tmp_path/"source.h5ad")
    data = ad.read_h5ad(original)
    expected = data.X.toarray()  # Deliberately tiny simulated fixture only.
    data.X = {"dense": lambda: expected, "csr": lambda: sparse.csr_matrix(expected),
              "csc": lambda: sparse.csc_matrix(expected)}[storage]()
    path = tmp_path/"registered.attachment"
    data.write_h5ad(path)
    before = path.read_bytes()
    inspection = inspect_scATAC(path)
    assert inspection["n_cells"] == 5
    pca = pca_embed_cells(path, tmp_path/"pca", n_components=2)
    assert pca["n_cells"] == 5
    feature = _feature(path, tmp_path/"features")
    manifest = json.loads(Path(feature["feature_space_path"]).read_text())
    assert manifest["schema_version"] == 2
    assert manifest["validation"]["source_sparse"] is (storage != "dense")
    assert manifest["biology"]["reference_compatibility_assessed"] is False
    result = pb.build_replicate_pseudobulk(
        feature["feature_space_path"], output_dir=tmp_path/"pseudobulk",
        replicate_key="donor", group_key="cell_type", condition_key="condition", group_source="raw_obs")
    observed = ad.read_h5ad(result["pseudobulk_path"])
    for i, row in enumerate(observed.obs.itertuples()):
        selected = (data.obs.cell_type == row.group) & (data.obs.donor == row.replicate) & (
            data.obs.condition == row.condition)
        np.testing.assert_array_equal(observed.X[i].toarray()[0], expected[selected].sum(axis=0))
    arguments = dict(feature_space_path=feature["feature_space_path"],
        output_dir=str(tmp_path/"pseudobulk"), replicate_key="donor",
        group_key="cell_type", condition_key="condition", group_source="raw_obs")
    assert verify_step(PlanStep("pb", "build_replicate_pseudobulk", arguments),
        arguments, result, build_default_tool_registry()).passed
    assert path.read_bytes() == before


def test_dense_source_slicing_is_bounded_and_noncounts_still_rejected():
    class Matrix:
        shape = (10000, 100000)
        dtype = np.dtype("int64")
    bounds = list(pb._matrix_chunks(Matrix()))
    assert max(last-first for first,last in bounds) * 100000 * 32 <= 64*1024**2
    for invalid in (np.array([[.5]]), np.array([[-1]]), np.array([[np.nan]]),
                    np.array([[float(2**63)]])):
        with pytest.raises(pb.M81ScientificError):
            pb._canonical_integer_chunk(invalid, 0, 1, "fragment_counts")


@pytest.mark.parametrize("species,assembly", [("rat", "rn7"), ("Homo sapiens", "hg38"), ("mouse", "mm39")])
def test_declared_new_assembly_and_repeated_single_base_coordinates(tmp_path, species, assembly):
    source = _source(tmp_path/"source.h5ad", coordinates=True)
    data = ad.read_h5ad(source)
    data.var["chromosome"] = "chr1"
    data.var["start"] = 1
    data.var["end"] = 1
    data.write_h5ad(source)
    arguments = dict(input_path=str(source), output_dir=str(tmp_path/"features"),
        matrix_source="X", matrix_semantics="fragment_counts", species=species,
        genome_assembly=assembly, coordinate_source="var_columns",
        feature_chrom_key="chromosome", feature_start_key="start",
        feature_end_key="end", coordinate_system="one_based_closed")
    result = pb.validate_scATAC_feature_space(**arguments)
    assert result["species"] == species
    assert result["genome_assembly"] == assembly
    assert result["n_features"] == 4
    registry = build_default_tool_registry()
    step = PlanStep("feature", "validate_scATAC_feature_space", arguments)
    assert verify_step(step, arguments, result, registry).passed
    # Every original feature remains distinct despite equal intervals.
    assert data.var_names.is_unique
    arguments["coordinate_system"] = "zero_based_half_open"
    with pytest.raises(pb.M81ScientificError, match="interval bounds"):
        pb.validate_scATAC_feature_space(**arguments)


def test_native_embedding_and_label_transfer_files_without_suffixes(tmp_path):
    from test_label_transfer import _sources
    from agent.tools.analysis.label_transfer import transfer_cell_labels
    from agent.tools.analysis.embedding_analysis import build_cell_neighbors
    from agent.tools.analysis.neighbors_publication import verify_result
    arguments = _sources(tmp_path)
    for name in ("reference_embedding_path", "query_embedding_path",
                 "reference_cell_ids_path", "query_cell_ids_path",
                 "reference_h5ad_path", "query_h5ad_path"):
        original = arguments[name]
        destination = tmp_path/name
        original.rename(destination)
        arguments[name] = destination
    result = transfer_cell_labels(**arguments)
    assert result["assigned_count"] == 2
    graph_args = dict(embedding_path=str(arguments["reference_embedding_path"]),
        cell_ids_path=str(arguments["reference_cell_ids_path"]),
        output_dir=str(tmp_path/"graph"), n_neighbors=2)
    graph = build_cell_neighbors(**graph_args)
    assert verify_result(graph_args, graph) == graph


@pytest.mark.skipif(os.environ.get("RUN_EDGER_INTEGRATION") != "1",
                    reason="Explicit native edgeR acceptance gate")
def test_native_da_completion_without_upstream_or_r_replay(tmp_path, monkeypatch):
    from agent.orchestration import differential_accessibility_verifier as oracle
    monkeypatch.setenv("AGENT_EDGER_RSCRIPT", "/home/likeyi/anaconda3/envs/agent-edger/bin/Rscript")
    raw, pseudobulk = _source_and_pseudobulk(tmp_path)
    arguments = _arguments(pseudobulk, tmp_path/"da", "independent")
    calls = []
    invoke = da._invoke_fixed_script
    def counted(*args, **kwargs):
        calls.append(args[1])
        return invoke(*args, **kwargs)
    monkeypatch.setattr(da, "_invoke_fixed_script", counted)
    result = da.run_replicate_differential_accessibility(**arguments)
    assert calls.count("run") == 1
    # Explicit qualification remains independent and does run the pinned oracle.
    oracle.verify_replicate_differential_accessibility(arguments, result)
    def forbidden(*args, **kwargs):
        pytest.fail("Production completion replayed upstream or native science")
    monkeypatch.setattr(da, "_invoke_fixed_script", forbidden)
    monkeypatch.setattr(oracle, "_invoke_verifier_r", forbidden)
    monkeypatch.setattr(pb, "_production_aggregate", forbidden)
    raw.unlink()  # A valid aggregate does not require its historical cell matrix.
    checked = verify_step(PlanStep("da", "run_replicate_differential_accessibility", arguments),
        arguments, result, build_default_tool_registry())
    assert checked.passed, checked
    assert any(c.name == "differential_accessibility_native_completion" for c in checked.checks)
    path = Path(result["da_path"])
    original = path.read_bytes()
    for change in ("replicate", "features", "statistic", "provenance"):
        data = ad.read_h5ad(path)
        if change == "replicate":
            data.obs["replicate"] = "wrong-experiment"
        elif change == "features":
            data.var_names = data.var_names[::-1]
        elif change == "statistic":
            data.var.loc[data.var["da_status"] == "tested", "PValue"] = -1
        else:
            data.uns[da.DA_PROVENANCE_KEY]["validation"]["upstream_aggregation_reconstructed"] = True
        data.write_h5ad(path)
        changed = dict(result, da_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with pytest.raises((ValueError, AssertionError)):
            da.verify_completion(arguments, changed)
        path.write_bytes(original)
