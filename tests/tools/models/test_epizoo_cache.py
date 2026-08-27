from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest
import torch

from agent.tools.models import epizoo_cache as cache


@pytest.fixture(autouse=True)
def empty_model_cache():
    cache.clear_epizoo_backend_cache()
    yield
    cache.clear_epizoo_backend_cache()


def test_first_call_initializes_once_and_second_call_reuses_exact_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[object] = []

    def fake_load_model(**kwargs):
        model = object()
        loads.append(model)
        return model

    monkeypatch.setattr(cache.epizoo_backend, "load_model", fake_load_model)
    checkpoint = tmp_path / "models" / "epizoo.pth"

    first = cache.get_cached_epizoo_model(checkpoint, device="cpu")
    second = cache.get_cached_epizoo_model(
        tmp_path / "models" / ".." / "models" / "epizoo.pth",
        device=torch.device("cpu"),
    )

    assert first is second
    assert loads == [first]


def test_different_initialization_configuration_uses_different_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[object] = []

    def fake_load_model(**kwargs):
        model = object()
        loads.append(model)
        return model

    monkeypatch.setattr(cache.epizoo_backend, "load_model", fake_load_model)
    first_checkpoint = tmp_path / "first.pth"
    second_checkpoint = tmp_path / "second.pth"

    first = cache.get_cached_epizoo_model(first_checkpoint, device="cpu")
    different_checkpoint = cache.get_cached_epizoo_model(
        second_checkpoint, device="cpu"
    )
    different_dtype = cache.get_cached_epizoo_model(
        first_checkpoint, device="cpu", dtype=torch.float16
    )

    assert len(loads) == 3
    assert len({id(first), id(different_checkpoint), id(different_dtype)}) == 3


def test_failed_initialization_is_not_cached_and_retry_can_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = 0
    recovered_model = object()

    def sometimes_failing_load(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic initialization failure")
        return recovered_model

    monkeypatch.setattr(
        cache.epizoo_backend, "load_model", sometimes_failing_load
    )
    checkpoint = tmp_path / "epizoo.pth"

    with pytest.raises(RuntimeError, match="synthetic initialization failure"):
        cache.get_cached_epizoo_model(checkpoint, device="cpu")

    retry = cache.get_cached_epizoo_model(checkpoint, device="cpu")
    reused = cache.get_cached_epizoo_model(checkpoint, device="cpu")
    assert retry is recovered_model
    assert reused is recovered_model
    assert attempts == 2


def test_clear_causes_next_call_to_initialize_a_new_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[object] = []

    def fake_load_model(**kwargs):
        model = object()
        loads.append(model)
        return model

    monkeypatch.setattr(cache.epizoo_backend, "load_model", fake_load_model)
    checkpoint = tmp_path / "epizoo.pth"

    first = cache.get_cached_epizoo_model(checkpoint, device="cpu")
    cache.clear_epizoo_backend_cache()
    second = cache.get_cached_epizoo_model(checkpoint, device="cpu")

    assert first is not second
    assert loads == [first, second]


def test_simultaneous_misses_initialize_only_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[object] = []

    def slow_fake_load(**kwargs):
        time.sleep(0.05)
        model = object()
        loads.append(model)
        return model

    monkeypatch.setattr(cache.epizoo_backend, "load_model", slow_fake_load)
    checkpoint = tmp_path / "epizoo.pth"

    with ThreadPoolExecutor(max_workers=8) as executor:
        models = list(
            executor.map(
                lambda _: cache.get_cached_epizoo_model(checkpoint, device="cpu"),
                range(8),
            )
        )

    assert len(loads) == 1
    assert all(model is loads[0] for model in models)
