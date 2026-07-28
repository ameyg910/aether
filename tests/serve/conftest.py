"""Shared fixtures: a real checkpoint on disk and an app serving it."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from aether.config.schemas import ModelConfig
from aether.models.aether_model import AetherModel
from aether.serve.app import ServerSettings, create_app

VOCAB, MASK, MAX_LEN = 257, 256, 64  # byte tokenizer + [MASK]


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = ModelConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=4, max_seq_len=MAX_LEN)
    torch.manual_seed(0)
    model = AetherModel(cfg)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 4242,
            "extra": {"model_config": dataclasses.asdict(cfg)},
        },
        path,
    )
    return path


@pytest.fixture
def settings(checkpoint: Path) -> ServerSettings:
    return ServerSettings(
        model_version=f"local:{checkpoint}",
        tokenizer="byte",
        device="cpu",
        max_batch_size=8,
        max_wait_ms=15.0,
    )


@pytest.fixture
def client(settings: ServerSettings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    """A server with no model, for readiness and 503 paths."""
    with TestClient(create_app(ServerSettings(model_version=None, device="cpu"))) as c:
        yield c
