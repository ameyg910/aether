"""The Hydra config composes and validates against the typed schema."""

from __future__ import annotations

from aether.config import AetherConfig, load_config


def test_load_config_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, AetherConfig)
    assert cfg.model.d_model == 128
    assert cfg.diffusion.mask_token_id == 0
    assert cfg.diffusion.schedule.kind == "linear"


def test_override_applies() -> None:
    cfg = load_config(overrides=["model.d_model=256"])
    assert cfg.model.d_model == 256


class TestConfigDirResolution:
    """Config directory resolution — the container vs source-tree distinction.

    A packaged install lives in site-packages, where walking up from ``__file__``
    lands inside the environment rather than at a repo root. Containers therefore
    set ``AETHER_CONFIG_DIR`` to where they copied the configs; the source-tree
    walk-up remains the fallback for editable installs and tests. This is the bug
    that surfaced in the Week 8 image smoke test.
    """

    @staticmethod
    def _reload(monkeypatch, value):  # type: ignore[no-untyped-def]
        import importlib

        import aether.config as cfg

        if value is None:
            monkeypatch.delenv("AETHER_CONFIG_DIR", raising=False)
        else:
            monkeypatch.setenv("AETHER_CONFIG_DIR", value)
        importlib.reload(cfg)
        return cfg

    def test_env_override_takes_precedence(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        cfg = self._reload(monkeypatch, "/some/container/path/configs")
        try:
            assert cfg._resolve_config_dir() == "/some/container/path/configs"
        finally:
            self._reload(monkeypatch, None)  # restore for other tests

    def test_falls_back_to_source_tree(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        cfg = self._reload(monkeypatch, None)
        assert cfg._resolve_config_dir().endswith("configs")

    def test_loads_config_from_override_dir(self, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import shutil
        from pathlib import Path

        import aether.config as cfg

        src = Path(cfg._resolve_config_dir())
        dest = Path(str(tmp_path)) / "app" / "configs"
        shutil.copytree(src, dest)
        try:
            cfg = self._reload(monkeypatch, str(dest))
            loaded = cfg.load_config(["serve.port=4321"])
            assert loaded.serve.port == 4321
        finally:
            self._reload(monkeypatch, None)


class TestCliHelp:
    """`--help` must exit cleanly, not crash in Hydra's override parser.

    Because the CLIs call Hydra's ``compose`` API directly rather than via the
    ``@hydra.main`` decorator, every argv token is treated as a ``key=value``
    override. ``--help`` is not one, so without interception Hydra raises a lexer
    error -- which is what broke the Week 8 image smoke test.
    """

    def test_help_flag_exits_zero(self) -> None:
        import pytest

        from aether.config import cli_overrides

        with pytest.raises(SystemExit) as exc:
            cli_overrides(["--help"], "usage text")
        assert exc.value.code == 0

    def test_short_help_flag_exits_zero(self) -> None:
        import pytest

        from aether.config import cli_overrides

        with pytest.raises(SystemExit) as exc:
            cli_overrides(["-h"], "usage text")
        assert exc.value.code == 0

    def test_normal_overrides_pass_through_untouched(self) -> None:
        from aether.config import cli_overrides

        argv = ["model=medium", "train.max_steps=100"]
        assert cli_overrides(argv, "usage") == argv
