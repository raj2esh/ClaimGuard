"""Lightweight structural tests for ClaimGuard.

Scope: package import, config loading, required config keys, path resolution.
No model loading, no GPU access, no downloads — see PROJECT_PLAN.md Step 5
for model smoke tests (not run yet).
"""

from __future__ import annotations

import unittest

import claimguard
from claimguard import config


class TestPackageImport(unittest.TestCase):
    def test_import_succeeds(self) -> None:
        self.assertTrue(hasattr(claimguard, "__version__"))


class TestConfigLoading(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        cfg = config.load_default_config()
        self.assertIsInstance(cfg, dict)

    def test_models_config_loads(self) -> None:
        cfg = config.load_models_config()
        self.assertIsInstance(cfg, dict)
        for role in ("generator", "embedding", "reranker", "verifier"):
            self.assertIn(role, cfg)

    def test_experiments_config_loads(self) -> None:
        cfg = config.load_experiments_config()
        self.assertIsInstance(cfg, dict)
        self.assertIn("experiments", cfg)


class TestRequiredConfigKeys(unittest.TestCase):
    def test_default_config_has_required_keys(self) -> None:
        cfg = config.load_default_config()
        config.validate_default_config(cfg)  # raises ConfigError on failure

    def test_missing_key_is_detected(self) -> None:
        incomplete = {"seed": 42}
        with self.assertRaises(config.ConfigError):
            config.validate_default_config(incomplete)


class TestPathResolution(unittest.TestCase):
    def test_project_root_exists(self) -> None:
        self.assertTrue(config.PROJECT_ROOT.exists())
        self.assertTrue((config.PROJECT_ROOT / "configs").is_dir())

    def test_resolve_path(self) -> None:
        resolved = config.resolve_path("configs")
        self.assertTrue(resolved.exists())
        self.assertTrue(resolved.is_dir())

    def test_resolve_data_dirs(self) -> None:
        for rel in ("data/raw", "data/processed", "data/evaluation"):
            resolved = config.resolve_path(rel)
            self.assertTrue(resolved.is_dir(), f"missing expected dir: {rel}")


if __name__ == "__main__":
    unittest.main()
