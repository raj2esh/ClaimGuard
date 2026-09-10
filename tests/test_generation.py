"""Tests for claimguard.generation (Step 17: Qwen3-8B generator integration).

Structural/unit tests use fake tokenizer/model objects (no GPU, no real
model download) so they stay fast and offline. One guarded test class
exercises the REAL Qwen3-8B checkpoint end-to-end and is skipped if it
isn't available in this environment (same pattern as
`TestLoadRealVerifierCheckpoint` in tests/test_verification.py).
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from claimguard import config as cg_config
from claimguard.generation import qwen3 as qwen3_mod
from claimguard.generation.types import GenerationResult


def _generation_source_files() -> list[Path]:
    import claimguard.generation as pkg
    return list(Path(pkg.__file__).parent.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    def test_no_ragtruth_truthfulqa_fever_halueval_imports(self) -> None:
        import ast

        for path in _generation_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            lowered = {m.lower() for m in imported}
            for banned in ("ragtruth", "truthfulqa", "fever", "halueval"):
                self.assertFalse(any(banned in m for m in lowered), f"{path} imports {banned}: {imported}")


class TestNoDatasetLabelAccess(unittest.TestCase):
    """Structural guard: generate()'s parameters carry only generic prompt
    text and generation settings - no dataset-label-shaped parameter."""

    def test_generate_has_no_dataset_label_parameter(self) -> None:
        sig = inspect.signature(qwen3_mod.generate)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "hallucination", "evidence", "fever", "ragtruth", "truthfulqa"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])

    def test_load_generator_has_no_dataset_label_parameter(self) -> None:
        sig = inspect.signature(qwen3_mod.load_generator)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "hallucination", "evidence", "fever", "ragtruth", "truthfulqa"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])


class TestConfigLoads(unittest.TestCase):
    def test_generator_baseline_config_loads_with_required_keys(self) -> None:
        cfg = cg_config.load_generator_baseline_config()
        self.assertIn("model", cfg)
        self.assertIn("generation", cfg)
        self.assertIn("chat_template", cfg)
        self.assertIn("name", cfg["model"])
        self.assertIn("max_new_tokens", cfg["generation"])
        self.assertIn("do_sample", cfg["generation"])
        self.assertIn("seed", cfg["generation"])
        self.assertIn("enable_thinking", cfg["chat_template"])

    def test_generator_baseline_model_identity_matches_models_yaml(self) -> None:
        """Prevents configs/generator_baseline.yaml silently drifting from
        configs/models.yaml's generator.primary - the single source of
        truth for model selection."""
        gen_cfg = cg_config.load_generator_baseline_config()
        models_cfg = cg_config.load_models_config()
        primary = models_cfg["generator"]["primary"]
        self.assertEqual(gen_cfg["model"]["name"], primary["name"])
        self.assertEqual(gen_cfg["model"]["dtype"], primary["dtype"])
        self.assertEqual(gen_cfg["model"]["device"], primary["device"])

    def test_generator_baseline_do_sample_is_false_for_deterministic_smoke_test(self) -> None:
        cfg = cg_config.load_generator_baseline_config()
        self.assertFalse(cfg["generation"]["do_sample"])

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(cg_config.ConfigError):
            cg_config.load_generator_baseline_config(config_dir=Path("/nonexistent/config/dir"))


class FakeTokenizer:
    """Minimal tokenizer fake exercising the same call shape as the real
    Qwen3 tokenizer's apply_chat_template + decode."""

    def __init__(self, supports_enable_thinking: bool = True):
        self.supports_enable_thinking = supports_enable_thinking
        self.last_messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt",
                             return_dict=True, enable_thinking=None):
        import torch

        if enable_thinking is not None and not self.supports_enable_thinking:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
        self.last_messages = messages
        # Deterministic fake token ids: one id per word across all message contents.
        n_tokens = sum(len(m["content"].split()) for m in messages) + 1  # +1 for generation-prompt token
        input_ids = torch.arange(1, n_tokens + 1).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


class FakeModel:
    """Fake causal LM: `.generate()` appends `n_new` fixed new token ids
    after the prompt, so tests can check prompt-stripping and determinism
    without a real model."""

    def __init__(self, n_new: int = 5):
        self.n_new = n_new
        self._device = "cpu"

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask, **gen_kwargs):
        import torch

        prompt_len = input_ids.shape[1]
        new_ids = torch.arange(1000, 1000 + self.n_new).unsqueeze(0)
        return torch.cat([input_ids, new_ids], dim=1)


class TestBuildChatInputs(unittest.TestCase):
    def test_user_only_message(self) -> None:
        tok = FakeTokenizer()
        qwen3_mod.build_chat_inputs(tok, "What is 2+2?", None, enable_thinking=False,
                                     fallback_on_unsupported_kwarg=True)
        self.assertEqual(tok.last_messages, [{"role": "user", "content": "What is 2+2?"}])

    def test_system_and_user_message(self) -> None:
        tok = FakeTokenizer()
        qwen3_mod.build_chat_inputs(tok, "What is 2+2?", "Be concise.", enable_thinking=False,
                                     fallback_on_unsupported_kwarg=True)
        self.assertEqual(tok.last_messages, [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2?"},
        ])

    def test_fallback_when_enable_thinking_unsupported(self) -> None:
        tok = FakeTokenizer(supports_enable_thinking=False)
        _, _, fallback_triggered = qwen3_mod.build_chat_inputs(
            tok, "hi", None, enable_thinking=False, fallback_on_unsupported_kwarg=True,
        )
        self.assertTrue(fallback_triggered)

    def test_no_fallback_when_disabled_reraises(self) -> None:
        tok = FakeTokenizer(supports_enable_thinking=False)
        with self.assertRaises(TypeError):
            qwen3_mod.build_chat_inputs(tok, "hi", None, enable_thinking=False,
                                         fallback_on_unsupported_kwarg=False)

    def test_supported_kwarg_no_fallback(self) -> None:
        tok = FakeTokenizer(supports_enable_thinking=True)
        _, _, fallback_triggered = qwen3_mod.build_chat_inputs(
            tok, "hi", None, enable_thinking=False, fallback_on_unsupported_kwarg=True,
        )
        self.assertFalse(fallback_triggered)


class TestBuildGenerationKwargs(unittest.TestCase):
    def test_do_sample_false_omits_temperature_and_top_p(self) -> None:
        kwargs = qwen3_mod._build_generation_kwargs(256, do_sample=False, temperature=0.7, top_p=0.9)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertEqual(kwargs["do_sample"], False)
        self.assertEqual(kwargs["max_new_tokens"], 256)

    def test_do_sample_true_includes_temperature_and_top_p(self) -> None:
        kwargs = qwen3_mod._build_generation_kwargs(100, do_sample=True, temperature=0.7, top_p=0.9)
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)

    def test_do_sample_true_without_temperature_or_top_p(self) -> None:
        kwargs = qwen3_mod._build_generation_kwargs(100, do_sample=True, temperature=None, top_p=None)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)


class TestGenerateWithFakes(unittest.TestCase):
    def test_output_type_and_structure(self) -> None:
        result = qwen3_mod.generate(
            FakeModel(n_new=5), FakeTokenizer(), "What is the capital of France?",
            model_identifier="fake/model", max_new_tokens=5, do_sample=False, seed=42,
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertIsInstance(result.generated_text, str)
        self.assertGreater(len(result.generated_text), 0)
        self.assertEqual(result.generated_token_count, 5)
        self.assertEqual(result.model_identifier, "fake/model")

    def test_generated_text_excludes_prompt(self) -> None:
        result = qwen3_mod.generate(
            FakeModel(n_new=3), FakeTokenizer(), "one two three four", model_identifier="fake/model",
            max_new_tokens=3, do_sample=False,
        )
        # FakeModel's new token ids are 1000/1001/1002 -> decoded as tok1000 tok1001 tok1002.
        # The prompt's own fake token ids are small (1..N), so no overlap is possible if
        # prompt-stripping is correct.
        self.assertEqual(result.generated_text, "tok1000 tok1001 tok1002")

    def test_deterministic_repeatability_with_do_sample_false(self) -> None:
        r1 = qwen3_mod.generate(FakeModel(n_new=4), FakeTokenizer(), "hello", model_identifier="fake/model",
                                 max_new_tokens=4, do_sample=False, seed=42)
        r2 = qwen3_mod.generate(FakeModel(n_new=4), FakeTokenizer(), "hello", model_identifier="fake/model",
                                 max_new_tokens=4, do_sample=False, seed=42)
        self.assertEqual(r1.generated_text, r2.generated_text)
        self.assertEqual(r1.prompt_token_count, r2.prompt_token_count)

    def test_generation_config_captures_settings(self) -> None:
        result = qwen3_mod.generate(
            FakeModel(n_new=2), FakeTokenizer(), "hi", model_identifier="fake/model",
            max_new_tokens=2, do_sample=False, seed=7, enable_thinking=False,
        )
        self.assertEqual(result.generation_config["max_new_tokens"], 2)
        self.assertEqual(result.generation_config["do_sample"], False)
        self.assertEqual(result.generation_config["seed"], 7)
        self.assertEqual(result.generation_config["enable_thinking"], False)
        self.assertIn("chat_template_fallback_triggered", result.generation_config)

    def test_metadata_is_json_serializable(self) -> None:
        result = qwen3_mod.generate(FakeModel(n_new=2), FakeTokenizer(), "hi", model_identifier="fake/model",
                                     max_new_tokens=2, do_sample=False)
        serialized = json.dumps(result.to_dict())
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["generated_text"], result.generated_text)
        self.assertEqual(reloaded["model_identifier"], "fake/model")

    def test_system_instruction_preserved_in_result(self) -> None:
        result = qwen3_mod.generate(
            FakeModel(n_new=2), FakeTokenizer(), "hi", model_identifier="fake/model",
            system_instruction="Be concise.", max_new_tokens=2, do_sample=False,
        )
        self.assertEqual(result.system_instruction, "Be concise.")

    def test_no_system_instruction_defaults_to_none(self) -> None:
        result = qwen3_mod.generate(FakeModel(n_new=2), FakeTokenizer(), "hi", model_identifier="fake/model",
                                     max_new_tokens=2, do_sample=False)
        self.assertIsNone(result.system_instruction)


class TestLoadGeneratorRealCheckpoint(unittest.TestCase):
    """Guarded by SkipTest if the real Qwen3-8B model isn't available
    (no GPU, insufficient VRAM, or no network access to the HF Hub)."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.model = None
        cls.skip_reason = None
        if not torch.cuda.is_available():
            cls.skip_reason = "CUDA not available"
            return
        free, _total = torch.cuda.mem_get_info(0)
        if free / 1024**3 < 20.0:
            cls.skip_reason = f"insufficient free VRAM ({free / 1024**3:.1f} GB < 20 GB safety threshold)"
            return
        try:
            cls.model, cls.tokenizer, cls.info = qwen3_mod.load_generator()
        except Exception as exc:  # noqa: BLE001
            cls.skip_reason = str(exc)

    def setUp(self) -> None:
        if self.model is None:
            raise unittest.SkipTest(f"Real Qwen3-8B not available: {self.skip_reason}")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.model is not None:
            qwen3_mod.unload_generator(cls.model)

    def test_real_model_is_qwen3_8b(self) -> None:
        self.assertEqual(self.info["model_identifier"], "Qwen/Qwen3-8B")
        self.assertGreater(self.info["n_params"], 7_000_000_000)
        self.assertIn("bfloat16", self.info["dtype"])

    def test_real_model_generates_finite_deterministic_output(self) -> None:
        r1 = qwen3_mod.generate(self.model, self.tokenizer, "What is the capital of France?",
                                 model_identifier=self.info["model_identifier"], max_new_tokens=10,
                                 do_sample=False, seed=42)
        r2 = qwen3_mod.generate(self.model, self.tokenizer, "What is the capital of France?",
                                 model_identifier=self.info["model_identifier"], max_new_tokens=10,
                                 do_sample=False, seed=42)
        self.assertIsInstance(r1.generated_text, str)
        self.assertGreater(len(r1.generated_text), 0)
        self.assertEqual(r1.generated_text, r2.generated_text)
        self.assertNotIn("What is the capital of France?", r1.generated_text)


if __name__ == "__main__":
    unittest.main()
