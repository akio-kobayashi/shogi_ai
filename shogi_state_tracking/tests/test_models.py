import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTest(unittest.TestCase):
    def setUp(self):
        from models import ModelConfig, T2MLRConfig

        torch.manual_seed(7)
        self.vanilla_config = ModelConfig(
            vocab_size=64,
            max_seq_len=32,
            d_model=32,
            n_layers=4,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
        )
        self.t2mlr_config = T2MLRConfig(
            vocab_size=64,
            max_seq_len=32,
            d_model=32,
            n_layers=4,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            l_start=1,
            l_end=2,
            jacobi_depth=2,
        )
        self.input_ids = torch.randint(0, 64, (2, 12))
        self.recurrent_mask = torch.zeros_like(self.input_ids, dtype=torch.bool)
        self.recurrent_mask[:, 5:] = True

    def test_vanilla_parallel_matches_kv_cached_exact(self):
        from models import VanillaTransformer

        model = VanillaTransformer(self.vanilla_config).eval()
        parallel = model(self.input_ids)
        exact = model(self.input_ids, exact_recurrence=True)
        torch.testing.assert_close(parallel.logits, exact.logits, rtol=1e-5, atol=1e-6)
        self.assertEqual(len(parallel.hidden_states), 5)

    def test_vanilla_and_llama_prefill_match_parallel_last_logit(self):
        from models import build_model

        for model_type in ("vanilla", "llama"):
            model = build_model(model_type, self.vanilla_config).eval()
            parallel = model(self.input_ids)
            logits, cache = model.prefill(self.input_ids)
            torch.testing.assert_close(logits, parallel.logits[:, -1:], rtol=1e-5, atol=1e-6)
            self.assertEqual(len(cache), self.vanilla_config.n_layers)

    def test_llama_parallel_matches_kv_cached_exact(self):
        from models import build_model

        model = build_model("llama", self.vanilla_config).eval()
        parallel = model(self.input_ids)
        exact = model(self.input_ids, exact_recurrence=True)
        torch.testing.assert_close(parallel.logits, exact.logits, rtol=1e-5, atol=1e-6)

    def test_t2mlr_shapes_expose_probe_states(self):
        from models import T2MLRTransformer

        model = T2MLRTransformer(self.t2mlr_config).eval()
        output = model(self.input_ids, recurrent_mask=self.recurrent_mask)
        self.assertEqual(tuple(output.logits.shape), (2, 12, 64))
        self.assertEqual(len(output.hidden_states), 5)
        self.assertEqual(tuple(output.recurrent_states.shape), (2, 12, 32))
        self.assertEqual(tuple(output.recurrent_gates.shape), (2, 12, 32))
        self.assertTrue(bool((output.recurrent_gates[:, :5] == 0).all()))

    def test_rezero_initialization_preserves_vanilla_backbone(self):
        from models import T2MLRTransformer, VanillaTransformer

        vanilla = VanillaTransformer(self.vanilla_config).eval()
        t2mlr = T2MLRTransformer(self.t2mlr_config).eval()
        missing, unexpected = t2mlr.load_state_dict(vanilla.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(name.startswith(("fusion.", "recurrent_norm.")) for name in missing))
        vanilla_output = vanilla(self.input_ids)
        t2mlr_output = t2mlr(
            self.input_ids, recurrent_mask=self.recurrent_mask
        )
        torch.testing.assert_close(
            vanilla_output.logits, t2mlr_output.logits, rtol=1e-5, atol=1e-6
        )

    def test_recurrence_changes_output_and_receives_gradient(self):
        from models import T2MLRTransformer

        model = T2MLRTransformer(self.t2mlr_config)
        with torch.no_grad():
            model.fusion.rezero_gamma.fill_(0.2)
        output = model(self.input_ids, recurrent_mask=self.recurrent_mask)
        loss = output.logits.square().mean()
        loss.backward()
        self.assertIsNotNone(model.fusion.gate.weight.grad)
        self.assertGreater(float(model.fusion.gate.weight.grad.abs().sum()), 0.0)

    def test_exact_recurrence_is_causal(self):
        from models import T2MLRTransformer

        model = T2MLRTransformer(self.t2mlr_config).eval()
        with torch.no_grad():
            model.fusion.rezero_gamma.fill_(0.2)
        changed = self.input_ids.clone()
        changed[:, 9:] = torch.randint(0, 64, changed[:, 9:].shape)
        first = model(
            self.input_ids,
            recurrent_mask=self.recurrent_mask,
            exact_recurrence=True,
        )
        second = model(
            changed,
            recurrent_mask=self.recurrent_mask,
            exact_recurrence=True,
        )
        torch.testing.assert_close(first.logits[:, :9], second.logits[:, :9])
        torch.testing.assert_close(
            first.recurrent_states[:, :9], second.recurrent_states[:, :9]
        )

    def test_prefix_hidden_state_matches_full_sequence_at_same_position(self):
        """未来tokenを追加しても，因果mask以前のh_preは変わらないことを確認する。"""
        prefix = self.input_ids[:, :7]
        full = self.input_ids
        position = prefix.shape[1] - 1
        for model_type in ("vanilla", "llama"):
            model = __import__("models", fromlist=["build_model"]).build_model(
                model_type, self.vanilla_config
            ).eval()
            prefix_output = model(prefix)
            full_output = model(full)
            for layer_prefix, layer_full in zip(prefix_output.hidden_states, full_output.hidden_states):
                torch.testing.assert_close(
                    layer_prefix[:, -1], layer_full[:, position], rtol=1e-5, atol=1e-6
                )

    def test_parameter_count_is_reported(self):
        from models import (
            T2MLRTransformer,
            VanillaTransformer,
            parameter_matched_vanilla_config,
        )

        vanilla = VanillaTransformer(self.vanilla_config)
        t2mlr = T2MLRTransformer(self.t2mlr_config)
        self.assertGreater(t2mlr.parameter_count(), vanilla.parameter_count())
        matched = VanillaTransformer(
            parameter_matched_vanilla_config(self.t2mlr_config)
        )
        relative_difference = abs(
            matched.parameter_count() - t2mlr.parameter_count()
        ) / t2mlr.parameter_count()
        self.assertLess(relative_difference, 0.005)


if __name__ == "__main__":
    unittest.main()
