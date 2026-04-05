#!/usr/bin/env python3
"""
Test suite for Metric Space Explorer (app.py)

Covers:
  - Environment safety helpers
  - Model config detection
  - Tokenization
  - Probe sentence generation
  - Hidden state extraction & delta computation
  - Neighbor computation
  - PCA computation
  - Grid probe generation
  - Output assembly
  - HTTP handler routing
  - Argument parsing
  - End-to-end processing pipeline
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from io import BytesIO
from http.server import HTTPServer
import threading
import urllib.request
import numpy as np
import time

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
# Prevent ensure_safe_env() from re-execing during import
os.environ["UV_EXCLUDE_NEWER"] = "2099-01-01T00:00:00Z"

import app


# ===================================================================
# 1. ENVIRONMENT SAFETY
# ===================================================================
class TestEnvironmentSafety(unittest.TestCase):
    """Tests for Section 1: Environment Safety helpers."""

    def test_compute_exclude_newer_date_format(self):
        """Return value must be an ISO-8601-ish UTC timestamp."""
        result = app.compute_exclude_newer_date(days_back=8)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_compute_exclude_newer_date_different_days(self):
        d8 = app.compute_exclude_newer_date(8)
        d1 = app.compute_exclude_newer_date(1)
        # 1 day back should be more recent (lexicographically greater)
        self.assertGreater(d1, d8)

    def test_should_set_exclude_newer_when_set(self):
        """If UV_EXCLUDE_NEWER is already set, should return False."""
        with patch.dict(os.environ, {"UV_EXCLUDE_NEWER": "already"}):
            self.assertFalse(app.should_set_exclude_newer())

    def test_should_set_exclude_newer_when_unset(self):
        """If UV_EXCLUDE_NEWER is absent, should return True."""
        env = os.environ.copy()
        env.pop("UV_EXCLUDE_NEWER", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(app.should_set_exclude_newer())


# ===================================================================
# 2. MODEL DETECTION & CONFIG HELPERS
# ===================================================================
class TestModelDetection(unittest.TestCase):
    """Tests for Section 2: detect_model_type, get_n_layers, get_hidden_dim."""

    @staticmethod
    def _cfg(**kwargs):
        """Build a lightweight mock config."""
        cfg = MagicMock()
        cfg.architectures = kwargs.get("architectures", [])
        cfg.is_decoder = kwargs.get("is_decoder", False)
        cfg.n_layer = kwargs.get("n_layer", None)
        cfg.num_hidden_layers = kwargs.get("num_hidden_layers", None)
        cfg.num_layers = kwargs.get("num_layers", None)
        cfg.n_embd = kwargs.get("n_embd", None)
        cfg.hidden_size = kwargs.get("hidden_size", None)
        cfg.d_model = kwargs.get("d_model", None)
        return cfg

    # -- detect_model_type --
    def test_detect_causal_gpt(self):
        cfg = self._cfg(architectures=["GPT2LMHeadModel"])
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_detect_causal_opt(self):
        cfg = self._cfg(architectures=["OPTForCausalLM"])
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_detect_masked_bert(self):
        cfg = self._cfg(architectures=["BertForMaskedLM"])
        self.assertEqual(app.detect_model_type(cfg), "masked")

    def test_detect_masked_roberta(self):
        cfg = self._cfg(architectures=["RobertaModel"])
        self.assertEqual(app.detect_model_type(cfg), "masked")

    def test_detect_causal_by_is_decoder(self):
        cfg = self._cfg(architectures=["SomeUnknownArch"], is_decoder=True)
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_detect_default_causal(self):
        cfg = self._cfg(architectures=["SomeUnknownArch"])
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_detect_none_architectures(self):
        cfg = self._cfg()
        cfg.architectures = None
        self.assertEqual(app.detect_model_type(cfg), "causal")

    # -- get_n_layers --
    def test_get_n_layers_n_layer(self):
        cfg = self._cfg(n_layer=12)
        self.assertEqual(app.get_n_layers(cfg), 12)

    def test_get_n_layers_num_hidden_layers(self):
        cfg = self._cfg(num_hidden_layers=24)
        cfg.n_layer = None
        self.assertEqual(app.get_n_layers(cfg), 24)

    def test_get_n_layers_fallback(self):
        cfg = self._cfg()
        cfg.n_layer = None
        cfg.num_hidden_layers = None
        cfg.num_layers = None
        self.assertEqual(app.get_n_layers(cfg), 12)

    # -- get_hidden_dim --
    def test_get_hidden_dim_n_embd(self):
        cfg = self._cfg(n_embd=768)
        self.assertEqual(app.get_hidden_dim(cfg), 768)

    def test_get_hidden_dim_hidden_size(self):
        cfg = self._cfg(hidden_size=1024)
        cfg.n_embd = None
        self.assertEqual(app.get_hidden_dim(cfg), 1024)

    def test_get_hidden_dim_fallback(self):
        cfg = self._cfg()
        cfg.n_embd = None
        cfg.hidden_size = None
        cfg.d_model = None
        self.assertEqual(app.get_hidden_dim(cfg), 768)


# ===================================================================
# 3. TOKENIZATION
# ===================================================================
class TestTokenization(unittest.TestCase):
    """Tests for Section 4: tokenize_text, decode_token_ids."""

    def _mock_tokenizer(self):
        tok = MagicMock()
        # Simulate tokenizer("hello world", return_tensors="pt")
        import torch
        tok.return_value = {"input_ids": torch.tensor([[15496, 995]])}
        tok.decode = lambda ids: {15496: "hello", 995: " world"}.get(ids[0], "?")
        return tok

    def test_tokenize_text_returns_ids_and_strings(self):
        tok = self._mock_tokenizer()
        ids, labels = app.tokenize_text(tok, "hello world")
        self.assertEqual(ids.shape[1], 2)
        self.assertEqual(len(labels), 2)

    def test_decode_token_ids_replaces_special_chars(self):
        import torch
        tok = MagicMock()
        tok.decode = lambda ids: "\u0120word" if ids[0] == 1 else "\u010anewline"
        result = app.decode_token_ids(tok, torch.tensor([1, 2]))
        self.assertEqual(result[0], " word")
        self.assertEqual(result[1], "\\nnewline")


# ===================================================================
# 4. PROBE SENTENCES
# ===================================================================
class TestProbeSentences(unittest.TestCase):
    """Tests for Section 5: get_probe_texts, tokenize_probes."""

    def test_get_probe_texts_returns_list(self):
        probes = app.get_probe_texts()
        self.assertIsInstance(probes, list)
        self.assertGreater(len(probes), 10)

    def test_get_probe_texts_all_strings(self):
        for p in app.get_probe_texts():
            self.assertIsInstance(p, str)
            self.assertGreater(len(p), 0)

    def test_tokenize_probes_flags_all_false(self):
        """All probe tokens should have is_real=False."""
        import torch
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tok.decode = lambda ids: "x"
        seqs, labels, flags = app.tokenize_probes(tok, ["a b c"])
        self.assertTrue(all(f is False for f in flags))
        self.assertEqual(len(labels), 3)


# ===================================================================
# 5. HIDDEN STATE EXTRACTION
# ===================================================================
class TestHiddenStateExtraction(unittest.TestCase):
    """Tests for Section 6: extract_hidden_states, compute_layer0_and_deltas."""

    def _fake_hidden_states(self, n_layers=3, seq_len=4, dim=8):
        """Create a tuple of (n_layers+1) hidden-state tensors."""
        import torch
        return tuple(
            torch.randn(1, seq_len, dim) for _ in range(n_layers + 1)
        )

    def test_compute_layer0_and_deltas_shapes(self):
        hs = self._fake_hidden_states(n_layers=3, seq_len=4, dim=8)
        l0, dl = app.compute_layer0_and_deltas(hs, n_layers=3)
        self.assertEqual(len(l0), 4)
        self.assertEqual(len(dl), 4)
        # Each token should have 3 deltas
        for d in dl:
            self.assertEqual(len(d), 3)
        # Each vector should be dim=8
        for v in l0:
            self.assertEqual(v.shape, (8,))

    def test_deltas_are_differences(self):
        """Delta[l] should equal hidden_states[l+1] - hidden_states[l]."""
        import torch
        hs = self._fake_hidden_states(n_layers=2, seq_len=2, dim=4)
        l0, dl = app.compute_layer0_and_deltas(hs, n_layers=2)
        for s in range(2):
            for l in range(2):
                expected = (hs[l + 1][0][s] - hs[l][0][s]).cpu().numpy()
                np.testing.assert_allclose(dl[s][l], expected, atol=1e-6)

    def test_run_all_sequences(self):
        """run_all_sequences should aggregate across multiple sequences."""
        import torch

        def fake_model(input_ids):
            seq_len = input_ids.shape[1]
            out = MagicMock()
            out.hidden_states = tuple(
                torch.randn(1, seq_len, 4) for _ in range(3)  # 2 layers
            )
            return out

        model = MagicMock(side_effect=fake_model)
        seqs = [torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])]
        l0, dl = app.run_all_sequences(model, seqs, n_layers=2)
        self.assertEqual(len(l0), 5)  # 2 + 3
        self.assertEqual(len(dl), 5)


# ===================================================================
# 6. NEIGHBOR COMPUTATION
# ===================================================================
class TestNeighborComputation(unittest.TestCase):
    """Tests for Section 7: compute_neighbors, find_k_neighbors."""

    def test_find_k_neighbors_basic(self):
        all_emb = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
        ])
        labels = ["a", "b", "c", "d"]
        is_real = [True, True, False, False]
        result = app.find_k_neighbors(0, all_emb[0], all_emb, labels, is_real, k=2)
        self.assertEqual(len(result), 2)
        # Nearest to [0,0] should be [1,0] and [0,1]
        idxs = {r["idx"] for r in result}
        self.assertEqual(idxs, {1, 2})

    def test_find_k_neighbors_excludes_self(self):
        all_emb = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a", "b"], [True, True], k=5)
        idxs = [r["idx"] for r in result]
        self.assertNotIn(0, idxs)

    def test_compute_neighbors_returns_correct_count(self):
        real = np.array([[0.0, 0.0], [1.0, 0.0]])
        all_emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        labels = ["a", "b", "c"]
        is_real = [True, True, False]
        result = app.compute_neighbors(real, all_emb, labels, is_real, k=2)
        self.assertEqual(len(result), 2)  # one per real token

    def test_neighbor_dist_is_float(self):
        all_emb = np.array([[0.0, 0.0], [3.0, 4.0]])
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a", "b"], [True, True], k=1)
        self.assertAlmostEqual(result[0]["dist"], 5.0, places=5)


# ===================================================================
# 7. PCA COMPUTATION
# ===================================================================
class TestPCAComputation(unittest.TestCase):
    """Tests for Section 8: compute_pca_basis."""

    def test_pca_basis_shapes(self):
        mat = np.random.randn(20, 8)
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 8)
        self.assertEqual(centroid.shape, (8,))
        self.assertEqual(centered.shape, (20, 8))
        self.assertEqual(pc1.shape, (8,))
        self.assertEqual(pc2.shape, (8,))
        self.assertEqual(p1.shape, (20,))
        self.assertEqual(p2.shape, (20,))

    def test_pca_centroid_is_mean(self):
        mat = np.random.randn(15, 4)
        centroid, _, _, _, _, _ = app.compute_pca_basis(mat, 4)
        np.testing.assert_allclose(centroid, mat.mean(axis=0), atol=1e-10)

    def test_pca_centered_zero_mean(self):
        mat = np.random.randn(15, 4)
        _, centered, _, _, _, _ = app.compute_pca_basis(mat, 4)
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-10)

    def test_pca_single_point_fallback(self):
        """With only 1 point, PCA should still return valid axes."""
        mat = np.array([[1.0, 2.0, 3.0, 4.0]])
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 4)
        self.assertEqual(pc1.shape, (4,))
        self.assertEqual(pc2.shape, (4,))

    def test_pca_orthogonality(self):
        """PC1 and PC2 should be orthogonal."""
        mat = np.random.randn(50, 8)
        _, _, pc1, pc2, _, _ = app.compute_pca_basis(mat, 8)
        dot = np.dot(pc1, pc2)
        self.assertAlmostEqual(dot, 0.0, places=10)


# ===================================================================
# 8. GRID PROBE GENERATION
# ===================================================================
class TestGridProbes(unittest.TestCase):
    """Tests for Section 9: grid range, coords, interpolation, grid probes."""

    def test_compute_grid_range_padding(self):
        proj = np.array([0.0, 10.0])
        mn, mx, r = app.compute_grid_range(proj, pad_frac=0.3)
        self.assertLess(mn, 0.0)
        self.assertGreater(mx, 10.0)
        self.assertAlmostEqual(r, 10.0)

    def test_make_grid_coords_count(self):
        g1 = np.linspace(0, 1, 5)
        g2 = np.linspace(0, 1, 4)
        coords = app.make_grid_coords(g1, g2)
        self.assertEqual(len(coords), 20)

    def test_interpolate_grid_embedding(self):
        centroid = np.array([1.0, 2.0, 3.0])
        pc1 = np.array([1.0, 0.0, 0.0])
        pc2 = np.array([0.0, 1.0, 0.0])
        result = app.interpolate_grid_embedding(2.0, 3.0, centroid, pc1, pc2)
        np.testing.assert_allclose(result, [3.0, 5.0, 3.0])

    def test_compute_grid_weights_sum_to_one(self):
        existing = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        weights = app.compute_grid_weights(0.5, 0.5, existing, sigma_nn=1.0)
        self.assertAlmostEqual(weights.sum(), 1.0, places=10)

    def test_interpolate_deltas_shape(self):
        n_layers = 2
        hidden_dim = 4
        all_deltas = [
            [np.ones(hidden_dim), np.ones(hidden_dim) * 2],
            [np.ones(hidden_dim) * 3, np.ones(hidden_dim) * 4],
        ]
        weights = np.array([0.5, 0.5])
        result = app.interpolate_deltas(weights, all_deltas, n_layers, hidden_dim)
        self.assertEqual(len(result), n_layers)
        self.assertEqual(result[0].shape, (hidden_dim,))

    def test_create_grid_probes_count(self):
        n_layers = 2
        hidden_dim = 4
        n_side = 5
        centroid = np.zeros(hidden_dim)
        pc1 = np.array([1, 0, 0, 0], dtype=float)
        pc2 = np.array([0, 1, 0, 0], dtype=float)
        proj1 = np.random.randn(10)
        proj2 = np.random.randn(10)
        existing_proj = np.stack([proj1, proj2], axis=1)
        all_deltas = [[np.random.randn(hidden_dim) for _ in range(n_layers)] for _ in range(10)]

        gl0, gd = app.create_grid_probes(
            centroid, pc1, pc2, proj1, proj2, existing_proj,
            all_deltas, n_layers, hidden_dim, n_side=n_side, pad_frac=0.3,
        )
        self.assertEqual(len(gl0), n_side * n_side)
        self.assertEqual(len(gd), n_side * n_side)
        for d in gd:
            self.assertEqual(len(d), n_layers)


# ===================================================================
# 9. OUTPUT ASSEMBLY
# ===================================================================
class TestOutputAssembly(unittest.TestCase):
    """Tests for Section 10: build_fixed_pos, build_deltas_array, build_output_data."""

    def test_build_fixed_pos(self):
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = app.build_fixed_pos(vecs)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_build_deltas_array_shape(self):
        n_layers = 2
        n_points = 3
        all_deltas = [
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
            for _ in range(n_points)
        ]
        result = app.build_deltas_array(all_deltas, n_layers, n_points)
        self.assertEqual(len(result), n_layers)
        self.assertEqual(len(result[0]), n_points)

    def test_build_output_data_keys(self):
        data = app.build_output_data(
            all_labels=["a", "b"],
            all_is_real=[True, False],
            n_layers=2,
            n_total=2,
            n_real=1,
            hidden_dim=4,
            fixed_pos=[[1, 2, 3, 4], [5, 6, 7, 8]],
            deltas=[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]],
            model_name="gpt2",
            text="hello",
            neighbors=[],
        )
        expected_keys = {
            "tokens", "is_real", "n_layers", "n_points", "n_real",
            "n_synth", "hidden_dim", "fixed_pos", "deltas",
            "model_name", "text", "neighbors",
        }
        self.assertEqual(set(data.keys()), expected_keys)
        self.assertEqual(data["n_synth"], 1)

    def test_build_output_data_json_serializable(self):
        data = app.build_output_data(
            ["tok"], [True], 1, 1, 1, 2,
            [[0.0, 0.0]], [[[0.1, 0.2]]], "gpt2", "hi", [],
        )
        # Should not raise
        json.dumps(data)


# ===================================================================
# 10. HTTP SERVER
# ===================================================================
class TestHTTPHandlers(unittest.TestCase):
    """Tests for Section 13: handle_get_index, handle_post_run."""

    def test_handle_get_index_returns_html(self):
        result = app.handle_get_index()
        self.assertIsInstance(result, bytes)
        self.assertIn(b"<!DOCTYPE html>", result)
        self.assertIn(b"Metric Space Explorer", result)

    def test_handle_post_run_empty_text_raises(self):
        body = json.dumps({"text": ""}).encode()
        with self.assertRaises(ValueError):
            app.handle_post_run(body)

    @patch("app.process_text", return_value='{"ok":true}')
    def test_handle_post_run_valid(self, mock_pt):
        body = json.dumps({"text": "hello", "model": "gpt2"}).encode()
        result = app.handle_post_run(body)
        self.assertEqual(result, b'{"ok":true}')
        mock_pt.assert_called_once_with("hello", "gpt2")

    @patch("app.process_text", return_value='{"ok":true}')
    def test_handle_post_run_no_model(self, mock_pt):
        body = json.dumps({"text": "hello"}).encode()
        app.handle_post_run(body)
        mock_pt.assert_called_once_with("hello", None)


# ===================================================================
# 11. ARGUMENT PARSING
# ===================================================================
class TestArgParsing(unittest.TestCase):
    """Tests for Section 15: parse_args."""

    def test_defaults(self):
        args = app.parse_args([])
        self.assertEqual(args.model, "gpt2")
        self.assertEqual(args.port, 8765)

    def test_custom_model(self):
        args = app.parse_args(["--model", "gpt2-medium"])
        self.assertEqual(args.model, "gpt2-medium")

    def test_custom_port(self):
        args = app.parse_args(["--port", "9000"])
        self.assertEqual(args.port, 9000)


# ===================================================================
# 12. SERVER CREATION
# ===================================================================
class TestServerCreation(unittest.TestCase):
    """Tests for Section 16: start_server."""

    def test_start_server_returns_httpserver(self):
        server = app.start_server("127.0.0.1", 0)  # port 0 = OS picks
        self.assertIsInstance(server, HTTPServer)
        server.server_close()


# ===================================================================
# 13. INTEGRATION: HTTP ROUND-TRIP (with mocked model)
# ===================================================================
class TestHTTPIntegration(unittest.TestCase):
    """Spin up the real HTTP server, hit endpoints, verify responses."""

    @classmethod
    def setUpClass(cls):
        cls.server = app.start_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_index(self):
        url = f"http://127.0.0.1:{self.port}/"
        resp = urllib.request.urlopen(url)
        self.assertEqual(resp.status, 200)
        body = resp.read()
        self.assertIn(b"Metric Space Explorer", body)

    def test_get_404(self):
        url = f"http://127.0.0.1:{self.port}/nonexistent"
        try:
            urllib.request.urlopen(url)
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    @patch("app.process_text", return_value='{"test":"ok"}')
    def test_post_run(self, mock_pt):
        url = f"http://127.0.0.1:{self.port}/run"
        data = json.dumps({"text": "hello world"}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.read())
        self.assertEqual(body["test"], "ok")

    def test_post_run_empty_text_500(self):
        url = f"http://127.0.0.1:{self.port}/run"
        data = json.dumps({"text": ""}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected 500")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 500)


# ===================================================================
# 14. FULL PIPELINE (requires model — marked for slow/integration)
# ===================================================================
class TestFullPipeline(unittest.TestCase):
    """
    End-to-end test that loads a real model.
    Skipped unless RUN_SLOW_TESTS=1 is set.
    """

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("RUN_SLOW_TESTS"):
            raise unittest.SkipTest("Set RUN_SLOW_TESTS=1 to run model-loading tests")
        app.load_model("gpt2")

    def test_process_text_returns_valid_json(self):
        result = app.process_text("Hello world")
        data = json.loads(result)
        self.assertIn("tokens", data)
        self.assertIn("deltas", data)
        self.assertIn("fixed_pos", data)
        self.assertGreater(data["n_real"], 0)
        self.assertGreater(data["n_points"], data["n_real"])
        self.assertEqual(data["model_name"], "gpt2")

    def test_process_text_layer_count(self):
        result = app.process_text("Test")
        data = json.loads(result)
        self.assertEqual(data["n_layers"], 12)
        self.assertEqual(len(data["deltas"]), 12)

    def test_process_text_dimensions(self):
        result = app.process_text("A B C")
        data = json.loads(result)
        self.assertEqual(data["hidden_dim"], 768)
        for vec in data["fixed_pos"]:
            self.assertEqual(len(vec), 768)

    def test_neighbors_present(self):
        result = app.process_text("The cat sat")
        data = json.loads(result)
        self.assertIn("neighbors", data)
        self.assertEqual(len(data["neighbors"]), data["n_real"])
        for nlist in data["neighbors"]:
            self.assertGreater(len(nlist), 0)
            self.assertIn("dist", nlist[0])
            self.assertIn("label", nlist[0])


# ===================================================================
# 15. HTML CONTENT SANITY
# ===================================================================
class TestHTMLContent(unittest.TestCase):
    """Verify the embedded HTML page has expected UI elements."""

    def test_contains_canvas(self):
        self.assertIn('<canvas id="cv">', app.HTML_PAGE)

    def test_contains_run_button(self):
        self.assertIn('id="btn-run"', app.HTML_PAGE)

    def test_contains_layer_slider(self):
        self.assertIn('id="sl-layer"', app.HTML_PAGE)

    def test_contains_model_selector(self):
        self.assertIn('id="sel-model"', app.HTML_PAGE)

    def test_contains_keyboard_handler(self):
        self.assertIn("onKey", app.HTML_PAGE)

    def test_contains_draw_function(self):
        self.assertIn("function draw()", app.HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
