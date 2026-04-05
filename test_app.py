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
import unittest
from unittest.mock import patch, MagicMock
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
        """Delta[lay] should equal hidden_states[lay+1] - hidden_states[lay]."""
        hs = self._fake_hidden_states(n_layers=2, seq_len=2, dim=4)
        l0, dl = app.compute_layer0_and_deltas(hs, n_layers=2)
        for s in range(2):
            for lay in range(2):
                expected = (hs[lay + 1][0][s] - hs[lay][0][s]).cpu().numpy()
                np.testing.assert_allclose(dl[s][lay], expected, atol=1e-6)

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
        np.testing.assert_allclose(centroid, mat.astype(np.float32).mean(axis=0), atol=1e-6)

    def test_pca_centered_zero_mean(self):
        mat = np.random.randn(15, 4)
        _, centered, _, _, _, _ = app.compute_pca_basis(mat, 4)
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-6)

    def test_pca_single_point_fallback(self):
        """With only 1 point, PCA should still return valid axes."""
        mat = np.array([[1.0, 2.0, 3.0, 4.0]])
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 4)
        self.assertEqual(pc1.shape, (4,))
        self.assertEqual(pc2.shape, (4,))

    def test_pca_orthogonality(self):
        mat = np.random.randn(50, 8)
        _, _, pc1, pc2, _, _ = app.compute_pca_basis(mat, 8)
        dot = np.dot(pc1, pc2)
        self.assertAlmostEqual(dot, 0.0, places=5)

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

        gl0, gd, gadd, gmlpd = app.create_grid_probes(
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

# ===================================================================
# 16. EDGE CASES: ENVIRONMENT SAFETY
# ===================================================================
class TestEnvironmentSafetyEdgeCases(unittest.TestCase):
    """Additional edge-case tests for environment safety helpers."""

    def test_compute_exclude_newer_date_zero_days(self):
        """Zero days back should be approximately now."""
        from datetime import datetime
        result = app.compute_exclude_newer_date(days_back=0)
        now_str = datetime.utcnow().strftime("%Y-%m-%dT")
        self.assertTrue(result.startswith(now_str))

    def test_compute_exclude_newer_date_large_days(self):
        """Large days_back should still produce a valid timestamp."""
        result = app.compute_exclude_newer_date(days_back=365)
        self.assertRegex(result, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_compute_exclude_newer_monotonic(self):
        """More days back should always produce an earlier date."""
        dates = [app.compute_exclude_newer_date(d) for d in [1, 5, 10, 30, 100]]
        for i in range(len(dates) - 1):
            self.assertGreater(dates[i], dates[i + 1])


# ===================================================================
# 17. EDGE CASES: MODEL DETECTION
# ===================================================================
class TestModelDetectionEdgeCases(unittest.TestCase):
    """Additional edge-case tests for model detection helpers."""

    @staticmethod
    def _cfg(**kwargs):
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

    def test_detect_pythia(self):
        cfg = self._cfg(architectures=["GPTNeoXForCausalLM"])
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_detect_electra(self):
        cfg = self._cfg(architectures=["ElectraForPreTraining"])
        self.assertEqual(app.detect_model_type(cfg), "masked")

    def test_detect_empty_architectures_list(self):
        cfg = self._cfg(architectures=[])
        self.assertEqual(app.detect_model_type(cfg), "causal")

    def test_get_n_layers_num_layers_fallback(self):
        """Test the num_layers attribute (third priority)."""
        cfg = self._cfg()
        cfg.n_layer = None
        cfg.num_hidden_layers = None
        cfg.num_layers = 6
        self.assertEqual(app.get_n_layers(cfg), 6)

    def test_get_hidden_dim_d_model(self):
        """Test the d_model attribute (third priority)."""
        cfg = self._cfg()
        cfg.n_embd = None
        cfg.hidden_size = None
        cfg.d_model = 512
        self.assertEqual(app.get_hidden_dim(cfg), 512)

    def test_get_n_layers_priority_order(self):
        """n_layer should take priority over num_hidden_layers."""
        cfg = self._cfg(n_layer=10, num_hidden_layers=20)
        cfg.num_layers = 30
        self.assertEqual(app.get_n_layers(cfg), 10)

    def test_get_hidden_dim_priority_order(self):
        """n_embd should take priority over hidden_size."""
        cfg = self._cfg(n_embd=256, hidden_size=512)
        cfg.d_model = 1024
        self.assertEqual(app.get_hidden_dim(cfg), 256)


# ===================================================================
# 18. TOKENIZATION EDGE CASES
# ===================================================================
class TestTokenizationEdgeCases(unittest.TestCase):
    """Edge cases for tokenization functions."""

    def test_tokenize_text_single_token(self):
        import torch
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[42]])}
        tok.decode = lambda ids: "hello"
        ids, labels = app.tokenize_text(tok, "hello")
        self.assertEqual(ids.shape[1], 1)
        self.assertEqual(len(labels), 1)

    def test_decode_token_ids_no_special_chars(self):
        import torch
        tok = MagicMock()
        tok.decode = lambda ids: "plain"
        result = app.decode_token_ids(tok, torch.tensor([1, 2, 3]))
        self.assertEqual(result, ["plain", "plain", "plain"])

    def test_decode_token_ids_empty(self):
        import torch
        tok = MagicMock()
        result = app.decode_token_ids(tok, torch.tensor([]))
        self.assertEqual(result, [])

    def test_decode_token_ids_both_special_chars(self):
        """Token containing both Ġ and Ċ replacements."""
        import torch
        tok = MagicMock()
        tok.decode = lambda ids: "\u0120hello\u010aworld"
        result = app.decode_token_ids(tok, torch.tensor([1]))
        self.assertEqual(result[0], " hello\\nworld")


# ===================================================================
# 19. PROBE SENTENCES EDGE CASES
# ===================================================================
class TestProbeSentencesEdgeCases(unittest.TestCase):
    """Additional probe sentence tests."""

    def test_probe_texts_no_duplicates(self):
        probes = app.get_probe_texts()
        self.assertEqual(len(probes), len(set(probes)))

    def test_tokenize_probes_multiple_texts(self):
        import torch
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tok.decode = lambda ids: "x"
        seqs, labels, flags = app.tokenize_probes(tok, ["a", "b", "c"])
        self.assertEqual(len(seqs), 3)
        self.assertEqual(len(labels), 6)  # 2 tokens * 3 texts
        self.assertEqual(len(flags), 6)
        self.assertTrue(all(f is False for f in flags))

    def test_tokenize_probes_empty_list(self):
        tok = MagicMock()
        seqs, labels, flags = app.tokenize_probes(tok, [])
        self.assertEqual(len(seqs), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(flags), 0)


# ===================================================================
# 20. HIDDEN STATE EXTRACTION EDGE CASES
# ===================================================================
class TestHiddenStateExtractionEdgeCases(unittest.TestCase):
    """Edge cases for hidden state extraction."""

    def test_compute_layer0_and_deltas_single_token(self):
        import torch
        hs = tuple(torch.randn(1, 1, 4) for _ in range(3))  # 2 layers
        l0, dl = app.compute_layer0_and_deltas(hs, n_layers=2)
        self.assertEqual(len(l0), 1)
        self.assertEqual(len(dl), 1)
        self.assertEqual(len(dl[0]), 2)

    def test_compute_layer0_and_deltas_single_layer(self):
        import torch
        hs = tuple(torch.randn(1, 3, 8) for _ in range(2))  # 1 layer
        l0, dl = app.compute_layer0_and_deltas(hs, n_layers=1)
        self.assertEqual(len(l0), 3)
        for d in dl:
            self.assertEqual(len(d), 1)

    def test_extract_hidden_states_calls_model(self):
        import torch
        mock_model = MagicMock()
        mock_out = MagicMock()
        mock_out.hidden_states = (torch.randn(1, 2, 4),)
        mock_model.return_value = mock_out
        input_ids = torch.tensor([[1, 2]])
        result = app.extract_hidden_states(mock_model, input_ids)
        mock_model.assert_called_once_with(input_ids)
        self.assertEqual(len(result), 1)

    def test_run_all_sequences_single_sequence(self):
        import torch

        def fake_model(input_ids):
            out = MagicMock()
            out.hidden_states = tuple(torch.randn(1, input_ids.shape[1], 4) for _ in range(3))
            return out

        model = MagicMock(side_effect=fake_model)
        seqs = [torch.tensor([[10, 20, 30]])]
        l0, dl = app.run_all_sequences(model, seqs, n_layers=2)
        self.assertEqual(len(l0), 3)
        self.assertEqual(len(dl), 3)

    def test_run_all_sequences_empty(self):
        model = MagicMock()
        l0, dl = app.run_all_sequences(model, [], n_layers=2)
        self.assertEqual(len(l0), 0)
        self.assertEqual(len(dl), 0)


# ===================================================================
# 21. NEIGHBOR COMPUTATION EDGE CASES
# ===================================================================
class TestNeighborComputationEdgeCases(unittest.TestCase):
    """Edge cases for neighbor computation."""

    def test_find_k_neighbors_returns_sorted_by_distance(self):
        all_emb = np.array([
            [0.0, 0.0],
            [3.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a", "b", "c", "d"], [True]*4, k=3)
        dists = [r["dist"] for r in result]
        self.assertEqual(dists, sorted(dists))

    def test_find_k_neighbors_includes_is_real_flag(self):
        all_emb = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a", "b", "c"], [True, False, True], k=2)
        is_real_map = {r["idx"]: r["is_real"] for r in result}
        self.assertFalse(is_real_map[1])
        self.assertTrue(is_real_map[2])

    def test_compute_neighbors_empty_real(self):
        real = np.empty((0, 2))
        all_emb = np.array([[0.0, 0.0]])
        result = app.compute_neighbors(real, all_emb, ["a"], [False], k=5)
        self.assertEqual(len(result), 0)

    def test_neighbor_result_has_all_keys(self):
        all_emb = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a", "b"], [True, True], k=1)
        self.assertEqual(set(result[0].keys()), {"idx", "label", "dist", "is_real"})


# ===================================================================
# 22. PCA COMPUTATION EDGE CASES
# ===================================================================
class TestPCAComputationEdgeCases(unittest.TestCase):
    """Edge cases for PCA computation."""

    def test_pca_two_points(self):
        """With exactly 2 points, PCA should still work."""
        mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 3)
        self.assertEqual(centroid.shape, (3,))
        self.assertEqual(p1.shape, (2,))

    def test_pca_identical_points(self):
        """All identical points — SVD should still not crash."""
        mat = np.ones((5, 4))
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 4)
        np.testing.assert_allclose(centered, 0.0, atol=1e-10)

    def test_pca_projections_reconstruct(self):
        """Projections onto PC1/PC2 should allow approximate reconstruction in 2D."""
        mat = np.random.randn(30, 6)
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 6)
        # Reconstruction in the PC1-PC2 plane
        reconstructed = np.outer(p1, pc1) + np.outer(p2, pc2)
        # This should capture most variance for random data (not all)
        residual = centered - reconstructed
        # Residual norm should be less than original norm (PCA captures some variance)
        self.assertLess(np.linalg.norm(residual), np.linalg.norm(centered))

    def test_pca_pc_unit_vectors(self):
        """PC1 and PC2 should be approximately unit vectors."""
        mat = np.random.randn(20, 8)
        _, _, pc1, pc2, _, _ = app.compute_pca_basis(mat, 8)
        self.assertAlmostEqual(np.linalg.norm(pc1), 1.0, places=10)
        self.assertAlmostEqual(np.linalg.norm(pc2), 1.0, places=10)


# ===================================================================
# 23. GRID PROBE EDGE CASES
# ===================================================================
class TestGridProbeEdgeCases(unittest.TestCase):
    """Edge cases for grid probe generation."""

    def test_compute_grid_range_zero_range(self):
        """All identical projections should still produce a valid range."""
        proj = np.array([5.0, 5.0, 5.0])
        mn, mx, r = app.compute_grid_range(proj, pad_frac=0.3)
        self.assertEqual(r, 1.0)  # fallback
        self.assertLess(mn, 5.0)
        self.assertGreater(mx, 5.0)

    def test_make_grid_coords_1x1(self):
        g1 = np.array([0.0])
        g2 = np.array([0.0])
        coords = app.make_grid_coords(g1, g2)
        self.assertEqual(len(coords), 1)
        self.assertEqual(coords[0], (0.0, 0.0))

    def test_interpolate_grid_embedding_origin(self):
        centroid = np.array([10.0, 20.0])
        pc1 = np.array([1.0, 0.0])
        pc2 = np.array([0.0, 1.0])
        result = app.interpolate_grid_embedding(0.0, 0.0, centroid, pc1, pc2)
        np.testing.assert_allclose(result, centroid)

    def test_compute_grid_weights_at_existing_point(self):
        """Weight should be heavily concentrated on the nearest point."""
        existing = np.array([[0.0, 0.0], [10.0, 10.0]])
        weights = app.compute_grid_weights(0.0, 0.0, existing, sigma_nn=0.1)
        self.assertGreater(weights[0], 0.99)

    def test_interpolate_deltas_uniform_weights(self):
        n_layers = 2
        hidden_dim = 3
        all_deltas = [
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            [np.array([3.0, 4.0, 5.0]), np.array([6.0, 7.0, 8.0])],
        ]
        weights = np.array([0.5, 0.5])
        result = app.interpolate_deltas(weights, all_deltas, n_layers, hidden_dim)
        # Layer 0: 0.5*[1,2,3] + 0.5*[3,4,5] = [2,3,4]
        np.testing.assert_allclose(result[0], [2.0, 3.0, 4.0])
        # Layer 1: 0.5*[4,5,6] + 0.5*[6,7,8] = [5,6,7]
        np.testing.assert_allclose(result[1], [5.0, 6.0, 7.0])

    def test_create_grid_probes_n_side_1(self):
        """Minimal grid: 1x1 = 1 probe."""
        n_layers = 1
        hidden_dim = 2
        centroid = np.zeros(hidden_dim)
        pc1 = np.array([1.0, 0.0])
        pc2 = np.array([0.0, 1.0])
        proj1 = np.array([0.0, 1.0])
        proj2 = np.array([0.0, 1.0])
        existing_proj = np.stack([proj1, proj2], axis=1)
        all_deltas = [[np.ones(hidden_dim)] for _ in range(2)]

        gl0, gd, gadd, gmlpd = app.create_grid_probes(
            centroid, pc1, pc2, proj1, proj2, existing_proj,
            all_deltas, n_layers, hidden_dim, n_side=1, pad_frac=0.3,
        )
        self.assertEqual(len(gl0), 1)
        self.assertEqual(len(gd), 1)

    def test_compute_grid_range_negative_values(self):
        proj = np.array([-10.0, -5.0, -1.0])
        mn, mx, r = app.compute_grid_range(proj, pad_frac=0.3)
        self.assertAlmostEqual(r, 9.0)
        self.assertLess(mn, -10.0)
        self.assertGreater(mx, -1.0)


# ===================================================================
# 24. OUTPUT ASSEMBLY EDGE CASES
# ===================================================================
class TestOutputAssemblyEdgeCases(unittest.TestCase):
    """Edge cases for output assembly."""

    def test_build_fixed_pos_empty(self):
        result = app.build_fixed_pos([])
        self.assertEqual(result, [])

    def test_build_deltas_array_single_layer_single_point(self):
        all_deltas = [[np.array([1.0, 2.0])]]
        result = app.build_deltas_array(all_deltas, n_layers=1, n_points=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0], [1.0, 2.0])

    def test_build_output_data_n_synth_calculation(self):
        data = app.build_output_data(
            ["a", "b", "c"], [True, True, False],
            n_layers=1, n_total=3, n_real=2, hidden_dim=2,
            fixed_pos=[[0, 0], [1, 1], [2, 2]],
            deltas=[[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]],
            model_name="test", text="ab", neighbors=[],
        )
        self.assertEqual(data["n_synth"], 1)
        self.assertEqual(data["n_real"], 2)
        self.assertEqual(data["n_points"], 3)

    def test_build_output_data_preserves_all_labels(self):
        labels = ["tok1", "tok2", "probe1"]
        data = app.build_output_data(
            labels, [True, True, False],
            1, 3, 2, 2,
            [[0, 0]] * 3, [[[0, 0]] * 3],
            "gpt2", "test", [],
        )
        self.assertEqual(data["tokens"], labels)

    def test_build_deltas_array_values_correct(self):
        """Verify the transposition from per-point to per-layer is correct."""
        # Point 0: layer0=[1,2], layer1=[3,4]
        # Point 1: layer0=[5,6], layer1=[7,8]
        all_deltas = [
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
        ]
        result = app.build_deltas_array(all_deltas, n_layers=2, n_points=2)
        # result[layer][point]
        self.assertEqual(result[0][0], [1.0, 2.0])
        self.assertEqual(result[0][1], [5.0, 6.0])
        self.assertEqual(result[1][0], [3.0, 4.0])
        self.assertEqual(result[1][1], [7.0, 8.0])


# ===================================================================
# 25. HTTP HANDLER EDGE CASES
# ===================================================================
class TestHTTPHandlerEdgeCases(unittest.TestCase):
    """Edge cases for HTTP handlers."""

    def test_handle_post_run_whitespace_only_text(self):
        body = json.dumps({"text": "   "}).encode()
        with self.assertRaises(ValueError):
            app.handle_post_run(body)

    def test_handle_post_run_missing_text_key(self):
        body = json.dumps({"model": "gpt2"}).encode()
        with self.assertRaises(ValueError):
            app.handle_post_run(body)

    def test_handle_post_run_empty_model_uses_none(self):
        with patch("app.process_text", return_value='{}') as mock_pt:
            body = json.dumps({"text": "hello", "model": ""}).encode()
            app.handle_post_run(body)
            mock_pt.assert_called_once_with("hello", None)

    def test_handle_post_run_invalid_json(self):
        body = b"not json at all"
        with self.assertRaises(json.JSONDecodeError):
            app.handle_post_run(body)

    def test_handle_get_index_encoding(self):
        result = app.handle_get_index()
        # Should be valid UTF-8
        decoded = result.decode("utf-8")
        self.assertIn("<!DOCTYPE html>", decoded)


# ===================================================================
# 26. ARGUMENT PARSING EDGE CASES
# ===================================================================
class TestArgParsingEdgeCases(unittest.TestCase):
    """Edge cases for argument parsing."""

    def test_unknown_model_name_accepted(self):
        """parse_args doesn't validate model names — any string is fine."""
        args = app.parse_args(["--model", "some-random/model-name"])
        self.assertEqual(args.model, "some-random/model-name")

    def test_port_zero(self):
        args = app.parse_args(["--port", "0"])
        self.assertEqual(args.port, 0)

    def test_both_args(self):
        args = app.parse_args(["--model", "bert-base-uncased", "--port", "3000"])
        self.assertEqual(args.model, "bert-base-uncased")
        self.assertEqual(args.port, 3000)


# ===================================================================
# 27. HTML CONTENT ADDITIONAL CHECKS
# ===================================================================
class TestHTMLContentAdditional(unittest.TestCase):
    """Additional HTML content sanity checks."""

    def test_contains_deformation_slider(self):
        self.assertIn('id="sl-t"', app.HTML_PAGE)

    def test_contains_amplify_slider(self):
        self.assertIn('id="sl-amp"', app.HTML_PAGE)

    def test_contains_dim_x_slider(self):
        self.assertIn('id="sl-dx"', app.HTML_PAGE)

    def test_contains_dim_y_slider(self):
        self.assertIn('id="sl-dy"', app.HTML_PAGE)

    def test_contains_sigma_slider(self):
        self.assertIn('id="sl-sig"', app.HTML_PAGE)

    def test_contains_grid_resolution_slider(self):
        self.assertIn('id="sl-gr"', app.HTML_PAGE)

    def test_contains_mode_selector(self):
        self.assertIn('id="sel-mode"', app.HTML_PAGE)

    def test_contains_all_mode_options(self):
        for mode in ["single", "cumfwd", "cumbwd", "embedding"]:
            self.assertIn(f'value="{mode}"', app.HTML_PAGE)

    def test_contains_strain_color_function(self):
        self.assertIn("function s2c(", app.HTML_PAGE)

    def test_contains_autoplay_function(self):
        self.assertIn("function togAP()", app.HTML_PAGE)

    def test_contains_reset_function(self):
        self.assertIn("function rstAll()", app.HTML_PAGE)

    def test_contains_fetch_run_endpoint(self):
        self.assertIn("fetch('/run'", app.HTML_PAGE)

    def test_contains_neighbor_panel(self):
        self.assertIn('id="neighbor-panel"', app.HTML_PAGE)

    def test_contains_all_model_options(self):
        for model in ["gpt2", "gpt2-medium", "gpt2-large", "bert-base-uncased"]:
            self.assertIn(f'value="{model}"', app.HTML_PAGE)

    def test_contains_checkbox_controls(self):
        for cb in ["cb-grid", "cb-heat", "cb-ref", "cb-tok", "cb-syn", "cb-sc", "cb-vec", "cb-nb"]:
            self.assertIn(f'id="{cb}"', app.HTML_PAGE)

    def test_html_is_valid_utf8_string(self):
        self.assertIsInstance(app.HTML_PAGE, str)
        # Should encode without error
        app.HTML_PAGE.encode("utf-8")


# ===================================================================
# 28. INTEGRATION: CONCURRENT HTTP REQUESTS
# ===================================================================
class TestHTTPConcurrency(unittest.TestCase):
    """Test that the server handles multiple concurrent requests."""

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

    def test_multiple_get_requests(self):
        """Multiple GET requests should all succeed."""
        results = []
        def fetch():
            url = f"http://127.0.0.1:{self.port}/"
            resp = urllib.request.urlopen(url)
            results.append(resp.status)

        threads = [threading.Thread(target=fetch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(s == 200 for s in results))

    def test_post_method_not_allowed_on_index(self):
        """POST to / should return 404 (only /run accepts POST)."""
        url = f"http://127.0.0.1:{self.port}/"
        data = b'{"text":"hello"}'
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected error")
        except urllib.error.HTTPError as e:
            # POST to / goes to do_POST which checks path — should be 404
            self.assertEqual(e.code, 404)


# ===================================================================
# 29. NUMERICAL STABILITY
# ===================================================================
class TestNumericalStability(unittest.TestCase):
    """Tests for numerical edge cases."""

    def test_find_k_neighbors_identical_points(self):
        """All points at the same location."""
        all_emb = np.zeros((5, 3))
        result = app.find_k_neighbors(0, all_emb[0], all_emb, ["a","b","c","d","e"], [True]*5, k=4)
        # All distances should be 0 (or inf for self)
        for r in result:
            self.assertAlmostEqual(r["dist"], 0.0, places=10)

    def test_pca_large_dimension_mismatch(self):
        """PCA with more dimensions than samples should still work."""
        mat = np.random.randn(3, 100)  # 3 samples, 100 dims
        centroid, centered, pc1, pc2, p1, p2 = app.compute_pca_basis(mat, 100)
        self.assertEqual(pc1.shape, (100,))
        self.assertEqual(pc2.shape, (100,))
        self.assertEqual(p1.shape, (3,))

    def test_interpolate_deltas_single_point(self):
        """Single point with weight 1.0 should return its deltas exactly."""
        n_layers = 3
        hidden_dim = 4
        all_deltas = [[np.array([1.0, 2.0, 3.0, 4.0]),
                        np.array([5.0, 6.0, 7.0, 8.0]),
                        np.array([9.0, 10.0, 11.0, 12.0])]]
        weights = np.array([1.0])
        result = app.interpolate_deltas(weights, all_deltas, n_layers, hidden_dim)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(result[1], [5.0, 6.0, 7.0, 8.0])
        np.testing.assert_allclose(result[2], [9.0, 10.0, 11.0, 12.0])

    def test_build_fixed_pos_preserves_precision(self):
        """Ensure float64 precision is maintained through conversion."""
        vec = np.array([1.123456789012345, 2.987654321098765])
        result = app.build_fixed_pos([vec])
        self.assertAlmostEqual(result[0][0], 1.123456789012345, places=12)
        self.assertAlmostEqual(result[0][1], 2.987654321098765, places=12)


# ===================================================================
# 30. PROCESS TEXT PIPELINE (mocked model)
# ===================================================================
class TestProcessTextMocked(unittest.TestCase):
    """Test the process_text pipeline with a fully mocked model."""

    def setUp(self):
        import torch
        self.orig_tokenizer = app.TOKENIZER
        self.orig_model = app.MODEL
        self.orig_config = app.MODEL_CONFIG
        self.orig_name = app.MODEL_NAME

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[10, 20, 30]])}
        mock_tok.decode = lambda ids: {10: "Hello", 20: " world", 30: "!"}.get(ids[0], "?")

        # Mock model that returns hidden states
        def fake_model_call(input_ids):
            seq_len = input_ids.shape[1]
            out = MagicMock()
            # 2 layers => 3 hidden state tensors
            out.hidden_states = tuple(
                torch.randn(1, seq_len, 8) for _ in range(3)
            )
            return out

        mock_model = MagicMock(side_effect=fake_model_call)

        # Mock config
        mock_config = MagicMock()
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_config.is_decoder = True
        mock_config.n_layer = 2
        mock_config.num_hidden_layers = None
        mock_config.num_layers = None
        mock_config.n_embd = 8
        mock_config.hidden_size = None
        mock_config.d_model = None

        app.TOKENIZER = mock_tok
        app.MODEL = mock_model
        app.MODEL_CONFIG = mock_config
        app.MODEL_NAME = "test-model"

    def tearDown(self):
        app.TOKENIZER = self.orig_tokenizer
        app.MODEL = self.orig_model
        app.MODEL_CONFIG = self.orig_config
        app.MODEL_NAME = self.orig_name

    def test_process_text_returns_valid_json(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        self.assertIn("tokens", data)
        self.assertIn("deltas", data)
        self.assertIn("fixed_pos", data)
        self.assertEqual(data["n_real"], 3)
        self.assertEqual(data["n_layers"], 2)
        self.assertEqual(data["hidden_dim"], 8)
        self.assertEqual(data["model_name"], "test-model")

    def test_process_text_has_grid_probes(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        # Should have real + probe + grid points
        self.assertGreater(data["n_points"], data["n_real"])
        self.assertGreater(data["n_synth"], 0)

    def test_process_text_deltas_shape(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        # deltas[layer][point] should have hidden_dim elements
        self.assertEqual(len(data["deltas"]), 2)  # n_layers
        for layer_deltas in data["deltas"]:
            self.assertEqual(len(layer_deltas), data["n_points"])
            for delta_vec in layer_deltas:
                self.assertEqual(len(delta_vec), 8)  # hidden_dim

    def test_process_text_fixed_pos_shape(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        self.assertEqual(len(data["fixed_pos"]), data["n_points"])
        for vec in data["fixed_pos"]:
            self.assertEqual(len(vec), 8)

    def test_process_text_neighbors_present(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        self.assertIn("neighbors", data)
        self.assertEqual(len(data["neighbors"]), data["n_real"])
        for nlist in data["neighbors"]:
            self.assertIsInstance(nlist, list)
            self.assertGreater(len(nlist), 0)

    def test_process_text_is_real_flags(self):
        result = app.process_text("Hello world!")
        data = json.loads(result)
        # First n_real should be True, rest False
        for i in range(data["n_real"]):
            self.assertTrue(data["is_real"][i])
        for i in range(data["n_real"], data["n_points"]):
            self.assertFalse(data["is_real"][i])

    def test_process_text_model_switch(self):
        """Requesting a different model name should trigger load_model."""
        with patch("app.load_model") as mock_load:
            app.process_text("Hello", "different-model")
            mock_load.assert_called_once_with("different-model")

    def test_process_text_same_model_no_reload(self):
        """Same model name should NOT trigger load_model."""
        with patch("app.load_model") as mock_load:
            app.process_text("Hello", "test-model")
            mock_load.assert_not_called()


# ===================================================================
# 31. LOAD MODEL FUNCTION
# ===================================================================
class TestLoadModel(unittest.TestCase):
    """Tests for the load_model function."""

    @patch("app.AutoModel")
    @patch("app.AutoTokenizer")
    def test_load_model_sets_globals(self, mock_tok_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.architectures = ["GPT2LMHeadModel"]
        mock_model.config.n_layer = 6
        mock_model.config.n_embd = 256
        mock_model.config.num_hidden_layers = None
        mock_model.config.num_layers = None
        mock_model.config.hidden_size = None
        mock_model.config.d_model = None
        mock_model.config.is_decoder = True
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        app.load_model("test-gpt2")

        self.assertEqual(app.MODEL_NAME, "test-gpt2")
        self.assertIsNotNone(app.TOKENIZER)
        self.assertIsNotNone(app.MODEL)
        self.assertIsNotNone(app.MODEL_CONFIG)
        mock_model.eval.assert_called_once()

    @patch("app.AutoModel")
    @patch("app.AutoTokenizer")
    def test_load_model_calls_pretrained(self, mock_tok_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        app.load_model("my-model")

        mock_tok_cls.from_pretrained.assert_called_once_with("my-model")
        mock_model_cls.from_pretrained.assert_called_once_with("my-model", output_hidden_states=True)


# ===================================================================
# 32. BROWSER OPENER
# ===================================================================
class TestBrowserOpener(unittest.TestCase):
    """Tests for the open_browser_delayed function."""

    @patch("app.webbrowser.open")
    def test_open_browser_delayed_calls_open(self, mock_open):
        app.open_browser_delayed(9999, delay=0.01)
        time.sleep(0.1)
        mock_open.assert_called_once_with("http://127.0.0.1:9999")

    @patch("app.webbrowser.open")
    def test_open_browser_delayed_correct_url(self, mock_open):
        app.open_browser_delayed(1234, delay=0.01)
        time.sleep(0.1)
        mock_open.assert_called_with("http://127.0.0.1:1234")


# ===================================================================
# 33. RESTART WITH UV
# ===================================================================
class TestRestartWithUV(unittest.TestCase):
    """Tests for restart_with_uv (mocked to prevent actual exec)."""

    @patch("os.execvpe")
    def test_restart_with_uv_calls_execvpe(self, mock_exec):
        app.restart_with_uv("/path/to/app.py", ["--port", "8000"], {"KEY": "VAL"})
        mock_exec.assert_called_once_with(
            "uv",
            ["uv", "run", "--quiet", "/path/to/app.py", "--port", "8000"],
            {"KEY": "VAL"},
        )

    @patch("os.execvpe")
    def test_restart_with_uv_no_args(self, mock_exec):
        app.restart_with_uv("/app.py", [], {"UV_EXCLUDE_NEWER": "2024-01-01"})
        mock_exec.assert_called_once_with(
            "uv",
            ["uv", "run", "--quiet", "/app.py"],
            {"UV_EXCLUDE_NEWER": "2024-01-01"},
        )


# ===================================================================
# 34. HANDLER CLASS DIRECT TESTS
# ===================================================================
class TestHandlerRouting(unittest.TestCase):
    """Test HTTP handler routing via a live server."""

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

    def test_get_index_html_content_type(self):
        url = f"http://127.0.0.1:{self.port}/"
        resp = urllib.request.urlopen(url)
        ct = resp.headers.get("Content-Type", "")
        self.assertIn("text/html", ct)

    def test_get_index_html_alias(self):
        url = f"http://127.0.0.1:{self.port}/index.html"
        resp = urllib.request.urlopen(url)
        self.assertEqual(resp.status, 200)
        body = resp.read()
        self.assertIn(b"Metric Space Explorer", body)

    def test_post_to_unknown_path_404(self):
        url = f"http://127.0.0.1:{self.port}/unknown"
        data = b'{"text":"hello"}'
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    @patch("app.process_text", return_value='{"result":"ok"}')
    def test_post_run_content_type(self, mock_pt):
        url = f"http://127.0.0.1:{self.port}/run"
        data = json.dumps({"text": "test"}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req)
        ct = resp.headers.get("Content-Type", "")
        self.assertIn("application/json", ct)

    def test_post_run_error_returns_json(self):
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
            body = json.loads(e.read())
            self.assertIn("error", body)


# ===================================================================
# 35. MAIN FUNCTION
# ===================================================================
class TestMainFunction(unittest.TestCase):
    """Tests for the main() entry point."""

    @patch("app.HTTPServer")
    @patch("app.load_model")
    @patch("app.open_browser_delayed")
    @patch("app.parse_args")
    def test_main_calls_load_model(self, mock_parse, mock_browser, mock_load, mock_server):
        mock_args = MagicMock()
        mock_args.model = "gpt2-medium"
        mock_args.port = 9999
        mock_parse.return_value = mock_args
        mock_srv_instance = MagicMock()
        mock_srv_instance.serve_forever = MagicMock(side_effect=KeyboardInterrupt)
        mock_server.return_value = mock_srv_instance

        try:
            app.main()
        except KeyboardInterrupt:
            pass

        mock_load.assert_called_once_with("gpt2-medium")


if __name__ == "__main__":
    unittest.main()

