#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
# ]
# ///

import argparse
import os
import sys
import json
import threading
from urllib.parse import urlparse
from datetime import datetime, timedelta, UTC


# ============================================================
# 1. ENVIRONMENT SAFETY
# ============================================================

def compute_exclude_newer_date(days_back=8):
    """Return a UTC timestamp string for `days_back` days ago."""
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")


def should_set_exclude_newer():
    """Check whether UV_EXCLUDE_NEWER is already set."""
    return not os.environ.get("UV_EXCLUDE_NEWER")


def restart_with_uv(script_path, args, env):
    """Re-exec the current script under `uv run`."""
    os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)


def ensure_safe_env():
    """Ensure uv only uses packages at least 8 days old."""
    if not should_set_exclude_newer():
        return

    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date

    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


# This must run BEFORE heavy imports
ensure_safe_env()

import webbrowser # noqa: E402
import time # noqa: E402
import numpy as np # noqa: E402
import torch # noqa: E402
from transformers import AutoTokenizer, AutoModel # noqa: E402
from http.server import HTTPServer, BaseHTTPRequestHandler # noqa: E402

# ============================================================
# 2. MODEL DETECTION & CONFIG HELPERS
# ============================================================

def detect_model_type(config):
    """Detect whether a model is 'causal' (decoder) or 'masked' (encoder)."""
    arch = getattr(config, "architectures", []) or []
    arch_str = " ".join(arch).lower()
    if any(k in arch_str for k in ["causal", "gpt", "opt", "pythia", "neox"]):
        return "causal"
    if any(k in arch_str for k in ["masked", "bert", "roberta", "electra"]):
        return "masked"
    if getattr(config, "is_decoder", False):
        return "causal"
    return "causal"


def get_n_layers(config):
    """Extract the number of layers from a model config."""
    for attr in ["n_layer", "num_hidden_layers", "num_layers"]:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return 12


def get_hidden_dim(config):
    """Extract the hidden dimension from a model config."""
    for attr in ["n_embd", "hidden_size", "d_model"]:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return 768


# ============================================================
# 3. MODEL LOADING (stateful — thin wrapper)
# ============================================================

MODEL_NAME = "gpt2"
TOKENIZER = None
MODEL = None
LM_MODEL = None
MODEL_CONFIG = None


def load_model(model_name):
    global TOKENIZER, MODEL, LM_MODEL, MODEL_NAME, MODEL_CONFIG
    MODEL_NAME = model_name
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    MODEL = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    MODEL.eval()
    MODEL_CONFIG = MODEL.config

    # Also load the LM head version for next-token prediction
    mtype = detect_model_type(MODEL_CONFIG)
    LM_MODEL = None
    if mtype == "causal":
        try:
            from transformers import AutoModelForCausalLM
            LM_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
            LM_MODEL.eval()
        except Exception as e:
            print(f"[Model] Could not load LM head: {e}")

# ============================================================
# 4. TOKENIZATION
# ============================================================

def tokenize_text(tokenizer, text):
    """Tokenize text and return (input_ids tensor, clean token strings)."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tokens_clean = decode_token_ids(tokenizer, input_ids[0])
    return input_ids, tokens_clean


def decode_token_ids(tokenizer, token_ids):
    """Decode a 1-D tensor of token IDs into cleaned strings."""
    tokens = []
    for tid in token_ids:
        t = tokenizer.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")
        tokens.append(t)
    return tokens


# ============================================================
# 5. PROBE SENTENCES
# ============================================================

def get_probe_texts():
    """Return a list of diverse probe sentences for embedding coverage."""
    return [
        "zero one two three four five six seven eight nine ten eleven twelve",
        "the a an and or but if then else while for each every some",
        "red blue green yellow purple orange black white pink gray brown",
        "happy sad angry calm excited tired hungry cold hot fast slow quiet",
        "dog cat bird fish horse cow pig sheep duck mouse rabbit snake",
        "water fire earth wind light dark sun moon star rain snow ice",
        "run walk jump fly swim climb fall push pull throw catch hold",
        "big small tall short wide narrow deep shallow thick thin long",
        "house tree road river mountain city forest ocean desert lake field",
        "king queen prince knight wizard dragon castle sword shield gold silver",
        "computer science math physics chemistry biology history music art language",
        "morning evening night today tomorrow yesterday always never often sometimes",
        "love hate fear hope dream think know believe want need feel see",
        "north south east west up down left right front back inside outside",
        "eat drink sleep wake work play read write speak listen learn teach",
        "mother father sister brother daughter son friend enemy teacher student",
        "car train plane boat bicycle truck bus ship rocket helicopter engine",
        "spring summer autumn winter january february march april may june july",
        "dollar euro pound yen bitcoin price cost value money bank trade",
        "hello goodbye please thanks sorry yes no maybe okay sure right wrong",
    ]


def tokenize_probes(tokenizer, probe_texts):
    """Tokenize all probe texts. Returns list of input_id tensors, labels, is_real flags."""
    all_seqs = []
    all_labels = []
    all_is_real = []
    for ptxt in probe_texts:
        pi = tokenizer(ptxt, return_tensors="pt")
        all_seqs.append(pi["input_ids"])
        for tid in pi["input_ids"][0]:
            t = tokenizer.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")
            all_labels.append(t)
        all_is_real.extend([False] * pi["input_ids"].shape[1])
    return all_seqs, all_labels, all_is_real


# ============================================================
# 6. HIDDEN STATE EXTRACTION
# ============================================================

def extract_hidden_states(model, input_ids):
    """Run a single sequence through the model, return hidden_states tuple."""
    with torch.no_grad():
        out = model(input_ids)
    return out.hidden_states


def compute_layer0_and_deltas(hidden_states, n_layers):
    """
    From hidden_states for one sequence, extract per-token:
      - layer-0 embedding (numpy)
      - list of per-layer deltas (numpy)
    Returns (list_of_layer0_vecs, list_of_delta_lists).
    """
    hs = hidden_states
    seq_len = hs[0].shape[1]
    layer0_vecs = []
    delta_lists = []
    for s in range(seq_len):
        layer0_vecs.append(hs[0][0][s].cpu().numpy())
        deltas = []
        for lay in range(n_layers):
            deltas.append((hs[lay + 1][0][s] - hs[lay][0][s]).cpu().numpy())
        delta_lists.append(deltas)
    return layer0_vecs, delta_lists


def run_all_sequences(model, all_seqs, n_layers):
    """
    Run every sequence through the model and collect layer0 + deltas.
    Returns (all_layer0, all_deltas_per_point).
    """
    all_layer0 = []
    all_deltas = []
    for seq_ids in all_seqs:
        hs = extract_hidden_states(model, seq_ids)
        l0, dl = compute_layer0_and_deltas(hs, n_layers)
        all_layer0.extend(l0)
        all_deltas.extend(dl)
    return all_layer0, all_deltas


# ============================================================
# 6b. COMPONENT-DECOMPOSED HIDDEN STATE EXTRACTION
#     (Attention vs MLP contribution per layer)
# ============================================================

def _get_transformer_blocks(model):
    """Return the list of transformer block modules for supported architectures."""
    # GPT-2
    if hasattr(model, 'h'):
        return list(model.h)
    # BERT / RoBERTa / DistilBERT
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return list(model.encoder.layer)
    # Pythia / GPT-NeoX
    if hasattr(model, 'layers'):
        return list(model.layers)
    # OPT
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        return list(model.decoder.layers)
    return []


def extract_component_deltas(model, input_ids, n_layers, hidden_dim):
    """
    Run a single sequence through the model with hooks to capture
    attention-output and MLP-output contributions separately.

    For GPT-2 style models, each block computes:
        attn_out = block.attn(ln_1(h))
        mlp_out  = block.mlp(ln_2(h + attn_out))
        h_next   = h + attn_out + mlp_out

    So the residual delta = attn_out + mlp_out.

    Returns:
        attn_deltas: list of list of numpy arrays [seq_len][n_layers]
        mlp_deltas:  list of list of numpy arrays [seq_len][n_layers]
    """
    blocks = _get_transformer_blocks(model)
    if len(blocks) == 0 or len(blocks) != n_layers:
        return None, None

    # Storage for captured outputs: layer -> tensor
    attn_outputs = {}
    mlp_outputs = {}
    hooks = []

    def _make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            # GPT-2 attn returns (attn_output, present, (attentions))
            # BERT self-attention output layer returns (hidden_states,) or tuple
            if isinstance(output, tuple):
                attn_outputs[layer_idx] = output[0].detach()
            else:
                attn_outputs[layer_idx] = output.detach()
        return hook_fn

    def _make_mlp_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                mlp_outputs[layer_idx] = output[0].detach()
            else:
                mlp_outputs[layer_idx] = output.detach()
        return hook_fn

    # Register hooks on each block's attention and MLP sub-modules
    for layer_idx, block in enumerate(blocks):
        # GPT-2: block.attn, block.mlp
        if hasattr(block, 'attn') and hasattr(block, 'mlp'):
            hooks.append(block.attn.register_forward_hook(_make_attn_hook(layer_idx)))
            hooks.append(block.mlp.register_forward_hook(_make_mlp_hook(layer_idx)))
        # BERT: block.attention, block.intermediate + block.output
        elif hasattr(block, 'attention') and hasattr(block, 'output'):
            hooks.append(block.attention.register_forward_hook(_make_attn_hook(layer_idx)))
            # For BERT, the "MLP" is intermediate + output, but the output module
            # also includes the residual. We hook the intermediate for the MLP signal.
            if hasattr(block, 'intermediate'):
                hooks.append(block.intermediate.register_forward_hook(_make_mlp_hook(layer_idx)))
            else:
                hooks.append(block.output.register_forward_hook(_make_mlp_hook(layer_idx)))
        # DistilBERT: block.attention, block.ffn
        elif hasattr(block, 'attention') and hasattr(block, 'ffn'):
            hooks.append(block.attention.register_forward_hook(_make_attn_hook(layer_idx)))
            hooks.append(block.ffn.register_forward_hook(_make_mlp_hook(layer_idx)))
        else:
            # Fallback: can't decompose this architecture
            for h in hooks:
                h.remove()
            return None, None

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    seq_len = input_ids.shape[1]

    # Build per-token, per-layer attn and mlp delta arrays
    attn_deltas_per_token = []  # [seq_len] x [n_layers] numpy arrays
    mlp_deltas_per_token = []

    for s in range(seq_len):
        attn_list = []
        mlp_list = []
        for lay in range(n_layers):
            if lay in attn_outputs:
                attn_list.append(attn_outputs[lay][0][s].cpu().numpy())
            else:
                attn_list.append(np.zeros(hidden_dim))
            if lay in mlp_outputs:
                mlp_list.append(mlp_outputs[lay][0][s].cpu().numpy())
            else:
                mlp_list.append(np.zeros(hidden_dim))
        attn_deltas_per_token.append(attn_list)
        mlp_deltas_per_token.append(mlp_list)

    return attn_deltas_per_token, mlp_deltas_per_token


def run_all_sequences_with_components(model, all_seqs, n_layers, hidden_dim):
    """
    Run every sequence through the model and collect:
      - layer0 embeddings
      - full residual deltas
      - attention-only deltas
      - MLP-only deltas
    Returns (all_layer0, all_deltas, all_attn_deltas, all_mlp_deltas).
    If decomposition fails, attn/mlp deltas will be None.
    """
    all_layer0 = []
    all_deltas = []
    all_attn_deltas = []
    all_mlp_deltas = []
    decomposition_ok = True

    for seq_ids in all_seqs:
        hs = extract_hidden_states(model, seq_ids)
        l0, dl = compute_layer0_and_deltas(hs, n_layers)
        all_layer0.extend(l0)
        all_deltas.extend(dl)

        if decomposition_ok:
            ad, md = extract_component_deltas(model, seq_ids, n_layers, hidden_dim)
            if ad is not None and md is not None:
                all_attn_deltas.extend(ad)
                all_mlp_deltas.extend(md)
            else:
                decomposition_ok = False

    if not decomposition_ok:
        return all_layer0, all_deltas, None, None

    return all_layer0, all_deltas, all_attn_deltas, all_mlp_deltas


# ============================================================
# 7. NEIGHBOR COMPUTATION
# ============================================================

def compute_neighbors(real_embeddings, all_embeddings, all_labels, all_is_real, k=10):
    """
    For each real token, find K nearest neighbors among all tokens.
    Returns list of neighbor-lists (one per real token).
    """
    n_real = real_embeddings.shape[0]
    neighbors = []
    for ri in range(n_real):
        nlist = find_k_neighbors(ri, real_embeddings[ri], all_embeddings, all_labels, all_is_real, k)
        neighbors.append(nlist)
    return neighbors


def find_k_neighbors(self_idx, query_vec, all_embeddings, all_labels, all_is_real, k):
    """Find k nearest neighbors for a single query vector, excluding self_idx."""
    dists = np.linalg.norm(all_embeddings - query_vec, axis=1)
    dists[self_idx] = np.inf
    nearest_idx = np.argsort(dists)[:k]
    result = []
    for ni in nearest_idx:
        result.append({
            "idx": int(ni),
            "label": all_labels[ni],
            "dist": float(dists[ni]),
            "is_real": all_is_real[ni],
        })
    return result

# ============================================================
# 7b. NEXT TOKEN PREDICTION & VOCABULARY NEIGHBORS
# ============================================================

def predict_next_token(tokenizer, model, input_ids, model_config, k=5):
    """Predict top-k next tokens using the pre-loaded LM model if available."""
    global LM_MODEL
    try:
        if LM_MODEL is None:
            print("[Model] No LM head model loaded — skipping next-token prediction")
            return []

        with torch.no_grad():
            outputs = LM_MODEL(input_ids)
            logits = outputs.logits[0, -1, :]  # last token's logits
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k)
            results = []
            for i in range(k):
                tid = topk.indices[i].item()
                prob = topk.values[i].item()
                token_str = tokenizer.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")
                results.append({"token": token_str, "prob": round(prob, 4)})
            return results
    except Exception as e:
        print(f"[Model] Next-token prediction failed: {e}")
        return []


def find_vocab_neighbors(tokenizer, model, layer0_vecs, n_real, k=5):
    """
    For each real token, find k nearest vocabulary tokens in embedding space.
    Returns list of lists (one per real token).
    """
    try:
        # Get the embedding matrix
        if hasattr(model, 'wte'):
            emb_matrix = model.wte.weight.detach().cpu().numpy()
        elif hasattr(model, 'embeddings'):
            emb_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
        elif hasattr(model, 'embed_tokens'):
            emb_matrix = model.embed_tokens.weight.detach().cpu().numpy()
        else:
            print("[Model] Could not find embedding matrix for vocab neighbors")
            return [[] for _ in range(n_real)]

        vocab_size = emb_matrix.shape[0]
        results = []
        for ri in range(n_real):
            vec = layer0_vecs[ri]
            dists = np.linalg.norm(emb_matrix - vec, axis=1)
            nearest = np.argsort(dists)[:k + 1]  # +1 to skip self
            neighbors = []
            for ni in nearest:
                token_str = tokenizer.decode([int(ni)]).replace("\u0120", " ").replace("\u010a", "\\n")
                # Skip if it's the same token (distance ~0)
                if dists[ni] < 1e-6:
                    continue
                neighbors.append({
                    "token": token_str,
                    "dist": round(float(dists[ni]), 3)
                })
                if len(neighbors) >= k:
                    break
            results.append(neighbors)
        return results
    except Exception as e:
        print(f"[Model] Vocab neighbor computation failed: {e}")
        return [[] for _ in range(n_real)]

# ============================================================
# 8. PCA COMPUTATION
# ============================================================

def compute_pca_basis(layer0_mat, hidden_dim):
    """
    Compute centroid, centered data, and top-2 PCA directions.
    Returns (centroid, centered, pc1, pc2, proj1, proj2).
    """
    centroid = np.mean(layer0_mat, axis=0)
    centered = layer0_mat - centroid
    n_total = layer0_mat.shape[0]

    if n_total >= 2:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pc1, pc2 = Vt[0], Vt[1]
    else:
        pc1 = np.zeros(hidden_dim)
        pc1[0] = 1.0
        pc2 = np.zeros(hidden_dim)
        pc2[1] = 1.0

    proj1 = centered @ pc1
    proj2 = centered @ pc2
    return centroid, centered, pc1, pc2, proj1, proj2


# ============================================================
# 9. GRID INTERSECTION PROBES
# ============================================================

def compute_grid_range(proj, pad_frac=0.3):
    """Compute padded min/max for a 1-D projection array."""
    mn, mx = float(proj.min()), float(proj.max())
    r = mx - mn if mx - mn > 1e-6 else 1.0
    return mn - pad_frac * r, mx + pad_frac * r, r


def make_grid_coords(g1, g2):
    """Return list of (v1, v2) pairs for a 2-D grid."""
    coords = []
    for v1 in g1:
        for v2 in g2:
            coords.append((v1, v2))
    return coords


def interpolate_grid_embedding(v1, v2, centroid, pc1, pc2):
    """Reconstruct a high-dim embedding from PCA coordinates."""
    return centroid + v1 * pc1 + v2 * pc2


def compute_grid_weights(v1, v2, existing_proj, sigma_nn):
    """Compute RBF weights for a grid point relative to existing projections."""
    dists = (existing_proj[:, 0] - v1) ** 2 + (existing_proj[:, 1] - v2) ** 2
    weights = np.exp(-dists / (2 * sigma_nn ** 2))
    weights /= weights.sum() + 1e-15
    return weights


def interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim):
    """Weighted-average the deltas from all points for one grid point."""
    n_total = len(all_deltas_per_point)
    point_deltas = []
    for lay in range(n_layers):
        d = np.zeros(hidden_dim)
        for pi in range(n_total):
            d += weights[pi] * all_deltas_per_point[pi][lay]
        point_deltas.append(d)
    return point_deltas


def create_grid_probes(centroid, pc1, pc2, proj1, proj2, existing_proj,
                       all_deltas_per_point, n_layers, hidden_dim, n_side=10, pad_frac=0.3,
                       all_attn_deltas=None, all_mlp_deltas=None):
    """
    Create grid intersection probes in PCA space.
    Returns (grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas).
    grid_attn_deltas and grid_mlp_deltas may be None if decomposition is unavailable.
    """
    mn1, mx1, r1 = compute_grid_range(proj1, pad_frac)
    mn2, mx2, r2 = compute_grid_range(proj2, pad_frac)
    g1 = np.linspace(mn1, mx1, n_side)
    g2 = np.linspace(mn2, mx2, n_side)
    sigma_nn = r1 * 0.2

    grid_layer0 = []
    grid_deltas = []
    grid_attn_deltas = [] if all_attn_deltas is not None else None
    grid_mlp_deltas = [] if all_mlp_deltas is not None else None

    for v1, v2 in make_grid_coords(g1, g2):
        emb = interpolate_grid_embedding(v1, v2, centroid, pc1, pc2)
        grid_layer0.append(emb)

        weights = compute_grid_weights(v1, v2, existing_proj, sigma_nn)
        point_deltas = interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim)
        grid_deltas.append(point_deltas)

        if all_attn_deltas is not None:
            attn_point_deltas = interpolate_deltas(weights, all_attn_deltas, n_layers, hidden_dim)
            grid_attn_deltas.append(attn_point_deltas)
        if all_mlp_deltas is not None:
            mlp_point_deltas = interpolate_deltas(weights, all_mlp_deltas, n_layers, hidden_dim)
            grid_mlp_deltas.append(mlp_point_deltas)

    return grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas


# ============================================================
# 10. OUTPUT ASSEMBLY
# ============================================================

def build_fixed_pos(all_layer0):
    """Convert list of numpy arrays to list of lists for JSON."""
    return [v.tolist() for v in all_layer0]


def build_deltas_array(all_deltas_per_point, n_layers, n_points):
    """Reshape per-point deltas into per-layer arrays for JSON."""
    deltas = []
    for lay in range(n_layers):
        layer_d = []
        for p in range(n_points):
            layer_d.append(all_deltas_per_point[p][lay].tolist())
        deltas.append(layer_d)
    return deltas


def build_output_data(all_labels, all_is_real, n_layers, n_total, n_real,
                      hidden_dim, fixed_pos, deltas, model_name, text, neighbors,
                      next_token_preds=None, vocab_neighbors=None,
                      attn_deltas=None, mlp_deltas=None, strain_stats=None):
    """Assemble the final JSON-serializable dict."""
    data = {
        "tokens": all_labels,
        "is_real": all_is_real,
        "n_layers": n_layers,
        "n_points": n_total,
        "n_real": n_real,
        "n_synth": n_total - n_real,
        "hidden_dim": hidden_dim,
        "fixed_pos": fixed_pos,
        "deltas": deltas,
        "model_name": model_name,
        "text": text,
        "neighbors": neighbors,
    }
    if next_token_preds is not None:
        data["next_token"] = next_token_preds
    if vocab_neighbors is not None:
        data["vocab_neighbors"] = vocab_neighbors
    if attn_deltas is not None:
        data["attn_deltas"] = attn_deltas
    if mlp_deltas is not None:
        data["mlp_deltas"] = mlp_deltas
    if strain_stats is not None:
        data["strain_stats"] = strain_stats
    return data


# ============================================================
# 10b. STRAIN STATISTICS COMPUTATION
# ============================================================

def compute_strain_stats(all_layer0, all_deltas_per_point, n_layers, n_real, hidden_dim):
    """
    Compute per-layer strain statistics using pairs of real tokens.
    For each layer, we compute the strain (deformed_dist / original_dist)
    for all pairs of real tokens, then aggregate.

    Returns a list of dicts, one per layer:
      {mean, max, min, variance, frac_expanding, frac_contracting, frac_isometric}
    """
    if n_real < 2:
        return [{"mean": 1.0, "max": 1.0, "min": 1.0, "variance": 0.0,
                 "frac_expanding": 0.0, "frac_contracting": 0.0, "frac_isometric": 1.0}
                for _ in range(n_layers)]

    # Precompute original pairwise distances (using all dimensions)
    # But since we project to 2D for visualization, we use a representative
    # approach: compute strain across all dimension pairs would be expensive.
    # Instead, compute the full high-dimensional strain.
    layer0_vecs = np.stack(all_layer0[:n_real], axis=0)  # (n_real, hidden_dim)

    stats = []
    for lay in range(n_layers):
        # Compute deformed positions: p + delta
        deltas = np.stack([all_deltas_per_point[p][lay] for p in range(n_real)], axis=0)
        deformed = layer0_vecs + deltas  # (n_real, hidden_dim)

        # Compute all pairwise distances
        strains = []
        for i in range(n_real):
            for j in range(i + 1, n_real):
                orig_dist = np.linalg.norm(layer0_vecs[i] - layer0_vecs[j])
                def_dist = np.linalg.norm(deformed[i] - deformed[j])
                if orig_dist > 1e-12:
                    strains.append(def_dist / orig_dist)

        strains = np.array(strains) if len(strains) > 0 else np.array([1.0])

        mean_s = float(np.mean(strains))
        max_s = float(np.max(strains))
        min_s = float(np.min(strains))
        var_s = float(np.var(strains))
        n_total_pairs = len(strains)
        frac_expanding = float(np.sum(strains > 1.05) / n_total_pairs)
        frac_contracting = float(np.sum(strains < 0.95) / n_total_pairs)
        frac_isometric = float(np.sum((strains >= 0.95) & (strains <= 1.05)) / n_total_pairs)

        stats.append({
            "mean": round(mean_s, 4),
            "max": round(max_s, 4),
            "min": round(min_s, 4),
            "variance": round(var_s, 6),
            "frac_expanding": round(frac_expanding, 4),
            "frac_contracting": round(frac_contracting, 4),
            "frac_isometric": round(frac_isometric, 4),
            "n_pairs": n_total_pairs,
        })

    return stats


# ============================================================
# 11. MAIN PROCESSING PIPELINE
# ============================================================

def process_text(text, model_name=None):
    global TOKENIZER, MODEL, MODEL_NAME, MODEL_CONFIG

    # Switch model if requested
    if model_name and model_name != MODEL_NAME:
        load_model(model_name)

    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    n_layers = get_n_layers(MODEL_CONFIG)

    # Tokenize real input
    real_ids, tokens_clean = tokenize_text(TOKENIZER, text)
    n_real = real_ids.shape[1]
    print(f"[Model] Tokens ({n_real}): {tokens_clean}")

    # Tokenize probes
    probe_texts = get_probe_texts()
    probe_seqs, probe_labels, probe_is_real = tokenize_probes(TOKENIZER, probe_texts)

    # Combine sequences
    all_seqs = [real_ids] + probe_seqs
    all_labels = list(tokens_clean) + probe_labels
    all_is_real = [True] * n_real + probe_is_real

    print(f"[Model] Running {len(all_seqs)} sequences through model (with component decomposition)...")

    # Extract hidden states WITH component decomposition
    all_layer0, all_deltas_per_point, all_attn_deltas, all_mlp_deltas = \
        run_all_sequences_with_components(MODEL, all_seqs, n_layers, hidden_dim)

    decomposition_available = all_attn_deltas is not None and all_mlp_deltas is not None
    if decomposition_available:
        print(f"[Model] Component decomposition: OK (attention + MLP deltas captured)")
    else:
        print(f"[Model] Component decomposition: UNAVAILABLE for this architecture")

    n_total = len(all_layer0)
    n_synth = n_total - n_real
    print(f"[Model] {n_total} points ({n_real} real + {n_synth} probes)")

    # Compute neighbors
    real_embeddings = np.stack(all_layer0[:n_real], axis=0)
    all_embeddings = np.stack(all_layer0, axis=0)
    neighbors = compute_neighbors(real_embeddings, all_embeddings, all_labels, all_is_real, k=10)

    # Predict next token
    print("[Model] Predicting next token...")
    next_token_preds = predict_next_token(TOKENIZER, MODEL, real_ids, MODEL_CONFIG, k=5)

    # Find vocabulary neighbors for each real token
    print("[Model] Finding vocabulary neighbors...")
    vocab_neighbors = find_vocab_neighbors(TOKENIZER, MODEL, all_layer0[:n_real], n_real, k=5)

    # PCA + grid probes
    print("[Model] Creating grid intersection probes...")
    layer0_mat = np.stack(all_layer0, axis=0)
    centroid, centered, pc1, pc2, proj1, proj2 = compute_pca_basis(layer0_mat, hidden_dim)
    existing_proj = np.stack([proj1, proj2], axis=1)

    grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas = create_grid_probes(
        centroid, pc1, pc2, proj1, proj2, existing_proj,
        all_deltas_per_point, n_layers, hidden_dim,
        n_side=10, pad_frac=0.3,
        all_attn_deltas=all_attn_deltas,
        all_mlp_deltas=all_mlp_deltas,
    )

    n_grid = len(grid_layer0)
    print(f"[Model] Added {n_grid} grid intersection probes")

    # Append grid probes
    for gi in range(n_grid):
        all_layer0.append(grid_layer0[gi])
        all_deltas_per_point.append(grid_deltas[gi])
        if decomposition_available:
            all_attn_deltas.append(grid_attn_deltas[gi])
            all_mlp_deltas.append(grid_mlp_deltas[gi])
        all_labels.append("\u00b7")
        all_is_real.append(False)

    n_total_final = len(all_layer0)

    # Compute strain statistics (on real tokens only, before grid appended)
    print("[Model] Computing strain statistics...")
    strain_stats = compute_strain_stats(all_layer0, all_deltas_per_point, n_layers, n_real, hidden_dim)
    # Find most active layer
    most_active_layer = max(range(n_layers), key=lambda l: strain_stats[l]["variance"])
    print(f"[Model] Most active layer (by strain variance): {most_active_layer}")

    # Build output
    fixed_pos = build_fixed_pos(all_layer0)
    deltas = build_deltas_array(all_deltas_per_point, n_layers, n_total_final)

    # Build component delta arrays if available
    attn_deltas_json = None
    mlp_deltas_json = None
    if decomposition_available:
        attn_deltas_json = build_deltas_array(all_attn_deltas, n_layers, n_total_final)
        mlp_deltas_json = build_deltas_array(all_mlp_deltas, n_layers, n_total_final)

    data = build_output_data(
        all_labels, all_is_real, n_layers, n_total_final, n_real,
        hidden_dim, fixed_pos, deltas, MODEL_NAME, text, neighbors,
        next_token_preds=next_token_preds,
        vocab_neighbors=vocab_neighbors,
        attn_deltas=attn_deltas_json,
        mlp_deltas=mlp_deltas_json,
        strain_stats=strain_stats,
    )

    json_str = json.dumps(data)
    print(f"[Model] JSON: {len(json_str)/1024/1024:.1f} MB")
    return json_str


# ============================================================
# 12. HTML PAGE WITH 2D/3D TOGGLE + DECOMPOSITION + STRAIN STATS
# ============================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Metric Space Explorer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden}
#side{width:370px;min-width:370px;background:#16213e;padding:12px;
  overflow-y:auto;border-right:2px solid #0f3460;
  display:flex;flex-direction:column;gap:6px}
#neighbor-panel{background:#0f3460;padding:6px;border-radius:4px;font-size:10px;
  line-height:1.5;max-height:300px;overflow-y:auto;display:none}
#selected-tokens{display:flex;flex-wrap:wrap;gap:3px;margin-top:4px;
  min-height:20px;max-height:120px;overflow-y:auto}
#side h2{color:#e94560;font-size:14px;margin-bottom:4px}
#side h3{color:#53a8b6;font-size:11px;margin-top:5px;border-bottom:1px solid #0f3460;padding-bottom:2px}
.cr{display:flex;align-items:center;gap:5px;font-size:11px}
.cr label{min-width:90px;text-align:right;color:#a0a0c0}
.cr input[type=range]{flex:1;accent-color:#e94560}
.cr .v{min-width:50px;color:#e94560;font-weight:bold;font-size:10px}
.cr select{flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;
  padding:2px;font-size:10px;min-width:0}
.cb{display:flex;align-items:center;gap:5px;font-size:11px;color:#a0a0c0}
.cb input{accent-color:#e94560}
#info{background:#0f3460;padding:6px;border-radius:4px;font-size:10px;line-height:1.4}
#info span{color:#e94560}
#text-area{display:flex;gap:4px;margin-bottom:4px}
#txt-in{flex:1;background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:6px;font-size:12px;border-radius:4px;font-family:monospace}
#btn-run{background:#e94560;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold}
#btn-run:hover{background:#c73e54}
#btn-run:disabled{background:#555;cursor:wait}
#model-area{display:flex;gap:4px;margin-bottom:4px;align-items:center}
#model-area label{font-size:11px;color:#a0a0c0;min-width:50px}
#sel-model{flex:1;background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:4px;font-size:11px;border-radius:4px}
#main{flex:1;display:flex;align-items:center;justify-content:center;position:relative}
canvas{background:#0d1117}
#status{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);background:rgba(15,52,96,.9);padding:4px 12px;border-radius:12px;font-size:10px;color:#53a8b6}
#legend{position:absolute;top:8px;right:8px;background:rgba(15,52,96,.9);padding:5px 8px;border-radius:5px;font-size:9px}
.li{display:flex;align-items:center;gap:4px;margin:2px 0}
.lc{width:16px;height:7px;border-radius:2px}
#keys{font-size:8px;color:#555;margin-top:4px;line-height:1.3}
#neighbor-panel h4{color:#e94560;font-size:11px;margin-bottom:4px}
.nb-item{color:#a0a0c0;cursor:pointer;padding:1px 4px;border-radius:2px}
.nb-item:hover{background:#1a1a2e;color:#fff}
.nb-item.is-real{color:#53a8b6;font-weight:bold}
.nb-item .nb-dist{color:#666;font-size:9px;margin-left:4px}
.sel-tok{background:#e94560;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;cursor:pointer;display:flex;align-items:center;gap:3px}
.sel-tok:hover{background:#c73e54}
.sel-tok .x{font-weight:bold;margin-left:2px}
.view-toggle{display:flex;gap:2px;margin-bottom:4px}
.view-toggle button{flex:1;padding:5px 8px;border:1px solid #0f3460;background:#1a1a2e;color:#a0a0c0;cursor:pointer;font-size:11px;font-weight:bold;border-radius:4px;transition:all 0.2s}
.view-toggle button.active{background:#e94560;color:#fff;border-color:#e94560}
.view-toggle button:hover:not(.active){background:#0f3460;color:#fff}
#strain-stats-panel{background:#0f3460;padding:6px;border-radius:4px;font-size:9px;
  line-height:1.3;max-height:280px;overflow-y:auto;display:none}
#strain-stats-panel table{width:100%;border-collapse:collapse}
#strain-stats-panel th{color:#53a8b6;text-align:left;padding:2px 3px;border-bottom:1px solid #1a1a2e;font-size:8px;position:sticky;top:0;background:#0f3460}
#strain-stats-panel td{padding:2px 3px;border-bottom:1px solid rgba(255,255,255,0.05)}
#strain-stats-panel tr.current-layer{background:rgba(233,69,96,0.2)}
#strain-stats-panel tr.most-active{background:rgba(83,168,182,0.15)}
#strain-stats-panel tr:hover{background:rgba(255,255,255,0.05)}
.strain-bar{display:inline-block;height:6px;border-radius:2px;vertical-align:middle}
</style></head><body>
<div id="side">
<h2>Metric Space Explorer</h2>
<div id="model-area">
<label>Model:</label>
<select id="sel-model">
<option value="gpt2">GPT-2 (124M)</option>
<option value="gpt2-medium">GPT-2 Medium (355M)</option>
<option value="gpt2-large">GPT-2 Large (774M)</option>
<option value="gpt2-xl">GPT-2 XL (1.5B)</option>
<option value="distilbert-base-uncased">DistilBERT</option>
<option value="bert-base-uncased">BERT Base</option>
<option value="bert-large-uncased">BERT Large</option>
<option value="roberta-base">RoBERTa Base</option>
<option value="EleutherAI/pythia-160m">Pythia 160M</option>
<option value="EleutherAI/pythia-410m">Pythia 410M</option>
<option value="facebook/opt-125m">OPT 125M</option>
</select>
</div>
<div id="text-area">
<input id="txt-in" type="text" placeholder="Enter text..." value="The quick brown fox jumps over the lazy dog">
<button id="btn-run" onclick="runText()">Run</button>
</div>
<div id="info">
Model: <span id="i-mod">-</span> |
Points: <span id="i-pts">-</span> (<span id="i-real">-</span> real + <span id="i-syn">-</span> probes)<br>
Layers: <span id="i-lay">-</span> | Dim: <span id="i-dim">-</span> |
Tokens: <span id="i-tok">-</span>
</div>
<h3>Predicted Next Token</h3>
<div id="next-token-panel" style="background:#0f3460;padding:6px;border-radius:4px;font-size:11px;line-height:1.6">
<span style="color:#555">Run a prompt to see predictions</span>
</div>
<h3>Selected Tokens (click canvas to select)</h3>
<div id="selected-tokens"><span style="color:#555;font-size:10px">Click on a token dot to select it</span></div>
<div id="neighbor-panel">
<h4 id="nb-title">Neighbors</h4>
<div id="nb-list"></div>
</div>
<h3>View Mode</h3>
<div class="view-toggle">
<button id="btn-2d" class="active" onclick="setViewMode('2d')">2D</button>
<button id="btn-3d" onclick="setViewMode('3d')">3D</button>
</div>
<h3>Layer &amp; Deformation</h3>
<div class="cr"><label>Layer:</label><input type="range" id="sl-layer" min="0" max="11" value="0" step="1"><span class="v" id="v-layer">0</span></div>
<div class="cr"><label>Deform t:</label><input type="range" id="sl-t" min="0" max="1" value="1.0" step="0.01"><span class="v" id="v-t">1.00</span></div>
<div class="cr"><label>Amplify:</label><input type="range" id="sl-amp" min="0.1" max="500" value="1" step="0.1"><span class="v" id="v-amp">1.0</span></div>
<div class="cr"><label>Mode:</label>
<select id="sel-mode">
<option value="single">This Layer Only</option>
<option value="cumfwd">Layers 0→L (Cumulative)</option>
<option value="cumbwd">Layers L→End (Cumulative)</option>
<option value="embedding">Raw Embedding Space</option>
</select>
</div>
<div class="cr"><label>Decomposition:</label>
<select id="sel-decomp">
<option value="full">Full Residual</option>
<option value="attn">Attention Only</option>
<option value="mlp">MLP Only</option>
</select>
</div>
<h3>Strain Statistics</h3>
<div id="strain-stats-panel"></div>
<h3>Dimensions</h3>
<div class="cr"><label>Dim X:</label><input type="range" id="sl-dx" min="0" max="767" value="0" step="1"><span class="v" id="v-dx">0</span></div>
<div class="cr"><label>Dim Y:</label><input type="range" id="sl-dy" min="0" max="767" value="1" step="1"><span class="v" id="v-dy">1</span></div>
<div class="cr" id="dz-row" style="display:none"><label>Dim Z:</label><input type="range" id="sl-dz" min="0" max="767" value="2" step="1"><span class="v" id="v-dz">2</span></div>
<h3>Neighbor Tracing</h3>
<div class="cr"><label>K Neighbors:</label><input type="range" id="sl-kn" min="1" max="20" value="5" step="1"><span class="v" id="v-kn">5</span></div>
<div class="cb"><input type="checkbox" id="cb-nb" checked><label for="cb-nb">Show Neighbor Lines</label></div>
<div class="cb"><input type="checkbox" id="cb-nblabel" checked><label for="cb-nblabel">Show Neighbor Labels</label></div>
<h3>RBF &amp; Grid</h3>
<div class="cr"><label>Bandwidth σ:</label><input type="range" id="sl-sig" min="0.01" max="20" value="1.0" step="0.01"><span class="v" id="v-sig">1.00</span></div>
<div class="cr"><label>Grid Res:</label><input type="range" id="sl-gr" min="10" max="80" value="30" step="1"><span class="v" id="v-gr">30</span></div>
<h3>Display</h3>
<div class="cb"><input type="checkbox" id="cb-grid" checked><label for="cb-grid">Deformed Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-vocnb" checked><label for="cb-vocnb">Show Nearby Vocab Words</label></div>
<div class="cb"><input type="checkbox" id="cb-heat" checked><label for="cb-heat">Strain Heatmap</label></div>
<div class="cb"><input type="checkbox" id="cb-ref" checked><label for="cb-ref">Reference Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-tok" checked><label for="cb-tok">Real Tokens</label></div>
<div class="cb"><input type="checkbox" id="cb-syn"><label for="cb-syn">Probe Points</label></div>
<div class="cb"><input type="checkbox" id="cb-sc" checked><label for="cb-sc">Strain Color</label></div>
<div class="cb"><input type="checkbox" id="cb-vec"><label for="cb-vec">Vector Arrows</label></div>
<div id="keys"><b>Keys:</b> ←→ Layer | ↑↓ t | A/Z Amp | Space Auto | R Reset | D Dims | 0 Reset Zoom<br>
<b>Mouse:</b> Scroll=Zoom | Shift+Drag=Pan | Click=Select Token</div>
</div>
<div id="main">
<canvas id="cv"></canvas>
<div id="legend">
<div class="li"><div class="lc" style="background:linear-gradient(90deg,#0077b6,#666,#e94560)"></div>Strain</div>
<div class="li"><div class="lc" style="background:#0077b6"></div>Contraction</div>
<div class="li"><div class="lc" style="background:#666"></div>Isometry</div>
<div class="li"><div class="lc" style="background:#e94560"></div>Expansion</div>
<div class="li"><div class="lc" style="background:#0f0"></div>Selected</div>
<div class="li"><div class="lc" style="background:rgba(0,255,200,0.5)"></div>Neighbor</div>
</div>
<div id="status">Enter text and click Run</div>
</div>
<script>
// Pan/Zoom state
var zoomLevel = 1.0;
var panX = 0, panY = 0;
var panActive = false, panLastX = 0, panLastY = 0;
var D=null,AP=null;
var selectedTokens=new Set();
var viewMode='2d';
// 3D rotation state
var rotX=-0.4, rotY=0.6, rotZ=0;
var dragActive=false, dragLastX=0, dragLastY=0;
var focalLength=600;

function setViewMode(mode){
    viewMode=mode;
    document.getElementById('btn-2d').className=(mode==='2d'?'active':'');
    document.getElementById('btn-3d').className=(mode==='3d'?'active':'');
    document.getElementById('dz-row').style.display=(mode==='3d'?'flex':'none');
    draw();
}

/** Return the active deltas array based on decomposition selector */
function getActiveDeltas(){
    if(!D) return null;
    var decomp = document.getElementById('sel-decomp').value;
    if(decomp === 'attn' && D.attn_deltas) return D.attn_deltas;
    if(decomp === 'mlp' && D.mlp_deltas) return D.mlp_deltas;
    return D.deltas;
}

/** Get a label for the current decomposition mode */
function getDecompLabel(){
    var decomp = document.getElementById('sel-decomp').value;
    if(decomp === 'attn' && D && D.attn_deltas) return 'Attn';
    if(decomp === 'mlp' && D && D.mlp_deltas) return 'MLP';
    return 'Full';
}

function runText(){
    var txt=document.getElementById('txt-in').value.trim();
    if(!txt)return;
    var modelName=document.getElementById('sel-model').value;
    var btn=document.getElementById('btn-run');
    btn.disabled=true;btn.textContent='Running...';
    document.getElementById('status').textContent='Processing (model: '+modelName+')...';
    fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:txt, model:modelName})})
    .then(function(r){if(!r.ok)throw new Error('Server error '+r.status);return r.json()})
    .then(function(d){
        D=d;selectedTokens.clear();updateSelectedUI();onData();
        btn.disabled=false;btn.textContent='Run';
    }).catch(function(e){
        document.getElementById('status').textContent='Error: '+e;
        btn.disabled=false;btn.textContent='Run';
    });
}

function updateStrainStatsPanel(){
    var panel = document.getElementById('strain-stats-panel');
    if(!D || !D.strain_stats || D.strain_stats.length === 0){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    var stats = D.strain_stats;
    var currentLayer = +document.getElementById('sl-layer').value;

    // Find most active layer by variance
    var mostActiveLayer = 0;
    var maxVar = 0;
    for(var i = 0; i < stats.length; i++){
        if(stats[i].variance > maxVar){
            maxVar = stats[i].variance;
            mostActiveLayer = i;
        }
    }

    // Find global max strain for bar scaling
    var globalMaxStrain = 0;
    for(var i = 0; i < stats.length; i++){
        if(stats[i].max > globalMaxStrain) globalMaxStrain = stats[i].max;
    }
    if(globalMaxStrain < 1.01) globalMaxStrain = 2.0;

    var html = '<table>';
    html += '<tr><th>L</th><th>Mean</th><th>Max</th><th>Var</th><th>Exp%</th><th>Con%</th><th>Iso%</th><th>Distribution</th></tr>';
    for(var i = 0; i < stats.length; i++){
        var s = stats[i];
        var rowClass = '';
        if(i === currentLayer && i === mostActiveLayer) rowClass = 'current-layer most-active';
        else if(i === currentLayer) rowClass = 'current-layer';
        else if(i === mostActiveLayer) rowClass = 'most-active';

        // Mini bar showing mean strain relative to 1.0
        var barWidth = Math.min(60, Math.max(2, Math.abs(s.mean - 1.0) / (globalMaxStrain - 1.0) * 60));
        var barColor = s.mean > 1.0 ? '#e94560' : '#0077b6';
        var barHtml = '<span class="strain-bar" style="width:'+barWidth+'px;background:'+barColor+'"></span>';

        // Markers
        var marker = '';
        if(i === currentLayer) marker += ' ◄';
        if(i === mostActiveLayer) marker += ' ★';

        html += '<tr class="'+rowClass+'" onclick="document.getElementById(\'sl-layer\').value='+i+';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))" style="cursor:pointer">';
        html += '<td style="color:#e94560;font-weight:bold">'+i+marker+'</td>';
        html += '<td>'+s.mean.toFixed(3)+'</td>';
        html += '<td style="color:'+(s.max > 1.5 ? '#e94560' : '#a0a0c0')+'">'+s.max.toFixed(3)+'</td>';
        html += '<td>'+s.variance.toFixed(4)+'</td>';
        html += '<td style="color:#e94560">'+(s.frac_expanding*100).toFixed(0)+'</td>';
        html += '<td style="color:#0077b6">'+(s.frac_contracting*100).toFixed(0)+'</td>';
        html += '<td style="color:#888">'+(s.frac_isometric*100).toFixed(0)+'</td>';
        html += '<td>'+barHtml+'</td>';
        html += '</tr>';
    }
    html += '</table>';
    html += '<div style="margin-top:4px;color:#888;font-size:8px">';
    html += '◄ = current layer | ★ = most active (highest variance) | Click row to jump to layer';
    html += '</div>';
    panel.innerHTML = html;
}

function onData(){
    document.getElementById('sl-layer').max=D.n_layers-1;
    document.getElementById('sl-dx').max=D.hidden_dim-1;
    document.getElementById('sl-dy').max=D.hidden_dim-1;
    document.getElementById('sl-dz').max=D.hidden_dim-1;
    document.getElementById('i-mod').textContent=D.model_name;
    document.getElementById('i-pts').textContent=D.n_points;
    document.getElementById('i-real').textContent=D.n_real;
    document.getElementById('i-syn').textContent=D.n_synth;
    document.getElementById('i-lay').textContent=D.n_layers;
    document.getElementById('i-dim').textContent=D.hidden_dim;
    document.getElementById('i-tok').textContent=D.tokens.slice(0,D.n_real).join(' ');
    document.getElementById('sel-model').value=D.model_name;

    // Update decomposition selector availability
    var decompSel = document.getElementById('sel-decomp');
    if(D.attn_deltas && D.mlp_deltas){
        decompSel.disabled = false;
        decompSel.title = 'Component decomposition available';
    } else {
        decompSel.disabled = true;
        decompSel.value = 'full';
        decompSel.title = 'Component decomposition not available for this model architecture';
    }

    // Update strain stats panel
    updateStrainStatsPanel();

    autoParams();
    draw();
    document.getElementById('status').textContent='Ready — '+D.n_real+' tokens, '+D.n_synth+' probes | Model: '+D.model_name;

    var ntp = document.getElementById('next-token-panel');
    if(D.next_token && D.next_token.length > 0){
        var html = '';
        for(var i=0; i<D.next_token.length; i++){
            var nt = D.next_token[i];
            var barW = Math.max(2, nt.prob * 200);
            html += '<div style="display:flex;align-items:center;gap:6px">';
            html += '<span style="color:#e94560;font-weight:bold;min-width:80px;font-family:monospace">' +
                    nt.token + '</span>';
            html += '<div style="background:#e94560;height:8px;width:'+barW+'px;border-radius:3px;opacity:0.7"></div>';
            html += '<span style="color:#888;font-size:9px">' + (nt.prob*100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        ntp.innerHTML = html;
    } else {
        ntp.innerHTML = '<span style="color:#555">No predictions available</span>';
    }
}

function autoParams(){
    if(!D)return;
    var dx=+document.getElementById('sl-dx').value;
    var dy=+document.getElementById('sl-dy').value;
    var nP=D.n_points;
    var mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity;
    for(var i=0;i<nP;i++){
        var x=D.fixed_pos[i][dx],y=D.fixed_pos[i][dy];
        if(x<mnx)mnx=x;if(x>mxx)mxx=x;if(y<mny)mny=y;if(y>mxy)mxy=y;
    }
    var range=Math.max(mxx-mnx,mxy-mny)||1;
    var sig=range*0.15;
    var slSig=document.getElementById('sl-sig');
    slSig.max=Math.max(20,range*2).toFixed(1);
    slSig.value=sig.toFixed(2);
    document.getElementById('v-sig').textContent=sig.toFixed(2);
    var activeDeltas = getActiveDeltas();
    if(!activeDeltas) activeDeltas = D.deltas;
    var norms=[];
    for(var l=0;l<D.n_layers;l++){
        for(var p=0;p<nP;p++){
            var ddx=activeDeltas[l][p][dx],ddy=activeDeltas[l][p][dy];
            norms.push(Math.sqrt(ddx*ddx+ddy*ddy));
        }
    }
    norms.sort(function(a,b){return a-b});
    var med=norms[Math.floor(norms.length*0.75)]||1;
    var amp=range*0.06/(med+1e-12);
    amp=Math.max(0.1,Math.min(500,amp));
    document.getElementById('sl-amp').value=amp.toFixed(1);
    document.getElementById('v-amp').textContent=amp.toFixed(1);
}

function updateSelectedUI(){
    var cont=document.getElementById('selected-tokens');
    cont.innerHTML='';
    if(selectedTokens.size===0){
        cont.innerHTML='<span style="color:#555;font-size:10px">Click on a token dot to select it</span>';
        document.getElementById('neighbor-panel').style.display='none';
        return;
    }
    selectedTokens.forEach(function(ti){
        var el=document.createElement('span');
        el.className='sel-tok';
        el.innerHTML='['+ti+'] '+D.tokens[ti]+' <span class="x">\u00d7</span>';
        el.onclick=function(){selectedTokens.delete(ti);updateSelectedUI();draw()};
        cont.appendChild(el);
    });
    updateNeighborPanel();
}

function updateNeighborPanel(){
    var panel=document.getElementById('neighbor-panel');
    var list=document.getElementById('nb-list');
    var title=document.getElementById('nb-title');
    if(!D||!D.neighbors||selectedTokens.size===0){
        panel.style.display='none';return;
    }
    panel.style.display='block';
    var kn=+document.getElementById('sl-kn').value;
    var html='';
    selectedTokens.forEach(function(ti){
        if(ti>=D.neighbors.length)return;
        html+='<div style="margin-bottom:6px"><b style="color:#e94560">['+ti+'] '+D.tokens[ti]+'</b> neighbors:';
        var nbs=D.neighbors[ti].slice(0,kn);
        for(var ni=0;ni<nbs.length;ni++){
            var nb=nbs[ni];
            var cls=nb.is_real?'nb-item is-real':'nb-item';
            html+='<div class="'+cls+'" onclick="clickNeighbor('+nb.idx+','+nb.is_real+')">';
            html+=(nb.is_real?'\u2605 ':'')+nb.label+'<span class="nb-dist">d='+nb.dist.toFixed(2)+'</span></div>';
        }
        html+='</div>';
    });
    list.innerHTML=html;
    title.textContent='Neighbors (K='+kn+')';
}

function clickNeighbor(idx, isReal){
    if(isReal && idx < D.n_real){
        selectedTokens.add(idx);
        updateSelectedUI();
    }
    draw();
}

// 3D rotation helpers
function rotatePoint3D(x, y, z){
    var cosY=Math.cos(rotY), sinY=Math.sin(rotY);
    var x1=x*cosY+z*sinY, z1=-x*sinY+z*cosY;
    var cosX=Math.cos(rotX), sinX=Math.sin(rotX);
    var y1=y*cosX-z1*sinX, z2=y*sinX+z1*cosX;
    return [x1, y1, z2];
}

function project3D(x, y, z, W, H){
    var r=rotatePoint3D(x, y, z);
    var scale=focalLength/(focalLength+r[2]);
    return [W/2+r[0]*scale, H/2+r[1]*scale, r[2], scale];
}

var cv3d=document.getElementById('cv');

cv3d.addEventListener('mousedown', function(e){
    if(viewMode==='3d'){
        if(e.button===0 && !e.shiftKey){
            dragActive=true;
            dragLastX=e.clientX;
            dragLastY=e.clientY;
            return;
        }
        if(e.button===1 || (e.button===0 && e.shiftKey)){
            e.preventDefault();
            panActive=true;
            panLastX=e.clientX;
            panLastY=e.clientY;
        }
        return;
    }
    if(e.button===1 || (e.button===0 && e.shiftKey)){
        e.preventDefault();
        panActive=true;
        panLastX=e.clientX;
        panLastY=e.clientY;
    }
});

window.addEventListener('mousemove', function(e){
    if(dragActive && viewMode==='3d'){
        var ddx=e.clientX-dragLastX, ddy=e.clientY-dragLastY;
        rotY+=ddx*0.005;
        rotX+=ddy*0.005;
        rotX=Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
        dragLastX=e.clientX;
        dragLastY=e.clientY;
        draw();
        return;
    }
    if(panActive){
        panX+=e.clientX-panLastX;
        panY+=e.clientY-panLastY;
        panLastX=e.clientX;
        panLastY=e.clientY;
        draw();
    }
});

window.addEventListener('mouseup', function(e){
    dragActive=false;
    panActive=false;
});

document.getElementById('cv').addEventListener('click', function(e){
    if(!D)return;
    if(dragActive)return;
    var cv=document.getElementById('cv');
    var rect=cv.getBoundingClientRect();
    var rawMx=e.clientX-rect.left, rawMy=e.clientY-rect.top;

    var mx, my;
    if(viewMode==='2d'){
        mx = (rawMx - panX) / zoomLevel;
        my = (rawMy - panY) / zoomLevel;
    } else {
        mx = rawMx;
        my = rawMy;
    }

    var dx=+document.getElementById('sl-dx').value;
    var dy=+document.getElementById('sl-dy').value;
    var dz=+document.getElementById('sl-dz').value;
    var nP=D.n_points, nR=D.n_real;
    var W=cv.width, H=cv.height, M2=42, dW2=W-2*M2, dH2=H-2*M2;

    var bestDist=Infinity, bestIdx=-1;

    if(viewMode==='2d'){
        var fx2=new Float64Array(nP),fy2=new Float64Array(nP);
        for(var i=0;i<nP;i++){fx2[i]=D.fixed_pos[i][dx];fy2[i]=D.fixed_pos[i][dy]}
        var mnx=fx2[0],mxx=fx2[0],mny=fy2[0],mxy=fy2[0];
        for(var i2=1;i2<nP;i2++){
            if(fx2[i2]<mnx)mnx=fx2[i2];if(fx2[i2]>mxx)mxx=fx2[i2];
            if(fy2[i2]<mny)mny=fy2[i2];if(fy2[i2]>mxy)mxy=fy2[i2];
        }
        var mr=Math.max(mxx-mnx,mxy-mny)||1;
        var cxv=(mnx+mxx)/2,cyv=(mny+mxy)/2;
        var pd2=0.12;
        var vx0=cxv-mr*(.5+pd2),vy0=cyv-mr*(.5+pd2),vw=mr*(1+2*pd2),vh=vw;
        function SX(x){return M2+((x-vx0)/vw)*dW2}
        function SY(y){return M2+((y-vy0)/vh)*dH2}
        for(var ti=0;ti<nR;ti++){
            var sx=SX(fx2[ti]), sy=SY(fy2[ti]);
            var dd=Math.hypot(mx-sx, my-sy);
            if(dd<bestDist){bestDist=dd;bestIdx=ti}
        }
    } else {
        var fx3=new Float64Array(nP),fy3=new Float64Array(nP),fz3=new Float64Array(nP);
        for(var i=0;i<nP;i++){fx3[i]=D.fixed_pos[i][dx];fy3[i]=D.fixed_pos[i][dy];fz3[i]=D.fixed_pos[i][dz]}
        var mnx3=Infinity,mxx3=-Infinity,mny3=Infinity,mxy3=-Infinity,mnz3=Infinity,mxz3=-Infinity;
        for(var i3=0;i3<nP;i3++){
            if(fx3[i3]<mnx3)mnx3=fx3[i3];if(fx3[i3]>mxx3)mxx3=fx3[i3];
            if(fy3[i3]<mny3)mny3=fy3[i3];if(fy3[i3]>mxy3)mxy3=fy3[i3];
            if(fz3[i3]<mnz3)mnz3=fz3[i3];if(fz3[i3]>mxz3)mxz3=fz3[i3];
        }
        var mr3=Math.max(mxx3-mnx3,mxy3-mny3,mxz3-mnz3)||1;
        var cx3=(mnx3+mxx3)/2,cy3=(mny3+mxy3)/2,cz3=(mnz3+mxz3)/2;
        var sc3=Math.min(dW2,dH2)*0.35/mr3;
        for(var ti=0;ti<nR;ti++){
            var px=(fx3[ti]-cx3)*sc3, py=(fy3[ti]-cy3)*sc3, pz=(fz3[ti]-cz3)*sc3;
            var proj=project3D(px,py,pz,W,H);
            var dd=Math.hypot(mx-proj[0], my-proj[1]);
            if(dd<bestDist){bestDist=dd;bestIdx=ti}
        }
    }

    if(bestIdx>=0 && bestDist<25){
        if(selectedTokens.has(bestIdx)) selectedTokens.delete(bestIdx);
        else selectedTokens.add(bestIdx);
        updateSelectedUI();draw();
    }
});

['sl-kn'].forEach(function(id){
    var s=document.getElementById(id);
    s.addEventListener('input',function(){
        document.getElementById('v-kn').textContent=s.value;
        updateSelectedUI();draw();
    });
});
['cb-nb','cb-nblabel'].forEach(function(id){
    document.getElementById(id).addEventListener('change',function(){draw()});
});

window.addEventListener('resize',function(){rsz();draw()});
function rsz(){
    var cv=document.getElementById('cv'),ct=document.getElementById('main');
    var s=Math.min(ct.clientWidth-16,ct.clientHeight-16);
    cv.width=Math.max(400,s);cv.height=cv.width;
}
rsz();

var SLS=[
    ['sl-layer','v-layer',0],['sl-t','v-t',2],['sl-amp','v-amp',1],
    ['sl-dx','v-dx',0],['sl-dy','v-dy',0],['sl-dz','v-dz',0],
    ['sl-sig','v-sig',2],['sl-gr','v-gr',0]
];
for(var i=0;i<SLS.length;i++)(function(c){
    var s=document.getElementById(c[0]),v=document.getElementById(c[1]),dec=c[2];
    v.textContent=parseFloat(s.value).toFixed(dec);
    s.addEventListener('input',function(){
        v.textContent=parseFloat(s.value).toFixed(dec);
        if(c[0]==='sl-dx'||c[0]==='sl-dy'||c[0]==='sl-dz')autoParams();
        if(c[0]==='sl-layer') updateStrainStatsPanel();
        draw();
    });
})(SLS[i]);

['cb-grid','cb-heat','cb-ref','cb-tok','cb-syn','cb-sc','cb-vec','cb-vocnb'].forEach(function(id){
    document.getElementById(id).addEventListener('change',draw);
});

document.getElementById('sel-mode').addEventListener('change',draw);
document.getElementById('sel-decomp').addEventListener('change',function(){
    autoParams();
    draw();
});
document.addEventListener('keydown',onKey);
document.getElementById('txt-in').addEventListener('keydown',function(e){if(e.key==='Enter')runText()});

function gp(){return{
    layer:+document.getElementById('sl-layer').value,
    t:+document.getElementById('sl-t').value,
    amp:+document.getElementById('sl-amp').value,
    mode:document.getElementById('sel-mode').value,
    dx:+document.getElementById('sl-dx').value,
    dy:+document.getElementById('sl-dy').value,
    dz:+document.getElementById('sl-dz').value,
    sig:+document.getElementById('sl-sig').value,
    gr:+document.getElementById('sl-gr').value,
    grid:document.getElementById('cb-grid').checked,
    heat:document.getElementById('cb-heat').checked,
    ref:document.getElementById('cb-ref').checked,
    tok:document.getElementById('cb-tok').checked,
    syn:document.getElementById('cb-syn').checked,
    sc:document.getElementById('cb-sc').checked,
    vec:document.getElementById('cb-vec').checked,
    nb:document.getElementById('cb-nb').checked,
    nblabel:document.getElementById('cb-nblabel').checked,
    kn:+document.getElementById('sl-kn').value
}}

function onKey(e){
    if(document.activeElement===document.getElementById('txt-in'))return;
    var sl=document.getElementById('sl-layer'),st=document.getElementById('sl-t'),sa=document.getElementById('sl-amp');
    if(e.key==='ArrowRight'){sl.value=Math.min(+sl.max,+sl.value+1);sl.dispatchEvent(new Event('input'))}
    else if(e.key==='ArrowLeft'){sl.value=Math.max(0,+sl.value-1);sl.dispatchEvent(new Event('input'))}
    else if(e.key==='ArrowUp'){e.preventDefault();st.value=Math.min(1,+st.value+.05).toFixed(2);st.dispatchEvent(new Event('input'))}
    else if(e.key==='ArrowDown'){e.preventDefault();st.value=Math.max(0,+st.value-.05).toFixed(2);st.dispatchEvent(new Event('input'))}
    else if(e.key==='a'||e.key==='A'){sa.value=Math.min(500,+sa.value*1.3).toFixed(1);sa.dispatchEvent(new Event('input'))}
    else if(e.key==='z'||e.key==='Z'){sa.value=Math.max(.1,+sa.value/1.3).toFixed(1);sa.dispatchEvent(new Event('input'))}
    else if(e.key===' '){e.preventDefault();togAP()}
    else if(e.key==='r'||e.key==='R'){rstAll()}
    else if(e.key==='d'||e.key==='D'){nxtD()}
    else if(e.key==='0'){zoomLevel=1.0;panX=0;panY=0;draw()}
}

function togAP(){
    if(AP){clearInterval(AP);AP=null;document.getElementById('status').textContent='Stopped';return}
    AP=setInterval(function(){
        var sl=document.getElementById('sl-layer');
        sl.value=(+sl.value+1)%(+sl.max+1);sl.dispatchEvent(new Event('input'));
    },600);
    document.getElementById('status').textContent='Autoplay (Space=stop)';
}
function rstAll(){
    document.getElementById('sl-layer').value='0';
    document.getElementById('sl-t').value='1.0';
    document.getElementById('sl-dx').value='0';
    document.getElementById('sl-dy').value='1';
    document.getElementById('sl-dz').value='2';
    document.getElementById('sl-gr').value='30';
    document.getElementById('sel-decomp').value='full';
    rotX=-0.4;rotY=0.6;rotZ=0;
    zoomLevel=1.0;panX=0;panY=0;
    selectedTokens.clear();updateSelectedUI();
    if(D)autoParams();
    ['sl-layer','sl-t','sl-amp','sl-dx','sl-dy','sl-dz','sl-sig','sl-gr'].forEach(function(id){
        document.getElementById(id).dispatchEvent(new Event('input'));
    });
}
function nxtD(){
    var dx=document.getElementById('sl-dx'),dy=document.getElementById('sl-dy');
    var x=+dx.value,y=+dy.value;
    y++;if(y>+dy.max){y=0;x++}if(x>=y)y=x+1;if(y>+dy.max){x=0;y=1}
    dx.value=x;dy.value=y;dx.dispatchEvent(new Event('input'));
}

function s2c(s){
    if(s<=.5)return[0,180,220];if(s>=1.5)return[233,69,96];
    if(s<1){var f=(s-.5)/.5;return[~~(f*120),~~(180-f*80),~~(220-f*120)]}
    var f2=(s-1)/.5;return[~~(120+f2*113),~~(100-f2*31),~~(100-f2*4)];
}

function draw(){
    if(!D)return;
    if(viewMode==='3d'){draw3D();return}
    draw2D();
}

function draw2D(){
    var p=gp(),cv=document.getElementById('cv'),c=cv.getContext('2d');
    var W=cv.width,H=cv.height;
    c.clearRect(0,0,W,H);

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    var nP=D.n_points,nR=D.n_real,dx=p.dx,dy=p.dy;
    var isEmb=p.mode==='embedding';

    // Use the active deltas based on decomposition selector
    var activeDeltas = getActiveDeltas();

    var fx=new Float64Array(nP),fy=new Float64Array(nP);
    for(var i=0;i<nP;i++){fx[i]=D.fixed_pos[i][dx];fy[i]=D.fixed_pos[i][dy]}

    var edx=new Float64Array(nP),edy=new Float64Array(nP);
    if(!isEmb){
        var layer=p.layer,amp=p.amp;
        for(var j=0;j<nP;j++){
            var sx2=0,sy2=0;
            if(p.mode==='single'){sx2=activeDeltas[layer][j][dx];sy2=activeDeltas[layer][j][dy]}
            else if(p.mode==='cumfwd'){for(var l=0;l<=layer;l++){sx2+=activeDeltas[l][j][dx];sy2+=activeDeltas[l][j][dy]}}
            else{for(var l2=layer;l2<D.n_layers;l2++){sx2+=activeDeltas[l2][j][dx];sy2+=activeDeltas[l2][j][dy]}}
            edx[j]=sx2*amp;edy[j]=sy2*amp;
        }
    }

    var mnx=fx[0],mxx=fx[0],mny=fy[0],mxy=fy[0];
    for(var i2=1;i2<nP;i2++){
        if(fx[i2]<mnx)mnx=fx[i2];if(fx[i2]>mxx)mxx=fx[i2];
        if(fy[i2]<mny)mny=fy[i2];if(fy[i2]>mxy)mxy=fy[i2];
    }
    var rx=mxx-mnx||1,ry=mxy-mny||1;
    var mr=Math.max(rx,ry);
    var cxv=(mnx+mxx)/2,cyv=(mny+mxy)/2;
    var pd2=0.12;
    var vx0=cxv-mr*(.5+pd2),vy0=cyv-mr*(.5+pd2),vw=mr*(1+2*pd2),vh=vw;

    var M=42,dW=W-2*M,dH=H-2*M;
    function SX(x){return M+((x-vx0)/vw)*dW}
    function SY(y){return M+((y-vy0)/vh)*dH}

    var N=p.gr,nV=(N+1)*(N+1);
    var oX=new Float64Array(nV),oY=new Float64Array(nV);
    var gX=new Float64Array(nV),gY=new Float64Array(nV);
    for(var gy=0;gy<=N;gy++)for(var gx=0;gx<=N;gx++){
        var gi=gy*(N+1)+gx;
        oX[gi]=vx0+(gx/N)*vw;oY[gi]=vy0+(gy/N)*vh;
    }

    var sig=p.sig,s2i=1/(2*sig*sig),t=p.t;
    if(isEmb){
        for(var gi2=0;gi2<nV;gi2++){gX[gi2]=oX[gi2];gY[gi2]=oY[gi2]}
    } else {
        for(var gi3=0;gi3<nV;gi3++){
            var px=oX[gi3],py=oY[gi3];
            var vx=0,vy=0,ws=0;
            for(var k=0;k<nP;k++){
                var ex=px-fx[k],ey=py-fy[k];
                var w=Math.exp(-(ex*ex+ey*ey)*s2i);
                vx+=w*edx[k];vy+=w*edy[k];ws+=w;
            }
            if(ws>1e-15){vx/=ws;vy/=ws}
            gX[gi3]=px+t*vx;gY[gi3]=py+t*vy;
        }
    }

    var sH=new Float64Array(N*(N+1)),sVa=new Float64Array((N+1)*N);
    for(var ey2=0;ey2<=N;ey2++)for(var ex2=0;ex2<N;ex2++){
        var a=ey2*(N+1)+ex2,b=a+1;
        var od=Math.hypot(oX[b]-oX[a],oY[b]-oY[a]);
        var dd=Math.hypot(gX[b]-gX[a],gY[b]-gY[a]);
        sH[ey2*N+ex2]=od>1e-12?dd/od:1;
    }
    for(var ey3=0;ey3<N;ey3++)for(var ex3=0;ex3<=N;ex3++){
        var a2=ey3*(N+1)+ex3,b2=(ey3+1)*(N+1)+ex3;
        var od2=Math.hypot(oX[b2]-oX[a2],oY[b2]-oY[a2]);
        var dd2=Math.hypot(gX[b2]-gX[a2],gY[b2]-gY[a2]);
        sVa[ey3*(N+1)+ex3]=od2>1e-12?dd2/od2:1;
    }

    if(p.heat&&!isEmb){
        for(var hy=0;hy<N;hy++)for(var hx=0;hx<N;hx++){
            var avg=(sH[hy*N+hx]+sH[(hy+1)*N+hx]+sVa[hy*(N+1)+hx]+sVa[hy*(N+1)+hx+1])/4;
            var co=s2c(avg);
            var i00=hy*(N+1)+hx,i10=i00+1,i01=(hy+1)*(N+1)+hx,i11=i01+1;
            c.beginPath();
            c.moveTo(SX(gX[i00]),SY(gY[i00]));c.lineTo(SX(gX[i10]),SY(gY[i10]));
            c.lineTo(SX(gX[i11]),SY(gY[i11]));c.lineTo(SX(gX[i01]),SY(gY[i01]));
            c.closePath();
            c.fillStyle='rgba('+co[0]+','+co[1]+','+co[2]+',0.3)';c.fill();
        }
    }

    if(p.ref){
        c.strokeStyle=isEmb?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.07)';c.lineWidth=0.5;
        for(var ry2=0;ry2<=N;ry2++){
            c.beginPath();
            for(var rx2=0;rx2<=N;rx2++){var ri=ry2*(N+1)+rx2;if(rx2===0)c.moveTo(SX(oX[ri]),SY(oY[ri]));else c.lineTo(SX(oX[ri]),SY(oY[ri]))}
            c.stroke();
        }
        for(var rx3=0;rx3<=N;rx3++){
            c.beginPath();
            for(var ry3=0;ry3<=N;ry3++){var ri3=ry3*(N+1)+rx3;if(ry3===0)c.moveTo(SX(oX[ri3]),SY(oY[ri3]));else c.lineTo(SX(oX[ri3]),SY(oY[ri3]))}
            c.stroke();
        }
    }

    if(p.grid&&!isEmb){
        c.lineWidth=1.2;
        for(var dhy=0;dhy<=N;dhy++)for(var dhx=0;dhx<N;dhx++){
            var di1=dhy*(N+1)+dhx,di2=di1+1;
            var es=sH[dhy*N+dhx];
            if(p.sc){var ec=s2c(es);c.strokeStyle='rgba('+ec[0]+','+ec[1]+','+ec[2]+',0.85)'}
            else c.strokeStyle='rgba(200,200,200,0.5)';
            c.beginPath();c.moveTo(SX(gX[di1]),SY(gY[di1]));c.lineTo(SX(gX[di2]),SY(gY[di2]));c.stroke();
        }
        for(var dvx=0;dvx<=N;dvx++)for(var dvy=0;dvy<N;dvy++){
            var dvi1=dvy*(N+1)+dvx,dvi2=(dvy+1)*(N+1)+dvx;
            var vs=sVa[dvy*(N+1)+dvx];
            if(p.sc){var vc=s2c(vs);c.strokeStyle='rgba('+vc[0]+','+vc[1]+','+vc[2]+',0.85)'}
            else c.strokeStyle='rgba(200,200,200,0.5)';
            c.beginPath();c.moveTo(SX(gX[dvi1]),SY(gY[dvi1]));c.lineTo(SX(gX[dvi2]),SY(gY[dvi2]));c.stroke();
        }
    }

    if(p.vec&&!isEmb){
        var step=Math.max(1,Math.floor(N/12));c.lineWidth=1.5;
        for(var viy=0;viy<=N;viy+=step)for(var vix=0;vix<=N;vix+=step){
            var vi=viy*(N+1)+vix;
            var ax=SX(oX[vi]),ay=SY(oY[vi]),bx=SX(gX[vi]),by=SY(gY[vi]);
            var al=Math.hypot(bx-ax,by-ay);if(al<3)continue;
            c.strokeStyle='rgba(255,255,100,0.6)';c.fillStyle='rgba(255,255,100,0.6)';
            c.beginPath();c.moveTo(ax,ay);c.lineTo(bx,by);c.stroke();
            var aa=Math.atan2(by-ay,bx-ax),hl=Math.min(7,al*.3);
            c.beginPath();c.moveTo(bx,by);
            c.lineTo(bx-hl*Math.cos(aa-.4),by-hl*Math.sin(aa-.4));
            c.lineTo(bx-hl*Math.cos(aa+.4),by-hl*Math.sin(aa+.4));
            c.closePath();c.fill();
        }
    }

    if(p.syn){
        for(var pi=nR;pi<nP;pi++){
            c.beginPath();c.arc(SX(fx[pi]),SY(fy[pi]),2.5,0,Math.PI*2);
            c.fillStyle='rgba(100,200,255,0.2)';c.fill();
        }
    }

    if(p.nb && D.neighbors && selectedTokens.size>0){
        var kn=p.kn;
        selectedTokens.forEach(function(ti){
            if(ti>=D.neighbors.length)return;
            var nbs=D.neighbors[ti].slice(0,kn);
            var tx=SX(fx[ti]),ty=SY(fy[ti]);
            for(var ni=0;ni<nbs.length;ni++){
                var nb=nbs[ni];
                var nidx=nb.idx;
                if(nidx>=nP)continue;
                var nx=SX(fx[nidx]),ny=SY(fy[nidx]);
                var alpha=Math.max(0.15, 1.0 - ni*0.08);
                c.strokeStyle='rgba(0,255,200,'+alpha.toFixed(2)+')';
                c.lineWidth=Math.max(0.5, 2.5 - ni*0.2);
                c.setLineDash([3,3]);
                c.beginPath();c.moveTo(tx,ty);c.lineTo(nx,ny);c.stroke();
                c.setLineDash([]);
                c.beginPath();c.arc(nx,ny,5,0,Math.PI*2);
                c.fillStyle=nb.is_real?'rgba(0,255,200,0.8)':'rgba(0,255,200,0.35)';
                c.fill();c.strokeStyle='rgba(0,255,200,0.6)';c.lineWidth=1;c.stroke();
                if(p.nblabel){
                    c.font='9px monospace';c.fillStyle='rgba(0,255,200,0.9)';
                    c.fillText(nb.label+' (d='+nb.dist.toFixed(1)+')',nx+8,ny-4);
                }
            }
        });
    }

    if(p.tok){
        var tc=['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22',
            '#f39c12','#d35400','#c0392b','#16a085','#27ae60',
            '#2980b9','#8e44ad','#2c3e50','#ecf0f1','#fd79a8'];
        for(var ti=0;ti<nR;ti++){
            var tx2=SX(fx[ti]),ty2=SY(fy[ti]),col=tc[ti%tc.length];
            var isSel=selectedTokens.has(ti);
            if(isSel){
                var grad2=c.createRadialGradient(tx2,ty2,0,tx2,ty2,30);
                grad2.addColorStop(0,'rgba(0,255,0,0.25)');grad2.addColorStop(1,'rgba(0,255,0,0)');
                c.beginPath();c.arc(tx2,ty2,30,0,Math.PI*2);c.fillStyle=grad2;c.fill();
            }
            var grad=c.createRadialGradient(tx2,ty2,0,tx2,ty2,20);
            grad.addColorStop(0,'rgba(255,255,255,0.08)');grad.addColorStop(1,'rgba(255,255,255,0)');
            c.beginPath();c.arc(tx2,ty2,20,0,Math.PI*2);c.fillStyle=grad;c.fill();
            c.beginPath();c.arc(tx2,ty2,isSel?9:7,0,Math.PI*2);
            c.fillStyle=col;c.fill();
            c.strokeStyle=isSel?'#0f0':'#fff';c.lineWidth=isSel?3:2;c.stroke();
            c.font='bold 11px monospace';c.lineWidth=3;c.strokeStyle='rgba(0,0,0,0.9)';
            var lb='['+ti+'] '+D.tokens[ti];
            c.strokeText(lb,tx2+12,ty2-10);c.fillStyle=isSel?'#0f0':'#fff';c.fillText(lb,tx2+12,ty2-10);
        }
        if(isEmb&&nR>1){
            c.strokeStyle='rgba(233,69,96,0.3)';c.lineWidth=1.5;c.setLineDash([4,4]);
            c.beginPath();c.moveTo(SX(fx[0]),SY(fy[0]));
            for(var ti2=1;ti2<nR;ti2++)c.lineTo(SX(fx[ti2]),SY(fy[ti2]));
            c.stroke();c.setLineDash([]);
        }
    }

    if(document.getElementById('cb-vocnb').checked && D.vocab_neighbors && p.tok){
        c.font='9px monospace';
        for(var vi2=0;vi2<nR;vi2++){
            if(!D.vocab_neighbors[vi2])continue;
            var vtx=SX(fx[vi2]),vty=SY(fy[vi2]);
            var vnbs=D.vocab_neighbors[vi2];
            for(var vni=0;vni<vnbs.length;vni++){
                var vnb=vnbs[vni];
                var angle=-Math.PI/2+(vni/(vnbs.length-1||1))*Math.PI;
                var radius=35+vni*8;
                var vnx=vtx+Math.cos(angle)*radius;
                var vny=vty+Math.sin(angle)*radius+20;
                c.fillStyle='rgba(150,150,170,0.45)';
                c.fillText(vnb.token,vnx,vny);
            }
        }
    }

    c.restore();

    // HUD text
    var decompLabel = getDecompLabel();
    c.font='11px monospace';c.fillStyle='rgba(255,255,255,0.45)';
    if(isEmb){
        c.fillText('EMBEDDING SPACE [2D]  Dims:'+dx+','+dy,42,18);
    } else {
        c.fillText('Layer '+p.layer+'/'+(D.n_layers-1)+'  t='+p.t.toFixed(2)+'  amp='+p.amp.toFixed(1)+'  Dims:'+dx+','+dy+'  Mode:'+p.mode+'  Decomp:'+decompLabel+'  [2D]',42,18);
    }
    c.font='10px monospace';c.fillStyle='rgba(255,255,255,0.35)';
    c.fillText('Zoom: '+zoomLevel.toFixed(2)+'x  (Scroll=zoom, Shift+drag=pan, 0=reset)',42,H-10);
}

// ===================== 3D DRAWING =====================
function draw3D(){
    var p=gp(),cv=document.getElementById('cv'),c=cv.getContext('2d');
    var W=cv.width,H=cv.height;
    c.clearRect(0,0,W,H);

    var nP=D.n_points,nR=D.n_real,dx=p.dx,dy=p.dy,dz=p.dz;
    var isEmb=p.mode==='embedding';

    // Use the active deltas based on decomposition selector
    var activeDeltas = getActiveDeltas();

    var fx=new Float64Array(nP),fy=new Float64Array(nP),fz=new Float64Array(nP);
    for(var i=0;i<nP;i++){
        fx[i]=D.fixed_pos[i][dx];
        fy[i]=D.fixed_pos[i][dy];
        fz[i]=D.fixed_pos[i][dz];
    }

    var edx3=new Float64Array(nP),edy3=new Float64Array(nP),edz3=new Float64Array(nP);
    if(!isEmb){
        var layer=p.layer,amp=p.amp;
        for(var j=0;j<nP;j++){
            var sx3=0,sy3=0,sz3=0;
            if(p.mode==='single'){
                sx3=activeDeltas[layer][j][dx];sy3=activeDeltas[layer][j][dy];sz3=activeDeltas[layer][j][dz];
            } else if(p.mode==='cumfwd'){
                for(var l=0;l<=layer;l++){sx3+=activeDeltas[l][j][dx];sy3+=activeDeltas[l][j][dy];sz3+=activeDeltas[l][j][dz]}
            } else {
                for(var l2=layer;l2<D.n_layers;l2++){sx3+=activeDeltas[l2][j][dx];sy3+=activeDeltas[l2][j][dy];sz3+=activeDeltas[l2][j][dz]}
            }
            edx3[j]=sx3*amp;edy3[j]=sy3*amp;edz3[j]=sz3*amp;
        }
    }

    var mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity,mnz=Infinity,mxz=-Infinity;
    for(var i2=0;i2<nP;i2++){
        if(fx[i2]<mnx)mnx=fx[i2];if(fx[i2]>mxx)mxx=fx[i2];
        if(fy[i2]<mny)mny=fy[i2];if(fy[i2]>mxy)mxy=fy[i2];
        if(fz[i2]<mnz)mnz=fz[i2];if(fz[i2]>mxz)mxz=fz[i2];
    }
    var mrx=mxx-mnx||1, mry=mxy-mny||1, mrz=mxz-mnz||1;
    var mr3=Math.max(mrx,mry,mrz);
    var cx3=(mnx+mxx)/2,cy3=(mny+mxy)/2,cz3=(mnz+mxz)/2;
    var sc3=Math.min(W,H)*0.3/mr3;

    var pd3=0.12;
    var vx0=cx3-mr3*(.5+pd3), vx1=cx3+mr3*(.5+pd3);
    var vy0=cy3-mr3*(.5+pd3), vy1=cy3+mr3*(.5+pd3);
    var vz0=cz3-mr3*(.5+pd3), vz1=cz3+mr3*(.5+pd3);

    var effSc3 = sc3 * zoomLevel;
    var cx2d = W/2 + panX;
    var cy2d = H/2 + panY;

    function proj3D(x, y, z){
        var r=rotatePoint3D(x, y, z);
        var scale=focalLength/(focalLength+r[2]);
        return [cx2d+r[0]*scale, cy2d+r[1]*scale, r[2], scale];
    }

    var N3=Math.min(Math.max(6, Math.round(p.gr/4)), 20);
    var nV3=(N3+1)*(N3+1)*(N3+1);

    function gIdx(ix,iy,iz){return iz*(N3+1)*(N3+1)+iy*(N3+1)+ix}

    var oX3=new Float64Array(nV3),oY3=new Float64Array(nV3),oZ3=new Float64Array(nV3);
    for(var iz=0;iz<=N3;iz++)for(var iy=0;iy<=N3;iy++)for(var ix=0;ix<=N3;ix++){
        var gi=gIdx(ix,iy,iz);
        oX3[gi]=vx0+(ix/N3)*(vx1-vx0);
        oY3[gi]=vy0+(iy/N3)*(vy1-vy0);
        oZ3[gi]=vz0+(iz/N3)*(vz1-vz0);
    }

    var gX3=new Float64Array(nV3),gY3=new Float64Array(nV3),gZ3=new Float64Array(nV3);
    var sig=p.sig, s2i3=1/(2*sig*sig), t=p.t;

    if(isEmb){
        for(var gi2=0;gi2<nV3;gi2++){gX3[gi2]=oX3[gi2];gY3[gi2]=oY3[gi2];gZ3[gi2]=oZ3[gi2]}
    } else {
        for(var gi3=0;gi3<nV3;gi3++){
            var gpx=oX3[gi3],gpy=oY3[gi3],gpz=oZ3[gi3];
            var vvx=0,vvy=0,vvz=0,ws=0;
            for(var k=0;k<nP;k++){
                var eex=gpx-fx[k],eey=gpy-fy[k],eez=gpz-fz[k];
                var w=Math.exp(-(eex*eex+eey*eey+eez*eez)*s2i3);
                vvx+=w*edx3[k];vvy+=w*edy3[k];vvz+=w*edz3[k];ws+=w;
            }
            if(ws>1e-15){vvx/=ws;vvy/=ws;vvz/=ws}
            gX3[gi3]=gpx+t*vvx;
            gY3[gi3]=gpy+t*vvy;
            gZ3[gi3]=gpz+t*vvz;
        }
    }

    function strain3(ax,ay,az,bx,by,bz,oax,oay,oaz,obx,oby,obz){
        var od=Math.sqrt((obx-oax)*(obx-oax)+(oby-oay)*(oby-oay)+(obz-oaz)*(obz-oaz));
        var dd=Math.sqrt((bx-ax)*(bx-ax)+(by-ay)*(by-ay)+(bz-az)*(bz-az));
        return od>1e-12?dd/od:1;
    }

    var edges3d=[];
    for(var iz=0;iz<=N3;iz++)for(var iy=0;iy<=N3;iy++)for(var ix=0;ix<N3;ix++){
        var a=gIdx(ix,iy,iz), b=gIdx(ix+1,iy,iz);
        var s=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],
                       oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
        edges3d.push({a:a,b:b,strain:s,dir:'x'});
    }
    for(var iz=0;iz<=N3;iz++)for(var iy=0;iy<N3;iy++)for(var ix=0;ix<=N3;ix++){
        var a=gIdx(ix,iy,iz), b=gIdx(ix,iy+1,iz);
        var s=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],
                       oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
        edges3d.push({a:a,b:b,strain:s,dir:'y'});
    }
    for(var iz=0;iz<N3;iz++)for(var iy=0;iy<=N3;iy++)for(var ix=0;ix<=N3;ix++){
        var a=gIdx(ix,iy,iz), b=gIdx(ix,iy,iz+1);
        var s=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],
                       oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
        edges3d.push({a:a,b:b,strain:s,dir:'z'});
    }

    var projV=[];
    for(var vi=0;vi<nV3;vi++){
        var px=(gX3[vi]-cx3)*effSc3, py=(gY3[vi]-cy3)*effSc3, pz=(gZ3[vi]-cz3)*effSc3;
        projV.push(proj3D(px,py,pz));
    }
    var projO=[];
    for(var vi=0;vi<nV3;vi++){
        var px=(oX3[vi]-cx3)*effSc3, py=(oY3[vi]-cy3)*effSc3, pz=(oZ3[vi]-cz3)*effSc3;
        projO.push(proj3D(px,py,pz));
    }

    for(var ei=0;ei<edges3d.length;ei++){
        var e=edges3d[ei];
        e.avgZ=(projV[e.a][2]+projV[e.b][2])/2;
    }
    edges3d.sort(function(a,b){return b.avgZ-a.avgZ});

    if(p.ref){
        c.strokeStyle=isEmb?'rgba(255,255,255,0.1)':'rgba(255,255,255,0.04)';
        c.lineWidth=0.4;
        for(var ei=0;ei<edges3d.length;ei++){
            var e=edges3d[ei];
            var pa=projO[e.a], pb=projO[e.b];
            c.beginPath();c.moveTo(pa[0],pa[1]);c.lineTo(pb[0],pb[1]);c.stroke();
        }
    }

    if(p.grid&&!isEmb){
        c.lineWidth=0.9;
        for(var ei=0;ei<edges3d.length;ei++){
            var e=edges3d[ei];
            var pa=projV[e.a], pb=projV[e.b];
            var depthAlpha=Math.max(0.1, Math.min(0.85, 0.6-e.avgZ*0.001));
            if(p.sc){
                var ec=s2c(e.strain);
                c.strokeStyle='rgba('+ec[0]+','+ec[1]+','+ec[2]+','+depthAlpha.toFixed(2)+')';
            } else {
                c.strokeStyle='rgba(200,200,200,'+depthAlpha.toFixed(2)+')';
            }
            c.beginPath();c.moveTo(pa[0],pa[1]);c.lineTo(pb[0],pb[1]);c.stroke();
        }
    }

    if(p.heat&&!isEmb){
        var faces=[];
        function addFace(a,b,cc2,d,s1,s2s,s3s,s4){
            var avgS=(s1+s2s+s3s+s4)/4;
            var avgZ2=(projV[a][2]+projV[b][2]+projV[cc2][2]+projV[d][2])/4;
            faces.push({verts:[a,b,cc2,d],strain:avgS,z:avgZ2});
        }
        for(var fiz=0;fiz<N3;fiz++)for(var fiy=0;fiy<N3;fiy++){
            var a=gIdx(0,fiy,fiz),b=gIdx(0,fiy+1,fiz),cc2=gIdx(0,fiy+1,fiz+1),d=gIdx(0,fiy,fiz+1);
            var s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            var s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            var s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            var s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
            a=gIdx(N3,fiy,fiz);b=gIdx(N3,fiy+1,fiz);cc2=gIdx(N3,fiy+1,fiz+1);d=gIdx(N3,fiy,fiz+1);
            s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
        }
        for(var fiz=0;fiz<N3;fiz++)for(var fix=0;fix<N3;fix++){
            var a=gIdx(fix,0,fiz),b=gIdx(fix+1,0,fiz),cc2=gIdx(fix+1,0,fiz+1),d=gIdx(fix,0,fiz+1);
            var s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            var s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            var s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            var s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
            a=gIdx(fix,N3,fiz);b=gIdx(fix+1,N3,fiz);cc2=gIdx(fix+1,N3,fiz+1);d=gIdx(fix,N3,fiz+1);
            s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
        }
        for(var fiy=0;fiy<N3;fiy++)for(var fix=0;fix<N3;fix++){
            var a=gIdx(fix,fiy,0),b=gIdx(fix+1,fiy,0),cc2=gIdx(fix+1,fiy+1,0),d=gIdx(fix,fiy+1,0);
            var s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            var s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            var s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            var s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
            a=gIdx(fix,fiy,N3);b=gIdx(fix+1,fiy,N3);cc2=gIdx(fix+1,fiy+1,N3);d=gIdx(fix,fiy+1,N3);
            s1=strain3(gX3[a],gY3[a],gZ3[a],gX3[b],gY3[b],gZ3[b],oX3[a],oY3[a],oZ3[a],oX3[b],oY3[b],oZ3[b]);
            s2s=strain3(gX3[b],gY3[b],gZ3[b],gX3[cc2],gY3[cc2],gZ3[cc2],oX3[b],oY3[b],oZ3[b],oX3[cc2],oY3[cc2],oZ3[cc2]);
            s3s=strain3(gX3[cc2],gY3[cc2],gZ3[cc2],gX3[d],gY3[d],gZ3[d],oX3[cc2],oY3[cc2],oZ3[cc2],oX3[d],oY3[d],oZ3[d]);
            s4=strain3(gX3[d],gY3[d],gZ3[d],gX3[a],gY3[a],gZ3[a],oX3[d],oY3[d],oZ3[d],oX3[a],oY3[a],oZ3[a]);
            addFace(a,b,cc2,d,s1,s2s,s3s,s4);
        }
        faces.sort(function(a,b){return b.z-a.z});
        for(var fi=0;fi<faces.length;fi++){
            var f=faces[fi];
            var co=s2c(f.strain);
            var va=projV[f.verts[0]],vb=projV[f.verts[1]],vc=projV[f.verts[2]],vd=projV[f.verts[3]];
            c.beginPath();
            c.moveTo(va[0],va[1]);c.lineTo(vb[0],vb[1]);
            c.lineTo(vc[0],vc[1]);c.lineTo(vd[0],vd[1]);
            c.closePath();
            c.fillStyle='rgba('+co[0]+','+co[1]+','+co[2]+',0.15)';c.fill();
        }
    }

    if(p.vec&&!isEmb){
        var step3=Math.max(1,Math.floor(N3/4));
        c.lineWidth=1.2;
        for(var viz=0;viz<=N3;viz+=step3)for(var viy=0;viy<=N3;viy+=step3)for(var vix=0;vix<=N3;vix+=step3){
            var vi=gIdx(vix,viy,viz);
            var pa=projO[vi], pb=projV[vi];
            var al=Math.hypot(pb[0]-pa[0],pb[1]-pa[1]);
            if(al<3)continue;
            c.strokeStyle='rgba(255,255,100,0.5)';c.fillStyle='rgba(255,255,100,0.5)';
            c.beginPath();c.moveTo(pa[0],pa[1]);c.lineTo(pb[0],pb[1]);c.stroke();
            var aa=Math.atan2(pb[1]-pa[1],pb[0]-pa[0]),hl=Math.min(6,al*.3);
            c.beginPath();c.moveTo(pb[0],pb[1]);
            c.lineTo(pb[0]-hl*Math.cos(aa-.4),pb[1]-hl*Math.sin(aa-.4));
            c.lineTo(pb[0]-hl*Math.cos(aa+.4),pb[1]-hl*Math.sin(aa+.4));
            c.closePath();c.fill();
        }
    }

    // 3D axes
    var axLen=mr3*0.5*effSc3;
    var axes=[
        {v:[1,0,0],label:'Dim '+dx,color:'#e94560'},
        {v:[0,1,0],label:'Dim '+dy,color:'#53a8b6'},
        {v:[0,0,1],label:'Dim '+dz,color:'#f5a623'}
    ];
    c.lineWidth=1.5;
    for(var ai=0;ai<3;ai++){
        var ax=axes[ai];
        var o3=proj3D(0,0,0);
        var e3=proj3D(ax.v[0]*axLen,ax.v[1]*axLen,ax.v[2]*axLen);
        c.strokeStyle=ax.color;c.globalAlpha=0.5;
        c.beginPath();c.moveTo(o3[0],o3[1]);c.lineTo(e3[0],e3[1]);c.stroke();
        c.globalAlpha=1;
        c.font='10px monospace';c.fillStyle=ax.color;
        c.fillText(ax.label,e3[0]+4,e3[1]-4);
    }

    // Points
    var points3d=[];
    for(var pi=0;pi<nP;pi++){
        var px3=(fx[pi]-cx3)*effSc3, py3=(fy[pi]-cy3)*effSc3, pz3=(fz[pi]-cz3)*effSc3;
        var proj=proj3D(px3,py3,pz3);
        points3d.push({idx:pi,sx:proj[0],sy:proj[1],z:proj[2],scale:proj[3],
            wx:px3,wy:py3,wz:pz3});
    }
    points3d.sort(function(a,b){return b.z-a.z});

    if(p.syn){
        for(var si=0;si<points3d.length;si++){
            var sp=points3d[si];
            if(sp.idx<nR)continue;
            var sr=Math.max(1,2.5*sp.scale);
            c.beginPath();c.arc(sp.sx,sp.sy,sr,0,Math.PI*2);
            c.fillStyle='rgba(100,200,255,0.15)';c.fill();
        }
    }

    if(p.nb && D.neighbors && selectedTokens.size>0){
        var kn=p.kn;
        selectedTokens.forEach(function(ti){
            if(ti>=D.neighbors.length)return;
            var nbs=D.neighbors[ti].slice(0,kn);
            var tpx=(fx[ti]-cx3)*effSc3,tpy=(fy[ti]-cy3)*effSc3,tpz=(fz[ti]-cz3)*effSc3;
            var tp=proj3D(tpx,tpy,tpz);
            for(var ni=0;ni<nbs.length;ni++){
                var nb=nbs[ni];
                var nidx=nb.idx;
                if(nidx>=nP)continue;
                var npx=(fx[nidx]-cx3)*effSc3,npy=(fy[nidx]-cy3)*effSc3,npz=(fz[nidx]-cz3)*effSc3;
                var np2=proj3D(npx,npy,npz);
                var alpha=Math.max(0.15,1.0-ni*0.08);
                c.strokeStyle='rgba(0,255,200,'+alpha.toFixed(2)+')';
                c.lineWidth=Math.max(0.5,2.5-ni*0.2);
                c.setLineDash([3,3]);
                c.beginPath();c.moveTo(tp[0],tp[1]);c.lineTo(np2[0],np2[1]);c.stroke();
                c.setLineDash([]);
                var nr=Math.max(2,5*np2[3]);
                c.beginPath();c.arc(np2[0],np2[1],nr,0,Math.PI*2);
                c.fillStyle=nb.is_real?'rgba(0,255,200,0.8)':'rgba(0,255,200,0.35)';
                c.fill();
                if(p.nblabel){
                    c.font='9px monospace';c.fillStyle='rgba(0,255,200,0.9)';
                    c.fillText(nb.label+' (d='+nb.dist.toFixed(1)+')',np2[0]+8,np2[1]-4);
                }
            }
        });
    }

    if(p.tok){
        var tc=['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22',
            '#f39c12','#d35400','#c0392b','#16a085','#27ae60',
            '#2980b9','#8e44ad','#2c3e50','#ecf0f1','#fd79a8'];
        for(var ri=0;ri<points3d.length;ri++){
            var rp=points3d[ri];
            if(rp.idx>=nR)continue;
            var ti3=rp.idx;
            var col=tc[ti3%tc.length];
            var isSel=selectedTokens.has(ti3);
            var dotR=Math.max(3,(isSel?9:7)*rp.scale);

            if(isSel){
                var grad2=c.createRadialGradient(rp.sx,rp.sy,0,rp.sx,rp.sy,30*rp.scale);
                grad2.addColorStop(0,'rgba(0,255,0,0.25)');grad2.addColorStop(1,'rgba(0,255,0,0)');
                c.beginPath();c.arc(rp.sx,rp.sy,30*rp.scale,0,Math.PI*2);c.fillStyle=grad2;c.fill();
            }

            c.beginPath();c.arc(rp.sx,rp.sy,dotR,0,Math.PI*2);
            c.fillStyle=col;c.fill();
            c.strokeStyle=isSel?'#0f0':'#fff';c.lineWidth=isSel?3:2;c.stroke();

            var fontSize=Math.max(8,11*rp.scale);
            c.font='bold '+Math.round(fontSize)+'px monospace';
            c.lineWidth=3;c.strokeStyle='rgba(0,0,0,0.9)';
            var lb='['+ti3+'] '+D.tokens[ti3];
            c.strokeText(lb,rp.sx+12,rp.sy-10);
            c.fillStyle=isSel?'#0f0':'#fff';c.fillText(lb,rp.sx+12,rp.sy-10);
        }

        if(isEmb&&nR>1){
            c.strokeStyle='rgba(233,69,96,0.3)';c.lineWidth=1.5;c.setLineDash([4,4]);
            c.beginPath();
            var fp0=(fx[0]-cx3)*effSc3,fp0y=(fy[0]-cy3)*effSc3,fp0z=(fz[0]-cz3)*effSc3;
            var pp0=proj3D(fp0,fp0y,fp0z);
            c.moveTo(pp0[0],pp0[1]);
            for(var ti4=1;ti4<nR;ti4++){
                var fpx=(fx[ti4]-cx3)*effSc3,fpy=(fy[ti4]-cy3)*effSc3,fpz=(fz[ti4]-cz3)*effSc3;
                var pp=proj3D(fpx,fpy,fpz);
                c.lineTo(pp[0],pp[1]);
            }
            c.stroke();c.setLineDash([]);
        }
    }

    // HUD text
    var decompLabel = getDecompLabel();
    c.font='11px monospace';c.fillStyle='rgba(255,255,255,0.45)';
    if(isEmb){
        c.fillText('EMBEDDING SPACE [3D]  Dims:'+dx+','+dy+','+dz+'  Drag to rotate',42,18);
    } else {
        c.fillText('Layer '+p.layer+'/'+(D.n_layers-1)+'  t='+p.t.toFixed(2)+'  amp='+p.amp.toFixed(1)+'  Dims:'+dx+','+dy+','+dz+'  Decomp:'+decompLabel+'  [3D]  Drag to rotate',42,18);
    }
    c.font='10px monospace';c.fillStyle='rgba(255,255,255,0.35)';
    c.fillText('Zoom: '+zoomLevel.toFixed(2)+'x  (Scroll=zoom, Shift+drag=pan, 0=reset)',42,H-10);
}

// Scroll-wheel zoom
cv3d.addEventListener('wheel', function(e) {
    e.preventDefault();
    var rect = cv3d.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    var oldZoom = zoomLevel;
    var zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    zoomLevel = Math.max(0.1, Math.min(50, zoomLevel * zoomFactor));

    if(viewMode==='2d'){
        panX = mx - (mx - panX) * (zoomLevel / oldZoom);
        panY = my - (my - panY) * (zoomLevel / oldZoom);
    } else {
        var W = cv3d.width, H = cv3d.height;
        var cx2dOld = W/2 + panX;
        var cy2dOld = H/2 + panY;
        var dmx = mx - cx2dOld;
        var dmy = my - cy2dOld;
        panX += dmx * (1 - zoomLevel / oldZoom);
        panY += dmy * (1 - zoomLevel / oldZoom);
    }

    draw();
}, { passive: false });

</script></body></html>"""


# ============================================================
# 13. HTTP SERVER (thin wrappers)
# ============================================================

def handle_get_index():
    """Return the HTML page as bytes."""
    return HTML_PAGE.encode("utf-8")


def handle_post_run(body_bytes):
    """Parse a /run POST request and return JSON response bytes."""
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    model_name = req.get("model", "").strip() or None
    if not text:
        raise ValueError("Empty text")
    print(f"\n[Server] Processing: {text[:60]}... (model: {model_name or MODEL_NAME})...")
    json_str = process_text(text, model_name)
    return json_str.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(handle_get_index())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/run":
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                response_bytes = handle_post_run(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


# ============================================================
# 14. BROWSER OPENER
# ============================================================

def open_browser_delayed(port, delay=1.0):
    """Open the browser after a short delay."""
    time.sleep(delay)
    webbrowser.open(f"http://127.0.0.1:{port}")


# ============================================================
# 15. ARGUMENT PARSING
# ============================================================

def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Metric Space Explorer")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Initial model to load (any HuggingFace model name)")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


# ============================================================
# 16. MAIN ENTRY POINT
# ============================================================

def start_server(host, port):
    """Create and return an HTTPServer instance."""
    return HTTPServer((host, port), Handler)


def main():
    args = parse_args()

    load_model(args.model)

    threading.Thread(
        target=open_browser_delayed,
        args=(args.port,),
        daemon=True,
    ).start()

    server = start_server("127.0.0.1", args.port)
    print(f"\n[Server] http://127.0.0.1:{args.port}")
    print("[Server] Ctrl+C to stop\n")
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
