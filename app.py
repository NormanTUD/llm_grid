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
from datetime import datetime, timedelta


# ============================================================
# 1. ENVIRONMENT SAFETY
# ============================================================

def compute_exclude_newer_date(days_back=8):
    """Return a UTC timestamp string for `days_back` days ago."""
    return (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")


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
                       all_deltas_per_point, n_layers, hidden_dim, n_side=10, pad_frac=0.3):
    """
    Create grid intersection probes in PCA space.
    Returns (grid_layer0, grid_deltas).
    """
    mn1, mx1, r1 = compute_grid_range(proj1, pad_frac)
    mn2, mx2, r2 = compute_grid_range(proj2, pad_frac)
    g1 = np.linspace(mn1, mx1, n_side)
    g2 = np.linspace(mn2, mx2, n_side)
    sigma_nn = r1 * 0.2

    grid_layer0 = []
    grid_deltas = []

    for v1, v2 in make_grid_coords(g1, g2):
        emb = interpolate_grid_embedding(v1, v2, centroid, pc1, pc2)
        grid_layer0.append(emb)

        weights = compute_grid_weights(v1, v2, existing_proj, sigma_nn)
        point_deltas = interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim)
        grid_deltas.append(point_deltas)

    return grid_layer0, grid_deltas


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
                      next_token_preds=None, vocab_neighbors=None):
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
    return data

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

    print(f"[Model] Running {len(all_seqs)} sequences through model...")

    # Extract hidden states
    all_layer0, all_deltas_per_point = run_all_sequences(MODEL, all_seqs, n_layers)

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

    grid_layer0, grid_deltas = create_grid_probes(
        centroid, pc1, pc2, proj1, proj2, existing_proj,
        all_deltas_per_point, n_layers, hidden_dim,
        n_side=10, pad_frac=0.3,
    )

    n_grid = len(grid_layer0)
    print(f"[Model] Added {n_grid} grid intersection probes")

    # Append grid probes
    for gi in range(n_grid):
        all_layer0.append(grid_layer0[gi])
        all_deltas_per_point.append(grid_deltas[gi])
        all_labels.append("·")
        all_is_real.append(False)

    n_total_final = len(all_layer0)

    # Build output
    fixed_pos = build_fixed_pos(all_layer0)
    deltas = build_deltas_array(all_deltas_per_point, n_layers, n_total_final)

    data = build_output_data(
        all_labels, all_is_real, n_layers, n_total_final, n_real,
        hidden_dim, fixed_pos, deltas, MODEL_NAME, text, neighbors,
        next_token_preds=next_token_preds,
        vocab_neighbors=vocab_neighbors,
    )

    json_str = json.dumps(data)
    print(f"[Model] JSON: {len(json_str)/1024/1024:.1f} MB")
    return json_str


# ============================================================
# 12. HTML (unchanged — kept as constant)
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
#neighbor-panel{background:#0f3460;padding:6px;border-radius:4px;font-size:10px;line-height:1.5;max-height:200px;overflow-y:auto;display:none}
#neighbor-panel h4{color:#e94560;font-size:11px;margin-bottom:4px}
.nb-item{color:#a0a0c0;cursor:pointer;padding:1px 4px;border-radius:2px}
.nb-item:hover{background:#1a1a2e;color:#fff}
.nb-item.is-real{color:#53a8b6;font-weight:bold}
.nb-item .nb-dist{color:#666;font-size:9px;margin-left:4px}
#selected-tokens{display:flex;flex-wrap:wrap;gap:3px;margin-top:4px;min-height:20px}
.sel-tok{background:#e94560;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;cursor:pointer;display:flex;align-items:center;gap:3px}
.sel-tok:hover{background:#c73e54}
.sel-tok .x{font-weight:bold;margin-left:2px}
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
<h3>Layer &amp; Deformation</h3>
<div class="cr"><label>Layer:</label><input type="range" id="sl-layer" min="0" max="11" value="0" step="1"><span class="v" id="v-layer">0</span></div>
<div class="cr"><label>Deform t:</label><input type="range" id="sl-t" min="0" max="1" value="1.0" step="0.01"><span class="v" id="v-t">1.00</span></div>
<div class="cr"><label>Amplify:</label><input type="range" id="sl-amp" min="0.1" max="500" value="1" step="0.1"><span class="v" id="v-amp">1.0</span></div>
<div class="cr"><label>Mode:</label>
<select id="sel-mode">
<option value="single">🔍 This Layer Only — show deformation from one layer</option>
<option value="cumfwd">⏩ Layers 0→L (Cumulative) — total deformation up to this layer</option>
<option value="cumbwd">⏪ Layers L→End (Cumulative) — remaining deformation after this layer</option>
<option value="embedding">📐 Raw Embedding Space — no deformation, just token positions</option>
</select>
</div>
<h3>Dimensions</h3>
<div class="cr"><label>Dim X:</label><input type="range" id="sl-dx" min="0" max="767" value="0" step="1"><span class="v" id="v-dx">0</span></div>
<div class="cr"><label>Dim Y:</label><input type="range" id="sl-dy" min="0" max="767" value="1" step="1"><span class="v" id="v-dy">1</span></div>
<h3>Neighbor Tracing</h3>
<div class="cr"><label>K Neighbors:</label><input type="range" id="sl-kn" min="1" max="20" value="5" step="1"><span class="v" id="v-kn">5</span></div>
<div class="cb"><input type="checkbox" id="cb-nb" checked><label for="cb-nb">Show Neighbor Lines</label></div>
<div class="cb"><input type="checkbox" id="cb-nblabel" checked><label for="cb-nblabel">Show Neighbor Labels</label></div>
<h3>RBF &amp; Grid</h3>
<div class="cr"><label>Bandwidth σ:</label><input type="range" id="sl-sig" min="0.01" max="20" value="1.0" step="0.01"><span class="v" id="v-sig">1.00</span></div>
<div class="cr"><label>Grid Res:</label><input type="range" id="sl-gr" min="10" max="80" value="30" step="1"><span class="v" id="v-gr">30</span></div>
<h3>Display</h3>
<div class="cb"><input type="checkbox" id="cb-grid" checked><label for="cb-grid">Deformed Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-vocnb" checked><label for="cb-vocnb">Show Nearby Vocab Words (greyed)</label></div>
<div class="cb"><input type="checkbox" id="cb-heat" checked><label for="cb-heat">Strain Heatmap</label></div>
<div class="cb"><input type="checkbox" id="cb-ref" checked><label for="cb-ref">Reference Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-tok" checked><label for="cb-tok">Real Tokens</label></div>
<div class="cb"><input type="checkbox" id="cb-syn"><label for="cb-syn">Probe Points</label></div>
<div class="cb"><input type="checkbox" id="cb-sc" checked><label for="cb-sc">Strain Color</label></div>
<div class="cb"><input type="checkbox" id="cb-vec"><label for="cb-vec">Vector Arrows</label></div>
<div id="keys"><b>Keys:</b> ←→ Layer | ↑↓ t | A/Z Amp | Space Auto | R Reset | D Dims | Click token to trace neighbors</div>
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
var D=null,AP=null;
var selectedTokens=new Set();

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

function onData(){
    document.getElementById('sl-layer').max=D.n_layers-1;
    document.getElementById('sl-dx').max=D.hidden_dim-1;
    document.getElementById('sl-dy').max=D.hidden_dim-1;
    document.getElementById('i-mod').textContent=D.model_name;
    document.getElementById('i-pts').textContent=D.n_points;
    document.getElementById('i-real').textContent=D.n_real;
    document.getElementById('i-syn').textContent=D.n_synth;
    document.getElementById('i-lay').textContent=D.n_layers;
    document.getElementById('i-dim').textContent=D.hidden_dim;
    document.getElementById('i-tok').textContent=D.tokens.slice(0,D.n_real).join(' ');
    document.getElementById('sel-model').value=D.model_name;
    autoParams();
    draw();
    document.getElementById('status').textContent='Ready — '+D.n_real+' tokens, '+D.n_synth+' probes | Model: '+D.model_name;

// Show next token predictions
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
    var norms=[];
    for(var l=0;l<D.n_layers;l++){
        for(var p=0;p<nP;p++){
            var ddx=D.deltas[l][p][dx],ddy=D.deltas[l][p][dy];
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

document.getElementById('cv').addEventListener('click', function(e){
    if(!D)return;
    var cv=document.getElementById('cv');
    var rect=cv.getBoundingClientRect();
    var mx=e.clientX-rect.left, my=e.clientY-rect.top;
    var dx=+document.getElementById('sl-dx').value;
    var dy=+document.getElementById('sl-dy').value;
    var nP=D.n_points, nR=D.n_real;
    var W=cv.width, H=cv.height, M2=42, dW2=W-2*M2, dH2=H-2*M2;
    var fx=new Float64Array(nP),fy=new Float64Array(nP);
    for(var i=0;i<nP;i++){fx[i]=D.fixed_pos[i][dx];fy[i]=D.fixed_pos[i][dy]}
    var mnx=fx[0],mxx=fx[0],mny=fy[0],mxy=fy[0];
    for(var i2=1;i2<nP;i2++){
        if(fx[i2]<mnx)mnx=fx[i2];if(fx[i2]>mxx)mxx=fx[i2];
        if(fy[i2]<mny)mny=fy[i2];if(fy[i2]>mxy)mxy=fy[i2];
    }
    var mr=Math.max(mxx-mnx,mxy-mny)||1;
    var cxv=(mnx+mxx)/2,cyv=(mny+mxy)/2;
    var pd2=0.12;
    var vx0=cxv-mr*(.5+pd2),vy0=cyv-mr*(.5+pd2),vw=mr*(1+2*pd2),vh=vw;
    function SX(x){return M2+((x-vx0)/vw)*dW2}
    function SY(y){return M2+((y-vy0)/vh)*dH2}
    var bestDist=Infinity, bestIdx=-1;
    for(var ti=0;ti<nR;ti++){
        var sx=SX(fx[ti]), sy=SY(fy[ti]);
        var dd=Math.hypot(mx-sx, my-sy);
        if(dd<bestDist){bestDist=dd;bestIdx=ti}
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
    ['sl-dx','v-dx',0],['sl-dy','v-dy',0],
    ['sl-sig','v-sig',2],['sl-gr','v-gr',0]
];
for(var i=0;i<SLS.length;i++)(function(c){
    var s=document.getElementById(c[0]),v=document.getElementById(c[1]),dec=c[2];
    v.textContent=parseFloat(s.value).toFixed(dec);
    s.addEventListener('input',function(){
        v.textContent=parseFloat(s.value).toFixed(dec);
        if(c[0]==='sl-dx'||c[0]==='sl-dy')autoParams();
        draw();
    });
})(SLS[i]);

['cb-grid','cb-heat','cb-ref','cb-tok','cb-syn','cb-sc','cb-vec','cb-vocnb'].forEach(function(id){
    document.getElementById(id).addEventListener('change',draw);
});

document.getElementById('sel-mode').addEventListener('change',draw);
document.addEventListener('keydown',onKey);
document.getElementById('txt-in').addEventListener('keydown',function(e){if(e.key==='Enter')runText()});

function gp(){return{
    layer:+document.getElementById('sl-layer').value,
    t:+document.getElementById('sl-t').value,
    amp:+document.getElementById('sl-amp').value,
    mode:document.getElementById('sel-mode').value,
    dx:+document.getElementById('sl-dx').value,
    dy:+document.getElementById('sl-dy').value,
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
    document.getElementById('sl-gr').value='30';
    selectedTokens.clear();updateSelectedUI();
    if(D)autoParams();
    ['sl-layer','sl-t','sl-amp','sl-dx','sl-dy','sl-sig','sl-gr'].forEach(function(id){
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
    var p=gp(),cv=document.getElementById('cv'),c=cv.getContext('2d');
    var W=cv.width,H=cv.height;
    c.clearRect(0,0,W,H);

    var nP=D.n_points,nR=D.n_real,dx=p.dx,dy=p.dy;
    var isEmb=p.mode==='embedding';

    var fx=new Float64Array(nP),fy=new Float64Array(nP);
    for(var i=0;i<nP;i++){fx[i]=D.fixed_pos[i][dx];fy[i]=D.fixed_pos[i][dy]}

    var edx=new Float64Array(nP),edy=new Float64Array(nP);
    if(!isEmb){
        var layer=p.layer,amp=p.amp;
        for(var j=0;j<nP;j++){
            var sx2=0,sy2=0;
            if(p.mode==='single'){sx2=D.deltas[layer][j][dx];sy2=D.deltas[layer][j][dy]}
            else if(p.mode==='cumfwd'){for(var l=0;l<=layer;l++){sx2+=D.deltas[l][j][dx];sy2+=D.deltas[l][j][dy]}}
            else{for(var l2=layer;l2<D.n_layers;l2++){sx2+=D.deltas[l2][j][dx];sy2+=D.deltas[l2][j][dy]}}
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
    for(var ey=0;ey<=N;ey++)for(var ex=0;ex<N;ex++){
        var a=ey*(N+1)+ex,b=a+1;
        var od=Math.hypot(oX[b]-oX[a],oY[b]-oY[a]);
        var dd=Math.hypot(gX[b]-gX[a],gY[b]-gY[a]);
        sH[ey*N+ex]=od>1e-12?dd/od:1;
    }
    for(var ey2=0;ey2<N;ey2++)for(var ex2=0;ex2<=N;ex2++){
        var a2=ey2*(N+1)+ex2,b2=(ey2+1)*(N+1)+ex2;
        var od2=Math.hypot(oX[b2]-oX[a2],oY[b2]-oY[a2]);
        var dd2=Math.hypot(gX[b2]-gX[a2],gY[b2]-gY[a2]);
        sVa[ey2*(N+1)+ex2]=od2>1e-12?dd2/od2:1;
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

    c.font='11px monospace';c.fillStyle='rgba(255,255,255,0.45)';
    if(isEmb){
        c.fillText('EMBEDDING SPACE  Dims:'+dx+','+dy,M,18);
        c.fillText('Token positions in layer-0 embedding space. Change dims to explore.',M,H-8);
    } else {
        c.fillText('Layer '+p.layer+'/'+(D.n_layers-1)+'  t='+p.t.toFixed(2)+'  amp='+p.amp.toFixed(1)+'  Dims:'+dx+','+dy+'  Mode:'+p.mode+'  \u03C3='+p.sig.toFixed(2),M,18);
        var sMin=Infinity,sMax=-Infinity,sSum=0,sN2=0;
        for(var q=0;q<sH.length;q++){if(sH[q]<sMin)sMin=sH[q];if(sH[q]>sMax)sMax=sH[q];sSum+=sH[q];sN2++}
        for(var q2=0;q2<sVa.length;q2++){if(sVa[q2]<sMin)sMin=sVa[q2];if(sVa[q2]>sMax)sMax=sVa[q2];sSum+=sVa[q2];sN2++}
        c.fillText('Strain: min='+sMin.toFixed(4)+' avg='+(sSum/sN2).toFixed(4)+' max='+sMax.toFixed(4),M,H-8);
    }

// Draw vocab neighbors (greyed out nearby words)
if(document.getElementById('cb-vocnb').checked && D.vocab_neighbors && p.tok){
    c.font='9px monospace';
    for(var vi=0;vi<nR;vi++){
        if(!D.vocab_neighbors[vi])continue;
        var vtx=SX(fx[vi]),vty=SY(fy[vi]);
        var vnbs=D.vocab_neighbors[vi];
        for(var vni=0;vni<vnbs.length;vni++){
            var vnb=vnbs[vni];
            // Position them in a fan around the token
            var angle = -Math.PI/2 + (vni/(vnbs.length-1||1)) * Math.PI;
            var radius = 35 + vni * 8;
            var vnx = vtx + Math.cos(angle) * radius;
            var vny = vty + Math.sin(angle) * radius + 20;
            c.fillStyle='rgba(150,150,170,0.45)';
            c.fillText(vnb.token, vnx, vny);
        }
    }
}
}
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
