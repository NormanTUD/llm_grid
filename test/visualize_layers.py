#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "networkx",
#   "scikit-learn",
#   "Pillow",
#   "seaborn",
# ]
# ///
"""
Weight Graph Analyzer — Interpret transformer weights as weighted graphs.
Loads GPT-2 (124M) and GPT-2 XL (1.5B) [or GPT-Neo 1.3B as GPT-3 proxy],
builds graph representations of every weight matrix, and visualizes:

  • Weight matrices as raw canvases (heatmaps)
  • Neuron-to-neuron weighted graphs per layer
  • Degree distributions, centrality, clustering coefficients
  • Spectral analysis (eigenvalues of weight-derived adjacency)
  • Community detection on weight graphs
  • Cross-model comparison dashboards
  • Attention head connectivity graphs
  • MLP weight flow graphs
  • Singular value spectra per layer
  • Weight norm landscapes
  • Cosine similarity between neurons
  • Small-world / scale-free diagnostics

Run:
    uv run weight_graph_analyzer.py

Or:
    uv run weight_graph_analyzer.py --model2 EleutherAI/gpt-neo-1.3B --output ./my_results
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, UTC
import argparse

# ============================================================
# AUTO-INSTALL / UV SAFETY (same pattern as your app.py)
# ============================================================

def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_safe_env():
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        os.environ["UV_EXCLUDE_NEWER"] = compute_exclude_newer_date(8)
        try:
            os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
        except FileNotFoundError:
            print("uv is not installed. Try: curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)

ensure_safe_env()

# ============================================================
# HEAVY IMPORTS
# ============================================================

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import networkx as nx
from scipy import sparse
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from transformers import AutoModel, AutoTokenizer, AutoConfig
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL1 = "gpt2"
DEFAULT_MODEL2 = "gpt2-xl"  # Proxy for "GPT-3 class" — swap to gpt-neo-1.3B etc.

# How many neurons to subsample for graph construction (full is too large)
MAX_NEURONS_FOR_GRAPH = 256
# Top-k edges per neuron for sparse graph
TOP_K_EDGES = 16
# Output directory
OUTPUT_DIR = Path("weight_graph_output")


def parse_args():
    parser = argparse.ArgumentParser(description="Weight Graph Analyzer")
    parser.add_argument("--model1", default=DEFAULT_MODEL1, help="First model (default: gpt2)")
    parser.add_argument("--model2", default=DEFAULT_MODEL2, help="Second model (default: gpt2-xl)")
    parser.add_argument("--output", default="weight_graph_output", help="Output directory")
    parser.add_argument("--max-neurons", type=int, default=MAX_NEURONS_FOR_GRAPH,
                        help="Max neurons for graph construction")
    parser.add_argument("--top-k-edges", type=int, default=TOP_K_EDGES,
                        help="Top-K edges per neuron in graph")
    parser.add_argument("--skip-model2", action="store_true",
                        help="Only analyze model1")
    return parser.parse_args()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_config(model_name):
    """Load a HuggingFace model, return (model, config, name)."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Layers: {get_n_layers(config)}")
    print(f"  Hidden dim: {get_hidden_dim(config)}")
    print(f"  Heads: {getattr(config, 'n_head', getattr(config, 'num_attention_heads', '?'))}")
    return model, config, model_name


def get_n_layers(config):
    for attr in ["n_layer", "num_hidden_layers", "num_layers"]:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return 12


def get_hidden_dim(config):
    for attr in ["n_embd", "hidden_size", "d_model"]:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return 768


def get_n_heads(config):
    for attr in ["n_head", "num_attention_heads"]:
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return 12


# ============================================================
# WEIGHT EXTRACTION
# ============================================================

def extract_layer_weights(model, config):
    """
    Extract all weight matrices organized by layer.
    Returns a list of dicts, one per layer, each containing:
        - 'attn_qkv': combined QKV weight or separate Q, K, V
        - 'attn_proj': attention output projection
        - 'mlp_up': MLP up-projection (fc1 / c_fc)
        - 'mlp_down': MLP down-projection (fc2 / c_proj)
        - 'ln1': LayerNorm 1 weights
        - 'ln2': LayerNorm 2 weights
    All as numpy arrays.
    """
    layers = []
    n_layers = get_n_layers(config)

    # Try to find transformer blocks
    blocks = None
    if hasattr(model, 'h'):
        blocks = list(model.h)  # GPT-2
    elif hasattr(model, 'layers'):
        blocks = list(model.layers)  # GPT-Neo, Pythia
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        blocks = list(model.encoder.layer)  # BERT
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        blocks = list(model.decoder.layers)  # OPT

    if blocks is None:
        print("  WARNING: Could not find transformer blocks, extracting from state_dict")
        return extract_weights_from_state_dict(model, n_layers)

    for layer_idx, block in enumerate(blocks):
        layer_data = {"layer_idx": layer_idx}

        # GPT-2 style
        if hasattr(block, 'attn'):
            attn = block.attn
            if hasattr(attn, 'c_attn'):
                layer_data['attn_qkv'] = attn.c_attn.weight.detach().cpu().float().numpy()
            if hasattr(attn, 'c_proj'):
                layer_data['attn_proj'] = attn.c_proj.weight.detach().cpu().float().numpy()

        # GPT-Neo style
        if hasattr(block, 'attention'):
            attn = block.attention
            if hasattr(attn, 'out_proj'):
                layer_data['attn_proj'] = attn.out_proj.weight.detach().cpu().float().numpy()
            # GPT-Neo uses local/global attention alternating
            if hasattr(attn, 'attention'):
                inner = attn.attention
                if hasattr(inner, 'q_proj'):
                    q = inner.q_proj.weight.detach().cpu().float().numpy()
                    k = inner.k_proj.weight.detach().cpu().float().numpy()
                    v = inner.v_proj.weight.detach().cpu().float().numpy()
                    layer_data['attn_q'] = q
                    layer_data['attn_k'] = k
                    layer_data['attn_v'] = v

        # MLP
        if hasattr(block, 'mlp'):
            mlp = block.mlp
            if hasattr(mlp, 'c_fc'):
                layer_data['mlp_up'] = mlp.c_fc.weight.detach().cpu().float().numpy()
            if hasattr(mlp, 'c_proj'):
                layer_data['mlp_down'] = mlp.c_proj.weight.detach().cpu().float().numpy()
            # GPT-Neo
            if hasattr(mlp, 'c_fc') is False and hasattr(mlp, 'dense_h_to_4h'):
                layer_data['mlp_up'] = mlp.dense_h_to_4h.weight.detach().cpu().float().numpy()
            if hasattr(mlp, 'c_proj') is False and hasattr(mlp, 'dense_4h_to_h'):
                layer_data['mlp_down'] = mlp.dense_4h_to_h.weight.detach().cpu().float().numpy()

        # LayerNorms
        if hasattr(block, 'ln_1'):
            layer_data['ln1'] = block.ln_1.weight.detach().cpu().float().numpy()
        if hasattr(block, 'ln_2'):
            layer_data['ln2'] = block.ln_2.weight.detach().cpu().float().numpy()

        layers.append(layer_data)

    print(f"  Extracted weights from {len(layers)} layers")
    return layers


def extract_weights_from_state_dict(model, n_layers):
    """Fallback: extract weights from state_dict by name matching."""
    sd = model.state_dict()
    layers = []
    for li in range(n_layers):
        layer_data = {"layer_idx": li}
        for name, param in sd.items():
            if f".{li}." in name or f"layer.{li}." in name:
                short_name = name.split(".")[-2] + "_" + name.split(".")[-1]
                layer_data[short_name] = param.cpu().float().numpy()
        layers.append(layer_data)
    return layers


# ============================================================
# GRAPH CONSTRUCTION FROM WEIGHTS
# ============================================================

def weight_matrix_to_graph(W, max_neurons=256, top_k=16, name="weight"):
    """
    Interpret a weight matrix W (out_dim x in_dim) as a bipartite weighted graph.
    Also create a neuron-neuron similarity graph based on weight vector cosine similarity.

    Returns:
        bipartite_G: nx.Graph — bipartite graph (input neurons -> output neurons)
        similarity_G: nx.Graph — cosine similarity graph among output neurons
        stats: dict of graph statistics
    """
    out_dim, in_dim = W.shape

    # Subsample if too large
    out_idx = np.linspace(0, out_dim - 1, min(max_neurons, out_dim), dtype=int)
    in_idx = np.linspace(0, in_dim - 1, min(max_neurons, in_dim), dtype=int)
    W_sub = W[np.ix_(out_idx, in_idx)]

    # ---- Bipartite Graph: input -> output ----
    bipartite_G = nx.Graph()
    for i, ii in enumerate(in_idx):
        bipartite_G.add_node(f"in_{ii}", bipartite=0, neuron_type="input")
    for o, oi in enumerate(out_idx):
        bipartite_G.add_node(f"out_{oi}", bipartite=1, neuron_type="output")

    # Add top-k strongest connections per output neuron
    for o_local, o_global in enumerate(out_idx):
        row = np.abs(W_sub[o_local])
        if top_k < len(row):
            top_indices = np.argsort(row)[-top_k:]
        else:
            top_indices = np.arange(len(row))
        for i_local in top_indices:
            i_global = in_idx[i_local]
            weight_val = float(W_sub[o_local, i_local])
            if abs(weight_val) > 1e-8:
                bipartite_G.add_edge(f"in_{i_global}", f"out_{o_global}",
                                     weight=abs(weight_val),
                                     raw_weight=weight_val)

    # ---- Similarity Graph: cosine similarity among output neurons ----
    # Each output neuron is characterized by its weight vector (row of W)
    n_out = len(out_idx)
    W_rows = W_sub  # (n_out, n_in)

    # Normalize rows
    norms = np.linalg.norm(W_rows, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    W_normed = W_rows / norms

    # Cosine similarity matrix
    cos_sim = W_normed @ W_normed.T  # (n_out, n_out)
    np.fill_diagonal(cos_sim, 0)  # no self-loops

    similarity_G = nx.Graph()
    for o, oi in enumerate(out_idx):
        similarity_G.add_node(f"n_{oi}", neuron_idx=int(oi))

    # Add top-k edges per neuron
    for o in range(n_out):
        row = cos_sim[o]
        if top_k < n_out:
            top_indices = np.argsort(np.abs(row))[-top_k:]
        else:
            top_indices = np.arange(n_out)
        for j in top_indices:
            if j != o and abs(row[j]) > 0.01:
                similarity_G.add_edge(f"n_{out_idx[o]}", f"n_{out_idx[j]}",
                                      weight=float(row[j]))

    # ---- Compute Statistics ----
    stats = {}
    stats['n_nodes_bipartite'] = bipartite_G.number_of_nodes()
    stats['n_edges_bipartite'] = bipartite_G.number_of_edges()
    stats['n_nodes_similarity'] = similarity_G.number_of_nodes()
    stats['n_edges_similarity'] = similarity_G.number_of_edges()

    if similarity_G.number_of_nodes() > 2 and similarity_G.number_of_edges() > 0:
        try:
            stats['avg_clustering'] = round(nx.average_clustering(similarity_G, weight='weight'), 6)
        except Exception:
            stats['avg_clustering'] = 0.0

        try:
            if nx.is_connected(similarity_G):
                stats['avg_path_length'] = round(nx.average_shortest_path_length(similarity_G), 4)
            else:
                largest_cc = max(nx.connected_components(similarity_G), key=len)
                sub = similarity_G.subgraph(largest_cc)
                stats['avg_path_length'] = round(nx.average_shortest_path_length(sub), 4)
                stats['n_connected_components'] = nx.number_connected_components(similarity_G)
        except Exception:
            stats['avg_path_length'] = float('inf')

        degrees = [d for _, d in similarity_G.degree()]
        stats['avg_degree'] = round(np.mean(degrees), 2)
        stats['max_degree'] = max(degrees)
        stats['degree_std'] = round(np.std(degrees), 2)

        # Degree centrality
        dc = nx.degree_centrality(similarity_G)
        stats['max_centrality_node'] = max(dc, key=dc.get)
        stats['max_centrality'] = round(max(dc.values()), 4)

        # Density
        stats['density'] = round(nx.density(similarity_G), 6)
    else:
        stats['avg_clustering'] = 0.0
        stats['avg_degree'] = 0.0

    return bipartite_G, similarity_G, cos_sim, stats


def compute_spectral_properties(W, max_neurons=256):
    """
    Compute spectral properties of a weight matrix:
    - Singular values
    - Eigenvalues of W^T W (Gram matrix)
    - Eigenvalues of the cosine similarity matrix (graph Laplacian proxy)
    """
    out_dim, in_dim = W.shape

    # Singular values
    sv = np.linalg.svd(W, compute_uv=False)

    # Eigenvalues of the Gram matrix (W^T W)
    if min(out_dim, in_dim) <= 1024:
        gram = W.T @ W
        gram_eigs = np.linalg.eigvalsh(gram)
        gram_eigs = np.sort(gram_eigs)[::-1]
    else:
        # Subsample for large matrices
        idx = np.linspace(0, min(out_dim, in_dim) - 1, 512, dtype=int)
        W_sub = W[np.ix_(idx if out_dim > 512 else np.arange(out_dim),
                          idx if in_dim > 512 else np.arange(in_dim))]
        gram = W_sub.T @ W_sub
        gram_eigs = np.linalg.eigvalsh(gram)
        gram_eigs = np.sort(gram_eigs)[::-1]

    # Cosine similarity matrix eigenvalues (neuron graph spectrum)
    n_sub = min(max_neurons, out_dim)
    idx_out = np.linspace(0, out_dim - 1, n_sub, dtype=int)
    W_rows = W[idx_out]
    norms = np.linalg.norm(W_rows, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    W_normed = W_rows / norms
    cos_sim = W_normed @ W_normed.T
    cos_eigs = np.linalg.eigvalsh(cos_sim)
    cos_eigs = np.sort(cos_eigs)[::-1]

    # Effective rank (from singular values)
    sv_norm = sv / max(sv.sum(), 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-12]
    effective_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))

    # Stable rank
    stable_rank = float((sv ** 2).sum() / max(sv[0] ** 2, 1e-12))

    return {
        'singular_values': sv,
        'gram_eigenvalues': gram_eigs,
        'cosine_eigenvalues': cos_eigs,
        'effective_rank': effective_rank,
        'stable_rank': stable_rank,
        'condition_number': float(sv[0] / max(sv[-1], 1e-12)),
        'frobenius_norm': float(np.linalg.norm(W, 'fro')),
        'spectral_norm': float(sv[0]),
        'nuclear_norm': float(sv.sum()),
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def setup_dark_fig(nrows, ncols, figsize, title=""):
    """Create a dark-themed figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white', y=0.98)
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            style_axis(ax)
    else:
        style_axis(axes)
    return fig, axes


def style_axis(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#a0a0c0', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1a1a3e')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('#a0a0c0')
    ax.yaxis.label.set_color('#a0a0c0')


def viz_weight_canvas(W, title="Weight Matrix", save_path=None, max_display=1024):
    """
    Visualize a weight matrix as a raw canvas / heatmap.
    """
    out_dim, in_dim = W.shape

    # Subsample for display if too large
    if out_dim > max_display:
        idx_o = np.linspace(0, out_dim - 1, max_display, dtype=int)
        W_disp = W[idx_o]
    else:
        W_disp = W
    if W_disp.shape[1] > max_display:
        idx_i = np.linspace(0, W_disp.shape[1] - 1, max_display, dtype=int)
        W_disp = W_disp[:, idx_i]

    fig, axes = setup_dark_fig(1, 3, (18, 5), title=f"{title} ({out_dim}×{in_dim})")

    # Raw weights
    vmax = np.percentile(np.abs(W_disp), 99)
    im0 = axes[0].imshow(W_disp, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0].set_title('Raw Weights', color='#e94560', fontsize=11)
    axes[0].set_xlabel('Input Neuron')
    axes[0].set_ylabel('Output Neuron')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Absolute weights
    im1 = axes[1].imshow(np.abs(W_disp), aspect='auto', cmap='inferno', interpolation='nearest')
    axes[1].set_title('|Weights| (Magnitude)', color='#f5a623', fontsize=11)
    axes[1].set_xlabel('Input Neuron')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Weight histogram (from full matrix)
    axes[2].hist(W.flatten(), bins=200, color='#53a8b6', alpha=0.8, density=True, log=True)
    axes[2].axvline(0, color='#e94560', linewidth=1, linestyle='--')
    axes[2].set_title('Weight Distribution', color='#53a8b6', fontsize=11)
    axes[2].set_xlabel('Weight Value')
    axes[2].set_ylabel('Density (log)')

    # Add stats text
    stats_text = (f"mean={W.mean():.4f}  std={W.std():.4f}\n"
                  f"min={W.min():.4f}  max={W.max():.4f}\n"
                  f"sparsity={np.mean(np.abs(W) < 0.01):.2%}")
    axes[2].text(0.95, 0.95, stats_text, transform=axes[2].transAxes,
                 fontsize=8, color='#a0a0c0', ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_similarity_graph(similarity_G, cos_sim, layer_idx, weight_name, save_path=None):
    """
    Visualize the neuron similarity graph with multiple views.
    """
    fig, axes = setup_dark_fig(2, 2, (14, 12),
                                title=f"Layer {layer_idx}: {weight_name} — Neuron Similarity Graph")

    n_nodes = similarity_G.number_of_nodes()
    if n_nodes < 2:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Too few neurons", transform=ax.transAxes,
                    ha='center', color='#666', fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return fig

    # ---- Panel 1: Graph layout ----
    ax = axes[0, 0]
    try:
        pos = nx.spring_layout(similarity_G, k=2.0/np.sqrt(n_nodes), iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(similarity_G, seed=42)

    # Color by degree centrality
    dc = nx.degree_centrality(similarity_G)
    node_colors = [dc.get(n, 0) for n in similarity_G.nodes()]

    # Edge weights for width
    edges = similarity_G.edges(data=True)
    edge_weights = [abs(d.get('weight', 0.1)) for _, _, d in edges]
    max_ew = max(edge_weights) if edge_weights else 1.0

    nx.draw_networkx_edges(similarity_G, pos, ax=ax,
                           edge_color='#1a3a5c', alpha=0.3,
                           width=[0.5 + 2.0 * w / max_ew for w in edge_weights])
    nodes = nx.draw_networkx_nodes(similarity_G, pos, ax=ax,
                                   node_color=node_colors, cmap=plt.cm.plasma,
                                   node_size=30, alpha=0.8)
    ax.set_title('Spring Layout (color=degree centrality)', color='#e94560', fontsize=10)
    if nodes:
        fig.colorbar(nodes, ax=ax, shrink=0.6, label='Degree Centrality')

    # ---- Panel 2: Cosine similarity heatmap ----
    ax = axes[0, 1]
    n_show = min(cos_sim.shape[0], 128)
    im = ax.imshow(cos_sim[:n_show, :n_show], aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('Cosine Similarity Matrix', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')
    fig.colorbar(im, ax=ax, shrink=0.6)

    # ---- Panel 3: Degree distribution ----
    ax = axes[1, 0]
    degrees = [d for _, d in similarity_G.degree()]
    ax.hist(degrees, bins=min(50, max(degrees) - min(degrees) + 1) if degrees else 10,
            color='#7b68ee', alpha=0.8, edgecolor='#0d1117')
    ax.set_title('Degree Distribution', color='#7b68ee', fontsize=10)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')

    # Log-log degree distribution (scale-free check)
    if len(set(degrees)) > 3:
        from collections import Counter
        deg_count = Counter(degrees)
        degs = sorted(deg_count.keys())
        counts = [deg_count[d] for d in degs]
        ax_inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        ax_inset.set_facecolor('#1a1a3e')
        ax_inset.loglog(degs, counts, 'o', color='#f5a623', markersize=3, alpha=0.7)
        ax_inset.set_title('log-log', fontsize=7, color='#a0a0c0')
        ax_inset.tick_params(labelsize=6, colors='#a0a0c0')

    # ---- Panel 4: Clustering coefficient distribution ----
    ax = axes[1, 1]
    try:
        clustering = nx.clustering(similarity_G, weight='weight')
        cc_vals = list(clustering.values())
        ax.hist(cc_vals, bins=50, color='#2ecc71', alpha=0.8, edgecolor='#0d1117')
        ax.set_title(f'Clustering Coeff (avg={np.mean(cc_vals):.4f})', color='#2ecc71', fontsize=10)
        ax.set_xlabel('Clustering Coefficient')
        ax.set_ylabel('Count')
    except Exception:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', color='#666')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_spectral_analysis(spectral_data, layer_idx, weight_name, save_path=None):
    """
    Visualize spectral properties of a weight matrix.
    """
    fig, axes = setup_dark_fig(2, 2, (14, 10),
                                title=f"Layer {layer_idx}: {weight_name} — Spectral Analysis")

    sv = spectral_data['singular_values']
    gram_eigs = spectral_data['gram_eigenvalues']
    cos_eigs = spectral_data['cosine_eigenvalues']

    # ---- Panel 1: Singular value spectrum ----
    ax = axes[0, 0]
    ax.semilogy(sv, color='#e94560', linewidth=1.5, alpha=0.9)
    ax.fill_between(range(len(sv)), sv, alpha=0.15, color='#e94560')
    ax.set_title(f'Singular Values (eff. rank={spectral_data["effective_rank"]:.1f})',
                 color='#e94560', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('σ (log scale)')

    # Mark effective rank
    eff_rank = int(spectral_data['effective_rank'])
    if eff_rank < len(sv):
        ax.axvline(eff_rank, color='#f5a623', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'Eff. rank ≈ {eff_rank}')
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    # Stats box
    stats_text = (f"σ₁={sv[0]:.3f}  σ_min={sv[-1]:.2e}\n"
                  f"cond={spectral_data['condition_number']:.1f}\n"
                  f"stable rank={spectral_data['stable_rank']:.1f}\n"
                  f"‖W‖_F={spectral_data['frobenius_norm']:.3f}\n"
                  f"‖W‖_*={spectral_data['nuclear_norm']:.3f}")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, color='#a0a0c0', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    # ---- Panel 2: Singular value distribution (histogram) ----
    ax = axes[0, 1]
    ax.hist(sv, bins=min(100, len(sv) // 2 + 1), color='#53a8b6', alpha=0.8,
            edgecolor='#0d1117', density=True)
    ax.set_title('Singular Value Distribution', color='#53a8b6', fontsize=10)
    ax.set_xlabel('σ')
    ax.set_ylabel('Density')

    # Marchenko-Pastur reference (if applicable)
    n, m = spectral_data.get('shape', (len(sv), len(sv)))
    if isinstance(n, (int, float)) and isinstance(m, (int, float)):
        gamma = min(n, m) / max(n, m) if max(n, m) > 0 else 1.0
        mp_upper = (1 + np.sqrt(gamma)) ** 2
        mp_lower = (1 - np.sqrt(gamma)) ** 2
        ax.axvline(mp_upper * np.median(sv), color='#f5a623', linestyle=':', linewidth=1,
                   alpha=0.5, label=f'MP upper (γ={gamma:.2f})')
        ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    # ---- Panel 3: Gram matrix eigenvalues ----
    ax = axes[1, 0]
    n_show = min(200, len(gram_eigs))
    ax.semilogy(gram_eigs[:n_show], color='#7b68ee', linewidth=1.5)
    ax.fill_between(range(n_show), gram_eigs[:n_show], alpha=0.1, color='#7b68ee')
    ax.set_title('Gram Matrix Eigenvalues (W^T W)', color='#7b68ee', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('λ (log scale)')

    # ---- Panel 4: Cosine similarity eigenvalues ----
    ax = axes[1, 1]
    ax.plot(cos_eigs, color='#2ecc71', linewidth=1.5)
    ax.fill_between(range(len(cos_eigs)), cos_eigs, alpha=0.1, color='#2ecc71')
    ax.axhline(0, color='#555', linewidth=0.5)
    ax.set_title('Cosine Similarity Spectrum', color='#2ecc71', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('λ')

    # Count positive/negative eigenvalues
    n_pos = np.sum(cos_eigs > 0.01)
    n_neg = np.sum(cos_eigs < -0.01)
    ax.text(0.95, 0.95, f'+{n_pos} / −{n_neg} eigenvalues',
            transform=ax.transAxes, fontsize=8, color='#a0a0c0', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_bipartite_graph(bipartite_G, layer_idx, weight_name, save_path=None):
    """
    Visualize the bipartite input->output neuron graph.
    """
    fig, axes = setup_dark_fig(1, 2, (16, 6),
                                title=f"Layer {layer_idx}: {weight_name} — Bipartite Weight Graph")

    if bipartite_G.number_of_nodes() < 2:
        for ax in axes:
            ax.text(0.5, 0.5, "Too few neurons", transform=ax.transAxes,
                    ha='center', color='#666', fontsize=14)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return fig

    # Separate input and output nodes
    input_nodes = [n for n, d in bipartite_G.nodes(data=True) if d.get('neuron_type') == 'input']
    output_nodes = [n for n, d in bipartite_G.nodes(data=True) if d.get('neuron_type') == 'output']

    # ---- Panel 1: Bipartite layout ----
    ax = axes[0]
    pos = {}
    for i, n in enumerate(input_nodes):
        pos[n] = (0, i / max(len(input_nodes) - 1, 1))
    for i, n in enumerate(output_nodes):
        pos[n] = (1, i / max(len(output_nodes) - 1, 1))

    # Edge colors by sign of raw weight
    edge_colors = []
    edge_widths = []
    for u, v, d in bipartite_G.edges(data=True):
        raw = d.get('raw_weight', 0)
        w = d.get('weight', abs(raw))
        edge_colors.append('#e94560' if raw > 0 else '#0077b6')
        edge_widths.append(0.3 + 2.0 * min(w, 1.0))

    nx.draw_networkx_edges(bipartite_G, pos, ax=ax,
                           edge_color=edge_colors, alpha=0.3,
                           width=edge_widths)
    nx.draw_networkx_nodes(bipartite_G, pos, nodelist=input_nodes, ax=ax,
                           node_color='#53a8b6', node_size=15, alpha=0.7)
    nx.draw_networkx_nodes(bipartite_G, pos, nodelist=output_nodes, ax=ax,
                           node_color='#e94560', node_size=15, alpha=0.7)
    ax.set_title('Bipartite Graph (input→output)', color='#f5a623', fontsize=10)
    ax.text(0.0, -0.05, 'Input', transform=ax.transAxes, color='#53a8b6', fontsize=9, ha='center')
    ax.text(1.0, -0.05, 'Output', transform=ax.transAxes, color='#e94560', fontsize=9, ha='center')

    # ---- Panel 2: Edge weight distribution ----
    ax = axes[1]
    raw_weights = [d.get('raw_weight', 0) for _, _, d in bipartite_G.edges(data=True)]
    if raw_weights:
        ax.hist(raw_weights, bins=80, color='#7b68ee', alpha=0.8, edgecolor='#0d1117')
        ax.axvline(0, color='#e94560', linewidth=1, linestyle='--')
    ax.set_title('Edge Weight Distribution', color='#7b68ee', fontsize=10)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_weight_norms_landscape(layers_data, model_name, save_path=None):
    """
    Visualize weight norms across all layers as a landscape.
    """
    n_layers = len(layers_data)
    weight_types = set()
    for ld in layers_data:
        for k in ld:
            if k != 'layer_idx' and isinstance(ld[k], np.ndarray) and ld[k].ndim == 2:
                weight_types.add(k)
    weight_types = sorted(weight_types)

    if not weight_types:
        return None

    n_types = len(weight_types)
    fig, axes = setup_dark_fig(1, 1, (max(14, n_layers * 0.8), 6),
                                title=f"{model_name} — Weight Norm Landscape")
    ax = axes

    x = np.arange(n_layers)
    bar_width = 0.8 / max(n_types, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_types))

    for ti, wtype in enumerate(weight_types):
        norms = []
        for ld in layers_data:
            if wtype in ld:
                norms.append(np.linalg.norm(ld[wtype], 'fro'))
            else:
                norms.append(0)
        offset = (ti - n_types / 2 + 0.5) * bar_width
        ax.bar(x + offset, norms, bar_width, color=colors[ti], alpha=0.8,
               label=wtype, edgecolor='#0d1117')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Frobenius Norm', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
    ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e',
              labelcolor='#a0a0c0', loc='upper left')
    ax.set_title('Weight Frobenius Norms per Layer', color='#e94560', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_cross_layer_sv_waterfall(layers_data, weight_key, model_name, save_path=None, max_sv=64):
    """
    Waterfall plot of singular values across layers for a specific weight type.
    """
    sv_per_layer = []
    for ld in layers_data:
        if weight_key in ld:
            W = ld[weight_key]
            sv = np.linalg.svd(W, compute_uv=False)
            sv_per_layer.append(sv[:max_sv])
        else:
            sv_per_layer.append(np.zeros(max_sv))

    if not sv_per_layer:
        return None

    n_layers = len(sv_per_layer)
    max_len = max(len(s) for s in sv_per_layer)
    sv_matrix = np.zeros((n_layers, max_len))
    for i, sv in enumerate(sv_per_layer):
        sv_matrix[i, :len(sv)] = sv

    fig, axes = setup_dark_fig(1, 2, (16, 6),
                                title=f"{model_name} — {weight_key} Singular Value Waterfall")

    # ---- Panel 1: Heatmap ----
    ax = axes[0]
    im = ax.imshow(np.log10(sv_matrix + 1e-12), aspect='auto', cmap='inferno',
                   interpolation='nearest')
    ax.set_title('log₁₀(σ) across layers', color='#e94560', fontsize=10)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Layer')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 2: Overlaid curves ----
    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0, 1, n_layers))
    for i in range(n_layers):
        ax.semilogy(sv_per_layer[i], color=cmap[i], alpha=0.6, linewidth=1)
    ax.set_title('SV Spectra (color=layer depth)', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('σ (log)')

    # Add colorbar for layer
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, n_layers - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8, label='Layer')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_community_detection(similarity_G, cos_sim, layer_idx, weight_name,
                            n_communities=6, save_path=None):
    """
    Detect and visualize communities in the neuron similarity graph.
    """
    n_nodes = similarity_G.number_of_nodes()
    if n_nodes < 4:
        return None

    fig, axes = setup_dark_fig(1, 2, (14, 6),
                                title=f"Layer {layer_idx}: {weight_name} — Community Structure")

    # ---- Community detection via spectral clustering on cosine similarity ----
    n_show = min(cos_sim.shape[0], 256)
    sim_sub = cos_sim[:n_show, :n_show]

    # Make it a valid affinity matrix (shift to non-negative)
    affinity = (sim_sub + 1) / 2  # map [-1,1] -> [0,1]
    np.fill_diagonal(affinity, 1.0)

    n_clust = min(n_communities, n_show - 1)
    try:
        sc = SpectralClustering(n_clusters=n_clust, affinity='precomputed',
                                random_state=42, n_init=10)
        labels = sc.fit_predict(affinity)
    except Exception:
        labels = np.zeros(n_show, dtype=int)

    # ---- Panel 1: Reordered similarity matrix by community ----
    ax = axes[0]
    order = np.argsort(labels)
    reordered = sim_sub[np.ix_(order, order)]
    im = ax.imshow(reordered, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_title(f'Similarity (reordered by {n_clust} communities)', color='#e94560', fontsize=10)
    ax.set_xlabel('Neuron (reordered)')
    ax.set_ylabel('Neuron (reordered)')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Draw community boundaries
    boundaries = []
    for c in range(n_clust):
        mask = labels[order] == c
        indices = np.where(mask)[0]
        if len(indices) > 0:
            boundaries.append(indices[-1] + 0.5)
    for b in boundaries[:-1]:
        ax.axhline(b, color='#f5a623', linewidth=1, alpha=0.7)
        ax.axvline(b, color='#f5a623', linewidth=1, alpha=0.7)

    # ---- Panel 2: Community graph layout ----
    ax = axes[1]
    nodes_list = list(similarity_G.nodes())[:n_show]
    sub_G = similarity_G.subgraph(nodes_list).copy()

    try:
        pos = nx.spring_layout(sub_G, k=2.0 / np.sqrt(n_show), iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(sub_G, seed=42)

    # Map labels to nodes
    node_to_label = {}
    for i, n in enumerate(nodes_list):
        if i < len(labels):
            node_to_label[n] = labels[i]
        else:
            node_to_label[n] = 0

    node_colors = [node_to_label.get(n, 0) for n in sub_G.nodes()]

    nx.draw_networkx_edges(sub_G, pos, ax=ax, edge_color='#1a3a5c', alpha=0.15, width=0.5)
    nodes_drawn = nx.draw_networkx_nodes(sub_G, pos, ax=ax,
                                          node_color=node_colors, cmap=plt.cm.Set2,
                                          node_size=20, alpha=0.8)
    ax.set_title(f'{n_clust} Communities (Spectral Clustering)', color='#53a8b6', fontsize=10)

    # Community size legend
    from collections import Counter
    comm_counts = Counter(labels)
    legend_text = "  ".join([f"C{c}:{cnt}" for c, cnt in sorted(comm_counts.items())])
    ax.text(0.5, -0.05, legend_text, transform=ax.transAxes,
            fontsize=8, color='#a0a0c0', ha='center',
            bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_attention_head_graph(W_qkv, n_heads, layer_idx, save_path=None):
    """
    Visualize attention heads as a graph: each head is a node,
    edges represent similarity between heads' weight patterns.
    """
    if W_qkv is None:
        return None

    # GPT-2 c_attn: (hidden_dim, 3*hidden_dim) — columns are [Q, K, V] concatenated
    # Or it could be (3*hidden_dim, hidden_dim) depending on convention
    total_dim = W_qkv.shape[0] * W_qkv.shape[1]
    hidden_dim = min(W_qkv.shape)

    # Try to split into Q, K, V
    if W_qkv.shape[1] == 3 * W_qkv.shape[0]:
        # (hidden_dim, 3*hidden_dim)
        d = W_qkv.shape[0]
        Q = W_qkv[:, :d]
        K = W_qkv[:, d:2*d]
        V = W_qkv[:, 2*d:]
    elif W_qkv.shape[0] == 3 * W_qkv.shape[1]:
        # (3*hidden_dim, hidden_dim)
        d = W_qkv.shape[1]
        Q = W_qkv[:d, :]
        K = W_qkv[d:2*d, :]
        V = W_qkv[2*d:, :]
    else:
        return None

    head_dim = d // n_heads

    fig, axes = setup_dark_fig(2, 2, (14, 12),
                                title=f"Layer {layer_idx}: Attention Head Analysis ({n_heads} heads)")

    # ---- Extract per-head weight matrices ----
    head_Q = [Q[:, h*head_dim:(h+1)*head_dim] for h in range(n_heads)]
    head_K = [K[:, h*head_dim:(h+1)*head_dim] for h in range(n_heads)]
    head_V = [V[:, h*head_dim:(h+1)*head_dim] for h in range(n_heads)]

    # ---- Panel 1: Head similarity graph (based on Q weights) ----
    ax = axes[0, 0]
    head_vecs_Q = np.stack([hq.flatten() for hq in head_Q])
    norms = np.linalg.norm(head_vecs_Q, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    head_vecs_Q_normed = head_vecs_Q / norms
    sim_Q = head_vecs_Q_normed @ head_vecs_Q_normed.T

    G_heads = nx.Graph()
    for h in range(n_heads):
        G_heads.add_node(h)
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            if abs(sim_Q[i, j]) > 0.1:
                G_heads.add_edge(i, j, weight=float(sim_Q[i, j]))

    pos = nx.spring_layout(G_heads, seed=42)
    edge_weights = [abs(d['weight']) for _, _, d in G_heads.edges(data=True)]
    max_ew = max(edge_weights) if edge_weights else 1.0

    edge_colors_list = ['#e94560' if d['weight'] > 0 else '#0077b6'
                        for _, _, d in G_heads.edges(data=True)]

    nx.draw_networkx_edges(G_heads, pos, ax=ax, edge_color=edge_colors_list,
                           alpha=0.5, width=[1 + 3 * w / max_ew for w in edge_weights])
    nx.draw_networkx_nodes(G_heads, pos, ax=ax, node_color=range(n_heads),
                           cmap=plt.cm.Set3, node_size=200, alpha=0.9)
    nx.draw_networkx_labels(G_heads, pos, ax=ax,
                            labels={h: f'H{h}' for h in range(n_heads)},
                            font_size=7, font_color='white')
    ax.set_title('Head Similarity (Q weights)', color='#e94560', fontsize=10)

    # ---- Panel 2: Head similarity heatmap ----
    ax = axes[0, 1]
    im = ax.imshow(sim_Q, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('Q-Weight Cosine Similarity', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Head')
    ax.set_ylabel('Head')
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_heads))
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 3: Per-head weight norms ----
    ax = axes[1, 0]
    q_norms = [np.linalg.norm(hq, 'fro') for hq in head_Q]
    k_norms = [np.linalg.norm(hk, 'fro') for hk in head_K]
    v_norms = [np.linalg.norm(hv, 'fro') for hv in head_V]

    x = np.arange(n_heads)
    w = 0.25
    ax.bar(x - w, q_norms, w, color='#e94560', alpha=0.8, label='Q')
    ax.bar(x, k_norms, w, color='#53a8b6', alpha=0.8, label='K')
    ax.bar(x + w, v_norms, w, color='#2ecc71', alpha=0.8, label='V')
    ax.set_title('Per-Head Weight Norms', color='#f5a623', fontsize=10)
    ax.set_xlabel('Head')
    ax.set_ylabel('‖W‖_F')
    ax.set_xticks(x)
    ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    # ---- Panel 4: QK^T pattern (attention pattern proxy) ----
    ax = axes[1, 1]
    # For each head, compute Q @ K^T pattern (averaged over input neurons)
    qk_patterns = []
    for h in range(n_heads):
        qk = head_Q[h].T @ head_K[h]  # (head_dim, head_dim)
        qk_patterns.append(qk)

    # Show the first head's QK pattern as example
    if qk_patterns:
        im = ax.imshow(qk_patterns[0], aspect='auto', cmap='RdBu_r',
                       interpolation='nearest')
        ax.set_title(f'Head 0: Q^T K pattern ({head_dim}×{head_dim})', color='#7b68ee', fontsize=10)
        ax.set_xlabel('K dimension')
        ax.set_ylabel('Q dimension')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_mlp_flow_graph(W_up, W_down, layer_idx, max_neurons=128, save_path=None):
    """
    Visualize the MLP as a flow graph: input -> hidden -> output.
    """
    if W_up is None or W_down is None:
        return None

    fig, axes = setup_dark_fig(1, 3, (18, 6),
                                title=f"Layer {layer_idx}: MLP Flow Graph")

    # W_up: (4*hidden, hidden) or (hidden, 4*hidden) — up-projection
    # W_down: (hidden, 4*hidden) or (4*hidden, hidden) — down-projection

    # ---- Panel 1: Up-projection as heatmap ----
    ax = axes[0]
    n_show = min(max_neurons, W_up.shape[0], W_up.shape[1])
    W_up_sub = W_up[:n_show, :n_show]
    vmax = np.percentile(np.abs(W_up_sub), 99)
    im = ax.imshow(W_up_sub, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_title('MLP Up-Projection', color='#e94560', fontsize=10)
    ax.set_xlabel('Input')
    ax.set_ylabel('Hidden')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 2: Down-projection as heatmap ----
    ax = axes[1]
    W_down_sub = W_down[:n_show, :n_show]
    vmax = np.percentile(np.abs(W_down_sub), 99)
    im = ax.imshow(W_down_sub, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_title('MLP Down-Projection', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Hidden')
    ax.set_ylabel('Output')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 3: Composed MLP (W_down @ W_up) bottleneck analysis ----
    ax = axes[2]
    # Compute the effective MLP transformation
    # If shapes are compatible, compute W_down @ W_up
    try:
        if W_down.shape[1] == W_up.shape[0]:
            W_composed = W_down @ W_up
        elif W_up.shape[1] == W_down.shape[0]:
            W_composed = W_up @ W_down
        else:
            # Subsample to make compatible
            min_dim = min(W_up.shape[0], W_down.shape[1])
            W_composed = W_down[:min_dim, :min_dim] @ W_up[:min_dim, :min_dim]
        n_show = min(128, W_composed.shape[0], W_composed.shape[1])
        W_comp_sub = W_composed[:n_show, :n_show]

        # SVD of composed MLP
        sv_composed = np.linalg.svd(W_composed, compute_uv=False)

        vmax = np.percentile(np.abs(W_comp_sub), 99)
        im = ax.imshow(W_comp_sub, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_title(f'Composed MLP (eff. rank={len(sv_composed[sv_composed > sv_composed[0]*0.01])})',
                     color='#2ecc71', fontsize=10)
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        fig.colorbar(im, ax=ax, shrink=0.8)

    except Exception as e:
        ax.text(0.5, 0.5, f"Composition failed:\n{e}", transform=ax.transAxes,
                ha='center', color='#666', fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_weight_pca_embedding(layers_data, weight_key, model_name, save_path=None):
    """
    PCA embedding of weight vectors across layers — each layer's weight matrix
    is flattened and projected into 2D to see how weights evolve.
    """
    vecs = []
    labels = []
    for ld in layers_data:
        if weight_key in ld:
            W = ld[weight_key]
            # Flatten and subsample for tractability
            flat = W.flatten()
            if len(flat) > 10000:
                idx = np.linspace(0, len(flat) - 1, 10000, dtype=int)
                flat = flat[idx]
            vecs.append(flat)
            labels.append(f"L{ld['layer_idx']}")

    if len(vecs) < 2:
        return None

    # Pad to same length
    max_len = max(len(v) for v in vecs)
    vecs_padded = np.zeros((len(vecs), max_len))
    for i, v in enumerate(vecs):
        vecs_padded[i, :len(v)] = v

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs_padded)

    fig, ax = setup_dark_fig(1, 1, (8, 8),
                              title=f"{model_name} — {weight_key} Weight PCA Trajectory")
    ax = ax if not isinstance(ax, np.ndarray) else ax

    # Color by layer depth
    colors = plt.cm.viridis(np.linspace(0, 1, len(vecs)))

    # Draw trajectory
    ax.plot(coords[:, 0], coords[:, 1], '-', color='#333', linewidth=1, alpha=0.5)
    for i in range(len(vecs)):
        ax.scatter(coords[i, 0], coords[i, 1], c=[colors[i]], s=80, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                    fontsize=8, color='#a0a0c0', textcoords="offset points", xytext=(5, 5))

    # Draw arrows between consecutive layers
    for i in range(len(vecs) - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        ax.annotate('', xy=(coords[i + 1, 0], coords[i + 1, 1]),
                    xytext=(coords[i, 0], coords[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_title(f'{weight_key} Weight Evolution Through Layers', color='#e94560', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_graph_metrics_dashboard(all_stats, model_name, save_path=None):
    """
    Dashboard of graph metrics across all layers and weight types.
    """
    # Organize stats by weight type
    by_type = {}
    for layer_idx, weight_name, stats in all_stats:
        if weight_name not in by_type:
            by_type[weight_name] = {'layers': [], 'clustering': [], 'density': [],
                                     'avg_degree': [], 'max_centrality': []}
        by_type[weight_name]['layers'].append(layer_idx)
        by_type[weight_name]['clustering'].append(stats.get('avg_clustering', 0))
        by_type[weight_name]['density'].append(stats.get('density', 0))
        by_type[weight_name]['avg_degree'].append(stats.get('avg_degree', 0))
        by_type[weight_name]['max_centrality'].append(stats.get('max_centrality', 0))

    n_types = len(by_type)
    if n_types == 0:
        return None

    fig, axes = setup_dark_fig(2, 2, (14, 10),
                                title=f"{model_name} — Graph Metrics Across Layers")

    metrics = [
        ('clustering', 'Avg Clustering Coefficient', '#e94560'),
        ('density', 'Graph Density', '#53a8b6'),
        ('avg_degree', 'Average Degree', '#7b68ee'),
        ('max_centrality', 'Max Degree Centrality', '#2ecc71'),
    ]

    for idx, (metric_key, metric_title, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for wtype, data in by_type.items():
            ax.plot(data['layers'], data[metric_key], 'o-', label=wtype,
                    alpha=0.8, markersize=4, linewidth=1.5)
        ax.set_title(metric_title, color=color, fontsize=10)
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric_title)
        ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#1a1a3e',
                  labelcolor='#a0a0c0', loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_weight_as_image(W, title="Weight Image", save_path=None, max_display=512):
    """
    Treat the weight matrix as a raw image — each value is a pixel.
    Show it in multiple color mappings for artistic/exploratory purposes.
    """
    out_dim, in_dim = W.shape

    # Subsample
    if out_dim > max_display:
        idx_o = np.linspace(0, out_dim - 1, max_display, dtype=int)
        W_disp = W[idx_o]
    else:
        W_disp = W
    if W_disp.shape[1] > max_display:
        idx_i = np.linspace(0, W_disp.shape[1] - 1, max_display, dtype=int)
        W_disp = W_disp[:, idx_i]

    fig, axes = setup_dark_fig(2, 3, (18, 10), title=f"{title} as Image ({out_dim}×{in_dim})")

    cmaps = [
        ('viridis', 'Viridis', '#2ecc71'),
        ('plasma', 'Plasma', '#e94560'),
        ('twilight_shifted', 'Twilight (cyclic)', '#7b68ee'),
        ('cubehelix', 'Cubehelix', '#f5a623'),
        ('terrain', 'Terrain', '#53a8b6'),
        ('hsv', 'HSV (phase)', '#ff6b9d'),
    ]

    for idx, (cmap_name, cmap_title, color) in enumerate(cmaps):
        ax = axes[idx // 3, idx % 3]
        # Normalize to [0, 1] for consistent display
        W_norm = (W_disp - W_disp.min()) / max(W_disp.max() - W_disp.min(), 1e-12)
        im = ax.imshow(W_norm, aspect='auto', cmap=cmap_name, interpolation='nearest')
        ax.set_title(cmap_title, color=color, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_row_column_norms(W, title="Row/Column Norms", save_path=None):
    """
    Visualize the L2 norms of rows and columns of a weight matrix.
    Rows = output neurons, Columns = input neurons.
    """
    row_norms = np.linalg.norm(W, axis=1)
    col_norms = np.linalg.norm(W, axis=0)

    fig, axes = setup_dark_fig(2, 2, (14, 10), title=title)

    # Row norms
    ax = axes[0, 0]
    ax.bar(range(len(row_norms)), row_norms, color='#e94560', alpha=0.7, width=1.0)
    ax.set_title(f'Row Norms (output neurons, n={len(row_norms)})', color='#e94560', fontsize=10)
    ax.set_xlabel('Output Neuron')
    ax.set_ylabel('‖w‖₂')

    # Column norms
    ax = axes[0, 1]
    ax.bar(range(len(col_norms)), col_norms, color='#53a8b6', alpha=0.7, width=1.0)
    ax.set_title(f'Column Norms (input neurons, n={len(col_norms)})', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Input Neuron')
    ax.set_ylabel('‖w‖₂')

    # Row norm distribution
    ax = axes[1, 0]
    ax.hist(row_norms, bins=50, color='#e94560', alpha=0.8, edgecolor='#0d1117')
    ax.set_title('Row Norm Distribution', color='#e94560', fontsize=10)
    ax.set_xlabel('‖w‖₂')
    ax.set_ylabel('Count')

    # Column norm distribution
    ax = axes[1, 1]
    ax.hist(col_norms, bins=50, color='#53a8b6', alpha=0.8, edgecolor='#0d1117')
    ax.set_title('Column Norm Distribution', color='#53a8b6', fontsize=10)
    ax.set_xlabel('‖w‖₂')
    ax.set_ylabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_weight_correlation_structure(W, title="Correlation Structure", save_path=None,
                                      max_neurons=256):
    """
    Visualize the correlation structure between neurons:
    - Row-row correlation (output neuron similarity)
    - Column-column correlation (input neuron similarity)
    - Eigenvalue spectrum of the correlation matrix
    """
    out_dim, in_dim = W.shape

    # Subsample
    n_out = min(max_neurons, out_dim)
    n_in = min(max_neurons, in_dim)
    idx_out = np.linspace(0, out_dim - 1, n_out, dtype=int)
    idx_in = np.linspace(0, in_dim - 1, n_in, dtype=int)

    W_out = W[idx_out]  # (n_out, in_dim)
    W_in = W[:, idx_in]  # (out_dim, n_in)

    # Row correlation (output neurons)
    norms_out = np.linalg.norm(W_out, axis=1, keepdims=True)
    norms_out = np.clip(norms_out, 1e-12, None)
    W_out_normed = W_out / norms_out
    corr_out = W_out_normed @ W_out_normed.T

    # Column correlation (input neurons)
    norms_in = np.linalg.norm(W_in, axis=0, keepdims=True)
    norms_in = np.clip(norms_in, 1e-12, None)
    W_in_normed = W_in / norms_in
    corr_in = W_in_normed.T @ W_in_normed

    fig, axes = setup_dark_fig(2, 2, (14, 12), title=title)

    # Output neuron correlation
    ax = axes[0, 0]
    im = ax.imshow(corr_out, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_title(f'Output Neuron Correlation ({n_out}×{n_out})', color='#e94560', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Input neuron correlation
    ax = axes[0, 1]
    im = ax.imshow(corr_in, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_title(f'Input Neuron Correlation ({n_in}×{n_in})', color='#53a8b6', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Eigenvalue spectrum of output correlation
    ax = axes[1, 0]
    eigs_out = np.linalg.eigvalsh(corr_out)[::-1]
    ax.plot(eigs_out, color='#e94560', linewidth=1.5)
    ax.fill_between(range(len(eigs_out)), eigs_out, alpha=0.15, color='#e94560')
    ax.axhline(0, color='#555', linewidth=0.5)
    ax.set_title('Output Correlation Eigenvalues', color='#e94560', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('λ')

    # Eigenvalue spectrum of input correlation
    ax = axes[1, 1]
    eigs_in = np.linalg.eigvalsh(corr_in)[::-1]
    ax.plot(eigs_in, color='#53a8b6', linewidth=1.5)
    ax.fill_between(range(len(eigs_in)), eigs_in, alpha=0.15, color='#53a8b6')
    ax.axhline(0, color='#555', linewidth=0.5)
    ax.set_title('Input Correlation Eigenvalues', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('λ')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_small_world_analysis(similarity_G, layer_idx, weight_name, save_path=None):
    """
    Analyze small-world and scale-free properties of the neuron similarity graph.
    """
    n_nodes = similarity_G.number_of_nodes()
    n_edges = similarity_G.number_of_edges()

    if n_nodes < 10 or n_edges < 5:
        return None

    fig, axes = setup_dark_fig(2, 2, (14, 10),
                                title=f"Layer {layer_idx}: {weight_name} — Small-World / Scale-Free Analysis")

    # ---- Panel 1: Degree distribution with power-law fit ----
    ax = axes[0, 0]
    degrees = [d for _, d in similarity_G.degree()]
    from collections import Counter
    deg_count = Counter(degrees)
    degs = sorted(deg_count.keys())
    counts = [deg_count[d] for d in degs]

    ax.loglog(degs, counts, 'o', color='#e94560', markersize=5, alpha=0.8)

    # Attempt power-law fit
    if len(degs) > 3 and min(degs) > 0:
        log_degs = np.log(degs)
        log_counts = np.log(counts)
        try:
            coeffs = np.polyfit(log_degs, log_counts, 1)
            gamma = -coeffs[0]
            fit_x = np.linspace(min(log_degs), max(log_degs), 100)
            fit_y = np.exp(coeffs[1]) * np.exp(fit_x) ** coeffs[0]
            ax.loglog(np.exp(fit_x), fit_y, '--', color='#f5a623', linewidth=1.5,
                      label=f'γ ≈ {gamma:.2f}')
            ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')
        except Exception:
            pass

    ax.set_title('Degree Distribution (log-log)', color='#e94560', fontsize=10)
    ax.set_xlabel('Degree k')
    ax.set_ylabel('P(k)')

    # Scale-free indicator
    is_scale_free = "Possibly" if len(degs) > 5 else "Insufficient data"
    ax.text(0.95, 0.95, f'Scale-free: {is_scale_free}',
            transform=ax.transAxes, fontsize=8, color='#a0a0c0', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    # ---- Panel 2: Clustering vs Degree ----
    ax = axes[0, 1]
    try:
        clustering = nx.clustering(similarity_G, weight='weight')
        node_degrees = dict(similarity_G.degree())
        degs_list = []
        cc_list = []
        for n in similarity_G.nodes():
            degs_list.append(node_degrees[n])
            cc_list.append(clustering[n])
        ax.scatter(degs_list, cc_list, c='#53a8b6', s=15, alpha=0.5)
        ax.set_title('Clustering vs Degree', color='#53a8b6', fontsize=10)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Clustering Coefficient')
    except Exception:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', color='#666')

    # ---- Panel 3: Small-world coefficient estimation ----
    ax = axes[1, 0]
    try:
        # Compare with random graph of same size and density
        avg_cc = nx.average_clustering(similarity_G, weight='weight')
        density = nx.density(similarity_G)

        # Generate equivalent random graph
        n_random_trials = 5
        random_cc_vals = []
        random_pl_vals = []
        for _ in range(n_random_trials):
            G_rand = nx.gnm_random_graph(n_nodes, n_edges, seed=np.random.randint(10000))
            if G_rand.number_of_edges() > 0:
                random_cc_vals.append(nx.average_clustering(G_rand))
                if nx.is_connected(G_rand):
                    random_pl_vals.append(nx.average_shortest_path_length(G_rand))

        avg_random_cc = np.mean(random_cc_vals) if random_cc_vals else 0.01
        avg_random_pl = np.mean(random_pl_vals) if random_pl_vals else 1.0

        # Small-world coefficient: σ = (C/C_rand) / (L/L_rand)
        # High σ (>> 1) indicates small-world
        if nx.is_connected(similarity_G):
            avg_pl = nx.average_shortest_path_length(similarity_G)
        else:
            largest_cc = max(nx.connected_components(similarity_G), key=len)
            sub = similarity_G.subgraph(largest_cc)
            avg_pl = nx.average_shortest_path_length(sub)

        gamma_sw = (avg_cc / max(avg_random_cc, 1e-6))
        lambda_sw = (avg_pl / max(avg_random_pl, 1e-6))
        sigma_sw = gamma_sw / max(lambda_sw, 1e-6)

        labels_sw = ['C_real', 'C_random', 'L_real', 'L_random', 'σ']
        values_sw = [avg_cc, avg_random_cc, avg_pl, avg_random_pl, sigma_sw]
        colors_sw = ['#e94560', '#666', '#53a8b6', '#666', '#f5a623']

        bars = ax.bar(labels_sw, values_sw, color=colors_sw, alpha=0.8, edgecolor='#0d1117')
        ax.set_title(f'Small-World Analysis (σ={sigma_sw:.2f})', color='#f5a623', fontsize=10)
        ax.set_ylabel('Value')

        # Annotate
        for bar, val in zip(bars, values_sw):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, color='#a0a0c0')

        sw_verdict = "YES" if sigma_sw > 1.5 else ("MAYBE" if sigma_sw > 1.0 else "NO")
        ax.text(0.95, 0.95, f'Small-world: {sw_verdict}',
                transform=ax.transAxes, fontsize=9, color='#f5a623', ha='right', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    except Exception as e:
        ax.text(0.5, 0.5, f"Analysis failed:\n{e}", transform=ax.transAxes,
                ha='center', color='#666', fontsize=9)

    # ---- Panel 4: Betweenness centrality distribution ----
    ax = axes[1, 1]
    try:
        bc = nx.betweenness_centrality(similarity_G, weight='weight')
        bc_vals = list(bc.values())
        ax.hist(bc_vals, bins=50, color='#7b68ee', alpha=0.8, edgecolor='#0d1117')
        ax.set_title(f'Betweenness Centrality (max={max(bc_vals):.4f})',
                     color='#7b68ee', fontsize=10)
        ax.set_xlabel('Betweenness Centrality')
        ax.set_ylabel('Count')

        # Mark top-5 most central nodes
        top_bc = sorted(bc.items(), key=lambda x: -x[1])[:5]
        top_text = "\n".join([f"{n}: {v:.4f}" for n, v in top_bc])
        ax.text(0.95, 0.95, f"Top 5:\n{top_text}",
                transform=ax.transAxes, fontsize=7, color='#a0a0c0', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))
    except Exception:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha='center', color='#666')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_cross_model_comparison(stats_model1, stats_model2, name1, name2, save_path=None):
    """
    Compare graph metrics between two models side by side.
    """
    # Find common weight types
    types1 = set(wn for _, wn, _ in stats_model1)
    types2 = set(wn for _, wn, _ in stats_model2)
    common_types = sorted(types1 & types2)

    if not common_types:
        return None

    n_metrics = 4
    fig, axes = setup_dark_fig(n_metrics, len(common_types),
                                (5 * len(common_types), 3 * n_metrics),
                                title=f"Cross-Model Comparison: {name1} vs {name2}")

    if len(common_types) == 1:
        axes = axes.reshape(-1, 1)

    metrics = ['avg_clustering', 'density', 'avg_degree', 'max_centrality']
    metric_labels = ['Avg Clustering', 'Density', 'Avg Degree', 'Max Centrality']
    metric_colors = ['#e94560', '#53a8b6', '#7b68ee', '#2ecc71']

    for ti, wtype in enumerate(common_types):
        # Extract data for this weight type
        data1 = [(li, s) for li, wn, s in stats_model1 if wn == wtype]
        data2 = [(li, s) for li, wn, s in stats_model2 if wn == wtype]

        layers1 = [d[0] for d in data1]
        layers2 = [d[0] for d in data2]

        for mi, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            ax = axes[mi, ti] if len(common_types) > 1 else axes[mi, 0]

            vals1 = [d[1].get(metric, 0) for d in data1]
            vals2 = [d[1].get(metric, 0) for d in data2]

            ax.plot(layers1, vals1, 'o-', color=color, alpha=0.8, label=name1, markersize=4)
            ax.plot(layers2, vals2, 's--', color=color, alpha=0.5, label=name2, markersize=4)

            if mi == 0:
                ax.set_title(wtype, color='white', fontsize=10, fontweight='bold')
            ax.set_ylabel(label, fontsize=8)
            if mi == n_metrics - 1:
                ax.set_xlabel('Layer')
            ax.legend(fontsize=6, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_layernorm_analysis(layers_data, model_name, save_path=None):
    """
    Visualize LayerNorm weights across layers — these are 1D vectors
    that scale each hidden dimension.
    """
    ln1_weights = []
    ln2_weights = []
    for ld in layers_data:
        if 'ln1' in ld:
            ln1_weights.append(ld['ln1'])
        if 'ln2' in ld:
            ln2_weights.append(ld['ln2'])

    if not ln1_weights and not ln2_weights:
        return None

    n_panels = (1 if ln1_weights else 0) + (1 if ln2_weights else 0)
    fig, axes = setup_dark_fig(2, n_panels, (7 * n_panels, 10),
                                title=f"{model_name} — LayerNorm Weight Analysis")
    if n_panels == 1:
        axes = axes.reshape(-1, 1)

    panel_idx = 0

    for ln_weights, ln_name, color in [
        (ln1_weights, 'LayerNorm 1 (pre-attn)', '#e94560'),
        (ln2_weights, 'LayerNorm 2 (pre-MLP)', '#53a8b6'),
    ]:
        if not ln_weights:
            continue

        # Stack into matrix: (n_layers, hidden_dim)
        ln_matrix = np.stack(ln_weights, axis=0)

        # Heatmap
        ax = axes[0, panel_idx]
        im = ax.imshow(ln_matrix, aspect='auto', cmap='RdBu_r',
                       vmin=ln_matrix.min(), vmax=ln_matrix.max(),
                       interpolation='nearest')
        ax.set_title(f'{ln_name} Weights', color=color, fontsize=10)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('Layer')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Overlay line plots per layer
        ax = axes[1, panel_idx]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(ln_weights)))
        for li in range(len(ln_weights)):
            ax.plot(ln_weights[li], color=cmap[li], alpha=0.5, linewidth=0.8)
        ax.set_title(f'{ln_name} Per-Layer Overlay', color=color, fontsize=10)
        ax.set_xlabel('Hidden Dimension')
        ax.set_ylabel('Weight Value')

        # Add colorbar for layer
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(ln_weights) - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.8, label='Layer')

        panel_idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_embedding_matrix(model, model_name, max_tokens=5000, save_path=None):
    """
    Visualize the token embedding matrix as an image and analyze its structure.
    """
    # Extract embedding matrix
    emb = None
    if hasattr(model, 'wte'):
        emb = model.wte.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
        emb = model.embeddings.word_embeddings.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'embed_tokens'):
        emb = model.embed_tokens.weight.detach().cpu().float().numpy()

    if emb is None:
        return None

    vocab_size, hidden_dim = emb.shape

    fig, axes = setup_dark_fig(2, 2, (14, 12),
                                title=f"{model_name} — Token Embedding Matrix ({vocab_size}×{hidden_dim})")

    # Subsample for display
    n_show = min(max_tokens, vocab_size)
    idx = np.linspace(0, vocab_size - 1, n_show, dtype=int)
    emb_sub = emb[idx]

    # ---- Panel 1: Embedding matrix heatmap ----
    ax = axes[0, 0]
    vmax = np.percentile(np.abs(emb_sub), 99)
    im = ax.imshow(emb_sub, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(f'Embedding Matrix ({n_show}×{hidden_dim})', color='#e94560', fontsize=10)
    ax.set_xlabel('Hidden Dimension')
    ax.set_ylabel('Token ID')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 2: Embedding norms ----
    ax = axes[0, 1]
    norms = np.linalg.norm(emb, axis=1)
    ax.plot(norms, color='#53a8b6', linewidth=0.5, alpha=0.7)
    ax.set_title(f'Token Embedding Norms (vocab={vocab_size})', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Token ID')
    ax.set_ylabel('‖e‖₂')

    # Highlight outliers
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    outliers = np.where(np.abs(norms - mean_norm) > 3 * std_norm)[0]
    if len(outliers) > 0:
        ax.scatter(outliers, norms[outliers], c='#e94560', s=10, zorder=5, label=f'{len(outliers)} outliers')
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    # ---- Panel 3: SVD of embedding matrix ----
    ax = axes[1, 0]
    sv = np.linalg.svd(emb, compute_uv=False)
    ax.semilogy(sv[:min(200, len(sv))], color='#7b68ee', linewidth=1.5)
    ax.fill_between(range(min(200, len(sv))), sv[:min(200, len(sv))], alpha=0.15, color='#7b68ee')
    ax.set_title('Embedding Singular Values', color='#7b68ee', fontsize=10)
    ax.set_xlabel('Index')
    ax.set_ylabel('σ (log)')

    # Effective rank
    sv_norm = sv / max(sv.sum(), 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-12]
    eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))
    ax.text(0.95, 0.95, f'Eff. rank ≈ {eff_rank:.1f}',
            transform=ax.transAxes, fontsize=9, color='#f5a623', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a3e', alpha=0.8))

    # ---- Panel 4: Embedding PCA ----
    ax = axes[1, 1]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_sub)
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.arange(n_show), cmap='viridis',
               s=3, alpha=0.5)
    ax.set_title(f'Embedding PCA (var={pca.explained_variance_ratio_.sum():.1%})',
                 color='#2ecc71', fontsize=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


def viz_inter_layer_weight_similarity(layers_data, weight_key, model_name, save_path=None):
    """
    Compute and visualize the cosine similarity between the same weight type
    across different layers. Shows how much the weight matrices change through depth.
    """
    vecs = []
    layer_indices = []
    for ld in layers_data:
        if weight_key in ld:
            W = ld[weight_key]
            vecs.append(W.flatten())
            layer_indices.append(ld['layer_idx'])

    if len(vecs) < 2:
        return None

    # Pad to same length
    max_len = max(len(v) for v in vecs)
    vecs_padded = np.zeros((len(vecs), max_len))
    for i, v in enumerate(vecs):
        vecs_padded[i, :len(v)] = v

    # Normalize
    norms = np.linalg.norm(vecs_padded, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    vecs_normed = vecs_padded / norms

    # Cosine similarity matrix
    sim_matrix = vecs_normed @ vecs_normed.T

    fig, axes = setup_dark_fig(1, 2, (14, 6),
                                title=f"{model_name} — {weight_key} Inter-Layer Similarity")

    # ---- Panel 1: Similarity heatmap ----
    ax = axes[0]
    im = ax.imshow(sim_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_title('Cosine Similarity Between Layers', color='#e94560', fontsize=10)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Layer')
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([str(l) for l in layer_indices], fontsize=7)
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels([str(l) for l in layer_indices], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # ---- Panel 2: Consecutive layer similarity ----
    ax = axes[1]
    consecutive_sim = [sim_matrix[i, i + 1] for i in range(len(vecs) - 1)]
    ax.plot(range(len(consecutive_sim)), consecutive_sim, 'o-', color='#53a8b6',
            markersize=6, linewidth=2)
    ax.fill_between(range(len(consecutive_sim)), consecutive_sim, alpha=0.15, color='#53a8b6')
    ax.set_title('Consecutive Layer Similarity', color='#53a8b6', fontsize=10)
    ax.set_xlabel('Layer Transition (L→L+1)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(-0.1, 1.1)

    # Highlight the biggest change
    if consecutive_sim:
        min_idx = np.argmin(consecutive_sim)
        ax.scatter([min_idx], [consecutive_sim[min_idx]], c='#e94560', s=100, zorder=5,
                   label=f'Biggest change: L{layer_indices[min_idx]}→L{layer_indices[min_idx+1]}')
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1a1a3e', labelcolor='#a0a0c0')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return fig


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_model(model, config, model_name, output_dir, args):
    """
    Run the full analysis pipeline on a single model.
    Returns all_stats for cross-model comparison.
    """
    model_dir = output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_name}")
    print(f"Output: {model_dir}")
    print(f"{'='*60}")

    # Extract weights
    print("\n[1/8] Extracting layer weights...")
    layers_data = extract_layer_weights(model, config)
    n_layers = len(layers_data)
    n_heads = get_n_heads(config)

    if n_layers == 0:
        print("  ERROR: No layers found!")
        return []

    # Identify weight types present
    weight_types = set()
    for ld in layers_data:
        for k in ld:
            if k != 'layer_idx' and isinstance(ld[k], np.ndarray) and ld[k].ndim == 2:
                weight_types.add(k)
    weight_types = sorted(weight_types)
    print(f"  Found weight types: {weight_types}")

    all_stats = []  # (layer_idx, weight_name, stats_dict)

    # ---- Per-layer analysis ----
    for layer_idx, ld in enumerate(layers_data):
        layer_dir = model_dir / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(exist_ok=True)

        print(f"\n[Layer {layer_idx}/{n_layers-1}] Analyzing...")

        for weight_name in weight_types:
            if weight_name not in ld:
                continue

            W = ld[weight_name]
            print(f"  {weight_name}: {W.shape}")

            wt_dir = layer_dir / weight_name
            wt_dir.mkdir(exist_ok=True)

            # 1. Weight canvas
            print(f"    [a] Weight canvas...")
            viz_weight_canvas(W, title=f"L{layer_idx} {weight_name}",
                              save_path=wt_dir / "weight_canvas.png")

            # 2. Weight as artistic image
            print(f"    [b] Weight image...")
            viz_weight_as_image(W, title=f"L{layer_idx} {weight_name}",
                                save_path=wt_dir / "weight_image.png")

            # 3. Row/column norms
            print(f"    [c] Row/column norms...")
            viz_row_column_norms(W, title=f"L{layer_idx} {weight_name} Norms",
                                 save_path=wt_dir / "row_col_norms.png")

            # 4. Graph construction
            print(f"    [d] Building graphs...")
            bipartite_G, similarity_G, cos_sim, stats = weight_matrix_to_graph(
                W, max_neurons=args.max_neurons, top_k=args.top_k_edges,
                name=weight_name
            )
            all_stats.append((layer_idx, weight_name, stats))

            # 5. Similarity graph visualization
            print(f"    [e] Similarity graph...")
            viz_similarity_graph(similarity_G, cos_sim, layer_idx, weight_name,
                                 save_path=wt_dir / "similarity_graph.png")

            # 6. Bipartite graph
            print(f"    [f] Bipartite graph...")
            viz_bipartite_graph(bipartite_G, layer_idx, weight_name,
                                save_path=wt_dir / "bipartite_graph.png")

            # 7. Spectral analysis
            print(f"    [g] Spectral analysis...")
            spectral = compute_spectral_properties(W, max_neurons=args.max_neurons)
            spectral['shape'] = W.shape
            viz_spectral_analysis(spectral, layer_idx, weight_name,
                                  save_path=wt_dir / "spectral_analysis.png")

            # 8. Community detection
            print(f"    [h] Community detection...")
            viz_community_detection(similarity_G, cos_sim, layer_idx, weight_name,
                                    save_path=wt_dir / "communities.png")

            # 9. Small-world analysis
            print(f"    [i] Small-world analysis...")
            viz_small_world_analysis(similarity_G, layer_idx, weight_name,
                                     save_path=wt_dir / "small_world.png")

            # 10. Correlation structure
            print(f"    [j] Correlation structure...")
            viz_weight_correlation_structure(W, title=f"L{layer_idx} {weight_name} Correlation",
                                             save_path=wt_dir / "correlation_structure.png",
                                             max_neurons=args.max_neurons)

        # ---- Attention head analysis ----
        if 'attn_qkv' in ld:
            print(f"  [Attention heads]...")
            viz_attention_head_graph(ld['attn_qkv'], n_heads, layer_idx,
                                     save_path=layer_dir / "attention_heads.png")

        # ---- MLP flow graph ----
        if 'mlp_up' in ld and 'mlp_down' in ld:
            print(f"  [MLP flow]...")
            viz_mlp_flow_graph(ld['mlp_up'], ld['mlp_down'], layer_idx,
                               save_path=layer_dir / "mlp_flow.png")

    # ---- Cross-layer analyses ----
    print(f"\n[2/8] Cross-layer weight norm landscape...")
    viz_weight_norms_landscape(layers_data, model_name,
                                save_path=model_dir / "weight_norms_landscape.png")

    print(f"[3/8] Singular value waterfalls...")
    for wt in weight_types:
        viz_cross_layer_sv_waterfall(layers_data, wt, model_name,
                                      save_path=model_dir / f"sv_waterfall_{wt}.png")

    print(f"[4/8] Weight PCA trajectories...")
    for wt in weight_types:
        viz_weight_pca_embedding(layers_data, wt, model_name,
                                  save_path=model_dir / f"weight_pca_{wt}.png")

    print(f"[5/8] Graph metrics dashboard...")
    viz_graph_metrics_dashboard(all_stats, model_name,
                                 save_path=model_dir / "graph_metrics_dashboard.png")

    print(f"[6/8] LayerNorm analysis...")
    viz_layernorm_analysis(layers_data, model_name,
                            save_path=model_dir / "layernorm_analysis.png")

    print(f"[7/8] Embedding matrix analysis...")
    viz_embedding_matrix(model, model_name,
                          save_path=model_dir / "embedding_matrix.png")

    print(f"[8/8] Inter-layer weight similarity...")
    for wt in weight_types:
        viz_inter_layer_weight_similarity(layers_data, wt, model_name,
                                           save_path=model_dir / f"inter_layer_sim_{wt}.png")

    # Save stats summary
    stats_summary = []
    for layer_idx, weight_name, stats in all_stats:
        stats_summary.append({
            'layer': layer_idx,
            'weight': weight_name,
            **{k: v for k, v in stats.items() if not isinstance(v, (np.ndarray, np.generic))}
        })

    import json
    with open(model_dir / "graph_stats.json", "w") as f:
        json.dump(stats_summary, f, indent=2, default=str)

    print(f"\n✅ {model_name} analysis complete! Results in: {model_dir}")
    return all_stats


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║         WEIGHT GRAPH ANALYZER                           ║
    ║   Interpreting Transformer Weights as Weighted Graphs   ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # ---- Load Model 1 ----
    model1, config1, name1 = load_model_and_config(args.model1)
    stats1 = analyze_model(model1, config1, name1, output_dir, args)

    # Free memory
    del model1
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc
    gc.collect()

    # ---- Load Model 2 (if not skipped) ----
    stats2 = None
    name2 = None
    if not args.skip_model2:
        try:
            model2, config2, name2 = load_model_and_config(args.model2)
            stats2 = analyze_model(model2, config2, name2, output_dir, args)

            # Free memory
            del model2
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            print(f"\n⚠️  Could not load model2 ({args.model2}): {e}")
            print("  Skipping cross-model comparison.")
            stats2 = None

    # ---- Cross-model comparison ----
    if stats1 and stats2 and name2:
        print(f"\n{'='*60}")
        print(f"CROSS-MODEL COMPARISON: {name1} vs {name2}")
        print(f"{'='*60}")

        viz_cross_model_comparison(stats1, stats2, name1, name2,
                                    save_path=output_dir / "cross_model_comparison.png")
        print(f"✅ Cross-model comparison saved!")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"ALL DONE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir.resolve()}")

    # Count output files
    n_files = sum(1 for _ in output_dir.rglob("*.png"))
    n_json = sum(1 for _ in output_dir.rglob("*.json"))
    print(f"Generated: {n_files} PNG images, {n_json} JSON files")

    # Print directory tree (top level)
    print(f"\nDirectory structure:")
    for item in sorted(output_dir.iterdir()):
        if item.is_dir():
            n_sub = sum(1 for _ in item.rglob("*"))
            print(f"  📁 {item.name}/ ({n_sub} files)")
        else:
            print(f"  📄 {item.name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
