#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "ripser",
#   "gudhi",
#   "scikit-learn",
#   "transformers",
#   "tiktoken",
#   "sentencepiece",
#   "protobuf",
#   "numpy",
#   "sae-lens",
#   "transformer-lens",
# ]
# ///

from pathlib import Path
import argparse
import os
import sys
import json
import threading
import numpy as np
import traceback
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from scipy.stats import wasserstein_distance
from urllib.parse import urlparse
from datetime import datetime, timedelta, UTC
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_SAE_RELEASE_ID = None
_SAE_N_LAYERS = 0
_SAE_LOAD_ATTEMPTED = set()  # track layers we already tried to load
default_k = 1

def visualize_curvature_landscape(curvature_data, tokens, save_path=None):
    """
    Render the Curvature Landscape across the model's depth.

    Args:
        curvature_data: dict returned by estimate_fiber_curvature().
        tokens: list of token strings.
        save_path: if provided, save figure to this path instead of showing.

    Returns:
        matplotlib Figure object.
    """
    orc = curvature_data['ollivier_ricci']           # [n_layers+1, seq_len]
    scalar = curvature_data['scalar_curvature']       # [n_layers, seq_len]
    log_det = curvature_data['metric_log_det']        # [n_layers+1, seq_len]
    procrustes = curvature_data['procrustes_deviation']  # [n_layers, seq_len]
    sectional = curvature_data['sectional_curvature']    # [n_layers, seq_len]

    n_layers_orc, seq_len = orc.shape
    n_layers = scalar.shape[0]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Holographic Curvature Landscape', fontsize=16, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#a0a0c0', labelsize=8)
            ax.spines['bottom'].set_color('#0f3460')
            ax.spines['top'].set_color('#0f3460')
            ax.spines['left'].set_color('#0f3460')
            ax.spines['right'].set_color('#0f3460')

    token_labels = [f'[{i}] {t}' for i, t in enumerate(tokens)]

    # ---- Helper: build a safe TwoSlopeNorm centered at 0 ----
    def safe_two_slope_norm(data_min, data_max, vcenter=0.0):
        """
        Build a TwoSlopeNorm that is guaranteed to satisfy vmin < vcenter < vmax.
        Falls back to a simple Normalize if the data doesn't straddle vcenter.
        """
        dmin = float(data_min)
        dmax = float(data_max)

        # Ensure we have finite values
        if not np.isfinite(dmin):
            dmin = -1.0
        if not np.isfinite(dmax):
            dmax = 1.0

        # Case 1: data straddles vcenter — normal TwoSlopeNorm
        if dmin < vcenter < dmax:
            return mcolors.TwoSlopeNorm(vmin=dmin, vcenter=vcenter, vmax=dmax)

        # Case 2: all data >= vcenter (e.g. all ORC values are positive)
        if dmin >= vcenter:
            # Push vmin below vcenter
            margin = max(dmax - vcenter, 0.01) * 0.1
            return mcolors.TwoSlopeNorm(vmin=vcenter - margin, vcenter=vcenter, vmax=max(dmax, vcenter + 0.01))

        # Case 3: all data <= vcenter (e.g. all ORC values are negative)
        if dmax <= vcenter:
            margin = max(vcenter - dmin, 0.01) * 0.1
            return mcolors.TwoSlopeNorm(vmin=min(dmin, vcenter - 0.01), vcenter=vcenter, vmax=vcenter + margin)

        # Fallback: shouldn't reach here, but just in case
        return mcolors.Normalize(vmin=dmin, vmax=dmax)

    # ---- Panel 1: Ollivier-Ricci Curvature ----
    ax = axes[0, 0]
    cmap_orc = plt.cm.RdBu_r
    norm_orc = safe_two_slope_norm(orc.min(), orc.max(), vcenter=0.0)
    im1 = ax.imshow(orc, aspect='auto', cmap=cmap_orc, norm=norm_orc, interpolation='nearest')
    ax.set_title('Ollivier-Ricci Curvature', color='#e94560', fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im1, ax=ax, shrink=0.8, label='ORC')

    # ---- Panel 2: Scalar Curvature (Volumetric Strain) ----
    ax = axes[0, 1]
    norm_sc = safe_two_slope_norm(scalar.min(), scalar.max(), vcenter=0.0)
    im2 = ax.imshow(scalar, aspect='auto', cmap='coolwarm', norm=norm_sc, interpolation='nearest')
    ax.set_title('Scalar Curvature (Volumetric Strain)', color='#53a8b6', fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im2, ax=ax, shrink=0.8, label='ΔlogVol')

    # ---- Panel 3: Procrustes Deviation (Connection Strength) ----
    ax = axes[1, 0]
    im3 = ax.imshow(procrustes, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_title('Procrustes Deviation ||R - I||_F (Connection)', color='#f5a623', fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im3, ax=ax, shrink=0.8, label='||R-I||')

    # ---- Panel 4: Sectional Curvature ----
    ax = axes[1, 1]
    im4 = ax.imshow(sectional, aspect='auto', cmap='inferno', interpolation='nearest')
    ax.set_title('Sectional Curvature (Holonomy Proxy)', color='#7b68ee', fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im4, ax=ax, shrink=0.8, label='Sectional κ')

    # ---- Panel 5: log(det(g)) — Metric Determinant ----
    ax = axes[2, 0]
    im5 = ax.imshow(log_det, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title('log det(g) — Metric Volume Element', color='#2ecc71', fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    fig.colorbar(im5, ax=ax, shrink=0.8, label='log det(g)')

    # ---- Panel 6: Curvature Singularity Detection ----
    ax = axes[2, 1]
    # Combine ORC and sectional curvature to find singularities
    # Use the layer-aligned portion of ORC (skip embedding layer for alignment)
    if orc.shape[0] > n_layers:
        orc_layers = orc[1:, :]  # (n_layers, seq_len) — skip embedding
    else:
        orc_layers = orc[:n_layers, :]
    combined = np.abs(orc_layers) + sectional  # both are (n_layers, seq_len)

    # Find top singularities
    flat_idx = np.argsort(combined.ravel())[::-1][:10]
    sing_layers, sing_tokens = np.unravel_index(flat_idx, combined.shape)

    ax.imshow(combined, aspect='auto', cmap='hot', interpolation='nearest')
    for sl, st in zip(sing_layers, sing_tokens):
        ax.plot(st, sl, 'o', markersize=8, markeredgecolor='cyan',
                markerfacecolor='none', markeredgewidth=2)
        if st < len(tokens):
            ax.annotate(f'{tokens[st]}', (st, sl), textcoords="offset points",
                        xytext=(5, -10), fontsize=7, color='cyan', fontweight='bold')

    ax.set_title('Curvature Singularities (|ORC| + Sectional)', color='#e94560',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Layer', color='#a0a0c0')
    ax.set_xlabel('Token', color='#a0a0c0')
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"[Curvature] Saved landscape to {save_path}")

    return fig

def correlate_metric_with_surprisal(curvature_data, hidden_states, tokenizer, model, input_ids):
    """
    Compute the correlation between det(g) (metric determinant) and
    token surprisal (information content = -log P(token)).

    Args:
        curvature_data: dict from estimate_fiber_curvature().
        hidden_states: tuple of hidden state tensors.
        tokenizer: the tokenizer.
        model: the LM model (with language modeling head).
        input_ids: tensor of input token IDs, shape (1, seq_len).

    Returns:
        dict with:
            'surprisal': np.ndarray of shape (seq_len,) — per-token surprisal
            'log_det_g_per_layer': np.ndarray of shape (n_layers+1, seq_len)
            'correlations': list of dicts per layer with pearson_r, spearman_rho, p_values
            'best_layer': int — layer with strongest correlation
            'summary': str — human-readable summary
    """
    log_det = curvature_data['metric_log_det']  # (n_layers+1, seq_len)
    n_layers_plus_one, seq_len = log_det.shape

    # ================================================================
    # Compute surprisal: -log2 P(token_i | token_{<i})
    # ================================================================
    surprisal = np.zeros(seq_len)

    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        # For token i, surprisal = -log2 P(token_i | context)
        # Token 0 has no context, so we assign surprisal = 0 or use uniform prior
        for i in range(1, seq_len):
            token_id = input_ids[0, i].item()
            # The prediction for token i comes from position i-1
            log_p = log_probs[i - 1, token_id].item()
            surprisal[i] = -log_p / np.log(2)  # convert to bits

        # Token 0: use uniform prior over vocab
        vocab_size = logits.shape[-1]
        surprisal[0] = np.log2(vocab_size)

    except Exception as e:
        print(f"[Curvature] Could not compute surprisal: {e}")
        surprisal = np.ones(seq_len) * np.log2(50257)  # fallback: uniform GPT-2 vocab

    # ================================================================
    # Compute correlations between log_det(g) and surprisal at each layer
    # ================================================================
    correlations = []
    best_layer = 0
    best_abs_r = 0.0

    for lay in range(n_layers_plus_one):
        log_det_l = log_det[lay]

        # Remove any NaN/Inf
        valid = np.isfinite(log_det_l) & np.isfinite(surprisal)
        if valid.sum() < 3:
            correlations.append({
                'layer': lay,
                'pearson_r': 0.0, 'pearson_p': 1.0,
                'spearman_rho': 0.0, 'spearman_p': 1.0,
            })
            continue

        ld_valid = log_det_l[valid]
        s_valid = surprisal[valid]

        # Pearson correlation
        try:
            pr, pp = pearsonr(ld_valid, s_valid)
        except Exception:
            pr, pp = 0.0, 1.0

        # Spearman rank correlation
        try:
            sr, sp = spearmanr(ld_valid, s_valid)
        except Exception:
            sr, sp = 0.0, 1.0

        correlations.append({
            'layer': lay,
            'pearson_r': round(float(pr), 4),
            'pearson_p': round(float(pp), 6),
            'spearman_rho': round(float(sr), 4),
            'spearman_p': round(float(sp), 6),
        })

        if abs(pr) > best_abs_r:
            best_abs_r = abs(pr)
            best_layer = lay

    # ================================================================
    # Generate summary
    # ================================================================
    best_corr = correlations[best_layer]
    direction = "positive" if best_corr['pearson_r'] > 0 else "negative"

    summary = (
        f"Strongest correlation between log det(g) and surprisal found at layer {best_layer}: "
        f"Pearson r = {best_corr['pearson_r']:.4f} (p = {best_corr['pearson_p']:.2e}), "
        f"Spearman ρ = {best_corr['spearman_rho']:.4f} (p = {best_corr['spearman_p']:.2e}). "
        f"This {direction} correlation suggests that tokens with "
        f"{'larger' if direction == 'positive' else 'smaller'} local metric volume "
        f"tend to carry {'more' if direction == 'positive' else 'less'} information content."
    )

    return {
        'surprisal': surprisal,
        'log_det_g_per_layer': log_det,
        'correlations': correlations,
        'best_layer': best_layer,
        'summary': summary,
    }



def decode_curvature_singularities(curvature_data, tokens, surprisal=None, top_k=10):
    """
    Identify curvature singularities and map them back to input tokens.
    Classify each singularity as a syntactic junction, entropy collapse,
    or gravitational source.

    Args:
        curvature_data: dict from estimate_fiber_curvature().
        tokens: list of token strings.
        surprisal: optional np.ndarray of shape (seq_len,) — per-token surprisal.
        top_k: number of top singularities to return.

    Returns:
        list of dicts, each describing a curvature singularity:
            {
                'token_idx': int,
                'token': str,
                'layer': int,
                'orc': float,
                'sectional': float,
                'scalar': float,
                'procrustes': float,
                'combined_score': float,
                'classification': str,  — 'syntactic_junction', 'entropy_collapse', 'gravitational_source'
                'description': str,
            }
    """
    orc = curvature_data['ollivier_ricci']            # (n_layers+1, seq_len)
    sectional = curvature_data['sectional_curvature']  # (n_layers, seq_len)
    scalar = curvature_data['scalar_curvature']        # (n_layers, seq_len)
    procrustes = curvature_data['procrustes_deviation']  # (n_layers, seq_len)

    n_layers = scalar.shape[0]

    # Align ORC to layer-only dimensions (skip embedding layer)
    orc_layers = orc[1:, :] if orc.shape[0] > n_layers else orc[:n_layers, :]

    # Compute a combined curvature score for singularity detection
    # Normalize each component to [0, 1] range before combining
    def safe_normalize(arr):
        arr_min = arr.min()
        arr_max = arr.max()
        rng = arr_max - arr_min
        if rng < 1e-15:
            return np.zeros_like(arr)
        return (arr - arr_min) / rng

    orc_norm = safe_normalize(np.abs(orc_layers))
    sectional_norm = safe_normalize(sectional)
    scalar_norm = safe_normalize(np.abs(scalar))
    procrustes_norm = safe_normalize(procrustes)

    # Weighted combination — ORC and sectional curvature are primary signals
    combined = (
        0.35 * orc_norm +
        0.30 * sectional_norm +
        0.20 * scalar_norm +
        0.15 * procrustes_norm
    )

    # Find top-k singularities
    flat_idx = np.argsort(combined.ravel())[::-1][:top_k]
    sing_layers, sing_tokens = np.unravel_index(flat_idx, combined.shape)

    singularities = []

    for rank, (sl, st) in enumerate(zip(sing_layers, sing_tokens)):
        token_str = tokens[st] if st < len(tokens) else f"[{st}]"

        orc_val = float(orc_layers[sl, st])
        sect_val = float(sectional[sl, st])
        scal_val = float(scalar[sl, st])
        proc_val = float(procrustes[sl, st])
        score = float(combined[sl, st])

        # ================================================================
        # Classification heuristics
        # ================================================================
        classification = 'unclassified'
        description = ''

        # 1. Gravitational Source: high positive ORC (tokens "attract" neighbors)
        #    + high sectional curvature (strong holonomy)
        if orc_val > 0 and sect_val > np.median(sectional):
            classification = 'gravitational_source'
            description = (
                f"Token '{token_str}' at layer {sl} acts as a gravitational source: "
                f"positive Ricci curvature (ORC={orc_val:.4f}) indicates neighboring tokens "
                f"converge toward this point. High sectional curvature ({sect_val:.4f}) "
                f"suggests strong holonomy — attention heads are actively rotating "
                f"the local frame around this token."
            )

        # 2. Entropy Collapse: large negative volumetric strain (space contracting)
        #    + high Procrustes deviation (tangent space rapidly reorienting)
        elif scal_val < -np.std(scalar) and proc_val > np.median(procrustes):
            classification = 'entropy_collapse'
            description = (
                f"Token '{token_str}' at layer {sl} exhibits entropy collapse: "
                f"negative volumetric strain (ΔlogVol={scal_val:.4f}) means the local "
                f"simplex is contracting — the model is 'deciding' on a meaning. "
                f"High Procrustes deviation ({proc_val:.4f}) indicates the tangent space "
                f"is rapidly reorienting as ambiguity resolves."
            )

        # 3. Syntactic Junction: high Procrustes deviation (connection strength)
        #    + moderate ORC (neither strongly positive nor negative)
        #    + token is at a structural boundary
        elif proc_val > np.median(procrustes) * 1.5:
            classification = 'syntactic_junction'
            description = (
                f"Token '{token_str}' at layer {sl} is a syntactic junction: "
                f"high connection strength (||R-I||={proc_val:.4f}) indicates the "
                f"parallel transport between layers undergoes significant rotation here. "
                f"This typically occurs at grammatical boundaries (subject→verb, "
                f"clause transitions) where the model shifts its processing mode."
            )

        # 4. Fallback: describe based on dominant signal
        else:
            if abs(orc_val) > abs(scal_val) and abs(orc_val) > proc_val:
                classification = 'curvature_anomaly'
                sign = 'positive' if orc_val > 0 else 'negative'
                description = (
                    f"Token '{token_str}' at layer {sl} shows anomalous {sign} "
                    f"Ricci curvature (ORC={orc_val:.4f}). "
                    f"{'Tokens converge here.' if orc_val > 0 else 'Tokens diverge here.'}"
                )
            elif abs(scal_val) > proc_val:
                classification = 'volume_anomaly'
                direction = 'expansion' if scal_val > 0 else 'contraction'
                description = (
                    f"Token '{token_str}' at layer {sl} shows volumetric {direction} "
                    f"(ΔlogVol={scal_val:.4f}). The local representation neighborhood "
                    f"is {'expanding' if scal_val > 0 else 'collapsing'}."
                )
            else:
                classification = 'transport_anomaly'
                description = (
                    f"Token '{token_str}' at layer {sl} shows high parallel transport "
                    f"deviation (||R-I||={proc_val:.4f}), indicating significant "
                    f"frame rotation between layers."
                )

        # ================================================================
        # Compute surprisal correlation if available
        # ================================================================
        surprisal_note = ''
        if surprisal is not None and st < len(surprisal):
            s_val = surprisal[st]
            surprisal_note = f" Token surprisal: {s_val:.2f} bits."

        singularities.append({
            'rank': rank + 1,
            'token_idx': int(st),
            'token': token_str,
            'layer': int(sl),
            'orc': round(orc_val, 6),
            'sectional': round(sect_val, 6),
            'scalar': round(scal_val, 6),
            'procrustes': round(proc_val, 6),
            'combined_score': round(score, 6),
            'classification': classification,
            'description': description + surprisal_note,
        })

    return singularities



def visualize_metric_surprisal_correlation(correlation_data, tokens, save_path=None):
    """
    Visualize the correlation between log det(g) and token surprisal.

    Args:
        correlation_data: dict from correlate_metric_with_surprisal().
        tokens: list of token strings.
        save_path: optional path to save figure.

    Returns:
        matplotlib Figure object.
    """
    surprisal = correlation_data['surprisal']
    log_det = correlation_data['log_det_g_per_layer']
    correlations = correlation_data['correlations']
    best_layer = correlation_data['best_layer']
    seq_len = len(surprisal)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Holographic Decoding: Metric Determinant vs. Information Content',
                 fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('#1a1a2e')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#0d1117')
            ax.tick_params(colors='#a0a0c0', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#0f3460')

    token_labels = [f'[{i}] {t}' for i, t in enumerate(tokens)]

    # ---- Panel 1: Scatter plot at best layer ----
    ax = axes[0, 0]
    ld_best = log_det[best_layer]
    valid = np.isfinite(ld_best) & np.isfinite(surprisal)

    colors_scatter = plt.cm.viridis(np.linspace(0, 1, seq_len))
    for i in range(seq_len):
        if valid[i]:
            ax.scatter(ld_best[i], surprisal[i], c=[colors_scatter[i]],
                       s=60, zorder=5, edgecolors='white', linewidths=0.5)
            ax.annotate(tokens[i], (ld_best[i], surprisal[i]),
                        fontsize=7, color='#a0a0c0',
                        textcoords="offset points", xytext=(5, 3))

    # Fit line
    if valid.sum() >= 2:
        z = np.polyfit(ld_best[valid], surprisal[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(ld_best[valid].min(), ld_best[valid].max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#e94560', linewidth=1.5, alpha=0.7)

    best_corr = correlations[best_layer]
    ax.set_title(f'Best Layer {best_layer}: r={best_corr["pearson_r"]:.3f}, '
                 f'ρ={best_corr["spearman_rho"]:.3f}',
                 color='#e94560', fontsize=10, fontweight='bold')
    ax.set_xlabel('log det(g)', color='#a0a0c0')
    ax.set_ylabel('Surprisal (bits)', color='#a0a0c0')

    # ---- Panel 2: Correlation strength across layers ----
    ax = axes[0, 1]
    layer_indices = [c['layer'] for c in correlations]
    pearson_vals = [c['pearson_r'] for c in correlations]
    spearman_vals = [c['spearman_rho'] for c in correlations]

    ax.bar(np.array(layer_indices) - 0.15, pearson_vals, width=0.3,
           color='#e94560', alpha=0.8, label='Pearson r')
    ax.bar(np.array(layer_indices) + 0.15, spearman_vals, width=0.3,
           color='#53a8b6', alpha=0.8, label='Spearman ρ')
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.axvline(x=best_layer, color='#f5a623', linewidth=2, linestyle='--',
               alpha=0.7, label=f'Best layer ({best_layer})')
    ax.set_title('Correlation Strength Across Layers',
                 color='#53a8b6', fontsize=10, fontweight='bold')
    ax.set_xlabel('Layer', color='#a0a0c0')
    ax.set_ylabel('Correlation', color='#a0a0c0')
    ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#0f3460',
              labelcolor='#a0a0c0')

    # ---- Panel 3: Surprisal profile ----
    ax = axes[1, 0]
    ax.bar(range(seq_len), surprisal, color='#7b68ee', alpha=0.8)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    ax.set_title('Token Surprisal Profile', color='#7b68ee',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('Surprisal (bits)', color='#a0a0c0')

    # ---- Panel 4: log det(g) at best layer ----
    ax = axes[1, 1]
    ax.bar(range(seq_len), ld_best, color='#2ecc71', alpha=0.8)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(f'log det(g) at Layer {best_layer}', color='#2ecc71',
                 fontsize=10, fontweight='bold')
    ax.set_ylabel('log det(g)', color='#a0a0c0')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[Curvature] Saved correlation plot to {save_path}")

    return fig

def handle_curvature_analysis(body_bytes):
    """
    Full holographic curvature analysis endpoint.
    Computes fiber curvature, singularities, and metric-surprisal correlation.
    """
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    k_neighbors = req.get("k_neighbors", 8)
    pca_d = req.get("pca_d", 16)
    top_k_singularities = req.get("top_k_singularities", 10)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    # Tokenize and extract hidden states
    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    seq_len = input_ids.shape[1]

    print(f"[Curvature] Analyzing {seq_len} tokens across {n_layers} layers "
          f"(k={k_neighbors}, d={pca_d})...")

    # ================================================================
    # Stage 1-3: Estimate fiber curvature
    # ================================================================
    curvature_data = estimate_fiber_curvature(
        hs, k_neighbors=k_neighbors, pca_d=pca_d
    )

    print(f"[Curvature] ORC shape: {curvature_data['ollivier_ricci'].shape}")
    print(f"[Curvature] Scalar curvature shape: {curvature_data['scalar_curvature'].shape}")

    # ================================================================
    # Stage 4a: Correlate with surprisal
    # ================================================================
    correlation_data = None
    if LM_MODEL is not None:
        correlation_data = correlate_metric_with_surprisal(
            curvature_data, hs, TOKENIZER, LM_MODEL, input_ids
        )
        print(f"[Curvature] {correlation_data['summary']}")
    else:
        print("[Curvature] No LM model — skipping surprisal correlation")

    # ================================================================
    # Stage 4b: Decode singularities
    # ================================================================
    surprisal = correlation_data['surprisal'] if correlation_data else None
    singularities = decode_curvature_singularities(
        curvature_data, tokens,
        surprisal=surprisal,
        top_k=top_k_singularities
    )

    print(f"[Curvature] Found {len(singularities)} singularities")
    for s in singularities[:3]:
        print(f"  #{s['rank']}: [{s['token_idx']}] '{s['token']}' "
              f"at L{s['layer']} — {s['classification']}")

    # ================================================================
    # Stage 5: Generate visualizations
    # ================================================================
    fig_landscape = visualize_curvature_landscape(
        curvature_data, tokens, save_path='/tmp/curvature_landscape.png'
    )
    plt.close(fig_landscape)

    if correlation_data:
        fig_corr = visualize_metric_surprisal_correlation(
            correlation_data, tokens, save_path='/tmp/curvature_correlation.png'
        )
        plt.close(fig_corr)

    # ================================================================
    # Build JSON response
    # ================================================================
    response = {
        "tokens": tokens,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "k_neighbors": k_neighbors,
        "pca_d": pca_d,

        # Curvature tensors (main output)
        "ollivier_ricci": curvature_data['ollivier_ricci'].tolist(),
        "scalar_curvature": curvature_data['scalar_curvature'].tolist(),
        "sectional_curvature": curvature_data['sectional_curvature'].tolist(),
        "metric_log_det": curvature_data['metric_log_det'].tolist(),
        "procrustes_deviation": curvature_data['procrustes_deviation'].tolist(),

        # Singularities
        "singularities": singularities,

        # Correlation analysis
        "correlation": {
            "summary": correlation_data['summary'],
            "best_layer": correlation_data['best_layer'],
            "surprisal": correlation_data['surprisal'].tolist(),
            "correlations_per_layer": correlation_data['correlations'],
        } if correlation_data else None,

        # Visualization paths
        "visualizations": {
            "landscape": "/tmp/curvature_landscape.png",
            "correlation": "/tmp/curvature_correlation.png" if correlation_data else None,
        },
    }

    return json.dumps(response, cls=SafeFloatEncoder).encode()

class SafeFloatEncoder(json.JSONEncoder):
    """JSON encoder that replaces inf/-inf/nan with 0.0 instead of crashing."""
    def default(self, obj):
        return super().default(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return 0.0
            if obj == float('inf') or obj == float('-inf'):
                return 0.0
            return obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj

# ============================================================
# 3b. SAE LOADING (per-layer sparse autoencoders)
# ============================================================

SAE_MODELS = {}  # layer_idx -> trained SAE

# Known SAE release mappings for popular models
# SAELens uses release IDs like "gpt2-small-res-jb" etc.
SAE_RELEASE_MAP = {
    "gpt2":                          "gpt2-small-res-jb",
    "gpt2-small":                    "gpt2-small-res-jb",
    "gpt2-medium":                   "gpt2-medium-res-jb",
    "gpt2-large":                    "gpt2-large-res-jb",
    "EleutherAI/pythia-70m-deduped": "pythia-70m-deduped-res-sm",
    "EleutherAI/pythia-160m":        "pythia-160m-deduped-res-sm",
    "EleutherAI/pythia-410m":        "pythia-410m-deduped-res-sm",
}

# Hook point template per architecture
SAE_HOOK_TEMPLATES = {
    "gpt2": "blocks.{layer}.hook_resid_post",
    "pythia": "blocks.{layer}.hook_resid_post",
    "default": "blocks.{layer}.hook_resid_post",
}


def get_sae_release_id(model_name):
    """Map a HuggingFace model name to a SAELens release ID."""
    # Direct lookup first
    if model_name in SAE_RELEASE_MAP:
        return SAE_RELEASE_MAP[model_name]
    # Fuzzy matching
    mn = model_name.lower()
    if "gpt2" in mn:
        if "xl" in mn:
            return None  # no public SAE for gpt2-xl yet
        if "large" in mn:
            return "gpt2-large-res-jb"
        if "medium" in mn:
            return "gpt2-medium-res-jb"
        return "gpt2-small-res-jb"
    if "pythia" in mn:
        for key in SAE_RELEASE_MAP:
            if key.lower() in mn:
                return SAE_RELEASE_MAP[key]
    return None

def get_sae_hook_template(model_name):
    """Get the hook point template string for a model."""
    mn = model_name.lower()
    if "gpt2" in mn:
        return SAE_HOOK_TEMPLATES["gpt2"]
    if "pythia" in mn:
        return SAE_HOOK_TEMPLATES["pythia"]
    return SAE_HOOK_TEMPLATES["default"]

def load_saes(model_name, n_layers):
    """Initialize SAE metadata but don't actually load weights yet (lazy loading)."""
    global SAE_MODELS, _SAE_RELEASE_ID, _SAE_N_LAYERS, _SAE_LOAD_ATTEMPTED
    SAE_MODELS = {}
    _SAE_RELEASE_ID = None
    _SAE_N_LAYERS = n_layers
    _SAE_LOAD_ATTEMPTED = set()

    release_id = get_sae_release_id(model_name)
    if release_id is None:
        print(f"[SAE] No known SAE release for model '{model_name}' — skipping")
        return

    _SAE_RELEASE_ID = release_id
    print(f"[SAE] SAEs will be lazy-loaded from release '{release_id}' for {n_layers} layers on demand")

def get_sae_for_layer(layer):
    """Lazy-load and return the SAE for a given layer, or None if unavailable."""
    global SAE_MODELS, _SAE_LOAD_ATTEMPTED

    # Already loaded
    if layer in SAE_MODELS:
        return SAE_MODELS[layer]

    # Already tried and failed
    if layer in _SAE_LOAD_ATTEMPTED:
        return None

    _SAE_LOAD_ATTEMPTED.add(layer)

    if _SAE_RELEASE_ID is None:
        return None

    from sae_lens import SAE

    # Try hook_resid_pre first (used by gpt2-small-res-jb), then hook_resid_post
    sae_id_candidates = [
        f"blocks.{layer}.hook_resid_pre",
        f"blocks.{layer}.hook_resid_post",
    ]

    for sae_id in sae_id_candidates:
        try:
            sae = SAE.from_pretrained(
                release=_SAE_RELEASE_ID,
                sae_id=sae_id,
            )
            # Handle the deprecation: from_pretrained now returns just the SAE
            if isinstance(sae, tuple):
                sae = sae[0]
            sae.eval()
            SAE_MODELS[layer] = sae
            d_sae = sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else '?'
            print(f"  [SAE] Layer {layer}: loaded ({d_sae} latents) via {sae_id}")
            return sae
        except Exception:
            continue

    print(f"  [SAE] Layer {layer}: not available in release {_SAE_RELEASE_ID}")
    return None

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
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def ensure_safe_env():
    """Ensure uv only uses packages at least 8 days old."""
    if not should_set_exclude_newer():
        return

    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date

    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


# This must run BEFORE heavy imports
ensure_safe_env()


# ============================================================
# 2. HEAVY IMPORTS
# ============================================================

try:
    import webbrowser  # noqa: E402
    import time  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from transformers import AutoTokenizer, AutoModel  # noqa: E402
    from http.server import HTTPServer, BaseHTTPRequestHandler  # noqa: E402
except KeyboardInterrupt:
    pass

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

    mtype = detect_model_type(MODEL_CONFIG)
    LM_MODEL = None
    if mtype == "causal":
        try:
            from transformers import AutoModelForCausalLM
            LM_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
            LM_MODEL.eval()
        except Exception as e:
            print(f"[Model] Could not load LM head: {e}")

    # Load SAEs for this model
    n_layers = get_n_layers(MODEL_CONFIG)
    load_saes(model_name, n_layers)

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
    hs = hidden_states
    seq_len = hs[0].shape[1]
    layer0_vecs = []
    delta_lists = []
    for s in range(seq_len):
        layer0_vecs.append(hs[0][0][s].cpu().float().numpy())  # .float() → float32
        deltas = []
        for lay in range(n_layers):
            deltas.append((hs[lay + 1][0][s] - hs[lay][0][s]).cpu().float().numpy())
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

    For BERT/RoBERTa style models, each block computes:
        attn_out = block.attention(h)          -> (hidden_dim,)
        intermediate = block.intermediate(...)  -> (intermediate_size,)  e.g. 3072
        mlp_out = block.output.dense(intermediate) -> (hidden_dim,)

    We hook block.output.dense (the down-projection) for the MLP signal,
    NOT block.intermediate (which is the up-projection to intermediate_size).

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

        # BERT / RoBERTa: block.attention, block.intermediate, block.output
        # block.intermediate projects hidden_dim -> intermediate_size (e.g. 768 -> 3072)
        # block.output.dense projects intermediate_size -> hidden_dim (e.g. 3072 -> 768)
        # We must hook block.output.dense for the MLP signal to get the correct shape.
        elif hasattr(block, 'attention') and hasattr(block, 'output'):
            hooks.append(block.attention.register_forward_hook(_make_attn_hook(layer_idx)))
            if hasattr(block, 'intermediate') and hasattr(block.output, 'dense'):
                # Hook the down-projection (intermediate_size -> hidden_dim)
                hooks.append(block.output.dense.register_forward_hook(_make_mlp_hook(layer_idx)))
            elif hasattr(block, 'intermediate'):
                # Fallback: hook block.output (includes residual + LayerNorm)
                hooks.append(block.output.register_forward_hook(_make_mlp_hook(layer_idx)))
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
                attn_vec = attn_outputs[lay][0][s].cpu().float().numpy()
                # Safety check: if shape doesn't match hidden_dim, zero-fill
                if attn_vec.shape[0] != hidden_dim:
                    attn_list.append(np.zeros(hidden_dim))
                else:
                    attn_list.append(attn_vec)
            else:
                attn_list.append(np.zeros(hidden_dim))
            if lay in mlp_outputs:
                mlp_vec = mlp_outputs[lay][0][s].cpu().float().numpy()
                # Safety check: if shape doesn't match hidden_dim, zero-fill
                if mlp_vec.shape[0] != hidden_dim:
                    mlp_list.append(np.zeros(hidden_dim))
                else:
                    mlp_list.append(mlp_vec)
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

# ============================================================
# 7c. PREDICTED TOKEN EMBEDDING & DELTA EXTRACTION
# ============================================================

def embed_predicted_tokens(tokenizer, model, lm_model, input_ids, model_config, k=default_k):
    """
    Predict the top-k next tokens, then for each one:
      1. Append it to the input sequence
      2. Run the extended sequence through the model
      3. Extract the embedding (layer 0) and per-layer deltas
         for the predicted token position

    Returns:
        pred_layer0: list of numpy arrays — embedding vectors for each predicted token
        pred_deltas: list of list of numpy arrays — [k][n_layers] deltas
        pred_labels: list of str — token labels
        pred_probs:  list of float — probabilities
        pred_token_ids: list of int — token IDs
    """
    if lm_model is None:
        return [], [], [], [], []

    n_layers = get_n_layers(model_config)

    try:
        with torch.no_grad():
            outputs = lm_model(input_ids)
            logits = outputs.logits[0, -1, :]  # last token's logits
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, k)
    except Exception as e:
        print(f"[Predicted] Could not get predictions: {e}")
        return [], [], [], [], []

    pred_layer0 = []
    pred_deltas = []
    pred_labels = []
    pred_probs = []
    pred_token_ids = []

    for i in range(k):
        tid = topk.indices[i].item()
        prob = topk.values[i].item()
        token_str = tokenizer.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")

        # Build extended sequence: original + this predicted token
        extended_ids = torch.cat([
            input_ids,
            torch.tensor([[tid]], device=input_ids.device)
        ], dim=1)

        try:
            with torch.no_grad():
                hs = extract_hidden_states(model, extended_ids)

            # The predicted token is at position -1 (last) in the extended sequence
            pred_pos = extended_ids.shape[1] - 1

            # Layer 0 embedding for the predicted token
            layer0_vec = hs[0][0][pred_pos].cpu().float().numpy()
            pred_layer0.append(layer0_vec)

            # Per-layer deltas for the predicted token
            deltas = []
            for lay in range(n_layers):
                delta = (hs[lay + 1][0][pred_pos] - hs[lay][0][pred_pos]).cpu().float().numpy()
                deltas.append(delta)
            pred_deltas.append(deltas)

            pred_labels.append(f"→{token_str}")
            pred_probs.append(round(prob, 4))
            pred_token_ids.append(tid)

        except Exception as e:
            print(f"[Predicted] Error embedding token '{token_str}': {e}")
            continue

    print(f"[Predicted] Embedded {len(pred_layer0)}/{k} predicted next tokens")
    return pred_layer0, pred_deltas, pred_labels, pred_probs, pred_token_ids

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

        emb_matrix.shape[0]
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
    # Ensure float32 or float64 for linalg compatibility
    layer0_mat = layer0_mat.astype(np.float32)

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
    r = mx - mn
    if r < 1e-8:
        r = 1.0
        mn = mn - 0.5
        mx = mx + 0.5
    return mn - pad_frac * r, mx + pad_frac * r, r

def make_grid_coords(g1, g2):
    """Return list of (v1, v2) pairs for a 2-D grid."""
    coords = []
    for v1 in g1:
        for v2 in g2:
            coords.append((v1, v2))
    return coords


def interpolate_grid_embedding(v1, v2, centroid, pc1, pc2):
    """Reconstruct a high-dim embedding from PCA coordinates, sanitized."""
    emb = centroid + v1 * pc1 + v2 * pc2
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    emb = np.clip(emb, -1e6, 1e6)
    return emb

def compute_grid_weights(v1, v2, existing_proj, sigma_nn):
    """Compute RBF weights for a grid point relative to existing projections."""
    sigma_nn = max(sigma_nn, 1e-6)  # prevent division by zero
    dists = (existing_proj[:, 0] - v1) ** 2 + (existing_proj[:, 1] - v2) ** 2
    # Clamp exponent to prevent overflow
    exponents = -dists / (2 * sigma_nn ** 2)
    exponents = np.clip(exponents, -500, 0)
    weights = np.exp(exponents)
    w_sum = weights.sum()
    if w_sum < 1e-30:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= w_sum
    return weights

# ============================================================
# 9b. PLUGGABLE INTERPOLATION METHODS
# ============================================================

def interpolate_rbf_2d(query_x, query_y, source_x, source_y, source_vals, sigma):
    """Gaussian RBF interpolation. Returns interpolated value (scalar per dim)."""
    sigma = max(sigma, 1e-6)
    s2i = 1.0 / (2 * sigma ** 2)
    dists_sq = (query_x - source_x) ** 2 + (query_y - source_y) ** 2
    exponents = np.clip(-dists_sq * s2i, -500, 0)
    weights = np.exp(exponents)
    w_sum = weights.sum()
    if w_sum < 1e-30:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= w_sum
    return weights @ source_vals


def interpolate_tps_weights(source_x, source_y, source_vals):
    """Thin Plate Spline: precompute coefficients. Returns (coeffs, pts, n)."""
    n = len(source_x)
    if n < 3:
        return None
    try:
        pts = np.stack([source_x, source_y], axis=1)
        D = cdist(pts, pts, metric='euclidean')
        D = np.clip(D, 1e-12, None)
        K = D ** 2 * np.log(D)
        np.fill_diagonal(K, 0.0)
        P = np.column_stack([np.ones(n), source_x, source_y])
        Z = np.zeros((3, 3))
        reg = 1e-6 * np.eye(n)
        top = np.hstack([K + reg, P])
        bot = np.hstack([P.T, Z])
        A = np.vstack([top, bot])
        rhs = np.concatenate([source_vals, np.zeros(3)])
        coeffs = np.linalg.solve(A, rhs)
        return coeffs, pts, n
    except (np.linalg.LinAlgError, ValueError):
        return None


def interpolate_tps_eval(query_x, query_y, coeffs, pts, n):
    """Evaluate TPS at a query point."""
    qpt = np.array([[query_x, query_y]])
    dq = cdist(qpt, pts, metric='euclidean').flatten()
    dq = np.clip(dq, 1e-12, None)
    kq = dq ** 2 * np.log(dq)
    return float(np.dot(kq, coeffs[:n]) + coeffs[n] + coeffs[n+1]*query_x + coeffs[n+2]*query_y)


def compute_itp_weights(v1, v2, existing_proj, sigma_nn, method='rbf'):
    """Compute interpolation weights for weight-based methods.
    Returns normalized weight vector of shape (n_points,)."""
    source_x = existing_proj[:, 0]
    source_y = existing_proj[:, 1]
    n = len(source_x)

    if method == 'rbf':
        return compute_grid_weights(v1, v2, existing_proj, sigma_nn)

    elif method == 'idw':
        p = 2.0
        dists = np.sqrt((v1 - source_x) ** 2 + (v2 - source_y) ** 2)
        dists = np.clip(dists, 1e-12, None)
        w = 1.0 / (dists ** p)
        w_sum = w.sum()
        return w / w_sum if w_sum > 1e-30 else np.ones(n) / n

    elif method == 'nn':
        dists_sq = (v1 - source_x) ** 2 + (v2 - source_y) ** 2
        w = np.zeros(n)
        w[np.argmin(dists_sq)] = 1.0
        return w

    elif method == 'wendland':
        R = max(3.0 * sigma_nn, 1e-6)
        dists = np.sqrt((v1 - source_x) ** 2 + (v2 - source_y) ** 2)
        r_norm = dists / R
        mask = r_norm < 1.0
        w = np.zeros(n)
        r_clipped = np.clip(1.0 - r_norm[mask], 0, 1)
        w[mask] = r_clipped ** 4 * (4.0 * r_norm[mask] + 1.0)
        w_sum = w.sum()
        if w_sum < 1e-30:
            # Fall back to RBF if all weights are zero
            return compute_grid_weights(v1, v2, existing_proj, sigma_nn)
        return w / w_sum

    elif method == 'mls' or method == 'tps':
        # MLS and TPS are not weight-based; return RBF weights as fallback
        # The actual MLS/TPS interpolation is handled per-dimension in create_grid_probes
        return compute_grid_weights(v1, v2, existing_proj, sigma_nn)

    else:
        return compute_grid_weights(v1, v2, existing_proj, sigma_nn)


def interpolate_deltas_mls(v1, v2, existing_proj, all_deltas_per_point, n_layers, hidden_dim, sigma_nn):
    """Moving Least Squares interpolation of deltas. Fits a local linear model."""
    source_x = existing_proj[:, 0]
    source_y = existing_proj[:, 1]
    n_total = len(all_deltas_per_point)
    sigma_nn = max(sigma_nn, 1e-6)
    s2i = 1.0 / (2 * sigma_nn ** 2)

    # Compute Gaussian weights
    dists_sq = (v1 - source_x) ** 2 + (v2 - source_y) ** 2
    W = np.exp(np.clip(-dists_sq * s2i, -500, 0))

    # Build weighted least squares: f(x,y) = a0 + a1*(x-v1) + a2*(y-v2)
    dx_local = source_x - v1
    dy_local = source_y - v2

    # A^T W A (3x3 symmetric)
    w_sum = W.sum()
    if w_sum < 1e-30:
        # Fall back to uniform weights
        W = np.ones(n_total) / n_total
        w_sum = 1.0

    AtwA = np.array([
        [W.sum(),              (W * dx_local).sum(),       (W * dy_local).sum()],
        [(W * dx_local).sum(), (W * dx_local**2).sum(),    (W * dx_local * dy_local).sum()],
        [(W * dy_local).sum(), (W * dx_local * dy_local).sum(), (W * dy_local**2).sum()],
    ])
    # Regularize
    AtwA += 1e-8 * np.eye(3)

    point_deltas = []
    for lay in range(n_layers):
        d = np.zeros(hidden_dim)
        # Stack all source deltas for this layer: (n_total, hidden_dim)
        src_deltas = np.stack([all_deltas_per_point[pi][lay] for pi in range(n_total)], axis=0)

        # A^T W v for all hidden dims at once: (3, hidden_dim)
        Atwv = np.array([
            (W[:, None] * src_deltas).sum(axis=0),
            (W[:, None] * dx_local[:, None] * src_deltas).sum(axis=0),
            (W[:, None] * dy_local[:, None] * src_deltas).sum(axis=0),
        ])

        try:
            # Solve (3x3) system for each hidden dim: coeffs shape (3, hidden_dim)
            coeffs = np.linalg.solve(AtwA, Atwv)
            d = coeffs[0]  # a0 = value at query point (local coords = 0)
        except np.linalg.LinAlgError:
            # Fall back to weighted average
            d = (W[:, None] * src_deltas).sum(axis=0) / max(w_sum, 1e-30)

        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        point_deltas.append(d)
    return point_deltas


def interpolate_deltas_tps(existing_proj, all_deltas_per_point, n_layers, hidden_dim):
    """
    Precompute TPS coefficients for all layers and all hidden dims.
    Returns a function that evaluates the TPS at any query point.

    TPS kernel: phi(r) = r^2 * log(r)
    System: [K+reg, P; P^T, 0] [w; a] = [v; 0]
    """
    source_x = existing_proj[:, 0]
    source_y = existing_proj[:, 1]
    n = len(source_x)

    if n < 3:
        return None  # Can't do TPS with fewer than 3 points

    pts = np.stack([source_x, source_y], axis=1)
    D = cdist(pts, pts, metric='euclidean')
    D = np.clip(D, 1e-12, None)
    K = D ** 2 * np.log(D)
    np.fill_diagonal(K, 0.0)

    P = np.column_stack([np.ones(n), source_x, source_y])
    Z = np.zeros((3, 3))
    reg = 1e-6 * np.eye(n)
    top = np.hstack([K + reg, P])
    bot = np.hstack([P.T, Z])
    A = np.vstack([top, bot])

    # Precompute coefficients for every layer and hidden dim
    # This is expensive but done once per grid rebuild
    all_coeffs = []  # [n_layers][hidden_dim] -> coeffs array of length n+3

    try:
        # LU factorize once, solve for all RHS
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(A)

        for lay in range(n_layers):
            src_deltas = np.stack([all_deltas_per_point[pi][lay] for pi in range(n)], axis=0)
            # src_deltas shape: (n, hidden_dim)
            # Build RHS: (n+3, hidden_dim)
            rhs = np.vstack([src_deltas, np.zeros((3, hidden_dim))])
            coeffs = lu_solve((lu, piv), rhs)  # (n+3, hidden_dim)
            all_coeffs.append(coeffs)

    except (np.linalg.LinAlgError, ValueError):
        return None

    def evaluate_tps(v1, v2):
        """Evaluate the precomputed TPS at query point (v1, v2)."""
        qpt = np.array([[v1, v2]])
        dq = cdist(qpt, pts, metric='euclidean').flatten()
        dq = np.clip(dq, 1e-12, None)
        kq = dq ** 2 * np.log(dq)  # (n,)

        point_deltas = []
        for lay in range(n_layers):
            coeffs = all_coeffs[lay]  # (n+3, hidden_dim)
            # val = kq . coeffs[:n] + coeffs[n] + coeffs[n+1]*v1 + coeffs[n+2]*v2
            d = kq @ coeffs[:n] + coeffs[n] + coeffs[n+1] * v1 + coeffs[n+2] * v2
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            point_deltas.append(d)
        return point_deltas

    return evaluate_tps

def interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim):
    """Weighted-average the deltas from all points for one grid point, sanitized."""
    n_total = len(all_deltas_per_point)
    point_deltas = []
    for lay in range(n_layers):
        d = np.zeros(hidden_dim)
        for pi in range(n_total):
            d += weights[pi] * all_deltas_per_point[pi][lay]
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        point_deltas.append(d)
    return point_deltas

def create_grid_probes(centroid, pc1, pc2, proj1, proj2, existing_proj,
                       all_deltas_per_point, n_layers, hidden_dim, n_side=10, pad_frac=0.3,
                       all_attn_deltas=None, all_mlp_deltas=None, itp_method='rbf'):
    """
    Create grid intersection probes in PCA space.
    Uses the specified interpolation method for delta interpolation.
    Returns (grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas).
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

    # For TPS, precompute coefficients once (expensive but amortized)
    tps_eval_fn = None
    tps_attn_eval_fn = None
    tps_mlp_eval_fn = None
    if itp_method == 'tps':
        print(f"[Grid] Precomputing TPS coefficients ({len(all_deltas_per_point)} points, {hidden_dim} dims)...")
        tps_eval_fn = interpolate_deltas_tps(existing_proj, all_deltas_per_point, n_layers, hidden_dim)
        if tps_eval_fn is None:
            print("[Grid] TPS precomputation failed, falling back to RBF")
            itp_method = 'rbf'
        else:
            if all_attn_deltas is not None:
                tps_attn_eval_fn = interpolate_deltas_tps(existing_proj, all_attn_deltas, n_layers, hidden_dim)
            if all_mlp_deltas is not None:
                tps_mlp_eval_fn = interpolate_deltas_tps(existing_proj, all_mlp_deltas, n_layers, hidden_dim)

    # Weight-based methods can reuse compute_itp_weights + interpolate_deltas
    weight_based = itp_method in ('rbf', 'idw', 'nn', 'wendland')

    for v1, v2 in make_grid_coords(g1, g2):
        emb = interpolate_grid_embedding(v1, v2, centroid, pc1, pc2)
        grid_layer0.append(emb)

        if itp_method == 'tps' and tps_eval_fn is not None:
            point_deltas = tps_eval_fn(v1, v2)
            grid_deltas.append(point_deltas)
            if all_attn_deltas is not None and tps_attn_eval_fn is not None:
                grid_attn_deltas.append(tps_attn_eval_fn(v1, v2))
            elif all_attn_deltas is not None:
                # Fallback for attn
                w = compute_grid_weights(v1, v2, existing_proj, sigma_nn)
                grid_attn_deltas.append(interpolate_deltas(w, all_attn_deltas, n_layers, hidden_dim))
            if all_mlp_deltas is not None and tps_mlp_eval_fn is not None:
                grid_mlp_deltas.append(tps_mlp_eval_fn(v1, v2))
            elif all_mlp_deltas is not None:
                w = compute_grid_weights(v1, v2, existing_proj, sigma_nn)
                grid_mlp_deltas.append(interpolate_deltas(w, all_mlp_deltas, n_layers, hidden_dim))

        elif itp_method == 'mls':
            point_deltas = interpolate_deltas_mls(v1, v2, existing_proj,
                                                   all_deltas_per_point, n_layers, hidden_dim, sigma_nn)
            grid_deltas.append(point_deltas)
            if all_attn_deltas is not None:
                grid_attn_deltas.append(interpolate_deltas_mls(v1, v2, existing_proj,
                                                                all_attn_deltas, n_layers, hidden_dim, sigma_nn))
            if all_mlp_deltas is not None:
                grid_mlp_deltas.append(interpolate_deltas_mls(v1, v2, existing_proj,
                                                               all_mlp_deltas, n_layers, hidden_dim, sigma_nn))

        elif weight_based:
            weights = compute_itp_weights(v1, v2, existing_proj, sigma_nn, method=itp_method)
            point_deltas = interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim)
            grid_deltas.append(point_deltas)
            if all_attn_deltas is not None:
                grid_attn_deltas.append(interpolate_deltas(weights, all_attn_deltas, n_layers, hidden_dim))
            if all_mlp_deltas is not None:
                grid_mlp_deltas.append(interpolate_deltas(weights, all_mlp_deltas, n_layers, hidden_dim))

        else:
            # Unknown method, fall back to RBF
            weights = compute_grid_weights(v1, v2, existing_proj, sigma_nn)
            point_deltas = interpolate_deltas(weights, all_deltas_per_point, n_layers, hidden_dim)
            grid_deltas.append(point_deltas)
            if all_attn_deltas is not None:
                grid_attn_deltas.append(interpolate_deltas(weights, all_attn_deltas, n_layers, hidden_dim))
            if all_mlp_deltas is not None:
                grid_mlp_deltas.append(interpolate_deltas(weights, all_mlp_deltas, n_layers, hidden_dim))

    return grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas

# ============================================================
# 9b. PLUGGABLE INTERPOLATION METHODS
# ============================================================

def interpolate_rbf(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Gaussian RBF interpolation (original default)."""
    sigma = max(sigma, 1e-6)
    s2i = 1.0 / (2 * sigma ** 2)
    dists_sq = (query_x - source_x) ** 2 + (query_y - source_y) ** 2
    exponents = np.clip(-dists_sq * s2i, -500, 0)
    weights = np.exp(exponents)
    w_sum = weights.sum()
    if w_sum < 1e-30:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= w_sum
    vx = np.dot(weights, source_vals_x)
    vy = np.dot(weights, source_vals_y)
    return vx, vy


def interpolate_tps(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Thin Plate Spline interpolation.
    Uses the TPS kernel phi(r) = r^2 * log(r) with a linear polynomial term.
    Falls back to RBF if the system is singular."""
    n = len(source_x)
    if n < 3:
        return interpolate_rbf(query_x, query_y, source_x, source_y,
                               source_vals_x, source_vals_y, sigma)
    try:
        # Build TPS kernel matrix K
        pts = np.stack([source_x, source_y], axis=1)  # (n, 2)
        D = cdist(pts, pts, metric='euclidean')
        D = np.clip(D, 1e-12, None)
        K = D ** 2 * np.log(D)
        np.fill_diagonal(K, 0.0)

        # Build the full TPS system: [K P; P^T 0] [w; a] = [v; 0]
        P = np.column_stack([np.ones(n), source_x, source_y])  # (n, 3)
        Z = np.zeros((3, 3))

        # Regularization for stability
        reg = 1e-6 * np.eye(n)
        top = np.hstack([K + reg, P])
        bot = np.hstack([P.T, Z])
        A = np.vstack([top, bot])

        # Solve for x-component
        rhs_x = np.concatenate([source_vals_x, np.zeros(3)])
        coeffs_x = np.linalg.solve(A, rhs_x)

        # Solve for y-component
        rhs_y = np.concatenate([source_vals_y, np.zeros(3)])
        coeffs_y = np.linalg.solve(A, rhs_y)

        # Evaluate at query point
        qpt = np.array([[query_x, query_y]])
        dq = cdist(qpt, pts, metric='euclidean').flatten()
        dq = np.clip(dq, 1e-12, None)
        kq = dq ** 2 * np.log(dq)

        vx = np.dot(kq, coeffs_x[:n]) + coeffs_x[n] + coeffs_x[n+1]*query_x + coeffs_x[n+2]*query_y
        vy = np.dot(kq, coeffs_y[:n]) + coeffs_y[n] + coeffs_y[n+1]*query_x + coeffs_y[n+2]*query_y
        return float(vx), float(vy)
    except (np.linalg.LinAlgError, ValueError):
        return interpolate_rbf(query_x, query_y, source_x, source_y,
                               source_vals_x, source_vals_y, sigma)


def interpolate_idw(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Inverse Distance Weighting (Shepard's method) with power p=2."""
    p = 2.0
    dists = np.sqrt((query_x - source_x) ** 2 + (query_y - source_y) ** 2)
    dists = np.clip(dists, 1e-12, None)
    weights = 1.0 / (dists ** p)
    w_sum = weights.sum()
    if w_sum < 1e-30:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights /= w_sum
    vx = np.dot(weights, source_vals_x)
    vy = np.dot(weights, source_vals_y)
    return float(vx), float(vy)


def interpolate_mls(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Moving Least Squares with local linear basis.
    Fits a local linear model weighted by Gaussian distance."""
    sigma = max(sigma, 1e-6)
    s2i = 1.0 / (2 * sigma ** 2)
    dists_sq = (query_x - source_x) ** 2 + (query_y - source_y) ** 2
    weights = np.exp(np.clip(-dists_sq * s2i, -500, 0))

    n = len(source_x)
    if n < 3:
        return interpolate_rbf(query_x, query_y, source_x, source_y,
                               source_vals_x, source_vals_y, sigma)
    try:
        # Build weighted least squares: f(x,y) = a0 + a1*x + a2*y
        W = np.diag(weights)
        A = np.column_stack([np.ones(n), source_x - query_x, source_y - query_y])
        WA = W @ A
        ATWA = A.T @ WA

        # Regularize
        ATWA += 1e-8 * np.eye(3)

        # Solve for x-component
        coeffs_x = np.linalg.solve(ATWA, A.T @ (W @ source_vals_x))
        vx = coeffs_x[0]  # evaluated at (0,0) in local coords = query point

        # Solve for y-component
        coeffs_y = np.linalg.solve(ATWA, A.T @ (W @ source_vals_y))
        vy = coeffs_y[0]

        return float(vx), float(vy)
    except (np.linalg.LinAlgError, ValueError):
        return interpolate_rbf(query_x, query_y, source_x, source_y,
                               source_vals_x, source_vals_y, sigma)


def interpolate_nn(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Nearest Neighbor interpolation (simplest possible)."""
    dists_sq = (query_x - source_x) ** 2 + (query_y - source_y) ** 2
    idx = np.argmin(dists_sq)
    return float(source_vals_x[idx]), float(source_vals_y[idx])


def interpolate_wendland(query_x, query_y, source_x, source_y, source_vals_x, source_vals_y, sigma):
    """Compactly Supported RBF using Wendland C2 kernel.
    phi(r) = (1 - r/R)^4 * (4r/R + 1) for r < R, else 0.
    R = 3 * sigma (compact support radius)."""
    R = max(3.0 * sigma, 1e-6)
    dists = np.sqrt((query_x - source_x) ** 2 + (query_y - source_y) ** 2)
    r_norm = dists / R
    # Wendland C2 kernel
    mask = r_norm < 1.0
    weights = np.zeros_like(dists)
    r_clipped = np.clip(1.0 - r_norm[mask], 0, 1)
    weights[mask] = r_clipped ** 4 * (4.0 * r_norm[mask] + 1.0)

    w_sum = weights.sum()
    if w_sum < 1e-30:
        # Fall back to RBF if all weights are zero (query too far from all sources)
        return interpolate_rbf(query_x, query_y, source_x, source_y,
                               source_vals_x, source_vals_y, sigma)
    weights /= w_sum
    vx = np.dot(weights, source_vals_x)
    vy = np.dot(weights, source_vals_y)
    return float(vx), float(vy)


# Registry of available interpolation methods
INTERPOLATION_METHODS = {
    'rbf': interpolate_rbf,
    'tps': interpolate_tps,
    'idw': interpolate_idw,
    'mls': interpolate_mls,
    'nn': interpolate_nn,
    'wendland': interpolate_wendland,
}

def get_interpolation_fn(method_name):
    """Return the interpolation function for the given method name."""
    return INTERPOLATION_METHODS.get(method_name, interpolate_rbf)

# ============================================================
# 10. OUTPUT ASSEMBLY
# ============================================================

def build_fixed_pos(all_layer0):
    result = []
    for v in all_layer0:
        v_safe = np.nan_to_num(np.array(v, dtype=np.float64), ...)
        result.append(v_safe.tolist())
    return result

def build_deltas_array(all_deltas_per_point, n_layers, n_points):
    """Reshape per-point deltas into per-layer arrays for JSON, sanitizing NaN/Inf."""
    deltas = []
    for lay in range(n_layers):
        layer_d = []
        for p in range(n_points):
            v = np.array(all_deltas_per_point[p][lay], dtype=np.float64)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            layer_d.append(v.tolist())
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

def process_text(text, model_name=None, itp_method='rbf'):
    global TOKENIZER, MODEL, MODEL_NAME, MODEL_CONFIG

    # Switch model if requested
    if model_name and model_name != MODEL_NAME:
        load_model(model_name)

    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    n_layers = get_n_layers(MODEL_CONFIG)

    # Tokenize real input ONLY — no probe sentences
    real_ids, tokens_clean = tokenize_text(TOKENIZER, text)
    n_real = real_ids.shape[1]
    print(f"[Model] Tokens ({n_real}): {tokens_clean}")
    print(f"[Model] Interpolation method: {itp_method}")

    # Run ONLY the real sequence through the model (with component decomposition)
    all_layer0, all_deltas_per_point, all_attn_deltas, all_mlp_deltas = \
        run_all_sequences_with_components(MODEL, [real_ids], n_layers, hidden_dim)

    decomposition_available = all_attn_deltas is not None and all_mlp_deltas is not None
    if decomposition_available:
        print("[Model] Component decomposition: OK (attention + MLP deltas captured)")
    else:
        print("[Model] Component decomposition: UNAVAILABLE for this architecture")

    all_labels = list(tokens_clean)
    all_is_real = [True] * n_real

    # ================================================================
    # NEW: Embed predicted next tokens and add them as source points
    # ================================================================
    print("[Model] Embedding predicted next tokens...")
    pred_layer0, pred_deltas, pred_labels, pred_probs, pred_token_ids = \
        embed_predicted_tokens(
            TOKENIZER, MODEL, LM_MODEL, real_ids, MODEL_CONFIG, k=default_k
        )

    n_predicted = len(pred_layer0)
    is_predicted = []  # track which points are predictions

    for pi in range(n_predicted):
        all_layer0.append(pred_layer0[pi])
        all_deltas_per_point.append(pred_deltas[pi])
        if decomposition_available:
            # For predicted tokens, we don't have attn/mlp decomposition
            # (would require hooking the extended sequence separately)
            # Use the full delta as both attn and mlp placeholder
            all_attn_deltas.append([np.zeros(hidden_dim) for _ in range(n_layers)])
            all_mlp_deltas.append([np.zeros(hidden_dim) for _ in range(n_layers)])
        all_labels.append(pred_labels[pi])
        all_is_real.append(False)
        is_predicted.append(True)

    n_after_predicted = len(all_layer0)
    print(f"[Model] {n_real} real + {n_predicted} predicted = {n_after_predicted} source points for interpolation")

    len(all_layer0)

    # Compute neighbors among real tokens only
    real_embeddings = np.stack(all_layer0[:n_real], axis=0)
    all_embeddings = np.stack(all_layer0, axis=0)
    neighbors = compute_neighbors(real_embeddings, all_embeddings, all_labels, all_is_real, k=10)

    # Predict next token (for display — we already have the data)
    print("[Model] Formatting next token predictions...")
    next_token_preds = [
        {"token": pred_labels[i].lstrip("→"), "prob": pred_probs[i]}
        for i in range(n_predicted)
    ]

    # Find vocabulary neighbors for each real token
    print("[Model] Finding vocabulary neighbors...")
    vocab_neighbors = find_vocab_neighbors(TOKENIZER, MODEL, all_layer0[:n_real], n_real, k=5)

    # PCA on ALL source points (real + predicted) for grid probe generation
    # This ensures the grid covers the predicted token region too
    print("[Model] Creating systematic grid probes around real + predicted points...")
    layer0_mat = np.stack(all_layer0, axis=0)

    if n_real < 2:
        print("[Model] WARNING: fewer than 2 real tokens, grid probes will be trivial")

    centroid, centered, pc1, pc2, proj1, proj2 = compute_pca_basis(layer0_mat, hidden_dim)
    existing_proj = np.stack([proj1, proj2], axis=1)

    # Grid probes now interpolate from BOTH real tokens AND predicted tokens
    # This means the diffeomorphism field incorporates predicted-token geometry
    grid_layer0, grid_deltas, grid_attn_deltas, grid_mlp_deltas = create_grid_probes(
        centroid, pc1, pc2, proj1, proj2, existing_proj,
        all_deltas_per_point, n_layers, hidden_dim,
        n_side=10, pad_frac=0.3,
        all_attn_deltas=all_attn_deltas,
        all_mlp_deltas=all_mlp_deltas,
        itp_method=itp_method,
    )

    n_grid = len(grid_layer0)
    print(f"[Model] Added {n_grid} systematic grid probes (itp={itp_method})")

    # Append grid probes as synthetic points
    for gi in range(n_grid):
        all_layer0.append(grid_layer0[gi])
        all_deltas_per_point.append(grid_deltas[gi])
        if decomposition_available:
            all_attn_deltas.append(grid_attn_deltas[gi])
            all_mlp_deltas.append(grid_mlp_deltas[gi])
        all_labels.append("\u00b7")
        all_is_real.append(False)

    n_total_final = len(all_layer0)

    # Compute strain statistics (on real tokens only)
    print("[Model] Computing strain statistics...")
    strain_stats = compute_strain_stats(all_layer0, all_deltas_per_point, n_layers, n_real, hidden_dim)
    most_active_layer = max(range(n_layers), key=lambda lay: strain_stats[lay]["variance"])
    print(f"[Model] Most active layer (by strain variance): {most_active_layer}")

    # Build output
    fixed_pos = build_fixed_pos(all_layer0)
    deltas = build_deltas_array(all_deltas_per_point, n_layers, n_total_final)

    attn_deltas_json = None
    mlp_deltas_json = None
    if decomposition_available:
        attn_deltas_json = build_deltas_array(all_attn_deltas, n_layers, n_total_final)
        mlp_deltas_json = build_deltas_array(all_mlp_deltas, n_layers, n_total_final)

    data_raw_activations = []

    for i in range(n_total_final):
        vec = np.array(all_layer0[i], dtype=np.float64)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        data_raw_activations.append(vec.tolist())

    data = build_output_data(
        all_labels, all_is_real, n_layers, n_total_final, n_real,
        hidden_dim, fixed_pos, deltas, MODEL_NAME, text, neighbors,
        next_token_preds=next_token_preds,
        vocab_neighbors=vocab_neighbors,
        attn_deltas=attn_deltas_json,
        mlp_deltas=mlp_deltas_json,
        strain_stats=strain_stats,
    )

    data["raw_activations"] = data_raw_activations

    # Include metadata about predicted points
    data["itp_method"] = itp_method
    data["n_predicted"] = n_predicted
    data["predicted_indices"] = list(range(n_real, n_real + n_predicted))
    data["predicted_probs"] = pred_probs

    json_str = json.dumps(data, cls=SafeFloatEncoder)
    print(f"[Model] JSON: {len(json_str)/1024/1024:.1f} MB")
    return json_str

# ============================================================
# 12. HTML PAGE WITH 2D/3D TOGGLE + DECOMPOSITION + STRAIN STATS
# ============================================================

script_js_file_content = Path("scripts.js").read_text(encoding="utf-8")

HTML_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Metric Space Explorer</title>
<style>
#curvature-singularities::-webkit-scrollbar{width:4px}
#curvature-singularities::-webkit-scrollbar-track{background:#0a0a1a}
#curvature-singularities::-webkit-scrollbar-thumb{background:#0f3460;border-radius:2px}
#curvature-panel .cr{display:flex;align-items:center;gap:5px;font-size:11px}
#curvature-panel .cr label{min-width:80px;text-align:right;color:#a0a0c0}
#curvature-panel .cr input[type=range]{flex:1;accent-color:#7b68ee}
#curvature-panel .cr .v{min-width:30px;color:#7b68ee;font-weight:bold;font-size:10px}
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',sans-serif;display:flex;height:100vh;overflow:hidden}
#side{width:370px;min-width:370px;background:#16213e;padding:12px;
  overflow-y:auto;border-right:2px solid #0f3460;
  display:flex;flex-direction:column;gap:6px}
#neighbor-panel{background:#0f3460;padding:6px;border-radius:4px;font-size:10px;
  line-height:1.5;max-height:300px;overflow-y:auto;display:none;flex-shrink: 0;}
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
  line-height:1.3;max-height:280px;overflow-y:auto;display:none;flex-shrink: 0;}
#strain-stats-panel table{width:100%;border-collapse:collapse}
#strain-stats-panel th{color:#53a8b6;text-align:left;padding:2px 3px;border-bottom:1px solid #1a1a2e;font-size:8px;position:sticky;top:0;background:#0f3460}
#strain-stats-panel td{padding:2px 3px;border-bottom:1px solid rgba(255,255,255,0.05)}
#strain-stats-panel tr.current-layer{background:rgba(233,69,96,0.2)}
#strain-stats-panel tr.most-active{background:rgba(83,168,182,0.15)}
#strain-stats-panel tr:hover{background:rgba(255,255,255,0.05)}
/* ---- Compare Mode ---- */
#compare-area{display:none;margin-top:4px}
#compare-area .cr{margin-bottom:3px}
#txt-b{flex:1;background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:6px;font-size:12px;border-radius:4px;font-family:monospace}
#btn-compare{background:#53a8b6;color:#fff;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold}
#btn-compare:hover{background:#3d8a96}
#btn-compare:disabled{background:#555;cursor:wait}
#compare-panel{background:#0a0a1a;padding:6px;border-radius:4px;max-height:700px;overflow:auto;display:none}
#compare-summary{background:#0f3460;padding:8px;border-radius:4px;font-size:10px;line-height:1.5;display:none}
.compare-row{display:flex;gap:2px;margin-bottom:8px;align-items:flex-start}
.compare-col{text-align:center}
.compare-col canvas{border:1px solid #0f3460;image-rendering:pixelated}
.compare-label{font-size:8px;margin-bottom:1px}
.divergence-bar{height:4px;border-radius:2px;margin-top:2px}
.onset-marker{color:#f5a623;font-weight:bold}
#compare-divergence-chart{margin:8px 0}
#compare-divergence-chart canvas{border:1px solid #0f3460;border-radius:4px}
</style></head><body>
<div id="side">

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: HEADER & STATUS                        -->
<!-- ═══════════════════════════════════════════════ -->
<h2>Metric Space Explorer</h2>
<div id="status">Enter text and click Run</div>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: INPUT                                  -->
<!-- ═══════════════════════════════════════════════ -->
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
    <option value="EleutherAI/pythia-1.4b">Pythia 1.4B</option>
    <option value="EleutherAI/pythia-2.8b">Pythia 2.8B</option>
    <option value="facebook/opt-1.3b">OPT 1.3B</option>
    <option value="microsoft/phi-2">Phi-2 (2.7B)</option>
    <option value="microsoft/deberta-v3-large">DeBERTa-v3 Large</option>
  </select>
</div>
<div id="text-area">
  <input id="txt-in" type="text" placeholder="Enter text..." value="The quick brown fox jumps over the lazy dog">
  <button id="btn-run" onclick="runText()">Run</button>
</div>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: MODEL INFO                             -->
<!-- ═══════════════════════════════════════════════ -->
<div id="info">
  Model: <span id="i-mod">-</span> |
  Points: <span id="i-pts">-</span> (<span id="i-real">-</span> real + <span id="i-syn">-</span> probes)<br>
  Layers: <span id="i-lay">-</span> | Dim: <span id="i-dim">-</span> |
  Tokens: <span id="i-tok">-</span> |
  ITP: <span id="i-itp">rbf</span>
</div>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: VIEW & NAVIGATION                      -->
<!-- ═══════════════════════════════════════════════ -->
<details open>
  <summary style="color:#e94560;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🗺️ View &amp; Navigation</summary>

  <h3 style="margin-top:6px">View Mode</h3>
  <div class="view-toggle">
    <button id="btn-fibre" onclick="setViewMode('fibre')">Fibre Bundle</button>
    <button id="btn-fibre3d" onclick="setViewMode('fibre3d')">Fibre 3D</button>
    <button id="btn-fibrekelp" onclick="setViewMode('fibrekelp')">Fibre Kelp</button>
    <button id="btn-2d" class="active" onclick="setViewMode('2d')">2D</button>
    <button id="btn-3d" onclick="setViewMode('3d')">3D</button>
    <button id="btn-multi-view" onclick="setViewMode('multi')" style="display:none">Multi Compare</button>
  </div>

  <h3>Layer &amp; Deformation</h3>
  <div class="cr"><label>Layer:</label><input type="range" id="sl-layer" min="0" max="11" value="0" step="1"><span class="v" id="v-layer">0</span></div>
  <div class="cr"><label>Deform t:</label><input type="range" id="sl-t" min="0" max="1" value="1.0" step="0.01"><span class="v" id="v-t">1.00</span></div>
  <div class="cr"><label>Amplify:</label><input type="range" id="sl-amp" min="0.1" max="500" value="1" step="0.1"><span class="v" id="v-amp">1.0</span></div>
  <div class="cr"><label>Mode:</label>
    <select id="sel-mode">
      <option value="cumfwd">Layers 0→L (Cumulative)</option>
      <option value="cumbwd">Layers L→End (Cumulative)</option>
      <option value="single">This Layer Only</option>
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

  <h3>Dimensions</h3>
  <div class="cr"><label>Dim X:</label><input type="range" id="sl-dx" min="0" max="767" value="0" step="1"><span class="v" id="v-dx">0</span></div>
  <div class="cr"><label>Dim Y:</label><input type="range" id="sl-dy" min="0" max="767" value="1" step="1"><span class="v" id="v-dy">1</span></div>
  <div class="cr" id="dz-row" style="display:none"><label>Dim Z:</label><input type="range" id="sl-dz" min="0" max="767" value="2" step="1"><span class="v" id="v-dz">2</span></div>
</details>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: DISPLAY OPTIONS                        -->
<!-- ═══════════════════════════════════════════════ -->
<details open>
  <summary style="color:#53a8b6;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🎨 Display Options</summary>

  <h3 style="margin-top:6px">Grid &amp; Interpolation</h3>
  <div class="cr"><label>Method:</label>
    <select id="sel-itp">
      <option value="rbf" selected>Gaussian RBF</option>
      <option value="tps">Thin Plate Spline</option>
      <option value="idw">Inverse Distance (Shepard)</option>
      <option value="mls">Moving Least Squares</option>
      <option value="nn">Nearest Neighbor</option>
      <option value="wendland">Wendland C2 (Compact)</option>
    </select>
  </div>
  <div class="cr"><label>Bandwidth σ:</label><input type="range" id="sl-sig" min="0.01" max="20" value="1.0" step="0.01"><span class="v" id="v-sig">1.00</span></div>
  <div class="cr"><label>Grid Res:</label><input type="range" id="sl-gr" min="10" max="80" value="30" step="1"><span class="v" id="v-gr">30</span></div>

  <h3>Visibility</h3>
  <div class="cb"><input type="checkbox" id="cb-grid" checked><label for="cb-grid">Deformed Grid</label></div>
  <div class="cb"><input type="checkbox" id="cb-heat" checked><label for="cb-heat">Strain Heatmap</label></div>
  <div class="cb"><input type="checkbox" id="cb-ref" checked><label for="cb-ref">Reference Grid</label></div>
  <div class="cb"><input type="checkbox" id="cb-tok" checked><label for="cb-tok">Real Tokens</label></div>
  <div class="cb"><input type="checkbox" id="cb-syn"><label for="cb-syn">Probe Points</label></div>
  <div class="cb"><input type="checkbox" id="cb-sc" checked><label for="cb-sc">Strain Color</label></div>
  <div class="cb"><input type="checkbox" id="cb-vec"><label for="cb-vec">Vector Arrows</label></div>
  <div class="cb"><input type="checkbox" id="cb-vocnb" checked><label for="cb-vocnb">Show Nearby Vocab Words</label></div>
</details>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: SELECTION & NEIGHBORS                  -->
<!-- ═══════════════════════════════════════════════ -->
<details open>
  <summary style="color:#f5a623;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🔍 Selection &amp; Neighbors</summary>

  <h3 style="margin-top:6px">Selected Tokens <span style="font-weight:normal;font-size:9px;color:#666">(click canvas)</span></h3>
  <div id="selected-tokens"><span style="color:#555;font-size:10px">Click on a token dot to select it</span></div>

  <div id="neighbor-panel">
    <h4 id="nb-title">Neighbors</h4>
    <div id="nb-list"></div>
  </div>

  <h3>Neighbor Tracing</h3>
  <div class="cr"><label>K Neighbors:</label><input type="range" id="sl-kn" min="1" max="20" value="5" step="1"><span class="v" id="v-kn">5</span></div>
  <div class="cb"><input type="checkbox" id="cb-nb" checked><label for="cb-nb">Show Neighbor Lines</label></div>
  <div class="cb"><input type="checkbox" id="cb-nblabel" checked><label for="cb-nblabel">Show Neighbor Labels</label></div>

  <h3>Predicted Next Token</h3>
  <div id="next-token-panel" style="background:#0f3460;padding:6px;border-radius:4px;font-size:11px;line-height:1.6">
    <span style="color:#555">Run a prompt to see predictions</span>
  </div>
</details>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: STRAIN STATISTICS                      -->
<!-- ═══════════════════════════════════════════════ -->
<details>
  <summary style="color:#2ecc71;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">📊 Strain Statistics</summary>
  <div id="strain-stats-panel" style="margin-top:6px"></div>
</details>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: ANALYSIS TOOLS                         -->
<!-- ═══════════════════════════════════════════════ -->

<!-- ---- Compare Mode ---- -->
<details>
  <summary style="color:#53a8b6;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">⚡ Compare Mode</summary>
  <div style="margin-top:6px">
    <div class="cb" style="margin-bottom:4px">
      <input type="checkbox" id="cb-compare" onchange="toggleCompareMode()">
      <label for="cb-compare" style="color:#53a8b6;font-weight:bold">Enable Compare Mode</label>
    </div>
    <div id="compare-area">
      <div style="font-size:10px;color:#888;margin-bottom:4px">
        Enter a second text to compare activations side-by-side.
        See where the model processes them differently.
      </div>
      <div style="display:flex;gap:4px;margin-bottom:4px">
        <input id="txt-b" type="text" placeholder="Second text (e.g. false version)..."
               value="The capital of France is Berlin">
        <button id="btn-compare" onclick="runCompare()">Compare</button>
      </div>
      <div id="compare-summary"></div>
      <div id="compare-divergence-chart"></div>
      <div id="compare-panel"></div>
    </div>
  </div>
</details>

<!-- ---- Multi-Sentence Compare ---- -->
<details>
  <summary style="color:#7b68ee;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🔬 Multi-Sentence Compare</summary>
  <div style="margin-top:6px">
    <div class="cb" style="margin-bottom:4px">
      <input type="checkbox" id="cb-multi" onchange="toggleMultiMode()">
      <label for="cb-multi" style="color:#7b68ee;font-weight:bold">Enable Multi-Compare</label>
    </div>
    <div id="multi-area" style="display:none">
      <div style="font-size:10px;color:#888;margin-bottom:4px">
        Enter multiple sentences (one per line). Each will be analyzed independently, then compared dimension-by-dimension.
      </div>
      <textarea id="multi-txt" rows="6" style="width:100%;background:#0d1117;color:#e0e0e0;border:1px solid #0f3460;padding:6px;font-size:11px;border-radius:4px;font-family:monospace;resize:vertical"
>The capital of France is Paris
The capital of France is Berlin
The capital of Germany is Berlin</textarea>
      <button id="btn-multi" onclick="runMulti()" style="margin-top:4px;width:100%;background:#7b68ee;color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold">
        Compare Sentences
      </button>
      <div id="multi-status" style="margin-top:4px;font-size:9px;color:#555"></div>
      <div id="multi-summary" style="display:none;margin-top:6px;background:#0f3460;padding:8px;border-radius:4px;font-size:10px;line-height:1.5"></div>
      <div id="multi-layer-select" style="display:none;margin-top:4px">
        <div class="cr">
          <label>Layer:</label>
          <select id="multi-layer" onchange="renderMultiLayer();if(viewMode==='multi')drawMultiCanvas()" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px"></select>
        </div>
        <div class="cr" style="margin-top:3px">
          <label>Show:</label>
          <select id="multi-show" onchange="renderMultiLayer();if(viewMode==='multi')drawMultiCanvas()" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
            <option value="most">Most Different Dims</option>
            <option value="least">Least Different Dims</option>
          </select>
        </div>
        <div class="cr" style="margin-top:3px">
          <label>Top K:</label>
          <input type="range" id="multi-topk" min="5" max="50" value="20" step="1" oninput="document.getElementById('v-multi-topk').textContent=this.value;renderMultiLayer();if(viewMode==='multi')drawMultiCanvas()">
          <span class="v" id="v-multi-topk">20</span>
        </div>
      </div>
      <div id="multi-dim-chart" style="display:none;margin-top:6px">
        <canvas id="multi-dim-cv" width="340" height="300" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
      </div>
      <div id="multi-pairwise" style="display:none;margin-top:6px">
        <div style="color:#53a8b6;font-size:9px;font-weight:bold;margin-bottom:2px">Pairwise Cosine Similarity</div>
        <canvas id="multi-pair-cv" width="200" height="200" style="border:1px solid #0f3460;border-radius:4px;display:block"></canvas>
      </div>
      <div id="multi-layer-profile" style="display:none;margin-top:6px">
        <div style="color:#e94560;font-size:9px;font-weight:bold;margin-bottom:2px">Layer Divergence Profile</div>
        <canvas id="multi-layerprof-cv" width="340" height="80" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
      </div>
    </div>
  </div>
</details>

<!-- ---- Neuron Activation Grid ---- -->
<details>
  <summary style="color:#53a8b6;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🧠 Neuron Activation Grid</summary>
  <div style="margin-top:6px">
    <div id="neuron-grid-controls" style="margin-bottom:4px">
      <div class="cr">
        <label>Norm:</label>
        <select id="ng-norm" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
          <option value="layer">Per-Layer (relative)</option>
          <option value="global">Global (absolute)</option>
        </select>
      </div>
      <div class="cr" style="margin-top:3px">
        <label>Pixel size:</label>
        <input type="range" id="ng-pixsize" min="1" max="6" value="2" step="1">
        <span class="v" id="v-ng-pixsize">2</span>
      </div>
      <div class="cb" style="margin-top:3px">
        <input type="checkbox" id="ng-absval"><label for="ng-absval">Use |activation| (absolute value)</label>
      </div>
      <button onclick="fetchNeuronGrid()" style="margin-top:4px;background:#53a8b6;color:#fff;border:none;padding:4px 12px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold;width:100%">
        Load Neuron Grid
      </button>
    </div>
    <div id="neuron-grid-panel" style="background:#0a0a1a;padding:6px;border-radius:4px;max-height:500px;overflow:auto;display:none">
    </div>
  </div>
</details>

<!-- ---- SAE Feature Inspector ---- -->
<details>
  <summary style="color:#2ecc71;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🔎 SAE Feature Inspector</summary>
  <div id="sae-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px;margin-top:6px">
    <div id="sae-status" style="color:#555;margin-bottom:6px">Loading SAE info...</div>
    <div id="sae-controls" style="display:none">
      <div class="cr">
        <label>Layer:</label>
        <select id="sae-layer" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px"></select>
      </div>
      <div class="cr" style="margin-top:4px">
        <label>Token:</label>
        <select id="sae-token" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
          <option value="0">Run text first</option>
        </select>
      </div>
      <div class="cr" style="margin-top:4px">
        <label>Top K:</label>
        <input type="range" id="sae-topk" min="5" max="50" value="20" step="1">
        <span class="v" id="v-sae-topk">20</span>
      </div>
      <button onclick="fetchSAEFeatures()" style="margin-top:6px;background:#53a8b6;color:#fff;border:none;padding:4px 12px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold;width:100%">Inspect Features</button>
      <div id="sae-features-list" style="margin-top:8px;max-height:250px;overflow-y:auto"></div>
      <div id="sae-intervention" style="display:none;margin-top:8px;border-top:1px solid #1a1a2e;padding-top:6px">
        <div style="color:#e94560;font-weight:bold;margin-bottom:4px">⚡ Intervention</div>
        <div class="cr">
          <label>Feature:</label>
          <input type="number" id="sae-int-feature" min="0" value="0" style="width:70px;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
        </div>
        <div class="cr" style="margin-top:3px">
          <label>Clamp to:</label>
          <input type="range" id="sae-int-clamp" min="-10" max="50" value="0" step="0.5" style="flex:1">
          <span class="v" id="v-sae-clamp">0.0</span>
        </div>
        <div style="display:flex;gap:4px;margin-top:4px">
          <button onclick="runSAEIntervention()" style="flex:1;background:#e94560;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold">Apply</button>
          <button onclick="clearSAEIntervention()" style="flex:1;background:#555;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:10px">Clear</button>
        </div>
        <div id="sae-int-results" style="margin-top:6px"></div>
      </div>
    </div>
  </div>
</details>

<!-- ---- Diffeomorphism Spectrum ---- -->
<details>
  <summary style="color:#f5a623;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🌊 Diffeomorphism Spectrum</summary>
  <div id="spectrum-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px;margin-top:6px">
    <div style="display:flex;gap:4px;margin-bottom:6px">
      <button onclick="fetchDiffeoSpectrum()" style="flex:1;background:#53a8b6;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold">
        Analyze Spectrum
      </button>
      <button onclick="fetchDiffeoSpectrum(document.getElementById('txt-b').value)" style="flex:1;background:#f5a623;color:#fff;border:none;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold">
        Compare Spectra
      </button>
    </div>
    <div id="spectrum-results" style="max-height:350px;overflow-y:auto"></div>
  </div>
</details>

<!-- ---- Contrastive Feature Scanner ---- -->
<details>
  <summary style="color:#e94560;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🎯 Contrastive Feature Scanner</summary>
  <div id="contrastive-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px;margin-top:6px">
    <div style="color:#888;font-size:9px;margin-bottom:6px">
      Automatically find which geometric operations distinguish behaviors.
      Click a behavior to scan:
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px">
      <button onclick="runContrastiveScan('math')" style="background:#e94560;color:#fff;border:none;padding:3px 10px;border-radius:10px;cursor:pointer;font-size:9px">🔢 Math</button>
      <button onclick="runContrastiveScan('refusal')" style="background:#7b68ee;color:#fff;border:none;padding:3px 10px;border-radius:10px;cursor:pointer;font-size:9px">🚫 Refusal</button>
      <button onclick="runContrastiveScan('code')" style="background:#2ecc71;color:#fff;border:none;padding:3px 10px;border-radius:10px;cursor:pointer;font-size:9px">💻 Code</button>
      <button onclick="runContrastiveScan('reasoning')" style="background:#f5a623;color:#fff;border:none;padding:3px 10px;border-radius:10px;cursor:pointer;font-size:9px">🧠 Reasoning</button>
    </div>
    <div id="contrastive-results" style="max-height:400px;overflow-y:auto"></div>
  </div>
</details>

<!-- ---- Holographic Curvature Analysis ---- -->
<details>
  <summary style="color:#7b68ee;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🌀 Holographic Curvature Analysis</summary>
  <div id="curvature-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px;margin-top:6px">
    <div style="color:#888;font-size:9px;margin-bottom:6px">
      Decode Ricci &amp; Sectional curvature from the fiber bundle structure.
      Identifies curvature singularities, syntactic junctions, entropy collapses,
      and gravitational sources.
    </div>
    <div class="cr">
      <label>k-NN:</label>
      <input type="range" id="curv-k" min="3" max="20" value="8" step="1">
      <span class="v" id="v-curv-k">8</span>
    </div>
    <div class="cr" style="margin-top:3px">
      <label>PCA d:</label>
      <input type="range" id="curv-d" min="4" max="64" value="16" step="1">
      <span class="v" id="v-curv-d">16</span>
    </div>
    <div class="cr" style="margin-top:3px">
      <label>Top K sing.:</label>
      <input type="range" id="curv-topk" min="3" max="25" value="10" step="1">
      <span class="v" id="v-curv-topk">10</span>
    </div>
    <div style="display:flex;gap:4px;margin-top:6px">
      <button id="btn-curvature" onclick="runCurvatureAnalysis()" style="flex:1;background:#7b68ee;color:#fff;border:none;padding:5px 10px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold">
      Analyze Curvature
    </button>
  </div>
  <div id="curvature-status" style="margin-top:6px;color:#555;font-size:9px">Run text first, then analyze curvature.</div>

  <!-- Correlation summary -->
  <div id="curvature-correlation-summary" style="display:none;margin-top:8px;background:#0a0a1a;padding:6px;border-radius:4px;font-size:9px;line-height:1.5;color:#a0a0c0"></div>

  <!-- Singularities list -->
  <div id="curvature-singularities" style="display:none;margin-top:8px;max-height:350px;overflow-y:auto"></div>

  <!-- Curvature heatmap selector -->
  <div id="curvature-heatmap-controls" style="display:none;margin-top:8px">
    <div class="cr">
      <label>Heatmap:</label>
      <select id="curv-heatmap-type" onchange="renderCurvatureHeatmap()" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
        <option value="ollivier_ricci">Ollivier-Ricci Curvature</option>
        <option value="scalar_curvature">Scalar Curvature (Vol. Strain)</option>
        <option value="sectional_curvature">Sectional Curvature</option>
        <option value="procrustes_deviation">Procrustes Deviation</option>
        <option value="metric_log_det">log det(g)</option>
      </select>
    </div>
    <canvas id="curv-heatmap-cv" width="340" height="160" style="margin-top:4px;border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
    <div id="curv-heatmap-legend" style="display:flex;justify-content:space-between;font-size:8px;color:#888;margin-top:2px">
      <span id="curv-hm-min">min</span>
      <span style="color:#a0a0c0" id="curv-hm-title">—</span>
      <span id="curv-hm-max">max</span>
    </div>
  </div>

  <!-- Surprisal correlation chart -->
  <div id="curvature-surprisal-chart" style="display:none;margin-top:8px">
    <div style="color:#2ecc71;font-weight:bold;font-size:10px;margin-bottom:3px">Metric det(g) vs Surprisal</div>
    <canvas id="curv-surprisal-cv" width="340" height="140" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
    <div id="curv-corr-bars" style="margin-top:4px"></div>
  </div>
</details>

<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: KEYBOARD SHORTCUTS                     -->
<!-- ═══════════════════════════════════════════════ -->
<details>
  <summary style="color:#888;font-size:11px;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">⌨️ Keyboard Shortcuts</summary>
  <div id="keys" style="margin-top:4px;font-size:8px;color:#555;line-height:1.3">
    <b>Keys:</b> ←→ Dim X | ↑↓ Dim Y | Shift+←→ Dim Z (±1) | Shift+↑↓ Dim Z (±10) | PgUp/PgDn Dim Z (3D) | [/] Layer | ;/' t | A/Z Amp | Space Auto | R Reset | D Next Dim Pair | 0 Reset Zoom<br>
    <b>Mouse:</b> Scroll=Zoom | Shift+Drag=Pan | Click=Select Token | Drag=Rotate (3D)
  </div>
</details>
<!-- ---- Persistent Homology / TDA ---- -->
<details>
  <summary style="color:#00bcd4;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🕸️ Persistent Homology (TDA)</summary>
  <div id="tda-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px;margin-top:6px">
    <div style="color:#888;font-size:9px;margin-bottom:6px">
      Track topological features (connected components, loops, voids) in the
      representation space as they appear and disappear across layers.
    </div>
    <div class="cr">
      <label>Max dim:</label>
      <select id="tda-maxdim" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
        <option value="1">H0 + H1 (components + loops)</option>
        <option value="2" selected>H0 + H1 + H2 (+ voids)</option>
      </select>
    </div>
    <div class="cr" style="margin-top:3px">
      <label>PCA dims:</label>
      <input type="range" id="tda-pca" min="3" max="64" value="16" step="1">
      <span class="v" id="v-tda-pca">16</span>
    </div>
    <div class="cr" style="margin-top:3px">
      <label>Max edge:</label>
      <input type="range" id="tda-maxedge" min="0.5" max="50" value="10" step="0.5">
      <span class="v" id="v-tda-maxedge">10.0</span>
    </div>
    <div class="cr" style="margin-top:3px">
      <label>Collapse ε:</label>
      <input type="range" id="tda-collapse" min="0" max="5" value="0" step="0.1">
      <span class="v" id="v-tda-collapse">0.0</span>
    </div>
    <div style="display:flex;gap:4px;margin-top:6px">
      <button id="btn-tda" onclick="runTDA()" style="flex:1;background:#00bcd4;color:#fff;border:none;padding:5px 10px;border-radius:3px;cursor:pointer;font-size:10px;font-weight:bold">
        Compute Persistent Homology
      </button>
    </div>
    <div id="tda-status" style="margin-top:6px;color:#555;font-size:9px">Run text first, then compute homology.</div>

    <!-- Summary -->
    <div id="tda-summary" style="display:none;margin-top:8px;background:#0a0a1a;padding:6px;border-radius:4px;font-size:9px;line-height:1.5;color:#a0a0c0"></div>

    <!-- Layer selector for diagrams -->
    <div id="tda-layer-controls" style="display:none;margin-top:8px">
      <div class="cr">
        <label>Layer:</label>
        <select id="tda-layer-sel" onchange="renderTDALayer()" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px"></select>
      </div>
      <div class="cr" style="margin-top:3px">
        <label>View:</label>
        <select id="tda-view-sel" onchange="renderTDALayer()" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
          <option value="persistence">Persistence Diagram</option>
          <option value="barcode">Barcode</option>
          <option value="landscape">Persistence Landscape</option>
          <option value="betti">Betti Curve</option>
        </select>
      </div>
    </div>

    <!-- Main persistence diagram canvas -->
    <div id="tda-diagram-wrap" style="display:none;margin-top:6px">
      <canvas id="tda-diagram-cv" width="340" height="300" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
      <div id="tda-diagram-legend" style="display:flex;justify-content:space-between;font-size:8px;color:#888;margin-top:2px">
        <span>Birth →</span>
        <span id="tda-diagram-title">Persistence Diagram</span>
        <span>← Death</span>
      </div>
    </div>

    <!-- Cross-layer Betti number evolution -->
    <div id="tda-betti-evolution" style="display:none;margin-top:8px">
      <div style="color:#00bcd4;font-weight:bold;font-size:10px;margin-bottom:3px">Betti Numbers Across Layers</div>
      <canvas id="tda-betti-cv" width="340" height="120" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
    </div>

    <!-- Topological events timeline -->
    <div id="tda-events" style="display:none;margin-top:8px;max-height:250px;overflow-y:auto"></div>

    <!-- Wasserstein distance heatmap between layers -->
    <div id="tda-wasserstein-wrap" style="display:none;margin-top:8px">
      <div style="color:#f5a623;font-weight:bold;font-size:10px;margin-bottom:3px">Wasserstein Distance Between Layers</div>
      <canvas id="tda-wasserstein-cv" width="340" height="200" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
    </div>

    <!-- Persistence entropy -->
    <div id="tda-entropy-wrap" style="display:none;margin-top:8px">
      <div style="color:#2ecc71;font-weight:bold;font-size:10px;margin-bottom:3px">Persistence Entropy Across Layers</div>
      <canvas id="tda-entropy-cv" width="340" height="80" style="border:1px solid #0f3460;border-radius:4px;display:block;width:100%"></canvas>
    </div>
  </div>
</details>
<!-- ═══════════════════════════════════════════════ -->
<!-- SECTION: PURE MORPHING ANALYSIS                 -->
<!-- ═══════════════════════════════════════════════ -->
<details>
  <summary style="color:#fd79a8;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">🌊 Pure Morphing (Transformation Space)</summary>

  <div id="morphing-panel" style="margin-top:6px">
    <div style="color:#888;font-size:9px;margin-bottom:6px;line-height:1.4">
      Extract and visualize the <b>morphings themselves</b> — the diffeomorphisms, Jacobians,
      eigenvalue flows, and connection forms — stripped of the data they act on.
    </div>

    <!-- Controls -->
    <div class="cr"><label>PCA dims:</label>
      <input type="range" id="morph-pca-d" min="4" max="64" value="16" step="2">
      <span class="v" id="v-morph-pca-d">16</span>
    </div>
    <div class="cr"><label>K neighbors:</label>
      <input type="range" id="morph-k" min="3" max="20" value="8" step="1">
      <span class="v" id="v-morph-k">8</span>
    </div>

    <div style="display:flex;gap:4px;margin:6px 0;flex-wrap:wrap">
      <button id="btn-morphing" onclick="runMorphingAnalysis()"
        style="background:#fd79a8;color:#fff;border:none;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:10px;font-weight:bold">
        Extract Morphings
      </button>
      <button onclick="toggleMorphingOverlay()"
        style="background:#0f3460;color:#a0a0c0;border:1px solid #0f3460;padding:5px 8px;border-radius:4px;cursor:pointer;font-size:10px">
        Toggle Overlay
      </button>
    </div>

    <div id="morphing-status" style="font-size:9px;color:#888;margin-bottom:4px"></div>

    <!-- Sub-view selector -->
    <div id="morphing-views" style="display:none">
      <div style="display:flex;gap:2px;margin-bottom:6px;flex-wrap:wrap">
        <button class="morph-tab active" data-tab="eigenflow" onclick="setMorphTab('eigenflow')"
          style="flex:1;padding:4px 6px;border:1px solid #0f3460;background:#fd79a8;color:#fff;cursor:pointer;font-size:9px;border-radius:3px;min-width:60px">
          Eigenflow
        </button>
        <button class="morph-tab" data-tab="jacobian" onclick="setMorphTab('jacobian')"
          style="flex:1;padding:4px 6px;border:1px solid #0f3460;background:#1a1a2e;color:#a0a0c0;cursor:pointer;font-size:9px;border-radius:3px;min-width:60px">
          Jacobian
        </button>
        <button class="morph-tab" data-tab="holonomy" onclick="setMorphTab('holonomy')"
          style="flex:1;padding:4px 6px;border:1px solid #0f3460;background:#1a1a2e;color:#a0a0c0;cursor:pointer;font-size:9px;border-radius:3px;min-width:60px">
          Holonomy
        </button>
        <button class="morph-tab" data-tab="connection" onclick="setMorphTab('connection')"
          style="flex:1;padding:4px 6px;border:1px solid #0f3460;background:#1a1a2e;color:#a0a0c0;cursor:pointer;font-size:9px;border-radius:3px;min-width:60px">
          Connection
        </button>
        <button class="morph-tab" data-tab="svwaterfall" onclick="setMorphTab('svwaterfall')"
          style="flex:1;padding:4px 6px;border:1px solid #0f3460;background:#1a1a2e;color:#a0a0c0;cursor:pointer;font-size:9px;border-radius:3px;min-width:60px">
          SV Waterfall
        </button>
      </div>

      <!-- Animation controls -->
      <div class="cr"><label>Anim speed:</label>
        <input type="range" id="morph-speed" min="0.1" max="5.0" value="1.0" step="0.1">
        <span class="v" id="v-morph-speed">1.0</span>
      </div>
      <div class="cb">
        <input type="checkbox" id="morph-animate" checked>
        <label for="morph-animate">Animate eigenvalue flow</label>
      </div>
      <div class="cb">
        <input type="checkbox" id="morph-show-labels">
        <label for="morph-show-labels">Show token labels on trajectories</label>
      </div>
      <div class="cb">
        <input type="checkbox" id="morph-log-scale">
        <label for="morph-log-scale">Log scale magnitudes</label>
      </div>

      <!-- Canvas for morphing visualization -->
      <div style="margin-top:6px">
        <div id="morph-canvas-title" style="color:#fd79a8;font-size:10px;font-weight:bold;margin-bottom:2px">
          Eigenvalue Flow
        </div>
        <canvas id="morph-canvas" width="340" height="240"
          style="border:1px solid #0f3460;border-radius:4px;width:100%;cursor:crosshair"></canvas>
      </div>

      <!-- Jacobian decomposition summary -->
      <div id="morph-jacobian-summary" style="display:none;margin-top:6px;font-size:9px;max-height:200px;overflow-y:auto">
      </div>

      <!-- Holonomy results -->
      <div id="morph-holonomy-results" style="display:none;margin-top:6px;font-size:9px;max-height:200px;overflow-y:auto">
      </div>

      <!-- Connection 1-form display -->
      <div id="morph-connection-display" style="display:none;margin-top:6px;font-size:9px;max-height:200px;overflow-y:auto">
      </div>

      <!-- Eigenvalue trajectory list -->
      <div id="morph-eigen-list" style="margin-top:6px;font-size:9px;max-height:160px;overflow-y:auto">
      </div>
    </div>
  </div>
</details>
<!-- ---- Jacobian Field Visualization (No Points) ---- -->
<details>
  <summary style="color:#ff6b9d;font-size:12px;font-weight:bold;cursor:pointer;padding:4px 0;border-bottom:1px solid #0f3460">
    🌊 Jacobian Field (No Points)
  </summary>
  <div id="jf-panel" style="margin-top:6px">
    <div style="color:#888;font-size:9px;margin-bottom:6px;line-height:1.4">
      The <b>morphing itself</b>, stripped of all data. Watch the Jacobian field
      breathe — divergence, curl, shear, eigenflows — as pure geometry.
    </div>

    <label style="color:#a0a0c0;font-size:9px;display:block;margin-top:4px">
      <input type="checkbox" id="jf-raw-dims" checked> Use selected Dim X,Y (no PCA)
    </label>

    <div class="cr"><label>Grid res:</label>
      <input type="range" id="jf-res" min="8" max="40" value="20" step="1">
      <span class="v" id="v-jf-res">20</span>
    </div>
    <div class="cr"><label>PCA dims:</label>
      <input type="range" id="jf-pca" min="4" max="64" value="16" step="2">
      <span class="v" id="v-jf-pca">16</span>
    </div>
    <div class="cr"><label>Layer:</label>
      <input type="range" id="jf-layer" min="0" max="11" value="0" step="1">
      <span class="v" id="v-jf-layer">0</span>
    </div>
    <div class="cr"><label>Render:</label>
      <select id="jf-render" style="flex:1;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;padding:2px;font-size:10px">
        <option value="divergence">Divergence (expansion/contraction)</option>
        <option value="curl">Curl (local rotation)</option>
        <option value="shear">Shear (anisotropic distortion)</option>
        <option value="det">Determinant (area change)</option>
        <option value="rotation">Rotation angle</option>
        <option value="condition">Condition number (anisotropy)</option>
        <option value="eigphase">Eigenvalue phase (rotation mode)</option>
        <option value="stretch">Principal stretches</option>
        <option value="flow">Flow field</option>
        <option value="composite">Composite (all channels)</option>
      </select>
    </div>
    <div class="cb"><input type="checkbox" id="jf-animate" checked>
      <label for="jf-animate">Animate flow</label></div>
    <div class="cb"><input type="checkbox" id="jf-eigvecs">
      <label for="jf-eigvecs">Show eigenvector directions</label></div>
    <div class="cb"><input type="checkbox" id="jf-stretch-ellipses" checked>
      <label for="jf-stretch-ellipses">Show stretch ellipses</label></div>
    <div class="cb"><input type="checkbox" id="jf-ghost-tokens">
      <label for="jf-ghost-tokens">Ghost token positions (faint)</label></div>

    <button id="btn-jf" onclick="runJacobianField()"
      style="margin-top:6px;width:100%;background:#ff6b9d;color:#fff;border:none;padding:6px;border-radius:4px;cursor:pointer;font-size:11px;font-weight:bold">
      Extract Jacobian Field
    </button>
    <div id="jf-status" style="font-size:9px;color:#888;margin-top:4px"></div>

    <div id="jf-canvas-wrap" style="display:none;margin-top:6px">
      <canvas id="jf-canvas" width="500" height="500"
        style="border:1px solid #0f3460;border-radius:4px;width:100%;cursor:crosshair;background:#050510"></canvas>
      <div style="display:flex;justify-content:space-between;font-size:8px;color:#888;margin-top:2px">
        <span id="jf-legend-min">min</span>
        <span id="jf-legend-title" style="color:#ff6b9d">—</span>
        <span id="jf-legend-max">max</span>
      </div>
    </div>
  </div>
</details>
</div>

<div id="main">
<div class="diffeo-canvas-wrap" id="diffeo-wrap" style="display:none">
  <canvas id="diffeo-canvas"></canvas>
</div>
<canvas id="cv"></canvas>
<div id="legend">
<div class="li"><div class="lc" style="background:linear-gradient(90deg,#0077b6,#666,#e94560)"></div>Strain</div>
<div class="li"><div class="lc" style="background:#0077b6"></div>Contraction</div>
<div class="li"><div class="lc" style="background:#666"></div>Isometry</div>
<div class="li"><div class="lc" style="background:#e94560"></div>Expansion</div>
<div class="li"><div class="lc" style="background:#0f0"></div>Selected</div>
<div class="li"><div class="lc" style="background:rgba(0,255,200,0.5)"></div>Neighbor</div>
<div class="li"><div class="lc" style="background:#f5a623"></div>Predicted Next</div>
</div>
  <canvas id="cv" width="800" height="800"></canvas>
</div>

<script>
""" + script_js_file_content + """
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
    itp_method = req.get("itp_method", "rbf").strip()
    if not text:
        raise ValueError("Empty text")
    print(f"\n[Server] Processing: {text[:60]}... (model: {model_name or MODEL_NAME}, itp: {itp_method})...")
    json_str = process_text(text, model_name, itp_method=itp_method)
    return json_str.encode("utf-8")

def compute_persistent_homology(hidden_states, max_dim=2, pca_dims=16, max_edge=10.0, collapse_eps=0.0):
    """
    Compute persistent homology of the token representation space at each layer.
    Uses per-layer normalization and distance-based filtration to ensure
    each layer's topology is captured independently.
    """
    try:
        from ripser import ripser
        HAS_RIPSER = True
    except ImportError:
        HAS_RIPSER = False

    try:
        from gudhi import RipsComplex
        HAS_GUDHI = True
    except ImportError:
        HAS_GUDHI = False

    if not HAS_GUDHI and not HAS_RIPSER:
        raise ImportError(
            "Neither gudhi nor ripser is installed. "
            "Install one: pip install gudhi  OR  pip install ripser"
        )

    from sklearn.decomposition import PCA
    from scipy.spatial.distance import pdist

    n_layers_plus_one = len(hidden_states)
    seq_len = hidden_states[0].shape[1]

    # Extract raw point clouds at each layer
    point_clouds = []
    for li in range(n_layers_plus_one):
        pts = hidden_states[li][0].detach().cpu().float().numpy()  # (seq_len, hidden_dim)
        point_clouds.append(pts)

    print(f"[TDA] {n_layers_plus_one} layers, {seq_len} tokens, dim={point_clouds[0].shape[1]}")

    persistence_diagrams = []
    layer_summaries = []

    for li in range(n_layers_plus_one):
        pts = point_clouds[li]
        layer_name = "Embedding" if li == 0 else f"Layer {li - 1}"

        # ---- Per-layer PCA ----
        # This is the KEY fix: each layer gets its OWN PCA projection
        # so we capture the geometry unique to that layer
        actual_pca_dims = min(pca_dims, pts.shape[1], pts.shape[0] - 1)
        if actual_pca_dims < pts.shape[1]:
            pca = PCA(n_components=actual_pca_dims)
            pts_reduced = pca.fit_transform(pts)
            explained = pca.explained_variance_ratio_.sum()
        else:
            pts_reduced = pts.copy()
            explained = 1.0

        # ---- Per-layer normalization ----
        # Normalize distances so that max_edge is meaningful across layers
        # (different layers can have very different scales)
        dists = pdist(pts_reduced)
        if len(dists) > 0 and dists.max() > 1e-12:
            median_dist = np.median(dists)
            # Scale so median pairwise distance = 1.0
            pts_normalized = pts_reduced / (median_dist + 1e-12)
        else:
            pts_normalized = pts_reduced

        if li == 0 or li == n_layers_plus_one - 1:
            raw_dists = pdist(pts_reduced)
            norm_dists = pdist(pts_normalized)
            print(f"  [TDA] {layer_name}: PCA explained={explained:.3f}, "
                  f"raw dist range=[{raw_dists.min():.3f}, {raw_dists.max():.3f}], "
                  f"norm dist range=[{norm_dists.min():.3f}, {norm_dists.max():.3f}]")

        # ---- Compute persistence ----
        diagrams_by_dim = {}

        if HAS_RIPSER:
            try:
                result = ripser(pts_normalized, maxdim=max_dim, thresh=max_edge)
                for dim_idx in range(len(result['dgms'])):
                    dgm = result['dgms'][dim_idx]
                    pairs = []
                    for pair in dgm:
                        b, d = float(pair[0]), float(pair[1])
                        if np.isinf(d):
                            d = float('inf')
                        pairs.append([b, d])
                    diagrams_by_dim[dim_idx] = pairs
            except Exception as e:
                print(f"  [TDA] ripser failed on {layer_name}: {e}")
                for dim_idx in range(max_dim + 1):
                    diagrams_by_dim[dim_idx] = []

        elif HAS_GUDHI:
            try:
                rips = RipsComplex(points=pts_normalized.tolist(), max_edge_length=max_edge)
                simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
                if collapse_eps > 0:
                    try:
                        simplex_tree.collapse_edges(nb_iterations=10)
                    except Exception:
                        pass
                simplex_tree.compute_persistence()
                for dim_idx in range(max_dim + 1):
                    intervals = simplex_tree.persistence_intervals_in_dimension(dim_idx)
                    pairs = []
                    for interval in intervals:
                        b, d = float(interval[0]), float(interval[1])
                        if np.isinf(d):
                            d = float('inf')
                        pairs.append([b, d])
                    diagrams_by_dim[dim_idx] = pairs
            except Exception as e:
                print(f"  [TDA] GUDHI failed on {layer_name}: {e}")
                for dim_idx in range(max_dim + 1):
                    diagrams_by_dim[dim_idx] = []

        persistence_diagrams.append(diagrams_by_dim)

        # ---- Summary stats ----
        betti_0 = len([p for p in diagrams_by_dim.get(0, []) if p[1] == float('inf')])
        betti_1 = len([p for p in diagrams_by_dim.get(1, []) if p[1] == float('inf')])
        betti_2 = (len([p for p in diagrams_by_dim.get(2, []) if p[1] == float('inf')])
                   if max_dim >= 2 else 0)
        n_features = sum(len(diagrams_by_dim.get(d, [])) for d in range(max_dim + 1))
        entropy = compute_persistence_entropy(diagrams_by_dim, max_dim)

        # Also compute total finite persistence as a measure of "topological activity"
        total_persistence = 0.0
        max_persistence = 0.0
        for d in range(max_dim + 1):
            for pair in diagrams_by_dim.get(d, []):
                if pair[1] != float('inf'):
                    pers = pair[1] - pair[0]
                    total_persistence += pers
                    max_persistence = max(max_persistence, pers)

        layer_summaries.append({
            'layer': li,
            'name': layer_name,
            'betti_0': betti_0,
            'betti_1': betti_1,
            'betti_2': betti_2,
            'n_features': n_features,
            'entropy': round(entropy, 6),
            'total_persistence': round(total_persistence, 6),
            'max_persistence': round(max_persistence, 6),
        })

        print(f"  [TDA] {layer_name}: β₀={betti_0}, β₁={betti_1}, β₂={betti_2}, "
              f"features={n_features}, entropy={entropy:.4f}, "
              f"total_pers={total_persistence:.4f}, max_pers={max_persistence:.4f}")

    # ---- Cross-layer analysis ----
    wasserstein_matrix = compute_wasserstein_matrix(persistence_diagrams, max_dim)
    events = detect_topological_events(persistence_diagrams, layer_summaries, max_dim)
    total_features = sum(ls['n_features'] for ls in layer_summaries)
    summary = generate_tda_summary(layer_summaries, events, max_dim)

    return {
        'persistence_diagrams': persistence_diagrams,
        'layer_summaries': layer_summaries,
        'wasserstein_distances': wasserstein_matrix,
        'topological_events': events,
        'total_features': total_features,
        'max_dim': max_dim,
        'n_layers': n_layers_plus_one - 1,
        'seq_len': seq_len,
        'summary': summary,
    }

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
        elif path == "/sae_info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(handle_sae_info(b"{}"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        handler_map = {
            "/run": handle_post_run,
            "/multi_run": handle_multi_run,
            "/diffeomorphism_spectrum": handle_diffeomorphism_spectrum,
            "/contrastive_spectrum": handle_contrastive_spectrum,
            "/compare": handle_compare,
            "/sae_features": handle_sae_features,
            "/sae_intervene": handle_sae_intervene,
            "/sae_info": handle_sae_info,
            "/neuron_grid": handle_neuron_grid,  # <-- ADD THIS
            "/curvature": handle_curvature_analysis,  # <-- NEW
            "/morphing_analysis": handle_morphing_analysis,
            "/tda": handle_tda,
            "/jacobian_field": handle_jacobian_field_viz,
        }

        handler = handler_map.get(path)
        if handler:
            try:
                response_bytes = handler(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def handle_sae_intervene(body_bytes):
    """Clamp an SAE feature and return modified next-token predictions."""
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    layer = req.get("layer", 0)
    feature_id = req.get("feature_id", 0)
    clamp_value = req.get("clamp_value", 0.0)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()
    if LM_MODEL is None:
        return json.dumps({"error": "No LM head model loaded"}).encode()
    sae = get_sae_for_layer(layer)
    if sae is None:
        return json.dumps({"error": f"No SAE for layer {layer}"}).encode()

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    sae = SAE_MODELS[layer]

    # Baseline predictions
    with torch.no_grad():
        baseline_out = LM_MODEL(input_ids)
        baseline_logits = baseline_out.logits[0, -1, :]
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        baseline_topk = torch.topk(baseline_probs, 10)

    baseline_results = []
    for i in range(10):
        tid = baseline_topk.indices[i].item()
        prob = baseline_topk.values[i].item()
        token_str = TOKENIZER.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")
        baseline_results.append({"token": token_str, "prob": round(prob, 4)})

    # Intervention hook
    def intervention_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            latents = sae.encode(h)
            latents[:, :, feature_id] = clamp_value
            h_new = sae.decode(latents)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    # Hook the LM model's transformer blocks
    lm_blocks = None
    if hasattr(LM_MODEL, 'transformer'):
        lm_blocks = _get_transformer_blocks(LM_MODEL.transformer)
    elif hasattr(LM_MODEL, 'model'):
        lm_blocks = _get_transformer_blocks(LM_MODEL.model)
    else:
        lm_blocks = _get_transformer_blocks(LM_MODEL)

    if not lm_blocks or layer >= len(lm_blocks):
        return json.dumps({"error": "Cannot hook LM model at this layer"}).encode()

    hook = lm_blocks[layer].register_forward_hook(intervention_hook)
    try:
        with torch.no_grad():
            outputs = LM_MODEL(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            topk = torch.topk(probs, 10)

        modified_results = []
        for i in range(10):
            tid = topk.indices[i].item()
            prob = topk.values[i].item()
            token_str = TOKENIZER.decode([tid]).replace("\u0120", " ").replace("\u010a", "\\n")
            modified_results.append({"token": token_str, "prob": round(prob, 4)})
    finally:
        hook.remove()

    return json.dumps({
        "baseline_predictions": baseline_results,
        "modified_predictions": modified_results,
        "feature_id": feature_id,
        "clamp_value": clamp_value,
        "layer": layer,
        "tokens": tokens,
    }).encode()

def handle_sae_info(body_bytes):
    """Return info about which SAE layers are available."""
    info = {
        "loaded_layers": sorted(SAE_MODELS.keys()),
        "model_name": MODEL_NAME,
        "total_layers": get_n_layers(MODEL_CONFIG) if MODEL_CONFIG else 0,
        "lazy_loading": _SAE_RELEASE_ID is not None,
        "release_id": _SAE_RELEASE_ID,
    }
    layer_info = {}
    for layer, sae in SAE_MODELS.items():
        d_sae = sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else None
        layer_info[str(layer)] = {"d_sae": d_sae}
    info["layer_info"] = layer_info
    return json.dumps(info).encode()


def handle_sae_features(body_bytes):
    """Return top-K active SAE features for a given token at a given layer."""
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    layer = req.get("layer", 0)
    token_idx = req.get("token_idx", 0)
    top_k = req.get("top_k", 20)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()
    if len(SAE_MODELS) == 0:
        return json.dumps({"error": "No SAEs loaded", "features": []}).encode()
    if layer not in SAE_MODELS:
        sae = get_sae_for_layer(layer)
        if sae is None:
            available = sorted(SAE_MODELS.keys())
            return json.dumps({
                "error": f"No SAE for layer {layer}. Available: {available}",
                "features": []
            }).encode()
    sae = SAE_MODELS[layer]

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    n_tokens = input_ids.shape[1]
    if token_idx < 0 or token_idx >= n_tokens:
        return json.dumps({"error": f"token_idx {token_idx} out of range [0, {n_tokens})"}).encode()

    hs = extract_hidden_states(MODEL, input_ids)
    sae = SAE_MODELS[layer]

    with torch.no_grad():
        h = hs[layer + 1].squeeze(0)  # (seq_len, hidden_dim)
        latents = sae.encode(h)       # (seq_len, d_sae)

    token_latents = latents[token_idx].cpu().numpy()
    top_indices = np.argsort(-np.abs(token_latents))[:top_k]

    features = []
    for idx in top_indices:
        act = float(token_latents[idx])
        if abs(act) < 1e-8 and len(features) >= 5:
            break
        features.append({
            "feature_id": int(idx),
            "activation": round(act, 6),
        })

    # Token activations for heatmap (top 10 features)
    all_token_acts = {}
    for f in features[:10]:
        fid = f["feature_id"]
        acts = latents[:, fid].cpu().numpy().tolist()
        all_token_acts[str(fid)] = [round(a, 6) for a in acts]

    return json.dumps({
        "features": features,
        "token_idx": token_idx,
        "layer": layer,
        "tokens": tokens,
        "n_latents": int(latents.shape[1]),
        "token_activations": all_token_acts,
    }).encode()

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

SAE_MODELS = {}  # layer_idx -> trained SAE


def extract_sae_features(hidden_states, n_layers):
    """For each layer, encode hidden states into sparse SAE latents."""
    sae_activations = {}
    for layer in range(n_layers):
        if layer not in SAE_MODELS:
            continue
        sae = SAE_MODELS[layer]
        h = hidden_states[layer + 1]
        with torch.no_grad():
            latents = sae.encode(h.squeeze(0))
        sae_activations[layer] = latents.cpu().numpy()
    return sae_activations

def intervene_sae_feature(hidden_states, layer, feature_idx, new_value, sae):
    """Encode, modify one latent, decode back."""
    h = hidden_states[layer + 1].clone()
    with torch.no_grad():
        latents = sae.encode(h)
        original_val = latents[0, :, feature_idx].clone()
        latents[0, :, feature_idx] = new_value
        h_modified = sae.decode(latents)
    return h_modified, original_val

def handle_neuron_grid(body_bytes):
    """Return per-layer activation vectors for each real token,
    normalized to [0,1] for pixel rendering."""
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    n_tokens = input_ids.shape[1]

    # For each layer, collect the activation vector for each token
    # hs[0] = embedding, hs[1] = after layer 0, ..., hs[n_layers] = after last layer
    layers_data = []

    # Compute global min/max across ALL layers for consistent normalization
    global_min = float('inf')
    global_max = float('-inf')

    raw_activations = []
    for layer_idx in range(n_layers + 1):  # include embedding layer (0)
        layer_acts = hs[layer_idx][0].cpu().float().numpy()  # (seq_len, hidden_dim)
        raw_activations.append(layer_acts)
        global_min = min(global_min, float(layer_acts.min()))
        global_max = max(global_max, float(layer_acts.max()))

    # Normalize to [0, 1] and build response
    range_val = global_max - global_min if (global_max - global_min) > 1e-12 else 1.0

    for layer_idx, layer_acts in enumerate(raw_activations):
        normalized = ((layer_acts - global_min) / range_val)  # (seq_len, hidden_dim)
        # Clamp to [0,1]
        normalized = np.clip(normalized, 0.0, 1.0)
        layers_data.append({
            "layer": layer_idx,  # 0 = embedding, 1 = after layer 0, etc.
            # Each token's activation as a list of floats in [0,1]
            "activations": normalized.tolist(),
        })

    # Also compute per-layer normalization (relative within each layer)
    layers_data_relative = []
    for layer_idx, layer_acts in enumerate(raw_activations):
        lmin = float(layer_acts.min())
        lmax = float(layer_acts.max())
        lrange = lmax - lmin if (lmax - lmin) > 1e-12 else 1.0
        normalized = np.clip((layer_acts - lmin) / lrange, 0.0, 1.0)
        layers_data_relative.append({
            "layer": layer_idx,
            "activations": normalized.tolist(),
        })

    return json.dumps({
        "tokens": tokens,
        "n_tokens": n_tokens,
        "n_layers": n_layers + 1,  # including embedding
        "hidden_dim": hidden_dim,
        "global_norm": layers_data,
        "layer_norm": layers_data_relative,
        "global_min": round(global_min, 6),
        "global_max": round(global_max, 6),
    }).encode()

def handle_compare(body_bytes):
    """Compare two texts: return per-layer activation grids for A, B, and A-B diff."""
    req = json.loads(body_bytes)
    text_a = req.get("text_a", "").strip()
    text_b = req.get("text_b", "").strip()
    if not text_a or not text_b:
        return json.dumps({"error": "Both text_a and text_b required"}).encode()

    # Tokenize both
    ids_a, tokens_a = tokenize_text(TOKENIZER, text_a)
    ids_b, tokens_b = tokenize_text(TOKENIZER, text_b)

    # Extract hidden states
    hs_a = extract_hidden_states(MODEL, ids_a)
    hs_b = extract_hidden_states(MODEL, ids_b)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    n_tok_a = ids_a.shape[1]
    n_tok_b = ids_b.shape[1]

    # We align by position index up to min length for the diff.
    # Tokens beyond the shorter sequence get no diff.
    n_common = min(n_tok_a, n_tok_b)

    # Collect raw activations: list of (n_layers+1) arrays, each (seq_len, hidden_dim)
    raw_a = []
    raw_b = []
    for li in range(n_layers + 1):
        raw_a.append(hs_a[li][0].cpu().float().numpy())
        raw_b.append(hs_b[li][0].cpu().float().numpy())

    # --- Normalize A and B independently (per-layer) for their own grids ---
    def normalize_per_layer(raw_list):
        result = []
        for layer_acts in raw_list:
            lmin = float(layer_acts.min())
            lmax = float(layer_acts.max())
            lr = lmax - lmin if (lmax - lmin) > 1e-12 else 1.0
            normed = np.clip((layer_acts - lmin) / lr, 0.0, 1.0)
            result.append(normed.tolist())
        return result

    norm_a = normalize_per_layer(raw_a)
    norm_b = normalize_per_layer(raw_b)

    # --- Compute diff for aligned tokens ---
    # diff = activation_A - activation_B (raw, then normalize to [-1, 1] -> [0, 1])
    diff_layers = []
    diff_magnitude_layers = []  # |diff| for magnitude heatmap

    # Track global diff range for consistent normalization
    global_diff_max = 0.0
    raw_diffs = []
    for li in range(n_layers + 1):
        if n_common > 0:
            d = raw_a[li][:n_common] - raw_b[li][:n_common]  # (n_common, hidden_dim)
        else:
            d = np.zeros((0, hidden_dim))
        raw_diffs.append(d)
        if d.size > 0:
            m = float(np.abs(d).max())
            if m > global_diff_max:
                global_diff_max = m

    if global_diff_max < 1e-12:
        global_diff_max = 1.0

    for li in range(n_layers + 1):
        d = raw_diffs[li]
        if d.size > 0:
            # Map diff from [-global_diff_max, +global_diff_max] to [0, 1]
            # 0.5 = no difference, 0 = B >> A, 1 = A >> B
            normed_diff = np.clip(d / (2 * global_diff_max) + 0.5, 0.0, 1.0)
            # Magnitude: 0 = no diff, 1 = max diff
            mag = np.clip(np.abs(d) / global_diff_max, 0.0, 1.0)
        else:
            normed_diff = np.zeros((0, hidden_dim))
            mag = np.zeros((0, hidden_dim))
        diff_layers.append(normed_diff.tolist())
        diff_magnitude_layers.append(mag.tolist())

    # --- Per-layer divergence summary (scalar per layer) ---
    # Mean absolute diff across all aligned tokens and dims
    layer_divergence = []
    for li in range(n_layers + 1):
        d = raw_diffs[li]
        if d.size > 0:
            layer_divergence.append(round(float(np.mean(np.abs(d))), 6))
        else:
            layer_divergence.append(0.0)

    # --- Per-token divergence at each layer ---
    # For each aligned token, mean |diff| across dims
    token_divergence = []  # list of (n_layers+1) lists of n_common floats
    for li in range(n_layers + 1):
        d = raw_diffs[li]
        if d.size > 0:
            per_tok = np.mean(np.abs(d), axis=1)  # (n_common,)
            token_divergence.append([round(float(v), 6) for v in per_tok])
        else:
            token_divergence.append([])

    # --- Top diverging dimensions per layer ---
    # Which hidden dims show the biggest difference?
    top_dims_per_layer = []
    for li in range(n_layers + 1):
        d = raw_diffs[li]
        if d.size > 0:
            mean_abs_per_dim = np.mean(np.abs(d), axis=0)  # (hidden_dim,)
            top_k = min(20, hidden_dim)
            top_idx = np.argsort(-mean_abs_per_dim)[:top_k]
            top_dims_per_layer.append([
                {"dim": int(idx), "mean_abs_diff": round(float(mean_abs_per_dim[idx]), 6)}
                for idx in top_idx
            ])
        else:
            top_dims_per_layer.append([])

    # --- Find the "divergence onset" layer ---
    # The layer where divergence first exceeds 2x the embedding-layer divergence
    onset_layer = -1
    if len(layer_divergence) > 1 and layer_divergence[0] > 1e-12:
        baseline_div = layer_divergence[0]
        for li in range(1, len(layer_divergence)):
            if layer_divergence[li] > baseline_div * 2:
                onset_layer = li - 1  # report as transformer layer index
                break

    return json.dumps({
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "n_tokens_a": n_tok_a,
        "n_tokens_b": n_tok_b,
        "n_common": n_common,
        "n_layers": n_layers + 1,
        "hidden_dim": hidden_dim,
        "activations_a": norm_a,
        "activations_b": norm_b,
        "diff": diff_layers,
        "diff_magnitude": diff_magnitude_layers,
        "layer_divergence": layer_divergence,
        "token_divergence": token_divergence,
        "top_dims_per_layer": top_dims_per_layer,
        "onset_layer": onset_layer,
        "global_diff_max": round(global_diff_max, 6),
    }).encode()


def _compute_perturbed_delta(model, input_ids, layer, token_idx, perturbation, original_hs):
    """
    Compute the residual delta at a specific layer and token
    when the hidden state is perturbed by `perturbation`.

    Uses a forward hook to inject the perturbation at the right layer.
    """
    blocks = _get_transformer_blocks(model)
    if not blocks or layer >= len(blocks):
        return None

    captured = {}

    def inject_hook(module, input, output):
        """Inject perturbation into the hidden state at the target layer."""
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[0, token_idx] = h[0, token_idx] + perturbation.to(h.device)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def capture_hook(module, input, output):
        """Capture the output of the next layer."""
        h = output[0] if isinstance(output, tuple) else output
        captured["output"] = h[0, token_idx].detach().cpu().float().numpy()

    # Hook the target layer to inject perturbation
    hook1 = blocks[layer].register_forward_hook(inject_hook)

    # If there's a next layer, hook it to capture the result
    if layer + 1 < len(blocks):
        hook2 = blocks[layer + 1].register_forward_hook(capture_hook)
    else:
        hook2 = None

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook1.remove()
        if hook2:
            hook2.remove()

    if "output" in captured:
        # The perturbed delta is the difference between the captured output
        # and the original hidden state at this layer
        h_base = original_hs[layer][0, token_idx].cpu().float().numpy()
        return captured["output"] - h_base

    return None


def _eigvec_to_top_dims(eigvec, principal_dirs, hidden_dim, top_k=10):
    """
    Convert a projected eigenvector back to hidden-space dimension indices.
    Returns the top_k hidden dimensions that this eigenvector acts on most.
    """
    # eigvec is in the K-dimensional projected space
    # principal_dirs is (K, hidden_dim)
    # Reconstruct in full hidden space
    full_vec = np.zeros(hidden_dim)
    for i in range(len(eigvec)):
        full_vec += eigvec[i].real * principal_dirs[i]

    # Find top dimensions by magnitude
    top_idx = np.argsort(-np.abs(full_vec))[:top_k]
    return [{"dim": int(idx), "weight": round(float(full_vec[idx]), 6)} for idx in top_idx]

def _compute_principal_directions(h_cloud, n_tokens, hidden_dim, max_k=32):
    """
    Compute principal directions from a token cloud via SVD.

    Args:
        h_cloud: np.ndarray of shape (n_tokens, hidden_dim)
        n_tokens: number of tokens
        hidden_dim: hidden dimension size
        max_k: maximum number of principal components

    Returns:
        K: int — number of principal directions
        principal_dirs: np.ndarray of shape (K, hidden_dim)
    """
    if n_tokens >= 3:
        cloud_centered = h_cloud - h_cloud.mean(axis=0)
        U, S, Vt = np.linalg.svd(cloud_centered, full_matrices=False)
        K = min(max_k, hidden_dim, n_tokens - 1)
        principal_dirs = Vt[:K]
    else:
        K = min(max_k, hidden_dim)
        principal_dirs = np.eye(hidden_dim)[:K]
    return K, principal_dirs

def _estimate_jacobian_from_cloud(hs, lay, n_tokens, hidden_dim, K, principal_dirs, h_cloud):
    """
    Estimate the projected Jacobian of the layer-to-layer map
    from the token cloud using least-squares regression.

    Args:
        hs: hidden states tuple
        lay: layer index
        n_tokens: number of tokens
        hidden_dim: hidden dimension
        K: number of principal components
        principal_dirs: np.ndarray of shape (K, hidden_dim)
        h_cloud: np.ndarray of shape (n_tokens, hidden_dim) at layer `lay`

    Returns:
        J_proj: np.ndarray of shape (K, K) — projected Jacobian
    """
    deltas_cloud = np.stack([
        (hs[lay + 1][0, t] - hs[lay][0, t]).cpu().float().numpy()
        for t in range(n_tokens)
    ], axis=0)

    if n_tokens >= 3:
        cloud_centered = h_cloud - h_cloud.mean(axis=0)
        pos_proj = cloud_centered @ principal_dirs.T
        del_centered = deltas_cloud - deltas_cloud.mean(axis=0)
        del_proj = del_centered @ principal_dirs.T
        try:
            J_proj = np.linalg.lstsq(pos_proj, del_proj, rcond=None)[0].T
        except Exception:
            J_proj = np.eye(K)
    else:
        J_proj = np.eye(K)

    return J_proj

def compute_spectrum_for_text(text):
    """
    Compute the diffeomorphism spectrum for a single text.

    Returns:
        dict with keys: tokens, n_tokens, n_layers, layer_spectra
        where layer_spectra[lay][tok] is a dict of geometric invariants.
    """
    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)
    n_tokens = input_ids.shape[1]
    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)

    layer_spectra = []

    for lay in range(n_layers):
        token_spectra = []

        h_cloud = hs[lay][0].cpu().float().numpy()
        K, principal_dirs = _compute_principal_directions(h_cloud, n_tokens, hidden_dim)

        # Estimate Jacobian from token cloud
        J_proj = _estimate_jacobian_from_cloud(hs, lay, n_tokens, hidden_dim, K, principal_dirs, h_cloud)

        for ti in range(n_tokens):
            delta = (hs[lay + 1][0, ti] - hs[lay][0, ti]).cpu().float().numpy()
            eigenvalues, eigenvectors = np.linalg.eig(J_proj)
            sort_idx = np.argsort(-np.abs(eigenvalues))
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            spec = _compute_geometric_invariants(eigenvalues, eigenvectors, J_proj, K, delta, principal_dirs, hidden_dim)
            token_spectra.append(spec)

        layer_spectra.append(token_spectra)

    return {
        "tokens": tokens,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "layer_spectra": layer_spectra,
    }

def _compute_geometric_invariants(eigenvalues, eigenvectors, J_proj, K, delta, principal_dirs, hidden_dim):
    """
    Compute geometric invariants from a Jacobian's eigendecomposition.

    Args:
        eigenvalues: np.ndarray of complex eigenvalues (sorted by magnitude)
        eigenvectors: np.ndarray of eigenvectors (columns, sorted)
        J_proj: np.ndarray of shape (K, K) — the projected Jacobian
        K: number of projected dimensions
        delta: np.ndarray of shape (hidden_dim,) — the residual delta for this token
        principal_dirs: np.ndarray of shape (K, hidden_dim)
        hidden_dim: int

    Returns:
        dict of geometric invariants
    """
    eig_real = eigenvalues.real
    eig_imag = eigenvalues.imag
    eig_magnitude = np.abs(eigenvalues)
    eig_phase = np.angle(eigenvalues)

    # Divergence = trace of Jacobian
    divergence = float(np.real(np.sum(eigenvalues)))

    # Curl = antisymmetric part
    J_antisym = (J_proj - J_proj.T) / 2
    curl_magnitude = float(np.linalg.norm(J_antisym, 'fro'))

    # Shear = traceless symmetric part
    J_sym = (J_proj + J_proj.T) / 2
    J_traceless = J_sym - np.eye(K) * np.trace(J_sym) / K
    shear_magnitude = float(np.linalg.norm(J_traceless, 'fro'))

    # Determinant (via log for stability)
    log_det = float(np.real(np.sum(np.log(np.abs(eigenvalues) + 1e-30))))
    det = float(np.exp(np.clip(log_det, -50, 50)))

    # Singular values and derived quantities
    sv = np.linalg.svd(J_proj, compute_uv=False)
    condition_number = float(sv[0] / max(sv[-1], 1e-12))

    sv_norm = sv / max(sv.sum(), 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-12]
    effective_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))

    n_expanding = int(np.sum(eig_magnitude > 1.05))
    n_contracting = int(np.sum(eig_magnitude < 0.95))
    n_rotating = int(np.sum(np.abs(eig_imag) > 0.05))

    # Holonomy proxy
    holonomy_proxy = curl_magnitude / max(K, 1)

    # Delta norm
    delta_norm = float(np.linalg.norm(delta))

    # Top eigenvector mapped back to hidden dims
    top_eigvec_dims = _eigvec_to_top_dims(
        eigenvectors[:, 0], principal_dirs, hidden_dim, top_k=10
    ) if len(eigenvectors) > 0 else []

    return {
        "eigenvalues_real": eig_real[:16].tolist(),
        "eigenvalues_imag": eig_imag[:16].tolist(),
        "eigenvalues_magnitude": eig_magnitude[:16].tolist(),
        "eigenvalues_phase": eig_phase[:16].tolist(),
        "divergence": round(divergence, 6),
        "curl": round(curl_magnitude, 6),
        "shear": round(shear_magnitude, 6),
        "determinant": round(det, 6),
        "condition_number": round(condition_number, 4),
        "effective_rank": round(effective_rank, 4),
        "holonomy": round(holonomy_proxy, 6),
        "n_expanding": n_expanding,
        "n_contracting": n_contracting,
        "n_rotating": n_rotating,
        "singular_values": sv[:16].tolist(),
        "delta_norm": round(delta_norm, 6),
        "top_eigenvector_dims": top_eigvec_dims,
    }

INVARIANT_KEYS = [
    "divergence", "curl", "shear", "determinant",
    "condition_number", "effective_rank", "holonomy",
    "n_expanding", "n_contracting", "n_rotating"
]


def _aggregate_contrastive_invariants(pos_spectra, neg_spectra, n_layers):
    """
    For each layer, compute mean geometric invariants for positive vs negative
    sets and find the biggest differences (effect sizes).

    Returns:
        layer_contrasts: list of dicts (one per layer)
        best_layer: int
        best_invariant: str
        best_effect: float
        ranked_layers: list of ints sorted by total contrast score
    """
    layer_contrasts = []

    for lay in range(n_layers):
        pos_vals = {k: [] for k in INVARIANT_KEYS}
        neg_vals = {k: [] for k in INVARIANT_KEYS}

        for spec in pos_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    for k in INVARIANT_KEYS:
                        pos_vals[k].append(tok_spec[k])

        for spec in neg_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    for k in INVARIANT_KEYS:
                        neg_vals[k].append(tok_spec[k])

        contrast = {"layer": lay}
        total_contrast_score = 0.0

        for k in INVARIANT_KEYS:
            pos_arr = np.array(pos_vals[k]) if pos_vals[k] else np.array([0.0])
            neg_arr = np.array(neg_vals[k]) if neg_vals[k] else np.array([0.0])

            pos_mean = float(np.mean(pos_arr))
            neg_mean = float(np.mean(neg_arr))
            pos_std = float(np.std(pos_arr))
            neg_std = float(np.std(neg_arr))

            pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2) if (pos_std + neg_std) > 1e-12 else 1.0
            effect_size = abs(pos_mean - neg_mean) / pooled_std

            contrast[k] = {
                "pos_mean": round(pos_mean, 6),
                "neg_mean": round(neg_mean, 6),
                "pos_std": round(pos_std, 6),
                "neg_std": round(neg_std, 6),
                "difference": round(pos_mean - neg_mean, 6),
                "effect_size": round(effect_size, 4),
            }
            total_contrast_score += effect_size

        contrast["total_contrast_score"] = round(total_contrast_score, 4)
        layer_contrasts.append(contrast)

    ranked_layers = sorted(
        range(n_layers),
        key=lambda lll: layer_contrasts[lll]["total_contrast_score"],
        reverse=True
    )

    best_invariant = None
    best_effect = 0.0
    best_layer = 0
    for lay in range(n_layers):
        for k in INVARIANT_KEYS:
            es = layer_contrasts[lay][k]["effect_size"]
            if es > best_effect:
                best_effect = es
                best_invariant = k
                best_layer = lay

    return layer_contrasts, best_layer, best_invariant, best_effect, ranked_layers

def _compare_eigenvalue_distributions(pos_spectra, neg_spectra, ranked_layers, top_n=3):
    """
    For the top-N most contrastive layers, compare eigenvalue magnitude
    distributions between positive and negative sets.

    Returns:
        list of dicts with histogram data and KL divergence
    """
    eigenvalue_comparisons = []

    for lay in ranked_layers[:top_n]:
        pos_eigs = []
        neg_eigs = []

        for spec in pos_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    pos_eigs.extend(tok_spec["eigenvalues_magnitude"])

        for spec in neg_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    neg_eigs.extend(tok_spec["eigenvalues_magnitude"])

        all_eigs = pos_eigs + neg_eigs
        if not all_eigs:
            continue

        bins = np.linspace(min(all_eigs), max(all_eigs), 30)
        pos_hist, _ = np.histogram(pos_eigs, bins=bins, density=True) if pos_eigs else (np.zeros(29), bins)
        neg_hist, _ = np.histogram(neg_eigs, bins=bins, density=True) if neg_eigs else (np.zeros(29), bins)

        pos_hist_safe = pos_hist + 1e-10
        neg_hist_safe = neg_hist + 1e-10
        pos_hist_safe /= pos_hist_safe.sum()
        neg_hist_safe /= neg_hist_safe.sum()
        kl_div = float(
            0.5 * np.sum(pos_hist_safe * np.log(pos_hist_safe / neg_hist_safe)) +
            0.5 * np.sum(neg_hist_safe * np.log(neg_hist_safe / pos_hist_safe))
        )

        eigenvalue_comparisons.append({
            "layer": lay,
            "pos_histogram": pos_hist.tolist(),
            "neg_histogram": neg_hist.tolist(),
            "bin_edges": bins.tolist(),
            "kl_divergence": round(kl_div, 6),
            "pos_mean_magnitude": round(float(np.mean(pos_eigs)), 6) if pos_eigs else 0.0,
            "neg_mean_magnitude": round(float(np.mean(neg_eigs)), 6) if neg_eigs else 0.0,
        })

    return eigenvalue_comparisons

def handle_contrastive_spectrum(body_bytes):
    """
    Given two sets of prompts, compute the diffeomorphism spectrum
    for each, then find which geometric features are most discriminative.
    """
    req = json.loads(body_bytes)
    positive_texts = req.get("positive", [])
    negative_texts = req.get("negative", [])
    behavior_name = req.get("behavior", "unknown")

    if not positive_texts or not negative_texts:
        return json.dumps({"error": "Need both positive and negative texts"}).encode()

    # Stage 1: Compute spectra for both sets
    pos_spectra = []
    for text in positive_texts:
        try:
            pos_spectra.append(compute_spectrum_for_text(text))
        except Exception as e:
            print(f"[ContrastiveSpectrum] Error on positive text: {e}")

    neg_spectra = []
    for text in negative_texts:
        try:
            neg_spectra.append(compute_spectrum_for_text(text))
        except Exception as e:
            print(f"[ContrastiveSpectrum] Error on negative text: {e}")

    if not pos_spectra or not neg_spectra:
        return json.dumps({"error": "Failed to compute spectra for one or both sets"}).encode()

    n_layers = pos_spectra[0]["n_layers"]

    # Stage 2: Aggregate and compare invariants
    layer_contrasts, best_layer, best_invariant, best_effect, ranked_layers = \
        _aggregate_contrastive_invariants(pos_spectra, neg_spectra, n_layers)

    # Stage 3: Compare eigenvalue distributions
    eigenvalue_comparisons = _compare_eigenvalue_distributions(
        pos_spectra, neg_spectra, ranked_layers, top_n=3
    )

    # Stage 4: Build geometric signature
    geometric_signature = {
        "behavior": behavior_name,
        "most_discriminative_layer": best_layer,
        "most_discriminative_invariant": best_invariant,
        "best_effect_size": round(best_effect, 4),
        "layer_ranking": ranked_layers[:5],
        "description": _generate_signature_description(
            best_layer, best_invariant, best_effect,
            layer_contrasts, behavior_name
        ),
    }

    return json.dumps({
        "behavior": behavior_name,
        "n_positive": len(pos_spectra),
        "n_negative": len(neg_spectra),
        "n_layers": n_layers,
        "layer_contrasts": layer_contrasts,
        "ranked_layers": ranked_layers,
        "eigenvalue_comparisons": eigenvalue_comparisons,
        "geometric_signature": geometric_signature,
        "example_positive_spectrum": pos_spectra[0] if pos_spectra else None,
        "example_negative_spectrum": neg_spectra[0] if neg_spectra else None,
    }, cls=SafeFloatEncoder).encode()

def _generate_signature_description(best_layer, best_invariant, best_effect,
                                     layer_contrasts, behavior_name):
    """Generate a human-readable description of the geometric signature."""
    descriptions = {
        "divergence": "expansion/contraction of the representation space",
        "curl": "rotational mixing of information between dimensions",
        "shear": "anisotropic distortion (stretching in some directions, compressing in others)",
        "determinant": "volume change of the local representation neighborhood",
        "condition_number": "anisotropy of the transformation (how directionally biased it is)",
        "effective_rank": "effective dimensionality of the active transformation",
        "holonomy": "parallel transport deficit (how much local frames rotate)",
        "n_expanding": "number of expanding eigenvalue directions",
        "n_contracting": "number of contracting eigenvalue directions",
        "n_rotating": "number of rotating (complex eigenvalue) directions",
    }

    inv_desc = descriptions.get(best_invariant, best_invariant)
    contrast = layer_contrasts[best_layer][best_invariant]
    direction = "higher" if contrast["difference"] > 0 else "lower"

    desc = (
        f"The '{behavior_name}' behavior is most geometrically distinguished at "
        f"layer {best_layer} by its {inv_desc}. "
        f"Prompts triggering this behavior show {direction} {best_invariant} "
        f"(effect size d={best_effect:.2f}). "
        f"Positive mean: {contrast['pos_mean']:.4f}, "
        f"Negative mean: {contrast['neg_mean']:.4f}."
    )

    # Add secondary insights
    top_layers = [layer_contrasts[lx] for lx in sorted(
        range(len(layer_contrasts)),
        key=lambda lll: layer_contrasts[lll]["total_contrast_score"],
        reverse=True
    )[:3]]

    if len(top_layers) >= 2:
        desc += (
            f" The top 3 most discriminative layers are "
            f"{top_layers[0]['layer']}, {top_layers[1]['layer']}"
        )
        if len(top_layers) >= 3:
            desc += f", and {top_layers[2]['layer']}"
        desc += "."

    return desc


def _compute_perturbed_delta(model, input_ids, layer, token_idx, perturbation, original_hs):
    """
    Compute the residual delta at a specific layer and token
    when the hidden state is perturbed by `perturbation`.

    Uses a forward hook to inject the perturbation at the right layer,
    then captures the output of the next layer to compute the new delta.
    """
    blocks = _get_transformer_blocks(model)
    if not blocks or layer >= len(blocks):
        return None

    captured = {}

    def inject_hook(module, input, output):
        """Inject perturbation into the hidden state at the target layer."""
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[0, token_idx] = h[0, token_idx] + perturbation.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    def capture_hook(module, input, output):
        """Capture the output of the next layer."""
        h = output[0] if isinstance(output, tuple) else output
        captured["output"] = h[0, token_idx].detach().cpu().float().numpy()

    # Hook the target layer to inject perturbation
    hook1 = blocks[layer].register_forward_hook(inject_hook)

    # Hook the next layer to capture the result
    if layer + 1 < len(blocks):
        hook2 = blocks[layer + 1].register_forward_hook(capture_hook)
    else:
        # If this is the last block, capture from the block itself
        # (the output after perturbation propagates through)
        hook2 = None
        # Instead, capture from the perturbed block's own output
        def capture_self_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured["output"] = h[0, token_idx].detach().cpu().float().numpy()

        # We need the output AFTER the perturbation, so we capture from
        # the block after the injected one. If there's no next block,
        # we use the injected block's output directly.
        hook2 = blocks[layer].register_forward_hook(capture_self_hook)
        # But this would conflict with inject_hook. Instead, skip.
        hook2.remove()
        hook2 = None

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        hook1.remove()
        if hook2:
            hook2.remove()

    if "output" in captured:
        # The perturbed delta: new output minus the original hidden state at this layer
        h_base = original_hs[layer][0, token_idx].cpu().float().numpy()
        return captured["output"] - h_base

    return None


def _eigvec_to_top_dims(eigvec, principal_dirs, hidden_dim, top_k=10):
    """
    Convert a projected eigenvector back to hidden-space dimension indices.
    Returns the top_k hidden dimensions that this eigenvector acts on most.
    """
    # eigvec is in the K-dimensional projected space
    # principal_dirs is (K, hidden_dim)
    # Reconstruct in full hidden space
    full_vec = np.zeros(hidden_dim)
    for i in range(min(len(eigvec), len(principal_dirs))):
        full_vec += float(eigvec[i].real) * principal_dirs[i]

    # Find top dimensions by magnitude
    top_idx = np.argsort(-np.abs(full_vec))[:top_k]
    return [
        {"dim": int(idx), "weight": round(float(full_vec[idx]), 6)}
        for idx in top_idx
    ]

def handle_diffeomorphism_spectrum(body_bytes):
    """Compute the Jacobian eigenstructure of each layer's diffeomorphism
    for a single text (or pair of texts for comparison)."""
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    text_b = req.get("text_b", None)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    n_tokens = input_ids.shape[1]

    # Reuse the spectrum computation from contrastive scanner
    # but for a single text
    layer_spectra = []

    for lay in range(n_layers):
        token_spectra = []
        h_cloud = hs[lay][0].cpu().float().numpy()

        if n_tokens >= 3:
            cloud_centered = h_cloud - h_cloud.mean(axis=0)
            U, S, Vt = np.linalg.svd(cloud_centered, full_matrices=False)
            K = min(32, hidden_dim, n_tokens - 1)
            principal_dirs = Vt[:K]
        else:
            K = min(32, hidden_dim)
            principal_dirs = np.eye(hidden_dim)[:K]

        # Estimate Jacobian from token cloud variation
        deltas_cloud = np.stack([
            (hs[lay + 1][0, t] - hs[lay][0, t]).cpu().float().numpy()
            for t in range(n_tokens)
        ], axis=0)

        if n_tokens >= 3:
            pos_proj = cloud_centered @ principal_dirs.T
            del_centered = deltas_cloud - deltas_cloud.mean(axis=0)
            del_proj = del_centered @ principal_dirs.T
            try:
                J_proj = np.linalg.lstsq(pos_proj, del_proj, rcond=None)[0].T
            except Exception:
                J_proj = np.eye(K)
        else:
            J_proj = np.eye(K)

        # Per-token: use the shared Jacobian but compute per-token invariants
        # from the token's specific delta magnitude and direction
        for ti in range(n_tokens):
            delta = (hs[lay + 1][0, ti] - hs[lay][0, ti]).cpu().float().numpy()
            principal_dirs @ delta  # project delta into K-space

            eigenvalues, eigenvectors = np.linalg.eig(J_proj)
            sort_idx = np.argsort(-np.abs(eigenvalues))
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            eig_real = eigenvalues.real
            eig_imag = eigenvalues.imag
            eig_magnitude = np.abs(eigenvalues)
            eig_phase = np.angle(eigenvalues)

            divergence = float(np.real(np.sum(eigenvalues)))
            J_antisym = (J_proj - J_proj.T) / 2
            curl_magnitude = float(np.linalg.norm(J_antisym, 'fro'))
            J_sym = (J_proj + J_proj.T) / 2
            J_traceless = J_sym - np.eye(K) * np.trace(J_sym) / K
            shear_magnitude = float(np.linalg.norm(J_traceless, 'fro'))

            sv = np.linalg.svd(J_proj, compute_uv=False)
            condition_number = float(sv[0] / max(sv[-1], 1e-12))
            sv_norm = sv / max(sv.sum(), 1e-12)
            sv_norm = sv_norm[sv_norm > 1e-12]
            effective_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))

            # Per-token delta magnitude (how much THIS token moves)
            delta_norm = float(np.linalg.norm(delta))

            token_spectra.append({
                "eigenvalues_real": eig_real[:16].tolist(),
                "eigenvalues_imag": eig_imag[:16].tolist(),
                "eigenvalues_magnitude": eig_magnitude[:16].tolist(),
                "eigenvalues_phase": eig_phase[:16].tolist(),
                "divergence": round(divergence, 6),
                "curl": round(curl_magnitude, 6),
                "shear": round(shear_magnitude, 6),
                "condition_number": round(condition_number, 4),
                "effective_rank": round(effective_rank, 4),
                "n_expanding": int(np.sum(eig_magnitude > 1.05)),
                "n_contracting": int(np.sum(eig_magnitude < 0.95)),
                "n_rotating": int(np.sum(np.abs(eig_imag) > 0.05)),
                "singular_values": sv[:16].tolist(),
                "delta_norm": round(delta_norm, 6),
                "top_eigenvector_dims": _eigvec_to_top_dims(
                    eigenvectors[:, 0], principal_dirs, hidden_dim, top_k=10
                ),
            })

        layer_spectra.append(token_spectra)

    # ================================================================
    # AUTOMATIC ANOMALY DETECTION
    # ================================================================
    all_curls = [layer_spectra[lx][t]["curl"]
                 for lx in range(n_layers) for t in range(n_tokens)]
    mean_curl = np.mean(all_curls) if all_curls else 0

    anomalies = []
    for lay in range(n_layers):
        for ti in range(n_tokens):
            spec = layer_spectra[lay][ti]

            if spec["curl"] > mean_curl * 2.5:
                anomalies.append({
                    "type": "high_curl",
                    "layer": lay, "token": ti,
                    "token_str": tokens[ti],
                    "value": spec["curl"],
                    "description": f"High rotational mixing at L{lay} token '{tokens[ti]}'"
                })

            if spec["condition_number"] > 50:
                anomalies.append({
                    "type": "high_anisotropy",
                    "layer": lay, "token": ti,
                    "token_str": tokens[ti],
                    "value": spec["condition_number"],
                    "description": f"Highly anisotropic map at L{lay} token '{tokens[ti]}'"
                })

            if lay > 0:
                prev_rank = layer_spectra[lay - 1][ti]["effective_rank"]
                curr_rank = spec["effective_rank"]
                if abs(curr_rank - prev_rank) > 2:
                    anomalies.append({
                        "type": "rank_change",
                        "layer": lay, "token": ti,
                        "token_str": tokens[ti],
                        "value": curr_rank - prev_rank,
                        "description": f"Rank {'expansion' if curr_rank > prev_rank else 'collapse'} at L{lay} '{tokens[ti]}'"
                    })

    anomalies.sort(key=lambda a: abs(a["value"]), reverse=True)

    # ================================================================
    # COMPARISON (if text_b provided)
    # ================================================================
    diff_spectra = None
    if text_b and text_b.strip():
        ids_b, tokens_b = tokenize_text(TOKENIZER, text_b.strip())
        hs_b = extract_hidden_states(MODEL, ids_b)
        n_tokens_b = ids_b.shape[1]
        n_common = min(n_tokens, n_tokens_b)

        diff_spectra = {
            "tokens_a": tokens,
            "tokens_b": tokens_b,
            "n_common": n_common,
            "layer_diffs": [],
        }

        for lay in range(n_layers):
            layer_diff = []
            for ti in range(n_common):
                spec_a = layer_spectra[lay][ti]

                # Compute spectrum for text_b at this layer/token
                delta_b = (hs_b[lay + 1][0, ti] - hs_b[lay][0, ti]).cpu().float().numpy()
                delta_b_norm = float(np.linalg.norm(delta_b))

                # Compute spectrum for text_b at this layer/token
                delta_b = (hs_b[lay + 1][0, ti] - hs_b[lay][0, ti]).cpu().float().numpy()
                delta_b_norm = float(np.linalg.norm(delta_b))

                # Compare the geometric invariants
                # We need the spectrum for text_b at this layer
                # Use the same shared Jacobian approach but with text_b's token cloud
                h_cloud_b = hs_b[lay][0].cpu().float().numpy()
                if n_tokens_b >= 3:
                    cloud_centered_b = h_cloud_b - h_cloud_b.mean(axis=0)
                    U_b, S_b, Vt_b = np.linalg.svd(cloud_centered_b, full_matrices=False)
                    K_b = min(32, hidden_dim, n_tokens_b - 1)
                    principal_dirs_b = Vt_b[:K_b]
                else:
                    K_b = min(32, hidden_dim)
                    principal_dirs_b = np.eye(hidden_dim)[:K_b]

                deltas_cloud_b = np.stack([
                    (hs_b[lay + 1][0, t] - hs_b[lay][0, t]).cpu().float().numpy()
                    for t in range(n_tokens_b)
                ], axis=0)

                if n_tokens_b >= 3:
                    pos_proj_b = cloud_centered_b @ principal_dirs_b.T
                    del_centered_b = deltas_cloud_b - deltas_cloud_b.mean(axis=0)
                    del_proj_b = del_centered_b @ principal_dirs_b.T
                    try:
                        J_proj_b = np.linalg.lstsq(pos_proj_b, del_proj_b, rcond=None)[0].T
                    except Exception:
                        J_proj_b = np.eye(K_b)
                else:
                    J_proj_b = np.eye(K_b)

                eigenvalues_b, eigenvectors_b = np.linalg.eig(J_proj_b)
                sort_idx_b = np.argsort(-np.abs(eigenvalues_b))
                eigenvalues_b = eigenvalues_b[sort_idx_b]

                eig_magnitude_b = np.abs(eigenvalues_b)
                eig_phase_b = np.angle(eigenvalues_b)

                divergence_b = float(np.real(np.sum(eigenvalues_b)))
                J_antisym_b = (J_proj_b - J_proj_b.T) / 2
                curl_b = float(np.linalg.norm(J_antisym_b, 'fro'))
                J_sym_b = (J_proj_b + J_proj_b.T) / 2
                J_traceless_b = J_sym_b - np.eye(K_b) * np.trace(J_sym_b) / K_b
                shear_b = float(np.linalg.norm(J_traceless_b, 'fro'))

                sv_b = np.linalg.svd(J_proj_b, compute_uv=False)
                condition_b = float(sv_b[0] / max(sv_b[-1], 1e-12))
                sv_norm_b = sv_b / max(sv_b.sum(), 1e-12)
                sv_norm_b = sv_norm_b[sv_norm_b > 1e-12]
                effective_rank_b = float(np.exp(-np.sum(sv_norm_b * np.log(sv_norm_b))))

                # Compute the differential spectrum
                layer_diff.append({
                    "token_a": tokens[ti] if ti < len(tokens) else "?",
                    "token_b": tokens_b[ti] if ti < len(tokens_b) else "?",
                    "delta_norm_a": round(float(np.linalg.norm(
                        (hs[lay + 1][0, ti] - hs[lay][0, ti]).cpu().float().numpy()
                    )), 6),
                    "delta_norm_b": round(delta_b_norm, 6),
                    # Divergence difference
                    "divergence_diff": round(spec_a["divergence"] - divergence_b, 6),
                    "curl_diff": round(spec_a["curl"] - curl_b, 6),
                    "shear_diff": round(spec_a["shear"] - shear_b, 6),
                    "condition_diff": round(spec_a["condition_number"] - condition_b, 4),
                    "effective_rank_diff": round(spec_a["effective_rank"] - effective_rank_b, 4),
                    # Eigenvalue spectrum distance
                    "eigenvalue_magnitude_diff": [
                        round(float(a - b), 6)
                        for a, b in zip(
                            spec_a["eigenvalues_magnitude"][:16],
                            eig_magnitude_b[:16].tolist()
                        )
                    ],
                    "eigenvalue_phase_diff": [
                        round(float(a - b), 6)
                        for a, b in zip(
                            spec_a["eigenvalues_phase"][:16],
                            eig_phase_b[:16].tolist()
                        )
                    ],
                    # Spectral distance: Frobenius norm of Jacobian difference
                    # This requires matching the projected spaces, so we use
                    # the eigenvalue spectra as a proxy
                    "spectral_distance": round(float(np.linalg.norm(
                        np.array(spec_a["eigenvalues_magnitude"][:min(K, K_b)]) -
                        eig_magnitude_b[:min(K, K_b)]
                    )), 6),
                })

            diff_spectra["layer_diffs"].append(layer_diff)

        # ================================================================
        # Compute aggregate differential metrics
        # ================================================================
        diff_summary = {
            "per_layer_spectral_distance": [],
            "per_layer_divergence_diff": [],
            "per_layer_curl_diff": [],
            "per_layer_shear_diff": [],
            "onset_layer": -1,
            "max_spectral_distance_layer": 0,
            "max_spectral_distance": 0.0,
        }

        max_sd = 0.0
        max_sd_layer = 0
        baseline_sd = None

        for lay_idx, layer_diff in enumerate(diff_spectra["layer_diffs"]):
            if not layer_diff:
                diff_summary["per_layer_spectral_distance"].append(0.0)
                diff_summary["per_layer_divergence_diff"].append(0.0)
                diff_summary["per_layer_curl_diff"].append(0.0)
                diff_summary["per_layer_shear_diff"].append(0.0)
                continue

            # Average spectral distance across aligned tokens
            avg_sd = float(np.mean([td["spectral_distance"] for td in layer_diff]))
            avg_div_diff = float(np.mean([abs(td["divergence_diff"]) for td in layer_diff]))
            avg_curl_diff = float(np.mean([abs(td["curl_diff"]) for td in layer_diff]))
            avg_shear_diff = float(np.mean([abs(td["shear_diff"]) for td in layer_diff]))

            diff_summary["per_layer_spectral_distance"].append(round(avg_sd, 6))
            diff_summary["per_layer_divergence_diff"].append(round(avg_div_diff, 6))
            diff_summary["per_layer_curl_diff"].append(round(avg_curl_diff, 6))
            diff_summary["per_layer_shear_diff"].append(round(avg_shear_diff, 6))

            if avg_sd > max_sd:
                max_sd = avg_sd
                max_sd_layer = lay_idx

            # Detect onset: first layer where spectral distance exceeds 2x the first layer
            if baseline_sd is None:
                baseline_sd = avg_sd
            elif diff_summary["onset_layer"] < 0 and baseline_sd > 1e-8:
                if avg_sd > baseline_sd * 2.5:
                    diff_summary["onset_layer"] = lay_idx

        diff_summary["max_spectral_distance_layer"] = max_sd_layer
        diff_summary["max_spectral_distance"] = round(max_sd, 6)
        diff_spectra["summary"] = diff_summary

    return json.dumps({
        "tokens": tokens,
        "tokens_b": tokens_b if text_b else None,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "K_projected": K,
        "layer_spectra": layer_spectra,
        "anomalies": anomalies[:50],
        "diff_spectra": diff_spectra,
    }, cls=SafeFloatEncoder).encode()

def estimate_fiber_curvature(hidden_states, k_neighbors=8, pca_d=16):
    """
    Extract Sectional and Ricci Curvature from transformer hidden states,
    treating the model as a principal fiber bundle E -> M.

    Args:
        hidden_states: tuple of tensors from model output, shape (1, seq_len, hidden_dim) each.
                       hidden_states[0] = embedding, hidden_states[l+1] = after layer l.
        k_neighbors: number of neighbors for local metric / ORC computation.
        pca_d: number of principal components for tangent space approximation.

    Returns:
        dict with keys:
            'ollivier_ricci':       np.ndarray of shape [n_layers, seq_len] — ORC per token per layer
            'scalar_curvature':     np.ndarray of shape [n_layers, seq_len] — volumetric strain proxy
            'metric_log_det':       np.ndarray of shape [n_layers+1, seq_len] — log(det(g)) per layer
            'procrustes_deviation': np.ndarray of shape [n_layers, seq_len] — ||R - I||_F per token
            'metric_eigenvalues':   list of list of np.ndarray — eigenvalues of local Gram matrix
            'sectional_curvature':  np.ndarray of shape [n_layers, seq_len] — sectional curvature proxy
    """
    n_layers_plus_one = len(hidden_states)
    n_layers = n_layers_plus_one - 1
    seq_len = hidden_states[0].shape[1]
    hidden_dim = hidden_states[0].shape[2]

    # Convert all hidden states to numpy: [n_layers+1, seq_len, hidden_dim]
    H = np.stack([
        hidden_states[lay][0].cpu().float().numpy() for lay in range(n_layers_plus_one)
    ], axis=0)  # (n_layers+1, seq_len, hidden_dim)

    k = min(k_neighbors, seq_len - 1)
    d = min(pca_d, hidden_dim, seq_len - 1)

    # ================================================================
    # STAGE 1: Metric Reconstruction (Local Gram / Covariance Matrix)
    # For each layer l and token i, define N(i) via k-NN,
    # compute local covariance -> eigenvalues = local "stretching"
    # ================================================================
    metric_eigenvalues = []   # [n_layers+1][seq_len] -> eigenvalue arrays
    metric_log_det = np.zeros((n_layers_plus_one, seq_len))

    for lay in range(n_layers_plus_one):
        layer_eigs = []
        # Pairwise distances for this layer
        dists = cdist(H[lay], H[lay], metric='euclidean')  # (seq_len, seq_len)

        for i in range(seq_len):
            # k-NN neighborhood (exclude self)
            neighbor_idx = np.argsort(dists[i])[1:k + 1]
            neighborhood = H[lay][neighbor_idx]  # (k, hidden_dim)

            # Local covariance (Gram) matrix
            centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
            # Use the top-d PCA subspace for tractability
            if centered.shape[0] >= 2:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                eigs = (S[:d] ** 2) / max(len(neighbor_idx) - 1, 1)
            else:
                eigs = np.ones(min(d, 1))

            layer_eigs.append(eigs)

            # log(det(g)) ≈ sum of log(eigenvalues) — the local volume element
            eigs_safe = np.clip(eigs, 1e-30, None)
            metric_log_det[lay, i] = np.sum(np.log(eigs_safe))

        metric_eigenvalues.append(layer_eigs)

    # ================================================================
    # STAGE 2: Parallel Transport & Connection (Procrustes Alignment)
    # For each token, align tangent spaces (top-d PCs) between layer l and l+1.
    # Deviation of R from I = local "twist" of the bundle.
    # ================================================================
    procrustes_deviation = np.zeros((n_layers, seq_len))
    procrustes_rotations = []  # store for sectional curvature

    for lay in range(n_layers):
        layer_rotations = []
        dists_l = cdist(H[lay], H[lay], metric='euclidean')
        dists_l1 = cdist(H[lay + 1], H[lay + 1], metric='euclidean')

        for i in range(seq_len):
            # Get neighborhoods at layer lay and lay+1
            nb_l = np.argsort(dists_l[i])[1:k + 1]
            nb_l1 = np.argsort(dists_l1[i])[1:k + 1]

            # Use the SAME token indices for both layers to track the fiber
            # (parallel transport along the "layer" direction)
            common_nb = np.intersect1d(nb_l, nb_l1)
            if len(common_nb) < d:
                # Fall back to using the layer-lay neighborhood
                common_nb = nb_l[:max(d, len(common_nb))]

            if len(common_nb) < 2:
                layer_rotations.append(np.eye(d))
                procrustes_deviation[lay, i] = 0.0
                continue

            # Local tangent spaces via PCA
            cloud_l = H[lay][common_nb] - H[lay][i]
            cloud_l1 = H[lay + 1][common_nb] - H[lay + 1][i]

            def get_tangent_basis(cloud, dim):
                if cloud.shape[0] < 2:
                    return np.eye(cloud.shape[1])[:dim]
                U, S, Vt = np.linalg.svd(cloud, full_matrices=False)
                return Vt[:dim]

            T_l = get_tangent_basis(cloud_l, d)    # (d, hidden_dim)
            T_l1 = get_tangent_basis(cloud_l1, d)  # (d, hidden_dim)

            # Project neighborhoods into their respective tangent spaces
            proj_l = cloud_l @ T_l.T     # (n_common, d)
            proj_l1 = cloud_l1 @ T_l1.T  # (n_common, d)

            # Truncate to same number of rows
            n_use = min(proj_l.shape[0], proj_l1.shape[0], d)
            if n_use < 2:
                layer_rotations.append(np.eye(d))
                procrustes_deviation[lay, i] = 0.0
                continue

            proj_l_use = proj_l[:n_use, :d]
            proj_l1_use = proj_l1[:n_use, :d]

            # Orthogonal Procrustes: find R such that ||proj_l1_use - proj_l_use @ R||_F is minimized
            try:
                R, scale = orthogonal_procrustes(proj_l_use, proj_l1_use)
            except (np.linalg.LinAlgError, ValueError):
                R = np.eye(d)

            layer_rotations.append(R)

            # Deviation from identity = magnitude of the connection
            procrustes_deviation[lay, i] = np.linalg.norm(R - np.eye(d), 'fro')

        procrustes_rotations.append(layer_rotations)

    # ================================================================
    # STAGE 3a: Ollivier-Ricci Curvature (ORC)
    # For each pair of neighboring tokens at each layer,
    # compute W1 distance between their softmax-weighted distributions.
    # ORC(x,y) = 1 - W1(mu_x, mu_y) / d(x,y)
    # ================================================================
    ollivier_ricci = np.zeros((n_layers_plus_one, seq_len))

    for lay in range(n_layers_plus_one):
        dists = cdist(H[lay], H[lay], metric='euclidean')

        # Build softmax-weighted distributions for each token
        # mu_i(j) = softmax(-dists[i] / temperature) — a "lazy random walk"
        temperature = np.median(dists[dists > 0]) + 1e-12
        distributions = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            weights = np.exp(-dists[i] / temperature)
            weights[i] = 0  # exclude self for the transport
            w_sum = weights.sum()
            if w_sum > 1e-15:
                distributions[i] = weights / w_sum
            else:
                distributions[i] = np.ones(seq_len) / (seq_len - 1)
                distributions[i, i] = 0

        # For each token, compute average ORC with its k neighbors
        for i in range(seq_len):
            neighbor_idx = np.argsort(dists[i])[1:k + 1]
            orc_vals = []
            for j in neighbor_idx:
                d_ij = dists[i, j]
                if d_ij < 1e-12:
                    orc_vals.append(0.0)
                    continue

                # W1 distance between distributions[i] and distributions[j]
                # using the ground metric dists
                # For efficiency, use 1D Wasserstein on sorted marginals
                # projected onto the i-j axis
                proj_i = dists[i]  # distances from i to all tokens
                proj_j = dists[j]  # distances from j to all tokens

                w1 = wasserstein_distance(
                    proj_i, proj_j,
                    u_weights=distributions[i],
                    v_weights=distributions[j]
                )

                orc = 1.0 - w1 / d_ij
                orc_vals.append(orc)

            ollivier_ricci[lay, i] = np.mean(orc_vals) if orc_vals else 0.0

    # ================================================================
    # STAGE 3b: Scalar Curvature Proxy (Volumetric Strain)
    # Measure how the volume of a d-simplex changes across layers.
    # ================================================================
    scalar_curvature = np.zeros((n_layers, seq_len))

    for lay in range(n_layers):
        for i in range(seq_len):
            dists_l = cdist(H[lay], H[lay], metric='euclidean')
            nb = np.argsort(dists_l[i])[1:d + 2]  # d+1 neighbors for a d-simplex

            if len(nb) < 2:
                scalar_curvature[lay, i] = 0.0
                continue

            # Simplex at layer lay
            simplex_l = H[lay][nb] - H[lay][i]
            # Simplex at layer lay+1
            simplex_l1 = H[lay + 1][nb] - H[lay + 1][i]

            # Volume proxy: det of Gram matrix (or its log)
            def log_volume(vecs):
                G = vecs @ vecs.T  # Gram matrix
                sign, logdet = np.linalg.slogdet(G)
                return 0.5 * logdet if sign > 0 else -50.0  # half because vol = sqrt(det(G))

            vol_l = log_volume(simplex_l)
            vol_l1 = log_volume(simplex_l1)

            # Volumetric strain = change in log-volume
            scalar_curvature[lay, i] = vol_l1 - vol_l

    # ================================================================
    # STAGE 3c: Sectional Curvature Proxy
    # From the Procrustes rotations, estimate the curvature of the
    # connection by looking at the "holonomy" around small loops.
    # Sectional curvature ~ ||[R_l, R_{l+1}]|| for consecutive layers
    # ================================================================
    sectional_curvature = np.zeros((n_layers, seq_len))

    for lay in range(n_layers):
        for i in range(seq_len):
            R_l = procrustes_rotations[lay][i] if lay < len(procrustes_rotations) else np.eye(d)

            # Use the Procrustes deviation directly as a curvature proxy
            # (the antisymmetric part of R encodes the local rotation = curvature)
            R_antisym = (R_l - R_l.T) / 2
            sectional_curvature[lay, i] = np.linalg.norm(R_antisym, 'fro')

            # If we have two consecutive layers, compute the commutator
            if lay + 1 < n_layers and lay + 1 < len(procrustes_rotations):
                R_l1 = procrustes_rotations[lay + 1][i] if i < len(procrustes_rotations[lay + 1]) else np.eye(d)
                # Ensure compatible shapes
                min_d = min(R_l.shape[0], R_l1.shape[0])
                R_l_sub = R_l[:min_d, :min_d]
                R_l1_sub = R_l1[:min_d, :min_d]
                commutator = R_l_sub @ R_l1_sub - R_l1_sub @ R_l_sub
                sectional_curvature[lay, i] += np.linalg.norm(commutator, 'fro')

    return {
        'ollivier_ricci': ollivier_ricci,           # [n_layers+1, seq_len]
        'scalar_curvature': scalar_curvature,        # [n_layers, seq_len]
        'metric_log_det': metric_log_det,            # [n_layers+1, seq_len]
        'procrustes_deviation': procrustes_deviation, # [n_layers, seq_len]
        'metric_eigenvalues': metric_eigenvalues,     # list of lists
        'sectional_curvature': sectional_curvature,   # [n_layers, seq_len]
    }

def handle_multi_run(body_bytes):
    """Process multiple sentences independently and return comparison data."""
    req = json.loads(body_bytes)
    sentences = req.get("sentences", [])
    model_name = req.get("model", "").strip() or None
    itp_method = req.get("itp_method", "rbf").strip()

    if not sentences or len(sentences) < 2:
        return json.dumps({"error": "Need at least 2 sentences"}).encode()

    if model_name and model_name != MODEL_NAME:
        load_model(model_name)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)

    all_results = []

    for idx, text in enumerate(sentences):
        text = text.strip()
        if not text:
            continue
        print(f"[MultiRun] Processing sentence {idx+1}/{len(sentences)}: {text[:60]}...")

        input_ids, tokens = tokenize_text(TOKENIZER, text)
        hs = extract_hidden_states(MODEL, input_ids)
        n_tokens = input_ids.shape[1]

        # Collect per-layer hidden states as numpy arrays
        # Shape per layer: (seq_len, hidden_dim)
        layer_activations = []
        for lay in range(n_layers + 1):
            layer_activations.append(hs[lay][0].cpu().float().numpy())

        # Compute mean activation per dimension per layer (aggregated across tokens)
        # This gives a "sentence fingerprint" per layer: (n_layers+1, hidden_dim)
        mean_per_layer = np.stack([la.mean(axis=0) for la in layer_activations], axis=0)

        # Also compute max absolute activation per dimension per layer
        max_abs_per_layer = np.stack([np.abs(la).max(axis=0) for la in layer_activations], axis=0)

        # Compute per-layer norms (L2 of mean activation)
        layer_norms = [float(np.linalg.norm(mean_per_layer[lay])) for lay in range(n_layers + 1)]

        all_results.append({
            "index": idx,
            "text": text,
            "tokens": tokens,
            "n_tokens": n_tokens,
            "mean_per_layer": mean_per_layer.tolist(),        # (n_layers+1, hidden_dim)
            "max_abs_per_layer": max_abs_per_layer.tolist(),  # (n_layers+1, hidden_dim)
            "layer_norms": layer_norms,
        })

    if len(all_results) < 2:
        return json.dumps({"error": "Need at least 2 non-empty sentences"}).encode()

    # ================================================================
    # CROSS-SENTENCE COMPARISON
    # ================================================================
    n_sentences = len(all_results)

    # For each layer, compute pairwise dimension-wise differences
    # and find which dimensions differ the most / least
    layer_comparisons = []

    for lay in range(n_layers + 1):
        # Collect mean activations for all sentences at this layer
        # Shape: (n_sentences, hidden_dim)
        means = np.stack([
            np.array(all_results[si]["mean_per_layer"][lay])
            for si in range(n_sentences)
        ], axis=0)

        # Variance across sentences per dimension
        dim_variance = np.var(means, axis=0)  # (hidden_dim,)

        # Mean across sentences per dimension
        dim_mean = np.mean(means, axis=0)  # (hidden_dim,)

        # Range (max - min) across sentences per dimension
        dim_range = np.ptp(means, axis=0)  # (hidden_dim,)

        # Top-K most differing dimensions
        top_k = min(50, hidden_dim)
        most_different_dims = np.argsort(-dim_variance)[:top_k]
        least_different_dims = np.argsort(dim_variance)[:top_k]

        # Pairwise cosine similarities between sentences at this layer
        from scipy.spatial.distance import cosine as cosine_dist
        pairwise_cosine = np.zeros((n_sentences, n_sentences))
        for i in range(n_sentences):
            for j in range(n_sentences):
                if i == j:
                    pairwise_cosine[i, j] = 1.0
                else:
                    pairwise_cosine[i, j] = 1.0 - cosine_dist(means[i], means[j])

        # Pairwise L2 distances
        pairwise_l2 = cdist(means, means, metric='euclidean')

        layer_comparisons.append({
            "layer": lay,
            "dim_variance": dim_variance.tolist(),
            "dim_mean": dim_mean.tolist(),
            "dim_range": dim_range.tolist(),
            "most_different_dims": [
                {
                    "dim": int(d),
                    "variance": round(float(dim_variance[d]), 8),
                    "range": round(float(dim_range[d]), 6),
                    "values": [round(float(means[si][d]), 6) for si in range(n_sentences)]
                }
                for d in most_different_dims
            ],
            "least_different_dims": [
                {
                    "dim": int(d),
                    "variance": round(float(dim_variance[d]), 8),
                    "range": round(float(dim_range[d]), 6),
                    "values": [round(float(means[si][d]), 6) for si in range(n_sentences)]
                }
                for d in least_different_dims
            ],
            "pairwise_cosine": pairwise_cosine.tolist(),
            "pairwise_l2": pairwise_l2.tolist(),
            "total_variance": round(float(np.sum(dim_variance)), 6),
        })

    # ================================================================
    # GLOBAL SUMMARY: Which layers show the most cross-sentence divergence?
    # ================================================================
    layer_total_variances = [lc["total_variance"] for lc in layer_comparisons]
    most_divergent_layer = int(np.argmax(layer_total_variances))
    least_divergent_layer = int(np.argmin(layer_total_variances))

    # Global top dimensions (across all layers)
    global_dim_variance = np.zeros(hidden_dim)
    for lc in layer_comparisons:
        global_dim_variance += np.array(lc["dim_variance"])
    global_most_different = np.argsort(-global_dim_variance)[:50]
    global_least_different = np.argsort(global_dim_variance)[:50]

    response = {
        "n_sentences": n_sentences,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model_name": MODEL_NAME,
        "sentences": all_results,
        "layer_comparisons": layer_comparisons,
        "summary": {
            "most_divergent_layer": most_divergent_layer,
            "least_divergent_layer": least_divergent_layer,
            "layer_total_variances": [round(v, 6) for v in layer_total_variances],
            "global_most_different_dims": [
                {"dim": int(d), "total_variance": round(float(global_dim_variance[d]), 8)}
                for d in global_most_different[:20]
            ],
            "global_least_different_dims": [
                {"dim": int(d), "total_variance": round(float(global_dim_variance[d]), 8)}
                for d in global_least_different[:20]
            ],
        }
    }

    full_results = []
    for idx, text in enumerate(sentences):
        text = text.strip()
        if not text:
            continue
        json_str = process_text(text, model_name, itp_method=itp_method)
        full_results.append(json.loads(json_str))

    # Return both the comparison data AND the full per-sentence data
    response["sentence_data"] = full_results  # <-- ADD THIS

    return json.dumps(response, cls=SafeFloatEncoder).encode()

def compute_persistence_entropy(diagrams_by_dim, max_dim):
    """Compute the persistence entropy of a persistence diagram."""
    lifetimes = []
    for dim in range(max_dim + 1):
        pairs = diagrams_by_dim.get(dim, [])
        for pair in pairs:
            b, d = pair[0], pair[1]
            if d == float('inf'):
                continue  # skip infinite persistence features
            lifetime = d - b
            if lifetime > 1e-12:
                lifetimes.append(lifetime)

    if len(lifetimes) == 0:
        return 0.0

    lifetimes = np.array(lifetimes)
    total = lifetimes.sum()
    if total < 1e-12:
        return 0.0

    probs = lifetimes / total
    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-30))
    return float(entropy)


def compute_wasserstein_matrix(persistence_diagrams, max_dim):
    """
    Compute pairwise Wasserstein distances between persistence diagrams
    at different layers.

    Returns:
        matrix: list of lists (n_layers x n_layers) of float distances
    """
    n_layers = len(persistence_diagrams)
    matrix = [[0.0] * n_layers for _ in range(n_layers)]

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            total_dist = 0.0
            for dim in range(max_dim + 1):
                pairs_i = persistence_diagrams[i].get(dim, [])
                pairs_j = persistence_diagrams[j].get(dim, [])

                # Extract finite-persistence lifetimes for Wasserstein comparison
                lifetimes_i = []
                lifetimes_j = []
                for p in pairs_i:
                    lt = p[1] - p[0] if p[1] != float('inf') else 0.0
                    if lt > 1e-12:
                        lifetimes_i.append(lt)
                for p in pairs_j:
                    lt = p[1] - p[0] if p[1] != float('inf') else 0.0
                    if lt > 1e-12:
                        lifetimes_j.append(lt)

                if len(lifetimes_i) > 0 and len(lifetimes_j) > 0:
                    try:
                        dist = wasserstein_distance(lifetimes_i, lifetimes_j)
                        total_dist += dist
                    except Exception:
                        pass
                elif len(lifetimes_i) > 0 or len(lifetimes_j) > 0:
                    # One is empty — distance is the total persistence of the other
                    total_dist += sum(lifetimes_i) + sum(lifetimes_j)

            matrix[i][j] = round(total_dist, 6)
            matrix[j][i] = round(total_dist, 6)

    return matrix


def detect_topological_events(persistence_diagrams, layer_summaries, max_dim):
    """
    Detect significant topological events across layers:
    - Birth of new persistent features
    - Death of features
    - Peaks in persistence

    Returns:
        list of event dicts
    """
    events = []
    n_layers = len(layer_summaries)

    for li in range(1, n_layers):
        prev = layer_summaries[li - 1]
        curr = layer_summaries[li]

        # Detect changes in Betti numbers
        for dim, (betti_key, dim_label) in enumerate([
            ('betti_0', 'H₀ (components)'),
            ('betti_1', 'H₁ (loops)'),
            ('betti_2', 'H₂ (voids)')
        ]):
            if dim > max_dim:
                break

            prev_b = prev.get(betti_key, 0)
            curr_b = curr.get(betti_key, 0)

            if curr_b > prev_b:
                events.append({
                    'layer': li,
                    'type': 'birth',
                    'dim': dim,
                    'description': f'{dim_label}: β increased from {prev_b} to {curr_b}'
                })
            elif curr_b < prev_b:
                events.append({
                    'layer': li,
                    'type': 'death',
                    'dim': dim,
                    'description': f'{dim_label}: β decreased from {prev_b} to {curr_b}'
                })

        # Detect entropy peaks
        if li >= 2:
            prev_prev = layer_summaries[li - 2]
            if (curr['entropy'] > prev['entropy'] and
                curr['entropy'] > prev_prev['entropy'] and
                curr['entropy'] > 0.1):
                events.append({
                    'layer': li,
                    'type': 'persistence_peak',
                    'dim': -1,
                    'description': f'Persistence entropy peak: {curr["entropy"]:.4f}'
                })

    # Sort by layer
    events.sort(key=lambda e: (e['layer'], e['dim']))

    return events


def generate_tda_summary(layer_summaries, events, max_dim):
    """Generate a human-readable summary of the TDA analysis."""
    n_layers = len(layer_summaries)

    # Find layer with most topological complexity
    max_features_layer = max(range(n_layers), key=lambda i: layer_summaries[i]['n_features'])
    max_entropy_layer = max(range(n_layers), key=lambda i: layer_summaries[i]['entropy'])

    # Find layer with most loops
    max_loops_layer = max(range(n_layers), key=lambda i: layer_summaries[i]['betti_1'])

    summary_parts = []

    summary_parts.append(
        f"Analyzed {n_layers} layers (embedding + {n_layers - 1} transformer layers)."
    )

    mf = layer_summaries[max_features_layer]
    summary_parts.append(
        f"Most topologically complex: {mf['name']} "
        f"({mf['n_features']} features, β₀={mf['betti_0']}, β₁={mf['betti_1']})."
    )

    me = layer_summaries[max_entropy_layer]
    summary_parts.append(
        f"Highest persistence entropy: {me['name']} ({me['entropy']:.4f})."
    )

    ml = layer_summaries[max_loops_layer]
    if ml['betti_1'] > 0:
        summary_parts.append(
            f"Most 1-cycles (loops): {ml['name']} (β₁={ml['betti_1']})."
        )

    n_births = sum(1 for e in events if e['type'] == 'birth')
    n_deaths = sum(1 for e in events if e['type'] == 'death')
    if n_births > 0 or n_deaths > 0:
        summary_parts.append(
            f"Topological events: {n_births} births, {n_deaths} deaths across layers."
        )

    return " ".join(summary_parts)


def handle_tda(body_bytes):
    """
    Endpoint handler for /tda — computes persistent homology across all layers.
    """
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    max_dim = req.get("max_dim", 2)
    pca_dims = req.get("pca_dims", 16)
    max_edge = req.get("max_edge", 10.0)
    collapse_eps = req.get("collapse_eps", 0.0)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    # Tokenize and extract hidden states
    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    print(f"[TDA] Computing persistent homology for '{text[:50]}...' "
          f"(max_dim={max_dim}, pca_dims={pca_dims}, max_edge={max_edge})")

    try:
        result = compute_persistent_homology(
            hs,
            max_dim=max_dim,
            pca_dims=pca_dims,
            max_edge=max_edge,
            collapse_eps=collapse_eps,
        )
    except ImportError as e:
        return json.dumps({"error": str(e)}).encode()
    except Exception as e:
        traceback.print_exc()
        return json.dumps({"error": f"TDA computation failed: {e}"}).encode()

    # Convert persistence diagrams: replace Python inf with a large number for JSON
    json_diagrams = []
    for layer_diags in result['persistence_diagrams']:
        json_layer = {}
        for dim_key, pairs in layer_diags.items():
            json_pairs = []
            for pair in pairs:
                b = pair[0]
                d = pair[1] if pair[1] != float('inf') else 1e18
                json_pairs.append([round(b, 8), round(d, 8)])
            json_layer[dim_key] = json_pairs
        json_diagrams.append(json_layer)

    result['persistence_diagrams'] = json_diagrams

    return json.dumps(result, cls=SafeFloatEncoder).encode()

def extract_pure_jacobian_field(hidden_states, n_layers, pca_d=16):
    """
    Extract the layer-to-layer Jacobian field as a standalone object.
    Returns the transformation itself, not the data it acts on.
    """
    results = []
    for lay in range(n_layers):
        h_cloud = hidden_states[lay][0].cpu().float().numpy()
        hidden_states[lay + 1][0].cpu().float().numpy()

        # The Jacobian IS the morphing
        K, principal_dirs = _compute_principal_directions(h_cloud, h_cloud.shape[0], h_cloud.shape[1])
        J = _estimate_jacobian_from_cloud(hidden_states, lay, h_cloud.shape[0], h_cloud.shape[1], K, principal_dirs, h_cloud)

        # Decompose into pure geometric operations
        eigenvalues, eigenvectors = np.linalg.eig(J)
        J_sym = (J + J.T) / 2      # pure stretch
        J_antisym = (J - J.T) / 2   # pure rotation

        results.append({
            'jacobian': J,
            'stretch': J_sym,
            'rotation': J_antisym,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'divergence': float(np.real(np.trace(J))),
            'curl_magnitude': float(np.linalg.norm(J_antisym, 'fro')),
            'det': float(np.real(np.linalg.det(J))),
        })
    return results

def compute_holonomy_loop(procrustes_rotations, token_i, token_j, layer_start, layer_end, d):
    """
    Transport a frame: token_i@layer_start -> token_i@layer_end ->
    token_j@layer_end -> token_j@layer_start -> token_i@layer_start

    The residual rotation IS the curvature, with no data attached.
    """
    R_total = np.eye(d)
    # Up through layers at token i
    for lay in range(layer_start, layer_end):
        R_total = procrustes_rotations[lay][token_i] @ R_total
    # Across tokens at top layer (would need token-to-token transport)
    # Down through layers at token j
    for lay in range(layer_end - 1, layer_start - 1, -1):
        R_total = procrustes_rotations[lay][token_j].T @ R_total

    # R_total should be identity if space is flat
    # The deviation IS the curvature
    holonomy_angle = np.arccos(np.clip((np.trace(R_total) - 1) / 2, -1, 1))
    return R_total, holonomy_angle

def extract_eigenvalue_flow(hidden_states, n_layers):
    """
    Track how the eigenvalue spectrum of the layer map evolves.
    This IS the morphing, viewed as a flow through 'transformation space'.
    """
    flow = []
    for lay in range(n_layers):
        h = hidden_states[lay][0].cpu().float().numpy()
        hidden_states[lay + 1][0].cpu().float().numpy()

        # Compute the map's spectrum
        K, dirs = _compute_principal_directions(h, h.shape[0], h.shape[1])
        J = _estimate_jacobian_from_cloud(hidden_states, lay, h.shape[0], h.shape[1], K, dirs, h)

        eigs = np.linalg.eigvals(J)
        sv = np.linalg.svd(J, compute_uv=False)

        flow.append({
            'layer': lay,
            'eigenvalue_magnitudes': np.abs(eigs).tolist(),
            'eigenvalue_phases': np.angle(eigs).tolist(),
            'singular_values': sv.tolist(),
            'divergence': float(np.real(np.trace(J))),          # <-- MUST be present
            'log_det': float(np.sum(np.log(np.abs(eigs) + 1e-30))),  # <-- MUST be present
            'spectral_gap': float(sv[0] / max(sv[-1], 1e-12)),  # <-- MUST be present
        })
    return flow

def handle_morphing_analysis(body_bytes):
    """
    Extract pure morphing data: Jacobian field, eigenvalue flow,
    holonomy loops, and connection 1-form.
    """
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    pca_d = req.get("pca_d", 16)
    k_neighbors = req.get("k_neighbors", 8)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    seq_len = input_ids.shape[1]

    print(f"[Morphing] Extracting pure morphings for {seq_len} tokens × {n_layers} layers...")

    # 1. Eigenvalue flow
    eigenvalue_flow = extract_eigenvalue_flow(hs, n_layers)

    # 2. Jacobian field decomposition
    jacobian_field_raw = extract_pure_jacobian_field(hs, n_layers, pca_d=pca_d)
    jacobian_field = []
    for jf in jacobian_field_raw:
        jacobian_field.append({
            'stretch': jf['stretch'].tolist(),
            'rotation': jf['rotation'].tolist(),
            'eigenvalue_magnitudes': np.abs(jf['eigenvalues']).tolist(),
            'eigenvalue_phases': np.angle(jf['eigenvalues']).tolist(),
            'divergence': jf['divergence'],
            'curl_magnitude': jf['curl_magnitude'],
            'det': jf['det'],
        })

    # 3. Holonomy loops (approximate from Procrustes deviations)
    curvature_data = estimate_fiber_curvature(hs, k_neighbors=k_neighbors, pca_d=pca_d)
    proc_dev = curvature_data['procrustes_deviation']  # (n_layers, seq_len)

    holonomy_loops = []
    max_pairs = min(seq_len, 8)
    for ti in range(max_pairs):
        for tj in range(ti + 1, max_pairs):
            total_dev = 0.0
            for lay in range(n_layers):
                total_dev += proc_dev[lay, ti] + proc_dev[lay, tj]
            holonomy_angle = float(np.arctan(total_dev / max(n_layers * 2, 1)))
            holonomy_loops.append({
                'token_i': int(ti),
                'token_j': int(tj),
                'token_i_str': tokens[ti],
                'token_j_str': tokens[tj],
                'layer_start': 0,
                'layer_end': n_layers,
                'holonomy_angle': round(holonomy_angle, 6),
                'frobenius_deviation': round(float(total_dev), 6),
            })
    holonomy_loops.sort(key=lambda x: -x['holonomy_angle'])

    # 4. Connection 1-form (approximate via Procrustes rotation matrices)
    # We use the Procrustes deviation as a proxy for ||log(R)||
    # and build per-token per-layer connection matrices from the curvature data
    connection_field = []
    d = min(pca_d, hidden_dim, seq_len - 1)

    # Re-extract Procrustes rotations for the connection field
    H = np.stack([
        hs[lay][0].cpu().float().numpy() for lay in range(n_layers + 1)
    ], axis=0)

    k = min(k_neighbors, seq_len - 1)

    for lay in range(n_layers):
        layer_conn = []
        dists_l = cdist(H[lay], H[lay], metric='euclidean')
        dists_l1 = cdist(H[lay + 1], H[lay + 1], metric='euclidean')

        for i in range(seq_len):
            nb_l = np.argsort(dists_l[i])[1:k + 1]
            nb_l1 = np.argsort(dists_l1[i])[1:k + 1]
            common_nb = np.intersect1d(nb_l, nb_l1)
            if len(common_nb) < d:
                common_nb = nb_l[:max(d, len(common_nb))]

            if len(common_nb) < 2:
                # Identity connection (zero Lie algebra element)
                A = np.zeros((d, d))
                layer_conn.append(A.tolist())
                continue

            cloud_l = H[lay][common_nb] - H[lay][i]
            cloud_l1 = H[lay + 1][common_nb] - H[lay + 1][i]

            def get_tangent_basis(cloud, dim):
                if cloud.shape[0] < 2:
                    return np.eye(cloud.shape[1])[:dim]
                U, S, Vt = np.linalg.svd(cloud, full_matrices=False)
                return Vt[:dim]

            T_l = get_tangent_basis(cloud_l, d)
            T_l1 = get_tangent_basis(cloud_l1, d)

            proj_l = cloud_l @ T_l.T
            proj_l1 = cloud_l1 @ T_l1.T

            n_use = min(proj_l.shape[0], proj_l1.shape[0], d)
            if n_use < 2:
                A = np.zeros((d, d))
                layer_conn.append(A.tolist())
                continue

            proj_l_use = proj_l[:n_use, :d]
            proj_l1_use = proj_l1[:n_use, :d]

            try:
                R, _ = orthogonal_procrustes(proj_l_use, proj_l1_use)
                # Connection = log(R) — the Lie algebra element
                from scipy.linalg import logm
                A = logm(R).real
            except Exception:
                A = np.zeros((d, d))

            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            layer_conn.append(A.tolist())

        connection_field.append(layer_conn)

    print(f"[Morphing] Done: {len(eigenvalue_flow)} eigenflow layers, "
          f"{len(jacobian_field)} Jacobian layers, "
          f"{len(holonomy_loops)} holonomy loops, "
          f"{len(connection_field)} connection layers")

    response = {
        "tokens": tokens,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "eigenvalue_flow": eigenvalue_flow,
        "jacobian_field": jacobian_field,
        "holonomy_loops": holonomy_loops,
        "connection_field": connection_field,
    }

    return json.dumps(response, cls=SafeFloatEncoder).encode()

def handle_jacobian_field_viz(body_bytes):
    """
    Extract the pure Jacobian field — the morphing itself, stripped of all data.
    Returns a dense grid of Jacobian decompositions (stretch, rotation, eigenvalues)
    at every point in the PCA-projected space, for every layer transition.

    This is the diffeomorphism watching itself in a mirror.
    """
    req = json.loads(body_bytes)
    text = req.get("text", "").strip()
    pca_d = req.get("pca_d", 16)
    grid_res = req.get("grid_res", 20)

    if not text:
        return json.dumps({"error": "Empty text"}).encode()

    input_ids, tokens = tokenize_text(TOKENIZER, text)
    hs = extract_hidden_states(MODEL, input_ids)

    n_layers = get_n_layers(MODEL_CONFIG)
    hidden_dim = get_hidden_dim(MODEL_CONFIG)
    seq_len = input_ids.shape[1]

    print(f"[JacobianField] Extracting pure morphing field: "
          f"{seq_len} tokens × {n_layers} layers, grid={grid_res}²")

    # Convert hidden states to numpy
    H = np.stack([
        hs[lay][0].cpu().float().numpy() for lay in range(n_layers + 1)
    ], axis=0)  # (n_layers+1, seq_len, hidden_dim)

    # For each layer transition, estimate the Jacobian field on a 2D grid
    # The grid lives in the top-2 PCA subspace of the token cloud

    layer_fields = []

    for lay in range(n_layers):
        h_cloud = H[lay]       # (seq_len, hidden_dim)
        h_next = H[lay + 1]    # (seq_len, hidden_dim)
        deltas = h_next - h_cloud  # (seq_len, hidden_dim)

        # PCA of the token cloud at this layer
        cloud_centered = h_cloud - h_cloud.mean(axis=0)
        K = min(pca_d, hidden_dim, seq_len - 1)

        if seq_len >= 3:
            U, S, Vt = np.linalg.svd(cloud_centered, full_matrices=False)
            principal_dirs = Vt[:K]  # (K, hidden_dim)
        else:
            principal_dirs = np.eye(hidden_dim)[:K]

        # Project tokens into PCA space
        pos_proj = cloud_centered @ principal_dirs.T  # (seq_len, K)
        del_proj = deltas @ principal_dirs.T           # (seq_len, K)

        # Use only the first 2 PCA dims for the visualization grid
        pos_2d = pos_proj[:, :2]  # (seq_len, 2)

        # Compute grid bounds
        margin = 0.2
        x_min = pos_2d[:, 0].min()
        x_max = pos_2d[:, 0].max()
        y_min = pos_2d[:, 1].min()
        y_max = pos_2d[:, 1].max()
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0
        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range

        grid_x = np.linspace(x_min, x_max, grid_res)
        grid_y = np.linspace(y_min, y_max, grid_res)

        # At each grid point, estimate the local Jacobian via
        # weighted least-squares regression of nearby token deltas
        sigma = max(x_range, y_range) * 0.25
        s2i = 1.0 / (2 * sigma ** 2)

        # For the full K-dimensional Jacobian estimation
        # We fit: delta_proj ≈ J @ (pos_proj - query_pos) + delta_0
        # using Gaussian-weighted least squares

        grid_data = []  # list of dicts per grid point

        for gy_idx in range(grid_res):
            for gx_idx in range(grid_res):
                qx = grid_x[gx_idx]
                qy = grid_y[gy_idx]
                query_2d = np.array([qx, qy])

                # Gaussian weights based on 2D distance
                dists_sq = np.sum((pos_2d - query_2d) ** 2, axis=1)
                weights = np.exp(-dists_sq * s2i)
                w_sum = weights.sum()

                if w_sum < 1e-15:
                    weights = np.ones(seq_len) / seq_len
                    w_sum = 1.0

                # Weighted mean delta (the "base flow" at this point)
                w_norm = weights / w_sum
                mean_delta = w_norm @ del_proj  # (K,)

                # Estimate 2D Jacobian via weighted linear regression
                # delta_2d ≈ J_2x2 @ (pos_2d - query) + mean_delta_2d
                dx_local = pos_2d - query_2d  # (seq_len, 2)
                del_2d = del_proj[:, :2]       # (seq_len, 2)

                # Weighted least squares: J = (X^T W X)^{-1} X^T W Y
                W = np.diag(weights)
                XtWX = dx_local.T @ W @ dx_local + 1e-8 * np.eye(2)
                XtWY = dx_local.T @ W @ del_2d

                try:
                    J_2d = np.linalg.solve(XtWX, XtWY).T  # (2, 2)
                except np.linalg.LinAlgError:
                    J_2d = np.eye(2)

                # ============================================
                # DECOMPOSE THE JACOBIAN — this IS the morphing
                # ============================================

                # 1. Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eig(J_2d)
                eig_mag = np.abs(eigenvalues)
                eig_phase = np.angle(eigenvalues)

                # 2. Polar decomposition: J = R @ S (rotation × stretch)
                # Using SVD: J = U Σ V^T, then R = U V^T, S = V Σ V^T
                U_j, sv, Vt_j = np.linalg.svd(J_2d)
                R_mat = U_j @ Vt_j                    # rotation part
                Vt_j.T @ np.diag(sv) @ Vt_j   # stretch part

                # 3. Rotation angle
                rotation_angle = float(np.arctan2(R_mat[1, 0], R_mat[0, 0]))

                # 4. Divergence = trace(J) — expansion/contraction
                divergence = float(np.trace(J_2d))

                # 5. Curl = J[1,0] - J[0,1] — local rotation rate
                curl = float(J_2d[1, 0] - J_2d[0, 1])

                # 6. Shear = traceless symmetric part
                J_sym = (J_2d + J_2d.T) / 2
                J_traceless = J_sym - np.eye(2) * np.trace(J_sym) / 2
                shear = float(np.linalg.norm(J_traceless, 'fro'))

                # 7. Determinant — area change
                det = float(np.linalg.det(J_2d))

                # 8. Condition number — anisotropy
                condition = float(sv[0] / max(sv[-1], 1e-12))

                # 9. Mean flow vector at this point
                flow_x = float(mean_delta[0])
                flow_y = float(mean_delta[1])

                # 10. Principal stretch directions (from SVD)
                stretch_dir1 = Vt_j[0].tolist()  # direction of max stretch
                stretch_dir2 = Vt_j[1].tolist()  # direction of min stretch
                stretch_mag1 = float(sv[0])
                stretch_mag2 = float(sv[1])

                # 11. Eigenvector directions (may be complex for rotations)
                evec1_real = eigenvectors[:, 0].real.tolist()
                evec2_real = eigenvectors[:, 1].real.tolist()

                grid_data.append({
                    'gx': round(float(qx), 6),
                    'gy': round(float(qy), 6),

                    # The morphing itself
                    'flow_x': round(flow_x, 6),
                    'flow_y': round(flow_y, 6),
                    'divergence': round(divergence, 6),
                    'curl': round(curl, 6),
                    'shear': round(shear, 6),
                    'det': round(det, 6),
                    'rotation_angle': round(rotation_angle, 6),
                    'condition': round(condition, 4),

                    # Eigenstructure
                    'eig_mag': [round(float(m), 6) for m in eig_mag],
                    'eig_phase': [round(float(p), 6) for p in eig_phase],
                    'evec1': [round(float(v), 6) for v in evec1_real],
                    'evec2': [round(float(v), 6) for v in evec2_real],

                    # Stretch decomposition
                    'stretch_dir1': [round(v, 6) for v in stretch_dir1],
                    'stretch_dir2': [round(v, 6) for v in stretch_dir2],
                    'stretch_mag1': round(stretch_mag1, 6),
                    'stretch_mag2': round(stretch_mag2, 6),

                    # Singular values
                    'sv': [round(float(s), 6) for s in sv],
                })

        layer_fields.append({
            'layer': lay,
            'grid_res': grid_res,
            'grid_x_range': [round(float(x_min), 6), round(float(x_max), 6)],
            'grid_y_range': [round(float(y_min), 6), round(float(y_max), 6)],
            'grid_data': grid_data,
            'token_positions_2d': pos_2d.tolist(),
        })

    print(f"[JacobianField] Done: {n_layers} layers × {grid_res}² = "
          f"{n_layers * grid_res * grid_res} Jacobian samples")

    return json.dumps({
        'tokens': tokens,
        'seq_len': seq_len,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'pca_d': pca_d,
        'grid_res': grid_res,
        'layer_fields': layer_fields,
    }, cls=SafeFloatEncoder).encode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
