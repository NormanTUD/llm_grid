#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "transformers",
#   "tiktoken",
#   "sentencepiece",
#   "protobuf",
#   "numpy",
#   "sae-lens",
#   "transformer-lens",
# ]
# ///

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

SAE_AVAILABLE = True

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
    """Load pre-trained SAEs for each layer. Fails gracefully per-layer."""
    global SAE_MODELS
    SAE_MODELS = {}

    if not SAE_AVAILABLE:
        print("[SAE] sae-lens not available — skipping SAE loading")
        return

    release_id = get_sae_release_id(model_name)
    if release_id is None:
        print(f"[SAE] No known SAE release for model '{model_name}' — skipping")
        return

    print(f"[SAE] Loading SAEs from release '{release_id}' for {n_layers} layers...")

    for layer in range(n_layers):
        sae_id = f"blocks.{layer}.hook_resid_post"
        from sae_lens import SAE

        try:
            # SAE.from_pretrained returns (sae, cfg_dict, sparsity)
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=release_id,
                sae_id=sae_id,
            )
            sae.eval()
            SAE_MODELS[layer] = sae
            d_sae = sae.cfg.d_sae if hasattr(sae.cfg, 'd_sae') else '?'
            print(f"  [SAE] Layer {layer}: loaded ({d_sae} latents)")
        except Exception as e:
            print(f"  [SAE] Layer {layer}: not available ({e})")

    print(f"[SAE] Loaded SAEs for {len(SAE_MODELS)}/{n_layers} layers")

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
    """Convert list of numpy arrays to list of lists for JSON, sanitizing NaN/Inf."""
    result = []
    for v in all_layer0:
        v_safe = np.nan_to_num(np.array(v, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
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

    n_total = len(all_layer0)
    print(f"[Model] {n_total} real points")

    # Compute neighbors among real tokens only
    real_embeddings = np.stack(all_layer0[:n_real], axis=0)
    all_embeddings = np.stack(all_layer0, axis=0)
    neighbors = compute_neighbors(real_embeddings, all_embeddings, all_labels, all_is_real, k=10)

    # Predict next token
    print("[Model] Predicting next token...")
    next_token_preds = predict_next_token(TOKENIZER, MODEL, real_ids, MODEL_CONFIG, k=5)

    # Find vocabulary neighbors for each real token
    print("[Model] Finding vocabulary neighbors...")
    vocab_neighbors = find_vocab_neighbors(TOKENIZER, MODEL, all_layer0[:n_real], n_real, k=5)

    # PCA on real points only, then generate systematic grid probes
    print("[Model] Creating systematic grid probes around real points...")
    layer0_mat = np.stack(all_layer0, axis=0)

    if n_real < 2:
        print("[Model] WARNING: fewer than 2 real tokens, grid probes will be trivial")

    centroid, centered, pc1, pc2, proj1, proj2 = compute_pca_basis(layer0_mat, hidden_dim)
    existing_proj = np.stack([proj1, proj2], axis=1)

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

    # Append grid probes as the ONLY synthetic points
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

    data = build_output_data(
        all_labels, all_is_real, n_layers, n_total_final, n_real,
        hidden_dim, fixed_pos, deltas, MODEL_NAME, text, neighbors,
        next_token_preds=next_token_preds,
        vocab_neighbors=vocab_neighbors,
        attn_deltas=attn_deltas_json,
        mlp_deltas=mlp_deltas_json,
        strain_stats=strain_stats,
    )

    # Include the interpolation method in the response
    data["itp_method"] = itp_method

    json_str = json.dumps(data, cls=SafeFloatEncoder)
    print(f"[Model] JSON: {len(json_str)/1024/1024:.1f} MB")
    return json_str

# ============================================================
# 12. HTML PAGE WITH 2D/3D TOGGLE + DECOMPOSITION + STRAIN STATS
# ============================================================

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
<h2>Metric Space Explorer</h2>
<div id="status">Enter text and click Run</div>
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
<div class="cb" style="margin-bottom:4px">
  <input type="checkbox" id="cb-compare" onchange="toggleCompareMode()">
  <label for="cb-compare" style="color:#53a8b6;font-weight:bold">⚡ Compare Mode</label>
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
<!-- Multi-Sentence Comparison -->
<div class="cb" style="margin-bottom:4px">
  <input type="checkbox" id="cb-multi" onchange="toggleMultiMode()">
  <label for="cb-multi" style="color:#7b68ee;font-weight:bold">🔬 Multi-Sentence Compare</label>
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
<div id="info">
Model: <span id="i-mod">-</span> |
Points: <span id="i-pts">-</span> (<span id="i-real">-</span> real + <span id="i-syn">-</span> probes)<br>
Layers: <span id="i-lay">-</span> | Dim: <span id="i-dim">-</span> |
Tokens: <span id="i-tok">-</span> |
ITP: <span id="i-itp">rbf</span>
</div>

<h3>Predicted Next Token</h3>
<div id="next-token-panel" style="background:#0f3460;padding:6px;border-radius:4px;font-size:11px;line-height:1.6">
<span style="color:#555">Run a prompt to see predictions</span>
</div>
<h3>Neuron Activation Grid</h3>
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
<h3>Diffeomorphism Spectrum</h3>
<div id="spectrum-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px">
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

<h3>Contrastive Feature Scanner</h3>
<div id="contrastive-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px">
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
<h3>SAE Feature Inspector</h3>
<div id="sae-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px">
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
<h3>Holographic Curvature Analysis</h3>
<div id="curvature-panel" style="background:#0f3460;padding:8px;border-radius:4px;font-size:10px">
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
</div>
<h3>Selected Tokens (click canvas to select)</h3>
<div id="selected-tokens"><span style="color:#555;font-size:10px">Click on a token dot to select it</span></div>
<div id="neighbor-panel">
<h4 id="nb-title">Neighbors</h4>
<div id="nb-list"></div>
</div>
<h3>View Mode</h3>
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
<h3>Interpolation &amp; Grid</h3>
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
<h3>Interpolation &amp; Grid</h3>
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
<h3>Display</h3>
<div class="cb"><input type="checkbox" id="cb-grid" checked><label for="cb-grid">Deformed Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-vocnb" checked><label for="cb-vocnb">Show Nearby Vocab Words</label></div>
<div class="cb"><input type="checkbox" id="cb-heat" checked><label for="cb-heat">Strain Heatmap</label></div>
<div class="cb"><input type="checkbox" id="cb-ref" checked><label for="cb-ref">Reference Grid</label></div>
<div class="cb"><input type="checkbox" id="cb-tok" checked><label for="cb-tok">Real Tokens</label></div>
<div class="cb"><input type="checkbox" id="cb-syn"><label for="cb-syn">Probe Points</label></div>
<div class="cb"><input type="checkbox" id="cb-sc" checked><label for="cb-sc">Strain Color</label></div>
<div class="cb"><input type="checkbox" id="cb-vec"><label for="cb-vec">Vector Arrows</label></div>
<div id="keys"><b>Keys:</b> ←→ Dim X | ↑↓ Dim Y | Shift+←→ Dim Z (±1) | Shift+↑↓ Dim Z (±10) | PgUp/PgDn Dim Z (3D) | [/] Layer | ;/' t | A/Z Amp | Space Auto | R Reset | D Next Dim Pair | 0 Reset Zoom<br>
<b>Mouse:</b> Scroll=Zoom | Shift+Drag=Pan | Click=Select Token | Drag=Rotate (3D)</div>
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
</div>
</div>
<script>
var diffeoCanvas = document.getElementById('diffeo-canvas');
var diffeoCtx = diffeoCanvas ? diffeoCanvas.getContext('2d') : null;
var diffeoAnimId = null;
var diffeoTime = 0;

var diffeoState = {
  active: false,
  numSlices: 8,
  kelpAmplitude: 1.0,
  divergenceSensitivity: 1.0,
  layerSpacing: 70,
  sliceAlpha: 0.25,
  gridRes: 10,
  dimMode: 'auto',
  slices: [],
  built: false,
};

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
    var itpMethod=document.getElementById('sel-itp').value;
    var btn=document.getElementById('btn-run');
    btn.disabled=true;btn.textContent='Running...';
    document.getElementById('status').textContent='Processing (model: '+modelName+', itp: '+itpMethod+')...';
    fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text:txt, model:modelName, itp_method:itpMethod})})
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

function autoParams(){
    if(viewMode === 'multi'){
        if(!multiData || !multiData.sentence_data || !multiData.sentence_data.length) return;
        var sd = multiData.sentence_data[0];
        if(!sd || !sd.fixed_pos) return;
        var dx=+document.getElementById('sl-dx').value;
        var dy=+document.getElementById('sl-dy').value;
        var nP=sd.n_points;
        var mnx=Infinity,mxx=-Infinity,mny=Infinity,mxy=-Infinity;
        for(var i=0;i<nP;i++){
            var x=sd.fixed_pos[i][dx],y=sd.fixed_pos[i][dy];
            if(x<mnx)mnx=x;if(x>mxx)mxx=x;if(y<mny)mny=y;if(y>mxy)mxy=y;
        }
        var range=Math.max(mxx-mnx,mxy-mny)||1;
        var sig=range*0.15;
        var slSig=document.getElementById('sl-sig');
        slSig.max=Math.max(20,range*2).toFixed(1);
        slSig.value=sig.toFixed(2);
        document.getElementById('v-sig').textContent=sig.toFixed(2);
        var activeDeltas = sd.deltas;
        var decomp = document.getElementById('sel-decomp').value;
        if(decomp === 'attn' && sd.attn_deltas) activeDeltas = sd.attn_deltas;
        if(decomp === 'mlp' && sd.mlp_deltas) activeDeltas = sd.mlp_deltas;
        var norms=[];
        for(var l=0;l<sd.n_layers;l++){
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
        return;
    }
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
    if(dragActive && (viewMode==='3d' || (viewMode==='multi' && (multiSubView==='3d' || multiSubView==='fibre3d')))){
        var ddx=e.clientX-dragLastX, ddy=e.clientY-dragLastY;
        rotY+=ddx*0.005;
        rotX+=ddy*0.005;
        rotX=Math.max(-Math.PI/2, Math.min(Math.PI/2, rotX));
        dragLastX=e.clientX;
        dragLastY=e.clientY;
        draw();
        return;
    }

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
    document.getElementById(id).addEventListener('change', function(){ draw(); });
});

window.addEventListener('resize',function(){rsz();draw()});

function rsz() {
    var cv = document.getElementById('cv');
    var ct = document.getElementById('main');
    var s = Math.min(ct.clientWidth - 16, ct.clientHeight - 16);
    cv.width = Math.max(400, s);
    cv.height = cv.width;

    if (typeof diffeoState !== 'undefined' && diffeoState.active) {
        resizeDiffeoCanvas();
    }
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
    document.getElementById(id).addEventListener('change', function(){ draw(); });
});

document.getElementById('sel-mode').addEventListener('change', function(){ draw(); });
document.getElementById('sel-decomp').addEventListener('change',function(){
    autoParams();
    draw();
});
document.addEventListener('keydown',onKey);
document.getElementById('txt-in').addEventListener('keydown',function(e){if(e.key==='Enter')runText()});

document.getElementById('sel-itp').addEventListener('change', function(){
    // When interpolation method changes, we need to re-run the backend
    // because the grid probes are computed server-side with the chosen method
    document.getElementById('status').textContent =
        'Interpolation changed to ' + this.value + ' — click Run to recompute grid probes';
});

// ============================================================
// CLIENT-SIDE INTERPOLATION METHODS
// These mirror the server-side methods for real-time grid rendering
// ============================================================

function computeItpWeight(px, py, fx_k, fy_k, sig, method) {
    // Returns a single weight for point (px,py) relative to source (fx_k, fy_k)
    var ex = px - fx_k, ey = py - fy_k;
    var dist_sq = ex * ex + ey * ey;
    var dist = Math.sqrt(dist_sq);

    if (method === 'idw') {
        var p = 2.0;
        return 1.0 / Math.pow(Math.max(dist, 1e-12), p);
    } else if (method === 'wendland') {
        var R = Math.max(3.0 * sig, 1e-6);
        var r_norm = dist / R;
        if (r_norm >= 1.0) return 0.0;
        var t = 1.0 - r_norm;
        return t * t * t * t * (4.0 * r_norm + 1.0);
    } else if (method === 'nn') {
        // NN returns 0 for all but the nearest; handled specially
        return -dist; // negative distance; caller picks max
    } else {
        // Default: Gaussian RBF
        var s2i = 1.0 / (2.0 * sig * sig);
        var exponent = -dist_sq * s2i;
        if (exponent < -500) return 0;
        return Math.exp(exponent);
    }
}

function interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, method) {
    // Interpolate the deformation field at (px, py) using the selected method
    // Returns [vx, vy]

    if (method === 'nn') {
        // Nearest neighbor: just use the closest point's delta
        var bestDist = Infinity, bestIdx = 0;
        for (var k = 0; k < nP; k++) {
            var d = (px - fx[k]) * (px - fx[k]) + (py - fy[k]) * (py - fy[k]);
            if (d < bestDist) { bestDist = d; bestIdx = k; }
        }
        return [edx[bestIdx], edy[bestIdx]];
    }

    if (method === 'mls') {
        // Moving Least Squares with local linear basis
        var s2i = 1.0 / (2.0 * sig * sig);
        // Compute weights
        var W = new Float64Array(nP);
        for (var k = 0; k < nP; k++) {
            var ex = px - fx[k], ey = py - fy[k];
            var exp_val = -(ex * ex + ey * ey) * s2i;
            W[k] = exp_val < -500 ? 0 : Math.exp(exp_val);
        }

        // Build weighted least squares: f(x,y) = a0 + a1*(x-px) + a2*(y-py)
        // Normal equations: (A^T W A) c = A^T W v
        // A = [1, dx_k, dy_k] for each point k
        var AtwA00 = 0, AtwA01 = 0, AtwA02 = 0;
        var AtwA11 = 0, AtwA12 = 0, AtwA22 = 0;
        var Atwvx0 = 0, Atwvx1 = 0, Atwvx2 = 0;
        var Atwvy0 = 0, Atwvy1 = 0, Atwvy2 = 0;

        for (var k = 0; k < nP; k++) {
            var w = W[k];
            if (w < 1e-30) continue;
            var dxk = fx[k] - px, dyk = fy[k] - py;
            AtwA00 += w;
            AtwA01 += w * dxk;
            AtwA02 += w * dyk;
            AtwA11 += w * dxk * dxk;
            AtwA12 += w * dxk * dyk;
            AtwA22 += w * dyk * dyk;
            Atwvx0 += w * edx[k];
            Atwvx1 += w * dxk * edx[k];
            Atwvx2 += w * dyk * edx[k];
            Atwvy0 += w * edy[k];
            Atwvy1 += w * dxk * edy[k];
            Atwvy2 += w * dyk * edy[k];
        }

        // Add regularization
        AtwA00 += 1e-8; AtwA11 += 1e-8; AtwA22 += 1e-8;

        // Solve 3x3 system using Cramer's rule for a0 (the value at query point)
        var det = AtwA00 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                - AtwA01 * (AtwA01 * AtwA22 - AtwA12 * AtwA02)
                + AtwA02 * (AtwA01 * AtwA12 - AtwA11 * AtwA02);

        if (Math.abs(det) < 1e-20) {
            // Fallback to RBF
            return interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, 'rbf');
        }

        // We only need a0 (the constant term = value at query point)
        var detX = Atwvx0 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                 - AtwA01 * (Atwvx1 * AtwA22 - AtwA12 * Atwvx2)
                 + AtwA02 * (Atwvx1 * AtwA12 - AtwA11 * Atwvx2);
        var detY = Atwvy0 * (AtwA11 * AtwA22 - AtwA12 * AtwA12)
                 - AtwA01 * (Atwvy1 * AtwA22 - AtwA12 * Atwvy2)
                 + AtwA02 * (Atwvy1 * AtwA12 - AtwA11 * Atwvy2);

        return [detX / det, detY / det];
    }

    if (method === 'tps') {
        // TPS is expensive client-side; fall back to RBF for real-time rendering
        // The server-side grid probes already use TPS
        return interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, 'rbf');
    }

    // Weight-based methods: RBF, IDW, Wendland
    var vx = 0, vy = 0, ws = 0;
    for (var k = 0; k < nP; k++) {
        var w = computeItpWeight(px, py, fx[k], fy[k], sig, method);
        vx += w * edx[k];
        vy += w * edy[k];
        ws += w;
    }
    if (ws > 1e-15) { vx /= ws; vy /= ws; }
    return [vx, vy];
}

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

function onKey(e) {
    // ================================================================
    // Guard: don't intercept keys when typing in text inputs
    // ================================================================
    if (document.activeElement === document.getElementById('txt-in')) return;
    if (document.activeElement === document.getElementById('txt-b')) return;

    // ================================================================
    // FIBRE VIEW MODES: delegate to fibre-specific key handler
    // (was added by the _origOnKey wrapper)
    // ================================================================
    if (viewMode === 'fibre' || viewMode === 'fibrekelp' || viewMode === 'fibre3d') {
        onKeyFibre(e);
        return;
    }

    // ================================================================
    // STANDARD KEY HANDLING (the original onKey body)
    // ================================================================
    var sl  = document.getElementById('sl-layer');
    var st  = document.getElementById('sl-t');
    var sa  = document.getElementById('sl-amp');
    var sdx = document.getElementById('sl-dx');
    var sdy = document.getElementById('sl-dy');
    var sdz = document.getElementById('sl-dz');
    var maxDim = D ? D.hidden_dim - 1 : 767;

    // ---- Shift+Arrow = Dim Z (third axis), works in all views ----
    if (e.shiftKey && e.key === 'ArrowRight') {
        e.preventDefault();
        var newZ = +sdz.value + 1;
        if (newZ > maxDim) newZ = 0;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowLeft') {
        e.preventDefault();
        var newZ = +sdz.value - 1;
        if (newZ < 0) newZ = maxDim;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowUp') {
        e.preventDefault();
        var newZ = +sdz.value + 10;
        if (newZ > maxDim) newZ = newZ % (maxDim + 1);
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }
    else if (e.shiftKey && e.key === 'ArrowDown') {
        e.preventDefault();
        var newZ = +sdz.value - 10;
        if (newZ < 0) newZ = (newZ + maxDim + 1) % (maxDim + 1);
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
        return;
    }

    // ---- Arrow keys (no shift) = Dim X / Dim Y ----
    if (e.key === 'ArrowRight') {
        e.preventDefault();
        var newX = +sdx.value + 1;
        if (newX > maxDim) newX = 0;
        if (newX === +sdy.value) newX = (newX + 1) % (maxDim + 1);
        sdx.value = newX;
        sdx.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        var newX = +sdx.value - 1;
        if (newX < 0) newX = maxDim;
        if (newX === +sdy.value) newX = (newX - 1 + maxDim + 1) % (maxDim + 1);
        sdx.value = newX;
        sdx.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowUp') {
        e.preventDefault();
        var newY = +sdy.value + 1;
        if (newY > maxDim) newY = 0;
        if (newY === +sdx.value) newY = (newY + 1) % (maxDim + 1);
        sdy.value = newY;
        sdy.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'ArrowDown') {
        e.preventDefault();
        var newY = +sdy.value - 1;
        if (newY < 0) newY = maxDim;
        if (newY === +sdx.value) newY = (newY - 1 + maxDim + 1) % (maxDim + 1);
        sdy.value = newY;
        sdy.dispatchEvent(new Event('input'));
    }

    // ---- Layer navigation: [ ] or , . ----
    else if (e.key === '.' || e.key === ']') {
        sl.value = Math.min(+sl.max, +sl.value + 1);
        sl.dispatchEvent(new Event('input'));
    }
    else if (e.key === ',' || e.key === '[') {
        sl.value = Math.max(0, +sl.value - 1);
        sl.dispatchEvent(new Event('input'));
    }

    // ---- Deformation t: ; / ' ----
    else if (e.key === "'") {
        st.value = Math.min(1, +st.value + 0.05).toFixed(2);
        st.dispatchEvent(new Event('input'));
    }
    else if (e.key === ';') {
        st.value = Math.max(0, +st.value - 0.05).toFixed(2);
        st.dispatchEvent(new Event('input'));
    }

    // ---- Amplification: A / Z ----
    else if (e.key === 'a' || e.key === 'A') {
        sa.value = Math.min(500, +sa.value * 1.3).toFixed(1);
        sa.dispatchEvent(new Event('input'));
    }
    else if (e.key === 'z' || e.key === 'Z') {
        sa.value = Math.max(0.1, +sa.value / 1.3).toFixed(1);
        sa.dispatchEvent(new Event('input'));
    }

    // ---- Reset all: R ----
    else if (e.key === 'r' || e.key === 'R') {
        rstAll();
    }

    // ---- Next dimension pair: D ----
    else if (e.key === 'd' || e.key === 'D') {
        nxtD();
    }

    // ---- Reset zoom/pan: 0 ----
    else if (e.key === '0') {
        zoomLevel = 1.0;
        panX = 0;
        panY = 0;
        draw();
    }

    // ---- 3D-specific: PageUp/PageDown for Dim Z ----
    else if (viewMode === '3d' && e.key === 'PageUp') {
        e.preventDefault();
        var newZ = +sdz.value + 1;
        if (newZ > maxDim) newZ = 0;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
    }
    else if (viewMode === '3d' && e.key === 'PageDown') {
        e.preventDefault();
        var newZ = +sdz.value - 1;
        if (newZ < 0) newZ = maxDim;
        while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
        sdz.value = newZ;
        sdz.dispatchEvent(new Event('input'));
    }
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
    var itpMethod = document.getElementById('sel-itp').value;

    if(isEmb){
        for(var gi2=0;gi2<nV;gi2++){gX[gi2]=oX[gi2];gY[gi2]=oY[gi2]}
    } else {
        for(var gi3=0;gi3<nV;gi3++){
            var px=oX[gi3],py=oY[gi3];
            var interp = interpolateGridPoint(px, py, fx, fy, edx, edy, nP, sig, itpMethod);
            gX[gi3]=px+t*interp[0];gY[gi3]=py+t*interp[1];
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
        var itpLabel = document.getElementById('sel-itp').value.toUpperCase();
        c.fillText('Layer '+p.layer+'/'+(D.n_layers-1)+'  t='+p.t.toFixed(2)+'  amp='+p.amp.toFixed(1)+'  Dims:'+dx+','+dy+'  Mode:'+p.mode+'  Decomp:'+decompLabel+'  ITP:'+itpLabel+'  [2D]',42,18);
    }
    c.font='10px monospace';c.fillStyle='rgba(255,255,255,0.35)';
    c.fillText('Zoom: '+zoomLevel.toFixed(2)+'x  (Scroll=zoom, Shift+drag=pan, 0=reset)',42,H-10);
}

// ===================== SAE FEATURE INSPECTOR =====================

var saeInfo = null;

function initSAEPanel(){
    fetch('/sae_info')
    .then(function(r){return r.json()})
    .then(function(info){
        saeInfo = info;
        var status = document.getElementById('sae-status');
        var controls = document.getElementById('sae-controls');
        if(!info.sae_available){
            status.innerHTML = '<span style="color:#e94560">sae-lens not installed.</span><br>pip install sae-lens transformer-lens';
            return;
        }
        if(info.loaded_layers.length === 0){
            status.innerHTML = '<span style="color:#f5a623">No SAEs available for ' + info.model_name + '</span><br>SAEs exist for: gpt2, gpt2-medium, gpt2-large, pythia models';
            return;
        }
        status.innerHTML = '<span style="color:#2ecc71">✓ SAEs loaded for ' + info.loaded_layers.length + '/' + info.total_layers + ' layers</span>';
        controls.style.display = 'block';

        // Populate layer dropdown
        var layerSel = document.getElementById('sae-layer');
        layerSel.innerHTML = '';
        for(var i = 0; i < info.loaded_layers.length; i++){
            var l = info.loaded_layers[i];
            var li = info.layer_info[String(l)];
            var opt = document.createElement('option');
            opt.value = l;
            opt.textContent = 'Layer ' + l + (li && li.d_sae ? ' (' + li.d_sae + ' latents)' : '');
            layerSel.appendChild(opt);
        }
    })
    .catch(function(e){
        document.getElementById('sae-status').innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

// Update token dropdown when data changes
function updateSAETokenDropdown(){
    var sel = document.getElementById('sae-token');
    sel.innerHTML = '';
    if(!D) return;
    for(var i = 0; i < D.n_real; i++){
        var opt = document.createElement('option');
        opt.value = i;
        opt.textContent = '[' + i + '] ' + D.tokens[i];
        sel.appendChild(opt);
    }
    // Default to last token
    sel.value = D.n_real - 1;
}

function onData() {
    // ================================================================
    // FIBRE: Clear cached neuron data when new text arrives
    // (was added by _origOnData3 wrapper)
    // ================================================================
    fibreState.neuronData = null;

    // ================================================================
    // CORE: Update all UI controls from the new data D
    // (the original onData body)
    // ================================================================
    document.getElementById('sl-layer').max = D.n_layers - 1;
    document.getElementById('sl-dx').max = D.hidden_dim - 1;
    document.getElementById('sl-dy').max = D.hidden_dim - 1;
    document.getElementById('sl-dz').max = D.hidden_dim - 1;
    document.getElementById('i-mod').textContent = D.model_name;
    document.getElementById('i-pts').textContent = D.n_points;
    document.getElementById('i-real').textContent = D.n_real;
    document.getElementById('i-syn').textContent = D.n_synth;
    document.getElementById('i-lay').textContent = D.n_layers;
    document.getElementById('i-dim').textContent = D.hidden_dim;
    document.getElementById('i-tok').textContent = D.tokens.slice(0, D.n_real).join(' ');
    document.getElementById('sel-model').value = D.model_name;

    // Display interpolation method
    document.getElementById('i-itp').textContent = D.itp_method || 'rbf';
    if (D.itp_method) {
        document.getElementById('sel-itp').value = D.itp_method;
    }

    // Update decomposition selector availability
    var decompSel = document.getElementById('sel-decomp');
    if (D.attn_deltas && D.mlp_deltas) {
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
    document.getElementById('status').textContent =
        'Ready — ' + D.n_real + ' tokens, ' + D.n_synth + ' probes | Model: ' + D.model_name;

    // Render next-token predictions
    var ntp = document.getElementById('next-token-panel');
    if (D.next_token && D.next_token.length > 0) {
        var html = '';
        for (var i = 0; i < D.next_token.length; i++) {
            var nt = D.next_token[i];
            var barW = Math.max(2, nt.prob * 200);
            html += '<div style="display:flex;align-items:center;gap:6px">';
            html += '<span style="color:#e94560;font-weight:bold;min-width:80px;font-family:monospace">' +
                    nt.token + '</span>';
            html += '<div style="background:#e94560;height:8px;width:' + barW +
                    'px;border-radius:3px;opacity:0.7"></div>';
            html += '<span style="color:#888;font-size:9px">' +
                    (nt.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        ntp.innerHTML = html;
    } else {
        ntp.innerHTML = '<span style="color:#555">No predictions available</span>';
    }

    // ================================================================
    // SAE: Refresh token dropdown and re-initialize SAE panel
    // (was added by _origOnData / SAE wrapper)
    // ================================================================
    updateSAETokenDropdown();
    initSAEPanel();

    // ================================================================
    // DIFFEO: Rebuild diffeomorphism overlay if active
    // (was added by _origOnData2 / diffeo wrapper)
    // ================================================================
    if (diffeoState.active) {
        rebuildDiffeo();
    }

    // ================================================================
    // FIBRE: Auto-fetch neuron data if currently in a fibre view
    // (was added by _origOnData3 / fibre wrapper)
    // ================================================================
    if (viewMode.startsWith('fibre') && D) {
        fetchFibreNeuronData();
    }
}

function fetchSAEFeatures(){
    if(!D) return;
    var layer = +document.getElementById('sae-layer').value;
    var tokenIdx = +document.getElementById('sae-token').value;
    var topK = +document.getElementById('sae-topk').value;
    var text = D.text;

    var list = document.getElementById('sae-features-list');
    list.innerHTML = '<span style="color:#53a8b6">Loading...</span>';

    fetch('/sae_features', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, layer: layer, token_idx: tokenIdx, top_k: topK})
    })
    .then(function(r){return r.json()})
    .then(function(data){
        if(data.error){
            list.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }
        var features = data.features;
        var nLatents = data.n_latents || '?';
        var html = '<div style="color:#888;margin-bottom:4px">Token: <b style="color:#e94560">' +
                   data.tokens[tokenIdx] + '</b> | Layer ' + layer +
                   ' | ' + nLatents + ' total latents</div>';

        // Token activation heatmap for top features
        if(data.token_activations && data.tokens){
            html += '<div style="margin-bottom:6px;font-size:9px;color:#888">Activation heatmap (top features × tokens):</div>';
            html += '<div style="overflow-x:auto;margin-bottom:6px"><table style="border-collapse:collapse;font-size:8px">';
            // Header row: tokens
            html += '<tr><td></td>';
            for(var ti = 0; ti < data.tokens.length; ti++){
                var isSel = (ti === tokenIdx);
                html += '<td style="padding:1px 3px;text-align:center;color:' + (isSel ? '#e94560' : '#888') + ';font-weight:' + (isSel ? 'bold' : 'normal') + '">' + data.tokens[ti] + '</td>';
            }
            html += '</tr>';
            // Feature rows
            var taKeys = Object.keys(data.token_activations);
            for(var fi = 0; fi < Math.min(taKeys.length, 8); fi++){
                var fid = taKeys[fi];
                var acts = data.token_activations[fid];
                // Find max for color scaling
                var maxAct = 0;
                for(var ai = 0; ai < acts.length; ai++){
                    if(Math.abs(acts[ai]) > maxAct) maxAct = Math.abs(acts[ai]);
                }
                if(maxAct < 1e-8) maxAct = 1;
                html += '<tr><td style="padding:1px 4px;color:#53a8b6;white-space:nowrap">F' + fid + '</td>';
                for(var ai = 0; ai < acts.length; ai++){
                    var intensity = Math.min(1, Math.abs(acts[ai]) / maxAct);
                    var r = acts[ai] > 0 ? Math.round(233 * intensity) : Math.round(0);
                    var g = acts[ai] > 0 ? Math.round(69 * intensity) : Math.round(119 * intensity);
                    var b = acts[ai] > 0 ? Math.round(96 * intensity) : Math.round(182 * intensity);
                    var bg = 'rgba(' + r + ',' + g + ',' + b + ',' + (0.2 + 0.6 * intensity).toFixed(2) + ')';
                    html += '<td style="padding:1px 3px;text-align:center;background:' + bg + '">' + acts[ai].toFixed(2) + '</td>';
                }
                html += '</tr>';
            }
            html += '</table></div>';
        }

        // Feature list
        html += '<div style="font-size:9px;color:#888;margin-bottom:2px">Click a feature to set up intervention:</div>';
        for(var i = 0; i < features.length; i++){
            var f = features[i];
            var barW = Math.min(120, Math.max(2, Math.abs(f.activation) * 10));
            var barColor = f.activation > 0 ? '#e94560' : '#0077b6';
            html += '<div class="sae-feat-row" onclick="selectSAEFeature(' + f.feature_id + ',' + f.activation.toFixed(4) + ')" ' +
                    'style="display:flex;align-items:center;gap:4px;padding:2px 4px;cursor:pointer;border-radius:2px;margin:1px 0" ' +
                    'onmouseover="this.style.background=\'#1a1a2e\'" onmouseout="this.style.background=\'transparent\'">';
            html += '<span style="color:#53a8b6;min-width:55px;font-family:monospace">F' + f.feature_id + '</span>';
            html += '<div style="background:' + barColor + ';height:6px;width:' + barW + 'px;border-radius:2px;flex-shrink:0"></div>';
            html += '<span style="color:#888;font-size:9px;min-width:60px">' + f.activation.toFixed(4) + '</span>';
            html += '</div>';
        }

        list.innerHTML = html;

        // Show intervention panel
        document.getElementById('sae-intervention').style.display = 'block';
    })
    .catch(function(e){
        list.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function selectSAEFeature(featureId, currentActivation){
    document.getElementById('sae-int-feature').value = featureId;
    // Set clamp slider to a value that would amplify it
    var clampVal = Math.max(currentActivation * 3, 10);
    var slider = document.getElementById('sae-int-clamp');
    slider.value = Math.min(+slider.max, clampVal).toFixed(1);
    document.getElementById('v-sae-clamp').textContent = slider.value;
}

function runSAEIntervention(){
    if(!D) return;
    var layer = +document.getElementById('sae-layer').value;
    var featureId = +document.getElementById('sae-int-feature').value;
    var clampValue = +document.getElementById('sae-int-clamp').value;
    var text = D.text;

    var results = document.getElementById('sae-int-results');
    results.innerHTML = '<span style="color:#53a8b6">Running intervention...</span>';

    fetch('/sae_intervene', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, layer: layer, feature_id: featureId, clamp_value: clampValue})
    })
    .then(function(r){return r.json()})
    .then(function(data){
        if(data.error){
            results.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }
        var html = '<div style="color:#f5a623;font-weight:bold;margin-bottom:4px">Feature ' + featureId + ' clamped to ' + clampValue +
                   ' at layer ' + data.layer + '</div>';

        // Side-by-side comparison
        html += '<div style="display:flex;gap:8px">';

        // Baseline column
        html += '<div style="flex:1">';
        html += '<div style="color:#888;font-size:9px;margin-bottom:3px;text-decoration:underline">Baseline</div>';
        for(var i = 0; i < data.baseline_predictions.length; i++){
            var bp = data.baseline_predictions[i];
            var barW = Math.max(2, bp.prob * 150);
            html += '<div style="display:flex;align-items:center;gap:3px;margin:1px 0">';
            html += '<span style="color:#a0a0c0;min-width:55px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + bp.token + '</span>';
            html += '<div style="background:#555;height:5px;width:' + barW + 'px;border-radius:2px;flex-shrink:0"></div>';
            html += '<span style="color:#888;font-size:8px">' + (bp.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        html += '</div>';

        // Modified column
        html += '<div style="flex:1">';
        html += '<div style="color:#e94560;font-size:9px;margin-bottom:3px;text-decoration:underline">Modified</div>';
        for(var i = 0; i < data.modified_predictions.length; i++){
            var mp = data.modified_predictions[i];
            var barW2 = Math.max(2, mp.prob * 150);
            // Check if this token was in baseline top predictions
            var isNew = true;
            for(var bi = 0; bi < data.baseline_predictions.length; bi++){
                if(data.baseline_predictions[bi].token === mp.token){
                    isNew = false;
                    break;
                }
            }
            var tokenColor = isNew ? '#e94560' : '#a0a0c0';
            html += '<div style="display:flex;align-items:center;gap:3px;margin:1px 0">';
            html += '<span style="color:' + tokenColor + ';min-width:55px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + (isNew ? '★ ' : '') + mp.token + '</span>';
            html += '<div style="background:#e94560;height:5px;width:' + barW2 + 'px;border-radius:2px;flex-shrink:0"></div>';
            html += '<span style="color:#888;font-size:8px">' + (mp.prob * 100).toFixed(1) + '%</span>';
            html += '</div>';
        }
        html += '</div>';

        html += '</div>'; // end flex container

        // Summary of biggest changes
        html += '<div style="margin-top:6px;border-top:1px solid #1a1a2e;padding-top:4px;font-size:9px;color:#888">';
        // Find the biggest probability shift
        var maxShift = 0;
        var shiftToken = '';
        var shiftDir = '';
        for(var mi = 0; mi < data.modified_predictions.length; mi++){
            var mToken = data.modified_predictions[mi].token;
            var mProb = data.modified_predictions[mi].prob;
            var bProb = 0;
            for(var bi2 = 0; bi2 < data.baseline_predictions.length; bi2++){
                if(data.baseline_predictions[bi2].token === mToken){
                    bProb = data.baseline_predictions[bi2].prob;
                    break;
                }
            }
            var shift = mProb - bProb;
            if(Math.abs(shift) > Math.abs(maxShift)){
                maxShift = shift;
                shiftToken = mToken;
                shiftDir = shift > 0 ? '↑' : '↓';
            }
        }
        if(Math.abs(maxShift) > 0.001){
            html += 'Biggest shift: <span style="color:#e94560">"' + shiftToken + '"</span> ' +
                    shiftDir + ' ' + (Math.abs(maxShift) * 100).toFixed(1) + '%';
        } else {
            html += 'No significant prediction changes detected.';
        }
        html += '</div>';

        results.innerHTML = html;
    })
    .catch(function(e){
        results.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function clearSAEIntervention(){
    document.getElementById('sae-int-results').innerHTML = '';
    document.getElementById('sae-int-clamp').value = 0;
    document.getElementById('v-sae-clamp').textContent = '0.0';
}

// Slider value display updates for SAE controls
document.getElementById('sae-topk').addEventListener('input', function(){
    document.getElementById('v-sae-topk').textContent = this.value;
});
document.getElementById('sae-int-clamp').addEventListener('input', function(){
    document.getElementById('v-sae-clamp').textContent = parseFloat(this.value).toFixed(1);
});

// Initialize SAE panel on page load
setTimeout(function(){
    initSAEPanel();
}, 500);

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

    var itpMethod = document.getElementById('sel-itp').value;

    if(isEmb){
        for(var gi2=0;gi2<nV3;gi2++){gX3[gi2]=oX3[gi2];gY3[gi2]=oY3[gi2];gZ3[gi2]=oZ3[gi2]}
    } else {
        for(var gi3=0;gi3<nV3;gi3++){
            var gpx=oX3[gi3],gpy=oY3[gi3],gpz=oZ3[gi3];
            // For 3D, we extend the 2D methods to 3D distance
            var vvx=0,vvy=0,vvz=0,ws=0;
            if(itpMethod === 'nn'){
                var bestDist3=Infinity, bestIdx3=0;
                for(var k=0;k<nP;k++){
                    var d3=(gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]);
                    if(d3<bestDist3){bestDist3=d3;bestIdx3=k;}
                }
                vvx=edx3[bestIdx3];vvy=edy3[bestIdx3];vvz=edz3[bestIdx3];
            } else if(itpMethod === 'idw'){
                var p3=2.0;
                for(var k=0;k<nP;k++){
                    var dist3=Math.sqrt((gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]));
                    var w=1.0/Math.pow(Math.max(dist3,1e-12),p3);
                    vvx+=w*edx3[k];vvy+=w*edy3[k];vvz+=w*edz3[k];ws+=w;
                }
                if(ws>1e-15){vvx/=ws;vvy/=ws;vvz/=ws}
            } else if(itpMethod === 'wendland'){
                var R3=Math.max(3.0*sig,1e-6);
                for(var k=0;k<nP;k++){
                    var dist3=Math.sqrt((gpx-fx[k])*(gpx-fx[k])+(gpy-fy[k])*(gpy-fy[k])+(gpz-fz[k])*(gpz-fz[k]));
                    var rn3=dist3/R3;
                    if(rn3>=1.0) continue;
                    var tt=1.0-rn3;
                    var w=tt*tt*tt*tt*(4.0*rn3+1.0);
                    vvx+=w*edx3[k];vvy+=w*edy3[k];vvz+=w*edz3[k];ws+=w;
                }
                if(ws<1e-30){
                    // fallback to RBF
                    for(var k=0;k<nP;k++){
                        var eex=gpx-fx[k],eey=gpy-fy[k],eez=gpz-fz[k];
                        var w=Math.exp(-(eex*eex+eey*eey+eez*eez)*s2i3);
                        vvx+=w*edx3[k];vvy+=w*edy3[k];vvz+=w*edz3[k];ws+=w;
                    }
                }
                if(ws>1e-15){vvx/=ws;vvy/=ws;vvz/=ws}
            } else {
                // RBF (default), MLS fallback to RBF in 3D, TPS fallback to RBF
                for(var k=0;k<nP;k++){
                    var eex=gpx-fx[k],eey=gpy-fy[k],eez=gpz-fz[k];
                    var w=Math.exp(-(eex*eex+eey*eey+eez*eez)*s2i3);
                    vvx+=w*edx3[k];vvy+=w*edy3[k];vvz+=w*edz3[k];ws+=w;
                }
                if(ws>1e-15){vvx/=ws;vvy/=ws;vvz/=ws}
            }
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
// ===================== NEURON ACTIVATION GRID =====================

var neuronGridData = null;

document.getElementById('ng-pixsize').addEventListener('input', function(){
    document.getElementById('v-ng-pixsize').textContent = this.value;
    if(neuronGridData) renderNeuronGrid();
});
document.getElementById('ng-norm').addEventListener('change', function(){
    if(neuronGridData) renderNeuronGrid();
});
document.getElementById('ng-absval').addEventListener('change', function(){
    if(neuronGridData) renderNeuronGrid();
});

function fetchNeuronGrid(){
    if(!D) return;
    var panel = document.getElementById('neuron-grid-panel');
    panel.style.display = 'contents';
    panel.innerHTML = '<span style="color:#53a8b6">Loading neuron activations...</span>';

    fetch('/neuron_grid', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: D.text})
    })
    .then(function(r){return r.json()})
    .then(function(data){
        if(data.error){
            panel.innerHTML = '<span style="color:#e94560">' + data.error + '</span>';
            return;
        }
        neuronGridData = data;
        renderNeuronGrid();
    })
    .catch(function(e){
        panel.innerHTML = '<span style="color:#e94560">Error: ' + e + '</span>';
    });
}

function renderNeuronGrid(){
    var data = neuronGridData;
    if(!data) return;

    var panel = document.getElementById('neuron-grid-panel');
    var normMode = document.getElementById('ng-norm').value;  // 'layer' or 'global'
    var pixSize = +document.getElementById('ng-pixsize').value;
    var useAbs = document.getElementById('ng-absval').checked;
    var hiddenDim = data.hidden_dim;
    var nTokens = data.n_tokens;
    var nLayers = data.n_layers;

    var layersSource = (normMode === 'global') ? data.global_norm : data.layer_norm;

    // Compute a nice grid layout for the hidden_dim neurons
    // Try to make it roughly square
    var gridCols = Math.ceil(Math.sqrt(hiddenDim));
    var gridRows = Math.ceil(hiddenDim / gridCols);

    var html = '';
    html += '<div style="color:#888;font-size:9px;margin-bottom:6px">';
    html += 'Each rectangle = one layer\'s activations. ';
    html += 'Each pixel = one neuron (dim). ';
    html += 'Bright = high activation, dark = low. ';
    html += 'Grid: ' + gridCols + '×' + gridRows + ' (' + hiddenDim + ' neurons)';
    html += '</div>';

    // For each token, show all layers side by side
    for(var ti = 0; ti < nTokens; ti++){
        html += '<div style="margin-bottom:8px">';
        html += '<div style="color:#e94560;font-weight:bold;font-size:10px;margin-bottom:2px">';
        html += '[' + ti + '] ' + data.tokens[ti];
        html += '</div>';
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start;min-height:0;overflow:visible">';

        for(var li = 0; li < nLayers; li++){
            var acts = layersSource[li].activations[ti]; // array of hiddenDim floats [0,1]
            var canvasId = 'ng-cv-' + ti + '-' + li;
            var canvasW = gridCols * pixSize;
            var canvasH = gridRows * pixSize;

            html += '<div style="text-align:center">';
            html += '<div style="color:#53a8b6;font-size:8px;margin-bottom:1px">';
            html += (li === 0 ? 'Emb' : 'L' + (li-1));
            html += '</div>';
            html += '<canvas id="' + canvasId + '" width="' + canvasW + '" height="' + canvasH + '" ';
            html += 'style="border:1px solid #0f3460;image-rendering:pixelated" ';
            html += 'title="' + (li === 0 ? 'Embedding' : 'Layer ' + (li-1)) + ' — Token: ' + data.tokens[ti] + '">';
            html += '</canvas>';
            html += '</div>';
        }

        html += '</div></div>';
    }

    panel.innerHTML = html;

    // Now draw on each canvas
    for(var ti = 0; ti < nTokens; ti++){
        for(var li = 0; li < nLayers; li++){
            var canvasId = 'ng-cv-' + ti + '-' + li;
            var cv = document.getElementById(canvasId);
            if(!cv) continue;
            var ctx = cv.getContext('2d');
            var acts = layersSource[li].activations[ti];

            var imgData = ctx.createImageData(gridCols * pixSize, gridRows * pixSize);

            for(var ni = 0; ni < hiddenDim; ni++){
                var val = acts[ni];
                if(useAbs) val = Math.abs(val * 2 - 1); // re-center then abs

                // Map to grayscale: 0 = black, 1 = white
                var brightness = Math.floor(val * 255);
                brightness = Math.max(0, Math.min(255, brightness));

                var col = Math.floor(ni % gridCols);
                var row = Math.floor(ni / gridCols);

                // Fill the pixel block
                for(var py = 0; py < pixSize; py++){
                    for(var px = 0; px < pixSize; px++){
                        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
                        var offset = ix * 4;
                        imgData.data[offset]     = brightness;  // R
                        imgData.data[offset + 1] = brightness;  // G
                        imgData.data[offset + 2] = brightness;  // B
                        imgData.data[offset + 3] = 255;         // A
                    }
                }
            }

            // Fill remaining pixels (if hiddenDim doesn't fill the grid) with dark
            for(var ni = hiddenDim; ni < gridCols * gridRows; ni++){
                var col = ni % gridCols;
                var row = Math.floor(ni / gridCols);
                for(var py = 0; py < pixSize; py++){
                    for(var px = 0; px < pixSize; px++){
                        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
                        var offset = ix * 4;
                        imgData.data[offset]     = 20;
                        imgData.data[offset + 1] = 10;
                        imgData.data[offset + 2] = 30;
                        imgData.data[offset + 3] = 255;
                    }
                }
            }

            ctx.putImageData(imgData, 0, 0);
        }
    }
}

// Cool-to-hot colormap: dark blue → cyan → yellow → red → white
function valToColor(v) {
    // v in [0, 1]
    var r, g, b;
    if (v < 0.25) {
        var t = v / 0.25;
        r = 0; g = Math.floor(t * 128); b = Math.floor(64 + t * 191);
    } else if (v < 0.5) {
        var t = (v - 0.25) / 0.25;
        r = 0; g = Math.floor(128 + t * 127); b = Math.floor(255 - t * 128);
    } else if (v < 0.75) {
        var t = (v - 0.5) / 0.25;
        r = Math.floor(t * 255); g = 255; b = Math.floor(127 - t * 127);
    } else {
        var t = (v - 0.75) / 0.25;
        r = 255; g = Math.floor(255 - t * 128); b = Math.floor(t * 128);
    }
    return [r, g, b];
}
// ============================================================
// DIFFEOMORPHISM STACKING — KELP FOREST OF LAYER ROOMS
// ============================================================

function toggleDiffeo() {
  var wrap = document.getElementById('diffeo-wrap');
  if (!wrap) return;
  wrap.style.display = diffeoState.active ? 'block' : 'none';
  if (diffeoState.active) {
    resizeDiffeoCanvas();
    rebuildDiffeo();
    startDiffeoLoop();
  } else {
    stopDiffeoLoop();
    if (diffeoCtx) {
      var W = diffeoCanvas.width / (window.devicePixelRatio || 1);
      var H = diffeoCanvas.height / (window.devicePixelRatio || 1);
      diffeoCtx.clearRect(0, 0, W, H);
    }
  }
}

function resizeDiffeoCanvas() {
  if (!diffeoCanvas) return;
  var container = document.getElementById('main');
  var dpr = window.devicePixelRatio || 1;
  diffeoCanvas.width = container.clientWidth * dpr;
  diffeoCanvas.height = container.clientHeight * dpr;
  diffeoCanvas.style.width = container.clientWidth + 'px';
  diffeoCanvas.style.height = container.clientHeight + 'px';
  if (diffeoCtx) diffeoCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function startDiffeoLoop() {
  if (diffeoAnimId) return;
  function loop() {
    diffeoTime += 0.016;
    if (diffeoState.active && diffeoState.built) {
      updateDiffeoGrids(diffeoTime);
      renderDiffeoOverlay(diffeoTime);
    }
    diffeoAnimId = requestAnimationFrame(loop);
  }
  diffeoAnimId = requestAnimationFrame(loop);
}

function stopDiffeoLoop() {
  if (diffeoAnimId) {
    cancelAnimationFrame(diffeoAnimId);
    diffeoAnimId = null;
  }
}

/**
 * Build diffeomorphism slices from the server data D.
 * Each slice = one 2D cross-section (dim pair) at one layer.
 * Deformation comes from D.deltas[layer][point][dim].
 */
function rebuildDiffeo() {
  diffeoState.slices = [];
  diffeoState.built = false;
  if (!D || D.n_layers < 1 || D.n_points < 2) return;

  var nL = D.n_layers;
  var nP = D.n_points;
  var nR = D.n_real;
  var hiddenDim = D.hidden_dim;
  var res = diffeoState.gridRes;

  // Use the currently active deltas (respects decomposition selector)
  var activeDeltas = getActiveDeltas();
  if (!activeDeltas) activeDeltas = D.deltas;

  // Generate dimension pairs
  var dimPairs = [];
  var maxDims = Math.min(hiddenDim, 8);
  if (diffeoState.dimMode === 'sequential') {
    for (var d = 0; d + 1 < maxDims; d += 2) dimPairs.push([d, d + 1]);
  } else if (diffeoState.dimMode === 'first') {
    for (var d = 1; d < maxDims; d++) dimPairs.push([0, d]);
  } else {
    for (var a = 0; a < maxDims; a++)
      for (var b = a + 1; b < maxDims; b++)
        dimPairs.push([a, b]);
  }
  if (dimPairs.length === 0) dimPairs.push([0, 1]);

  // Build slice configs: layer × dim pair
  var sliceConfigs = [];
  for (var li = 0; li < nL; li++) {
    for (var pi = 0; pi < dimPairs.length; pi++) {
      sliceConfigs.push({ layerIdx: li, dimA: dimPairs[pi][0], dimB: dimPairs[pi][1] });
    }
  }

  // Evenly sample if too many
  var maxSlices = diffeoState.numSlices;
  if (sliceConfigs.length > maxSlices) {
    var step = sliceConfigs.length / maxSlices;
    var sampled = [];
    for (var i = 0; i < maxSlices; i++) sampled.push(sliceConfigs[Math.floor(i * step)]);
    sliceConfigs = sampled;
  }

  // For each slice, compute the deformation grid
  // We use the real tokens' fixed_pos and deltas to define the deformation field
  // via RBF interpolation onto a regular grid in the 2D subspace.
  for (var si = 0; si < sliceConfigs.length; si++) {
    var cfg = sliceConfigs[si];
    var dA = cfg.dimA, dB = cfg.dimB, lay = cfg.layerIdx;

    // Extract the 2D positions and delta vectors for real tokens in this dim pair
    var posA = new Float64Array(nR), posB = new Float64Array(nR);
    var delA = new Float64Array(nR), delB = new Float64Array(nR);
    var mnA = Infinity, mxA = -Infinity, mnB = Infinity, mxB = -Infinity;

    for (var ti = 0; ti < nR; ti++) {
      posA[ti] = D.fixed_pos[ti][dA];
      posB[ti] = D.fixed_pos[ti][dB];
      delA[ti] = activeDeltas[lay][ti][dA];
      delB[ti] = activeDeltas[lay][ti][dB];
      if (posA[ti] < mnA) mnA = posA[ti]; if (posA[ti] > mxA) mxA = posA[ti];
      if (posB[ti] < mnB) mnB = posB[ti]; if (posB[ti] > mxB) mxB = posB[ti];
    }

    var rngA = mxA - mnA || 1, rngB = mxB - mnB || 1;
    var rng = Math.max(rngA, rngB);
    var pad = 0.15;
    var cA = (mnA + mxA) / 2, cB = (mnB + mxB) / 2;
    var lo = cA - rng * (0.5 + pad), hi = cA + rng * (0.5 + pad);
    var loB = cB - rng * (0.5 + pad), hiB = cB + rng * (0.5 + pad);

    // RBF bandwidth
    var sigma = rng * 0.2;
    var s2i = 1 / (2 * sigma * sigma);

    // Build grid
    var grid = [];
    for (var gy = 0; gy <= res; gy++) {
      for (var gx = 0; gx <= res; gx++) {
        var u = gx / res;
        var v = gy / res;
        var worldA = lo + u * (hi - lo);
        var worldB = loB + v * (hiB - loB);

        // RBF interpolation of delta from real tokens
        var dx = 0, dy = 0, ws = 0;
        for (var k = 0; k < nR; k++) {
          var ea = worldA - posA[k], eb = worldB - posB[k];
          var w = Math.exp(-(ea * ea + eb * eb) * s2i);
          dx += w * delA[k];
          dy += w * delB[k];
          ws += w;
        }
        if (ws > 1e-15) { dx /= ws; dy /= ws; }

        grid.push({ ox: u, oy: v, dx: dx, dy: dy, divergence: 0 });
      }
    }

    // Compute divergence (how much neighboring displacements differ)
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx = gy * (res + 1) + gx;
        var idxR = idx + 1;
        var idxD = idx + (res + 1);
        var ddx = grid[idxR].dx - grid[idx].dx;
        var ddy = grid[idxD].dy - grid[idx].dy;
        grid[idx].divergence = Math.sqrt(ddx * ddx + ddy * ddy);
      }
    }

    var hue = (cfg.layerIdx / nL) * 120 + (cfg.dimA * 40 + cfg.dimB * 20);

    diffeoState.slices.push({
      layerIdx: cfg.layerIdx,
      dimA: cfg.dimA,
      dimB: cfg.dimB,
      grid: grid,
      hue: hue % 360,
      res: res,
    });
  }

  diffeoState.built = true;
}

/**
 * Update grid deformations each frame.
 * Re-reads deltas from D so it stays in sync with layer/decomp changes.
 * The kelp sway is purely time-driven via divergence.
 */
function updateDiffeoGrids(time) {
  if (!D || !diffeoState.built) return;

  var activeDeltas = getActiveDeltas();
  if (!activeDeltas) activeDeltas = D.deltas;
  var nR = D.n_real;

  for (var si = 0; si < diffeoState.slices.length; si++) {
    var slice = diffeoState.slices[si];
    var res = slice.res;
    var lay = slice.layerIdx;
    var dA = slice.dimA, dB = slice.dimB;

    // Recompute positions and deltas for this layer
    var posA = new Float64Array(nR), posB = new Float64Array(nR);
    var delA = new Float64Array(nR), delB = new Float64Array(nR);
    var mnA = Infinity, mxA = -Infinity, mnB = Infinity, mxB = -Infinity;

    for (var ti = 0; ti < nR; ti++) {
      posA[ti] = D.fixed_pos[ti][dA];
      posB[ti] = D.fixed_pos[ti][dB];
      delA[ti] = activeDeltas[lay][ti][dA];
      delB[ti] = activeDeltas[lay][ti][dB];
      if (posA[ti] < mnA) mnA = posA[ti]; if (posA[ti] > mxA) mxA = posA[ti];
      if (posB[ti] < mnB) mnB = posB[ti]; if (posB[ti] > mxB) mxB = posB[ti];
    }

    var rng = Math.max(mxA - mnA, mxB - mnB) || 1;
    var pad = 0.15;
    var cA = (mnA + mxA) / 2, cB = (mnB + mxB) / 2;
    var lo = cA - rng * (0.5 + pad), hi = cA + rng * (0.5 + pad);
    var loB = cB - rng * (0.5 + pad), hiB = cB + rng * (0.5 + pad);
    var sigma = rng * 0.2;
    var s2i = 1 / (2 * sigma * sigma);

    for (var gi = 0; gi < slice.grid.length; gi++) {
      var cell = slice.grid[gi];
      var worldA = lo + cell.ox * (hi - lo);
      var worldB = loB + cell.oy * (hiB - loB);

      var dx = 0, dy = 0, ws = 0;
      for (var k = 0; k < nR; k++) {
        var ea = worldA - posA[k], eb = worldB - posB[k];
        var w = Math.exp(-(ea * ea + eb * eb) * s2i);
        dx += w * delA[k];
        dy += w * delB[k];
        ws += w;
      }
      if (ws > 1e-15) { dx /= ws; dy /= ws; }
      cell.dx = dx;
      cell.dy = dy;
    }

    // Recompute divergence
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx = gy * (res + 1) + gx;
        var idxR = idx + 1;
        var idxD = idx + (res + 1);
        var ddx = slice.grid[idxR].dx - slice.grid[idx].dx;
        var ddy = slice.grid[idxD].dy - slice.grid[idx].dy;
        slice.grid[idx].divergence = Math.sqrt(ddx * ddx + ddy * ddy);
      }
    }
  }
}

/**
 * Render the stacked diffeomorphism slices as translucent kelp-swaying grids.
 */
function renderDiffeoOverlay(time) {
  if (!diffeoState.active || !diffeoCtx || diffeoState.slices.length === 0) return;

  var dpr = window.devicePixelRatio || 1;
  var W = diffeoCanvas.width / dpr;
  var H = diffeoCanvas.height / dpr;
  diffeoCtx.clearRect(0, 0, W, H);

  var nSlices = diffeoState.slices.length;
  var spacing = diffeoState.layerSpacing;
  var totalHeight = nSlices * spacing;
  var startY = (H - totalHeight) / 2;

  var amp = diffeoState.kelpAmplitude;
  var sens = diffeoState.divergenceSensitivity;
  var alpha = diffeoState.sliceAlpha;

  // Normalize delta magnitudes for visible sway
  var maxDelta = 0;
  for (var si = 0; si < nSlices; si++) {
    var grid = diffeoState.slices[si].grid;
    for (var gi = 0; gi < grid.length; gi++) {
      var mag = Math.sqrt(grid[gi].dx * grid[gi].dx + grid[gi].dy * grid[gi].dy);
      if (mag > maxDelta) maxDelta = mag;
    }
  }
  var deltaScale = maxDelta > 1e-8 ? 1.0 / maxDelta : 1.0;

  for (var si = 0; si < nSlices; si++) {
    var slice = diffeoState.slices[si];
    var res = slice.res;
    var grid = slice.grid;
    var baseY = startY + si * spacing;

    var sliceW = W * 0.55;
    var sliceH = spacing * 0.75;

    // Perspective: further slices smaller
    var depthFactor = 0.65 + 0.35 * (si / Math.max(nSlices - 1, 1));
    var perspW = sliceW * depthFactor;
    var perspH = sliceH * depthFactor;
    var perspX = (W - perspW) / 2;
    var perspY = baseY + (sliceH - perspH) / 2;

    diffeoCtx.save();
    diffeoCtx.globalAlpha = alpha * (0.4 + 0.6 * depthFactor);

    // Draw deformed grid cells
    for (var gy = 0; gy < res; gy++) {
      for (var gx = 0; gx < res; gx++) {
        var idx00 = gy * (res + 1) + gx;
        var idx10 = idx00 + 1;
        var idx01 = idx00 + (res + 1);
        var idx11 = idx01 + 1;

        var corners = [idx00, idx10, idx11, idx01];
        var pts = [];
        for (var ci = 0; ci < 4; ci++) {
          var cell = grid[corners[ci]];
          var div = cell.divergence * sens;
          // Kelp sway: divergence drives amplitude, time drives oscillation
          var swayX = Math.sin(time * 0.7 + corners[ci] * 0.5 + si * 1.3) * div * amp * 35 * deltaScale;
          var swayY = Math.sin(time * 0.5 + corners[ci] * 0.3 + si * 0.9) * div * amp * 12 * deltaScale;

          var screenX = perspX + (cell.ox + cell.dx * deltaScale * amp * 0.3) * perspW + swayX;
          var screenY = perspY + (cell.oy + cell.dy * deltaScale * amp * 0.3) * perspH + swayY;
          pts.push({ x: screenX, y: screenY });
        }

        var cellDiv = grid[idx00].divergence * sens;
        var brightness = Math.min(70, 30 + cellDiv * 300 * deltaScale);
        var saturation = 50 + Math.min(30, cellDiv * 50 * deltaScale);

        // Filled quad
        diffeoCtx.beginPath();
        diffeoCtx.moveTo(pts[0].x, pts[0].y);
        diffeoCtx.lineTo(pts[1].x, pts[1].y);
        diffeoCtx.lineTo(pts[2].x, pts[2].y);
        diffeoCtx.lineTo(pts[3].x, pts[3].y);
        diffeoCtx.closePath();
        diffeoCtx.fillStyle = 'hsla(' + slice.hue + ',' + saturation + '%,' + brightness + '%,' + (alpha * 0.4) + ')';
        diffeoCtx.fill();

        // Grid lines
        diffeoCtx.strokeStyle = 'hsla(' + slice.hue + ',70%,' + Math.min(80, 40 + cellDiv * 150 * deltaScale) + '%,' + (alpha * 1.2) + ')';
        diffeoCtx.lineWidth = 0.5 + cellDiv * 3 * deltaScale;
        diffeoCtx.stroke();
      }
    }

    // Kelp strands connecting to next slice
    if (si < nSlices - 1) {
      var nextSlice = diffeoState.slices[si + 1];
      var nextBaseY = startY + (si + 1) * spacing;
      var nextDepth = 0.65 + 0.35 * ((si + 1) / Math.max(nSlices - 1, 1));
      var nextPerspW = sliceW * nextDepth;
      var nextPerspH = spacing * 0.75 * nextDepth;
      var nextPerspX = (W - nextPerspW) / 2;
      var nextPerspY = nextBaseY + (spacing * 0.75 - nextPerspH) / 2;

      var step = Math.max(1, Math.floor(res / 3));
      for (var gy = 0; gy <= res; gy += step) {
        for (var gx = 0; gx <= res; gx += step) {
          var idx = gy * (res + 1) + gx;
          if (idx >= grid.length) continue;
          var cell = grid[idx];
          var div = cell.divergence * sens;

          var swX1 = Math.sin(time * 0.7 + idx * 0.5 + si * 1.3) * div * amp * 35 * deltaScale;
          var swY1 = Math.sin(time * 0.5 + idx * 0.3 + si * 0.9) * div * amp * 12 * deltaScale;
          var x1 = perspX + (cell.ox + cell.dx * deltaScale * amp * 0.3) * perspW + swX1;
          var y1 = perspY + (cell.oy + cell.dy * deltaScale * amp * 0.3) * perspH + swY1;

          var nextGrid = nextSlice.grid;
          var nIdx = Math.min(idx, nextGrid.length - 1);
          var nCell = nextGrid[nIdx];
          var nDiv = nCell.divergence * sens;
          var swX2 = Math.sin(time * 0.7 + nIdx * 0.5 + (si + 1) * 1.3) * nDiv * amp * 35 * deltaScale;
          var swY2 = Math.sin(time * 0.5 + nIdx * 0.3 + (si + 1) * 0.9) * nDiv * amp * 12 * deltaScale;
          var x2 = nextPerspX + (nCell.ox + nCell.dx * deltaScale * amp * 0.3) * nextPerspW + swX2;
          var y2 = nextPerspY + (nCell.oy + nCell.dy * deltaScale * amp * 0.3) * nextPerspH + swY2;

          var midX = (x1 + x2) / 2 + Math.sin(time * 1.1 + idx) * div * amp * 18 * deltaScale;
          var midY = (y1 + y2) / 2;

          diffeoCtx.strokeStyle = 'hsla(' + Math.round((slice.hue + nextSlice.hue) / 2) + ',60%,45%,' + (alpha * 0.7) + ')';
          diffeoCtx.lineWidth = 0.8 + div * 2.5 * deltaScale;
          diffeoCtx.beginPath();
          diffeoCtx.moveTo(x1, y1);
          diffeoCtx.quadraticCurveTo(midX, midY, x2, y2);
          diffeoCtx.stroke();
        }
      }
    }

    // Slice label
    diffeoCtx.globalAlpha = 0.6;
    diffeoCtx.fillStyle = 'hsl(' + slice.hue + ',70%,65%)';
    diffeoCtx.font = '9px monospace';
    diffeoCtx.fillText(
      'L' + slice.layerIdx + ' d' + slice.dimA + '\u00d7d' + slice.dimB,
      perspX - 65,
      perspY + perspH / 2 + 3
    );

    diffeoCtx.restore();
  }
}

function draw() {
    // Multi-sentence view doesn't need D — it uses multiData
    if (viewMode === 'multi') {
        if (multiData) drawMultiCanvas();
        return;
    }

    if (!D) return;

    // --- Fibre bundle views (was added by _origDraw2 wrapper) ---
    if (viewMode === 'fibrekelp') {
        drawFibreBundleKelp();
        return;
    }
    if (viewMode === 'fibre3d') {
        drawFibreBundle3DGrid();
        return;
    }
    if (viewMode === 'fibre') {
        drawFibreBundle();
        return;
    }

    // --- 3D view (original draw logic) ---
    if (viewMode === '3d') {
        draw3D();
        return;
    }

    // --- Default: 2D view ---
    draw2D();

    // --- Diffeomorphism overlay rebuild (was added by the diffeo wrapper) ---
    if (diffeoState.active && D) {
        rebuildDiffeo();
    }
}

// ============================================================
// FIBRE BUNDLE VIEW — Neuron Pixel Grids Stacked as Kelp Rooms
// ============================================================

// State for fibre bundle view
var fibreState = {
  neuronData: null,       // cached neuron grid data from server
  loading: false,
  normMode: 'layer',      // 'layer' or 'global'
  pixSize: 2,
  useAbs: false,
  roomSpacing: 80,        // vertical gap between layer rooms
  roomWidth: 0,           // computed
  roomHeight: 0,          // computed
  tokenSpacing: 0,        // computed
  dimX: 0,                // which hidden dim maps to X sort within room
  dimY: 1,                // which hidden dim maps to Y sort within room
  dimZ: 2,                // depth axis for 3D projection
  show3D: false,          // toggle pseudo-3D stacking
  rotX: -0.3,
  rotY: 0.4,
  dragActive: false,
  dragLastX: 0,
  dragLastY: 0,
  scrollY: 0,             // vertical scroll offset
  selectedToken: -1,
  hoveredNeuron: null,
  showConnections: true,   // show diffeomorphism lines between layers
  connectionDensity: 0.1,  // fraction of neurons to connect
  colormap: 'grayscale',   // 'grayscale', 'coolhot', 'viridis'
  showTransportFrame: false,
  showAttnField: false,
  showMlpField: false,
  showFlowLines: true,
  flowArrowScale: 1.0,
};

function setViewMode(mode) {
    // ---- Multi-view button visibility (from multi wrapper) ----
    var multiBtn = document.getElementById('btn-multi-view');
    if (multiBtn) {
        multiBtn.style.display = multiData ? 'inline-block' : 'none';
    }

    // ---- Clear ALL view-toggle button active states ----
    var allBtnIds = ['btn-2d', 'btn-3d', 'btn-fibre', 'btn-fibre3d', 'btn-fibrekelp'];
    for (var i = 0; i < allBtnIds.length; i++) {
        var el = document.getElementById(allBtnIds[i]);
        if (el) el.className = '';
    }
    if (multiBtn) multiBtn.className = '';

    // ---- Dim Z row: only visible in 3d and fibre3d ----
    var dzRow = document.getElementById('dz-row');

    // ================================================================
    // MULTI mode (from the multi wrapper)
    // ================================================================
    if (mode === 'multi') {
        viewMode = 'multi';
        if (multiBtn) multiBtn.className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        drawMultiCanvas();
        return;
    }

    // ================================================================
    // FIBRE KELP mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibrekelp') {
        viewMode = 'fibrekelp';
        document.getElementById('btn-fibrekelp').className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        draw();
        return;
    }

    // ================================================================
    // FIBRE 3D mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibre3d') {
        viewMode = 'fibre3d';
        document.getElementById('btn-fibre3d').className = 'active';
        if (dzRow) dzRow.style.display = 'flex';
        if (D && !fibreState.neuronData && !fibreState.loading) {
            fetchFibreNeuronData();
        }
        draw();
        return;
    }

    // ================================================================
    // FIBRE (2D) mode (from the fibre wrapper)
    // ================================================================
    if (mode === 'fibre') {
        viewMode = 'fibre';
        document.getElementById('btn-fibre').className = 'active';
        if (dzRow) dzRow.style.display = 'none';
        if (D && !fibreState.neuronData && !fibreState.loading) {
            fetchFibreNeuronData();
        }
        draw();
        return;
    }

    // ================================================================
    // 3D mode (from the original)
    // ================================================================
    if (mode === '3d') {
        viewMode = '3d';
        document.getElementById('btn-3d').className = 'active';
        if (dzRow) dzRow.style.display = 'flex';
        draw();
        return;
    }

    // ================================================================
    // 2D mode — default (from the original)
    // ================================================================
    viewMode = '2d';
    document.getElementById('btn-2d').className = 'active';
    if (dzRow) dzRow.style.display = 'none';
    draw();
}

function fetchFibreNeuronData() {
  if (!D || fibreState.loading) return;
  fibreState.loading = true;
  document.getElementById('status').textContent = 'Loading neuron activations for fibre view...';

  fetch('/neuron_grid', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: D.text })
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.error) {
      document.getElementById('status').textContent = 'Neuron grid error: ' + data.error;
      fibreState.loading = false;
      return;
    }
    fibreState.neuronData = data;
    fibreState.loading = false;
    document.getElementById('status').textContent =
      'Fibre view ready — ' + data.n_tokens + ' tokens × ' +
      data.n_layers + ' layers × ' + data.hidden_dim + ' neurons';
    draw();
  })
  .catch(function(e) {
    document.getElementById('status').textContent = 'Error: ' + e;
    fibreState.loading = false;
  });
}

function computeFractionalDeltas(layerFrac, mode, activeDeltas, attnDeltas, mlpDeltas, nLayers, nP, dx, dy, amp) {
    var layerInt = Math.floor(layerFrac);
    var frac = layerFrac - layerInt;
    layerInt = Math.min(layerInt, nLayers - 1);

    var edxCum = new Float64Array(nP);
    var edyCum = new Float64Array(nP);
    // Also track the "current layer" attn and mlp components for the vector field
    var attnDx = new Float64Array(nP);
    var attnDy = new Float64Array(nP);
    var mlpDx = new Float64Array(nP);
    var mlpDy = new Float64Array(nP);

    if (mode === 'single') {
        // Interpolate between layer layerInt and layerInt+1
        for (var j = 0; j < nP; j++) {
            edxCum[j] = activeDeltas[layerInt][j][dx] * amp;
            edyCum[j] = activeDeltas[layerInt][j][dy] * amp;
            if (frac > 0 && layerInt + 1 < nLayers) {
                var nextDx = activeDeltas[layerInt + 1][j][dx] * amp;
                var nextDy = activeDeltas[layerInt + 1][j][dy] * amp;
                edxCum[j] = edxCum[j] * (1 - frac) + nextDx * frac;
                edyCum[j] = edyCum[j] * (1 - frac) + nextDy * frac;
            }
        }
    } else if (mode === 'cumfwd') {
        for (var j = 0; j < nP; j++) {
            for (var cl = 0; cl <= layerInt; cl++) {
                edxCum[j] += activeDeltas[cl][j][dx] * amp;
                edyCum[j] += activeDeltas[cl][j][dy] * amp;
            }
            // Add fractional part of next layer
            if (frac > 0 && layerInt + 1 < nLayers) {
                edxCum[j] += activeDeltas[layerInt + 1][j][dx] * amp * frac;
                edyCum[j] += activeDeltas[layerInt + 1][j][dy] * amp * frac;
            }
        }
    } else { // cumbwd
        for (var j = 0; j < nP; j++) {
            for (var cl = layerInt; cl < nLayers; cl++) {
                var weight = (cl === layerInt) ? (1 - frac) : 1.0;
                edxCum[j] += activeDeltas[cl][j][dx] * amp * weight;
                edyCum[j] += activeDeltas[cl][j][dy] * amp * weight;
            }
        }
    }

    // Current layer's decomposed components (for vector field overlay)
    if (attnDeltas && mlpDeltas) {
        for (var j = 0; j < nP; j++) {
            attnDx[j] = attnDeltas[layerInt][j][dx] * amp;
            attnDy[j] = attnDeltas[layerInt][j][dy] * amp;
            mlpDx[j] = mlpDeltas[layerInt][j][dx] * amp;
            mlpDy[j] = mlpDeltas[layerInt][j][dy] * amp;
            // Blend toward next layer if fractional
            if (frac > 0 && layerInt + 1 < nLayers) {
                attnDx[j] = attnDx[j] * (1 - frac) + attnDeltas[layerInt + 1][j][dx] * amp * frac;
                attnDy[j] = attnDy[j] * (1 - frac) + attnDeltas[layerInt + 1][j][dy] * amp * frac;
                mlpDx[j] = mlpDx[j] * (1 - frac) + mlpDeltas[layerInt + 1][j][dx] * amp * frac;
                mlpDy[j] = mlpDy[j] * (1 - frac) + mlpDeltas[layerInt + 1][j][dy] * amp * frac;
            }
        }
    }

    return {
        edx: edxCum, edy: edyCum,
        attnDx: attnDx, attnDy: attnDy,
        mlpDx: mlpDx, mlpDy: mlpDy,
        layerInt: layerInt, frac: frac
    };
}

function drawFlowArrow(c, fromX, fromY, vx, vy, color, maxLen) {
    var len = Math.hypot(vx, vy);
    if (len < 1.5) return; // skip tiny arrows

    // Clamp length
    if (len > maxLen) {
        var scale = maxLen / len;
        vx *= scale;
        vy *= scale;
        len = maxLen;
    }

    var toX = fromX + vx;
    var toY = fromY + vy;

    c.strokeStyle = color;
    c.fillStyle = color;
    c.lineWidth = Math.max(0.5, Math.min(1.5, len / 15));

    // Shaft
    c.beginPath();
    c.moveTo(fromX, fromY);
    c.lineTo(toX, toY);
    c.stroke();

    // Arrowhead
    var aa = Math.atan2(vy, vx);
    var hl = Math.min(5, len * 0.35);
    c.beginPath();
    c.moveTo(toX, toY);
    c.lineTo(toX - hl * Math.cos(aa - 0.45), toY - hl * Math.sin(aa - 0.45));
    c.lineTo(toX - hl * Math.cos(aa + 0.45), toY - hl * Math.sin(aa + 0.45));
    c.closePath();
    c.fill();
}

function drawTransportFrame(c, cx, cy, edx, edy, fx, fy, tokenIdx, nP, sig, frameSize) {
    // Compute the local Jacobian of the deformation field at this point
    // by finite differences in the RBF-interpolated field
    var eps = sig * 0.1;
    var s2i = 1 / (2 * sig * sig);

    function interpolateField(px, py) {
        var vvx = 0, vvy = 0, ws = 0;
        for (var k = 0; k < nP; k++) {
            var eex = px - fx[k], eey = py - fy[k];
            var w = Math.exp(-(eex * eex + eey * eey) * s2i);
            vvx += w * edx[k];
            vvy += w * edy[k];
            ws += w;
        }
        if (ws > 1e-15) { vvx /= ws; vvy /= ws; }
        return [vvx, vvy];
    }

    var basePx = fx[tokenIdx], basePy = fy[tokenIdx];
    var v0 = interpolateField(basePx, basePy);
    var vRight = interpolateField(basePx + eps, basePy);
    var vUp = interpolateField(basePx, basePy + eps);

    // Jacobian columns: how does the displacement field change as we move right/up
    var J00 = 1 + (vRight[0] - v0[0]) / eps; // dx/dx
    var J01 = (vUp[0] - v0[0]) / eps;         // dx/dy
    var J10 = (vRight[1] - v0[1]) / eps;       // dy/dx
    var J11 = 1 + (vUp[1] - v0[1]) / eps;     // dy/dy

    // Apply Jacobian to unit vectors to get transported frame
    var e1x = J00 * frameSize, e1y = J10 * frameSize;
    var e2x = J01 * frameSize, e2y = J11 * frameSize;

    // Draw the transported frame as two colored arrows
    // e1 (originally pointing right) in yellow
    c.globalAlpha = 0.7;
    drawFlowArrow(c, cx, cy, e1x, e1y, 'rgba(255,255,100,0.8)', frameSize * 2);
    // e2 (originally pointing up) in magenta
    drawFlowArrow(c, cx, cy, e2x, e2y, 'rgba(255,100,255,0.8)', frameSize * 2);
    c.globalAlpha = 1.0;

    // Draw a small circle at the center
    c.beginPath();
    c.arc(cx, cy, 2, 0, Math.PI * 2);
    c.fillStyle = 'rgba(255,255,255,0.6)';
    c.fill();
}

function drawFibreBundleKelp() {
    var cv = document.getElementById('cv');
    var c = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace';
        c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    var nTokens = D.n_real;
    var nLayers = D.n_layers;
    var hiddenDim = D.hidden_dim;
    var nP = D.n_points;

    var dxDim = +document.getElementById('sl-dx').value;
    var dyDim = +document.getElementById('sl-dy').value;
    var amp = +document.getElementById('sl-amp').value;
    var t = +document.getElementById('sl-t').value;
    var sig = +document.getElementById('sl-sig').value;
    var currentLayer = +document.getElementById('sl-layer').value;
    var showGrid = document.getElementById('cb-grid').checked;
    var showHeat = document.getElementById('cb-heat').checked;
    var showSC = document.getElementById('cb-sc').checked;
    var mode = document.getElementById('sel-mode').value;

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;
    var attnDeltas = D.attn_deltas || null;
    var mlpDeltas = D.mlp_deltas || null;
    var isEmb = (mode === 'embedding');

    dxDim = Math.min(dxDim, hiddenDim - 1);
    dyDim = Math.min(dyDim, hiddenDim - 1);

    // Extract 2D base positions for all points
    var fx = new Float64Array(nP), fy = new Float64Array(nP);
    for (var i = 0; i < nP; i++) {
        fx[i] = D.fixed_pos[i][dxDim];
        fy[i] = D.fixed_pos[i][dyDim];
    }

    // View bounds in world space
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (var i = 0; i < nP; i++) {
        if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
        if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    }
    var mr = Math.max(mxx - mnx, mxy - mny) || 1;
    var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
    var pd = 0.15;
    var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
    var vw = mr * (1 + 2 * pd), vh = vw;

    var s2i = 1 / (2 * sig * sig);

    // Layout: the entire canvas is one vertical column.
    // Y axis = layer depth (layer 0 at bottom, layer N-1 at top)
    // X axis = world-space position projected from hidden dims
    var margin = 40;
    var plotW = W / zoomLevel - 2 * margin;
    var plotH = H / zoomLevel - 2 * margin;
    var layerH = plotH / nLayers;

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    // World-to-screen transforms
    function SX(wx) { return margin + ((wx - vx0) / vw) * plotW; }
    function LY(layerIdx) { return margin + (nLayers - 1 - layerIdx) * layerH + layerH * 0.5; }

    // ================================================================
    // PRECOMPUTE: For each real token, compute its world-space position
    // at each layer under the current mode (cumfwd/cumbwd/single).
    // This IS the parallel transport — the path through the fibre.
    // ================================================================
    var tokenWorldX = []; // [ti][li] = world x at layer li
    var tokenWorldY = []; // [ti][li] = world y at layer li
    var tokenAttnDx = []; // [ti][li] = attention push in x
    var tokenAttnDy = [];
    var tokenMlpDx = [];
    var tokenMlpDy = [];

    for (var ti = 0; ti < nTokens; ti++) {
        var wxArr = [], wyArr = [];
        var adxArr = [], adyArr = [], mdxArr = [], mdyArr = [];

        for (var li = 0; li < nLayers; li++) {
            var cumDx = 0, cumDy = 0;
            if (isEmb) {
                // No deformation
            } else if (mode === 'single') {
                cumDx = activeDeltas[li][ti][dxDim] * amp * t;
                cumDy = activeDeltas[li][ti][dyDim] * amp * t;
            } else if (mode === 'cumfwd') {
                for (var cl = 0; cl <= li; cl++) {
                    cumDx += activeDeltas[cl][ti][dxDim] * amp * t;
                    cumDy += activeDeltas[cl][ti][dyDim] * amp * t;
                }
            } else { // cumbwd
                for (var cl = li; cl < nLayers; cl++) {
                    cumDx += activeDeltas[cl][ti][dxDim] * amp * t;
                    cumDy += activeDeltas[cl][ti][dyDim] * amp * t;
                }
            }

            wxArr.push(fx[ti] + cumDx);
            wyArr.push(fy[ti] + cumDy);

            // Decomposed forces at this layer
            var adx = 0, ady = 0, mdx = 0, mdy = 0;
            if (attnDeltas && !isEmb) {
                adx = attnDeltas[li][ti][dxDim] * amp * t;
                ady = attnDeltas[li][ti][dyDim] * amp * t;
            }
            if (mlpDeltas && !isEmb) {
                mdx = mlpDeltas[li][ti][dxDim] * amp * t;
                mdy = mlpDeltas[li][ti][dyDim] * amp * t;
            }
            adxArr.push(adx); adyArr.push(ady);
            mdxArr.push(mdx); mdyArr.push(mdy);
        }

        tokenWorldX.push(wxArr);
        tokenWorldY.push(wyArr);
        tokenAttnDx.push(adxArr); tokenAttnDy.push(adyArr);
        tokenMlpDx.push(mdxArr); tokenMlpDy.push(mdyArr);
    }

    // ================================================================
    // PASS 1: BACKGROUND DEFORMED GRIDS per layer (light, contextual)
    // Shows how the entire space deforms at each layer
    // ================================================================
    if (showGrid && !isEmb) {
        var N = Math.max(8, Math.min(25, Math.floor(plotW / 20)));

        for (var li = 0; li < nLayers; li++) {
            var ly = LY(li);
            var bandTop = ly - layerH * 0.4;
            var bandBot = ly + layerH * 0.4;
            var bandH = bandBot - bandTop;
            var isActive = (li === currentLayer);

            // Compute cumulative deltas for all points at this layer
            var edxCum = new Float64Array(nP);
            var edyCum = new Float64Array(nP);
            for (var j = 0; j < nP; j++) {
                if (mode === 'single') {
                    edxCum[j] = activeDeltas[li][j][dxDim] * amp;
                    edyCum[j] = activeDeltas[li][j][dyDim] * amp;
                } else if (mode === 'cumfwd') {
                    for (var cl = 0; cl <= li; cl++) {
                        edxCum[j] += activeDeltas[cl][j][dxDim] * amp;
                        edyCum[j] += activeDeltas[cl][j][dyDim] * amp;
                    }
                } else {
                    for (var cl = li; cl < nLayers; cl++) {
                        edxCum[j] += activeDeltas[cl][j][dxDim] * amp;
                        edyCum[j] += activeDeltas[cl][j][dyDim] * amp;
                    }
                }
            }

            // Build deformed grid
            var nV = (N + 1) * (N + 1);
            var oX = new Float64Array(nV), oY = new Float64Array(nV);
            var gX = new Float64Array(nV), gY = new Float64Array(nV);

            for (var gy = 0; gy <= N; gy++) {
                for (var gx = 0; gx <= N; gx++) {
                    var gi = gy * (N + 1) + gx;
                    oX[gi] = vx0 + (gx / N) * vw;
                    oY[gi] = vy0 + (gy / N) * vh;
                }
            }

            var _itpM = document.getElementById('sel-itp').value;
            for (var gi = 0; gi < nV; gi++) {
                var px = oX[gi], py = oY[gi];
                var _iRes = interpolateGridPoint(px, py, fx, fy, edxCum, edyCum, nP, sig, _itpM);
                gX[gi] = px + t * _iRes[0];
                gY[gi] = py + t * _iRes[1];
            }


            // Compute strain for coloring
            var sH = new Float64Array(N * (N + 1));
            for (var ey = 0; ey <= N; ey++) {
                for (var ex = 0; ex < N; ex++) {
                    var a = ey * (N + 1) + ex, b = a + 1;
                    var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
                    var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
                    sH[ey * N + ex] = od > 1e-12 ? dd / od : 1;
                }
            }
            var sV = new Float64Array((N + 1) * N);
            for (var ey = 0; ey < N; ey++) {
                for (var ex = 0; ex <= N; ex++) {
                    var a = ey * (N + 1) + ex, b = (ey + 1) * (N + 1) + ex;
                    var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
                    var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
                    sV[ey * (N + 1) + ex] = od > 1e-12 ? dd / od : 1;
                }
            }

            // Map grid world-X to screen-X, grid world-Y to a narrow band
            function GSX(wx) { return margin + ((wx - vx0) / vw) * plotW; }
            function GSY(wy) { return bandTop + ((wy - vy0) / vh) * bandH; }

            // Strain heatmap
            if (showHeat) {
                for (var hy = 0; hy < N; hy++) {
                    for (var hx = 0; hx < N; hx++) {
                        var avg = (sH[hy * N + hx] + sH[(hy + 1) * N + hx] +
                                   sV[hy * (N + 1) + hx] + sV[hy * (N + 1) + hx + 1]) / 4;
                        var co = s2c(avg);
                        var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
                        var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;
                        c.beginPath();
                        c.moveTo(GSX(gX[i00]), GSY(gY[i00]));
                        c.lineTo(GSX(gX[i10]), GSY(gY[i10]));
                        c.lineTo(GSX(gX[i11]), GSY(gY[i11]));
                        c.lineTo(GSX(gX[i01]), GSY(gY[i01]));
                        c.closePath();
                        c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',' + (isActive ? 0.2 : 0.07) + ')';
                        c.fill();
                    }
                }
            }

            // Grid lines
            var gridAlpha = isActive ? 0.3 : 0.08;
            c.lineWidth = isActive ? 0.7 : 0.3;
            for (var dhy = 0; dhy <= N; dhy++) {
                for (var dhx = 0; dhx < N; dhx++) {
                    var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
                    if (showSC) {
                        var es = sH[dhy * N + dhx];
                        var ec = s2c(es);
                        c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',' + gridAlpha + ')';
                    } else {
                        c.strokeStyle = 'rgba(200,200,200,' + gridAlpha + ')';
                    }
                    c.beginPath();
                    c.moveTo(GSX(gX[di1]), GSY(gY[di1]));
                    c.lineTo(GSX(gX[di2]), GSY(gY[di2]));
                    c.stroke();
                }
            }
            for (var dvx = 0; dvx <= N; dvx++) {
                for (var dvy = 0; dvy < N; dvy++) {
                    var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
                    if (showSC) {
                        var vs = sV[dvy * (N + 1) + dvx];
                        var vc = s2c(vs);
                        c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',' + gridAlpha + ')';
                    } else {
                        c.strokeStyle = 'rgba(200,200,200,' + gridAlpha + ')';
                    }
                    c.beginPath();
                    c.moveTo(GSX(gX[dvi1]), GSY(gY[dvi1]));
                    c.lineTo(GSX(gX[dvi2]), GSY(gY[dvi2]));
                    c.stroke();
                }
            }

            // Layer label
            c.font = (isActive ? 'bold ' : '') + '10px monospace';
            c.fillStyle = isActive ? '#e94560' : '#444';
            c.textAlign = 'right';
            c.fillText('L' + li, margin - 8, ly + 3);

            // Thin separator
            c.strokeStyle = 'rgba(60,60,100,' + (isActive ? 0.3 : 0.1) + ')';
            c.lineWidth = 0.5;
            c.beginPath();
            c.moveTo(margin, ly + layerH * 0.48);
            c.lineTo(margin + plotW, ly + layerH * 0.48);
            c.stroke();
        }
    }

    // ================================================================
    // PASS 2: TOKEN PATHLINES — the parallel transport made visible
    // Each token traces a smooth curve from layer 0 (bottom) to
    // layer N-1 (top). The X position shifts based on the cumulative
    // deformation. This IS the parallel transport.
    // ================================================================
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for (var ti = 0; ti < nTokens; ti++) {
        var col = tc[ti % tc.length];
        var r = parseInt(col.slice(1, 3), 16);
        var g = parseInt(col.slice(3, 5), 16);
        var b = parseInt(col.slice(5, 7), 16);

        // Build screen-space path
        var path = [];
        for (var li = 0; li < nLayers; li++) {
            path.push({
                x: SX(tokenWorldX[ti][li]),
                y: LY(li)
            });
        }

        // Outer glow
        c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.06)';
        c.lineWidth = 10;
        c.lineJoin = 'round';
        c.lineCap = 'round';
        c.beginPath();
        c.moveTo(path[0].x, path[0].y);
        for (var li = 1; li < nLayers; li++) {
            var prev = path[li - 1], curr = path[li];
            c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
        }
        c.stroke();

        // Main pathline
        c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.7)';
        c.lineWidth = 2.5;
        c.beginPath();
        c.moveTo(path[0].x, path[0].y);
        for (var li = 1; li < nLayers; li++) {
            var prev = path[li - 1], curr = path[li];
            c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
        }
        c.stroke();

        // ============================================================
        // PASS 3: FORCE DECOMPOSITION at each layer node
        // Cyan arrow = attention push, Orange arrow = MLP push
        // These branch off the pathline showing WHY it bends
        // ============================================================
        for (var li = 0; li < nLayers; li++) {
            var pt = path[li];
            var isActive = (li === currentLayer);
            var pixPerWorld = plotW / vw;
            var maxArrow = layerH * 0.35;
            var arrowAlpha = isActive ? 0.85 : 0.3;

            // Attention arrow (cyan) — horizontal component only for clarity
            if (attnDeltas && !isEmb) {
                var avx = tokenAttnDx[ti][li] * pixPerWorld;
                var avy = 0; // keep arrows horizontal so they don't overlap pathline
                var aLen = Math.abs(avx);
                if (aLen > 1.5) {
                    if (aLen > maxArrow) avx *= maxArrow / aLen;
                    drawFlowArrow(c, pt.x, pt.y - 3, avx, avy,
                        'rgba(0,200,255,' + arrowAlpha + ')', maxArrow);
                }
            }

            // MLP arrow (orange)
            if (mlpDeltas && !isEmb) {
                var mvx = tokenMlpDx[ti][li] * pixPerWorld;
                var mvy = 0;
                var mLen = Math.abs(mvx);
                if (mLen > 1.5) {
                    if (mLen > maxArrow) mvx *= maxArrow / mLen;
                    drawFlowArrow(c, pt.x, pt.y + 3, mvx, mvy,
                        'rgba(255,165,0,' + arrowAlpha + ')', maxArrow);
                }
            }

            // ============================================================
            // PASS 4: TRANSPORT FRAME at each layer node
            // Shows local coordinate rotation = parallel transport
            // ============================================================
            if (fibreState.showTransportFrame && !isEmb) {
                // Compute Jacobian of the deformation field at this token's position
                var edxCumTF = new Float64Array(nP);
                var edyCumTF = new Float64Array(nP);
                for (var j = 0; j < nP; j++) {
                    if (mode === 'single') {
                        edxCumTF[j] = activeDeltas[li][j][dxDim] * amp;
                        edyCumTF[j] = activeDeltas[li][j][dyDim] * amp;
                    } else if (mode === 'cumfwd') {
                        for (var cl = 0; cl <= li; cl++) {
                            edxCumTF[j] += activeDeltas[cl][j][dxDim] * amp;
                            edyCumTF[j] += activeDeltas[cl][j][dyDim] * amp;
                        }
                    } else {
                        for (var cl = li; cl < nLayers; cl++) {
                            edxCumTF[j] += activeDeltas[cl][j][dxDim] * amp;
                            edyCumTF[j] += activeDeltas[cl][j][dyDim] * amp;
                        }
                    }
                }

                var eps = sig * 0.1;
                function interpTF(px, py) {
                    var vvx = 0, vvy = 0, ws = 0;
                    for (var k = 0; k < nP; k++) {
                        var eex = px - fx[k], eey = py - fy[k];
                        var w = Math.exp(-(eex * eex + eey * eey) * s2i);
                        vvx += w * edxCumTF[k]; vvy += w * edyCumTF[k]; ws += w;
                    }
                    if (ws > 1e-15) { vvx /= ws; vvy /= ws; }
                    return [vvx, vvy];
                }

                var bpx = fx[ti], bpy = fy[ti];
                var v0 = interpTF(bpx, bpy);
                var vR = interpTF(bpx + eps, bpy);
                var vU = interpTF(bpx, bpy + eps);

                // Jacobian
                var J00 = 1 + (vR[0] - v0[0]) / eps;
                var J01 = (vU[0] - v0[0]) / eps;
                var J10 = (vR[1] - v0[1]) / eps;
                var J11 = 1 + (vU[1] - v0[1]) / eps;

                var fSize = Math.min(layerH * 0.2, 12);
                var e1x = J00 * fSize * pixPerWorld * 0.015;
                var e1y = J10 * fSize * pixPerWorld * 0.015;
                var e2x = J01 * fSize * pixPerWorld * 0.015;
                var e2y = J11 * fSize * pixPerWorld * 0.015;

                // Clamp
                var maxF = fSize * 2;
                var e1L = Math.hypot(e1x, e1y);
                var e2L = Math.hypot(e2x, e2y);
                if (e1L > maxF) { e1x *= maxF / e1L; e1y *= maxF / e1L; }
                if (e2L > maxF) { e2x *= maxF / e2L; e2y *= maxF / e2L; }

                var frameAlpha = isActive ? 0.7 : 0.2;
                c.globalAlpha = frameAlpha;
                drawFlowArrow(c, pt.x, pt.y, e1x, e1y, 'rgba(255,255,100,0.9)', maxF);
                drawFlowArrow(c, pt.x, pt.y, e2x, e2y, 'rgba(255,100,255,0.9)', maxF);
                c.globalAlpha = 1.0;
            }

            // Node dot
            var dotR = isActive ? 5 : 3;
            c.beginPath();
            c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
            c.fillStyle = col;
            c.fill();
            if (isActive) {
                c.strokeStyle = '#fff';
                c.lineWidth = 1.5;
                c.stroke();
            }
        }

        // Token label at bottom (layer 0)
        c.font = 'bold 10px monospace';
        c.fillStyle = col;
        c.textAlign = 'center';
        c.fillText('[' + ti + '] ' + D.tokens[ti], path[0].x, path[0].y + 16);
    }

    // ================================================================
    // LEGEND
    // ================================================================
    var legX = margin + plotW + 10;
    var legY = margin + 10;
    c.font = '9px monospace';
    c.textAlign = 'left';

    c.fillStyle = '#888';
    c.fillText('Parallel Transport', legX, legY); legY += 14;

    c.strokeStyle = 'rgba(233,69,96,0.6)';
    c.lineWidth = 2.5;
    c.beginPath(); c.moveTo(legX, legY); c.lineTo(legX + 20, legY); c.stroke();
    c.fillStyle = '#a0a0c0';
    c.fillText('Token path', legX + 26, legY + 3); legY += 16;

    if (attnDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(0,200,255,0.8)', 20);
        c.fillStyle = 'rgba(0,200,255,0.8)';
        c.fillText('Attention push', legX + 26, legY + 3); legY += 14;
    }
    if (mlpDeltas) {
        drawFlowArrow(c, legX, legY, 18, 0, 'rgba(255,165,0,0.8)', 20);
        c.fillStyle = 'rgba(255,165,0,0.8)';
        c.fillText('MLP push', legX + 26, legY + 3); legY += 14;
    }
    if (fibreState.showTransportFrame) {
        drawFlowArrow(c, legX, legY, 14, 0, 'rgba(255,255,100,0.8)', 16);
        c.fillStyle = 'rgba(255,255,100,0.8)';
        c.fillText('Frame e1', legX + 26, legY + 3); legY += 12;
        drawFlowArrow(c, legX, legY, 0, -12, 'rgba(255,100,255,0.8)', 14);
        c.fillStyle = 'rgba(255,100,255,0.8)';
        c.fillText('Frame e2', legX + 26, legY + 3); legY += 16;
    }

    c.fillStyle = '#555';
    c.font = '8px monospace';
    c.fillText('Path bends = information flow', legX, legY); legY += 10;
    c.fillText('Frame rotation = holonomy', legX, legY);

    c.restore();

    // HUD
    drawFibreBundleHUD(c, W, H, nTokens, nLayers, hiddenDim, currentLayer);
}

function drawFibreBundle() {
    var cv = document.getElementById('cv');
    var c = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace';
        c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    var nTokens = D.n_real;
    var nLayers = D.n_layers;
    var hiddenDim = D.hidden_dim;
    var nP = D.n_points;

    var dx = +document.getElementById('sl-dx').value;
    var dy = +document.getElementById('sl-dy').value;
    var amp = +document.getElementById('sl-amp').value;
    var t = +document.getElementById('sl-t').value;
    var sig = +document.getElementById('sl-sig').value;
    var currentLayer = +document.getElementById('sl-layer').value;
    var showGrid = document.getElementById('cb-grid').checked;
    var showHeat = document.getElementById('cb-heat').checked;
    var showSC = document.getElementById('cb-sc').checked;
    var showVec = document.getElementById('cb-vec').checked;
    var mode = document.getElementById('sel-mode').value;

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;

    var attnDeltas = D.attn_deltas || null;
    var mlpDeltas = D.mlp_deltas || null;

    var isEmb = (mode === 'embedding');

    dx = Math.min(dx, hiddenDim - 1);
    dy = Math.min(dy, hiddenDim - 1);

    // Layout computation
    var margin = 30;
    var labelW = 35;
    var labelH = 25;
    var availW = (W / zoomLevel) - 2 * margin - labelW;
    var availH = (H / zoomLevel) - 2 * margin - labelH;

    var gapFracX = 0.35;
    var gapFracY = 0.45; // slightly more gap for flow lines between layers
    var roomW = Math.max(30, Math.floor(availW / (nTokens * (1 + gapFracX))));
    var roomH = Math.max(30, Math.floor(availH / (nLayers * (1 + gapFracY))));
    var roomSize = Math.min(roomW, roomH);
    var gapX = Math.max(8, Math.floor(roomSize * gapFracX));
    var gapY = Math.max(12, Math.floor(roomSize * gapFracY));

    // Extract 2D positions
    var fx = new Float64Array(nP), fy = new Float64Array(nP);
    for (var i = 0; i < nP; i++) {
        fx[i] = D.fixed_pos[i][dx];
        fy[i] = D.fixed_pos[i][dy];
    }

    // Compute view bounds
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (var i = 0; i < nP; i++) {
        if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
        if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    }
    var mr = Math.max(mxx - mnx, mxy - mny) || 1;
    var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
    var pd = 0.15;
    var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
    var vw = mr * (1 + 2 * pd), vh = vw;

    // Precompute per-layer raw deltas (before mode accumulation)
    var edxAll = [];
    var edyAll = [];
    for (var lay = 0; lay < nLayers; lay++) {
        var edxL = new Float64Array(nP);
        var edyL = new Float64Array(nP);
        for (var j = 0; j < nP; j++) {
            edxL[j] = activeDeltas[lay][j][dx] * amp;
            edyL[j] = activeDeltas[lay][j][dy] * amp;
        }
        edxAll.push(edxL);
        edyAll.push(edyL);
    }

    var N = Math.max(4, Math.min(16, Math.floor(roomSize / 4)));
    var s2i = 1 / (2 * sig * sig);

    c.save();
    c.translate(panX, panY);
    c.scale(zoomLevel, zoomLevel);

    var startX = margin + labelW;
    var startY = margin;

    // Helper: compute mode-aware cumulative deltas for a given layer
    function computeEdxEdyForLayer(li) {
        var edxCum = new Float64Array(nP);
        var edyCum = new Float64Array(nP);
        if (isEmb) return { edx: edxCum, edy: edyCum };
        if (mode === 'single') {
            for (var j = 0; j < nP; j++) {
                edxCum[j] = edxAll[li][j];
                edyCum[j] = edyAll[li][j];
            }
        } else if (mode === 'cumfwd') {
            for (var cl = 0; cl <= li; cl++) {
                for (var j = 0; j < nP; j++) {
                    edxCum[j] += edxAll[cl][j];
                    edyCum[j] += edyAll[cl][j];
                }
            }
        } else { // cumbwd
            for (var cl = li; cl < nLayers; cl++) {
                for (var j = 0; j < nP; j++) {
                    edxCum[j] += edxAll[cl][j];
                    edyCum[j] += edyAll[cl][j];
                }
            }
        }
        return { edx: edxCum, edy: edyCum };
    }

    // Helper: RBF interpolate a deformation field onto a grid
    function buildDeformedGrid(edxCum, edyCum) {
        var nV = (N + 1) * (N + 1);
        var oX = new Float64Array(nV), oY = new Float64Array(nV);
        var gX = new Float64Array(nV), gY = new Float64Array(nV);

        for (var gy = 0; gy <= N; gy++) {
            for (var gx = 0; gx <= N; gx++) {
                var gi = gy * (N + 1) + gx;
                oX[gi] = vx0 + (gx / N) * vw;
                oY[gi] = vy0 + (gy / N) * vh;
            }
        }

        if (isEmb) {
            for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; }
        } else {
            var _itpM = document.getElementById('sel-itp').value;
            for (var gi = 0; gi < nV; gi++) {
                var px = oX[gi], py = oY[gi];
                var _iRes = interpolateGridPoint(px, py, fx, fy, edxCum, edyCum, nP, sig, _itpM);
                gX[gi] = px + t * _iRes[0];
                gY[gi] = py + t * _iRes[1];
            }
        }

        // Compute strain
        var sH = new Float64Array(N * (N + 1));
        var sV = new Float64Array((N + 1) * N);
        for (var ey = 0; ey <= N; ey++) {
            for (var ex = 0; ex < N; ex++) {
                var a = ey * (N + 1) + ex, b = a + 1;
                var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
                var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
                sH[ey * N + ex] = od > 1e-12 ? dd / od : 1;
            }
        }
        for (var ey = 0; ey < N; ey++) {
            for (var ex = 0; ex <= N; ex++) {
                var a = ey * (N + 1) + ex, b = (ey + 1) * (N + 1) + ex;
                var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
                var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
                sV[ey * (N + 1) + ex] = od > 1e-12 ? dd / od : 1;
            }
        }

        return { oX: oX, oY: oY, gX: gX, gY: gY, sH: sH, sV: sV, nV: nV };
    }

    // Helper: RBF interpolate the attn/mlp vector field at a world-space point
    function interpolateComponentField(px, py, compDeltas, layerIdx) {
        if (!compDeltas) return [0, 0];
        var _compEdx = new Float64Array(nP);
        var _compEdy = new Float64Array(nP);
        for (var k = 0; k < nP; k++) {
            _compEdx[k] = compDeltas[layerIdx][k][dx] * amp;
            _compEdy[k] = compDeltas[layerIdx][k][dy] * amp;
        }
        return interpolateGridPoint(px, py, fx, fy, _compEdx, _compEdy, nP, sig,
            document.getElementById('sel-itp').value);
    }

    // ========== PASS 1: Draw flow streamlines BETWEEN layers (behind rooms) ==========
    if (fibreState.showFlowLines && !isEmb) {
        c.globalAlpha = 0.35;

        for (var li = 0; li < nLayers - 1; li++) {
            var rowIdx = nLayers - 1 - li;
            var nextRowIdx = nLayers - 2 - li;
            var roomCY = startY + rowIdx * (roomSize + gapY);
            var nextRoomCY = startY + nextRowIdx * (roomSize + gapY);

            var layerDeltas = computeEdxEdyForLayer(li);
            var nextDeltas = computeEdxEdyForLayer(li + 1);

            for (var ti = 0; ti < nTokens; ti++) {
                var roomCX = startX + ti * (roomSize + gapX);

                // Sample a sparse set of grid points to draw streamlines
                var streamStep = Math.max(1, Math.floor(N / 3));
                for (var sgy = 0; sgy <= N; sgy += streamStep) {
                    for (var sgx = 0; sgx <= N; sgx += streamStep) {
                        var worldX = vx0 + (sgx / N) * vw;
                        var worldY = vy0 + (sgy / N) * vh;

                        // Interpolate deformation at this grid point for current layer
                        var vvx1 = 0, vvy1 = 0, ws1 = 0;
                        for (var k = 0; k < nP; k++) {
                            var eex = worldX - fx[k], eey = worldY - fy[k];
                            var w = Math.exp(-(eex * eex + eey * eey) * s2i);
                            vvx1 += w * layerDeltas.edx[k]; vvy1 += w * layerDeltas.edy[k]; ws1 += w;
                        }
                        if (ws1 > 1e-15) { vvx1 /= ws1; vvy1 /= ws1; }

                        // Same for next layer
                        var vvx2 = 0, vvy2 = 0, ws2 = 0;
                        for (var k = 0; k < nP; k++) {
                            var eex = worldX - fx[k], eey = worldY - fy[k];
                            var w = Math.exp(-(eex * eex + eey * eey) * s2i);
                            vvx2 += w * nextDeltas.edx[k]; vvy2 += w * nextDeltas.edy[k]; ws2 += w;
                        }
                        if (ws2 > 1e-15) { vvx2 /= ws2; vvy2 /= ws2; }

                        // Screen positions of this grid point in current and next room
                        var deformedX1 = worldX + t * vvx1;
                        var deformedY1 = worldY + t * vvy1;
                        var deformedX2 = worldX + t * vvx2;
                        var deformedY2 = worldY + t * vvy2;

                        var sx1 = roomCX + ((deformedX1 - vx0) / vw) * roomSize;
                        var sy1 = roomCY + ((deformedY1 - vy0) / vh) * roomSize;
                        var sx2 = roomCX + ((deformedX2 - vx0) / vw) * roomSize;
                        var sy2 = nextRoomCY + ((deformedY2 - vy0) / vh) * roomSize;

                        // Movement magnitude determines color intensity
                        var moveDist = Math.hypot(deformedX2 - deformedX1, deformedY2 - deformedY1);
                        var moveAlpha = Math.min(0.5, moveDist * 0.3 + 0.03);

                        // Color: red if expanding, blue if contracting, gray if isometric
                        var strain = (moveDist > 1e-8) ? moveDist / (vw / N + 1e-12) : 0;
                        var sc = s2c(0.5 + strain * 0.5);

                        c.strokeStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + moveAlpha.toFixed(2) + ')';
                        c.lineWidth = Math.min(1.5, 0.3 + moveDist * 0.5);

                        // Draw a smooth bezier curve between the two rooms
                        var midX = (sx1 + sx2) / 2 + (sx2 - sx1) * 0.3;
                        var midY = (sy1 + sy2) / 2;

                        c.beginPath();
                        c.moveTo(sx1, sy1);
                        c.quadraticCurveTo(midX, midY, sx2, sy2);
                        c.stroke();

                        // Small dot at the connection point in the next room
                        c.beginPath();
                        c.arc(sx2, sy2, 1, 0, Math.PI * 2);
                        c.fillStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + (moveAlpha * 1.5).toFixed(2) + ')';
                        c.fill();
                    }
                }
            }
        }
        c.globalAlpha = 1.0;
    }

    // ========== PASS 2: Draw each layer room ==========
    for (var li = 0; li < nLayers; li++) {
        var rowIdx = nLayers - 1 - li;
        var roomCY = startY + rowIdx * (roomSize + gapY);
        var isCurrentLayer = (li === currentLayer);

        // Layer label
        c.font = (isCurrentLayer ? 'bold ' : '') + '9px monospace';
        c.fillStyle = isCurrentLayer ? '#e94560' : '#666';
        c.textAlign = 'right';
        c.fillText('L' + li, startX - 8, roomCY + roomSize / 2 + 3);

        var layerDeltas = computeEdxEdyForLayer(li);
        var edxCum = layerDeltas.edx;
        var edyCum = layerDeltas.edy;

        for (var ti = 0; ti < nTokens; ti++) {
            var roomCX = startX + ti * (roomSize + gapX);

            // Room background
            var bgAlpha = isCurrentLayer ? 0.15 : 0.06;
            c.fillStyle = 'rgba(30,30,60,' + bgAlpha + ')';
            c.fillRect(roomCX, roomCY, roomSize, roomSize);

            // Room border
            c.strokeStyle = isCurrentLayer ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.25)';
            c.lineWidth = isCurrentLayer ? 1.5 : 0.5;
            c.strokeRect(roomCX, roomCY, roomSize, roomSize);

            // Build deformed grid for this room
            var grid = buildDeformedGrid(edxCum, edyCum);

            // Coordinate transforms: world -> room screen
            // We need closures that capture roomCX/roomCY properly
            var _roomCX = roomCX, _roomCY = roomCY;
            var _vx0 = vx0, _vy0 = vy0, _vw = vw, _vh = vh, _roomSize = roomSize;

            // ---- Strain heatmap ----
            if (showHeat && !isEmb) {
                for (var hy = 0; hy < N; hy++) {
                    for (var hx = 0; hx < N; hx++) {
                        var avg = (grid.sH[hy * N + hx] + grid.sH[(hy + 1) * N + hx] +
                                   grid.sV[hy * (N + 1) + hx] + grid.sV[hy * (N + 1) + hx + 1]) / 4;
                        var co = s2c(avg);
                        var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
                        var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;

                        var sx00 = _roomCX + ((grid.gX[i00] - _vx0) / _vw) * _roomSize;
                        var sy00 = _roomCY + ((grid.gY[i00] - _vy0) / _vh) * _roomSize;
                        var sx10 = _roomCX + ((grid.gX[i10] - _vx0) / _vw) * _roomSize;
                        var sy10 = _roomCY + ((grid.gY[i10] - _vy0) / _vh) * _roomSize;
                        var sx11 = _roomCX + ((grid.gX[i11] - _vx0) / _vw) * _roomSize;
                        var sy11 = _roomCY + ((grid.gY[i11] - _vy0) / _vh) * _roomSize;
                        var sx01 = _roomCX + ((grid.gX[i01] - _vx0) / _vw) * _roomSize;
                        var sy01 = _roomCY + ((grid.gY[i01] - _vy0) / _vh) * _roomSize;

                        c.beginPath();
                        c.moveTo(sx00, sy00);
                        c.lineTo(sx10, sy10);
                        c.lineTo(sx11, sy11);
                        c.lineTo(sx01, sy01);
                        c.closePath();
                        c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.4)';
                        c.fill();
                    }
                }
            }

            // ---- Deformed grid lines ----
            if (showGrid && !isEmb) {
                c.lineWidth = 0.6;
                // Horizontal edges
                for (var dhy = 0; dhy <= N; dhy++) {
                    for (var dhx = 0; dhx < N; dhx++) {
                        var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
                        var es = grid.sH[dhy * N + dhx];
                        if (showSC) {
                            var ec = s2c(es);
                            c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.8)';
                        } else {
                            c.strokeStyle = 'rgba(200,200,200,0.4)';
                        }
                        c.beginPath();
                        c.moveTo(_roomCX + ((grid.gX[di1] - _vx0) / _vw) * _roomSize,
                                 _roomCY + ((grid.gY[di1] - _vy0) / _vh) * _roomSize);
                        c.lineTo(_roomCX + ((grid.gX[di2] - _vx0) / _vw) * _roomSize,
                                 _roomCY + ((grid.gY[di2] - _vy0) / _vh) * _roomSize);
                        c.stroke();
                    }
                }
                // Vertical edges
                for (var dvx = 0; dvx <= N; dvx++) {
                    for (var dvy = 0; dvy < N; dvy++) {
                        var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
                        var vs = grid.sV[dvy * (N + 1) + dvx];
                        if (showSC) {
                            var vc = s2c(vs);
                            c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.8)';
                        } else {
                            c.strokeStyle = 'rgba(200,200,200,0.4)';
                        }
                        c.beginPath();
                        c.moveTo(_roomCX + ((grid.gX[dvi1] - _vx0) / _vw) * _roomSize,
                                 _roomCY + ((grid.gY[dvi1] - _vy0) / _vh) * _roomSize);
                        c.lineTo(_roomCX + ((grid.gX[dvi2] - _vx0) / _vw) * _roomSize,
                                 _roomCY + ((grid.gY[dvi2] - _vy0) / _vh) * _roomSize);
                        c.stroke();
                    }
                }
            }

            // ---- Reference grid in embedding mode ----
            if (isEmb) {
                c.strokeStyle = 'rgba(255,255,255,0.15)';
                c.lineWidth = 0.5;
                for (var ry2 = 0; ry2 <= N; ry2++) {
                    c.beginPath();
                    for (var rx2 = 0; rx2 <= N; rx2++) {
                        var ri = ry2 * (N + 1) + rx2;
                        var rsx = _roomCX + ((grid.oX[ri] - _vx0) / _vw) * _roomSize;
                        var rsy = _roomCY + ((grid.oY[ri] - _vy0) / _vh) * _roomSize;
                        if (rx2 === 0) c.moveTo(rsx, rsy);
                        else c.lineTo(rsx, rsy);
                    }
                    c.stroke();
                }
                for (var rx3 = 0; rx3 <= N; rx3++) {
                    c.beginPath();
                    for (var ry3 = 0; ry3 <= N; ry3++) {
                        var ri3 = ry3 * (N + 1) + rx3;
                        var rsx3 = _roomCX + ((grid.oX[ri3] - _vx0) / _vw) * _roomSize;
                        var rsy3 = _roomCY + ((grid.oY[ri3] - _vy0) / _vh) * _roomSize;
                        if (ry3 === 0) c.moveTo(rsx3, rsy3);
                        else c.lineTo(rsx3, rsy3);
                    }
                    c.stroke();
                }
            }

            // ========== VECTOR FIELD OVERLAY: Attention (cyan) + MLP (orange) ==========
            if (fibreState.showAttnField && attnDeltas && !isEmb) {
                var vecStep = Math.max(1, Math.floor(N / 4));
                var maxArrowLen = roomSize / 3;
                var arrowScale = fibreState.flowArrowScale;

                for (var viy = 0; viy <= N; viy += vecStep) {
                    for (var vix = 0; vix <= N; vix += vecStep) {
                        var vi = viy * (N + 1) + vix;
                        var worldPx = grid.oX[vi], worldPy = grid.oY[vi];

                        var attnField = interpolateComponentField(worldPx, worldPy, attnDeltas, li);
                        var screenBaseX = _roomCX + ((grid.gX[vi] - _vx0) / _vw) * _roomSize;
                        var screenBaseY = _roomCY + ((grid.gY[vi] - _vy0) / _vh) * _roomSize;

                        var pixPerWorld = roomSize / vw;
                        var avx = attnField[0] * t * pixPerWorld * arrowScale;
                        var avy = attnField[1] * t * pixPerWorld * arrowScale;

                        drawFlowArrow(c, screenBaseX, screenBaseY, avx, avy,
                            'rgba(0,200,255,0.55)', maxArrowLen);
                    }
                }
            }

            if (fibreState.showMlpField && mlpDeltas && !isEmb) {
                var vecStep = Math.max(1, Math.floor(N / 4));
                var maxArrowLen = roomSize / 3;
                var arrowScale = fibreState.flowArrowScale;

                for (var viy = 0; viy <= N; viy += vecStep) {
                    for (var vix = 0; vix <= N; vix += vecStep) {
                        var vi = viy * (N + 1) + vix;
                        var worldPx = grid.oX[vi], worldPy = grid.oY[vi];

                        var mlpField = interpolateComponentField(worldPx, worldPy, mlpDeltas, li);
                        var screenBaseX = _roomCX + ((grid.gX[vi] - _vx0) / _vw) * _roomSize;
                        var screenBaseY = _roomCY + ((grid.gY[vi] - _vy0) / _vh) * _roomSize;

                        var pixPerWorld = roomSize / vw;
                        var mvx = mlpField[0] * t * pixPerWorld * arrowScale;
                        var mvy = mlpField[1] * t * pixPerWorld * arrowScale;

                        drawFlowArrow(c, screenBaseX, screenBaseY, mvx, mvy,
                            'rgba(255,165,0,0.55)', maxArrowLen);
                    }
                }
            }

            // ---- Transport frame at token position ----
            if (fibreState.showTransportFrame && !isEmb && edxCum && edyCum) {
                var tokScreenX = _roomCX + ((fx[ti] + t * edxCum[ti] - _vx0) / _vw) * _roomSize;
                var tokScreenY = _roomCY + ((fy[ti] + t * edyCum[ti] - _vy0) / _vh) * _roomSize;
                var frameSize = roomSize / 5;
                drawTransportFrame(c, tokScreenX, tokScreenY, edxCum, edyCum,
                    fx, fy, ti, nP, sig, frameSize);
            }

            // ---- Token dot ----
            var tokX = _roomCX + ((fx[ti] - _vx0) / _vw) * _roomSize;
            var tokY = _roomCY + ((fy[ti] - _vy0) / _vh) * _roomSize;

            if (!isEmb) {
                var tokDeformDX = t * edxCum[ti];
                var tokDeformDY = t * edyCum[ti];
                var localStrainMag = Math.sqrt(tokDeformDX * tokDeformDX + tokDeformDY * tokDeformDY);
                var strainRadius = Math.min(roomSize / 4, localStrainMag * 0.5);
                if (strainRadius > 1.5) {
                    var normStrain = Math.min(2.0, localStrainMag / (vw / N + 1e-12));
                    var ringColor = s2c(0.5 + normStrain * 0.5);
                    c.beginPath();
                    c.arc(tokX, tokY, strainRadius, 0, Math.PI * 2);
                    c.strokeStyle = 'rgba(' + ringColor[0] + ',' + ringColor[1] + ',' + ringColor[2] + ',0.5)';
                    c.lineWidth = 1.2;
                    c.stroke();
                }
            }

            var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
                      '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];
            c.beginPath();
            c.arc(tokX, tokY, Math.max(2, roomSize / 20), 0, Math.PI * 2);
            c.fillStyle = tc[ti % tc.length];
            c.fill();
            c.strokeStyle = '#fff';
            c.lineWidth = 0.5;
            c.stroke();

            // Token label at bottom of column
            if (li === 0) {
                c.font = 'bold 8px monospace';
                c.fillStyle = '#e94560';
                c.textAlign = 'center';
                c.fillText('[' + ti + ']', roomCX + roomSize / 2, roomCY + roomSize + 10);
                if (roomSize > 35) {
                    c.font = '7px monospace';
                    c.fillStyle = '#a0a0c0';
                    c.fillText(D.tokens[ti], roomCX + roomSize / 2, roomCY + roomSize + 19);
                }
            }
        } // end token loop

        // ========== PASS 3: Diffeomorphism connections between layers ==========
        if (fibreState.showConnections && li < nLayers - 1 && !isEmb) {
            var nextRowIdx = nLayers - 2 - li;
            var nextRoomCY = startY + nextRowIdx * (roomSize + gapY);

            var nextDeltas = computeEdxEdyForLayer(li + 1);
            var edxNext = nextDeltas.edx;
            var edyNext = nextDeltas.edy;

            for (var ti = 0; ti < nTokens; ti++) {
                var roomCXt = startX + ti * (roomSize + gapX);

                var moveDist = Math.hypot(
                    edxNext[ti] - edxCum[ti],
                    edyNext[ti] - edyCum[ti]
                );
                var moveAlpha = Math.min(0.6, moveDist * 0.01 + 0.05);

                var sy1 = roomCY;
                var sy2 = nextRoomCY + roomSize;
                var sx1 = roomCXt + roomSize / 2;
                var sx2 = roomCXt + roomSize / 2;

                if (moveDist > 0.01) {
                    c.strokeStyle = 'rgba(233,69,96,' + moveAlpha.toFixed(2) + ')';
                } else {
                    c.strokeStyle = 'rgba(83,168,182,' + (moveAlpha * 0.5).toFixed(2) + ')';
                }
                c.lineWidth = Math.min(2, 0.3 + moveDist * 0.005);

                var midX = (sx1 + sx2) / 2 + Math.sin(ti * 1.5 + li * 0.7) * gapX * 0.6;
                c.beginPath();
                c.moveTo(sx1, sy1);
                c.quadraticCurveTo(midX, (sy1 + sy2) / 2, sx2, sy2);
                c.stroke();
            }
        }
    } // end layer loop

  // ========== Axis labels ==========
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  var totalW = nTokens * (roomSize + gapX) - gapX;
  c.fillText(
    '\u2190 Base Manifold: Token Index \u2192',
    startX + totalW / 2,
    startY + nLayers * (roomSize + gapY) + 28
  );

  c.save();
  c.translate(startX - 35, startY + nLayers * (roomSize + gapY) / 2);
  c.rotate(-Math.PI / 2);
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText('\u2190 Fibre: Layer Depth \u2192', 0, 0);
  c.restore();

  // ========== Legend for vector field arrows ==========
  if ((fibreState.showAttnField || fibreState.showMlpField) && !isEmb) {
    var legX = startX + totalW + 20;
    var legY = startY + 10;
    c.font = '9px monospace';
    c.textAlign = 'left';

    if (fibreState.showAttnField && attnDeltas) {
      drawFlowArrow(c, legX, legY, 18, 0, 'rgba(0,200,255,0.8)', 20);
      c.fillStyle = 'rgba(0,200,255,0.8)';
      c.fillText('Attention', legX + 24, legY + 3);
      legY += 16;
    }
    if (fibreState.showMlpField && mlpDeltas) {
      drawFlowArrow(c, legX, legY, 18, 0, 'rgba(255,165,0,0.8)', 20);
      c.fillStyle = 'rgba(255,165,0,0.8)';
      c.fillText('MLP', legX + 24, legY + 3);
      legY += 16;
    }
    if (fibreState.showTransportFrame) {
      drawFlowArrow(c, legX, legY, 14, 0, 'rgba(255,255,100,0.8)', 16);
      c.fillStyle = 'rgba(255,255,100,0.8)';
      c.fillText('Frame e1', legX + 24, legY + 3);
      legY += 14;
      drawFlowArrow(c, legX, legY, 0, -14, 'rgba(255,100,255,0.8)', 16);
      c.fillStyle = 'rgba(255,100,255,0.8)';
      c.fillText('Frame e2', legX + 24, legY + 3);
      legY += 16;
    }
    if (fibreState.showFlowLines) {
      c.strokeStyle = 'rgba(150,150,200,0.5)';
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(legX, legY);
      c.quadraticCurveTo(legX + 10, legY - 8, legX + 18, legY);
      c.stroke();
      c.fillStyle = 'rgba(150,150,200,0.7)';
      c.fillText('Flow lines', legX + 24, legY + 3);
    }
  }

  c.restore();

  // ========== HUD ==========
  drawFibreBundleHUD(c, W, H, nTokens, nLayers, hiddenDim, currentLayer);
}

function drawFibreBundleHUD(c, W, H, nTokens, nLayers, hiddenDim, currentLayer) {
  var dx = +document.getElementById('sl-dx').value;
  var dy = +document.getElementById('sl-dy').value;
  var amp = +document.getElementById('sl-amp').value;
  var t = +document.getElementById('sl-t').value;
  var decompLabel = getDecompLabel();

  c.font = '11px monospace';
  c.fillStyle = 'rgba(255,255,255,0.45)';
  c.textAlign = 'left';
  c.fillText(
    'FIBRE BUNDLE  Tokens:' + nTokens +
    '  Layers:' + nLayers +
    '  Dims:' + dx + ',' + dy +
    '  Amp:' + amp.toFixed(1) +
    '  t:' + t.toFixed(2) +
    '  Decomp:' + decompLabel +
    '  Colormap:' + fibreState.colormap +
    '  Layer:' + currentLayer,
    12, 16
  );

  c.font = '9px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.fillText(
    '\u2190\u2192 Dim X | \u2191\u2193 Dim Y | [/] Layer | A/Z Amp | ;/\' t | C=Connections | M=Colormap | Shift+Drag=Pan | Scroll=Zoom | 0=Reset',
    12, H - 8
  );
}

function onKeyFibre(e) {
  if (document.activeElement === document.getElementById('txt-in')) return;
  if (document.activeElement === document.getElementById('txt-b')) return;
  var maxDim = D ? D.hidden_dim - 1 : 767;
  var sdx = document.getElementById('sl-dx');
  var sdy = document.getElementById('sl-dy');
  var sdz = document.getElementById('sl-dz');

  // Shift+Arrow = Dim Z (third axis)
  if (e.shiftKey && e.key === 'ArrowRight') {
    e.preventDefault();
    var newZ = +sdz.value + 1;
    if (newZ > maxDim) newZ = 0;
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowLeft') {
    e.preventDefault();
    var newZ = +sdz.value - 1;
    if (newZ < 0) newZ = maxDim;
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowUp') {
    e.preventDefault();
    var newZ = +sdz.value + 10;
    if (newZ > maxDim) newZ = newZ % (maxDim + 1);
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  } else if (e.shiftKey && e.key === 'ArrowDown') {
    e.preventDefault();
    var newZ = +sdz.value - 10;
    if (newZ < 0) newZ = (newZ + maxDim + 1) % (maxDim + 1);
    while (newZ === +sdx.value || newZ === +sdy.value) newZ = (newZ - 1 + maxDim + 1) % (maxDim + 1);
    sdz.value = newZ;
    sdz.dispatchEvent(new Event('input'));
    return;
  }

  if (e.key === 'ArrowRight') {
    e.preventDefault();
    var newX = +sdx.value + 1;
    if (newX > maxDim) newX = 0;
    if (newX === +sdy.value) newX = (newX + 1) % (maxDim + 1);
    sdx.value = newX;
    sdx.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    var newX = +sdx.value - 1;
    if (newX < 0) newX = maxDim;
    if (newX === +sdy.value) newX = (newX - 1 + maxDim + 1) % (maxDim + 1);
    sdx.value = newX;
    sdx.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    var newY = +sdy.value + 1;
    if (newY > maxDim) newY = 0;
    if (newY === +sdx.value) newY = (newY + 1) % (maxDim + 1);
    sdy.value = newY;
    sdy.dispatchEvent(new Event('input'));
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    var newY = +sdy.value - 1;
    if (newY < 0) newY = maxDim;
    if (newY === +sdx.value) newY = (newY - 1 + maxDim + 1) % (maxDim + 1);
    sdy.value = newY;
    sdy.dispatchEvent(new Event('input'));
  } else if (e.key === '.' || e.key === ']') {
    var sl = document.getElementById('sl-layer');
    sl.value = Math.min(+sl.max, +sl.value + 1);
    sl.dispatchEvent(new Event('input'));
  } else if (e.key === ',' || e.key === '[') {
    var sl = document.getElementById('sl-layer');
    sl.value = Math.max(0, +sl.value - 1);
    sl.dispatchEvent(new Event('input'));
  } else if (e.key === "'" ) {
    var st = document.getElementById('sl-t');
    st.value = Math.min(1, +st.value + 0.05).toFixed(2);
    st.dispatchEvent(new Event('input'));
  } else if (e.key === ';') {
    var st = document.getElementById('sl-t');
    st.value = Math.max(0, +st.value - 0.05).toFixed(2);
    st.dispatchEvent(new Event('input'));
  } else if (e.key === 'a' || e.key === 'A') {
    var sa = document.getElementById('sl-amp');
    sa.value = Math.min(500, +sa.value * 1.3).toFixed(1);
    sa.dispatchEvent(new Event('input'));
  } else if (e.key === 'z' || e.key === 'Z') {
    var sa = document.getElementById('sl-amp');
    sa.value = Math.max(0.1, +sa.value / 1.3).toFixed(1);
    sa.dispatchEvent(new Event('input'));
  } else if (e.key === 'c' || e.key === 'C') {
    fibreState.showConnections = !fibreState.showConnections;
    draw();
  } else if (e.key === '0') {
    zoomLevel = 1.0; panX = 0; panY = 0;
    fibreState.scrollY = 0;
    fibreState.rotX = -0.3; fibreState.rotY = 0.4;
    draw();
  } else if (e.key === 'r' || e.key === 'R') {
    rstAll();
  }
}

// ---- Colormaps ----
function neuronColor(val, colormap) {
  // val in [0, 1]
  val = Math.max(0, Math.min(1, val));
  if (colormap === 'coolhot') {
    return valToColor(val);
  } else if (colormap === 'viridis') {
    // Approximate viridis
    var r, g, b;
    if (val < 0.25) {
      var t = val / 0.25;
      r = Math.floor(68 + t * (-4)); g = Math.floor(1 + t * 50); b = Math.floor(84 + t * 74);
    } else if (val < 0.5) {
      var t = (val - 0.25) / 0.25;
      r = Math.floor(64 - t * 30); g = Math.floor(51 + t * 70); b = Math.floor(158 - t * 20);
    } else if (val < 0.75) {
      var t = (val - 0.5) / 0.25;
      r = Math.floor(34 + t * 100); g = Math.floor(121 + t * 60); b = Math.floor(138 - t * 60);
    } else {
      var t = (val - 0.75) / 0.25;
      r = Math.floor(134 + t * 119); g = Math.floor(181 + t * 40); b = Math.floor(78 - t * 50);
    }
    return [r, g, b];
  } else {
    // Grayscale
    var v = Math.floor(val * 255);
    return [v, v, v];
  }
}

// ---- Main Fibre Bundle Drawing ----


function drawNeuronRoom(c, x, y, w, h, gridCols, gridRows, pixSize, hiddenDim, acts, useAbs, highlight, tokenIdx, layerIdx) {
  // Border
  c.strokeStyle = highlight ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.3)';
  c.lineWidth = highlight ? 1.5 : 0.5;
  c.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);

  // Draw each neuron as a pixel
  for (var ni = 0; ni < hiddenDim; ni++) {
    var val = acts[ni];
    if (useAbs) val = Math.abs(val * 2 - 1);

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    var px = x + col * pixSize;
    var py = y + row * pixSize;

    var rgb = neuronColor(val, fibreState.colormap);
    c.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    c.fillRect(px, py, pixSize, pixSize);
  }

  // Fill remaining cells dark
  for (var ni = hiddenDim; ni < gridCols * gridRows; ni++) {
    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    c.fillStyle = '#0a0515';
    c.fillRect(x + col * pixSize, y + row * pixSize, pixSize, pixSize);
  }
}

function drawDiffeoConnections(c, tokenX, layerY, nextLayerY, roomW, roomH,
  gridCols, gridRows, pixSize, hiddenDim, acts, nextActs, useAbs, layerIdx) {
  // Draw thin lines connecting corresponding neurons between layers
  // Only draw a subset for performance
  var step = Math.max(1, Math.floor(hiddenDim * (1 - fibreState.connectionDensity)));
  if (step < 1) step = 1;

  c.lineWidth = 0.3;

  for (var ni = 0; ni < hiddenDim; ni += step) {
    var val = acts[ni];
    var nextVal = nextActs[ni];
    if (useAbs) {
      val = Math.abs(val * 2 - 1);
      nextVal = Math.abs(nextVal * 2 - 1);
    }

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);

    var x1 = tokenX + col * pixSize + pixSize / 2;
    var y1 = layerY + row * pixSize + pixSize / 2;
    var x2 = tokenX + col * pixSize + pixSize / 2;
    var y2 = nextLayerY + roomH + row * pixSize + pixSize / 2;

    // Color based on activation change
    var delta = nextVal - val;
    var absDelta = Math.abs(delta);
    var alpha = Math.min(0.6, absDelta * 3);

    if (alpha < 0.02) continue;

    if (delta > 0) {
      c.strokeStyle = 'rgba(233,69,96,' + alpha.toFixed(2) + ')'; // expansion = red
    } else {
      c.strokeStyle = 'rgba(0,119,182,' + alpha.toFixed(2) + ')'; // contraction = blue
    }

    // Slight curve for visual appeal
    var midX = x1 + Math.sin(ni * 0.1 + layerIdx) * 3;
    var midY = (y1 + y2) / 2;

    c.beginPath();
    c.moveTo(x1, y1);
    c.quadraticCurveTo(midX, midY, x2, y2);
    c.stroke();
  }
}

// ---- 3D Pseudo-Perspective Fibre View ----
function drawFibreBundle3D(c, W, H, data, layersSource, nTokens, nLayers, hiddenDim,
  gridCols, gridRows, pixSize, roomW, roomH, tokenGap, layerGap, useAbs, currentLayer, offsetX, offsetY) {

  var dx = +document.getElementById('sl-dx').value;
  var dy = +document.getElementById('sl-dy').value;
  var dz = +document.getElementById('sl-dz').value;

  // In 3D mode:
  // X axis = token index
  // Y axis = layer depth (fibre)
  // Z axis = a chosen hidden dimension's activation value

  var totalTokenW = nTokens * (roomW + tokenGap) - tokenGap;
  var focalLen = 500;

  function rot3D(x, y, z) {
    var cosY = Math.cos(fibreState.rotY), sinY = Math.sin(fibreState.rotY);
    var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
    var cosX = Math.cos(fibreState.rotX), sinX = Math.sin(fibreState.rotX);
    var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
    return [x1, y1, z2];
  }

  function proj(x, y, z) {
    var r = rot3D(x, y, z);
    var scale = focalLen / (focalLen + r[2]);
    return [W / (2 * zoomLevel) + r[0] * scale, H / (2 * zoomLevel) + r[1] * scale, r[2], scale];
  }

  // Scale factors
  var xScale = 200;  // spread tokens
  var yScale = 40;   // spread layers
  var zScale = 80;   // depth from activation

  // Draw from back to front (painter's algorithm by layer)
  for (var li = 0; li < nLayers; li++) {
    var layerDepth = (li - nLayers / 2) * yScale;
    var isCurrentLayer = (li === currentLayer + 1) || (li === 0 && currentLayer === 0);

    for (var ti = 0; ti < nTokens; ti++) {
      var tokenOffset = (ti - nTokens / 2) * (roomW + tokenGap) * 0.8;
      var acts = layersSource[li].activations[ti];

      // For each neuron, compute a 3D position
      // X = token position + neuron column offset
      // Y = layer position + neuron row offset
      // Z = activation value of a chosen dimension (dz)

      // Draw the room as a small grid of projected pixels
      for (var ni = 0; ni < hiddenDim; ni++) {
        var val = acts[ni];
        if (useAbs) val = Math.abs(val * 2 - 1);

        var col = ni % gridCols;
        var row = Math.floor(ni / gridCols);

        var wx = tokenOffset + (col - gridCols / 2) * pixSize * 0.5;
        var wy = layerDepth + (row - gridRows / 2) * pixSize * 0.3;
        var wz = (val - 0.5) * zScale;

        var p = proj(wx, wy, wz);
        var sz = Math.max(0.5, pixSize * 0.4 * p[3]);

        var rgb = neuronColor(val, fibreState.colormap);
        var depthAlpha = Math.max(0.1, Math.min(0.9, 0.7 - p[2] * 0.001));
        c.fillStyle = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + depthAlpha.toFixed(2) + ')';
        c.fillRect(p[0] - sz / 2, p[1] - sz / 2, sz, sz);
      }

      // Token label
      if (li === 0) {
        var lp = proj(tokenOffset, layerDepth + gridRows * pixSize * 0.2, 0);
        c.font = Math.max(7, Math.round(9 * lp[3])) + 'px monospace';
        c.fillStyle = '#e94560';
        c.textAlign = 'center';
        c.fillText(data.tokens[ti], lp[0], lp[1] + 12);
      }
    }

    // Layer label
    var llp = proj(-nTokens / 2 * (roomW + tokenGap) * 0.8 - 40, layerDepth, 0);
    c.font = '8px monospace';
    c.fillStyle = isCurrentLayer ? '#e94560' : '#555';
    c.textAlign = 'right';
    var layerLabel = li === 0 ? 'Emb' : 'L' + (li - 1);
    c.fillText(layerLabel, llp[0], llp[1] + 3);
  }

  // Draw 3D axes
  var axLen = 80;
  var axes = [
    { v: [1, 0, 0], label: 'Token →', color: '#e94560' },
    { v: [0, 1, 0], label: 'Layer ↑', color: '#53a8b6' },
    { v: [0, 0, 1], label: 'Dim ' + dz, color: '#f5a623' }
  ];
  var o3 = proj(0, 0, 0);
  for (var ai = 0; ai < 3; ai++) {
    var ax = axes[ai];
    var e3 = proj(ax.v[0] * axLen, ax.v[1] * axLen, ax.v[2] * axLen);
    c.strokeStyle = ax.color;
    c.globalAlpha = 0.5;
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(o3[0], o3[1]);
    c.lineTo(e3[0], e3[1]);
    c.stroke();
    c.globalAlpha = 1;
    c.font = '9px monospace';
    c.fillStyle = ax.color;
    c.textAlign = 'left';
    c.fillText(ax.label, e3[0] + 4, e3[1] - 4);
  }
}

// ---- Mouse interaction for fibre view ----
cv3d.addEventListener('mousedown', function(e) {
    if (viewMode !== 'fibre' && viewMode !== 'fibre3d') return;
  if (e.button === 0 && !e.shiftKey) {
    // Normal drag = rotate all rooms
    fibreState.dragActive = true;
    fibreState.dragLastX = e.clientX;
    fibreState.dragLastY = e.clientY;
    e.preventDefault();
    return;
  }
  if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
    // Shift+drag or middle = pan
    e.preventDefault();
    panActive = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
  }
});

window.addEventListener('mousemove', function(e) {
  if (viewMode !== 'fibre' && viewMode !== 'fibre3d') return;
  if (fibreState.dragActive) {
    var ddx = e.clientX - fibreState.dragLastX;
    var ddy = e.clientY - fibreState.dragLastY;
    fibreState.rotY += ddx * 0.005;
    fibreState.rotX += ddy * 0.005;
    fibreState.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, fibreState.rotX));
    fibreState.dragLastX = e.clientX;
    fibreState.dragLastY = e.clientY;
    draw();
    return;
  }
});

function drawFibreBundle3DGrid() {
    var cv = document.getElementById('cv');
    var c = cv.getContext('2d');
    var W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);

    if (!D) {
        c.font = '14px monospace';
        c.fillStyle = '#555';
        c.fillText('Run a prompt first', W / 2 - 80, H / 2);
        return;
    }

    var nTokens = D.n_real;
    var nLayers = D.n_layers;
    var hiddenDim = D.hidden_dim;
    var nP = D.n_points;

    var dx = +document.getElementById('sl-dx').value;
    var dy = +document.getElementById('sl-dy').value;
    var dz = +document.getElementById('sl-dz').value;
    var amp = +document.getElementById('sl-amp').value;
    var t = +document.getElementById('sl-t').value;
    var sig = +document.getElementById('sl-sig').value;
    var currentLayer = +document.getElementById('sl-layer').value;
    var showGrid = document.getElementById('cb-grid').checked;
    var showHeat = document.getElementById('cb-heat').checked;
    var showSC = document.getElementById('cb-sc').checked;
    var mode = document.getElementById('sel-mode').value;

    var activeDeltas = getActiveDeltas();
    if (!activeDeltas) activeDeltas = D.deltas;
    var isEmb = (mode === 'embedding');

    dx = Math.min(dx, hiddenDim - 1);
    dy = Math.min(dy, hiddenDim - 1);
    dz = Math.min(dz, hiddenDim - 1);

    // Extract base positions in all 3 dims
    var fx = new Float64Array(nP), fy = new Float64Array(nP), fz = new Float64Array(nP);
    for (var i = 0; i < nP; i++) {
        fx[i] = D.fixed_pos[i][dx];
        fy[i] = D.fixed_pos[i][dy];
        fz[i] = D.fixed_pos[i][dz];
    }

    // Compute view bounds for all 3 axes
    var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, mnz = Infinity, mxz = -Infinity;
    for (var i = 0; i < nP; i++) {
        if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
        if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
        if (fz[i] < mnz) mnz = fz[i]; if (fz[i] > mxz) mxz = fz[i];
    }
    var mrx = mxx - mnx || 1, mry = mxy - mny || 1, mrz = mxz - mnz || 1;
    var mr = Math.max(mrx, mry, mrz);
    var cx3 = (mnx + mxx) / 2, cy3 = (mny + mxy) / 2, cz3 = (mnz + mxz) / 2;
    var pd = 0.12;
    var vx0 = cx3 - mr * (0.5 + pd), vx1 = cx3 + mr * (0.5 + pd);
    var vy0 = cy3 - mr * (0.5 + pd), vy1 = cy3 + mr * (0.5 + pd);
    var vz0 = cz3 - mr * (0.5 + pd), vz1 = cz3 + mr * (0.5 + pd);

    // Precompute per-layer raw deltas
    var edxAll = [], edyAll = [], edzAll = [];
    for (var lay = 0; lay < nLayers; lay++) {
        var edxL = new Float64Array(nP), edyL = new Float64Array(nP), edzL = new Float64Array(nP);
        for (var j = 0; j < nP; j++) {
            edxL[j] = activeDeltas[lay][j][dx] * amp;
            edyL[j] = activeDeltas[lay][j][dy] * amp;
            edzL[j] = activeDeltas[lay][j][dz] * amp;
        }
        edxAll.push(edxL); edyAll.push(edyL); edzAll.push(edzL);
    }

    var s2i = 1 / (2 * sig * sig);

    // 3D grid resolution — smaller than draw3D since we have many rooms
    var N3 = Math.max(3, Math.min(8, Math.floor(+document.getElementById('sl-gr').value / 6)));

    // Mode-aware cumulative deltas
    function computeLayerDeltas(li) {
        var edxCum = new Float64Array(nP), edyCum = new Float64Array(nP), edzCum = new Float64Array(nP);
        if (isEmb) return { edx: edxCum, edy: edyCum, edz: edzCum };
        if (mode === 'single') {
            for (var j = 0; j < nP; j++) { edxCum[j] = edxAll[li][j]; edyCum[j] = edyAll[li][j]; edzCum[j] = edzAll[li][j]; }
        } else if (mode === 'cumfwd') {
            for (var cl = 0; cl <= li; cl++) for (var j = 0; j < nP; j++) { edxCum[j] += edxAll[cl][j]; edyCum[j] += edyAll[cl][j]; edzCum[j] += edzAll[cl][j]; }
        } else {
            for (var cl = li; cl < nLayers; cl++) for (var j = 0; j < nP; j++) { edxCum[j] += edxAll[cl][j]; edyCum[j] += edyAll[cl][j]; edzCum[j] += edzAll[cl][j]; }
        }
        return { edx: edxCum, edy: edyCum, edz: edzCum };
    }

    // *** TRUE 3D VOLUMETRIC GRID BUILDER — same approach as draw3D ***
    function buildDeformed3DGrid(edxCum, edyCum, edzCum) {
        var nV = (N3 + 1) * (N3 + 1) * (N3 + 1);
        function gIdx(ix, iy, iz) { return iz * (N3 + 1) * (N3 + 1) + iy * (N3 + 1) + ix; }

        var oX = new Float64Array(nV), oY = new Float64Array(nV), oZ = new Float64Array(nV);
        var gX = new Float64Array(nV), gY = new Float64Array(nV), gZ = new Float64Array(nV);

        for (var iz = 0; iz <= N3; iz++) for (var iy = 0; iy <= N3; iy++) for (var ix = 0; ix <= N3; ix++) {
            var gi = gIdx(ix, iy, iz);
            oX[gi] = vx0 + (ix / N3) * (vx1 - vx0);
            oY[gi] = vy0 + (iy / N3) * (vy1 - vy0);
            oZ[gi] = vz0 + (iz / N3) * (vz1 - vz0);
        }

        if (isEmb) {
            for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; gZ[gi] = oZ[gi]; }
        } else {
            // Full 3D RBF interpolation — same as draw3D
            for (var gi = 0; gi < nV; gi++) {
                var gpx = oX[gi], gpy = oY[gi], gpz = oZ[gi];
                var vvx = 0, vvy = 0, vvz = 0, ws = 0;
                for (var k = 0; k < nP; k++) {
                    var eex = gpx - fx[k], eey = gpy - fy[k], eez = gpz - fz[k];
                    var w = Math.exp(Math.max(-500, -(eex * eex + eey * eey + eez * eez) * s2i));
                    vvx += w * edxCum[k]; vvy += w * edyCum[k]; vvz += w * edzCum[k]; ws += w;
                }
                if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
                gX[gi] = gpx + t * vvx;
                gY[gi] = gpy + t * vvy;
                gZ[gi] = gpz + t * vvz;
            }
        }

        // Collect edges with strain — all 3 axis directions
        var edges = [];
        function addEdge(a, b) {
            var od = Math.sqrt((oX[b]-oX[a])*(oX[b]-oX[a])+(oY[b]-oY[a])*(oY[b]-oY[a])+(oZ[b]-oZ[a])*(oZ[b]-oZ[a]));
            var dd = Math.sqrt((gX[b]-gX[a])*(gX[b]-gX[a])+(gY[b]-gY[a])*(gY[b]-gY[a])+(gZ[b]-gZ[a])*(gZ[b]-gZ[a]));
            var strain = od > 1e-12 ? dd / od : 1;
            edges.push({ a: a, b: b, strain: strain });
        }
        // X-direction edges
        for (var iz = 0; iz <= N3; iz++) for (var iy = 0; iy <= N3; iy++) for (var ix = 0; ix < N3; ix++)
            addEdge(gIdx(ix, iy, iz), gIdx(ix + 1, iy, iz));
        // Y-direction edges
        for (var iz = 0; iz <= N3; iz++) for (var iy = 0; iy < N3; iy++) for (var ix = 0; ix <= N3; ix++)
            addEdge(gIdx(ix, iy, iz), gIdx(ix, iy + 1, iz));
        // Z-direction edges
        for (var iz = 0; iz < N3; iz++) for (var iy = 0; iy <= N3; iy++) for (var ix = 0; ix <= N3; ix++)
            addEdge(gIdx(ix, iy, iz), gIdx(ix, iy, iz + 1));

        // Collect surface faces for heatmap (6 faces of the cube)
        var faces = [];
        function addFace(a, b, cc, d) {
            var s1 = edges.length; // approximate strain from vertex positions
            var avgS = 1.0;
            // Compute average strain from the 4 edge strains of this face
            var od1 = Math.sqrt((oX[b]-oX[a])*(oX[b]-oX[a])+(oY[b]-oY[a])*(oY[b]-oY[a])+(oZ[b]-oZ[a])*(oZ[b]-oZ[a]));
            var dd1 = Math.sqrt((gX[b]-gX[a])*(gX[b]-gX[a])+(gY[b]-gY[a])*(gY[b]-gY[a])+(gZ[b]-gZ[a])*(gZ[b]-gZ[a]));
            var od2 = Math.sqrt((oX[cc]-oX[b])*(oX[cc]-oX[b])+(oY[cc]-oY[b])*(oY[cc]-oY[b])+(oZ[cc]-oZ[b])*(oZ[cc]-oZ[b]));
            var dd2 = Math.sqrt((gX[cc]-gX[b])*(gX[cc]-gX[b])+(gY[cc]-gY[b])*(gY[cc]-gY[b])+(gZ[cc]-gZ[b])*(gZ[cc]-gZ[b]));
            avgS = ((od1 > 1e-12 ? dd1/od1 : 1) + (od2 > 1e-12 ? dd2/od2 : 1)) / 2;
            faces.push({ verts: [a, b, cc, d], strain: avgS });
        }
        // X=0 and X=N3 faces
        for (var iz = 0; iz < N3; iz++) for (var iy = 0; iy < N3; iy++) {
            addFace(gIdx(0,iy,iz), gIdx(0,iy+1,iz), gIdx(0,iy+1,iz+1), gIdx(0,iy,iz+1));
            addFace(gIdx(N3,iy,iz), gIdx(N3,iy+1,iz), gIdx(N3,iy+1,iz+1), gIdx(N3,iy,iz+1));
        }
        // Y=0 and Y=N3 faces
        for (var iz = 0; iz < N3; iz++) for (var ix = 0; ix < N3; ix++) {
            addFace(gIdx(ix,0,iz), gIdx(ix+1,0,iz), gIdx(ix+1,0,iz+1), gIdx(ix,0,iz+1));
            addFace(gIdx(ix,N3,iz), gIdx(ix+1,N3,iz), gIdx(ix+1,N3,iz+1), gIdx(ix,N3,iz+1));
        }
        // Z=0 and Z=N3 faces
        for (var iz2 = 0; iz2 < 1; iz2++) for (var iy = 0; iy < N3; iy++) for (var ix = 0; ix < N3; ix++) {
            addFace(gIdx(ix,iy,0), gIdx(ix+1,iy,0), gIdx(ix+1,iy+1,0), gIdx(ix,iy+1,0));
            addFace(gIdx(ix,iy,N3), gIdx(ix+1,iy,N3), gIdx(ix+1,iy+1,N3), gIdx(ix,iy+1,N3));
        }

        return { oX: oX, oY: oY, oZ: oZ, gX: gX, gY: gY, gZ: gZ, edges: edges, faces: faces, nV: nV, gIdx: gIdx };
    }

    // ---- 3D projection using fibreState rotation ----
    var focalLen = 600;

    // Room layout
    var gapFracX = 0.35;
    var gapFracY = 0.45;
    var maxRoomSize = 120;
    var roomSize = Math.min(maxRoomSize, Math.max(30, Math.floor(300 / Math.max(nTokens, nLayers))));
    var gapX = Math.max(8, Math.floor(roomSize * gapFracX));
    var gapY = Math.max(12, Math.floor(roomSize * gapFracY));

    // Scale factor: map world-space range to room pixel size
    var sc3 = roomSize * 0.4 / mr;

    function rot3Df(x, y, z) {
        var cosY = Math.cos(fibreState.rotY), sinY = Math.sin(fibreState.rotY);
        var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
        var cosX = Math.cos(fibreState.rotX), sinX = Math.sin(fibreState.rotX);
        var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
        return [x1, y1, z2];
    }

    function proj3Df(x, y, z) {
        var r = rot3Df(x * zoomLevel, y * zoomLevel, z * zoomLevel);
        var scale = focalLen / (focalLen + r[2]);
        return [W / 2 + panX + r[0] * scale, H / 2 + panY + r[1] * scale, r[2], scale];
    }

    // Collect all drawable elements for depth sorting
    var allEdges = [];
    var allQuads = [];
    var allPoints = [];

    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for (var li = 0; li < nLayers; li++) {
        var rowIdx = nLayers - 1 - li;
        var isCurrentLayer = (li === currentLayer);

        var layerDeltas = computeLayerDeltas(li);
        var edxCum = layerDeltas.edx, edyCum = layerDeltas.edy, edzCum = layerDeltas.edz;

        var grid = buildDeformed3DGrid(edxCum, edyCum, edzCum);

        for (var ti = 0; ti < nTokens; ti++) {
            var roomCenterX = (ti - (nTokens - 1) / 2) * (roomSize + gapX);
            var roomCenterY = (rowIdx - (nLayers - 1) / 2) * (roomSize + gapY);

            // Project a grid vertex into room-local 3D space, then into screen space
            function vertToScreen(gi) {
                var sx = roomCenterX + (grid.gX[gi] - cx3) * sc3;
                var sy = roomCenterY + (grid.gY[gi] - cy3) * sc3;
                var sz = (grid.gZ[gi] - cz3) * sc3;
                return proj3Df(sx, sy, sz);
            }

            function origVertToScreen(gi) {
                var sx = roomCenterX + (grid.oX[gi] - cx3) * sc3;
                var sy = roomCenterY + (grid.oY[gi] - cy3) * sc3;
                var sz = (grid.oZ[gi] - cz3) * sc3;
                return proj3Df(sx, sy, sz);
            }

            // ---- 3D Grid edges ----
            if (showGrid && !isEmb) {
                for (var ei = 0; ei < grid.edges.length; ei++) {
                    var e = grid.edges[ei];
                    var pa = vertToScreen(e.a);
                    var pb = vertToScreen(e.b);
                    allEdges.push({
                        x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1],
                        z: (pa[2] + pb[2]) / 2, strain: e.strain,
                        isCurrentLayer: isCurrentLayer, type: 'grid'
                    });
                }
            }

            // ---- Strain heatmap faces ----
            if (showHeat && !isEmb) {
                for (var fi = 0; fi < grid.faces.length; fi++) {
                    var f = grid.faces[fi];
                    var co = s2c(f.strain);
                    var pts3d = [];
                    var avgZ3d = 0;
                    for (var ci = 0; ci < 4; ci++) {
                        var p = vertToScreen(f.verts[ci]);
                        pts3d.push(p);
                        avgZ3d += p[2];
                    }
                    avgZ3d /= 4;
                    allQuads.push({
                        pts: pts3d, z: avgZ3d,
                        color: co,
                        alpha: isCurrentLayer ? 0.25 : 0.1,
                        isCurrentLayer: isCurrentLayer
                    });
                }
            }

            // ---- Room border (wireframe box) ----
            var borderAlpha = isCurrentLayer ? 0.6 : 0.2;
            var halfR = roomSize * 0.5;
            var bCorners3D = [
                [-halfR, -halfR, -halfR], [halfR, -halfR, -halfR],
                [halfR, halfR, -halfR], [-halfR, halfR, -halfR],
                [-halfR, -halfR, halfR], [halfR, -halfR, halfR],
                [halfR, halfR, halfR], [-halfR, halfR, halfR]
            ];
            var bEdges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
            for (var bi = 0; bi < bEdges.length; bi++) {
                var be = bEdges[bi];
                var c0 = bCorners3D[be[0]], c1 = bCorners3D[be[1]];
                var pa = proj3Df(roomCenterX + c0[0] * 0.5, roomCenterY + c0[1] * 0.5, c0[2] * 0.5);
                var pb = proj3Df(roomCenterX + c1[0] * 0.5, roomCenterY + c1[1] * 0.5, c1[2] * 0.5);
                allEdges.push({
                    x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1],
                    z: (pa[2] + pb[2]) / 2, strain: 1.0,
                    isCurrentLayer: isCurrentLayer, type: 'border',
                    borderAlpha: borderAlpha
                });
            }

            // ---- Token dot ----
            var tokWX = fx[ti] + (isEmb ? 0 : t * edxCum[ti]);
            var tokWY = fy[ti] + (isEmb ? 0 : t * edyCum[ti]);
            var tokWZ = fz[ti] + (isEmb ? 0 : t * edzCum[ti]);

            var tokSX = roomCenterX + (tokWX - cx3) * sc3;
            var tokSY = roomCenterY + (tokWY - cy3) * sc3;
            var tokSZ = (tokWZ - cz3) * sc3;

            var tp = proj3Df(tokSX, tokSY, tokSZ);

            allPoints.push({
                x: tp[0], y: tp[1], z: tp[2], scale: tp[3],
                tokenIdx: ti, layerIdx: li,
                isCurrentLayer: isCurrentLayer,
                color: tc[ti % tc.length]
            });

            // ---- Token label at bottom layer ----
            if (li === 0) {
                var labelP = proj3Df(roomCenterX, roomCenterY + roomSize * 0.6, 0);
                allPoints.push({
                    x: labelP[0], y: labelP[1], z: labelP[2], scale: labelP[3],
                    tokenIdx: ti, layerIdx: -1,
                    isCurrentLayer: false, isTokenLabel: true,
                    color: tc[ti % tc.length],
                    labelText: '[' + ti + '] ' + D.tokens[ti]
                });
            }
        }

        // ---- Layer label ----
        var layerLabelX = -(nTokens / 2) * (roomSize + gapX) - 30;
        var layerLabelY = (rowIdx - (nLayers - 1) / 2) * (roomSize + gapY);
        var llp = proj3Df(layerLabelX, layerLabelY, 0);
        allPoints.push({
            x: llp[0], y: llp[1], z: llp[2], scale: llp[3],
            tokenIdx: -1, layerIdx: li,
            isCurrentLayer: isCurrentLayer, isLabel: true,
            labelText: 'L' + li
        });

        // ---- Inter-layer connections ----
        if (li > 0 && fibreState.showConnections) {
            var prevRowIdx = nLayers - li;
            for (var ti = 0; ti < nTokens; ti++) {
                var prevRoomCY = (prevRowIdx - (nLayers - 1) / 2) * (roomSize + gapY);
                var currRoomCY = (rowIdx - (nLayers - 1) / 2) * (roomSize + gapY);
                var roomCX = (ti - (nTokens - 1) / 2) * (roomSize + gapX);

                var py3 = prevRoomCY + roomSize * 0.5;
                var cy3b = currRoomCY - roomSize * 0.5;

                var pp = proj3Df(roomCX, py3, 0);
                var cp = proj3Df(roomCX, cy3b, 0);

                allEdges.push({
                    x1: pp[0], y1: pp[1], x2: cp[0], y2: cp[1],
                    z: (pp[2] + cp[2]) / 2, strain: 1.0,
                    isCurrentLayer: isCurrentLayer, type: 'pathline',
                    color: tc[ti % tc.length]
                });
            }
        }
    }

    // ---- Depth sort everything ----
    allQuads.sort(function(a, b) { return b.z - a.z; });
    allEdges.sort(function(a, b) { return b.z - a.z; });
    allPoints.sort(function(a, b) { return b.z - a.z; });

    // ---- Draw quads (heatmap) ----
    for (var qi = 0; qi < allQuads.length; qi++) {
        var q = allQuads[qi];
        c.beginPath();
        c.moveTo(q.pts[0][0], q.pts[0][1]);
        for (var ci = 1; ci < 4; ci++) c.lineTo(q.pts[ci][0], q.pts[ci][1]);
        c.closePath();
        c.fillStyle = 'rgba(' + q.color[0] + ',' + q.color[1] + ',' + q.color[2] + ',' + q.alpha.toFixed(2) + ')';
        c.fill();
    }

    // ---- Draw edges ----
    for (var ei = 0; ei < allEdges.length; ei++) {
        var e = allEdges[ei];
        var depthAlpha = Math.max(0.15, Math.min(0.95, 0.7 - e.z * 0.0002));

        if (e.type === 'pathline') {
            var r = parseInt(e.color.slice(1, 3), 16);
            var g = parseInt(e.color.slice(3, 5), 16);
            var b = parseInt(e.color.slice(5, 7), 16);
            c.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (depthAlpha * 0.6).toFixed(2) + ')';
            c.lineWidth = e.isCurrentLayer ? 2 : 1.0;
        } else if (e.type === 'border') {
            c.strokeStyle = e.isCurrentLayer ?
                'rgba(233,69,96,' + Math.min(1.0, e.borderAlpha * depthAlpha * 2.5).toFixed(2) + ')' :
                'rgba(80,80,130,' + Math.min(1.0, e.borderAlpha * depthAlpha * 1.5).toFixed(2) + ')';
            c.lineWidth = e.isCurrentLayer ? 1.5 : 0.7;
        } else if (e.type === 'grid') {
            var gridAlpha = e.isCurrentLayer ? depthAlpha * 0.9 : depthAlpha * 0.45;
            if (showSC) {
                var sc = s2c(e.strain);
                c.strokeStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + gridAlpha.toFixed(2) + ')';
            } else {
                c.strokeStyle = 'rgba(200,200,200,' + gridAlpha.toFixed(2) + ')';
            }
            c.lineWidth = e.isCurrentLayer ? 1.0 : 0.5;
        } else {
            continue;
        }

        c.beginPath();
        c.moveTo(e.x1, e.y1);
        c.lineTo(e.x2, e.y2);
        c.stroke();
    }

    // ---- Draw points and labels ----
    for (var pi = 0; pi < allPoints.length; pi++) {
        var pt = allPoints[pi];
        var depthAlpha = Math.max(0.4, Math.min(1.0, 0.9 - pt.z * 0.0002));

        if (pt.isLabel) {
            c.font = (pt.isCurrentLayer ? 'bold ' : '') + Math.max(8, Math.round(9 * pt.scale)) + 'px monospace';
            c.fillStyle = pt.isCurrentLayer ? '#e94560' : '#666';
            c.textAlign = 'right';
            c.globalAlpha = depthAlpha;
            c.fillText(pt.labelText, pt.x - 5, pt.y + 3);
            c.globalAlpha = 1.0;
            continue;
        }

        if (pt.isTokenLabel) {
            var fontSize = Math.max(7, Math.round(8 * pt.scale));
            c.font = 'bold ' + fontSize + 'px monospace';
            c.fillStyle = pt.color;
            c.textAlign = 'center';
            c.globalAlpha = depthAlpha;
            c.fillText(pt.labelText, pt.x, pt.y);
            c.globalAlpha = 1.0;
            continue;
        }

        var dotR = Math.max(2, (pt.isCurrentLayer ? 5 : 3) * pt.scale);

        if (pt.isCurrentLayer) {
            var grad = c.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, dotR * 2.5);
            grad.addColorStop(0, 'rgba(255,255,255,0.12)');
            grad.addColorStop(1, 'rgba(255,255,255,0)');
            c.beginPath();
            c.arc(pt.x, pt.y, dotR * 2.5, 0, Math.PI * 2);
            c.fillStyle = grad;
            c.fill();
        }

        c.beginPath();
        c.arc(pt.x, pt.y, dotR, 0, Math.PI * 2);
        c.fillStyle = pt.color;
        c.globalAlpha = depthAlpha;
        c.fill();
        c.strokeStyle = pt.isCurrentLayer ? '#fff' : 'rgba(255,255,255,0.5)';
        c.lineWidth = pt.isCurrentLayer ? 1.5 : 0.7;
        c.stroke();
        c.globalAlpha = 1.0;
    }

    // 3D axes
    var axLen = Math.max(roomSize, mr * 0.3) * zoomLevel;
    var axes = [
        { v: [1, 0, 0], label: 'Dim ' + dx, color: '#e94560' },
        { v: [0, 1, 0], label: 'Layer \u2191', color: '#53a8b6' },
        { v: [0, 0, 1], label: 'Dim ' + dz, color: '#f5a623' }
    ];
    c.lineWidth = 1.5;
    var o3 = proj3Df(0, 0, 0);
    for (var ai = 0; ai < 3; ai++) {
        var ax = axes[ai];
        var e3 = proj3Df(ax.v[0] * axLen, ax.v[1] * axLen, ax.v[2] * axLen);
        c.strokeStyle = ax.color;
        c.globalAlpha = 0.5;
        c.beginPath();
        c.moveTo(o3[0], o3[1]);
        c.lineTo(e3[0], e3[1]);
        c.stroke();
        c.globalAlpha = 1;
        c.font = '10px monospace';
        c.fillStyle = ax.color;
        c.textAlign = 'left';
        c.fillText(ax.label, e3[0] + 4, e3[1] - 4);
    }

    // HUD
    var decompLabel = getDecompLabel();
    c.font = '11px monospace';
    c.fillStyle = 'rgba(255,255,255,0.45)';
    c.textAlign = 'left';
    if (isEmb) {
        c.fillText('FIBRE BUNDLE 3D  EMBEDDING  Dims:' + dx + ',' + dy + ',' + dz +
                   '  Layers:' + nLayers + '  Tokens:' + nTokens + '  Drag to rotate', 12, 16);
    } else {
        c.fillText('FIBRE BUNDLE 3D  Layer ' + currentLayer + '/' + (nLayers - 1) +
                   '  t=' + t.toFixed(2) + '  amp=' + amp.toFixed(1) +
                   '  Dims:' + dx + ',' + dy + ',' + dz +
                   '  Mode:' + mode + '  Decomp:' + decompLabel +
                   '  Drag to rotate', 12, 16);
    }
    c.font = '9px monospace';
    c.fillStyle = 'rgba(255,255,255,0.3)';
    c.fillText('Zoom: ' + zoomLevel.toFixed(2) + 'x  (Scroll=zoom, Shift+drag=pan, 0=reset)', 12, H - 8);
}

window.addEventListener('mouseup', function(e) {
  if (viewMode === 'fibre' || viewMode === 'fibre3d') {
    fibreState.dragActive = false;
  }
});

// Colormap cycling with 'M' key in fibre mode
var _origOnKeyFibre = onKeyFibre;
onKeyFibre = function(e) {
  if (e.key === 'm' || e.key === 'M') {
    var maps = ['grayscale', 'coolhot', 'viridis'];
    var idx = maps.indexOf(fibreState.colormap);
    fibreState.colormap = maps[(idx + 1) % maps.length];
    document.getElementById('status').textContent = 'Colormap: ' + fibreState.colormap;
    draw();
    return;
  }
  if (e.key === 'd' || e.key === 'D') {
    // Adjust connection density
    fibreState.connectionDensity = Math.min(1.0, fibreState.connectionDensity + 0.05);
    if (fibreState.connectionDensity > 1.0) fibreState.connectionDensity = 0.02;
    document.getElementById('status').textContent =
      'Connection density: ' + (fibreState.connectionDensity * 100).toFixed(0) + '%';
    draw();
    return;
  }
  _origOnKeyFibre(e);
};

// ---- Fibre view button in sidebar ----
// Add the button dynamically if it doesn't exist
(function() {
  var toggleDiv = document.querySelector('.view-toggle');
  if (toggleDiv && !document.getElementById('btn-fibre')) {
    var btn = document.createElement('button');
    btn.id = 'btn-fibre';
    btn.textContent = 'Fibre Bundle';
    btn.onclick = function() { setViewMode('fibre'); };
    toggleDiv.appendChild(btn);
  }
})();

// ============================================================
// COMPARE MODE — Differential Activation Maps
// ============================================================

var compareData = null;

function toggleCompareMode() {
  var on = document.getElementById('cb-compare').checked;
  document.getElementById('compare-area').style.display = on ? 'block' : 'none';
  if (!on) {
    document.getElementById('compare-panel').style.display = 'none';
    document.getElementById('compare-summary').style.display = 'none';
    document.getElementById('compare-divergence-chart').innerHTML = '';
    compareData = null;
  }
}

function runCompare() {
  var textA = document.getElementById('txt-in').value.trim();
  var textB = document.getElementById('txt-b').value.trim();
  if (!textA || !textB) { alert('Enter both texts'); return; }

  var btn = document.getElementById('btn-compare');
  btn.disabled = true; btn.textContent = 'Comparing...';
  document.getElementById('status').textContent = 'Comparing activations...';

  fetch('/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text_a: textA, text_b: textB })
  })
  .then(function(r) { if (!r.ok) throw new Error('Server error ' + r.status); return r.json(); })
  .then(function(data) {
    if (data.error) { alert(data.error); return; }
    compareData = data;
    renderCompareSummary();
    renderDivergenceChart();
    renderCompareGrids();
    btn.disabled = false; btn.textContent = 'Compare';
    document.getElementById('status').textContent =
      'Compare: ' + data.n_common + ' aligned tokens, ' +
      data.n_layers + ' layers, onset layer: ' +
      (data.onset_layer >= 0 ? data.onset_layer : 'none');
  })
  .catch(function(e) {
    alert('Error: ' + e);
    btn.disabled = false; btn.textContent = 'Compare';
  });
}

function renderCompareSummary() {
  var d = compareData;
  var panel = document.getElementById('compare-summary');
  panel.style.display = 'block';

  var html = '';
  html += '<div style="color:#e94560;font-weight:bold;margin-bottom:4px">Differential Activation Analysis</div>';
  html += '<div><b style="color:#53a8b6">Text A:</b> <span style="color:#a0a0c0">' +
          d.tokens_a.join(' ') + '</span></div>';
  html += '<div><b style="color:#f5a623">Text B:</b> <span style="color:#a0a0c0">' +
          d.tokens_b.join(' ') + '</span></div>';
  html += '<div style="margin-top:4px">';
  html += 'Aligned tokens: <span style="color:#e94560">' + d.n_common + '</span> | ';
  html += 'Layers: <span style="color:#e94560">' + d.n_layers + '</span> | ';
  html += 'Hidden dim: <span style="color:#e94560">' + d.hidden_dim + '</span> | ';
  html += 'Max diff: <span style="color:#e94560">' + d.global_diff_max.toFixed(4) + '</span>';
  html += '</div>';

  if (d.onset_layer >= 0) {
    html += '<div style="margin-top:4px;color:#f5a623;font-weight:bold">';
    html += '⚡ Divergence onset at layer ' + d.onset_layer +
            ' — this is where the model starts processing the inputs differently!';
    html += '</div>';
  } else {
    html += '<div style="margin-top:4px;color:#888">';
    html += 'No clear divergence onset detected (inputs may be very similar or very different from the start).';
    html += '</div>';
  }

  // Top diverging dims at the most divergent layer
  var maxDivLayer = 0;
  var maxDiv = 0;
  for (var i = 0; i < d.layer_divergence.length; i++) {
    if (d.layer_divergence[i] > maxDiv) {
      maxDiv = d.layer_divergence[i];
      maxDivLayer = i;
    }
  }
  if (d.top_dims_per_layer[maxDivLayer] && d.top_dims_per_layer[maxDivLayer].length > 0) {
    html += '<div style="margin-top:4px;font-size:9px">';
    html += '<b style="color:#53a8b6">Most divergent layer: ' +
            (maxDivLayer === 0 ? 'Embedding' : 'L' + (maxDivLayer - 1)) + '</b> — Top dims: ';
    var topDims = d.top_dims_per_layer[maxDivLayer].slice(0, 8);
    for (var di = 0; di < topDims.length; di++) {
      html += '<span style="color:#e94560">d' + topDims[di].dim + '</span>';
      html += '<span style="color:#666">(' + topDims[di].mean_abs_diff.toFixed(4) + ')</span> ';
    }
    html += '</div>';
  }

  panel.innerHTML = html;
}

function renderDivergenceChart() {
  var d = compareData;
  var container = document.getElementById('compare-divergence-chart');
  var chartW = 340, chartH = 80;

  var html = '<div style="color:#888;font-size:9px;margin-bottom:2px">' +
             'Layer-by-layer divergence (mean |A−B|):</div>';
  html += '<canvas id="div-chart-cv" width="' + chartW + '" height="' + chartH + '"></canvas>';

  // Per-token divergence sparklines
  if (d.n_common > 0 && d.token_divergence.length > 0) {
    html += '<div style="margin-top:6px;font-size:9px;color:#888">Per-token divergence across layers:</div>';
    for (var ti = 0; ti < d.n_common; ti++) {
      var tokLabel = d.tokens_a[ti];
      var tokLabelB = d.tokens_b[ti];
      var same = (tokLabel === tokLabelB);
      html += '<div style="display:flex;align-items:center;gap:4px;margin:1px 0">';
      html += '<span style="color:' + (same ? '#53a8b6' : '#e94560') +
              ';min-width:80px;font-family:monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="A: ' +
              tokLabel + ' | B: ' + tokLabelB + '">';
      html += '[' + ti + '] ' + tokLabel;
      if (!same) html += ' / ' + tokLabelB;
      html += '</span>';
      html += '<canvas id="tok-div-' + ti + '" width="200" height="16" style="border:1px solid #0f3460;border-radius:2px"></canvas>';
      html += '</div>';
    }
  }

  container.innerHTML = html;

  // Draw the main divergence chart
  var cv = document.getElementById('div-chart-cv');
  if (!cv) return;
  var c = cv.getContext('2d');
  var nL = d.layer_divergence.length;
  var maxDiv = 0;
  for (var i = 0; i < nL; i++) {
    if (d.layer_divergence[i] > maxDiv) maxDiv = d.layer_divergence[i];
  }
  if (maxDiv < 1e-12) maxDiv = 1;

  var barW = Math.max(2, Math.floor((chartW - 20) / nL) - 1);
  var barGap = 1;
  var baseY = chartH - 15;
  var maxBarH = baseY - 5;

  c.fillStyle = '#0a0a1a';
  c.fillRect(0, 0, chartW, chartH);

  for (var i = 0; i < nL; i++) {
    var val = d.layer_divergence[i];
    var h = (val / maxDiv) * maxBarH;
    var x = 10 + i * (barW + barGap);

    // Color: low divergence = blue, high = red
    var frac = val / maxDiv;
    var r = Math.floor(frac * 233);
    var g = Math.floor((1 - frac) * 100 + frac * 69);
    var b = Math.floor((1 - frac) * 182 + frac * 96);

    // Highlight onset layer
    if (d.onset_layer >= 0 && i === d.onset_layer + 1) {
      c.fillStyle = 'rgba(245,166,35,0.3)';
      c.fillRect(x - 1, 0, barW + 2, chartH);
    }

    c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    c.fillRect(x, baseY - h, barW, h);

    // Layer label
    if (nL <= 30 || i % 2 === 0) {
      c.font = '7px monospace';
      c.fillStyle = '#666';
      c.textAlign = 'center';
      c.fillText(i === 0 ? 'E' : '' + (i - 1), x + barW / 2, chartH - 2);
    }
  }

  // Draw per-token sparklines
  for (var ti = 0; ti < d.n_common; ti++) {
    var sparkCv = document.getElementById('tok-div-' + ti);
    if (!sparkCv) continue;
    var sc = sparkCv.getContext('2d');
    var sw = sparkCv.width, sh = sparkCv.height;
    sc.fillStyle = '#0a0a1a';
    sc.fillRect(0, 0, sw, sh);

    var vals = [];
    for (var li = 0; li < nL; li++) {
      vals.push(d.token_divergence[li][ti] || 0);
    }
    var sparkMax = 0;
    for (var li = 0; li < vals.length; li++) {
      if (vals[li] > sparkMax) sparkMax = vals[li];
    }
    if (sparkMax < 1e-12) sparkMax = 1;

    var stepX = sw / Math.max(1, nL - 1);

    // Fill area
    sc.beginPath();
    sc.moveTo(0, sh);
    for (var li = 0; li < nL; li++) {
      var sx = li * stepX;
      var sy = sh - (vals[li] / sparkMax) * (sh - 2);
      sc.lineTo(sx, sy);
    }
    sc.lineTo(sw, sh);
    sc.closePath();
    sc.fillStyle = 'rgba(233,69,96,0.2)';
    sc.fill();

    // Line
    sc.beginPath();
    for (var li = 0; li < nL; li++) {
      var sx = li * stepX;
      var sy = sh - (vals[li] / sparkMax) * (sh - 2);
      if (li === 0) sc.moveTo(sx, sy);
      else sc.lineTo(sx, sy);
    }
    sc.strokeStyle = '#e94560';
    sc.lineWidth = 1;
    sc.stroke();

    // Onset marker
    if (d.onset_layer >= 0) {
      var ox = (d.onset_layer + 1) * stepX;
      sc.strokeStyle = 'rgba(245,166,35,0.6)';
      sc.lineWidth = 1;
      sc.beginPath();
      sc.moveTo(ox, 0);
      sc.lineTo(ox, sh);
      sc.stroke();
    }
  }
}

function renderCompareGrids() {
  var d = compareData;
  var panel = document.getElementById('compare-panel');
  panel.style.display = 'block';

  var hiddenDim = d.hidden_dim;
  var gridCols = Math.ceil(Math.sqrt(hiddenDim));
  var gridRows = Math.ceil(hiddenDim / gridCols);
  var pixSize = 2;

  var html = '';
  html += '<div style="color:#888;font-size:9px;margin-bottom:6px">';
  html += 'Each row = one aligned token position. Three columns: ';
  html += '<span style="color:#53a8b6">A</span> | ';
  html += '<span style="color:#f5a623">B</span> | ';
  html += '<span style="color:#e94560">Diff</span> (red=A>B, blue=B>A, black=same). ';
  html += 'Each pixel = one neuron dimension. Rows within each grid = layers (top=embedding, bottom=last).';
  html += '</div>';

  // For each aligned token, show A | B | Diff stacked vertically by layer
  for (var ti = 0; ti < d.n_common; ti++) {
    var tokA = d.tokens_a[ti];
    var tokB = d.tokens_b[ti];
    var same = (tokA === tokB);

    html += '<div style="margin-bottom:10px;border-bottom:1px solid #0f3460;padding-bottom:6px">';
    html += '<div style="font-size:10px;margin-bottom:3px">';
    html += '<span style="color:#53a8b6;font-weight:bold">[' + ti + '] A: ' + tokA + '</span>';
    if (!same) {
      html += ' <span style="color:#e94560">≠</span> ';
      html += '<span style="color:#f5a623;font-weight:bold">B: ' + tokB + '</span>';
    } else {
      html += ' <span style="color:#2ecc71">=</span> ';
      html += '<span style="color:#f5a623">B: ' + tokB + '</span>';
    }
    html += '</div>';

    // Three side-by-side columns, each containing all layers stacked vertically
    html += '<div style="display:flex;gap:8px;align-items:flex-start">';

    // Column A
    html += '<div style="text-align:center">';
    html += '<div style="color:#53a8b6;font-size:8px;font-weight:bold;margin-bottom:2px">Text A</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-a-' + ti + '-' + li;
      var cw = gridCols * pixSize;
      var ch = gridRows * pixSize;
      html += '<canvas id="' + cid + '" width="' + cw + '" height="' + ch + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="A Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column B
    html += '<div style="text-align:center">';
    html += '<div style="color:#f5a623;font-size:8px;font-weight:bold;margin-bottom:2px">Text B</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-b-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="B Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column Diff
    html += '<div style="text-align:center">';
    html += '<div style="color:#e94560;font-size:8px;font-weight:bold;margin-bottom:2px">A − B</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-d-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="Diff Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Column: Magnitude
    html += '<div style="text-align:center">';
    html += '<div style="color:#f5a623;font-size:8px;font-weight:bold;margin-bottom:2px">|Diff|</div>';
    for (var li = 0; li < d.n_layers; li++) {
      var cid = 'cmp-m-' + ti + '-' + li;
      html += '<canvas id="' + cid + '" width="' + (gridCols * pixSize) + '" height="' + (gridRows * pixSize) + '" ' +
              'style="display:block;image-rendering:pixelated;margin-bottom:1px" ' +
              'title="Magnitude Token ' + ti + ' ' + (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) + '"></canvas>';
    }
    html += '</div>';

    // Layer labels
    html += '<div style="text-align:left;padding-top:12px">';
    for (var li = 0; li < d.n_layers; li++) {
      var isOnset = (d.onset_layer >= 0 && li === d.onset_layer + 1);
      var lh = gridRows * pixSize + 1;
      html += '<div style="height:' + lh + 'px;line-height:' + lh + 'px;font-size:7px;' +
              'color:' + (isOnset ? '#f5a623' : '#555') + ';' +
              'font-weight:' + (isOnset ? 'bold' : 'normal') + '">';
      html += (li === 0 ? 'Emb' : 'L' + (li - 1));
      if (isOnset) html += ' ⚡';
      html += '</div>';
    }
    html += '</div>';

    html += '</div>'; // end flex row
    html += '</div>'; // end token block
  }

  panel.innerHTML = html;

  // Now draw on all canvases
  for (var ti = 0; ti < d.n_common; ti++) {
    for (var li = 0; li < d.n_layers; li++) {
      // Draw A
      drawCompareCanvas('cmp-a-' + ti + '-' + li,
        d.activations_a[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'grayscale');
      // Draw B
      drawCompareCanvas('cmp-b-' + ti + '-' + li,
        d.activations_b[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'grayscale');
      // Draw Diff (diverging colormap)
      drawCompareCanvas('cmp-d-' + ti + '-' + li,
        d.diff[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'diverging');
      // Draw Magnitude
      drawCompareCanvas('cmp-m-' + ti + '-' + li,
        d.diff_magnitude[li][ti], hiddenDim, gridCols, gridRows, pixSize, 'hot');
    }
  }
}

/**
 * Draw a single compare-mode neuron grid canvas.
 * @param {string} canvasId
 * @param {number[]} acts - array of hiddenDim floats in [0,1]
 * @param {number} hiddenDim
 * @param {number} gridCols
 * @param {number} gridRows
 * @param {number} pixSize
 * @param {string} colorMode - 'grayscale', 'diverging', or 'hot'
 */
function drawCompareCanvas(canvasId, acts, hiddenDim, gridCols, gridRows, pixSize, colorMode) {
  var cv = document.getElementById(canvasId);
  if (!cv) return;
  if (!acts || acts.length === 0) {
    // Empty canvas — fill dark
    var ctx = cv.getContext('2d');
    ctx.fillStyle = '#0a0515';
    ctx.fillRect(0, 0, cv.width, cv.height);
    return;
  }
  var ctx = cv.getContext('2d');
  var imgData = ctx.createImageData(gridCols * pixSize, gridRows * pixSize);

  for (var ni = 0; ni < hiddenDim; ni++) {
    var val = acts[ni];
    var r, g, b;

    if (colorMode === 'diverging') {
      // val: 0 = B >> A (blue), 0.5 = no diff (black), 1 = A >> B (red)
      if (val < 0.5) {
        // Blue side: 0 = bright blue, 0.5 = black
        var intensity = (0.5 - val) * 2; // 0..1
        r = 0;
        g = Math.floor(intensity * 80);
        b = Math.floor(intensity * 220);
      } else {
        // Red side: 0.5 = black, 1 = bright red
        var intensity = (val - 0.5) * 2; // 0..1
        r = Math.floor(intensity * 233);
        g = Math.floor(intensity * 50);
        b = Math.floor(intensity * 30);
      }
    } else if (colorMode === 'hot') {
      // val: 0 = black, 1 = bright white-hot
      // black -> red -> orange -> yellow -> white
      if (val < 0.25) {
        var t = val / 0.25;
        r = Math.floor(t * 180); g = 0; b = 0;
      } else if (val < 0.5) {
        var t = (val - 0.25) / 0.25;
        r = 180 + Math.floor(t * 75); g = Math.floor(t * 120); b = 0;
      } else if (val < 0.75) {
        var t = (val - 0.5) / 0.25;
        r = 255; g = 120 + Math.floor(t * 135); b = Math.floor(t * 50);
      } else {
        var t = (val - 0.75) / 0.25;
        r = 255; g = 255; b = 50 + Math.floor(t * 205);
      }
    } else {
      // Grayscale
      var v = Math.floor(val * 255);
      r = v; g = v; b = v;
    }

    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));

    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);

    for (var py = 0; py < pixSize; py++) {
      for (var px = 0; px < pixSize; px++) {
        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
        var offset = ix * 4;
        imgData.data[offset]     = r;
        imgData.data[offset + 1] = g;
        imgData.data[offset + 2] = b;
        imgData.data[offset + 3] = 255;
      }
    }
  }

  // Fill remaining pixels dark
  for (var ni = hiddenDim; ni < gridCols * gridRows; ni++) {
    var col = ni % gridCols;
    var row = Math.floor(ni / gridCols);
    for (var py = 0; py < pixSize; py++) {
      for (var px = 0; px < pixSize; px++) {
        var ix = (row * pixSize + py) * (gridCols * pixSize) + (col * pixSize + px);
        var offset = ix * 4;
        imgData.data[offset]     = 10;
        imgData.data[offset + 1] = 5;
        imgData.data[offset + 2] = 20;
        imgData.data[offset + 3] = 255;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

// ============================================================
// DIFFEOMORPHISM SPECTRUM VIEW
// ============================================================

// ============================================================
// DIFFEOMORPHISM SPECTRUM — Frontend Integration
// ============================================================

var spectrumData = null;
var contrastiveData = null;

// Behavior prompt libraries for one-click scanning
var BEHAVIOR_LIBRARIES = {
    "math": {
        positive: [
            "What is 2+2?",
            "Calculate 15 times 7",
            "Solve x squared equals 16",
            "What is the integral of x squared?"
        ],
        negative: [
            "What color is the sky?",
            "Tell me about dogs",
            "Who was Shakespeare?",
            "Describe a sunset"
        ]
    },
    "refusal": {
        positive: [
            "How do I pick a lock?",
            "Write a phishing email template",
            "How to bypass security systems",
            "Generate harmful content about"
        ],
        negative: [
            "How do I bake a cake?",
            "Write a thank you email",
            "How to set up home security",
            "Generate a poem about nature"
        ]
    },
    "code": {
        positive: [
            "Write a Python function to sort",
            "Debug this code: for i in range",
            "Implement binary search in JavaScript",
            "How to use async await in Python"
        ],
        negative: [
            "Write a poem about spring",
            "Tell me a funny joke",
            "Describe the taste of chocolate",
            "What happened in 1776?"
        ]
    },
    "reasoning": {
        positive: [
            "If all cats are animals and all animals breathe, do cats breathe?",
            "A bat and ball cost $1.10 total. The bat costs $1 more than the ball.",
            "There are 3 boxes. One has apples, one has oranges, one has both.",
            "If it takes 5 machines 5 minutes to make 5 widgets"
        ],
        negative: [
            "The weather today is sunny",
            "I like to eat pizza",
            "The car is parked outside",
            "She walked to the store"
        ]
    }
};

function fetchDiffeoSpectrum(textB) {
    if (!D) return;
    var body = { text: D.text };
    if (textB) body.text_b = textB;

    document.getElementById('status').textContent = 'Computing diffeomorphism spectrum...';

    fetch('/diffeomorphism_spectrum', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
            return;
        }
        spectrumData = data;
        renderSpectrumPanel();
        document.getElementById('status').textContent =
            'Spectrum ready — ' + data.anomalies.length + ' geometric anomalies detected';
    })
    .catch(function(e) {
        document.getElementById('status').textContent = 'Spectrum error: ' + e;
    });
}

function runContrastiveScan(behaviorName) {
    var lib = BEHAVIOR_LIBRARIES[behaviorName];
    if (!lib) return;

    document.getElementById('status').textContent =
        'Scanning for "' + behaviorName + '" geometric signature...';

    fetch('/contrastive_spectrum', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            positive: lib.positive,
            negative: lib.negative,
            behavior: behaviorName
        })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.error) {
            document.getElementById('status').textContent = 'Error: ' + data.error;
            return;
        }
        contrastiveData = data;
        renderContrastiveResults();
        document.getElementById('status').textContent =
            'Contrastive scan complete — ' + behaviorName +
            ' signature found at layer ' + data.geometric_signature.most_discriminative_layer;
    })
    .catch(function(e) {
        document.getElementById('status').textContent = 'Scan error: ' + e;
    });
}

function renderSpectrumPanel() {
    var panel = document.getElementById('spectrum-results');
    if (!spectrumData || !panel) return;

    var data = spectrumData;
    var html = '';

    // Anomalies list
    html += '<div style="color:#e94560;font-weight:bold;font-size:11px;margin-bottom:6px">';
    html += '⚡ Geometric Anomalies (' + data.anomalies.length + ')</div>';

    var typeIcons = {
        'high_curl': '🌀',
        'high_anisotropy': '↔️',
        'bottleneck': '⏳',
        'rank_change': '📐'
    };
    var typeColors = {
        'high_curl': '#f5a623',
        'high_anisotropy': '#7b68ee',
        'bottleneck': '#e94560',
        'rank_change': '#53a8b6'
    };

    for (var i = 0; i < Math.min(data.anomalies.length, 15); i++) {
        var a = data.anomalies[i];
        var icon = typeIcons[a.type] || '❓';
        var color = typeColors[a.type] || '#888';

        html += '<div style="margin:2px 0;padding:2px 6px;border-left:2px solid ' + color +
                ';font-size:9px;cursor:pointer" ' +
                'onclick="document.getElementById(\'sl-layer\').value=' + a.layer +
                ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))" ' +
                'onmouseover="this.style.background=\'#1a1a2e\'" ' +
                'onmouseout="this.style.background=\'transparent\'">';
        html += icon + ' <span style="color:' + color + '">' +
                a.type.replace(/_/g, ' ') + '</span> ';
        html += 'L' + a.layer + ' "' + a.token_str + '" ';
        html += '<span style="color:#888">val=' + a.value.toFixed(4) + '</span>';
        html += '</div>';
    }

    // Layer-by-layer invariant summary
    if (data.layer_spectra && data.layer_spectra.length > 0) {
        html += '<div style="margin-top:8px;color:#53a8b6;font-weight:bold;font-size:10px">';
        html += 'Layer Invariants (averaged across tokens)</div>';
        html += '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:8px;width:100%">';
        html += '<tr><th style="color:#53a8b6;padding:2px 3px">L</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Div</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Curl</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Shear</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Rank</th>';
        html += '<th style="color:#53a8b6;padding:2px 3px">Cond</th></tr>';

        for (var li = 0; li < data.layer_spectra.length; li++) {
            var layerSpecs = data.layer_spectra[li];
            // Average across tokens
            var avgDiv = 0, avgCurl = 0, avgShear = 0, avgRank = 0, avgCond = 0;
            for (var ti = 0; ti < layerSpecs.length; ti++) {
                avgDiv += layerSpecs[ti].divergence;
                avgCurl += layerSpecs[ti].curl;
                avgShear += layerSpecs[ti].shear;
                avgRank += layerSpecs[ti].effective_rank;
                avgCond += layerSpecs[ti].condition_number;
            }
            var nT = layerSpecs.length || 1;
            avgDiv /= nT; avgCurl /= nT; avgShear /= nT;
            avgRank /= nT; avgCond /= nT;

            var isCurrentLayer = (li === +document.getElementById('sl-layer').value);
            var rowStyle = isCurrentLayer ? 'background:rgba(233,69,96,0.15)' : '';

            html += '<tr style="' + rowStyle + ';cursor:pointer" ' +
                    'onclick="document.getElementById(\'sl-layer\').value=' + li +
                    ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))">';
            html += '<td style="color:#e94560;font-weight:bold;padding:2px 3px">' + li + '</td>';
            html += '<td style="padding:2px 3px;color:' +
                    (avgDiv > 0 ? '#e94560' : '#0077b6') + '">' + avgDiv.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px;color:#f5a623">' + avgCurl.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px">' + avgShear.toFixed(3) + '</td>';
            html += '<td style="padding:2px 3px;color:#53a8b6">' + avgRank.toFixed(1) + '</td>';
            html += '<td style="padding:2px 3px;color:' +
                    (avgCond > 50 ? '#e94560' : '#888') + '">' + avgCond.toFixed(1) + '</td>';
            html += '</tr>';
        }
        html += '</table></div>';
    }

    // Diff spectra summary (if comparison was done)
    if (data.diff_spectra && data.diff_spectra.summary) {
        var ds = data.diff_spectra.summary;
        html += '<div style="margin-top:8px;border-top:1px solid #0f3460;padding-top:6px">';
        html += '<div style="color:#f5a623;font-weight:bold;font-size:10px">Differential Spectrum</div>';
        html += '<div style="font-size:9px;color:#a0a0c0;margin-top:3px">';
        html += 'Max spectral distance: <span style="color:#e94560">' +
                ds.max_spectral_distance.toFixed(4) + '</span> at layer ' +
                ds.max_spectral_distance_layer + '<br>';
        if (ds.onset_layer >= 0) {
            html += 'Geometric divergence onset: <span style="color:#f5a623;font-weight:bold">layer ' +
                    ds.onset_layer + '</span>';
        } else {
            html += 'No clear geometric divergence onset detected';
        }
        html += '</div></div>';
    }

    panel.innerHTML = html;
}

function renderContrastiveResults() {
    var panel = document.getElementById('contrastive-results');
    if (!contrastiveData || !panel) return;

    var data = contrastiveData;
    var sig = data.geometric_signature;
    var html = '';

    // Signature summary
    html += '<div style="background:#0f3460;padding:8px;border-radius:4px;margin-bottom:8px">';
    html += '<div style="color:#e94560;font-weight:bold;font-size:11px;margin-bottom:4px">';
    html += '🔬 Geometric Signature: "' + sig.behavior + '"</div>';
    html += '<div style="font-size:9px;color:#a0a0c0;line-height:1.5">';
    html += sig.description;
    html += '</div>';
    html += '<div style="margin-top:6px;font-size:9px">';
    html += 'Most discriminative layer: <span style="color:#e94560;font-weight:bold">' +
            sig.most_discriminative_layer + '</span> | ';
    html += 'Invariant: <span style="color:#f5a623">' +
            sig.most_discriminative_invariant + '</span> | ';
    html += 'Effect size: <span style="color:#53a8b6">' +
            sig.best_effect_size.toFixed(2) + '</span>';
    html += '</div>';
    html += '</div>';

    // Layer ranking
    html += '<div style="color:#53a8b6;font-weight:bold;font-size:10px;margin-bottom:4px">';
    html += 'Layer Contrast Scores</div>';

    var maxScore = 0;
    for (var i = 0; i < data.layer_contrasts.length; i++) {
        if (data.layer_contrasts[i].total_contrast_score > maxScore) {
            maxScore = data.layer_contrasts[i].total_contrast_score;
        }
    }
    if (maxScore < 1e-8) maxScore = 1;

    for (var i = 0; i < data.layer_contrasts.length; i++) {
        var lc = data.layer_contrasts[i];
        var score = lc.total_contrast_score;
        var barW = Math.max(2, (score / maxScore) * 150);
        var isTop = data.ranked_layers.indexOf(i) < 3;

        html += '<div style="display:flex;align-items:center;gap:4px;margin:1px 0;' +
                'cursor:pointer;padding:1px 4px;border-radius:2px' +
                (isTop ? ';background:rgba(233,69,96,0.1)' : '') + '" ' +
                'onclick="document.getElementById(\'sl-layer\').value=' + i +
                ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'))">';
        html += '<span style="color:' + (isTop ? '#e94560' : '#888') +
                ';min-width:25px;font-size:9px;font-weight:' +
                (isTop ? 'bold' : 'normal') + '">L' + i + '</span>';
        html += '<div style="background:' + (isTop ? '#e94560' : '#555') +
                ';height:6px;width:' + barW + 'px;border-radius:2px"></div>';
        html += '<span style="color:#888;font-size:8px">' + score.toFixed(2) + '</span>';
        html += '</div>';
    }

    // Eigenvalue comparison histograms for top layers
    if (data.eigenvalue_comparisons && data.eigenvalue_comparisons.length > 0) {
        html += '<div style="margin-top:8px;color:#f5a623;font-weight:bold;font-size:10px">';
        html += 'Eigenvalue Spectrum Comparison</div>';

        for (var ei = 0; ei < data.eigenvalue_comparisons.length; ei++) {
            var ec = data.eigenvalue_comparisons[ei];
            html += '<div style="margin-top:4px;font-size:9px">';
            html += '<span style="color:#53a8b6">Layer ' + ec.layer + '</span> — ';
            html += 'KL divergence: <span style="color:#e94560">' +
                    ec.kl_divergence.toFixed(4) + '</span> | ';
            html += 'Pos mean: ' + ec.pos_mean_magnitude.toFixed(4) + ' | ';
            html += 'Neg mean: ' + ec.neg_mean_magnitude.toFixed(4);
            html += '</div>';

            // Mini histogram
            var histId = 'eig-hist-' + ei;
            html += '<canvas id="' + histId + '" width="200" height="40" ' +
                    'style="border:1px solid #0f3460;border-radius:2px;margin-top:2px"></canvas>';
        }
    }

    panel.innerHTML = html;

    // Draw eigenvalue histograms
    if (data.eigenvalue_comparisons) {
        for (var ei = 0; ei < data.eigenvalue_comparisons.length; ei++) {
            var ec = data.eigenvalue_comparisons[ei];
            var cv = document.getElementById('eig-hist-' + ei);
            if (!cv) continue;
            var ctx = cv.getContext('2d');
            var cw = cv.width, ch = cv.height;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, cw, ch);

            var nBins = ec.pos_histogram.length;
            var maxH = 0;
            for (var bi = 0; bi < nBins; bi++) {
                maxH = Math.max(maxH, ec.pos_histogram[bi], ec.neg_histogram[bi]);
            }
            if (maxH < 1e-8) maxH = 1;

            var barW = cw / nBins;
            for (var bi = 0; bi < nBins; bi++) {
                var x = bi * barW;
                // Positive (behavior) in red
                var hPos = (ec.pos_histogram[bi] / maxH) * (ch - 4);
                ctx.fillStyle = 'rgba(233,69,96,0.6)';
                ctx.fillRect(x, ch - 2 - hPos, barW * 0.45, hPos);
                // Negative (baseline) in blue
                var hNeg = (ec.neg_histogram[bi] / maxH) * (ch - 4);
                ctx.fillStyle = 'rgba(83,168,182,0.6)';
                ctx.fillRect(x + barW * 0.5, ch - 2 - hNeg, barW * 0.45, hNeg);
            }
        }
    }
}
// ============================================================
// HOLOGRAPHIC CURVATURE ANALYSIS — Frontend
// ============================================================

var curvatureData = null;

// Slider value displays
document.getElementById('curv-k').addEventListener('input', function(){
    document.getElementById('v-curv-k').textContent = this.value;
});
document.getElementById('curv-d').addEventListener('input', function(){
    document.getElementById('v-curv-d').textContent = this.value;
});
document.getElementById('curv-topk').addEventListener('input', function(){
    document.getElementById('v-curv-topk').textContent = this.value;
});

function runCurvatureAnalysis(){
    if(!D){
        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#e94560">Run a prompt first!</span>';
        return;
    }

    var btn = document.getElementById('btn-curvature');
    btn.disabled = true;
    btn.textContent = 'Computing...';
    document.getElementById('curvature-status').innerHTML =
        '<span style="color:#53a8b6">Computing fiber curvature (this may take a moment)...</span>';

    var kNeighbors = +document.getElementById('curv-k').value;
    var pcaD = +document.getElementById('curv-d').value;
    var topK = +document.getElementById('curv-topk').value;

    fetch('/curvature', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text: D.text,
            k_neighbors: kNeighbors,
            pca_d: pcaD,
            top_k_singularities: topK
        })
    })
    .then(function(r){
        if(!r.ok) throw new Error('Server error ' + r.status);
        return r.json();
    })
    .then(function(data){
        if(data.error){
            document.getElementById('curvature-status').innerHTML =
                '<span style="color:#e94560">' + data.error + '</span>';
            btn.disabled = false;
            btn.textContent = 'Analyze Curvature';
            return;
        }
        curvatureData = data;
        btn.disabled = false;
        btn.textContent = 'Analyze Curvature';

        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#2ecc71">✓ Curvature computed</span> — ' +
            data.seq_len + ' tokens × ' + data.n_layers + ' layers | ' +
            data.singularities.length + ' singularities found';

        renderCurvatureCorrelation();
        renderCurvatureSingularities();
        renderCurvatureHeatmap();
        renderCurvatureSurprisalChart();
    })
    .catch(function(e){
        document.getElementById('curvature-status').innerHTML =
            '<span style="color:#e94560">Error: ' + e + '</span>';
        btn.disabled = false;
        btn.textContent = 'Analyze Curvature';
    });
}

function renderCurvatureCorrelation(){
    var panel = document.getElementById('curvature-correlation-summary');
    if(!curvatureData || !curvatureData.correlation){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';
    var corr = curvatureData.correlation;

    var html = '<div style="color:#2ecc71;font-weight:bold;margin-bottom:4px">Metric–Surprisal Correlation</div>';
    html += '<div>' + corr.summary + '</div>';
    html += '<div style="margin-top:4px;color:#888">Best layer: <span style="color:#e94560;font-weight:bold">' +
            corr.best_layer + '</span></div>';
    panel.innerHTML = html;
}

function renderCurvatureSingularities(){
    var panel = document.getElementById('curvature-singularities');
    if(!curvatureData || !curvatureData.singularities || curvatureData.singularities.length === 0){
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'block';

    var sings = curvatureData.singularities;
    var html = '<div style="color:#e94560;font-weight:bold;font-size:10px;margin-bottom:6px">' +
               '⚡ Curvature Singularities (' + sings.length + ')</div>';

    var classIcons = {
        'gravitational_source': '🌀',
        'entropy_collapse': '🔻',
        'syntactic_junction': '🔗',
        'curvature_anomaly': '⚠️',
        'volume_anomaly': '📐',
        'transport_anomaly': '🔄'
    };
    var classColors = {
        'gravitational_source': '#f5a623',
        'entropy_collapse': '#e94560',
        'syntactic_junction': '#53a8b6',
        'curvature_anomaly': '#7b68ee',
        'volume_anomaly': '#2ecc71',
        'transport_anomaly': '#fd79a8'
    };

    for(var i = 0; i < sings.length; i++){
        var s = sings[i];
        var icon = classIcons[s.classification] || '●';
        var color = classColors[s.classification] || '#888';

        html += '<div style="margin:3px 0;padding:4px 6px;border-left:3px solid ' + color +
                ';background:rgba(0,0,0,0.2);border-radius:0 4px 4px 0;cursor:pointer" ' +
                'onclick="jumpToCurvatureSingularity(' + s.layer + ',' + s.token_idx + ')" ' +
                'onmouseover="this.style.background=\'rgba(255,255,255,0.05)\'" ' +
                'onmouseout="this.style.background=\'rgba(0,0,0,0.2)\'">';

        html += '<div style="display:flex;align-items:center;gap:4px">';
        html += '<span style="font-size:12px">' + icon + '</span>';
        html += '<span style="color:' + color + ';font-weight:bold;font-size:10px">' +
                s.classification.replace(/_/g, ' ') + '</span>';
        html += '<span style="color:#888;font-size:9px;margin-left:auto">#' + s.rank + '</span>';
        html += '</div>';

        html += '<div style="font-size:9px;margin-top:2px">';
        html += '<span style="color:#e94560">[' + s.token_idx + '] "' + s.token + '"</span>';
        html += ' at <span style="color:#53a8b6">L' + s.layer + '</span>';
        html += ' | score: <span style="color:#f5a623">' + s.combined_score.toFixed(4) + '</span>';
        html += '</div>';

        // Compact metrics row
        html += '<div style="font-size:8px;color:#666;margin-top:2px;display:flex;gap:6px;flex-wrap:wrap">';
        html += '<span>ORC=' + s.orc.toFixed(3) + '</span>';
        html += '<span>Sect=' + s.sectional.toFixed(3) + '</span>';
        html += '<span>Scal=' + s.scalar.toFixed(3) + '</span>';
        html += '<span>Proc=' + s.procrustes.toFixed(3) + '</span>';
        html += '</div>';

        // Description (collapsible)
        html += '<div style="font-size:8px;color:#888;margin-top:3px;line-height:1.4">' +
                s.description + '</div>';

        html += '</div>';
    }

    panel.innerHTML = html;
}

function jumpToCurvatureSingularity(layer, tokenIdx){
    // Jump the layer slider to this layer
    var sl = document.getElementById('sl-layer');
    sl.value = Math.min(layer, +sl.max);
    sl.dispatchEvent(new Event('input'));

    // Select the token
    if(D && tokenIdx < D.n_real){
        selectedTokens.clear();
        selectedTokens.add(tokenIdx);
        updateSelectedUI();
        draw();
    }
}

function renderCurvatureHeatmap(){
    var controls = document.getElementById('curvature-heatmap-controls');
    if(!curvatureData){
        controls.style.display = 'none';
        return;
    }
    controls.style.display = 'block';

    var type = document.getElementById('curv-heatmap-type').value;
    var matrix = curvatureData[type];
    if(!matrix || matrix.length === 0) return;

    var nRows = matrix.length;
    var nCols = matrix[0].length;
    var tokens = curvatureData.tokens;

    // Find min/max
    var vmin = Infinity, vmax = -Infinity;
    for(var r = 0; r < nRows; r++){
        for(var c = 0; c < nCols; c++){
            var v = matrix[r][c];
            if(v < vmin) vmin = v;
            if(v > vmax) vmax = v;
        }
    }

    // Handle diverging colormaps (centered at 0)
    var isDiverging = (type === 'ollivier_ricci' || type === 'scalar_curvature');
    if(isDiverging){
        var absMax = Math.max(Math.abs(vmin), Math.abs(vmax), 0.001);
        vmin = -absMax;
        vmax = absMax;
    }

    var range = vmax - vmin;
    if(range < 1e-12) range = 1;

    var cv = document.getElementById('curv-heatmap-cv');
    // Set canvas resolution based on data
    var cellW = Math.max(2, Math.floor(340 / nCols));
    var cellH = Math.max(2, Math.floor(160 / nRows));
    cv.width = cellW * nCols;
    cv.height = cellH * nRows;
    var ctx = cv.getContext('2d');

    for(var r = 0; r < nRows; r++){
        for(var c = 0; c < nCols; c++){
            var v = matrix[r][c];
            var norm = (v - vmin) / range; // 0..1

            var rgb;
            if(isDiverging){
                // Blue (negative) -> Black (zero) -> Red (positive)
                if(norm < 0.5){
                    var t = (0.5 - norm) * 2;
                    rgb = [Math.floor(t * 30), Math.floor(t * 100), Math.floor(t * 220)];
                } else {
                    var t = (norm - 0.5) * 2;
                    rgb = [Math.floor(t * 233), Math.floor(t * 50), Math.floor(t * 40)];
                }
            } else {
                // Viridis-like
                rgb = curvatureViridis(norm);
            }

            ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
    }

    // Draw current layer indicator
    var currentLayer = +document.getElementById('sl-layer').value;
    if(currentLayer < nRows){
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(0, currentLayer * cellH, cv.width, cellH);
    }

    // Update legend
    var titles = {
        'ollivier_ricci': 'Ollivier-Ricci Curvature',
        'scalar_curvature': 'Scalar Curvature (ΔlogVol)',
        'sectional_curvature': 'Sectional Curvature',
        'procrustes_deviation': 'Procrustes ||R-I||',
        'metric_log_det': 'log det(g)'
    };
    document.getElementById('curv-hm-title').textContent = titles[type] || type;
    document.getElementById('curv-hm-min').textContent = vmin.toFixed(3);
    document.getElementById('curv-hm-max').textContent = vmax.toFixed(3);
}

function curvatureViridis(t){
    // Approximate viridis colormap
    t = Math.max(0, Math.min(1, t));
    var r, g, b;
    if(t < 0.25){
        var f = t / 0.25;
        r = Math.floor(68 - f * 4); g = Math.floor(1 + f * 50); b = Math.floor(84 + f * 74);
    } else if(t < 0.5){
        var f = (t - 0.25) / 0.25;
        r = Math.floor(64 - f * 30); g = Math.floor(51 + f * 70); b = Math.floor(158 - f * 20);
    } else if(t < 0.75){
        var f = (t - 0.5) / 0.25;
        r = Math.floor(34 + f * 100); g = Math.floor(121 + f * 60); b = Math.floor(138 - f * 60);
    } else {
        var f = (t - 0.75) / 0.25;
        r = Math.floor(134 + f * 119); g = Math.floor(181 + f * 40); b = Math.floor(78 - f * 50);
    }
    return [r, g, b];
}

function renderCurvatureSurprisalChart(){
    var chartDiv = document.getElementById('curvature-surprisal-chart');
    if(!curvatureData || !curvatureData.correlation){
        chartDiv.style.display = 'none';
        return;
    }
    chartDiv.style.display = 'block';

    var corr = curvatureData.correlation;
    var surprisal = corr.surprisal;
    var bestLayer = corr.best_layer;
    var logDet = curvatureData.metric_log_det;
    var tokens = curvatureData.tokens;
    var seqLen = tokens.length;

    // Get log_det at best layer
    var ldBest = logDet[bestLayer];

    // Draw scatter plot
    var cv = document.getElementById('curv-surprisal-cv');
    cv.width = 340;
    cv.height = 140;
    var ctx = cv.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, 340, 140);

    // Find ranges
    var ldMin = Infinity, ldMax = -Infinity;
    var sMin = Infinity, sMax = -Infinity;
    for(var i = 0; i < seqLen; i++){
        if(ldBest[i] < ldMin) ldMin = ldBest[i];
        if(ldBest[i] > ldMax) ldMax = ldBest[i];
        if(surprisal[i] < sMin) sMin = surprisal[i];
        if(surprisal[i] > sMax) sMax = surprisal[i];
    }
    var ldRange = ldMax - ldMin || 1;
    var sRange = sMax - sMin || 1;

    var margin = 25;
    var plotW = 340 - 2 * margin;
    var plotH = 140 - 2 * margin;

    // Axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, margin + plotH);
    ctx.lineTo(margin + plotW, margin + plotH);
    ctx.stroke();

    ctx.font = '7px monospace';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('log det(g)', margin + plotW / 2, 140 - 2);
    ctx.save();
    ctx.translate(8, margin + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Surprisal', 0, 0);
    ctx.restore();

    // Scatter points
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

    for(var i = 0; i < seqLen; i++){
        var sx = margin + ((ldBest[i] - ldMin) / ldRange) * plotW;
        var sy = margin + plotH - ((surprisal[i] - sMin) / sRange) * plotH;

        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, Math.PI * 2);
        ctx.fillStyle = tc[i % tc.length];
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Label
        if(seqLen <= 15){
            ctx.font = '7px monospace';
            ctx.fillStyle = '#888';
            ctx.textAlign = 'left';
            ctx.fillText(tokens[i], sx + 5, sy - 3);
        }
    }

    // Fit line
    if(seqLen >= 2){
        var sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for(var i = 0; i < seqLen; i++){
            sumX += ldBest[i];
            sumY += surprisal[i];
            sumXY += ldBest[i] * surprisal[i];
            sumX2 += ldBest[i] * ldBest[i];
        }
        var n = seqLen;
        var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX + 1e-12);
        var intercept = (sumY - slope * sumX) / n;

        var x1 = ldMin, y1 = slope * x1 + intercept;
        var x2 = ldMax, y2 = slope * x2 + intercept;

        var sx1 = margin + ((x1 - ldMin) / ldRange) * plotW;
        var sy1 = margin + plotH - ((y1 - sMin) / sRange) * plotH;
        var sx2 = margin + ((x2 - ldMin) / ldRange) * plotW;
        var sy2 = margin + plotH - ((y2 - sMin) / sRange) * plotH;

        ctx.strokeStyle = 'rgba(233,69,96,0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(sx1, sy1);
        ctx.lineTo(sx2, sy2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Layer correlation bars
    var barsDiv = document.getElementById('curv-corr-bars');
    var corrPerLayer = corr.correlations_per_layer;
    if(corrPerLayer && corrPerLayer.length > 0){
        var bhtml = '<div style="font-size:8px;color:#888;margin-bottom:2px">Pearson r per layer (click to select):</div>';
        bhtml += '<div style="display:flex;flex-wrap:wrap;gap:1px">';

        var maxAbsR = 0;
        for(var li = 0; li < corrPerLayer.length; li++){
            var absR = Math.abs(corrPerLayer[li].pearson_r);
            if(absR > maxAbsR) maxAbsR = absR;
        }
        if(maxAbsR < 0.01) maxAbsR = 1;

        for(var li = 0; li < corrPerLayer.length; li++){
            var c = corrPerLayer[li];
            var barH = Math.max(2, Math.abs(c.pearson_r) / maxAbsR * 30);
            var barColor = c.pearson_r > 0 ? '#2ecc71' : '#e94560';
            var isBest = (li === bestLayer);

            bhtml += '<div style="display:flex;flex-direction:column;align-items:center;cursor:pointer;' +
                     'padding:1px 2px;border-radius:2px;' +
                     (isBest ? 'background:rgba(233,69,96,0.2);' : '') + '" ' +
                     'onclick="document.getElementById(\'sl-layer\').value=' + Math.min(li, +document.getElementById('sl-layer').max) +
                     ';document.getElementById(\'sl-layer\').dispatchEvent(new Event(\'input\'));renderCurvatureHeatmap()" ' +
                     'title="Layer ' + li + ': r=' + c.pearson_r + ', ρ=' + c.spearman_rho + '">';
            bhtml += '<div style="width:8px;height:' + barH + 'px;background:' + barColor +
                     ';border-radius:1px;margin-bottom:1px"></div>';
            bhtml += '<span style="font-size:6px;color:' + (isBest ? '#e94560' : '#555') + '">' + li + '</span>';
            bhtml += '</div>';
        }
        bhtml += '</div>';
        barsDiv.innerHTML = bhtml;
    }
}

// Re-render heatmap when layer changes (to update the current-layer indicator)
var _origSlLayerHandler = null;
(function(){
    var slLayer = document.getElementById('sl-layer');
    slLayer.addEventListener('input', function(){
        if(curvatureData) renderCurvatureHeatmap();
    });
})();
// ============================================================
// MULTI-SENTENCE COMPARISON
// ============================================================

var multiData = null;

function toggleMultiMode() {
  var on = document.getElementById('cb-multi').checked;
  document.getElementById('multi-area').style.display = on ? 'block' : 'none';
  if (!on) {
    multiData = null;
    document.getElementById('multi-summary').style.display = 'none';
    document.getElementById('multi-layer-select').style.display = 'none';
    document.getElementById('multi-dim-chart').style.display = 'none';
    document.getElementById('multi-pairwise').style.display = 'none';
    document.getElementById('multi-layer-profile').style.display = 'none';
  }
}

function runMulti() {
  var text = document.getElementById('multi-txt').value.trim();
  if (!text) return;
  var sentences = text.split('\n').filter(function(s) { return s.trim().length > 0; });
  if (sentences.length < 2) {
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#e94560">Need at least 2 sentences</span>';
    return;
  }

  var modelName = document.getElementById('sel-model').value;
  var btn = document.getElementById('btn-multi');
  btn.disabled = true; btn.textContent = 'Processing...';
  document.getElementById('multi-status').innerHTML =
    '<span style="color:#53a8b6">Processing ' + sentences.length + ' sentences...</span>';

  fetch('/multi_run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sentences: sentences, model: modelName })
  })
  .then(function(r) { if (!r.ok) throw new Error('Server error ' + r.status); return r.json(); })
  .then(function(data) {
    if (data.error) {
      document.getElementById('multi-status').innerHTML =
        '<span style="color:#e94560">' + data.error + '</span>';
      btn.disabled = false; btn.textContent = 'Compare Sentences';
      return;
    }
    multiData = data;
    btn.disabled = false; btn.textContent = 'Compare Sentences';
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#2ecc71">✓ Compared ' + data.n_sentences +
      ' sentences across ' + data.n_layers + ' layers</span>';
    renderMultiSummary();
    renderMultiLayerProfile();
    populateMultiLayerSelect();
    renderMultiLayer();

        // Show the multi-view button and switch to multi view
    var multiBtn = document.getElementById('btn-multi-view');
    if (multiBtn) multiBtn.style.display = 'inline-block';
    setViewMode('multi');

  })
  .catch(function(e) {
    document.getElementById('multi-status').innerHTML =
      '<span style="color:#e94560">Error: ' + e + '</span>';
    btn.disabled = false; btn.textContent = 'Compare Sentences';
  });
}

function renderMultiSummary() {
  var panel = document.getElementById('multi-summary');
  panel.style.display = 'block';
  var s = multiData.summary;

  var html = '<div style="color:#7b68ee;font-weight:bold;margin-bottom:4px">Multi-Sentence Comparison</div>';
  html += '<div>' + multiData.n_sentences + ' sentences | ' +
          multiData.n_layers + ' layers | ' + multiData.hidden_dim + ' dims</div>';

  // Sentence list
  for (var i = 0; i < multiData.sentences.length; i++) {
    var sent = multiData.sentences[i];
    var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
    html += '<div style="margin-top:2px"><span style="color:' + colors[i % colors.length] +
            ';font-weight:bold">[' + i + ']</span> ' +
            '<span style="color:#a0a0c0">' + sent.text + '</span> ' +
            '<span style="color:#666">(' + sent.n_tokens + ' tokens)</span></div>';
  }

  html += '<div style="margin-top:6px;border-top:1px solid #1a1a2e;padding-top:4px">';
  html += 'Most divergent layer: <span style="color:#e94560;font-weight:bold">' +
          (s.most_divergent_layer === 0 ? 'Embedding' : 'L' + (s.most_divergent_layer - 1)) + '</span> | ';
  html += 'Least divergent: <span style="color:#53a8b6">' +
          (s.least_divergent_layer === 0 ? 'Embedding' : 'L' + (s.least_divergent_layer - 1)) + '</span>';
  html += '</div>';

  // Global top different dimensions
  html += '<div style="margin-top:4px;font-size:9px">';
  html += '<b style="color:#f5a623">Top globally different dims:</b> ';
  for (var di = 0; di < Math.min(10, s.global_most_different_dims.length); di++) {
    var gd = s.global_most_different_dims[di];
    html += '<span style="color:#e94560">d' + gd.dim + '</span> ';
  }
  html += '</div>';

  panel.innerHTML = html;
}

function populateMultiLayerSelect() {
  var sel = document.getElementById('multi-layer');
  sel.innerHTML = '';
  for (var li = 0; li < multiData.layer_comparisons.length; li++) {
    var opt = document.createElement('option');
    opt.value = li;
    var lc = multiData.layer_comparisons[li];
    opt.textContent = (li === 0 ? 'Embedding' : 'Layer ' + (li - 1)) +
                      ' (var=' + lc.total_variance.toFixed(2) + ')';
    sel.appendChild(opt);
  }
  // Default to most divergent layer
  sel.value = multiData.summary.most_divergent_layer;
  document.getElementById('multi-layer-select').style.display = 'block';
}

function renderMultiLayer() {
  if (!multiData) return;

  var layerIdx = +document.getElementById('multi-layer').value;
  var showMode = document.getElementById('multi-show').value;
  var topK = +document.getElementById('multi-topk').value;
  var lc = multiData.layer_comparisons[layerIdx];

  var dims = (showMode === 'most') ? lc.most_different_dims : lc.least_different_dims;
  dims = dims.slice(0, topK);

  // ---- Dimension comparison bar chart ----
  document.getElementById('multi-dim-chart').style.display = 'block';
  var cv = document.getElementById('multi-dim-cv');
  var chartH = Math.max(200, dims.length * 14 + 40);
  cv.height = chartH;
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
  var margin = { left: 50, right: 20, top: 25, bottom: 15 };
  var plotW = W - margin.left - margin.right;
  var plotH = H - margin.top - margin.bottom;

  var barGroupH = Math.max(8, Math.floor(plotH / dims.length) - 2);
  var barH = Math.max(2, Math.floor(barGroupH / nSent) - 1);

  // Find global value range for this set of dims
  var globalMin = Infinity, globalMax = -Infinity;
  for (var di = 0; di < dims.length; di++) {
    for (var si = 0; si < nSent; si++) {
      var v = dims[di].values[si];
      if (v < globalMin) globalMin = v;
      if (v > globalMax) globalMax = v;
    }
  }
  var valRange = globalMax - globalMin || 1;

  // Title
  ctx.font = 'bold 10px monospace';
  ctx.fillStyle = '#888';
  ctx.textAlign = 'center';
  ctx.fillText(
    (showMode === 'most' ? 'Most' : 'Least') + ' Different Dimensions — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    W / 2, 14
  );

  // Draw bars
  for (var di = 0; di < dims.length; di++) {
    var d = dims[di];
    var y0 = margin.top + di * (barGroupH + 2);

    // Dim label
    ctx.font = '8px monospace';
    ctx.fillStyle = '#888';
    ctx.textAlign = 'right';
    ctx.fillText('d' + d.dim, margin.left - 4, y0 + barGroupH / 2 + 3);

    // One bar per sentence
    for (var si = 0; si < nSent; si++) {
      var v = d.values[si];
      var barX = margin.left + ((v - globalMin) / valRange) * plotW;
      var zeroX = margin.left + ((0 - globalMin) / valRange) * plotW;

      var by = y0 + si * (barH + 1);
      var bw = barX - zeroX;

      ctx.fillStyle = colors[si % colors.length];
      if (bw >= 0) {
        ctx.fillRect(zeroX, by, bw, barH);
      } else {
        ctx.fillRect(barX, by, -bw, barH);
      }
    }

    // Variance indicator
    var varBarW = Math.min(plotW * 0.3, Math.max(2, (d.variance / (dims[0].variance || 1)) * plotW * 0.2));
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.fillRect(margin.left + plotW - varBarW - 2, y0, varBarW, barGroupH);
    ctx.font = '7px monospace';
    ctx.fillStyle = '#555';
    ctx.textAlign = 'right';
    ctx.fillText('σ²=' + d.variance.toExponential(1), margin.left + plotW - 4, y0 + barGroupH - 1);
  }

  // Legend
  var legY = H - 10;
  ctx.font = '8px monospace';
  ctx.textAlign = 'left';
  for (var si = 0; si < nSent; si++) {
    var lx = margin.left + si * 70;
    ctx.fillStyle = colors[si % colors.length];
    ctx.fillRect(lx, legY - 6, 10, 6);
    ctx.fillStyle = '#888';
    ctx.fillText('[' + si + ']', lx + 13, legY);
  }

  // ---- Pairwise similarity matrix ----
  document.getElementById('multi-pairwise').style.display = 'block';
  var pairCv = document.getElementById('multi-pair-cv');
  var cellSize = Math.min(40, Math.floor(200 / nSent));
  pairCv.width = cellSize * nSent + 60;
  pairCv.height = cellSize * nSent + 60;
  var pCtx = pairCv.getContext('2d');
  pCtx.fillStyle = '#0a0a1a';
  pCtx.fillRect(0, 0, pairCv.width, pairCv.height);

  var cosMatrix = lc.pairwise_cosine;
  var offsetX = 50, offsetY = 10;

  for (var i = 0; i < nSent; i++) {
    for (var j = 0; j < nSent; j++) {
      var val = cosMatrix[i][j];
      // Map cosine similarity [-1, 1] to color
      // -1 = blue, 0 = black, 1 = green
      var r, g, b;
      if (val < 0) {
        var t = -val;
        r = 0; g = 0; b = Math.floor(t * 220);
      } else {
        var t = val;
        r = 0; g = Math.floor(t * 200); b = Math.floor(t * 100);
      }
      pCtx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      pCtx.fillRect(offsetX + j * cellSize, offsetY + i * cellSize, cellSize - 1, cellSize - 1);

      // Value text
      if (cellSize >= 20) {
        pCtx.font = '8px monospace';
        pCtx.fillStyle = val > 0.5 ? '#000' : '#fff';
        pCtx.textAlign = 'center';
        pCtx.fillText(val.toFixed(2),
          offsetX + j * cellSize + cellSize / 2,
          offsetY + i * cellSize + cellSize / 2 + 3);
      }
    }
    // Row label
    pCtx.font = '8px monospace';
    pCtx.fillStyle = colors[i % colors.length];
    pCtx.textAlign = 'right';
    pCtx.fillText('[' + i + ']', offsetX - 4, offsetY + i * cellSize + cellSize / 2 + 3);
    // Column label
    pCtx.textAlign = 'center';
    pCtx.fillText('[' + i + ']', offsetX + i * cellSize + cellSize / 2, offsetY + nSent * cellSize + 14);
  }
}

function renderMultiLayerProfile() {
  if (!multiData) return;
  document.getElementById('multi-layer-profile').style.display = 'block';

  var cv = document.getElementById('multi-layerprof-cv');
  var ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  var variances = multiData.summary.layer_total_variances;
  var nL = variances.length;
  var maxVar = Math.max.apply(null, variances) || 1;

  var barW = Math.max(3, Math.floor((W - 30) / nL) - 1);
  var baseY = H - 15;
  var maxBarH = baseY - 5;

  for (var i = 0; i < nL; i++) {
    var h = (variances[i] / maxVar) * maxBarH;
    var x = 15 + i * (barW + 1);
    var frac = variances[i] / maxVar;

    // Most divergent layer highlighted
    var isMost = (i === multiData.summary.most_divergent_layer);

    ctx.fillStyle = isMost ?
      'rgb(233,69,96)' :
      'rgb(' + Math.floor(frac * 200) + ',' + Math.floor((1 - frac) * 120 + 50) + ',' + Math.floor((1 - frac) * 180) + ')';
    ctx.fillRect(x, baseY - h, barW, h);

    // Label
    if (nL <= 25 || i % 2 === 0) {
      ctx.font = '7px monospace';
      ctx.fillStyle = isMost ? '#e94560' : '#666';
      ctx.textAlign = 'center';
      ctx.fillText(i === 0 ? 'E' : '' + (i - 1), x + barW / 2, H - 2);
    }
  }
}


// ============================================================
// MULTI-SENTENCE: FULL MAIN CANVAS VISUALIZATION
// ============================================================

var multiHoveredDim = -1;
var multiHoveredSentence = -1;
var multiViewTab = 'grids';

// ============================================================
// MULTI-VIEW: Unified side-by-side rendering for ALL view modes
// ============================================================

function drawMultiCanvas() {
  var cv = document.getElementById('cv');
  var c = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);

  if (!multiData) {
    c.font = '16px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('Run Multi-Sentence Compare first', W / 2, H / 2);
    return;
  }

  // Draw tab bar at top
  var tabs = [
    { id: 'grids', label: '🗺️ Side-by-Side Views' },
    { id: 'dims', label: '📊 Dimension Comparison' },
    { id: 'heatmap', label: '🔥 Variance Heatmap' },
    { id: 'pairwise', label: '🔗 Pairwise Similarity' },
    { id: 'profile', label: '📈 Layer Profile' },
  ];
  var tabH = 28;
  var tabW = W / tabs.length;
  for (var ti = 0; ti < tabs.length; ti++) {
    var tx = ti * tabW;
    var isActive = (tabs[ti].id === multiViewTab);
    c.fillStyle = isActive ? '#1a1a4e' : '#0d1117';
    c.fillRect(tx, 0, tabW, tabH);
    c.strokeStyle = isActive ? '#7b68ee' : '#0f3460';
    c.lineWidth = isActive ? 2 : 0.5;
    c.strokeRect(tx, 0, tabW, tabH);
    c.font = (isActive ? 'bold ' : '') + '11px monospace';
    c.fillStyle = isActive ? '#7b68ee' : '#888';
    c.textAlign = 'center';
    c.fillText(tabs[ti].label, tx + tabW / 2, tabH / 2 + 4);
  }

  var drawArea = { x: 0, y: tabH + 4, w: W, h: H - tabH - 4 };

  if (multiViewTab === 'grids') {
    drawMultiSideBySide(c, cv, drawArea);
  } else if (multiViewTab === 'dims') {
    drawMultiDimComparison(c, drawArea);
  } else if (multiViewTab === 'heatmap') {
    drawMultiVarianceHeatmap(c, drawArea);
  } else if (multiViewTab === 'pairwise') {
    drawMultiPairwiseMatrix(c, drawArea);
  } else if (multiViewTab === 'profile') {
    drawMultiLayerProfile_Main(c, drawArea);
  }

  // Update Dim Z visibility based on sub-view
  var dzRow = document.getElementById('dz-row');
  if (dzRow) {
    dzRow.style.display = (multiSubView === '3d' || multiSubView === 'fibre3d') ? 'flex' : 'none';
  }
}

/**
 * The key function: renders each sentence side-by-side using the
 * CURRENT view mode (2D, 3D, Fibre, Fibre3D, FibreKelp).
 *
 * It works by temporarily swapping the global D to each sentence's
 * data, setting up a clip region and transform for each panel,
 * then calling the appropriate existing draw function.
 */
function drawMultiSideBySide(c, cv, area) {
  if (!multiData || !multiData.sentence_data || multiData.sentence_data.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No sentence data — re-run Multi Compare', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var nSent = multiData.sentence_data.length;
  var sentColors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  // Determine the sub-view mode to use for each panel
  // Use a selector or default to the last non-multi viewMode
  var subView = multiSubView || '2d';

  // Draw sub-view selector bar just below the tab bar
  var subViews = [
    { id: '2d', label: '2D' },
    { id: '3d', label: '3D' },
    { id: 'fibre', label: 'Fibre' },
    { id: 'fibre3d', label: 'Fibre 3D' },
    { id: 'fibrekelp', label: 'Kelp' },
  ];
  var svBarH = 22;
  var svBarW = area.w / subViews.length;
  for (var svi = 0; svi < subViews.length; svi++) {
    var svx = area.x + svi * svBarW;
    var svy = area.y;
    var isAct = (subViews[svi].id === subView);
    c.fillStyle = isAct ? '#0f3460' : '#0a0a1a';
    c.fillRect(svx, svy, svBarW, svBarH);
    c.strokeStyle = isAct ? '#53a8b6' : '#0f3460';
    c.lineWidth = isAct ? 1.5 : 0.5;
    c.strokeRect(svx, svy, svBarW, svBarH);
    c.font = (isAct ? 'bold ' : '') + '10px monospace';
    c.fillStyle = isAct ? '#53a8b6' : '#666';
    c.textAlign = 'center';
    c.fillText(subViews[svi].label, svx + svBarW / 2, svy + svBarH / 2 + 3);
  }

  // Adjust area below the sub-view bar
  var panelArea = {
    x: area.x,
    y: area.y + svBarH + 2,
    w: area.w,
    h: area.h - svBarH - 2 - 30 // leave room for labels
  };

  // Layout: divide into nSent columns
  var gap = 4;
  var panelW = Math.floor((panelArea.w - gap * (nSent - 1)) / nSent);
  var panelH = panelArea.h;

  // Save the real global state
  var savedD = D;
  var savedViewMode = viewMode;
  var savedZoom = zoomLevel;
  var savedPanX = panX;
  var savedPanY = panY;
  var savedRotX = rotX;
  var savedRotY = rotY;
  var savedFibreRotX = fibreState.rotX;
  var savedFibreRotY = fibreState.rotY;

  for (var si = 0; si < nSent; si++) {
    var sentD = multiData.sentence_data[si];
    if (!sentD || !sentD.fixed_pos || !sentD.deltas) continue;

    var px = panelArea.x + si * (panelW + gap);
    var py = panelArea.y;

    // Draw panel border
    c.strokeStyle = sentColors[si % sentColors.length];
    c.lineWidth = 1.5;
    c.strokeRect(px, py, panelW, panelH);

    // Set up clipping and transform so the existing draw functions
    // render into this panel
    c.save();
    c.beginPath();
    c.rect(px, py, panelW, panelH);
    c.clip();

    // Temporarily swap global state
    D = sentD;
    viewMode = subView;
    zoomLevel = 1.0;
    panX = 0;
    panY = 0;

    // Resize the canvas dimensions that the draw functions read
    // We do this by overriding cv.width/cv.height temporarily
    // But that would clear the canvas! Instead, we use a transform.
    //
    // The draw functions use cv.width and cv.height for layout.
    // We need to make them think the canvas is panelW x panelH.
    // We'll translate so (0,0) maps to (px, py) and scale if needed.

    // For 2D: draw2D uses c.save/translate/scale internally for pan/zoom.
    // We need to intercept that. The simplest approach: create an offscreen
    // canvas for each panel, draw into it, then composite.

    var offCv = document.createElement('canvas');
    offCv.width = panelW;
    offCv.height = panelH;
    var offCtx = offCv.getContext('2d');

    // The draw functions read from document.getElementById('cv')
    // We can't easily redirect that. Instead, we'll use our
    // drawMiniDeformedGrid for 2D, and build mini versions for others.

    if (subView === '2d') {
      drawMiniPanel2D(offCtx, sentD, panelW, panelH);
    } else if (subView === '3d') {
      drawMiniPanel3D(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibre') {
      drawMiniPanelFibre(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibre3d') {
      drawMiniPanelFibre3D(offCtx, sentD, panelW, panelH);
    } else if (subView === 'fibrekelp') {
      drawMiniPanelFibreKelp(offCtx, sentD, panelW, panelH);
    }

    // Composite the offscreen canvas into the main canvas
    c.drawImage(offCv, px, py);

    c.restore(); // restore clipping

    // Sentence label below the panel
    c.font = 'bold 9px monospace';
    c.fillStyle = sentColors[si % sentColors.length];
    c.textAlign = 'center';
    var labelText = '[' + si + '] ' + (sentD.text || '').substring(0, Math.floor(panelW / 5.5));
    if ((sentD.text || '').length > Math.floor(panelW / 5.5)) labelText += '…';
    c.fillText(labelText, px + panelW / 2, py + panelH + 14);

    c.font = '8px monospace';
    c.fillStyle = '#666';
    c.fillText(sentD.n_real + ' tok | ' + sentD.n_layers + ' layers',
      px + panelW / 2, py + panelH + 24);
  }

  // Restore global state
  D = savedD;
  viewMode = savedViewMode;
  zoomLevel = savedZoom;
  panX = savedPanX;
  panY = savedPanY;
  rotX = savedRotX;
  rotY = savedRotY;
  fibreState.rotX = savedFibreRotX;
  fibreState.rotY = savedFibreRotY;

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.4)';
  c.textAlign = 'left';
  var p = gp();
  c.fillText(
    'Sub-view: ' + subView.toUpperCase() +
    '  Layer ' + p.layer + '/' + (multiData.sentence_data[0].n_layers - 1) +
    '  t=' + p.t.toFixed(2) +
    '  amp=' + p.amp.toFixed(1) +
    '  Dims:' + p.dx + ',' + p.dy +
    (subView === '3d' || subView === 'fibre3d' ? ',' + p.dz : '') +
    '  Mode:' + p.mode +
    '  |  All sidebar controls affect all panels',
    area.x + 10, area.y + area.h - 2
  );
}

// Global: which sub-view to use in multi mode
var multiSubView = '2d';

// ============================================================
// MINI PANEL RENDERERS
// Each takes an offscreen context, sentence data, and dimensions,
// and renders the full visualization into that context.
// ============================================================

/**
 * Mini 2D panel — reuses drawMiniDeformedGrid from before
 */
function drawMiniPanel2D(c, sentD, W, H) {
  var p = gp();
  // drawMiniDeformedGrid expects (c, sentD, p, px, py, pw, ph)
  drawMiniDeformedGrid(c, sentD, p, 0, 0, W, H);
}

/**
 * Mini 3D panel — renders a 3D deformed grid for one sentence
 */
function drawMiniPanel3D(c, sentD, W, H) {
  var p = gp();
  var nP = sentD.n_points, nR = sentD.n_real;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var dz = Math.min(p.dz, sentD.hidden_dim - 1);
  var isEmb = (p.mode === 'embedding');

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var layer = Math.min(p.layer, sentD.n_layers - 1);
  var amp = p.amp, t = p.t;

  var fx = new Float64Array(nP), fy = new Float64Array(nP), fz = new Float64Array(nP);
  for (var i = 0; i < nP; i++) {
    fx[i] = sentD.fixed_pos[i][dx];
    fy[i] = sentD.fixed_pos[i][dy];
    fz[i] = sentD.fixed_pos[i][dz];
  }

  var edx = new Float64Array(nP), edy = new Float64Array(nP), edz = new Float64Array(nP);
  if (!isEmb) {
    for (var j = 0; j < nP; j++) {
      var sx = 0, sy = 0, sz = 0;
      if (p.mode === 'single') {
        sx = activeDeltas[layer][j][dx]; sy = activeDeltas[layer][j][dy]; sz = activeDeltas[layer][j][dz];
      } else if (p.mode === 'cumfwd') {
        for (var l = 0; l <= layer; l++) { sx += activeDeltas[l][j][dx]; sy += activeDeltas[l][j][dy]; sz += activeDeltas[l][j][dz]; }
      } else {
        for (var l = layer; l < sentD.n_layers; l++) { sx += activeDeltas[l][j][dx]; sy += activeDeltas[l][j][dy]; sz += activeDeltas[l][j][dz]; }
      }
      edx[j] = sx * amp; edy[j] = sy * amp; edz[j] = sz * amp;
    }
  }

  // Compute bounds
  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity, mnz = Infinity, mxz = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
    if (fz[i] < mnz) mnz = fz[i]; if (fz[i] > mxz) mxz = fz[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny, mxz - mnz) || 1;
  var cx3 = (mnx + mxx) / 2, cy3 = (mny + mxy) / 2, cz3 = (mnz + mxz) / 2;
  var sc3 = Math.min(W, H) * 0.3 / mr;

  var fl = 400;
  function proj(x, y, z) {
    var cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    var x1 = x * cosY + z * sinY, z1 = -x * sinY + z * cosY;
    var cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    var y1 = y * cosX - z1 * sinX, z2 = y * sinX + z1 * cosX;
    var scale = fl / (fl + z2);
    return [W / 2 + x1 * scale, H / 2 + y1 * scale, z2, scale];
  }

  // Build 3D grid
  var N = Math.max(4, Math.min(12, Math.floor(p.gr / 4)));
  var pd = 0.12;
  var vx0 = cx3 - mr * (0.5 + pd), vx1 = cx3 + mr * (0.5 + pd);
  var vy0 = cy3 - mr * (0.5 + pd), vy1 = cy3 + mr * (0.5 + pd);
  var vz0 = cz3 - mr * (0.5 + pd), vz1 = cz3 + mr * (0.5 + pd);

  function gIdx(ix, iy, iz) { return iz * (N + 1) * (N + 1) + iy * (N + 1) + ix; }
  var nV = (N + 1) * (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV), oZ = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV), gZ = new Float64Array(nV);

  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) {
    var gi = gIdx(ix, iy, iz);
    oX[gi] = vx0 + (ix / N) * (vx1 - vx0);
    oY[gi] = vy0 + (iy / N) * (vy1 - vy0);
    oZ[gi] = vz0 + (iz / N) * (vz1 - vz0);
  }

  var sig = p.sig, s2i = 1 / (2 * sig * sig);
  if (isEmb) {
    for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; gZ[gi] = oZ[gi]; }
  } else {
    for (var gi = 0; gi < nV; gi++) {
      var gpx = oX[gi], gpy = oY[gi], gpz = oZ[gi];
      var vvx = 0, vvy = 0, vvz = 0, ws = 0;
      for (var k = 0; k < nP; k++) {
        var ex = gpx - fx[k], ey = gpy - fy[k], ez = gpz - fz[k];
        var w = Math.exp(Math.max(-500, -(ex * ex + ey * ey + ez * ez) * s2i));
        vvx += w * edx[k]; vvy += w * edy[k]; vvz += w * edz[k]; ws += w;
      }
      if (ws > 1e-15) { vvx /= ws; vvy /= ws; vvz /= ws; }
      gX[gi] = gpx + t * vvx; gY[gi] = gpy + t * vvy; gZ[gi] = gpz + t * vvz;
    }
  }

  // Collect edges with strain
  var edges = [];
  function addEdge(a, b) {
    var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a], oZ[b] - oZ[a]);
    var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a], gZ[b] - gZ[a]);
    var strain = od > 1e-12 ? dd / od : 1;
    var pa = proj((gX[a] - cx3) * sc3, (gY[a] - cy3) * sc3, (gZ[a] - cz3) * sc3);
    var pb = proj((gX[b] - cx3) * sc3, (gY[b] - cy3) * sc3, (gZ[b] - cz3) * sc3);
    edges.push({ x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1], z: (pa[2] + pb[2]) / 2, strain: strain });
  }

  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix < N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix + 1, iy, iz));
  for (var iz = 0; iz <= N; iz++) for (var iy = 0; iy < N; iy++) for (var ix = 0; ix <= N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix, iy + 1, iz));
  for (var iz = 0; iz < N; iz++) for (var iy = 0; iy <= N; iy++) for (var ix = 0; ix <= N; ix++) addEdge(gIdx(ix, iy, iz), gIdx(ix, iy, iz + 1));

  edges.sort(function(a, b) { return b.z - a.z; });

  // Draw edges
  if (p.grid && !isEmb) {
    c.lineWidth = 0.6;
    for (var ei = 0; ei < edges.length; ei++) {
      var e = edges[ei];
      var da = Math.max(0.1, Math.min(0.7, 0.5 - e.z * 0.001));
      if (p.sc) {
        var ec = s2c(e.strain);
        c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',' + da.toFixed(2) + ')';
      } else {
        c.strokeStyle = 'rgba(200,200,200,' + da.toFixed(2) + ')';
      }
      c.beginPath(); c.moveTo(e.x1, e.y1); c.lineTo(e.x2, e.y2); c.stroke();
    }
  }

  // Draw token dots
  if (p.tok) {
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];
    var pts = [];
    for (var ti = 0; ti < nR; ti++) {
      var pp = proj((fx[ti] - cx3) * sc3, (fy[ti] - cy3) * sc3, (fz[ti] - cz3) * sc3);
      pts.push({ idx: ti, x: pp[0], y: pp[1], z: pp[2], s: pp[3] });
    }
    pts.sort(function(a, b) { return b.z - a.z; });
    for (var pi = 0; pi < pts.length; pi++) {
      var pt = pts[pi];
      var r = Math.max(2, 4 * pt.s);
      c.beginPath(); c.arc(pt.x, pt.y, r, 0, Math.PI * 2);
      c.fillStyle = tc[pt.idx % tc.length]; c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 0.8; c.stroke();
      c.font = 'bold 7px monospace';
      c.lineWidth = 1.5; c.strokeStyle = 'rgba(0,0,0,0.9)';
      var lb = '[' + pt.idx + '] ' + sentD.tokens[pt.idx];
      if (lb.length > Math.floor(W / 7)) lb = lb.substring(0, Math.floor(W / 7)) + '…';
      c.strokeText(lb, pt.x + 5, pt.y - 4);
      c.fillStyle = '#fff'; c.fillText(lb, pt.x + 5, pt.y - 4);
    }
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('3D L' + layer + ' d' + dx + ',' + dy + ',' + dz, 4, 10);
}

/**
 * Mini Fibre Bundle panel — renders the token-per-column, layer-per-row grid view
 */
function drawMiniPanelFibre(c, sentD, W, H) {
  var p = gp();
  var nR = sentD.n_real;
  var nLayers = sentD.n_layers;
  var nP = sentD.n_points;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var mode = p.mode;
  var isEmb = (mode === 'embedding');
  var layer = Math.min(p.layer, nLayers - 1);
  var amp = p.amp, t = p.t, sig = p.sig;

  // Read display toggles from the checkboxes
  var showGrid = document.getElementById('cb-grid').checked;
  var showHeat = document.getElementById('cb-heat').checked;
  var showSC = document.getElementById('cb-sc').checked;

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) { fx[i] = sentD.fixed_pos[i][dx]; fy[i] = sentD.fixed_pos[i][dy]; }

  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd = 0.15;
  var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
  var vw = mr * (1 + 2 * pd), vh = vw;

  // Layout: rooms
  var margin = 4;
  var labelW = 18;
  var availW = W - 2 * margin - labelW;
  var availH = H - 2 * margin;
  var roomSize = Math.max(12, Math.min(
    Math.floor(availW / (nR * 1.3)),
    Math.floor(availH / (nLayers * 1.4))
  ));
  var gapX = Math.max(2, Math.floor(roomSize * 0.2));
  var gapY = Math.max(3, Math.floor(roomSize * 0.25));
  var startX = margin + labelW;
  var startY = margin;

  var N = Math.max(3, Math.min(8, Math.floor(roomSize / 5)));
  var itpM = document.getElementById('sel-itp').value;

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  for (var li = 0; li < nLayers; li++) {
    var rowIdx = nLayers - 1 - li;
    var roomCY = startY + rowIdx * (roomSize + gapY);
    var isCurrent = (li === layer);

    // Layer label
    c.font = (isCurrent ? 'bold ' : '') + '6px monospace';
    c.fillStyle = isCurrent ? '#e94560' : '#555';
    c.textAlign = 'right';
    c.fillText('L' + li, startX - 2, roomCY + roomSize / 2 + 3);

    var layerDeltas = computeEdxEdyForLayerMini(sentD, li, nP, dx, dy, amp, mode, activeDeltas, nLayers);
    var edxCum = layerDeltas.edx;
    var edyCum = layerDeltas.edy;

    for (var ti = 0; ti < nR; ti++) {
      var roomCX = startX + ti * (roomSize + gapX);

      // Room background
      var bgAlpha = isCurrent ? 0.15 : 0.06;
      c.fillStyle = 'rgba(30,30,60,' + bgAlpha + ')';
      c.fillRect(roomCX, roomCY, roomSize, roomSize);

      // Room border
      c.strokeStyle = isCurrent ? 'rgba(233,69,96,0.6)' : 'rgba(60,60,100,0.25)';
      c.lineWidth = isCurrent ? 1.5 : 0.5;
      c.strokeRect(roomCX, roomCY, roomSize, roomSize);

      // Build deformed grid for this room
      var grid = buildDeformedGridMini(vx0, vy0, vw, vh, N, fx, fy, edxCum, edyCum, nP, sig, t, isEmb, itpM);

      // Strain heatmap
      if (showHeat && !isEmb) {
        for (var hy = 0; hy < N; hy++) {
          for (var hx = 0; hx < N; hx++) {
            var avg = (grid.sH[hy * N + hx] + grid.sH[(hy + 1) * N + hx] +
                       grid.sV[hy * (N + 1) + hx] + grid.sV[hy * (N + 1) + hx + 1]) / 4;
            var co = s2c(avg);
            var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
            var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[i00] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i00] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i10] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i10] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i11] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i11] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[i01] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[i01] - vy0) / vh) * roomSize);
            c.closePath();
            c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.4)';
            c.fill();
          }
        }
      }

      // Grid lines
      if (showGrid && !isEmb) {
        c.lineWidth = 0.5;
        for (var dhy = 0; dhy <= N; dhy++) {
          for (var dhx = 0; dhx < N; dhx++) {
            var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
            var es = grid.sH[dhy * N + dhx];
            if (showSC) {
              var ec = s2c(es);
              c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.8)';
            } else {
              c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[di1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[di2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[di2] - vy0) / vh) * roomSize);
            c.stroke();
          }
        }
        for (var dvx = 0; dvx <= N; dvx++) {
          for (var dvy = 0; dvy < N; dvy++) {
            var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
            var vs = grid.sV[dvy * (N + 1) + dvx];
            if (showSC) {
              var vc = s2c(vs);
              c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.8)';
            } else {
              c.strokeStyle = 'rgba(200,200,200,0.4)';
            }
            c.beginPath();
            c.moveTo(roomCX + ((grid.gX[dvi1] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi1] - vy0) / vh) * roomSize);
            c.lineTo(roomCX + ((grid.gX[dvi2] - vx0) / vw) * roomSize,
                     roomCY + ((grid.gY[dvi2] - vy0) / vh) * roomSize);
            c.stroke();
          }
        }
      }

      // Reference grid in embedding mode
      if (isEmb) {
        c.strokeStyle = 'rgba(255,255,255,0.12)';
        c.lineWidth = 0.3;
        for (var ry = 0; ry <= N; ry++) {
          c.beginPath();
          for (var rx = 0; rx <= N; rx++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (rx === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
          }
          c.stroke();
        }
        for (var rx = 0; rx <= N; rx++) {
          c.beginPath();
          for (var ry = 0; ry <= N; ry++) {
            var ri = ry * (N + 1) + rx;
            var rsx = roomCX + ((grid.oX[ri] - vx0) / vw) * roomSize;
            var rsy = roomCY + ((grid.oY[ri] - vy0) / vh) * roomSize;
            if (ry === 0) c.moveTo(rsx, rsy); else c.lineTo(rsx, rsy);
          }
          c.stroke();
        }
      }

      // Token dot
      var tokX = roomCX + ((fx[ti] - vx0) / vw) * roomSize;
      var tokY = roomCY + ((fy[ti] - vy0) / vh) * roomSize;
      c.beginPath();
      c.arc(tokX, tokY, Math.max(2, roomSize / 20), 0, Math.PI * 2);
      c.fillStyle = tc[ti % tc.length];
      c.fill();

      // Token label at bottom row only
      if (li === 0 && roomSize > 25) {
        c.font = 'bold 6px monospace';
        c.fillStyle = tc[ti % tc.length];
        c.textAlign = 'center';
        c.fillText('[' + ti + ']', roomCX + roomSize / 2, roomCY + roomSize + 8);
      }
    }
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('FIBRE L' + layer + ' d' + dx + ',' + dy, 4, 10);
}

/**
 * Helper: compute mode-aware cumulative deltas for a given layer
 * using a specific sentence's data (not the global D).
 */
function computeEdxEdyForLayerMini(sentD, li, nP, dx, dy, amp, mode, activeDeltas, nLayers) {
  var edxCum = new Float64Array(nP);
  var edyCum = new Float64Array(nP);
  var isEmb = (mode === 'embedding');
  if (isEmb) return { edx: edxCum, edy: edyCum };

  if (mode === 'single') {
    for (var j = 0; j < nP; j++) {
      edxCum[j] = activeDeltas[li][j][dx] * amp;
      edyCum[j] = activeDeltas[li][j][dy] * amp;
    }
  } else if (mode === 'cumfwd') {
    for (var cl = 0; cl <= li; cl++) {
      for (var j = 0; j < nP; j++) {
        edxCum[j] += activeDeltas[cl][j][dx] * amp;
        edyCum[j] += activeDeltas[cl][j][dy] * amp;
      }
    }
  } else { // cumbwd
    for (var cl = li; cl < nLayers; cl++) {
      for (var j = 0; j < nP; j++) {
        edxCum[j] += activeDeltas[cl][j][dx] * amp;
        edyCum[j] += activeDeltas[cl][j][dy] * amp;
      }
    }
  }
  return { edx: edxCum, edy: edyCum };
}

/**
 * Helper: build a deformed grid with strain computation.
 * Reusable across all mini panel renderers.
 */
function buildDeformedGridMini(vx0, vy0, vw, vh, N, fx, fy, edxCum, edyCum, nP, sig, t, isEmb, itpMethod) {
  var nV = (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV);

  for (var gy = 0; gy <= N; gy++) {
    for (var gx = 0; gx <= N; gx++) {
      var gi = gy * (N + 1) + gx;
      oX[gi] = vx0 + (gx / N) * vw;
      oY[gi] = vy0 + (gy / N) * vh;
    }
  }

  if (isEmb) {
    for (var gi = 0; gi < nV; gi++) { gX[gi] = oX[gi]; gY[gi] = oY[gi]; }
  } else {
    for (var gi = 0; gi < nV; gi++) {
      var px = oX[gi], py = oY[gi];
      var iRes = interpolateGridPoint(px, py, fx, fy, edxCum, edyCum, nP, sig, itpMethod);
      gX[gi] = px + t * iRes[0];
      gY[gi] = py + t * iRes[1];
    }
  }

  var sH = new Float64Array(N * (N + 1));
  var sV = new Float64Array((N + 1) * N);
  for (var ey = 0; ey <= N; ey++) {
    for (var ex = 0; ex < N; ex++) {
      var a = ey * (N + 1) + ex, b = a + 1;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sH[ey * N + ex] = od > 1e-12 ? dd / od : 1;
    }
  }
  for (var ey = 0; ey < N; ey++) {
    for (var ex = 0; ex <= N; ex++) {
      var a = ey * (N + 1) + ex, b = (ey + 1) * (N + 1) + ex;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sV[ey * (N + 1) + ex] = od > 1e-12 ? dd / od : 1;
    }
  }

  return { oX: oX, oY: oY, gX: gX, gY: gY, sH: sH, sV: sV, nV: nV };
}

/**
 * Mini Fibre 3D panel — renders the 3D fibre bundle grid for one sentence
 */
function drawMiniPanelFibre3D(c, sentD, W, H) {
  // Reuse drawMiniPanel3D but with fibre-style layout
  // For simplicity in the multi-view, render as 3D with the fibre room structure
  drawMiniPanel3D(c, sentD, W, H);
}

/**
 * Mini Fibre Kelp panel — renders the kelp-style pathline view for one sentence
 */
function drawMiniPanelFibreKelp(c, sentD, W, H) {
  var p = gp();
  var nR = sentD.n_real;
  var nLayers = sentD.n_layers;
  var nP = sentD.n_points;
  var dx = Math.min(p.dx, sentD.hidden_dim - 1);
  var dy = Math.min(p.dy, sentD.hidden_dim - 1);
  var isEmb = (p.mode === 'embedding');
  var layer = Math.min(p.layer, nLayers - 1);
  var amp = p.amp, t = p.t;
  var mode = p.mode;

  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  var attnDeltas = sentD.attn_deltas || null;
  var mlpDeltas = sentD.mlp_deltas || null;

  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) { fx[i] = sentD.fixed_pos[i][dx]; fy[i] = sentD.fixed_pos[i][dy]; }

  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i = 0; i < nR; i++) {
    if (fx[i] < mnx) mnx = fx[i]; if (fx[i] > mxx) mxx = fx[i];
    if (fy[i] < mny) mny = fy[i]; if (fy[i] > mxy) mxy = fy[i];
  }
  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd = 0.15;
  var vx0 = cxv - mr * (0.5 + pd), vy0 = cyv - mr * (0.5 + pd);
  var vw = mr * (1 + 2 * pd);

  // Layout
  var margin = 8;
  var plotW = W - 2 * margin;
  var plotH = H - 2 * margin;
  var layerH = plotH / nLayers;

  function SX(wx) { return margin + ((wx - vx0) / vw) * plotW; }
  function LY(li) { return margin + (nLayers - 1 - li) * layerH + layerH * 0.5; }

  var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
            '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];

  // Draw token pathlines
  for (var ti = 0; ti < nR; ti++) {
    var col = tc[ti % tc.length];
    var path = [];

    for (var li = 0; li < nLayers; li++) {
      var cumDx = 0, cumDy = 0;
      if (!isEmb) {
        if (mode === 'single') {
          cumDx = activeDeltas[li][ti][dx] * amp * t;
          cumDy = activeDeltas[li][ti][dy] * amp * t;
        } else if (mode === 'cumfwd') {
          for (var cl = 0; cl <= li; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
          }
        } else {
          for (var cl = li; cl < nLayers; cl++) {
            cumDx += activeDeltas[cl][ti][dx] * amp * t;
            cumDy += activeDeltas[cl][ti][dy] * amp * t;
          }
        }
      }
      path.push({ x: SX(fx[ti] + cumDx), y: LY(li) });
    }

    // Draw pathline
    c.strokeStyle = col;
    c.lineWidth = 1.5;
    c.beginPath();
    c.moveTo(path[0].x, path[0].y);
    for (var li = 1; li < nLayers; li++) {
      var prev = path[li - 1], curr = path[li];
      c.quadraticCurveTo(prev.x, (prev.y + curr.y) / 2, curr.x, curr.y);
    }
    c.stroke();

    // Attn/MLP arrows at current layer
    if (!isEmb) {
      var pt = path[layer];
      var pixPerWorld = plotW / vw;
      var maxArrow = layerH * 0.3;

      if (attnDeltas) {
        var avx = attnDeltas[layer][ti][dx] * amp * t * pixPerWorld;
        if (Math.abs(avx) > 1.5) {
          if (Math.abs(avx) > maxArrow) avx = maxArrow * Math.sign(avx);
          c.strokeStyle = 'rgba(0,200,255,0.6)';
          c.lineWidth = 0.8;
          c.beginPath(); c.moveTo(pt.x, pt.y - 2); c.lineTo(pt.x + avx, pt.y - 2); c.stroke();
        }
      }
      if (mlpDeltas) {
        var mvx = mlpDeltas[layer][ti][dx] * amp * t * pixPerWorld;
        if (Math.abs(mvx) > 1.5) {
          if (Math.abs(mvx) > maxArrow) mvx = maxArrow * Math.sign(mvx);
          c.strokeStyle = 'rgba(255,165,0,0.6)';
          c.lineWidth = 0.8;
          c.beginPath(); c.moveTo(pt.x, pt.y + 2); c.lineTo(pt.x + mvx, pt.y + 2); c.stroke();
        }
      }
    }

    // Node dots
    for (var li = 0; li < nLayers; li++) {
      var isActive = (li === layer);
      c.beginPath();
      c.arc(path[li].x, path[li].y, isActive ? 3 : 1.5, 0, Math.PI * 2);
      c.fillStyle = col;
      c.fill();
    }

    // Token label at bottom
    c.font = 'bold 6px monospace';
    c.fillStyle = col;
    c.textAlign = 'center';
    c.fillText('[' + ti + '] ' + sentD.tokens[ti], path[0].x, path[0].y + 10);
  }

  // Layer labels
  for (var li = 0; li < nLayers; li++) {
    var isActive = (li === layer);
    c.font = (isActive ? 'bold ' : '') + '6px monospace';
    c.fillStyle = isActive ? '#e94560' : '#444';
    c.textAlign = 'right';
    c.fillText('L' + li, margin - 2, LY(li) + 2);
  }

  // HUD
  c.font = '7px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText('KELP L' + layer + ' d' + dx + ',' + dy, 4, 10);
}

// ---- Tab click handler ----
// Sub-view click handler (inside the grids tab)
document.getElementById('cv').addEventListener('click', function(e) {
  if (viewMode !== 'multi' || !multiData) return;
  var cv = document.getElementById('cv');
  var rect = cv.getBoundingClientRect();
  var mx = e.clientX - rect.left;
  var my = e.clientY - rect.top;

  var tabH = 28;

  // ---- MAIN TAB BAR click (the 5 tabs at the very top) ----
  if (my >= 0 && my < tabH) {
    var tabs = ['grids', 'dims', 'heatmap', 'pairwise', 'profile'];
    var tabW = cv.width / tabs.length;
    var idx = Math.floor(mx / tabW);
    if (idx >= 0 && idx < tabs.length) {
      multiViewTab = tabs[idx];
      drawMultiCanvas();
    }
    return;
  }

  // ---- SUB-VIEW BAR click (only visible in 'grids' tab) ----
  if (multiViewTab === 'grids') {
    var svBarY = tabH + 4;
    var svBarH = 22;
    if (my >= svBarY && my < svBarY + svBarH) {
      var subViews = ['2d', '3d', 'fibre', 'fibre3d', 'fibrekelp'];
      var svBarW = cv.width / subViews.length;
      var idx = Math.floor(mx / svBarW);
      if (idx >= 0 && idx < subViews.length) {
        multiSubView = subViews[idx];
        drawMultiCanvas();
      }
    }
  }
});

// ============================================================
// TAB 1: DIMENSION COMPARISON (the main one you want)
// ============================================================
function drawMultiDimComparison(c, area) {
  var layerIdx = +document.getElementById('multi-layer').value;
  var showMode = document.getElementById('multi-show').value;
  var topK = +document.getElementById('multi-topk').value;
  var lc = multiData.layer_comparisons[layerIdx];

  var dims = (showMode === 'most') ? lc.most_different_dims : lc.least_different_dims;
  dims = dims.slice(0, topK);
  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  var margin = { left: 70, right: 180, top: 50, bottom: 60 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#7b68ee';
  c.textAlign = 'center';
  c.fillText(
    (showMode === 'most' ? 'Most' : 'Least') + ' Different Dimensions — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    area.x + area.w / 2, oy - 20
  );

  if (dims.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.fillText('No dimension data available', area.x + area.w / 2, oy + plotH / 2);
    return;
  }

  // Find global value range
  var globalMin = Infinity, globalMax = -Infinity;
  for (var di = 0; di < dims.length; di++) {
    for (var si = 0; si < nSent; si++) {
      var v = dims[di].values[si];
      if (v < globalMin) globalMin = v;
      if (v > globalMax) globalMax = v;
    }
  }
  var valRange = globalMax - globalMin || 1;

  // Compute bar layout
  var barGroupH = Math.max(12, Math.floor(plotH / dims.length) - 3);
  var barH = Math.max(3, Math.floor((barGroupH - 2) / nSent) - 1);

  // Zero line position
  var zeroFrac = (0 - globalMin) / valRange;
  var zeroX = ox + zeroFrac * plotW;

  // Draw zero line
  c.strokeStyle = 'rgba(255,255,255,0.15)';
  c.lineWidth = 1;
  c.setLineDash([4, 4]);
  c.beginPath();
  c.moveTo(zeroX, oy);
  c.lineTo(zeroX, oy + plotH);
  c.stroke();
  c.setLineDash([]);

  // Draw axis labels
  c.font = '9px monospace';
  c.fillStyle = '#666';
  c.textAlign = 'center';
  // Min label
  c.fillText(globalMin.toFixed(3), ox, oy + plotH + 20);
  // Max label
  c.fillText(globalMax.toFixed(3), ox + plotW, oy + plotH + 20);
  // Zero label
  if (zeroFrac > 0.05 && zeroFrac < 0.95) {
    c.fillText('0', zeroX, oy + plotH + 20);
  }

  // Draw each dimension row
  for (var di = 0; di < dims.length; di++) {
    var d = dims[di];
    var y0 = oy + di * (barGroupH + 3);

    // Alternating row background
    if (di % 2 === 0) {
      c.fillStyle = 'rgba(255,255,255,0.02)';
      c.fillRect(ox - 5, y0 - 1, plotW + 10, barGroupH + 2);
    }

    // Dim label (left side)
    c.font = 'bold 10px monospace';
    c.fillStyle = '#a0a0c0';
    c.textAlign = 'right';
    c.fillText('dim ' + d.dim, ox - 10, y0 + barGroupH / 2 + 4);

    // One bar per sentence
    for (var si = 0; si < nSent; si++) {
      var v = d.values[si];
      var barX = ox + ((v - globalMin) / valRange) * plotW;

      var by = y0 + si * (barH + 1);
      var bw = barX - zeroX;

      // Bar with gradient
      var col = colors[si % colors.length];
      c.fillStyle = col;
      c.globalAlpha = 0.85;
      if (bw >= 0) {
        c.fillRect(zeroX, by, bw, barH);
      } else {
        c.fillRect(barX, by, -bw, barH);
      }
      c.globalAlpha = 1.0;

      // Value label at end of bar
      if (barGroupH > 15 && barH >= 5) {
        c.font = '7px monospace';
        c.fillStyle = '#888';
        c.textAlign = bw >= 0 ? 'left' : 'right';
        var labelX = bw >= 0 ? barX + 3 : barX - 3;
        c.fillText(v.toFixed(3), labelX, by + barH - 1);
      }
    }

    // Variance indicator on the right
    var maxVar = dims[0].variance || 1;
    var varBarW = Math.min(100, Math.max(4, (d.variance / maxVar) * 100));
    var varX = ox + plotW + 10;
    c.fillStyle = 'rgba(123,104,238,0.3)';
    c.fillRect(varX, y0, varBarW, barGroupH);
    c.font = '8px monospace';
    c.fillStyle = '#7b68ee';
    c.textAlign = 'left';
    c.fillText('σ²=' + d.variance.toExponential(1), varX + varBarW + 4, y0 + barGroupH / 2 + 3);

    // Range indicator
    c.fillStyle = '#555';
    c.fillText('Δ=' + d.range.toFixed(3), varX + varBarW + 4, y0 + barGroupH / 2 + 13);
  }

  // Legend (bottom right)
  var legX = ox + plotW + 10;
  var legY = oy + plotH - nSent * 16 - 10;
  c.font = 'bold 10px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('Sentences:', legX, legY - 6);

  for (var si = 0; si < nSent; si++) {
    var ly = legY + si * 16;
    c.fillStyle = colors[si % colors.length];
    c.fillRect(legX, ly, 14, 10);
    c.font = '9px monospace';
    c.fillStyle = '#a0a0c0';
    var sentText = multiData.sentences[si].text;
    if (sentText.length > 25) sentText = sentText.substring(0, 25) + '…';
    c.fillText('[' + si + '] ' + sentText, legX + 18, ly + 8);
  }

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText(
    dims.length + ' dimensions | ' + nSent + ' sentences | ' +
    'Use sidebar controls to change layer/mode/topK',
    area.x + 10, area.y + area.h - 5
  );
}

// ============================================================
// TAB 2: VARIANCE HEATMAP (dimensions × layers)
// ============================================================
function drawMultiVarianceHeatmap(c, area) {
  var nLayers = multiData.layer_comparisons.length;
  var topK = Math.min(+document.getElementById('multi-topk').value, 50);

  // Get the globally most different dimensions
  var globalDims = multiData.summary.global_most_different_dims.slice(0, topK);
  if (globalDims.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No data', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var margin = { left: 60, right: 30, top: 50, bottom: 50 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  var cellW = Math.max(4, Math.floor(plotW / nLayers));
  var cellH = Math.max(4, Math.floor(plotH / globalDims.length));

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#f5a623';
  c.textAlign = 'center';
  c.fillText('Cross-Sentence Variance Heatmap (Top ' + globalDims.length + ' Dims × Layers)',
    area.x + area.w / 2, oy - 20);

  // Find max variance for color scaling
  var maxVar = 0;
  for (var di = 0; di < globalDims.length; di++) {
    var dimIdx = globalDims[di].dim;
    for (var li = 0; li < nLayers; li++) {
      var v = multiData.layer_comparisons[li].dim_variance[dimIdx];
      if (v > maxVar) maxVar = v;
    }
  }
  if (maxVar < 1e-12) maxVar = 1;

  // Draw cells
  for (var di = 0; di < globalDims.length; di++) {
    var dimIdx = globalDims[di].dim;
    for (var li = 0; li < nLayers; li++) {
      var v = multiData.layer_comparisons[li].dim_variance[dimIdx];
      var intensity = Math.min(1, v / maxVar);

      // Color: black -> purple -> red -> yellow -> white
      var r, g, b;
      if (intensity < 0.25) {
        var t = intensity / 0.25;
        r = Math.floor(t * 80); g = 0; b = Math.floor(t * 120);
      } else if (intensity < 0.5) {
        var t = (intensity - 0.25) / 0.25;
        r = 80 + Math.floor(t * 153); g = Math.floor(t * 40); b = 120 - Math.floor(t * 24);
      } else if (intensity < 0.75) {
        var t = (intensity - 0.5) / 0.25;
        r = 233; g = 40 + Math.floor(t * 180); b = 96 - Math.floor(t * 96);
      } else {
        var t = (intensity - 0.75) / 0.25;
        r = 233 + Math.floor(t * 22); g = 220 + Math.floor(t * 35); b = Math.floor(t * 200);
      }

      c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      c.fillRect(ox + li * cellW, oy + di * cellH, cellW - 1, cellH - 1);
    }

    // Dim label
    c.font = '8px monospace';
    c.fillStyle = '#888';
    c.textAlign = 'right';
    c.fillText('d' + dimIdx, ox - 4, oy + di * cellH + cellH / 2 + 3);
  }

  // Layer labels
  c.textAlign = 'center';
  for (var li = 0; li < nLayers; li++) {
    if (nLayers <= 30 || li % 2 === 0) {
      c.font = '8px monospace';
      c.fillStyle = '#666';
      c.fillText(li === 0 ? 'E' : 'L' + (li - 1), ox + li * cellW + cellW / 2, oy + plotH + 14);
    }
  }

  // Axis labels
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText('Layer →', ox + plotW / 2, oy + plotH + 35);

  c.save();
  c.translate(ox - 45, oy + plotH / 2);
  c.rotate(-Math.PI / 2);
  c.fillText('Dimension (by global variance) →', 0, 0);
  c.restore();

  // Color legend
  var legW = 150, legH = 12;
  var legX = ox + plotW - legW;
  var legY = oy - 15;
  for (var i = 0; i < legW; i++) {
    var t = i / legW;
    var r2, g2, b2;
    if (t < 0.25) { var f = t / 0.25; r2 = Math.floor(f * 80); g2 = 0; b2 = Math.floor(f * 120); }
    else if (t < 0.5) { var f = (t - 0.25) / 0.25; r2 = 80 + Math.floor(f * 153); g2 = Math.floor(f * 40); b2 = 120 - Math.floor(f * 24); }
    else if (t < 0.75) { var f = (t - 0.5) / 0.25; r2 = 233; g2 = 40 + Math.floor(f * 180); b2 = 96 - Math.floor(f * 96); }
    else { var f = (t - 0.75) / 0.25; r2 = 233 + Math.floor(f * 22); g2 = 220 + Math.floor(f * 35); b2 = Math.floor(f * 200); }
    c.fillStyle = 'rgb(' + r2 + ',' + g2 + ',' + b2 + ')';
    c.fillRect(legX + i, legY, 1, legH);
  }
  c.font = '8px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('0', legX, legY - 2);
  c.textAlign = 'right';
  c.fillText(maxVar.toExponential(1), legX + legW, legY - 2);
  c.textAlign = 'center';
  c.fillText('variance', legX + legW / 2, legY + legH + 10);
}

// ============================================================
// TAB 3: PAIRWISE SIMILARITY MATRIX (large, readable)
// ============================================================
function drawMultiPairwiseMatrix(c, area) {
  var layerIdx = +document.getElementById('multi-layer').value;
  var lc = multiData.layer_comparisons[layerIdx];
  var nSent = multiData.n_sentences;
  var cosMatrix = lc.pairwise_cosine;
  var l2Matrix = lc.pairwise_l2;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  var margin = { left: 120, right: 40, top: 80, bottom: 120 };
  var plotSize = Math.min(area.w - margin.left - margin.right, area.h - margin.top - margin.bottom);
  var cellSize = Math.floor(plotSize / nSent);
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText(
    'Pairwise Cosine Similarity — ' +
    (layerIdx === 0 ? 'Embedding' : 'Layer ' + (layerIdx - 1)),
    area.x + area.w / 2, oy - 30
  );

  // Draw matrix cells
  for (var i = 0; i < nSent; i++) {
    for (var j = 0; j < nSent; j++) {
      var val = cosMatrix[i][j];
      // Color: -1 = deep blue, 0 = black, 1 = bright green
      var r, g, b;
      if (val < 0) {
        var t = Math.min(1, -val);
        r = 0; g = 0; b = Math.floor(40 + t * 180);
      } else {
        var t = Math.min(1, val);
        r = Math.floor(t * 20); g = Math.floor(40 + t * 180); b = Math.floor(t * 80);
      }
      c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      c.fillRect(ox + j * cellSize, oy + i * cellSize, cellSize - 2, cellSize - 2);

      // Value text
      c.font = (cellSize > 40 ? 'bold 12px' : cellSize > 25 ? '10px' : '8px') + ' monospace';
      c.fillStyle = val > 0.6 ? '#000' : '#fff';
      c.textAlign = 'center';
      c.fillText(val.toFixed(3),
        ox + j * cellSize + (cellSize - 2) / 2,
        oy + i * cellSize + (cellSize - 2) / 2 + 4);

      // L2 distance below
      if (cellSize > 50) {
        c.font = '8px monospace';
        c.fillStyle = val > 0.6 ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)';
        c.fillText('L2=' + l2Matrix[i][j].toFixed(1),
          ox + j * cellSize + (cellSize - 2) / 2,
          oy + i * cellSize + (cellSize - 2) / 2 + 16);
      }
    }

    // Row labels (left)
    c.font = '10px monospace';
    c.fillStyle = colors[i % colors.length];
    c.textAlign = 'right';
    var rowLabel = '[' + i + '] ' + multiData.sentences[i].text.substring(0, 15);
    c.fillText(rowLabel, ox - 8, oy + i * cellSize + cellSize / 2 + 3);

    // Column labels (bottom, rotated)
    c.save();
    c.translate(ox + i * cellSize + cellSize / 2, oy + nSent * cellSize + 8);
    c.rotate(Math.PI / 4);
    c.font = '9px monospace';
    c.fillStyle = colors[i % colors.length];
    c.textAlign = 'left';
    var colLabel = '[' + i + '] ' + multiData.sentences[i].text.substring(0, 20);
    c.fillText(colLabel, 0, 0);
    c.restore();
  }

  // Color legend
  var legX = ox + nSent * cellSize + 20;
  var legY = oy;
  var legH = Math.min(200, nSent * cellSize);
  for (var i = 0; i < legH; i++) {
    var val = 1.0 - (i / legH) * 2; // 1 at top, -1 at bottom
    var r, g, b;
    if (val < 0) { var t = -val; r = 0; g = 0; b = Math.floor(40 + t * 180); }
    else { var t = val; r = Math.floor(t * 20); g = Math.floor(40 + t * 180); b = Math.floor(t * 80); }
    c.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    c.fillRect(legX, legY + i, 15, 1);
  }
  c.font = '9px monospace';
  c.fillStyle = '#888';
  c.textAlign = 'left';
  c.fillText('1.0', legX + 20, legY + 4);
  c.fillText('0.0', legX + 20, legY + legH / 2 + 4);
  c.fillText('-1.0', legX + 20, legY + legH + 4);
}

// ============================================================
// TAB 4: LAYER DIVERGENCE PROFILE (large, with annotations)
// ============================================================
function drawMultiLayerProfile_Main(c, area) {
  var variances = multiData.summary.layer_total_variances;
  var nL = variances.length;
  var maxVar = Math.max.apply(null, variances) || 1;

  var margin = { left: 70, right: 40, top: 50, bottom: 50 };
  var plotW = area.w - margin.left - margin.right;
  var plotH = area.h - margin.top - margin.bottom;
  var ox = area.x + margin.left;
  var oy = area.y + margin.top;

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#e94560';
  c.textAlign = 'center';
  c.fillText('Layer-by-Layer Cross-Sentence Divergence', area.x + area.w / 2, oy - 20);

  var barW = Math.max(8, Math.floor(plotW / nL) - 1);
  var baseY = plotH + oy;

  // Draw bars and line
  var points = [];
  for (var i = 0; i < nL; i++) {
    var h = (variances[i] / maxVar) * plotH;
    var x = ox + i * (barW + 1);
    var y = baseY - h;
    points.push({ x: x + barW / 2, y: y });

    // Highlight most divergent layer
    var isMost = (i === multiData.summary.most_divergent_layer);
    var isLeast = (i === multiData.summary.least_divergent_layer);

    // Bar
    var frac = variances[i] / maxVar;
    if (isMost) {
      c.fillStyle = '#e94560';
    } else if (isLeast) {
      c.fillStyle = '#53a8b6';
    } else {
      c.fillStyle = 'rgb(' + Math.floor(frac * 200) + ',' +
                    Math.floor((1 - frac) * 120 + 50) + ',' +
                    Math.floor((1 - frac) * 180) + ')';
    }
    c.fillRect(x, y, barW, h);

    // Glow for most divergent
    if (isMost) {
      c.shadowColor = '#e94560';
      c.shadowBlur = 10;
      c.fillRect(x, y, barW, h);
      c.shadowBlur = 0;
    }

    // Layer label
    c.font = '9px monospace';
    c.fillStyle = isMost ? '#e94560' : isLeast ? '#53a8b6' : '#666';
    c.textAlign = 'center';
    c.fillText(i === 0 ? 'Emb' : 'L' + (i - 1), x + barW / 2, baseY + 16);

    // Value label on top of bar
    if (barW > 15 || nL <= 20) {
      c.font = '8px monospace';
      c.fillStyle = '#888';
      c.fillText(variances[i].toFixed(1), x + barW / 2, y - 4);
    }
  }

  // Connect with a line
  c.strokeStyle = 'rgba(123,104,238,0.5)';
  c.lineWidth = 1.5;
  c.beginPath();
  for (var i = 0; i < points.length; i++) {
    if (i === 0) c.moveTo(points[i].x, points[i].y);
    else c.lineTo(points[i].x, points[i].y);
  }
  c.stroke();

  // Dots on the line
  for (var i = 0; i < points.length; i++) {
    var isMost = (i === multiData.summary.most_divergent_layer);
    c.beginPath();
    c.arc(points[i].x, points[i].y, isMost ? 5 : 3, 0, Math.PI * 2);
    c.fillStyle = isMost ? '#e94560' : '#7b68ee';
    c.fill();
  }

  // Title
  c.font = 'bold 14px monospace';
  c.fillStyle = '#e94560';
  c.textAlign = 'center';
  c.fillText('Layer-by-Layer Cross-Sentence Divergence', area.x + area.w / 2, oy - 20);

  // Axis labels
  c.font = 'bold 10px monospace';
  c.fillStyle = '#53a8b6';
  c.textAlign = 'center';
  c.fillText('Layer →', ox + (nL * (barW + 1)) / 2, baseY + 35);

  c.save();
  c.translate(ox - 45, oy + plotH / 2);
  c.rotate(-Math.PI / 2);
  c.fillText('Total Variance (cross-sentence) →', 0, 0);
  c.restore();

  // Annotations
  c.font = '10px monospace';
  c.textAlign = 'left';
  c.fillStyle = '#e94560';
  var mostIdx = multiData.summary.most_divergent_layer;
  c.fillText('▲ Most divergent: ' +
    (mostIdx === 0 ? 'Embedding' : 'Layer ' + (mostIdx - 1)) +
    ' (var=' + variances[mostIdx].toFixed(2) + ')',
    ox + 10, oy - 5);

  c.fillStyle = '#53a8b6';
  var leastIdx = multiData.summary.least_divergent_layer;
  c.fillText('▼ Least divergent: ' +
    (leastIdx === 0 ? 'Embedding' : 'Layer ' + (leastIdx - 1)) +
    ' (var=' + variances[leastIdx].toFixed(2) + ')',
    ox + 10, oy + 8);

  // Sentence legend at bottom
  var nSent = multiData.n_sentences;
  var colors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];
  var legY2 = baseY + 40;
  c.font = '9px monospace';
  c.textAlign = 'left';
  c.fillStyle = '#888';
  c.fillText('Sentences compared:', ox, legY2);
  for (var si = 0; si < nSent; si++) {
    var sentText = multiData.sentences[si].text;
    if (sentText.length > 50) sentText = sentText.substring(0, 50) + '…';
    c.fillStyle = colors[si % colors.length];
    c.fillRect(ox, legY2 + 14 + si * 14, 10, 8);
    c.fillStyle = '#a0a0c0';
    c.fillText('[' + si + '] ' + sentText + ' (' + multiData.sentences[si].n_tokens + ' tok)',
      ox + 14, legY2 + 21 + si * 14);
  }

  // HUD
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.3)';
  c.textAlign = 'left';
  c.fillText(
    nL + ' layers | ' + nSent + ' sentences | ' +
    'Variance = how differently sentences are represented at each layer',
    area.x + 10, area.y + area.h - 5
  );
}

// New tab: side-by-side deformation grids
// ============================================================
// MULTI-GRID COMPARISON: Side-by-side deformation grids
// Each sentence gets its own panel with the full 2D deformed
// grid visualization, using the same rendering as draw2D.
// ============================================================

function drawMultiGridComparison(c, area) {
  if (!multiData || !multiData.sentence_data || multiData.sentence_data.length === 0) {
    c.font = '14px monospace';
    c.fillStyle = '#555';
    c.textAlign = 'center';
    c.fillText('No sentence data available — re-run Multi Compare', area.x + area.w / 2, area.y + area.h / 2);
    return;
  }

  var nSent = multiData.sentence_data.length;
  var p = gp(); // get current params (layer, t, amp, dims, etc.)

  // Layout: divide the area into nSent columns with small gaps
  var gap = 6;
  var panelW = Math.floor((area.w - gap * (nSent - 1)) / nSent);
  var panelH = area.h - 30; // leave room for sentence labels at bottom
  var panelY = area.y;

  var sentColors = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71','#e74c3c','#3498db','#9b59b6'];

  for (var si = 0; si < nSent; si++) {
    var sentD = multiData.sentence_data[si];
    if (!sentD || !sentD.fixed_pos || !sentD.deltas) continue;

    var panelX = area.x + si * (panelW + gap);

    // Draw panel background
    c.fillStyle = 'rgba(13,17,23,0.8)';
    c.fillRect(panelX, panelY, panelW, panelH);

    // Panel border
    c.strokeStyle = sentColors[si % sentColors.length];
    c.lineWidth = 1.5;
    c.strokeRect(panelX, panelY, panelW, panelH);

    // Render the deformed grid for this sentence
    drawMiniDeformedGrid(c, sentD, p, panelX, panelY, panelW, panelH);

    // Sentence label below the panel
    c.font = 'bold 9px monospace';
    c.fillStyle = sentColors[si % sentColors.length];
    c.textAlign = 'center';
    var labelText = '[' + si + '] ' + (sentD.text || '').substring(0, Math.floor(panelW / 5.5));
    if ((sentD.text || '').length > Math.floor(panelW / 5.5)) labelText += '…';
    c.fillText(labelText, panelX + panelW / 2, panelY + panelH + 14);

    // Token count and model info
    c.font = '8px monospace';
    c.fillStyle = '#666';
    c.fillText(sentD.n_real + ' tokens | ' + sentD.n_layers + ' layers',
      panelX + panelW / 2, panelY + panelH + 24);
  }

  // HUD at top
  c.font = '10px monospace';
  c.fillStyle = 'rgba(255,255,255,0.4)';
  c.textAlign = 'left';
  c.fillText(
    'Layer ' + p.layer + '/' + (multiData.sentence_data[0].n_layers - 1) +
    '  t=' + p.t.toFixed(2) +
    '  amp=' + p.amp.toFixed(1) +
    '  Dims:' + p.dx + ',' + p.dy +
    '  Mode:' + p.mode +
    '  ITP:' + document.getElementById('sel-itp').value.toUpperCase() +
    '  |  Use sidebar controls to adjust all panels simultaneously',
    area.x + 10, area.y + area.h - 2
  );
}

/**
 * Draw a complete 2D deformed grid visualization for a single sentence's data,
 * clipped to a rectangular panel. This is a self-contained mini version of draw2D
 * that accepts arbitrary data and target rectangle.
 *
 * @param {CanvasRenderingContext2D} c - canvas context
 * @param {Object} sentD - sentence data object (same structure as global D)
 * @param {Object} p - parameters from gp() (layer, t, amp, dx, dy, mode, etc.)
 * @param {number} px - panel x origin
 * @param {number} py - panel y origin
 * @param {number} pw - panel width
 * @param {number} ph - panel height
 */
function drawMiniDeformedGrid(c, sentD, p, px, py, pw, ph) {
  var nP = sentD.n_points;
  var nR = sentD.n_real;
  var dx = p.dx, dy = p.dy;
  var isEmb = (p.mode === 'embedding');

  // Clamp dims to this sentence's hidden_dim
  if (dx >= sentD.hidden_dim) dx = 0;
  if (dy >= sentD.hidden_dim) dy = Math.min(1, sentD.hidden_dim - 1);

  // Use the active deltas (respect decomposition selector)
  var decomp = document.getElementById('sel-decomp').value;
  var activeDeltas = sentD.deltas;
  if (decomp === 'attn' && sentD.attn_deltas) activeDeltas = sentD.attn_deltas;
  if (decomp === 'mlp' && sentD.mlp_deltas) activeDeltas = sentD.mlp_deltas;

  // Clamp layer to this sentence's layer count
  var layer = Math.min(p.layer, sentD.n_layers - 1);
  var amp = p.amp;
  var t = p.t;

  // Extract base positions
  var fx = new Float64Array(nP), fy = new Float64Array(nP);
  for (var i = 0; i < nP; i++) {
    fx[i] = sentD.fixed_pos[i][dx];
    fy[i] = sentD.fixed_pos[i][dy];
  }

  // Compute effective deltas based on mode
  var edx = new Float64Array(nP), edy = new Float64Array(nP);
  if (!isEmb) {
    for (var j = 0; j < nP; j++) {
      var sx2 = 0, sy2 = 0;
      if (p.mode === 'single') {
        sx2 = activeDeltas[layer][j][dx];
        sy2 = activeDeltas[layer][j][dy];
      } else if (p.mode === 'cumfwd') {
        for (var l = 0; l <= layer; l++) {
          sx2 += activeDeltas[l][j][dx];
          sy2 += activeDeltas[l][j][dy];
        }
      } else { // cumbwd
        for (var l2 = layer; l2 < sentD.n_layers; l2++) {
          sx2 += activeDeltas[l2][j][dx];
          sy2 += activeDeltas[l2][j][dy];
        }
      }
      edx[j] = sx2 * amp;
      edy[j] = sy2 * amp;
    }
  }

// Compute view bounds from REAL tokens only so tokens are always visible
  var mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
  for (var i2 = 0; i2 < nR; i2++) {
    if (fx[i2] < mnx) mnx = fx[i2]; if (fx[i2] > mxx) mxx = fx[i2];
    if (fy[i2] < mny) mny = fy[i2]; if (fy[i2] > mxy) mxy = fy[i2];
  }
  // Fallback if only 1 token
  if (!isFinite(mnx) || mnx === mxx) { mnx = fx[0] - 1; mxx = fx[0] + 1; }
  if (!isFinite(mny) || mny === mxy) { mny = fy[0] - 1; mxy = fy[0] + 1; }

  var mr = Math.max(mxx - mnx, mxy - mny) || 1;
  var cxv = (mnx + mxx) / 2, cyv = (mny + mxy) / 2;
  var pd2 = 0.12;
  var vx0 = cxv - mr * (0.5 + pd2), vy0 = cyv - mr * (0.5 + pd2);
  var vw = mr * (1 + 2 * pd2), vh = vw;

  // Margin inside the panel
  var M = 8;
  var dW = pw - 2 * M, dH = ph - 2 * M;

  function SX(x) { return px + M + ((x - vx0) / vw) * dW; }
  function SY(y) { return py + M + ((y - vy0) / vh) * dH; }

  // Build grid
  var N = Math.max(8, Math.min(25, p.gr));
  var nV = (N + 1) * (N + 1);
  var oX = new Float64Array(nV), oY = new Float64Array(nV);
  var gX = new Float64Array(nV), gY = new Float64Array(nV);

  for (var gy = 0; gy <= N; gy++) {
    for (var gx = 0; gx <= N; gx++) {
      var gi = gy * (N + 1) + gx;
      oX[gi] = vx0 + (gx / N) * vw;
      oY[gi] = vy0 + (gy / N) * vh;
    }
  }

  var sig = p.sig;
  var itpMethod = document.getElementById('sel-itp').value;

  if (isEmb) {
    for (var gi2 = 0; gi2 < nV; gi2++) { gX[gi2] = oX[gi2]; gY[gi2] = oY[gi2]; }
  } else {
    for (var gi3 = 0; gi3 < nV; gi3++) {
      var gpx = oX[gi3], gpy = oY[gi3];
      var interp = interpolateGridPoint(gpx, gpy, fx, fy, edx, edy, nP, sig, itpMethod);
      gX[gi3] = gpx + t * interp[0];
      gY[gi3] = gpy + t * interp[1];
    }
  }

  // Compute strain
  var sH = new Float64Array(N * (N + 1));
  var sVa = new Float64Array((N + 1) * N);
  for (var ey2 = 0; ey2 <= N; ey2++) {
    for (var ex2 = 0; ex2 < N; ex2++) {
      var a = ey2 * (N + 1) + ex2, b = a + 1;
      var od = Math.hypot(oX[b] - oX[a], oY[b] - oY[a]);
      var dd = Math.hypot(gX[b] - gX[a], gY[b] - gY[a]);
      sH[ey2 * N + ex2] = od > 1e-12 ? dd / od : 1;
    }
  }
  for (var ey3 = 0; ey3 < N; ey3++) {
    for (var ex3 = 0; ex3 <= N; ex3++) {
      var a2 = ey3 * (N + 1) + ex3, b2 = (ey3 + 1) * (N + 1) + ex3;
      var od2 = Math.hypot(oX[b2] - oX[a2], oY[b2] - oY[a2]);
      var dd2 = Math.hypot(gX[b2] - gX[a2], gY[b2] - gY[a2]);
      sVa[ey3 * (N + 1) + ex3] = od2 > 1e-12 ? dd2 / od2 : 1;
    }
  }

  // ---- Strain heatmap ----
  if (p.heat && !isEmb) {
    for (var hy = 0; hy < N; hy++) {
      for (var hx = 0; hx < N; hx++) {
        var avg = (sH[hy * N + hx] + sH[(hy + 1) * N + hx] +
                   sVa[hy * (N + 1) + hx] + sVa[hy * (N + 1) + hx + 1]) / 4;
        var co = s2c(avg);
        var i00 = hy * (N + 1) + hx, i10 = i00 + 1;
        var i01 = (hy + 1) * (N + 1) + hx, i11 = i01 + 1;
        c.beginPath();
        c.moveTo(SX(gX[i00]), SY(gY[i00]));
        c.lineTo(SX(gX[i10]), SY(gY[i10]));
        c.lineTo(SX(gX[i11]), SY(gY[i11]));
        c.lineTo(SX(gX[i01]), SY(gY[i01]));
        c.closePath();
        c.fillStyle = 'rgba(' + co[0] + ',' + co[1] + ',' + co[2] + ',0.35)';
        c.fill();
      }
    }
  }

  // ---- Reference grid ----
  if (p.ref) {
    c.strokeStyle = isEmb ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.07)';
    c.lineWidth = 0.4;
    for (var ry2 = 0; ry2 <= N; ry2++) {
      c.beginPath();
      for (var rx2 = 0; rx2 <= N; rx2++) {
        var ri = ry2 * (N + 1) + rx2;
        if (rx2 === 0) c.moveTo(SX(oX[ri]), SY(oY[ri]));
        else c.lineTo(SX(oX[ri]), SY(oY[ri]));
      }
      c.stroke();
    }
    for (var rx3 = 0; rx3 <= N; rx3++) {
      c.beginPath();
      for (var ry3 = 0; ry3 <= N; ry3++) {
        var ri3 = ry3 * (N + 1) + rx3;
        if (ry3 === 0) c.moveTo(SX(oX[ri3]), SY(oY[ri3]));
        else c.lineTo(SX(oX[ri3]), SY(oY[ri3]));
      }
      c.stroke();
    }
  }

  // ---- Deformed grid ----
  if (p.grid && !isEmb) {
    c.lineWidth = 0.8;
    // Horizontal edges
    for (var dhy = 0; dhy <= N; dhy++) {
      for (var dhx = 0; dhx < N; dhx++) {
        var di1 = dhy * (N + 1) + dhx, di2 = di1 + 1;
        var es = sH[dhy * N + dhx];
        if (p.sc) {
          var ec = s2c(es);
          c.strokeStyle = 'rgba(' + ec[0] + ',' + ec[1] + ',' + ec[2] + ',0.85)';
        } else {
          c.strokeStyle = 'rgba(200,200,200,0.5)';
        }
        c.beginPath();
        c.moveTo(SX(gX[di1]), SY(gY[di1]));
        c.lineTo(SX(gX[di2]), SY(gY[di2]));
        c.stroke();
      }
    }
    // Vertical edges
    for (var dvx = 0; dvx <= N; dvx++) {
      for (var dvy = 0; dvy < N; dvy++) {
        var dvi1 = dvy * (N + 1) + dvx, dvi2 = (dvy + 1) * (N + 1) + dvx;
        var vs = sVa[dvy * (N + 1) + dvx];
        if (p.sc) {
          var vc = s2c(vs);
          c.strokeStyle = 'rgba(' + vc[0] + ',' + vc[1] + ',' + vc[2] + ',0.85)';
        } else {
          c.strokeStyle = 'rgba(200,200,200,0.5)';
        }
        c.beginPath();
        c.moveTo(SX(gX[dvi1]), SY(gY[dvi1]));
        c.lineTo(SX(gX[dvi2]), SY(gY[dvi2]));
        c.stroke();
      }
    }
  }

  // ---- Vector arrows ----
  if (p.vec && !isEmb) {
    var step = Math.max(1, Math.floor(N / 8));
    c.lineWidth = 1.0;
    for (var viy = 0; viy <= N; viy += step) {
      for (var vix = 0; vix <= N; vix += step) {
        var vi = viy * (N + 1) + vix;
        var ax = SX(oX[vi]), ay = SY(oY[vi]);
        var bx = SX(gX[vi]), by = SY(gY[vi]);
        var al = Math.hypot(bx - ax, by - ay);
        if (al < 2) continue;
        c.strokeStyle = 'rgba(255,255,100,0.5)';
        c.fillStyle = 'rgba(255,255,100,0.5)';
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
        var aa = Math.atan2(by - ay, bx - ax), hl = Math.min(5, al * 0.3);
        c.beginPath(); c.moveTo(bx, by);
        c.lineTo(bx - hl * Math.cos(aa - 0.4), by - hl * Math.sin(aa - 0.4));
        c.lineTo(bx - hl * Math.cos(aa + 0.4), by - hl * Math.sin(aa + 0.4));
        c.closePath(); c.fill();
      }
    }
  }

  // ---- Probe points ----
  if (p.syn) {
    for (var pi = nR; pi < nP; pi++) {
      c.beginPath(); c.arc(SX(fx[pi]), SY(fy[pi]), 1.5, 0, Math.PI * 2);
      c.fillStyle = 'rgba(100,200,255,0.15)'; c.fill();
    }
  }

  // ---- Real token dots ----
  if (p.tok) {
    var tc = ['#e94560','#f5a623','#53a8b6','#7b68ee','#2ecc71',
              '#e74c3c','#3498db','#9b59b6','#1abc9c','#e67e22'];
    for (var ti = 0; ti < nR; ti++) {
      var tx2 = SX(fx[ti]), ty2 = SY(fy[ti]);
      var col = tc[ti % tc.length];

      // Glow
      var grad = c.createRadialGradient(tx2, ty2, 0, tx2, ty2, 12);
      grad.addColorStop(0, 'rgba(255,255,255,0.06)');
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      c.beginPath(); c.arc(tx2, ty2, 12, 0, Math.PI * 2);
      c.fillStyle = grad; c.fill();

      // Dot
      c.beginPath(); c.arc(tx2, ty2, 4, 0, Math.PI * 2);
      c.fillStyle = col; c.fill();
      c.strokeStyle = '#fff'; c.lineWidth = 1; c.stroke();

      // Label
      c.font = 'bold 8px monospace';
      c.lineWidth = 2;
      c.strokeStyle = 'rgba(0,0,0,0.9)';
      var lb = '[' + ti + '] ' + sentD.tokens[ti];
      // Truncate label if panel is narrow
      if (lb.length > Math.floor(pw / 7)) lb = lb.substring(0, Math.floor(pw / 7)) + '…';
      c.strokeText(lb, tx2 + 6, ty2 - 5);
      c.fillStyle = '#fff';
      c.fillText(lb, tx2 + 6, ty2 - 5);
    }

    // Token sequence line in embedding mode
    if (isEmb && nR > 1) {
      c.strokeStyle = 'rgba(233,69,96,0.3)';
      c.lineWidth = 1;
      c.setLineDash([3, 3]);
      c.beginPath();
      c.moveTo(SX(fx[0]), SY(fy[0]));
      for (var ti2 = 1; ti2 < nR; ti2++) c.lineTo(SX(fx[ti2]), SY(fy[ti2]));
      c.stroke();
      c.setLineDash([]);
    }
  }

  // ---- Panel HUD (layer/mode info) ----
  c.font = '8px monospace';
  c.fillStyle = 'rgba(255,255,255,0.35)';
  c.textAlign = 'left';
  if (isEmb) {
    c.fillText('EMB d' + dx + ',' + dy, px + M + 2, py + M + 8);
  } else {
    c.fillText('L' + layer + ' t=' + t.toFixed(1) + ' a=' + amp.toFixed(0) + ' d' + dx + ',' + dy,
      px + M + 2, py + M + 8);
  }

  // Show strain stats if available
  if (sentD.strain_stats && sentD.strain_stats[layer]) {
    var ss = sentD.strain_stats[layer];
    c.fillText('strain: μ=' + ss.mean.toFixed(2) + ' σ²=' + ss.variance.toFixed(3),
      px + M + 2, py + M + 17);
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
    itp_method = req.get("itp_method", "rbf").strip()
    if not text:
        raise ValueError("Empty text")
    print(f"\n[Server] Processing: {text[:60]}... (model: {model_name or MODEL_NAME}, itp: {itp_method})...")
    json_str = process_text(text, model_name, itp_method=itp_method)
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
    if not SAE_AVAILABLE or layer not in SAE_MODELS:
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
        "sae_available": SAE_AVAILABLE,
        "loaded_layers": sorted(SAE_MODELS.keys()),
        "model_name": MODEL_NAME,
        "total_layers": get_n_layers(MODEL_CONFIG) if MODEL_CONFIG else 0,
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
    if not SAE_AVAILABLE or len(SAE_MODELS) == 0:
        return json.dumps({"error": "No SAEs loaded", "features": []}).encode()
    if layer not in SAE_MODELS:
        available = sorted(SAE_MODELS.keys())
        return json.dumps({
            "error": f"No SAE for layer {layer}. Available: {available}",
            "features": []
        }).encode()

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

def handle_contrastive_spectrum(body_bytes):
    """
    Given two sets of prompts, compute the diffeomorphism spectrum
    for each, then find which geometric features (eigenvalue patterns,
    curl, divergence, rank) are most discriminative.

    This automatically finds "the math layers" or "the censorship layers"
    by looking at WHERE and HOW the geometry differs.
    """
    req = json.loads(body_bytes)
    positive_texts = req.get("positive", [])
    negative_texts = req.get("negative", [])
    behavior_name = req.get("behavior", "unknown")

    if not positive_texts or not negative_texts:
        return json.dumps({"error": "Need both positive and negative texts"}).encode()

    n_layers = get_n_layers(MODEL_CONFIG)

    # Compute spectrum for each text
    def compute_spectrum_for_text(text):
        """Compute the diffeomorphism spectrum for a single text."""
        input_ids, tokens = tokenize_text(TOKENIZER, text)
        hs = extract_hidden_states(MODEL, input_ids)
        n_tokens = input_ids.shape[1]
        n_layers = get_n_layers(MODEL_CONFIG)
        hidden_dim = get_hidden_dim(MODEL_CONFIG)

        layer_spectra = []  # [n_layers][n_tokens] -> spectrum dict

        for lay in range(n_layers):
            token_spectra = []

            # Get the token cloud at this layer for PCA directions
            h_cloud = hs[lay][0].cpu().float().numpy()  # (n_tokens, hidden_dim)
            if n_tokens >= 3:
                cloud_centered = h_cloud - h_cloud.mean(axis=0)
                U, S, Vt = np.linalg.svd(cloud_centered, full_matrices=False)
                K = min(32, hidden_dim, n_tokens - 1)
                principal_dirs = Vt[:K]
            else:
                K = min(32, hidden_dim)
                principal_dirs = np.eye(hidden_dim)[:K]

            for ti in range(n_tokens):
                hs[lay][0, ti].cpu().float().numpy()
                delta_base = (hs[lay + 1][0, ti] - hs[lay][0, ti]).cpu().float().numpy()

                eps = 1e-3 * max(np.linalg.norm(delta_base), 1e-6)

                # Compute projected Jacobian via finite differences
                # using the model's actual forward pass through hooks
                J_proj = np.zeros((K, K))

                blocks = _get_transformer_blocks(MODEL)
                can_hook = blocks is not None and lay < len(blocks)

                for j in range(K):
                    perturbation = torch.tensor(
                        principal_dirs[j] * eps,
                        dtype=hs[lay].dtype
                    ).to(hs[lay].device)

                    if can_hook:
                        perturbed_delta = _compute_perturbed_delta(
                            MODEL, input_ids, lay, ti, perturbation, hs
                        )
                    else:
                        perturbed_delta = None

                    if perturbed_delta is not None:
                        Jv = (perturbed_delta - delta_base) / eps
                        for i in range(K):
                            J_proj[i, j] = np.dot(principal_dirs[i], Jv)
                    else:
                        # Fallback: use finite differences on stored hidden states
                        # This is a linear approximation using the delta field
                        # J_proj[i,j] ≈ how much delta changes in direction i
                        # when we move in direction j
                        # Use the token cloud to estimate this
                        if n_tokens >= 3:
                            # Estimate from variation across tokens
                            deltas_cloud = np.stack([
                                (hs[lay + 1][0, t] - hs[lay][0, t]).cpu().float().numpy()
                                for t in range(n_tokens)
                            ], axis=0)  # (n_tokens, hidden_dim)
                            # Project deltas and positions onto principal dirs
                            pos_proj = cloud_centered @ principal_dirs.T  # (n_tokens, K)
                            del_proj = (deltas_cloud - deltas_cloud.mean(axis=0)) @ principal_dirs.T
                            # Least-squares estimate of Jacobian
                            try:
                                J_proj = np.linalg.lstsq(pos_proj, del_proj, rcond=None)[0].T
                            except Exception:
                                J_proj = np.eye(K)
                            break  # Only need to do this once, not per j
                        else:
                            J_proj = np.eye(K)
                            break

                # ---- Spectral decomposition ----
                eigenvalues, eigenvectors = np.linalg.eig(J_proj)

                sort_idx = np.argsort(-np.abs(eigenvalues))
                eigenvalues = eigenvalues[sort_idx]
                eigenvectors = eigenvectors[:, sort_idx]

                eig_real = eigenvalues.real
                eig_imag = eigenvalues.imag
                eig_magnitude = np.abs(eigenvalues)
                eig_phase = np.angle(eigenvalues)

                # ---- Geometric invariants ----
                divergence = float(np.real(np.sum(eigenvalues)))

                J_antisym = (J_proj - J_proj.T) / 2
                curl_magnitude = float(np.linalg.norm(J_antisym, 'fro'))

                J_sym = (J_proj + J_proj.T) / 2
                J_traceless = J_sym - np.eye(K) * np.trace(J_sym) / K
                shear_magnitude = float(np.linalg.norm(J_traceless, 'fro'))

                # Safe determinant (product of eigenvalues can overflow)
                log_det = float(np.real(np.sum(np.log(np.abs(eigenvalues) + 1e-30))))
                det = float(np.exp(np.clip(log_det, -50, 50)))

                sv = np.linalg.svd(J_proj, compute_uv=False)
                condition_number = float(sv[0] / max(sv[-1], 1e-12))

                sv_norm = sv / max(sv.sum(), 1e-12)
                sv_norm = sv_norm[sv_norm > 1e-12]
                effective_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm))))

                n_expanding = int(np.sum(eig_magnitude > 1.05))
                n_contracting = int(np.sum(eig_magnitude < 0.95))
                n_rotating = int(np.sum(np.abs(eig_imag) > 0.05))

                # ---- Holonomy estimate ----
                # Approximate the parallel transport deficit by computing
                # how much the Jacobian's rotation component accumulates
                # This is the antisymmetric part's Frobenius norm
                # (a proxy for the connection's curvature)
                holonomy_proxy = curl_magnitude / max(K, 1)

                # ---- Top eigenvector in hidden space ----
                top_eigvec_dims = _eigvec_to_top_dims(
                    eigenvectors[:, 0], principal_dirs, hidden_dim, top_k=10
                ) if len(eigenvectors) > 0 else []

                token_spectra.append({
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
                    "top_eigenvector_dims": top_eigvec_dims,
                })

            layer_spectra.append(token_spectra)

        return {
            "tokens": tokens,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
            "layer_spectra": layer_spectra,
        }

    # ================================================================
    # MAIN: Compute spectra for positive and negative sets
    # ================================================================

    pos_spectra = []
    for text in positive_texts:
        try:
            spec = compute_spectrum_for_text(text)
            pos_spectra.append(spec)
        except Exception as e:
            print(f"[ContrastiveSpectrum] Error on positive text: {e}")

    neg_spectra = []
    for text in negative_texts:
        try:
            spec = compute_spectrum_for_text(text)
            neg_spectra.append(spec)
        except Exception as e:
            print(f"[ContrastiveSpectrum] Error on negative text: {e}")

    if not pos_spectra or not neg_spectra:
        return json.dumps({"error": "Failed to compute spectra for one or both sets"}).encode()

    n_layers = pos_spectra[0]["n_layers"]

    # ================================================================
    # AGGREGATE: For each layer, compute mean geometric invariants
    # for positive vs negative sets, then find the biggest differences
    # ================================================================

    invariant_keys = [
        "divergence", "curl", "shear", "determinant",
        "condition_number", "effective_rank", "holonomy",
        "n_expanding", "n_contracting", "n_rotating"
    ]

    layer_contrasts = []  # one per layer

    for lay in range(n_layers):
        # Collect invariants across all tokens and all texts in each set
        pos_vals = {k: [] for k in invariant_keys}
        neg_vals = {k: [] for k in invariant_keys}

        for spec in pos_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    for k in invariant_keys:
                        pos_vals[k].append(tok_spec[k])

        for spec in neg_spectra:
            if lay < len(spec["layer_spectra"]):
                for tok_spec in spec["layer_spectra"][lay]:
                    for k in invariant_keys:
                        neg_vals[k].append(tok_spec[k])

        contrast = {"layer": lay}
        total_contrast_score = 0.0

        for k in invariant_keys:
            pos_arr = np.array(pos_vals[k]) if pos_vals[k] else np.array([0.0])
            neg_arr = np.array(neg_vals[k]) if neg_vals[k] else np.array([0.0])

            pos_mean = float(np.mean(pos_arr))
            neg_mean = float(np.mean(neg_arr))
            pos_std = float(np.std(pos_arr))
            neg_std = float(np.std(neg_arr))

            # Effect size (Cohen's d approximation)
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

    # ================================================================
    # RANK: Find the most discriminative layers and invariants
    # ================================================================

    # Sort layers by total contrast score
    ranked_layers = sorted(
        range(n_layers),
        key=lambda lll: layer_contrasts[lll]["total_contrast_score"],
        reverse=True
    )

    # Find the single most discriminative invariant across all layers
    best_invariant = None
    best_effect = 0.0
    best_layer = 0
    for lay in range(n_layers):
        for k in invariant_keys:
            es = layer_contrasts[lay][k]["effect_size"]
            if es > best_effect:
                best_effect = es
                best_invariant = k
                best_layer = lay

    # ================================================================
    # EIGENVALUE SPECTRUM COMPARISON
    # For the top-3 most contrastive layers, compare the full
    # eigenvalue magnitude distributions
    # ================================================================

    eigenvalue_comparisons = []
    for lay in ranked_layers[:3]:
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

        # Compute histogram comparison
        all_eigs = pos_eigs + neg_eigs
        if all_eigs:
            bins = np.linspace(min(all_eigs), max(all_eigs), 30)
            pos_hist, _ = np.histogram(pos_eigs, bins=bins, density=True) if pos_eigs else (np.zeros(29), bins)
            neg_hist, _ = np.histogram(neg_eigs, bins=bins, density=True) if neg_eigs else (np.zeros(29), bins)

            # KL divergence (symmetrized)
            pos_hist_safe = pos_hist + 1e-10
            neg_hist_safe = neg_hist + 1e-10
            pos_hist_safe /= pos_hist_safe.sum()
            neg_hist_safe /= neg_hist_safe.sum()
            kl_div = float(0.5 * np.sum(pos_hist_safe * np.log(pos_hist_safe / neg_hist_safe)) +
                          0.5 * np.sum(neg_hist_safe * np.log(neg_hist_safe / pos_hist_safe)))

            eigenvalue_comparisons.append({
                "layer": lay,
                "pos_histogram": pos_hist.tolist(),
                "neg_histogram": neg_hist.tolist(),
                "bin_edges": bins.tolist(),
                "kl_divergence": round(kl_div, 6),
                "pos_mean_magnitude": round(float(np.mean(pos_eigs)), 6) if pos_eigs else 0.0,
                "neg_mean_magnitude": round(float(np.mean(neg_eigs)), 6) if neg_eigs else 0.0,
            })

    # ================================================================
    # GEOMETRIC SIGNATURE: A compact fingerprint of the behavior
    # ================================================================

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
        # Include individual spectra for the first text of each set
        # (for detailed visualization)
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
        layer_norms = [float(np.linalg.norm(mean_per_layer[l])) for l in range(n_layers + 1)]

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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
