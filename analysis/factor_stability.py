"""Factor stability analysis utilities for SGFA qMAP-PD analysis.

This module implements factor stability assessment using cosine similarity
matching across multiple independent MCMC chains, following the methodology
from Ferreira et al. 2024.

Key Concepts:
- Multiple independent chains are run with different random seeds
- Factors are matched across chains using cosine similarity
- A factor is considered "robust" if it appears in >50% of chains
- Effective factors are those not shrunk to zero by sparsity priors
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)


def procrustes_align_loadings(
    W_target: np.ndarray,
    W_source: np.ndarray,
    scale_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Align loading matrices using Procrustes rotation to handle rotational invariance.

    ASPECT 2: ROTATIONAL INVARIANCE (Rigorous Solution)

    Given two loading matrices W_target (D×K) and W_source (D×K) that may represent
    the same factor subspace but rotated, find the optimal orthogonal rotation R
    such that ||W_target - W_source @ R||² is minimized.

    This handles the fundamental indeterminacy: W @ P and P.T @ Z are equivalent
    for any orthogonal matrix P.

    Mathematical Background
    -----------------------
    The Procrustes problem solves:
        min_R ||W_target - W_source @ R||²_F
        s.t. R.T @ R = I  (R is orthogonal)

    Solution via SVD:
        If W_target.T @ W_source = U @ Σ @ V.T
        Then R = V @ U.T

    Scale Normalization (Optional)
    -------------------------------
    ASPECT 4: SCALE INDETERMINACY

    Before rotation, optionally normalize both matrices to unit Frobenius norm:
        W_norm = W / ||W||_F

    This handles scale indeterminacy: (W/c, c*Z) are equivalent for c > 0.

    Parameters
    ----------
    W_target : np.ndarray, shape (D, K)
        Target (reference) loading matrix
    W_source : np.ndarray, shape (D, K)
        Source loading matrix to be aligned to target
    scale_normalize : bool, default=True
        If True, normalize both matrices to unit Frobenius norm before alignment.
        Handles scale indeterminacy (Aspect 4).

    Returns
    -------
    W_aligned : np.ndarray, shape (D, K)
        Source loadings after Procrustes rotation (and optional scaling)
    R : np.ndarray, shape (K, K)
        Optimal orthogonal rotation matrix
    info : Dict
        Diagnostic information:
            - disparity: Procrustes disparity (normalized alignment error)
            - scale_target: Original Frobenius norm of target
            - scale_source: Original Frobenius norm of source
            - scale_ratio: scale_source / scale_target
            - rotation_angle: Maximum rotation angle in degrees

    Notes
    -----
    - Handles rotational invariance rigorously (better than cosine similarity)
    - Preserves orthogonality: If W_source factors are orthogonal, so are aligned factors
    - Scale normalization is recommended to separate rotation from scaling

    References
    ----------
    - Schönemann (1966) "A generalized solution of the orthogonal Procrustes problem"
    - Gower & Dijksterhuis (2004) "Procrustes Problems"

    Examples
    --------
    >>> W1 = np.random.randn(100, 5)
    >>> # Create rotated version
    >>> P = scipy.stats.special_ortho_group.rvs(5)
    >>> W2 = W1 @ P + 0.1 * np.random.randn(100, 5)  # rotated + noise
    >>> W_aligned, R, info = procrustes_align_loadings(W1, W2)
    >>> print(f"Disparity: {info['disparity']:.4f}")
    """
    D, K = W_target.shape
    assert W_source.shape == (D, K), f"Shape mismatch: {W_source.shape} != {(D, K)}"

    # Store original norms for diagnostics
    scale_target = np.linalg.norm(W_target, ord='fro')
    scale_source = np.linalg.norm(W_source, ord='fro')

    # Handle edge case: zero matrices
    if scale_target < 1e-10 or scale_source < 1e-10:
        logger.warning("Procrustes: One or both matrices near-zero, returning identity rotation")
        return W_source, np.eye(K), {
            'disparity': np.nan,
            'scale_target': scale_target,
            'scale_source': scale_source,
            'scale_ratio': scale_source / (scale_target + 1e-10),
            'rotation_angle': 0.0,
        }

    # Optional: Normalize to unit Frobenius norm (handles scale indeterminacy)
    if scale_normalize:
        W_target_norm = W_target / scale_target
        W_source_norm = W_source / scale_source
    else:
        W_target_norm = W_target
        W_source_norm = W_source

    # Compute Procrustes rotation using scipy
    R, scale = orthogonal_procrustes(W_source_norm, W_target_norm)
    # Note: scipy returns (R, scale) where optimal transformation is scale * W_source @ R
    # For orthogonal Procrustes, scale should be ≈1.0 if matrices are normalized

    # Apply rotation to source (use original or normalized depending on scale_normalize)
    if scale_normalize:
        # Align normalized matrices, then restore target's scale
        W_aligned = (W_source_norm @ R) * scale_target
    else:
        # Direct alignment without normalization
        W_aligned = W_source @ R

    # Compute disparity (normalized alignment error)
    disparity = np.linalg.norm(W_target_norm - W_source_norm @ R, ord='fro')

    # Compute maximum rotation angle (trace of R relates to rotation magnitude)
    # For orthogonal matrix R: trace(R) = sum of cos(θ_i)
    # Maximum rotation angle: arccos(λ_min(R)) where λ_min is smallest eigenvalue
    trace_R = np.trace(R)
    # Average rotation angle from trace
    avg_cos = trace_R / K
    avg_cos = np.clip(avg_cos, -1.0, 1.0)  # Numerical stability
    avg_rotation_deg = np.degrees(np.arccos(avg_cos))

    info = {
        'disparity': float(disparity),
        'scale_target': float(scale_target),
        'scale_source': float(scale_source),
        'scale_ratio': float(scale_source / scale_target),
        'rotation_angle': float(avg_rotation_deg),
        'trace_R': float(trace_R),
        'scale_from_procrustes': float(scale),
    }

    return W_aligned, R, info


def assess_factor_stability_cosine(
    chain_results: List[Dict],
    threshold: float = 0.8,
    min_match_rate: float = 0.5,
) -> Dict:
    """Assess factor stability using cosine similarity across chains.

    This function implements the factor stability analysis methodology from
    Ferreira et al. 2024 Section 2.7, using cosine similarity to match
    factors across independent MCMC chains.

    Algorithm:
    1. Average W (factor loadings) within each chain across posterior samples
    2. Use chain 0 as reference
    3. For each factor k in chain 0:
       - Find best matching factor in other chains using cosine similarity
       - Count matches where similarity > threshold
       - Factor is "robust" if match_count / n_chains > min_match_rate

    Parameters
    ----------
    chain_results : List[Dict]
        List of chain results, each containing:
        - "W": Factor loadings (D, K) or list of (D_m, K)
        - "samples": Optional dict with "W" samples (n_samples, D, K)
        - "chain_id": Chain identifier
    threshold : float, default=0.8
        Minimum cosine similarity for two factors to be considered matching
    min_match_rate : float, default=0.5
        Minimum fraction of chains in which a factor must appear to be
        considered robust (e.g., 0.5 means >50% of chains)

    Returns
    -------
    Dict
        Stability analysis results:
        - n_stable_factors: Number of robust factors
        - stable_factor_indices: List of robust factor indices
        - total_factors: Total number of factors (K)
        - threshold: Similarity threshold used
        - min_match_rate: Match rate threshold used
        - per_factor_details: List of dicts with per-factor matching info
        - similarity_matrix: (n_chains, n_chains, K) array of similarities
        - consensus_W: Consensus factor loadings for stable factors (D, n_stable)
        - consensus_Z: Consensus factor scores for stable factors (N, n_stable) or None

    Examples
    --------
    >>> chain_results = [
    ...     {"chain_id": 0, "W": W0, "samples": {"W": W0_samples}},
    ...     {"chain_id": 1, "W": W1, "samples": {"W": W1_samples}},
    ...     {"chain_id": 2, "W": W2, "samples": {"W": W2_samples}},
    ...     {"chain_id": 3, "W": W3, "samples": {"W": W3_samples}},
    ... ]
    >>> stability = assess_factor_stability_cosine(chain_results, threshold=0.8)
    >>> print(f"Found {stability['n_stable_factors']} stable factors")
    """
    logger.info("=" * 80)
    logger.info("ASSESS_FACTOR_STABILITY_COSINE - STARTING")
    logger.info("=" * 80)
    logger.info(f"Number of chains: {len(chain_results)}")
    logger.info(f"Similarity threshold: {threshold}")
    logger.info(f"Min match rate: {min_match_rate}")

    # Log chain information
    for i, result in enumerate(chain_results):
        chain_id = result.get("chain_id", i)
        W = result.get("W")
        logger.info(f"Chain {chain_id}: W type={type(W)}, samples={'samples' in result}")

    # Step 1: Average W within each chain across posterior samples
    W_chain_avg = []
    for i, result in enumerate(chain_results):
        chain_id = result.get("chain_id", i)

        # Check if we have posterior samples
        if "samples" in result and "W" in result.get("samples", {}):
            W_samples = result["samples"]["W"]  # (n_samples, D, K) or (n_samples, sum(D_m), K)
            logger.info(f"Chain {chain_id}: Averaging {W_samples.shape[0]} posterior samples")

            # Handle multi-view case (list of W matrices)
            if isinstance(result["W"], list):
                # W is a list of views - concatenate them
                W_concat = np.vstack([W_view for W_view in result["W"]])
                W_chain_avg.append(W_concat)
                logger.info(f"Chain {chain_id}: Multi-view W concatenated to shape {W_concat.shape}")
            else:
                # Single concatenated W matrix - average across samples
                W_avg = np.mean(W_samples, axis=0)  # (D, K)
                W_chain_avg.append(W_avg)
                logger.info(f"Chain {chain_id}: Averaged W shape {W_avg.shape}")
        else:
            # No samples, use point estimate
            W = result["W"]
            if isinstance(W, list):
                # Concatenate multi-view W
                W_concat = np.vstack([W_view for W_view in W])
                W_chain_avg.append(W_concat)
                logger.info(f"Chain {chain_id}: Multi-view W (no samples) shape {W_concat.shape}")
            else:
                W_chain_avg.append(W)
                logger.info(f"Chain {chain_id}: W (no samples) shape {W.shape}")

    n_chains = len(W_chain_avg)
    D, K = W_chain_avg[0].shape

    logger.info(f"Factor loading matrices: D={D} features, K={K} factors")

    # Step 1b: Average Z within each chain (for consensus Z computation later)
    Z_chain_avg = []
    for i, result in enumerate(chain_results):
        chain_id = result.get("chain_id", i)

        # Check if we have Z scores
        Z = result.get("Z")
        if Z is not None:
            if "samples" in result and "Z" in result.get("samples", {}):
                Z_samples = result["samples"]["Z"]  # (n_samples, N, K)
                Z_avg = np.mean(Z_samples, axis=0)  # (N, K)
                Z_chain_avg.append(Z_avg)
                logger.info(f"Chain {chain_id}: Averaged Z shape {Z_avg.shape}")
            else:
                Z_chain_avg.append(Z)
                logger.info(f"Chain {chain_id}: Z (no samples) shape {Z.shape}")
        else:
            logger.warning(f"Chain {chain_id}: No Z scores available")
            Z_chain_avg.append(None)

    # Validate all chains have same shape
    for i, W in enumerate(W_chain_avg):
        if W.shape != (D, K):
            raise ValueError(
                f"Chain {i} has shape {W.shape}, expected ({D}, {K}). "
                "All chains must have the same number of features and factors."
            )

    # Step 2: Match factors across chains using cosine similarity
    logger.info(f"Step 2: Matching {K} factors across {n_chains} chains")
    logger.info(f"  Using sign-invariant cosine similarity (handles factor sign flips)")
    logger.info(f"  Zero-vector detection enabled (threshold: 1e-10)")
    stable_factors = []
    per_factor_matches = []

    # Store similarity matrix for all factor pairs across chains
    # Shape: (n_chains, n_chains, K) - similarity of factor k in chain i to best match in chain j
    similarity_matrix = np.zeros((n_chains, n_chains, K))

    # Track edge cases
    n_zero_ref = 0
    n_zero_factor = 0
    n_sign_flips = 0

    for k in range(K):
        if k % 5 == 0:  # Log every 5th factor to avoid log spam
            logger.info(f"  Processing factor {k}/{K}")
        ref_factor = W_chain_avg[0][:, k]  # Reference from chain 0
        matches = 1  # Chain 0 matches itself
        matched_indices = [k]  # Which factor index matched in each chain
        matched_similarities = [1.0]  # Similarity scores for each match

        # Match this factor to all other chains
        for chain_idx in range(1, n_chains):
            # Find best matching factor in this chain
            similarities = []
            for k2 in range(K):
                factor = W_chain_avg[chain_idx][:, k2]

                # Compute norms for numerical stability check
                ref_norm = np.linalg.norm(ref_factor)
                factor_norm = np.linalg.norm(factor)

                # Handle zero/near-zero vectors (shrunk factors from ARD)
                if ref_norm < 1e-10 and factor_norm < 1e-10:
                    # Both factors shrunk to zero - consider them matched
                    sim = 1.0
                    n_zero_ref += 1
                    n_zero_factor += 1
                elif ref_norm < 1e-10:
                    # Reference shrunk, factor active - not a match
                    sim = 0.0
                    n_zero_ref += 1
                elif factor_norm < 1e-10:
                    # Factor shrunk, reference active - not a match
                    sim = 0.0
                    n_zero_factor += 1
                else:
                    # Normal case: compute cosine similarity
                    # Use absolute value for sign invariance (factors can flip signs)
                    sim_raw = 1 - cosine(ref_factor, factor)
                    sim = abs(sim_raw)  # Sign-invariant: [1,2,3] matches [-1,-2,-3]

                    # Track sign flips (when raw similarity is negative)
                    if sim_raw < -0.5:  # Strong negative correlation = sign flip
                        n_sign_flips += 1

                similarities.append(sim)

            best_sim = max(similarities)
            best_k = int(np.argmax(similarities))

            # Store similarity
            similarity_matrix[0, chain_idx, k] = best_sim

            if best_sim > threshold:
                matches += 1
                matched_indices.append(best_k)
                matched_similarities.append(float(best_sim))
            else:
                matched_indices.append(None)
                matched_similarities.append(float(best_sim))

        match_rate = matches / n_chains
        is_robust = match_rate > min_match_rate

        per_factor_matches.append({
            "factor_index": int(k),
            "matches": int(matches),
            "match_rate": float(match_rate),
            "matched_in_chains": matched_indices,
            "matched_similarities": matched_similarities,
            "is_robust": bool(is_robust),
            "reference_chain": 0,
        })

        # Factor is robust if it appears in >min_match_rate of chains
        if is_robust:
            stable_factors.append(k)
            logger.info(
                f"Factor {k}: ROBUST (matched in {matches}/{n_chains} chains, "
                f"rate={match_rate:.2%})"
            )
        else:
            logger.debug(
                f"Factor {k}: unstable (matched in {matches}/{n_chains} chains, "
                f"rate={match_rate:.2%})"
            )

    # Log edge case summary
    logger.info(f"\n📊 Cosine Similarity Edge Case Summary:")
    logger.info(f"  Zero reference vectors encountered: {n_zero_ref}")
    logger.info(f"  Zero comparison vectors encountered: {n_zero_factor}")
    logger.info(f"  Sign flips detected (|sim_raw| < -0.5): {n_sign_flips}")
    total_comparisons = K * (n_chains - 1) * K  # For each ref factor, compare to all factors in other chains
    logger.info(f"  Total factor comparisons: {total_comparisons}")
    if n_sign_flips > 0:
        logger.info(f"  ⚠️  Sign flips are normal - factors can flip signs and still be valid")

    # Step 3: Compute consensus factor loadings and scores
    # IMPORTANT: Always compute consensus for ALL factors, not just stable ones
    # Users need these files even if factors aren't deemed "stable" by the threshold
    consensus_W = None
    consensus_Z = None

    # Compute consensus for all factors (not just stable ones)
    all_factor_indices = list(range(K))
    if len(all_factor_indices) > 0:
        logger.info(f"Computing consensus loadings for all {K} factors")
        logger.info(f"  (Stable factors: {len(stable_factors)}/{K})")
        consensus_W = _compute_consensus_loadings(
            W_chain_avg, per_factor_matches, all_factor_indices
        )

        # Compute consensus Z if Z scores are available
        consensus_Z_result = None
        if all(Z is not None for Z in Z_chain_avg):
            logger.info(f"Computing consensus scores for all {K} factors")
            consensus_Z_result = _compute_consensus_scores(
                Z_chain_avg, per_factor_matches, all_factor_indices
            )
        else:
            logger.warning("Not all chains have Z scores - consensus_Z will be None")

    # Step 4: Compile results
    logger.info("=" * 80)
    logger.info("ASSESS_FACTOR_STABILITY_COSINE - COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Results:")
    logger.info(f"  - Stable factors: {len(stable_factors)}/{K}")
    logger.info(f"  - Stability rate: {len(stable_factors)/K:.1%}")
    logger.info(f"  - Stable factor indices: {stable_factors}")
    logger.info(f"  - Consensus W shape: {consensus_W.shape if consensus_W is not None else 'None'}")
    if consensus_Z_result is not None:
        logger.info(f"  - Consensus Z shape: {consensus_Z_result['median'].shape}")
        logger.info(f"  - Z posterior uncertainty quantified (std, CI)")
    else:
        logger.info(f"  - Consensus Z shape: None")

    result = {
        "n_stable_factors": len(stable_factors),
        "stable_factor_indices": stable_factors,
        "all_factor_indices": all_factor_indices,  # For consensus saving (all factors)
        "total_factors": K,
        "n_chains": n_chains,
        "threshold": threshold,
        "min_match_rate": min_match_rate,
        "per_factor_details": per_factor_matches,
        "similarity_matrix": similarity_matrix,
        "consensus_W": consensus_W,
        "consensus_Z": consensus_Z_result["median"] if consensus_Z_result is not None else None,
        "consensus_Z_std": consensus_Z_result["std"] if consensus_Z_result is not None else None,
        "consensus_Z_q025": consensus_Z_result["q025"] if consensus_Z_result is not None else None,
        "consensus_Z_q975": consensus_Z_result["q975"] if consensus_Z_result is not None else None,
        "consensus_Z_n_chains": consensus_Z_result["n_chains_per_factor"] if consensus_Z_result is not None else None,
        "stability_rate": len(stable_factors) / K if K > 0 else 0.0,
    }

    logger.info(f"Factor stability analysis complete:")
    logger.info(f"  - {len(stable_factors)}/{K} factors are robust ({result['stability_rate']:.1%})")
    logger.info(f"  - Stable factor indices: {stable_factors}")

    return result


def assess_factor_stability_procrustes(
    chain_results: List[Dict],
    scale_normalize: bool = True,
    disparity_threshold: float = 0.3,
    min_match_rate: float = 0.5,
) -> Dict:
    """
    Assess factor stability using Procrustes alignment (rigorous rotational invariance).

    ASPECT 2: ROTATIONAL INVARIANCE (Rigorous Solution)
    ASPECT 4: SCALE INDETERMINACY (Explicit Handling)

    This is a more rigorous alternative to cosine-based matching. Instead of matching
    individual factors independently, Procrustes aligns the entire loading subspace,
    properly handling joint rotations of multiple factors.

    Advantages over cosine similarity:
    - Handles joint rotations: Factors can rotate together as a group
    - Explicit scale normalization: Separates rotation from scaling (Aspect 4)
    - Theoretical guarantees: Optimal alignment under Frobenius norm
    - Diagnostic metrics: Disparity, rotation angles, scale ratios

    Parameters
    ----------
    chain_results : List[Dict]
        List of MCMC chain results, each containing 'W' and optionally 'Z'
    scale_normalize : bool, default=True
        Normalize loading matrices to unit Frobenius norm before alignment.
        Recommended to handle scale indeterminacy (Aspect 4).
    disparity_threshold : float, default=0.3
        Maximum Procrustes disparity to consider chains aligned.
        Disparity ≈ 0: Perfect alignment
        Disparity > 0.3: Significant differences (different factor structure)
    min_match_rate : float, default=0.5
        Minimum fraction of chains a factor must appear in to be considered robust

    Returns
    -------
    Dict containing:
        - consensus_W: np.ndarray, consensus loading matrix
        - consensus_Z: np.ndarray or None, consensus scores if available
        - rotation_matrices: List[np.ndarray], rotation for each chain
        - disparities: List[float], alignment quality per chain
        - scale_ratios: List[float], scale differences per chain
        - n_aligned_chains: int, number of chains below disparity threshold
        - alignment_rate: float, fraction of chains well-aligned

    Notes
    -----
    - Uses chain 0 as reference
    - All other chains are rotated (and optionally scaled) to match chain 0
    - High disparity indicates different factor structure (not just rotation)
    - Scale ratios reveal if chains converged to different scaling regimes

    See Also
    --------
    procrustes_align_loadings : Core Procrustes alignment function
    assess_factor_stability_cosine : Alternative using cosine similarity
    """
    logger.info("=" * 80)
    logger.info("ASSESS_FACTOR_STABILITY_PROCRUSTES - START")
    logger.info("=" * 80)
    logger.info(f"Analyzing {len(chain_results)} chains using Procrustes alignment")
    logger.info(f"  Scale normalization: {scale_normalize}")
    logger.info(f"  Disparity threshold: {disparity_threshold}")
    logger.info(f"  Min match rate: {min_match_rate}")

    n_chains = len(chain_results)
    if n_chains < 2:
        raise ValueError("Need at least 2 chains for stability analysis")

    # Extract loading matrices
    W_chains = [result['W'] for result in chain_results]
    N, K = W_chains[0].shape[0], W_chains[0].shape[1]
    D = W_chains[0].shape[0]

    # Check Z availability
    has_Z = all('Z' in result and result['Z'] is not None for result in chain_results)
    if has_Z:
        Z_chains = [result['Z'] for result in chain_results]
    else:
        Z_chains = None

    # Use chain 0 as reference
    W_ref = W_chains[0]
    logger.info(f"\nUsing chain 0 as reference: W shape = {W_ref.shape}")

    # Align all other chains to reference using Procrustes
    W_aligned = [W_ref]  # Chain 0 is reference
    Z_aligned = [Z_chains[0]] if has_Z else [None]
    rotation_matrices = [np.eye(K)]  # Identity for reference
    disparities = [0.0]  # Zero disparity for reference
    scale_ratios = [1.0]  # Ratio of 1.0 for reference
    rotation_angles = [0.0]

    logger.info(f"\nAligning chains 1-{n_chains-1} to reference...")

    for chain_idx in range(1, n_chains):
        W_source = W_chains[chain_idx]

        # Procrustes alignment
        W_aligned_chain, R, info = procrustes_align_loadings(
            W_ref, W_source, scale_normalize=scale_normalize
        )

        W_aligned.append(W_aligned_chain)
        rotation_matrices.append(R)
        disparities.append(info['disparity'])
        scale_ratios.append(info['scale_ratio'])
        rotation_angles.append(info['rotation_angle'])

        logger.info(
            f"  Chain {chain_idx}: disparity={info['disparity']:.4f}, "
            f"scale_ratio={info['scale_ratio']:.4f}, "
            f"rotation={info['rotation_angle']:.1f}°"
        )

        # Align Z if available (Z_new = Z_old @ R.T because X = Z @ W.T)
        if has_Z:
            Z_source = Z_chains[chain_idx]
            Z_aligned_chain = Z_source @ R.T
            Z_aligned.append(Z_aligned_chain)

    # Count well-aligned chains
    n_aligned = sum(1 for d in disparities if d <= disparity_threshold)
    alignment_rate = n_aligned / n_chains

    logger.info(f"\n📊 Procrustes Alignment Summary:")
    logger.info(f"  Disparities: min={min(disparities):.4f}, max={max(disparities):.4f}, "
                f"mean={np.mean(disparities):.4f}")
    logger.info(f"  Scale ratios: min={min(scale_ratios):.4f}, max={max(scale_ratios):.4f}")
    logger.info(f"  Rotation angles: min={min(rotation_angles):.1f}°, max={max(rotation_angles):.1f}°")
    logger.info(f"  Well-aligned chains: {n_aligned}/{n_chains} ({alignment_rate:.1%})")

    if alignment_rate < 0.5:
        logger.warning("⚠️  < 50% of chains are well-aligned!")
        logger.warning("   This suggests different factor structures, not just rotation.")
        logger.warning("   Consider: (1) longer MCMC runs, (2) better priors, (3) fewer factors")

    # Compute consensus as median (more robust than mean)
    logger.info("\nComputing consensus loadings (median across aligned chains)...")
    W_stack = np.stack(W_aligned, axis=0)  # (n_chains, D, K)
    consensus_W = np.median(W_stack, axis=0)  # (D, K)

    consensus_Z = None
    if has_Z:
        logger.info("Computing consensus scores (median across aligned chains)...")
        Z_stack = np.stack(Z_aligned, axis=0)  # (n_chains, N, K)
        consensus_Z = np.median(Z_stack, axis=0)  # (N, K)

    # Compute post-alignment correlations to check for remaining rotation
    W_stack_std = np.std(W_stack, axis=0)  # (D, K)
    mean_std_per_factor = np.mean(W_stack_std, axis=0)  # (K,)
    logger.info(f"\n📊 Post-alignment variability (mean std per factor):")
    for k in range(K):
        logger.info(f"  Factor {k}: {mean_std_per_factor[k]:.4f}")

    # Build result
    result = {
        'method': 'procrustes',
        'consensus_W': consensus_W,
        'consensus_Z': consensus_Z,
        'rotation_matrices': rotation_matrices,
        'disparities': disparities,
        'scale_ratios': scale_ratios,
        'rotation_angles': rotation_angles,
        'n_aligned_chains': n_aligned,
        'alignment_rate': alignment_rate,
        'n_chains': n_chains,
        'scale_normalize': scale_normalize,
        'disparity_threshold': disparity_threshold,
        'mean_std_per_factor': mean_std_per_factor.tolist(),
    }

    logger.info("=" * 80)
    logger.info("ASSESS_FACTOR_STABILITY_PROCRUSTES - COMPLETED")
    logger.info("=" * 80)

    return result


def diagnose_scale_indeterminacy(
    chain_results: List[Dict],
    tau_W_key: str = "tauW",
    tau_Z_key: str = "tauZ",
) -> Dict:
    """
    Diagnose scale indeterminacy by checking ARD hyperparameters and W/Z scales.

    ASPECT 4: SCALE INDETERMINACY (Explicit Diagnostic)

    Mathematical Reality:
        W/c and c*Z produce the same reconstruction for any c > 0
        X = Z @ W.T = (c*Z) @ (W/c).T

    The model handles this through ARD priors:
        Z ~ N(0, 1) * tauZ * lmbZ
        W ~ N(0, 1) * tauW * lmbW

    However, scale can still vary across chains if priors are weak. This function
    diagnoses whether scale is well-constrained by checking:
    1. Consistency of tauW and tauZ across chains
    2. Product tauW * tauZ (should be stable)
    3. Frobenius norms of W and Z matrices

    Parameters
    ----------
    chain_results : List[Dict]
        List of MCMC chain results containing 'W', 'Z', and hyperparameter samples
    tau_W_key : str, default="tauW"
        Key for W scale hyperparameter in samples dict
    tau_Z_key : str, default="tauZ"
        Key for Z scale hyperparameter in samples dict

    Returns
    -------
    Dict containing:
        - W_norms: List[float], Frobenius norm of W per chain
        - Z_norms: List[float], Frobenius norm of Z per chain (if available)
        - tau_W_means: List[float], mean tauW per chain (if available)
        - tau_Z_means: List[float], mean tauZ per chain (if available)
        - scale_products: List[float], ||W|| * ||Z|| per chain
        - tau_products: List[float], tauW * tauZ per chain (if available)
        - W_norm_cv: float, coefficient of variation of W norms
        - Z_norm_cv: float, coefficient of variation of Z norms
        - product_cv: float, coefficient of variation of products
        - scale_is_stable: bool, whether scale is consistent across chains
        - recommendations: List[str], diagnostic messages

    Notes
    -----
    - Low CV (< 0.2): Scale is well-constrained
    - Medium CV (0.2-0.5): Some scale variation but acceptable
    - High CV (> 0.5): Scale is poorly constrained, consider stronger priors

    The product ||W|| * ||Z|| should be more stable than individual norms
    because the reconstruction X = Z @ W.T constrains the product.
    """
    logger.info("=" * 80)
    logger.info("DIAGNOSE SCALE INDETERMINACY - START")
    logger.info("=" * 80)

    n_chains = len(chain_results)
    W_norms = []
    Z_norms = []
    scale_products = []
    tau_W_means = []
    tau_Z_means = []
    tau_products = []

    has_Z = all('Z' in result and result['Z'] is not None for result in chain_results)
    has_samples = all('samples' in result for result in chain_results)

    for chain_idx, result in enumerate(chain_results):
        W = result['W']
        W_norm = np.linalg.norm(W, ord='fro')
        W_norms.append(W_norm)

        if has_Z:
            Z = result['Z']
            Z_norm = np.linalg.norm(Z, ord='fro')
            Z_norms.append(Z_norm)
            scale_products.append(W_norm * Z_norm)

        # Extract ARD hyperparameters if available
        if has_samples and 'samples' in result:
            samples = result['samples']

            # Look for tauW (may be tauW1, tauW2, etc. for multi-view)
            tau_W_values = []
            for key in samples.keys():
                if key.startswith(tau_W_key):
                    tau_W_samples = samples[key]
                    tau_W_values.append(np.mean(tau_W_samples))

            if tau_W_values:
                tau_W_mean = np.mean(tau_W_values)  # Average across views
                tau_W_means.append(tau_W_mean)

            # Look for tauZ
            if tau_Z_key in samples:
                tau_Z_samples = samples[tau_Z_key]
                tau_Z_mean = np.mean(tau_Z_samples)
                tau_Z_means.append(tau_Z_mean)

                # Compute product
                if tau_W_values:
                    tau_products.append(tau_W_mean * tau_Z_mean)

    # Compute coefficients of variation (CV = std / mean)
    def cv(values):
        if len(values) < 2:
            return np.nan
        return np.std(values) / (np.mean(values) + 1e-10)

    W_norm_cv = cv(W_norms)
    Z_norm_cv = cv(Z_norms) if Z_norms else np.nan
    product_cv = cv(scale_products) if scale_products else np.nan
    tau_W_cv = cv(tau_W_means) if tau_W_means else np.nan
    tau_Z_cv = cv(tau_Z_means) if tau_Z_means else np.nan
    tau_product_cv = cv(tau_products) if tau_products else np.nan

    # Determine if scale is stable
    # Use product CV as primary metric (most constrained by data)
    if not np.isnan(product_cv):
        scale_is_stable = product_cv < 0.3
    elif not np.isnan(W_norm_cv):
        scale_is_stable = W_norm_cv < 0.3
    else:
        scale_is_stable = True  # Insufficient data

    # Generate recommendations
    recommendations = []

    logger.info(f"\n📊 Scale Diagnostics ({n_chains} chains):")
    logger.info(f"\nFrobenius Norms:")
    logger.info(f"  ||W||_F: mean={np.mean(W_norms):.2f}, std={np.std(W_norms):.2f}, CV={W_norm_cv:.3f}")

    if Z_norms:
        logger.info(f"  ||Z||_F: mean={np.mean(Z_norms):.2f}, std={np.std(Z_norms):.2f}, CV={Z_norm_cv:.3f}")
        logger.info(f"  ||W||*||Z||: mean={np.mean(scale_products):.2f}, std={np.std(scale_products):.2f}, CV={product_cv:.3f}")

    if tau_W_means:
        logger.info(f"\nARD Hyperparameters:")
        logger.info(f"  tauW: mean={np.mean(tau_W_means):.4f}, std={np.std(tau_W_means):.4f}, CV={tau_W_cv:.3f}")

    if tau_Z_means:
        logger.info(f"  tauZ: mean={np.mean(tau_Z_means):.4f}, std={np.std(tau_Z_means):.4f}, CV={tau_Z_cv:.3f}")

    if tau_products:
        logger.info(f"  tauW*tauZ: mean={np.mean(tau_products):.4f}, std={np.std(tau_products):.4f}, CV={tau_product_cv:.3f}")

    # Interpretation
    logger.info(f"\n🔍 Scale Indeterminacy Assessment:")

    if scale_is_stable:
        logger.info("  ✅ GOOD: Scale is well-constrained across chains")
        if product_cv < 0.1:
            logger.info("     Product ||W||*||Z|| is very stable (CV < 0.1)")
            recommendations.append("Scale indeterminacy is well-handled by ARD priors")
        elif product_cv < 0.3:
            logger.info("     Product ||W||*||Z|| is reasonably stable (CV < 0.3)")
            recommendations.append("Scale is adequately constrained, no action needed")
    else:
        logger.warning("  ⚠️  CAUTION: Scale varies significantly across chains!")

        if product_cv > 0.5:
            logger.warning(f"     Product CV = {product_cv:.3f} > 0.5")
            recommendations.append("WARNING: High scale variation across chains")
            recommendations.append("Consider: (1) Stronger ARD priors, (2) Explicit normalization")

        if W_norm_cv > 0.5:
            logger.warning(f"     ||W|| CV = {W_norm_cv:.3f} > 0.5")
            recommendations.append("W scale varies significantly - check convergence")

        if not np.isnan(Z_norm_cv) and Z_norm_cv > 0.5:
            logger.warning(f"     ||Z|| CV = {Z_norm_cv:.3f} > 0.5")
            recommendations.append("Z scale varies significantly - check convergence")

    # Check if ARD priors are constraining scale
    if tau_products and tau_product_cv < 0.3:
        logger.info(f"  ✅ ARD hyperparameters are stable (tauW*tauZ CV = {tau_product_cv:.3f})")
        recommendations.append("ARD priors successfully constrain scale")
    elif tau_products and tau_product_cv > 0.5:
        logger.warning(f"  ⚠️  ARD hyperparameters vary significantly (CV = {tau_product_cv:.3f})")
        recommendations.append("ARD priors may be too weak - consider informative priors")

    result = {
        'W_norms': W_norms,
        'Z_norms': Z_norms if Z_norms else None,
        'tau_W_means': tau_W_means if tau_W_means else None,
        'tau_Z_means': tau_Z_means if tau_Z_means else None,
        'scale_products': scale_products if scale_products else None,
        'tau_products': tau_products if tau_products else None,
        'W_norm_cv': float(W_norm_cv),
        'Z_norm_cv': float(Z_norm_cv) if not np.isnan(Z_norm_cv) else None,
        'product_cv': float(product_cv) if not np.isnan(product_cv) else None,
        'tau_W_cv': float(tau_W_cv) if not np.isnan(tau_W_cv) else None,
        'tau_Z_cv': float(tau_Z_cv) if not np.isnan(tau_Z_cv) else None,
        'tau_product_cv': float(tau_product_cv) if not np.isnan(tau_product_cv) else None,
        'scale_is_stable': scale_is_stable,
        'recommendations': recommendations,
    }

    logger.info("\n💡 Recommendations:")
    for rec in recommendations:
        logger.info(f"  - {rec}")

    logger.info("=" * 80)
    logger.info("DIAGNOSE SCALE INDETERMINACY - COMPLETED")
    logger.info("=" * 80)

    return result


def diagnose_slab_saturation(
    chain_results: List[Dict],
    slab_scale: float = None,
    saturation_threshold: float = 0.8,
) -> Dict:
    """
    Diagnose whether loadings/scores are saturating at the slab scale ceiling.

    ASPECT 11: SLAB SCALE (c²) AND REGULARIZED HORSESHOE BEHAVIOR

    Mathematical Background
    -----------------------
    The regularized horseshoe prior has TWO regimes:

    Regularized variance formula:
        σ̃² = (c² * τ² * λ²) / (c² + τ² * λ²)

    Regime 1: Small coefficients (τ²λ² << c²)
        σ̃² ≈ τ²λ²  →  Aggressive shrinkage toward zero

    Regime 2: Large coefficients (τ²λ² >> c²)
        σ̃² ≈ c²     →  Saturates at slab ceiling

    The slab scale c acts as a CEILING on variance:
        |W[j,k]| can grow at most to ≈ sqrt(c²) = c
        |Z[n,k]| can grow at most to ≈ sqrt(c²) = c

    Why This Matters
    ----------------
    1. **Over-regularization**: If important features hit ceiling, they're being
       constrained when they shouldn't be.

    2. **Data scale dependence**: c should match data scale!
       - Standardized data (mean=0, std=1): c ≈ 3-5 typical
       - Raw data (arbitrary units): c needs adjustment

    3. **Bug detection**: If loadings >> c, something is wrong!
       - Either c² is being ignored (implementation bug)
       - Or data preprocessing is broken (not standardized)

    Parameters
    ----------
    chain_results : List[Dict]
        MCMC chain results containing 'W', 'Z', and 'samples' with cW, cZ
    slab_scale : float, optional
        Expected slab_scale hyperparameter (typically 3-6).
        If None, attempts to extract from samples.
    saturation_threshold : float, default=0.8
        Fraction of ceiling to consider "saturated"
        (e.g., 0.8 means |W| > 0.8*c triggers warning)

    Returns
    -------
    Dict containing:
        - W_max_per_chain: List[float], maximum |W| per chain
        - Z_max_per_chain: List[float], maximum |Z| per chain (if available)
        - cW_mean: float, mean slab scale for W (if available)
        - cZ_mean: float, mean slab scale for Z (if available)
        - W_saturation_pct: float, percentage of |W| > threshold*c
        - Z_saturation_pct: float, percentage of |Z| > threshold*c
        - W_is_saturated: bool, whether W shows saturation
        - Z_is_saturated: bool, whether Z shows saturation
        - data_scale_issue: bool, whether loadings >> c (BUG!)
        - recommendations: List[str], actionable diagnostics

    Notes
    -----
    Expected behavior for STANDARDIZED data:
    - Most |W| < c: Features properly regularized ✓
    - Few |W| ≈ c: Important features at ceiling (OK if < 10%)
    - Many |W| ≈ c: c too small, increase slab_scale
    - Any |W| >> c: BUG! Data not standardized or c ignored

    Goldilocks Zone for c:
    - c < 2: Over-regularization, even important features shrunk
    - c ≈ 3-6: Typical for standardized data
    - c > 10: Under-regularization, numerical instability risk

    Examples
    --------
    >>> result = diagnose_slab_saturation(chain_results, slab_scale=5.5)
    >>> if result['W_is_saturated']:
    ...     print(f"Warning: {result['W_saturation_pct']:.1%} of loadings saturated")
    ...     print(f"Consider increasing slab_scale from {result['cW_mean']:.2f}")
    >>> if result['data_scale_issue']:
    ...     print("CRITICAL: Loadings >> slab scale → data preprocessing bug!")
    """
    logger.info("=" * 80)
    logger.info("DIAGNOSE SLAB SATURATION - START")
    logger.info("=" * 80)

    n_chains = len(chain_results)
    W_max_per_chain = []
    Z_max_per_chain = []
    cW_values = []
    cZ_values = []

    has_Z = all('Z' in result and result['Z'] is not None for result in chain_results)
    has_samples = all('samples' in result for result in chain_results)

    # Extract maximum absolute values and slab scales
    for chain_idx, result in enumerate(chain_results):
        W = result['W']
        W_max = np.max(np.abs(W))
        W_max_per_chain.append(W_max)

        if has_Z:
            Z = result['Z']
            Z_max = np.max(np.abs(Z))
            Z_max_per_chain.append(Z_max)

        # Try to extract cW and cZ from samples
        if has_samples and 'samples' in result:
            samples = result['samples']

            if 'cW' in samples:
                cW_samples = samples['cW']
                # cW has shape (n_samples, M, K) - take mean
                cW_mean_chain = np.mean(cW_samples)
                cW_values.append(cW_mean_chain)

            if 'cZ' in samples:
                cZ_samples = samples['cZ']
                cZ_mean_chain = np.mean(cZ_samples)
                cZ_values.append(cZ_mean_chain)

    # Compute mean slab scales across chains
    cW_mean = np.mean(cW_values) if cW_values else slab_scale
    cZ_mean = np.mean(cZ_values) if cZ_values else slab_scale

    # If still None, use default
    if cW_mean is None:
        logger.warning("⚠️  Could not extract slab scale, using default c=5.0")
        cW_mean = 5.0
        cZ_mean = 5.0

    # Compute saturation metrics
    W_max_overall = np.max(W_max_per_chain)
    W_ceiling = saturation_threshold * cW_mean

    # Count saturated elements across all chains
    W_saturated_count = 0
    W_total_count = 0
    for result in chain_results:
        W = result['W']
        W_saturated_count += np.sum(np.abs(W) > W_ceiling)
        W_total_count += W.size

    W_saturation_pct = W_saturated_count / W_total_count if W_total_count > 0 else 0.0

    # Similar for Z
    Z_max_overall = np.max(Z_max_per_chain) if Z_max_per_chain else np.nan
    Z_ceiling = saturation_threshold * cZ_mean if cZ_mean else np.nan

    if has_Z and cZ_mean:
        Z_saturated_count = 0
        Z_total_count = 0
        for result in chain_results:
            Z = result['Z']
            Z_saturated_count += np.sum(np.abs(Z) > Z_ceiling)
            Z_total_count += Z.size
        Z_saturation_pct = Z_saturated_count / Z_total_count if Z_total_count > 0 else 0.0
    else:
        Z_saturation_pct = np.nan

    # Determine if saturated
    W_is_saturated = W_saturation_pct > 0.1  # >10% saturated
    Z_is_saturated = Z_saturation_pct > 0.1 if not np.isnan(Z_saturation_pct) else False

    # CRITICAL: Check for data scale issues (loadings way bigger than c)
    data_scale_issue = W_max_overall > 2.0 * cW_mean

    # Generate recommendations
    recommendations = []

    logger.info(f"\n📊 Slab Saturation Diagnostics ({n_chains} chains):")
    logger.info(f"\nSlab Scale Parameters:")
    if cW_values:
        logger.info(f"  cW (loadings): mean={cW_mean:.2f}, std={np.std(cW_values):.2f}")
    else:
        logger.info(f"  cW (loadings): {cW_mean:.2f} (default)")

    if cZ_values:
        logger.info(f"  cZ (scores): mean={cZ_mean:.2f}, std={np.std(cZ_values):.2f}")
    elif cZ_mean:
        logger.info(f"  cZ (scores): {cZ_mean:.2f} (default)")

    logger.info(f"\nMaximum Absolute Values:")
    logger.info(f"  max|W| across chains: {W_max_overall:.2f}")
    logger.info(f"  Saturation ceiling (W): {W_ceiling:.2f} ({saturation_threshold*100:.0f}% of c)")

    if has_Z:
        logger.info(f"  max|Z| across chains: {Z_max_overall:.2f}")
        logger.info(f"  Saturation ceiling (Z): {Z_ceiling:.2f}")

    logger.info(f"\nSaturation Analysis:")
    logger.info(f"  W elements > ceiling: {W_saturation_pct:.1%}")

    if has_Z and not np.isnan(Z_saturation_pct):
        logger.info(f"  Z elements > ceiling: {Z_saturation_pct:.1%}")

    # Critical bug check
    if data_scale_issue:
        logger.error("=" * 80)
        logger.error("🔴 CRITICAL: DATA SCALE ISSUE DETECTED!")
        logger.error("=" * 80)
        logger.error(f"  max|W| = {W_max_overall:.2f} >> 2*c = {2*cW_mean:.2f}")
        logger.error("")
        logger.error("This indicates ONE of two bugs:")
        logger.error("  1. Data is NOT standardized (mean=0, std=1)")
        logger.error("  2. Slab regularization is being IGNORED in implementation")
        logger.error("")
        logger.error("Expected for standardized data: max|W| < 3*c")
        logger.error(f"Observed: max|W| = {W_max_overall:.2f}")
        logger.error("")
        logger.error("ACTION REQUIRED:")
        logger.error("  1. Check data preprocessing - verify standardization")
        logger.error("  2. Check model implementation - verify c² is applied")
        logger.error("  3. Review lines 148-150 in core/run_analysis.py")
        logger.error("=" * 80)

        recommendations.append("CRITICAL: Data scale issue - loadings >> slab scale")
        recommendations.append("ACTION: Verify data standardization and model implementation")

    elif W_is_saturated:
        logger.warning("\n⚠️  SATURATION DETECTED (W)")
        logger.warning(f"  {W_saturation_pct:.1%} of loadings exceed {saturation_threshold*100:.0f}% of slab ceiling")
        logger.warning(f"  This means model wants LARGER coefficients but c={cW_mean:.2f} constrains them")
        logger.warning("")
        logger.warning("Interpretation:")
        if W_saturation_pct > 0.3:
            logger.warning("  • >30% saturated: c is TOO SMALL for this data")
            logger.warning(f"  • Consider increasing slab_scale: {cW_mean:.1f} → {cW_mean*1.5:.1f}")
            recommendations.append(f"Increase slab_scale from {cW_mean:.1f} to {cW_mean*1.5:.1f}")
        else:
            logger.warning("  • 10-30% saturated: Borderline, monitor closely")
            logger.warning("  • These are likely the most important features")
            recommendations.append("Monitor saturation - consider slightly larger slab_scale")

    else:
        logger.info("\n✅ GOOD: No significant saturation detected")
        if W_saturation_pct > 0 and W_saturation_pct < 0.05:
            logger.info(f"  {W_saturation_pct:.1%} of loadings near ceiling (expected for important features)")
            recommendations.append("Slab scale is appropriate - minor saturation is normal")
        else:
            logger.info("  Loadings well below ceiling - slab working as intended")
            recommendations.append("Slab scale is appropriate")

    # Check if c is in the Goldilocks zone
    if cW_mean < 2:
        logger.warning(f"\n⚠️  CAUTION: c={cW_mean:.2f} < 2 may over-regularize")
        logger.warning("  Even important features will be aggressively shrunk")
        recommendations.append("c < 2: Consider increasing to 3-5 for better flexibility")
    elif cW_mean > 10:
        logger.warning(f"\n⚠️  CAUTION: c={cW_mean:.2f} > 10 may under-regularize")
        logger.warning("  Little effective regularization, risk of numerical instability")
        recommendations.append("c > 10: Consider decreasing to 5-8 for stability")
    else:
        logger.info(f"\n✅ GOOD: c={cW_mean:.2f} in Goldilocks zone (2-10)")

    # Build result
    result = {
        'W_max_per_chain': W_max_per_chain,
        'Z_max_per_chain': Z_max_per_chain if Z_max_per_chain else None,
        'cW_mean': float(cW_mean) if cW_mean else None,
        'cZ_mean': float(cZ_mean) if cZ_mean else None,
        'W_max_overall': float(W_max_overall),
        'Z_max_overall': float(Z_max_overall) if not np.isnan(Z_max_overall) else None,
        'W_ceiling': float(W_ceiling),
        'Z_ceiling': float(Z_ceiling) if not np.isnan(Z_ceiling) else None,
        'W_saturation_pct': float(W_saturation_pct),
        'Z_saturation_pct': float(Z_saturation_pct) if not np.isnan(Z_saturation_pct) else None,
        'W_is_saturated': W_is_saturated,
        'Z_is_saturated': Z_is_saturated,
        'data_scale_issue': data_scale_issue,
        'saturation_threshold': saturation_threshold,
        'recommendations': recommendations,
    }

    logger.info("\n💡 Recommendations:")
    for rec in recommendations:
        logger.info(f"  - {rec}")

    logger.info("=" * 80)
    logger.info("DIAGNOSE SLAB SATURATION - COMPLETED")
    logger.info("=" * 80)

    return result


def _compute_consensus_loadings(
    W_chain_avg: List[np.ndarray],
    per_factor_matches: List[Dict],
    stable_factors: List[int],
) -> np.ndarray:
    """Compute consensus factor loadings by averaging matched factors across chains.

    CRITICAL: Aligns factor signs before averaging to prevent attenuation.
    Factors can flip signs (W → -W) and still represent the same model.
    Without sign alignment, averaging [+0.5, -0.5, +0.5, +0.5] gives +0.25 (WRONG!)
    With sign alignment, averaging [+0.5, +0.5, +0.5, +0.5] gives +0.5 (CORRECT!)

    Parameters
    ----------
    W_chain_avg : List[np.ndarray]
        List of averaged W matrices from each chain, shape (D, K)
    per_factor_matches : List[Dict]
        Per-factor matching information
    stable_factors : List[int]
        Indices of stable factors

    Returns
    -------
    np.ndarray
        Consensus loadings, shape (D, len(stable_factors))
    """
    D, K = W_chain_avg[0].shape
    n_stable = len(stable_factors)
    consensus_W = np.zeros((D, n_stable))

    for i, factor_idx in enumerate(stable_factors):
        factor_match_info = per_factor_matches[factor_idx]
        matched_indices = factor_match_info["matched_in_chains"]

        # Collect loadings from all chains where this factor matched
        matched_loadings = []
        for chain_idx, matched_k in enumerate(matched_indices):
            if matched_k is not None:
                matched_loadings.append(W_chain_avg[chain_idx][:, matched_k])

        if matched_loadings:
            # CRITICAL FIX: Align signs before averaging
            # Use first chain as reference, flip others if needed
            reference = matched_loadings[0]
            aligned_loadings = [reference]  # Reference doesn't need alignment

            for loading in matched_loadings[1:]:
                # Check if this loading has opposite sign from reference
                correlation = np.dot(reference, loading)
                if correlation < 0:
                    # Flip sign to match reference
                    aligned_loadings.append(-loading)
                else:
                    aligned_loadings.append(loading)

            # Use MEDIAN instead of MEAN for robustness
            # Median is robust to outliers and equals mode for symmetric posteriors
            # Avoids attenuation from averaging multimodal distributions
            consensus_W[:, i] = np.median(aligned_loadings, axis=0)

    return consensus_W


def _compute_consensus_scores(
    Z_chain_avg: List[np.ndarray],
    per_factor_matches: List[Dict],
    stable_factors: List[int],
) -> Dict:
    """Compute consensus factor scores with posterior uncertainty quantification.

    CRITICAL: Aligns factor signs before aggregation to prevent attenuation.
    Factors can flip signs (Z → -Z) and still represent the same model.
    Without sign alignment, patient scores can be severely attenuated or wrong.
    Example: averaging [+2.5, -2.5, +2.5, +2.5] gives +1.25 (WRONG!)
             should be [+2.5, +2.5, +2.5, +2.5] → +2.5 (CORRECT!)

    IMPORTANT: Returns UNCERTAINTY along with point estimate!
    Z scores are indeterminate (given W, infinitely many Z are valid).
    Bayesian approach quantifies this uncertainty via posterior distribution.

    Parameters
    ----------
    Z_chain_avg : List[np.ndarray]
        Factor scores from each chain, each shape (N, K)
    per_factor_matches : List[Dict]
        Matching information for each factor from assess_factor_stability_cosine
    stable_factors : List[int]
        Indices of stable factors

    Returns
    -------
    Dict containing:
        - median: np.ndarray (N, n_stable), posterior median (robust point estimate)
        - std: np.ndarray (N, n_stable), posterior standard deviation (uncertainty)
        - q025: np.ndarray (N, n_stable), 2.5th percentile (lower 95% CI)
        - q975: np.ndarray (N, n_stable), 97.5th percentile (upper 95% CI)
        - n_chains_per_factor: list of int, number of chains contributing to each factor
    """
    N, K = Z_chain_avg[0].shape
    n_stable = len(stable_factors)

    consensus_Z_median = np.zeros((N, n_stable))
    consensus_Z_std = np.zeros((N, n_stable))
    consensus_Z_q025 = np.zeros((N, n_stable))
    consensus_Z_q975 = np.zeros((N, n_stable))
    n_chains_per_factor = []

    for i, factor_idx in enumerate(stable_factors):
        factor_match_info = per_factor_matches[factor_idx]
        matched_indices = factor_match_info["matched_in_chains"]

        # Collect scores from all chains where this factor matched
        matched_scores = []
        for chain_idx, matched_k in enumerate(matched_indices):
            if matched_k is not None:
                matched_scores.append(Z_chain_avg[chain_idx][:, matched_k])

        if matched_scores:
            # CRITICAL FIX: Align signs before aggregation
            # Use first chain as reference, flip others if needed
            reference = matched_scores[0]
            aligned_scores = [reference]  # Reference doesn't need alignment

            for scores in matched_scores[1:]:
                # Check if these scores have opposite sign from reference
                correlation = np.dot(reference, scores)
                if correlation < 0:
                    # Flip sign to match reference
                    aligned_scores.append(-scores)
                else:
                    aligned_scores.append(scores)

            # Convert to array for easy computation: (n_chains, N)
            aligned_scores_array = np.array(aligned_scores)
            n_chains_per_factor.append(len(aligned_scores))

            # Compute posterior statistics across chains
            # Use MEDIAN (robust to outliers) instead of MEAN
            consensus_Z_median[:, i] = np.median(aligned_scores_array, axis=0)
            consensus_Z_std[:, i] = np.std(aligned_scores_array, axis=0, ddof=1)
            consensus_Z_q025[:, i] = np.percentile(aligned_scores_array, 2.5, axis=0)
            consensus_Z_q975[:, i] = np.percentile(aligned_scores_array, 97.5, axis=0)

    logger.info(f"Consensus Z computed with uncertainty quantification")
    logger.info(f"  Median Z: shape {consensus_Z_median.shape}")
    logger.info(f"  Mean std per subject: {consensus_Z_std.mean():.3f}")
    logger.info(f"  Max std per subject: {consensus_Z_std.max():.3f}")

    # Log subjects with high uncertainty
    high_uncertainty_threshold = np.percentile(consensus_Z_std, 95)
    high_uncertainty_subjects = np.any(consensus_Z_std > high_uncertainty_threshold, axis=1)
    n_high_uncertainty = high_uncertainty_subjects.sum()

    if n_high_uncertainty > 0:
        logger.warning(f"  ⚠️  {n_high_uncertainty} subjects have high uncertainty (>95th percentile)")
        logger.warning(f"     Consider: More chains, longer sampling, or model issues")

    return {
        "median": consensus_Z_median,
        "std": consensus_Z_std,
        "q025": consensus_Z_q025,
        "q975": consensus_Z_q975,
        "n_chains_per_factor": n_chains_per_factor,
    }


def count_effective_factors(
    W: np.ndarray,
    sparsity_threshold: float = 0.001,
    min_nonzero_pct: float = 0.01,
) -> Dict:
    """Count how many factors have meaningful (non-zero) loadings.

    A factor is "effective" if sparsity priors haven't shrunk it to zero.
    This helps quantify how many of the K latent factors are actually used
    by the model.

    Criteria for effective factor:
    1. Maximum loading magnitude > sparsity_threshold
    2. At least min_nonzero_pct fraction of loadings > sparsity_threshold

    Parameters
    ----------
    W : np.ndarray
        Factor loading matrix, shape (D, K) or list of (D_m, K)
    sparsity_threshold : float, default=0.001
        Minimum loading magnitude to be considered non-zero.
        Adjusted for standardized data (0.001 = 0.1% of a standard deviation).
    min_nonzero_pct : float, default=0.01
        Minimum fraction of features with non-zero loadings (at least 1% of features).

    Returns
    -------
    Dict
        - n_effective: Number of effective factors
        - effective_indices: List of effective factor indices
        - shrinkage_rate: Fraction of factors shrunk to zero (1 - n_effective/K)
        - per_factor_stats: List of dicts with per-factor statistics

    Examples
    --------
    >>> W = np.random.randn(100, 20)
    >>> effective = count_effective_factors(W, sparsity_threshold=0.001)
    >>> print(f"{effective['n_effective']}/{W.shape[1]} factors are effective")
    """
    # Handle multi-view case
    # CRITICAL: Do NOT vstack! This double-counts variance.
    # Factor variance is SHARED across views (through Z), not additive.
    # Report per-view statistics separately.

    is_multiview = isinstance(W, list)

    if is_multiview:
        W_list = W
        M = len(W_list)
        K = W_list[0].shape[1]
        D_total = sum(W_view.shape[0] for W_view in W_list)
        logger.info(f"Counting effective factors in multi-view W: {M} views, {D_total} total features, K={K}")
    else:
        W_list = [W]
        M = 1
        K = W.shape[1]
        D_total = W.shape[0]
        logger.info(f"Counting effective factors in W matrix: D={D_total}, K={K}")

    effective_factors = []
    per_factor_stats = []

    for k in range(K):
        # Compute PER-VIEW statistics to avoid double-counting
        per_view_stats = []

        for view_idx, W_view in enumerate(W_list):
            loadings_view = W_view[:, k]
            D_view = W_view.shape[0]

            # Statistics for this view
            max_loading_view = float(np.max(np.abs(loadings_view)))
            mean_loading_view = float(np.mean(np.abs(loadings_view)))
            nonzero_count_view = int(np.sum(np.abs(loadings_view) > sparsity_threshold))
            nonzero_pct_view = nonzero_count_view / D_view

            per_view_stats.append({
                "view_idx": view_idx,
                "max_loading": max_loading_view,
                "mean_loading": mean_loading_view,
                "nonzero_count": nonzero_count_view,
                "nonzero_pct": nonzero_pct_view,
            })

        # Aggregate across views using WEIGHTED average (not sum!)
        # Weight by number of features per view
        D_views = np.array([W_view.shape[0] for W_view in W_list])
        weights = D_views / D_total

        # Weighted average of statistics
        max_loading = max([vs["max_loading"] for vs in per_view_stats])  # Global max
        mean_loading = sum(vs["mean_loading"] * w for vs, w in zip(per_view_stats, weights))
        nonzero_pct = sum(vs["nonzero_pct"] * w for vs, w in zip(per_view_stats, weights))

        # Factor is effective if ANY view shows strong loadings
        # (Factor can be specific to one view in multi-view GFA)
        is_effective_per_view = [
            (vs["max_loading"] > sparsity_threshold) and (vs["nonzero_pct"] > min_nonzero_pct)
            for vs in per_view_stats
        ]
        is_effective = any(is_effective_per_view)

        per_factor_stats.append({
            "factor_index": int(k),
            "max_loading": max_loading,
            "mean_loading": mean_loading,
            "nonzero_pct": float(nonzero_pct),
            "is_effective": bool(is_effective),
            "per_view_stats": per_view_stats,  # NEW: Keep per-view breakdown
            "is_effective_per_view": is_effective_per_view,  # NEW: Which views are active
        })

        if is_effective:
            effective_factors.append(k)
            active_views = [i for i, eff in enumerate(is_effective_per_view) if eff]
            logger.debug(
                f"Factor {k}: EFFECTIVE (max={max_loading:.3f}, "
                f"nonzero={nonzero_pct:.1%}, active in views {active_views})"
            )
        else:
            logger.debug(
                f"Factor {k}: shrunk (max={max_loading:.3f}, "
                f"nonzero={nonzero_pct:.1%})"
            )

    shrinkage_rate = 1 - len(effective_factors) / K if K > 0 else 0.0

    result = {
        "n_effective": len(effective_factors),
        "effective_indices": effective_factors,
        "total_factors": K,
        "shrinkage_rate": float(shrinkage_rate),
        "per_factor_stats": per_factor_stats,
        "sparsity_threshold": sparsity_threshold,
        "min_nonzero_pct": min_nonzero_pct,
    }

    logger.info(f"Effective factor analysis:")
    logger.info(f"  - {len(effective_factors)}/{K} factors are effective")
    logger.info(f"  - Shrinkage rate: {shrinkage_rate:.1%}")
    logger.info(f"  - Effective factor indices: {effective_factors}")

    return result


def count_effective_factors_from_samples(
    W_samples: np.ndarray,
    sparsity_threshold: float = 0.001,
    min_nonzero_pct: float = 0.01,
    credible_interval: float = 0.95,
) -> Dict:
    """Count effective factors using full posterior samples (not just mean).

    This is more robust than using posterior means because it checks whether
    factors are CONSISTENTLY non-zero across the posterior, not just whether
    the mean is non-zero.

    A factor is "effective" if:
    1. Median absolute loading > sparsity_threshold
    2. Lower bound of credible interval > sparsity_threshold (consistently non-zero)
    3. At least min_nonzero_pct fraction of features have consistently non-zero loadings

    Parameters
    ----------
    W_samples : np.ndarray
        Factor loading samples, shape (n_samples, D, K) or (n_samples, D_m, K) for single view
        For multi-view, should be concatenated: (n_samples, sum(D_m), K)
    sparsity_threshold : float, default=0.001
        Minimum loading magnitude to be considered non-zero
    min_nonzero_pct : float, default=0.01
        Minimum fraction of features with non-zero loadings
    credible_interval : float, default=0.95
        Credible interval to use (e.g., 0.95 for 95% CI)

    Returns
    -------
    Dict
        Same structure as count_effective_factors, plus:
        - median_loadings_per_factor: median absolute loading for each factor
        - ci_lower_per_factor: lower bound of credible interval for each factor
    """
    n_samples, D, K = W_samples.shape
    logger.info(f"Counting effective factors from {n_samples} posterior samples: D={D}, K={K}")

    ci_lower_pct = (1 - credible_interval) / 2 * 100
    ci_upper_pct = (1 - ci_lower_pct / 100) * 100

    effective_factors = []
    per_factor_stats = []
    median_loadings = []
    ci_lower_values = []

    for k in range(K):
        loadings_abs = np.abs(W_samples[:, :, k])  # (n_samples, D)

        # Compute median absolute loading across samples and features
        median_abs_loading = float(np.median(loadings_abs))
        median_loadings.append(median_abs_loading)

        # Compute credible interval for maximum loading
        max_loadings_per_sample = np.max(loadings_abs, axis=1)  # (n_samples,)
        ci_lower = float(np.percentile(max_loadings_per_sample, ci_lower_pct))
        ci_upper = float(np.percentile(max_loadings_per_sample, ci_upper_pct))
        ci_lower_values.append(ci_lower)

        # Check how many features are consistently non-zero
        # For each feature, check if it's non-zero in most samples
        feature_nonzero_rate = np.mean(loadings_abs > sparsity_threshold, axis=0)  # (D,)
        consistently_nonzero = np.sum(feature_nonzero_rate > 0.5)  # Features non-zero in >50% of samples
        nonzero_pct = consistently_nonzero / D

        # Additional statistics
        mean_abs_loading = float(np.mean(loadings_abs))
        std_abs_loading = float(np.std(loadings_abs))

        # Factor is effective if:
        # 1. Median loading is above threshold
        # 2. Lower CI is above threshold (consistently non-zero)
        # 3. Sufficient fraction of features are consistently non-zero
        is_effective = (
            median_abs_loading > sparsity_threshold
            and ci_lower > sparsity_threshold
            and nonzero_pct > min_nonzero_pct
        )

        per_factor_stats.append({
            "factor_index": int(k),
            "median_abs_loading": median_abs_loading,
            "mean_abs_loading": mean_abs_loading,
            "std_abs_loading": std_abs_loading,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "consistently_nonzero_count": int(consistently_nonzero),
            "nonzero_pct": float(nonzero_pct),
            "is_effective": bool(is_effective),
        })

        if is_effective:
            effective_factors.append(k)
            logger.debug(
                f"Factor {k}: EFFECTIVE (median={median_abs_loading:.3f}, "
                f"CI=[{ci_lower:.3f}, {ci_upper:.3f}], nonzero={nonzero_pct:.1%})"
            )
        else:
            logger.debug(
                f"Factor {k}: shrunk (median={median_abs_loading:.3f}, "
                f"CI=[{ci_lower:.3f}, {ci_upper:.3f}], nonzero={nonzero_pct:.1%})"
            )

    shrinkage_rate = 1 - len(effective_factors) / K if K > 0 else 0.0

    result = {
        "n_effective": len(effective_factors),
        "effective_indices": effective_factors,
        "total_factors": K,
        "shrinkage_rate": float(shrinkage_rate),
        "per_factor_stats": per_factor_stats,
        "median_loadings_per_factor": median_loadings,
        "ci_lower_per_factor": ci_lower_values,
        "method": "posterior_samples",
        "n_samples_used": n_samples,
    }

    logger.info(f"Effective factor analysis (from samples):")
    logger.info(f"  - {len(effective_factors)}/{K} factors are effective")
    logger.info(f"  - Shrinkage rate: {shrinkage_rate:.1%}")
    logger.info(f"  - Effective factor indices: {effective_factors}")

    return result


def count_effective_factors_from_ard(
    tau_W_samples: np.ndarray,
    precision_threshold: float = 100.0,
    variance_threshold: float = 0.01,
) -> Dict:
    """Count effective factors using ARD precision hyperparameters.

    This is the most principled approach for models with ARD priors, as it
    directly uses the learned precision parameters that control shrinkage.

    A factor is "shrunk" (not effective) if:
    - tau_W[k] > precision_threshold (high precision = shrunk toward zero)
    OR equivalently:
    - 1/tau_W[k] < variance_threshold (low variance = shrunk toward zero)

    Parameters
    ----------
    tau_W_samples : np.ndarray
        ARD precision parameter samples, shape (n_samples, K) or (n_chains, n_samples, K)
        If 3D, will be flattened to (n_chains * n_samples, K)
    precision_threshold : float, default=100.0
        Factors with median tau_W > this are considered shrunk
    variance_threshold : float, default=0.01
        Factors with median 1/tau_W < this are considered shrunk

    Returns
    -------
    Dict
        - n_effective: Number of effective factors
        - effective_indices: List of effective factor indices
        - shrinkage_rate: Fraction of factors shrunk
        - per_factor_stats: List of dicts with ARD statistics
        - method: "ard_precision"
    """
    # Handle both 2D and 3D arrays
    if tau_W_samples.ndim == 3:
        n_chains, n_samples, K = tau_W_samples.shape
        tau_W_flat = tau_W_samples.reshape(-1, K)  # (n_chains * n_samples, K)
        logger.info(f"Counting effective factors from ARD priors: {n_chains} chains, {n_samples} samples, K={K}")
    else:
        tau_W_flat = tau_W_samples
        K = tau_W_flat.shape[1]
        logger.info(f"Counting effective factors from ARD priors: {tau_W_flat.shape[0]} samples, K={K}")

    effective_factors = []
    per_factor_stats = []

    for k in range(K):
        tau_k = tau_W_flat[:, k]  # Precision samples for factor k

        # Compute statistics
        median_tau = float(np.median(tau_k))
        mean_tau = float(np.mean(tau_k))
        std_tau = float(np.std(tau_k))

        # Variance = 1 / precision
        variance_k = 1.0 / tau_k
        median_variance = float(np.median(variance_k))
        mean_variance = float(np.mean(variance_k))

        # Factor is shrunk if precision is high OR variance is low
        is_shrunk = (median_tau > precision_threshold) or (median_variance < variance_threshold)
        is_effective = not is_shrunk

        per_factor_stats.append({
            "factor_index": int(k),
            "median_precision": median_tau,
            "mean_precision": mean_tau,
            "std_precision": std_tau,
            "median_variance": median_variance,
            "mean_variance": mean_variance,
            "is_effective": bool(is_effective),
            "is_shrunk": bool(is_shrunk),
        })

        if is_effective:
            effective_factors.append(k)
            logger.debug(
                f"Factor {k}: EFFECTIVE (median tau={median_tau:.2f}, "
                f"median var={median_variance:.4f})"
            )
        else:
            logger.debug(
                f"Factor {k}: SHRUNK (median tau={median_tau:.2f}, "
                f"median var={median_variance:.4f})"
            )

    shrinkage_rate = 1 - len(effective_factors) / K if K > 0 else 0.0

    result = {
        "n_effective": len(effective_factors),
        "effective_indices": effective_factors,
        "total_factors": K,
        "shrinkage_rate": float(shrinkage_rate),
        "per_factor_stats": per_factor_stats,
        "method": "ard_precision",
        "precision_threshold": precision_threshold,
        "variance_threshold": variance_threshold,
    }

    logger.info(f"Effective factor analysis (from ARD):")
    logger.info(f"  - {len(effective_factors)}/{K} factors are effective")
    logger.info(f"  - Shrinkage rate: {shrinkage_rate:.1%}")
    logger.info(f"  - Effective factor indices: {effective_factors}")

    return result


def compute_communality(
    W_consensus: np.ndarray,
    sigma_samples: np.ndarray,
    data_is_standardized: bool = True,
    heywood_tolerance: float = 0.1,
) -> Dict:
    """Compute communality (h²) and uniqueness (u²) for each feature.

    Communality measures the proportion of variance in each feature explained
    by the common factors. Critical for understanding model quality and detecting
    improper solutions (Heywood cases).

    Mathematical Definition:
    ----------------------
    For feature j:
        h²[j] = sum_k(W[j,k]²)     # Variance explained by factors
        u²[j] = Ψ[j]               # Unique/residual variance

    For STANDARDIZED data (Var(X[j]) = 1):
        h²[j] + u²[j] ≈ 1

    Heywood Case (Improper Solution):
        h²[j] > 1 indicates model failure

    Parameters
    ----------
    W_consensus : np.ndarray
        Consensus factor loadings, shape (D, K) or list of (D_m, K) for multi-view
    sigma_samples : np.ndarray
        Noise precision samples, shape (n_samples, M) where M = number of views
        NOTE: Uniqueness = 1/sigma (variance = 1/precision)
    data_is_standardized : bool, default=True
        Whether input data was standardized (Var=1)
        If True, will check h² + u² ≈ 1
    heywood_tolerance : float, default=0.1
        Tolerance for Heywood case detection (h² > 1 + tolerance)

    Returns
    -------
    Dict containing:
        - communalities: np.ndarray (D,), h² values
        - uniquenesses: np.ndarray (D,), u² values
        - total_variance_explained: np.ndarray (D,), h² + u² (should ≈ 1 if standardized)
        - n_heywood_cases: int, number of Heywood cases
        - heywood_indices: list, indices of Heywood cases
        - low_communality_indices: list, features with h² < 0.3
        - per_view_stats: list of dicts with per-view breakdown

    Raises
    ------
    ValueError
        If Heywood cases detected (h² > 1 + tolerance)

    Examples
    --------
    >>> W = np.random.randn(100, 5) * 0.3
    >>> sigma = np.random.gamma(2, 2, size=(1000, 1))
    >>> result = compute_communality(W, sigma)
    >>> print(f"Mean communality: {result['communalities'].mean():.3f}")

    References
    ----------
    - Harman 1976: "Modern Factor Analysis"
    - Gorsuch 1983: "Factor Analysis" (2nd ed), Chapter 9
    - Heywood 1931: "On finite sequences of real numbers"
    """
    logger.info("=" * 80)
    logger.info("COMPUTING COMMUNALITY & UNIQUENESS")
    logger.info("=" * 80)

    # Handle multi-view case
    is_multiview = isinstance(W_consensus, list)

    if is_multiview:
        W_list = W_consensus
        M = len(W_list)
        K = W_list[0].shape[1]
        D_per_view = [W_view.shape[0] for W_view in W_list]
        D_total = sum(D_per_view)
        logger.info(f"Multi-view: {M} views, {D_total} total features, K={K}")
    else:
        W_list = [W_consensus]
        M = 1
        K = W_consensus.shape[1]
        D_total = W_consensus.shape[0]
        logger.info(f"Single-view: D={D_total}, K={K}")

    # Check sigma_samples shape
    n_samples, M_sigma = sigma_samples.shape
    if M_sigma != M:
        raise ValueError(f"sigma_samples has {M_sigma} views but W has {M} views!")

    logger.info(f"Using {n_samples} posterior samples for sigma")

    # Compute communalities (h²) per view
    communalities = np.zeros(D_total)
    uniquenesses = np.zeros(D_total)
    per_view_stats = []

    start_idx = 0
    for view_idx, W_view in enumerate(W_list):
        D_view = W_view.shape[0]
        end_idx = start_idx + D_view

        logger.info(f"\nView {view_idx}: {D_view} features")

        # Communality: h²[j] = sum_k(W[j,k]²)
        h_squared_view = np.sum(W_view**2, axis=1)  # (D_view,)
        communalities[start_idx:end_idx] = h_squared_view

        # Uniqueness: u²[j] = 1/sigma[view]
        # Take posterior median of 1/sigma
        sigma_view = sigma_samples[:, view_idx]  # (n_samples,)
        u_squared_view = np.median(1.0 / sigma_view)  # Scalar (same for all features in view)
        uniquenesses[start_idx:end_idx] = u_squared_view

        # Statistics for this view
        mean_h2 = h_squared_view.mean()
        median_h2 = np.median(h_squared_view)
        max_h2 = h_squared_view.max()
        min_h2 = h_squared_view.min()

        per_view_stats.append({
            "view_idx": view_idx,
            "n_features": D_view,
            "mean_communality": float(mean_h2),
            "median_communality": float(median_h2),
            "max_communality": float(max_h2),
            "min_communality": float(min_h2),
            "uniqueness": float(u_squared_view),
            "mean_total_variance": float(mean_h2 + u_squared_view),
        })

        logger.info(f"  Communality (h²): mean={mean_h2:.3f}, median={median_h2:.3f}, range=[{min_h2:.3f}, {max_h2:.3f}]")
        logger.info(f"  Uniqueness (u²): {u_squared_view:.3f}")
        logger.info(f"  Total variance: {mean_h2 + u_squared_view:.3f} (should ≈ 1.0 if standardized)")

        start_idx = end_idx

    # Overall statistics
    total_variance_explained = communalities + uniquenesses

    # Detect Heywood cases (h² > 1)
    heywood_mask = communalities > (1.0 + heywood_tolerance)
    n_heywood = heywood_mask.sum()
    heywood_indices = np.where(heywood_mask)[0].tolist()

    # Detect low communality features (poorly explained)
    low_comm_mask = communalities < 0.3
    low_comm_indices = np.where(low_comm_mask)[0].tolist()

    logger.info("\n" + "=" * 80)
    logger.info("COMMUNALITY SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Mean communality (h²): {communalities.mean():.3f}")
    logger.info(f"Median communality (h²): {np.median(communalities):.3f}")
    logger.info(f"Range: [{communalities.min():.3f}, {communalities.max():.3f}]")
    logger.info(f"Mean uniqueness (u²): {uniquenesses.mean():.3f}")
    logger.info(f"Mean total variance: {total_variance_explained.mean():.3f}")

    if data_is_standardized:
        deviation_from_1 = np.abs(total_variance_explained - 1.0).mean()
        logger.info(f"Mean |h² + u² - 1|: {deviation_from_1:.3f}")
        if deviation_from_1 > 0.2:
            logger.warning(f"  ⚠️  LARGE DEVIATION: Expected ≈0 for standardized data!")
            logger.warning(f"     Possible causes:")
            logger.warning(f"       1. Data not properly standardized")
            logger.warning(f"       2. Model misspecification")
            logger.warning(f"       3. MCMC non-convergence")

    # Heywood case detection
    if n_heywood > 0:
        logger.error(f"\n🚨 HEYWOOD CASES DETECTED: {n_heywood} features")
        logger.error(f"   Indices: {heywood_indices[:10]}{'...' if n_heywood > 10 else ''}")
        logger.error(f"   Max h²: {communalities[heywood_indices].max():.3f}")
        logger.error(f"\n   This indicates an IMPROPER SOLUTION!")
        logger.error(f"   Possible causes:")
        logger.error(f"     1. Data not standardized (Var ≠ 1)")
        logger.error(f"     2. Too many factors (K too large)")
        logger.error(f"     3. Numerical instability")
        logger.error(f"     4. MCMC non-convergence")
        logger.error(f"     5. Prior misspecification")
    else:
        logger.info(f"✓ No Heywood cases detected (all h² ≤ {1.0 + heywood_tolerance:.2f})")

    # Low communality warning
    if len(low_comm_indices) > 0:
        pct_low = 100 * len(low_comm_indices) / D_total
        logger.warning(f"\n⚠️  {len(low_comm_indices)} features ({pct_low:.1f}%) have low communality (h² < 0.3)")
        logger.warning(f"   These features are poorly explained by the factors")
        logger.warning(f"   Consider:")
        logger.warning(f"     1. Adding more factors")
        logger.warning(f"     2. Removing these features")
        logger.warning(f"     3. Checking for preprocessing issues")
    else:
        logger.info(f"✓ All features have adequate communality (h² ≥ 0.3)")

    result = {
        "communalities": communalities,
        "uniquenesses": uniquenesses,
        "total_variance_explained": total_variance_explained,
        "n_heywood_cases": int(n_heywood),
        "heywood_indices": heywood_indices,
        "n_low_communality": len(low_comm_indices),
        "low_communality_indices": low_comm_indices,
        "per_view_stats": per_view_stats,
        "mean_communality": float(communalities.mean()),
        "median_communality": float(np.median(communalities)),
        "mean_uniqueness": float(uniquenesses.mean()),
        "data_is_standardized": data_is_standardized,
    }

    # Raise error if Heywood cases found
    if n_heywood > 0:
        raise ValueError(
            f"Heywood cases detected ({n_heywood} features with h² > 1). "
            f"This indicates an improper solution. Check data standardization and model specification."
        )

    return result


def compute_factor_correlations(
    Z_consensus: np.ndarray,
    orthogonality_threshold: float = 0.3,
) -> Dict:
    """Compute inter-factor correlations to assess orthogonality assumption.

    Factor models often assume orthogonal (uncorrelated) factors via independent priors.
    However, the posterior can yield correlated factors due to:
    1. Likelihood coupling through X = W @ Z.T
    2. Real-world correlations in latent structure
    3. Model misspecification

    High correlations (|ρ| > 0.3) suggest oblique factors, which:
    - Are interpretable (real phenomena are correlated)
    - May require oblique rotation methods
    - Affect variance decomposition (overlap, not additive)

    Parameters
    ----------
    Z_consensus : np.ndarray
        Consensus factor scores, shape (N, K)
    orthogonality_threshold : float, default=0.3
        Threshold for flagging high correlations

    Returns
    -------
    Dict containing:
        - correlation_matrix: np.ndarray (K, K), factor correlation matrix
        - off_diagonal_corr: np.ndarray, all off-diagonal correlations
        - mean_abs_correlation: float, mean |ρ| for off-diagonal
        - max_abs_correlation: float, maximum |ρ| for off-diagonal
        - n_high_correlations: int, number with |ρ| > threshold
        - high_correlation_pairs: list of (i, j, ρ) tuples
        - is_oblique: bool, True if any |ρ| > threshold

    Examples
    --------
    >>> Z = np.random.randn(100, 5)
    >>> result = compute_factor_correlations(Z)
    >>> if result['is_oblique']:
    ...     print("Factors are oblique (correlated)")
    """
    N, K = Z_consensus.shape
    logger.info("=" * 80)
    logger.info("FACTOR CORRELATION ANALYSIS (Orthogonality Check)")
    logger.info("=" * 80)
    logger.info(f"Computing correlations for K={K} factors, N={N} subjects")

    # Compute correlation matrix
    correlation_matrix = np.corrcoef(Z_consensus.T)  # (K, K)

    # Extract off-diagonal correlations
    off_diagonal_mask = ~np.eye(K, dtype=bool)
    off_diagonal_corr = correlation_matrix[off_diagonal_mask]

    # Statistics
    mean_abs_corr = np.mean(np.abs(off_diagonal_corr))
    max_abs_corr = np.max(np.abs(off_diagonal_corr))
    std_abs_corr = np.std(np.abs(off_diagonal_corr))

    # Find high correlations
    high_corr_mask = np.abs(correlation_matrix) > orthogonality_threshold
    high_corr_mask = high_corr_mask & off_diagonal_mask  # Exclude diagonal

    high_correlation_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if high_corr_mask[i, j]:
                high_correlation_pairs.append((i, j, correlation_matrix[i, j]))

    n_high_corr = len(high_correlation_pairs)
    is_oblique = n_high_corr > 0

    # Log results
    logger.info(f"\nCorrelation Statistics:")
    logger.info(f"  Mean |ρ|: {mean_abs_corr:.3f}")
    logger.info(f"  Max |ρ|: {max_abs_corr:.3f}")
    logger.info(f"  Std |ρ|: {std_abs_corr:.3f}")
    logger.info(f"  Range: [{off_diagonal_corr.min():.3f}, {off_diagonal_corr.max():.3f}]")

    if is_oblique:
        logger.warning(f"\n⚠️  OBLIQUE FACTORS DETECTED: {n_high_corr} factor pairs with |ρ| > {orthogonality_threshold}")
        logger.warning(f"   High correlations:")
        for i, j, rho in sorted(high_correlation_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            logger.warning(f"     Factor {i} ↔ Factor {j}: ρ = {rho:+.3f}")
        if n_high_corr > 5:
            logger.warning(f"     ... and {n_high_corr - 5} more pairs")

        logger.warning(f"\n   Implications:")
        logger.warning(f"     1. Factors share common variance (not independent)")
        logger.warning(f"     2. Variance decomposition has overlap (not additive)")
        logger.warning(f"     3. May need oblique rotation methods (promax, oblimin)")
        logger.warning(f"     4. Interpretation: factors capture correlated phenomena")
        logger.warning(f"\n   This is NOT necessarily bad! Real-world latent factors are often correlated.")
        logger.warning(f"   Example: 'motor symptoms' and 'cognitive decline' may correlate in PD progression.")
    else:
        logger.info(f"\n✓ ORTHOGONAL FACTORS: All |ρ| ≤ {orthogonality_threshold}")
        logger.info(f"  Factors are approximately independent")
        logger.info(f"  Variance decomposition is additive")
        logger.info(f"  Consistent with independent prior assumption")

    result = {
        "correlation_matrix": correlation_matrix,
        "off_diagonal_corr": off_diagonal_corr,
        "mean_abs_correlation": float(mean_abs_corr),
        "max_abs_correlation": float(max_abs_corr),
        "std_abs_correlation": float(std_abs_corr),
        "n_high_correlations": n_high_corr,
        "high_correlation_pairs": high_correlation_pairs,
        "is_oblique": bool(is_oblique),
        "orthogonality_threshold": orthogonality_threshold,
    }

    return result


def compute_aligned_rhat(
    samples_per_chain: np.ndarray,
    per_factor_matches: List[Dict],
    reference_chain: int = 0,
) -> Dict:
    """Compute R-hat after aligning chains to account for sign/rotation ambiguity.

    Factor models have inherent sign and rotational indeterminacy, meaning chains
    may converge to equivalent solutions that differ only in sign flips or factor
    ordering. Standard R-hat diagnostics don't account for this and will incorrectly
    indicate poor convergence. This function aligns factors across chains before
    computing R-hat.

    Parameters
    ----------
    samples_per_chain : np.ndarray
        MCMC samples with shape (n_chains, n_samples, D, K) for W
        or (n_chains, n_samples, N, K) for Z
    per_factor_matches : List[Dict]
        Factor matching information from assess_factor_stability_cosine,
        containing "matched_in_chains" for each factor
    reference_chain : int, default=0
        Which chain to use as reference for alignment

    Returns
    -------
    Dict
        - max_rhat_per_factor: np.ndarray of shape (K,) with max R-hat per factor
        - mean_rhat_per_factor: np.ndarray of shape (K,) with mean R-hat per factor
        - max_rhat_overall: float, maximum R-hat across all parameters
        - mean_rhat_overall: float, mean R-hat across all parameters
        - n_aligned: int, number of chains successfully aligned
        - aligned_samples: np.ndarray of shape (n_chains, n_samples, dim1, K) with aligned samples

    Examples
    --------
    >>> # W_samples has shape (4 chains, 10000 samples, 1808 features, 5 factors)
    >>> aligned_rhat = compute_aligned_rhat(W_samples, per_factor_matches)
    >>> print(f"Aligned max R-hat: {aligned_rhat['max_rhat_overall']:.4f}")
    """
    from numpyro.diagnostics import split_gelman_rubin

    n_chains, n_samples, dim1, K = samples_per_chain.shape

    logger.info("Computing aligned R-hat (accounting for factor sign/rotation ambiguity)")
    logger.info(f"  Input shape: {samples_per_chain.shape}")
    logger.info(f"  Reference chain: {reference_chain}")

    # Create aligned samples array
    aligned_samples = np.zeros_like(samples_per_chain)

    # Reference chain stays as is
    aligned_samples[reference_chain] = samples_per_chain[reference_chain]
    n_aligned = 1

    # Track factor matching statistics for summary
    unmatched_factors_per_chain = {}

    # Align each other chain to reference
    for chain_idx in range(n_chains):
        if chain_idx == reference_chain:
            continue

        # Get samples from this chain
        chain_samples = samples_per_chain[chain_idx]  # (n_samples, dim1, K)

        # Create aligned version by reordering and flipping factors
        aligned_chain = np.zeros_like(chain_samples)

        # Track unmatched factors for this chain
        unmatched_factors = []

        # For each factor in reference chain, find its match in this chain
        for ref_k in range(K):
            match_info = per_factor_matches[ref_k]
            matched_indices = match_info["matched_in_chains"]

            # Get which factor index in this chain matches ref_k
            if chain_idx < len(matched_indices):
                matched_k = matched_indices[chain_idx]
            else:
                matched_k = None

            if matched_k is not None:
                # Copy matched factor to aligned position
                factor_samples = chain_samples[:, :, matched_k]  # (n_samples, dim1)

                # Check if we need to flip sign
                # Compare sign of mean loadings/scores to reference
                ref_mean = np.mean(aligned_samples[reference_chain, :, :, ref_k], axis=0)  # (dim1,)
                chain_mean = np.mean(factor_samples, axis=0)  # (dim1,)

                # Cosine similarity to detect sign flip
                from scipy.spatial.distance import cosine
                similarity = 1 - cosine(ref_mean, chain_mean)

                # If similarity is negative, flip the sign
                if similarity < 0:
                    factor_samples = -factor_samples
                    logger.debug(f"  Chain {chain_idx}, factor {matched_k} → ref factor {ref_k} (flipped)")
                else:
                    logger.debug(f"  Chain {chain_idx}, factor {matched_k} → ref factor {ref_k}")

                aligned_chain[:, :, ref_k] = factor_samples
            else:
                # No match found - use zeros (this factor not stable across chains)
                unmatched_factors.append(ref_k)
                aligned_chain[:, :, ref_k] = 0.0

        aligned_samples[chain_idx] = aligned_chain
        n_aligned += 1

        # Store unmatched factors for this chain
        if unmatched_factors:
            unmatched_factors_per_chain[chain_idx] = unmatched_factors

    logger.info(f"  Aligned {n_aligned}/{n_chains} chains to reference")

    # Determine which factors matched across all chains
    # A factor is "matched" if it has a match in ALL non-reference chains
    matched_factors = []
    for ref_k in range(K):
        match_info = per_factor_matches[ref_k]
        matched_indices = match_info["matched_in_chains"]

        # Check if this factor matched in all chains (excluding reference)
        is_fully_matched = True
        for chain_idx in range(n_chains):
            if chain_idx == reference_chain:
                continue
            if chain_idx >= len(matched_indices) or matched_indices[chain_idx] is None:
                is_fully_matched = False
                break

        if is_fully_matched:
            matched_factors.append(ref_k)

    # Calculate factor match rate
    factor_match_rate = len(matched_factors) / K if K > 0 else 0.0

    # Log summary of unmatched factors
    if unmatched_factors_per_chain:
        total_unmatched = sum(len(factors) for factors in unmatched_factors_per_chain.values())
        total_possible = (n_chains - 1) * K  # Exclude reference chain
        match_rate = 1.0 - (total_unmatched / total_possible)

        logger.warning(f"  ⚠️  Factor instability detected:")
        logger.warning(f"      {total_unmatched}/{total_possible} factors unmatched across chains ({match_rate*100:.1f}% match rate)")
        logger.warning(f"      {len(matched_factors)}/{K} factors matched across ALL chains ({factor_match_rate*100:.1f}%)")
        logger.warning(f"      {len(unmatched_factors_per_chain)}/{n_chains-1} chains have unmatched factors")

        # Show per-chain summary (condensed)
        for chain_idx in sorted(unmatched_factors_per_chain.keys()):
            n_unmatched = len(unmatched_factors_per_chain[chain_idx])
            logger.warning(f"      Chain {chain_idx}: {n_unmatched}/{K} factors unmatched")

        logger.warning(f"  ⚠️  This indicates poor convergence - chains exploring different factor structures")

        # Decide whether to compute R-hat
        if len(matched_factors) > 0:
            logger.warning(f"  → Computing R-hat only on {len(matched_factors)} matched factors (more reliable)")
        else:
            logger.warning(f"  → Cannot compute reliable R-hat (no factors matched across all chains)")
    else:
        logger.info(f"  ✓ All factors matched across chains (good stability)")

    # Now compute R-hat only on matched factors (if any)
    if len(matched_factors) > 0:
        # Extract only matched factors for R-hat computation
        matched_samples = aligned_samples[:, :, :, matched_factors]  # (n_chains, n_samples, dim1, n_matched)

        # Reshape to (n_chains, n_samples, -1) for split_gelman_rubin
        matched_flat = matched_samples.reshape(n_chains, n_samples, -1)

        logger.info(f"  Computing split Gelman-Rubin on {len(matched_factors)} matched factors...")
        rhat_values = split_gelman_rubin(matched_flat)  # Shape: (dim1 * n_matched,)

        # Reshape back to (dim1, n_matched) for per-factor analysis
        n_matched = len(matched_factors)
        rhat_matrix = rhat_values.reshape(dim1, n_matched)

        # Compute per-factor statistics (only for matched factors)
        max_rhat_per_factor_matched = np.max(rhat_matrix, axis=0)  # Max R-hat across features for each matched factor
        mean_rhat_per_factor_matched = np.mean(rhat_matrix, axis=0)  # Mean R-hat across features for each matched factor

        max_rhat_overall = float(np.max(rhat_values))
        mean_rhat_overall = float(np.mean(rhat_values))

        logger.info(f"  Aligned R-hat results (matched factors only):")
        logger.info(f"    Max R-hat overall:  {max_rhat_overall:.4f}")
        logger.info(f"    Mean R-hat overall: {mean_rhat_overall:.4f}")

        # Count how many parameters have good convergence
        n_converged = np.sum(rhat_values < 1.1)
        n_total = len(rhat_values)
        convergence_pct = 100 * n_converged / n_total
        logger.info(f"    Parameters with R-hat < 1.1: {n_converged}/{n_total} ({convergence_pct:.1f}%)")

        # Expand per-factor R-hat arrays to include unmatched factors (filled with NaN)
        max_rhat_per_factor = np.full(K, np.nan)
        mean_rhat_per_factor = np.full(K, np.nan)
        for i, factor_idx in enumerate(matched_factors):
            max_rhat_per_factor[factor_idx] = max_rhat_per_factor_matched[i]
            mean_rhat_per_factor[factor_idx] = mean_rhat_per_factor_matched[i]

    else:
        # No matched factors - cannot compute R-hat
        logger.error(f"  ❌ Cannot compute R-hat: no factors matched across all chains")
        max_rhat_overall = np.inf
        mean_rhat_overall = np.inf
        convergence_pct = 0.0
        max_rhat_per_factor = np.full(K, np.inf)
        mean_rhat_per_factor = np.full(K, np.inf)

    return {
        "max_rhat_per_factor": max_rhat_per_factor,
        "mean_rhat_per_factor": mean_rhat_per_factor,
        "max_rhat_overall": max_rhat_overall,
        "mean_rhat_overall": mean_rhat_overall,
        "n_aligned": n_aligned,
        "convergence_rate": convergence_pct / 100,  # Fraction with R-hat < 1.1
        "factor_match_rate": factor_match_rate,  # NEW: Fraction of factors matched across all chains
        "n_matched_factors": len(matched_factors),  # NEW: Number of factors matched
        "matched_factor_indices": matched_factors,  # NEW: Which factors matched
        "aligned_samples": aligned_samples,  # NEW: Aligned samples for R-hat evolution plots
    }


def create_stability_summary_table(stability_results: Dict) -> pd.DataFrame:
    """Create a summary table of factor stability results.

    Parameters
    ----------
    stability_results : Dict
        Output from assess_factor_stability_cosine()

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - Factor_Index
        - Matches (count)
        - Match_Rate (%)
        - Status (Robust/Unstable)
        - Matched_Similarities (mean)
    """
    per_factor = stability_results["per_factor_details"]

    rows = []
    for factor_info in per_factor:
        rows.append({
            "Factor_Index": factor_info["factor_index"],
            "Matches": factor_info["matches"],
            "Match_Rate_%": factor_info["match_rate"] * 100,
            "Status": "Robust" if factor_info["is_robust"] else "Unstable",
            "Mean_Similarity": np.mean([
                s for s in factor_info["matched_similarities"] if s is not None
            ]),
        })

    df = pd.DataFrame(rows)
    return df


def create_effective_factors_summary_table(effective_results: Dict) -> pd.DataFrame:
    """Create a summary table of effective factors.

    Parameters
    ----------
    effective_results : Dict
        Output from count_effective_factors()

    Returns
    -------
    pd.DataFrame
        Summary table with per-factor statistics
    """
    per_factor = effective_results["per_factor_stats"]
    df = pd.DataFrame(per_factor)

    # Round numeric columns for readability
    numeric_cols = ["max_loading", "mean_loading", "median_loading", "std_loading", "nonzero_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)

    return df


def save_stability_results(
    stability_results: Dict,
    effective_results: Dict,
    output_dir: str,
    subject_ids: Optional[List[str]] = None,
    view_names: Optional[List[str]] = None,
    feature_names: Optional[Dict[str, List[str]]] = None,
    Dm: Optional[List[int]] = None,
) -> None:
    """Save factor stability and effectiveness results to files.

    Parameters
    ----------
    stability_results : Dict
        Output from assess_factor_stability_cosine()
    effective_results : Dict
        Output from count_effective_factors()
    output_dir : str
        Directory to save results
    subject_ids : Optional[List[str]]
        Subject/patient IDs for Z score indexing
    view_names : Optional[List[str]]
        Names of views (e.g., ['volume_sn_voxels', 'clinical'])
    feature_names : Optional[Dict[str, List[str]]]
        Feature names per view
    Dm : Optional[List[int]]
        Dimensions per view (e.g., [850, 14] for imaging + clinical)

    Original effective_results parameter docs:
        Output from count_effective_factors()
    output_dir : str
        Directory to save results
    subject_ids : Optional[List[str]], default=None
        Patient/subject IDs for indexing Z scores. If None, uses generic labels.
    view_names : Optional[List[str]], default=None
        Names of data views (e.g., ['clinical', 'imaging']). Used with feature_names.
    feature_names : Optional[Dict[str, List[str]]], default=None
        Feature names for each view. Keys are view names, values are feature name lists.
    """
    from pathlib import Path
    from core.io_utils import save_csv, save_json, save_numpy

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving stability results to {output_path}")

    # Save summary JSON
    summary = {
        "stability": {
            "n_stable_factors": stability_results["n_stable_factors"],
            "total_factors": stability_results["total_factors"],
            "stability_rate": stability_results["stability_rate"],
            "threshold": stability_results["threshold"],
            "min_match_rate": stability_results["min_match_rate"],
            "n_chains": stability_results["n_chains"],
        },
        "effectiveness": {
            "n_effective": effective_results["n_effective"],
            "total_factors": effective_results["total_factors"],
            "shrinkage_rate": effective_results["shrinkage_rate"],
            "sparsity_threshold": effective_results["sparsity_threshold"],
        },
    }
    save_json(summary, output_path / "factor_stability_summary.json", indent=2)

    # Save detailed stability results
    stability_table = create_stability_summary_table(stability_results)
    save_csv(stability_table, output_path / "factor_stability_details.csv", index=False)

    # Save effective factors results
    effective_table = create_effective_factors_summary_table(effective_results)
    save_csv(effective_table, output_path / "effective_factors_details.csv", index=False)

    # Save consensus loadings if available
    if stability_results["consensus_W"] is not None:
        consensus_W = stability_results["consensus_W"]
        # Use all_factor_indices since we now compute consensus for all factors
        factor_indices = stability_results.get("all_factor_indices", stability_results["stable_factor_indices"])
        factor_names = [f"Factor_{i}" for i in factor_indices]
        consensus_W_df = pd.DataFrame(
            consensus_W,
            columns=factor_names,
        )

        # Construct concatenated feature names from all views if available
        if view_names and feature_names:
            concatenated_feature_names = []
            for view_name in view_names:
                if view_name in feature_names:
                    concatenated_feature_names.extend(feature_names[view_name])

            # Use concatenated feature names if the length matches
            if len(concatenated_feature_names) == consensus_W.shape[0]:
                consensus_W_df.index = concatenated_feature_names
            else:
                consensus_W_df.index = [f"Feature_{j}" for j in range(consensus_W.shape[0])]
                logger.warning(
                    f"Feature names length mismatch: {len(concatenated_feature_names)} names "
                    f"vs {consensus_W.shape[0]} rows. Using generic labels."
                )
        else:
            consensus_W_df.index = [f"Feature_{j}" for j in range(consensus_W.shape[0])]

        consensus_W_df.index.name = "Feature"
        save_csv(consensus_W_df, output_path / "consensus_factor_loadings.csv", index=True)
        logger.info(f"  ✅ Saved consensus factor loadings (W): {consensus_W.shape}")

        # Save per-view consensus loadings if Dm is available
        if Dm is not None and view_names is not None:
            logger.info(f"  Saving per-view consensus loadings...")
            start_idx = 0
            for view_idx, (view_name, view_dim) in enumerate(zip(view_names, Dm)):
                end_idx = start_idx + view_dim
                view_W = consensus_W[start_idx:end_idx, :]

                # Create DataFrame for this view
                view_W_df = pd.DataFrame(
                    view_W,
                    columns=factor_names,
                )

                # Use feature names for this view if available
                if feature_names and view_name in feature_names:
                    view_feature_names = feature_names[view_name]
                    if len(view_feature_names) == view_dim:
                        view_W_df.index = view_feature_names
                    else:
                        view_W_df.index = [f"{view_name}_Feature_{j}" for j in range(view_dim)]
                else:
                    view_W_df.index = [f"{view_name}_Feature_{j}" for j in range(view_dim)]

                view_W_df.index.name = "Feature"

                # First, save the standard CSV format (D_m × K: features/voxels × factors)
                # This is the consensus W matrix showing factor loadings
                view_filename = f"consensus_factor_loadings_{view_name}.csv"
                save_csv(view_W_df, output_path / view_filename, index=True)
                logger.info(f"    ✅ Saved {view_name} loadings (W): {view_W.shape}")

                # Check if this is a volume/voxel view
                is_volume_view = "volume" in view_name.lower() or "voxel" in view_name.lower()

                # Additionally, save per-factor reconstructions (N × D_m matrices)
                # Reconstruction = Z[:, k] × W[:, k].T gives subject-specific factor maps
                consensus_Z = stability_results.get("consensus_Z")

                if consensus_Z is not None:
                    if is_volume_view:
                        # Extract ROI name from view_name
                        # e.g., "volume_sn_voxels" -> "SN"
                        roi_name = view_name.replace("volume_", "").replace("_voxels", "")

                        # Capitalize ROI name appropriately
                        if roi_name.lower() == "sn":
                            roi_name = "SN"
                        elif roi_name.lower() == "all":
                            roi_name = "All"
                        else:
                            roi_name = roi_name.capitalize()

                        # Save each factor as a separate TSV file (N subjects × D_m voxels)
                        for factor_idx, factor_col in enumerate(view_W_df.columns):
                            # Get loadings for this factor (D_m × 1)
                            factor_loadings = view_W[:, factor_idx:factor_idx+1]  # Shape: (D_m, 1)

                            # Get scores for this factor (N × 1)
                            factor_scores = consensus_Z[:, factor_idx:factor_idx+1]  # Shape: (N, 1)

                            # Compute reconstruction: outer product gives N × D_m
                            reconstruction = factor_scores @ factor_loadings.T  # Shape: (N, D_m)

                            # Create filename: consensus_loadings_SN_Factor_0.tsv
                            factor_filename = f"consensus_loadings_{roi_name}_{factor_col}.tsv"

                            # Save as TSV without index or header (just data matrix)
                            # Each row = subject, each column = voxel
                            np.savetxt(
                                output_path / factor_filename,
                                reconstruction,
                                delimiter='\t',
                                fmt='%.10f'
                            )
                            logger.info(f"    ✅ Saved {roi_name} {factor_col} reconstruction: {reconstruction.shape[0]} subjects × {reconstruction.shape[1]} voxels")
                    else:
                        # Non-volume view (e.g., clinical): save per-factor reconstructions as CSV
                        for factor_idx, factor_col in enumerate(view_W_df.columns):
                            # Get loadings for this factor (D_m × 1)
                            factor_loadings = view_W[:, factor_idx:factor_idx+1]  # Shape: (D_m, 1)

                            # Get scores for this factor (N × 1)
                            factor_scores = consensus_Z[:, factor_idx:factor_idx+1]  # Shape: (N, 1)

                            # Compute reconstruction: outer product gives N × D_m
                            reconstruction = factor_scores @ factor_loadings.T  # Shape: (N, D_m)

                            # Create DataFrame with feature names as columns
                            reconstruction_df = pd.DataFrame(
                                reconstruction,
                                columns=view_W_df.index  # Use feature names from view_W_df index
                            )

                            # Add subject IDs as row index if available
                            if subject_ids and len(subject_ids) == reconstruction.shape[0]:
                                reconstruction_df.index = subject_ids
                            else:
                                reconstruction_df.index = [f"Subject_{j}" for j in range(reconstruction.shape[0])]

                            # Create filename: consensus_loadings_clinical_Factor_0.csv
                            factor_filename = f"consensus_loadings_{view_name}_{factor_col}.csv"

                            # Save as CSV with header row (feature names) preserved
                            reconstruction_df.to_csv(
                                output_path / factor_filename,
                                index=True  # Keep subject IDs as first column
                            )
                            logger.info(f"    ✅ Saved {view_name} {factor_col} reconstruction: {reconstruction.shape[0]} subjects × {reconstruction.shape[1]} features")
                else:
                    logger.info(f"    ℹ️  Skipping reconstructions for {view_name}: consensus_Z not available")

                start_idx = end_idx

    # Save consensus scores if available
    if stability_results.get("consensus_Z") is not None:
        consensus_Z = stability_results["consensus_Z"]
        # Use all_factor_indices since we now compute consensus for all factors
        factor_indices = stability_results.get("all_factor_indices", stability_results["stable_factor_indices"])
        factor_names = [f"Factor_{i}" for i in factor_indices]

        # Helper function to create DataFrame with proper indexing
        def create_score_df(data, factor_names, subject_ids):
            df = pd.DataFrame(data, columns=factor_names)
            if subject_ids and len(subject_ids) == data.shape[0]:
                df.index = subject_ids
            else:
                df.index = [f"Subject_{j}" for j in range(data.shape[0])]
            df.index.name = "Patient ID"
            return df

        # Save posterior median (primary estimate)
        consensus_Z_df = create_score_df(consensus_Z, factor_names, subject_ids)
        save_csv(consensus_Z_df, output_path / "consensus_factor_scores.csv", index=True)
        logger.info(f"  ✅ Saved consensus factor scores (Z median): {consensus_Z.shape}")

        # Save posterior uncertainty if available (Aspect 6: Z Indeterminacy)
        if stability_results.get("consensus_Z_std") is not None:
            consensus_Z_std = stability_results["consensus_Z_std"]
            consensus_Z_std_df = create_score_df(consensus_Z_std, factor_names, subject_ids)
            save_csv(consensus_Z_std_df, output_path / "consensus_factor_scores_std.csv", index=True)
            logger.info(f"  ✅ Saved consensus Z posterior std: {consensus_Z_std.shape}")

        if stability_results.get("consensus_Z_q025") is not None:
            consensus_Z_q025 = stability_results["consensus_Z_q025"]
            consensus_Z_q025_df = create_score_df(consensus_Z_q025, factor_names, subject_ids)
            save_csv(consensus_Z_q025_df, output_path / "consensus_factor_scores_q025.csv", index=True)
            logger.info(f"  ✅ Saved consensus Z 2.5th percentile (lower 95% CI)")

        if stability_results.get("consensus_Z_q975") is not None:
            consensus_Z_q975 = stability_results["consensus_Z_q975"]
            consensus_Z_q975_df = create_score_df(consensus_Z_q975, factor_names, subject_ids)
            save_csv(consensus_Z_q975_df, output_path / "consensus_factor_scores_q975.csv", index=True)
            logger.info(f"  ✅ Saved consensus Z 97.5th percentile (upper 95% CI)")
            logger.info(f"  📊 Z posterior uncertainty quantified (Aspect 6: Z Indeterminacy)")

    # Save similarity matrix
    if "similarity_matrix" in stability_results:
        similarity_matrix = stability_results["similarity_matrix"]
        save_numpy(
            similarity_matrix,
            output_path / "similarity_matrix.npy",
        )

        # Also save as CSV for easier inspection
        # Average across factors for summary (shape: n_chains x n_chains x K -> n_chains x n_chains)
        if similarity_matrix.ndim == 3:
            similarity_matrix_avg = np.mean(similarity_matrix, axis=2)
        else:
            similarity_matrix_avg = similarity_matrix

        similarity_df = pd.DataFrame(
            similarity_matrix_avg,
            index=[f"Chain_{i}" for i in range(similarity_matrix_avg.shape[0])],
            columns=[f"Chain_{i}" for i in range(similarity_matrix_avg.shape[1])]
        )
        similarity_df.to_csv(output_path / "similarity_matrix.csv")
        logger.info(f"  ✅ Saved similarity matrix: .npy and .csv formats")

    logger.info("Factor stability results saved successfully")
