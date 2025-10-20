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

logger = logging.getLogger(__name__)


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
    stable_factors = []
    per_factor_matches = []

    # Store similarity matrix for all factor pairs across chains
    # Shape: (n_chains, n_chains, K) - similarity of factor k in chain i to best match in chain j
    similarity_matrix = np.zeros((n_chains, n_chains, K))

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
                # Cosine similarity = 1 - cosine distance
                sim = 1 - cosine(ref_factor, factor)
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

    # Step 3: Compute consensus factor loadings and scores for stable factors
    consensus_W = None
    consensus_Z = None
    if stable_factors:
        logger.info(f"Computing consensus loadings for {len(stable_factors)} stable factors")
        consensus_W = _compute_consensus_loadings(
            W_chain_avg, per_factor_matches, stable_factors
        )

        # Compute consensus Z if Z scores are available
        if all(Z is not None for Z in Z_chain_avg):
            logger.info(f"Computing consensus scores for {len(stable_factors)} stable factors")
            consensus_Z = _compute_consensus_scores(
                Z_chain_avg, per_factor_matches, stable_factors
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
    logger.info(f"  - Consensus Z shape: {consensus_Z.shape if consensus_Z is not None else 'None'}")

    result = {
        "n_stable_factors": len(stable_factors),
        "stable_factor_indices": stable_factors,
        "total_factors": K,
        "n_chains": n_chains,
        "threshold": threshold,
        "min_match_rate": min_match_rate,
        "per_factor_details": per_factor_matches,
        "similarity_matrix": similarity_matrix,
        "consensus_W": consensus_W,
        "consensus_Z": consensus_Z,
        "stability_rate": len(stable_factors) / K if K > 0 else 0.0,
    }

    logger.info(f"Factor stability analysis complete:")
    logger.info(f"  - {len(stable_factors)}/{K} factors are robust ({result['stability_rate']:.1%})")
    logger.info(f"  - Stable factor indices: {stable_factors}")

    return result


def _compute_consensus_loadings(
    W_chain_avg: List[np.ndarray],
    per_factor_matches: List[Dict],
    stable_factors: List[int],
) -> np.ndarray:
    """Compute consensus factor loadings by averaging matched factors across chains.

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

        # Average loadings from all chains where this factor matched
        matched_loadings = []
        for chain_idx, matched_k in enumerate(matched_indices):
            if matched_k is not None:
                matched_loadings.append(W_chain_avg[chain_idx][:, matched_k])

        if matched_loadings:
            consensus_W[:, i] = np.mean(matched_loadings, axis=0)

    return consensus_W


def _compute_consensus_scores(
    Z_chain_avg: List[np.ndarray],
    per_factor_matches: List[Dict],
    stable_factors: List[int],
) -> np.ndarray:
    """Compute consensus factor scores by averaging across chains.

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
    np.ndarray
        Consensus scores, shape (N, len(stable_factors))
    """
    N, K = Z_chain_avg[0].shape
    n_stable = len(stable_factors)
    consensus_Z = np.zeros((N, n_stable))

    for i, factor_idx in enumerate(stable_factors):
        factor_match_info = per_factor_matches[factor_idx]
        matched_indices = factor_match_info["matched_in_chains"]

        # Average scores from all chains where this factor matched
        matched_scores = []
        for chain_idx, matched_k in enumerate(matched_indices):
            if matched_k is not None:
                matched_scores.append(Z_chain_avg[chain_idx][:, matched_k])

        if matched_scores:
            consensus_Z[:, i] = np.mean(matched_scores, axis=0)

    return consensus_Z


def count_effective_factors(
    W: np.ndarray,
    sparsity_threshold: float = 0.01,
    min_nonzero_pct: float = 0.05,
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
    sparsity_threshold : float, default=0.01
        Minimum loading magnitude to be considered non-zero
    min_nonzero_pct : float, default=0.05
        Minimum fraction of features with non-zero loadings

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
    >>> effective = count_effective_factors(W, sparsity_threshold=0.01)
    >>> print(f"{effective['n_effective']}/{W.shape[1]} factors are effective")
    """
    # Handle multi-view case
    if isinstance(W, list):
        W = np.vstack(W)

    D, K = W.shape
    logger.info(f"Counting effective factors in W matrix: D={D}, K={K}")

    effective_factors = []
    per_factor_stats = []

    for k in range(K):
        loadings = W[:, k]

        # Calculate statistics
        max_loading = float(np.max(np.abs(loadings)))
        mean_loading = float(np.mean(np.abs(loadings)))
        median_loading = float(np.median(np.abs(loadings)))
        std_loading = float(np.std(np.abs(loadings)))
        nonzero_count = int(np.sum(np.abs(loadings) > sparsity_threshold))
        nonzero_pct = nonzero_count / D

        # Factor is effective if it has strong loadings
        is_effective = (max_loading > sparsity_threshold) and (nonzero_pct > min_nonzero_pct)

        per_factor_stats.append({
            "factor_index": int(k),
            "max_loading": max_loading,
            "mean_loading": mean_loading,
            "median_loading": median_loading,
            "std_loading": std_loading,
            "nonzero_count": nonzero_count,
            "nonzero_pct": float(nonzero_pct),
            "is_effective": bool(is_effective),
        })

        if is_effective:
            effective_factors.append(k)
            logger.debug(
                f"Factor {k}: EFFECTIVE (max={max_loading:.3f}, "
                f"nonzero={nonzero_pct:.1%})"
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
        factor_names = [f"Factor_{i}" for i in stability_results["stable_factor_indices"]]
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

                # Save per-view file
                view_filename = f"consensus_factor_loadings_{view_name}.csv"
                save_csv(view_W_df, output_path / view_filename, index=True)
                logger.info(f"    ✅ Saved {view_name}: {view_W.shape}")

                start_idx = end_idx

    # Save consensus scores if available
    if stability_results.get("consensus_Z") is not None:
        consensus_Z = stability_results["consensus_Z"]
        factor_names = [f"Factor_{i}" for i in stability_results["stable_factor_indices"]]
        consensus_Z_df = pd.DataFrame(
            consensus_Z,
            columns=factor_names,
        )
        # Use patient IDs if available, otherwise use generic subject labels
        if subject_ids and len(subject_ids) == consensus_Z.shape[0]:
            consensus_Z_df.index = subject_ids
        else:
            consensus_Z_df.index = [f"Subject_{j}" for j in range(consensus_Z.shape[0])]
        consensus_Z_df.index.name = "Patient ID"
        save_csv(consensus_Z_df, output_path / "consensus_factor_scores.csv", index=True)
        logger.info(f"  ✅ Saved consensus factor scores (Z): {consensus_Z.shape}")

    # Save similarity matrix
    if "similarity_matrix" in stability_results:
        save_numpy(
            stability_results["similarity_matrix"],
            output_path / "similarity_matrix.npy",
        )

    logger.info("Factor stability results saved successfully")
