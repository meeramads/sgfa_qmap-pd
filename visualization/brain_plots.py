"""
Brain visualization module for neuroimaging data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from core.io_utils import save_plot

logger = logging.getLogger(__name__)


class BrainVisualizer:
    """Creates brain-specific visualizations for neuroimaging data."""

    def __init__(self, config):
        self.config = config
        self.setup_style()

        # Check for factor mapping availability
        self.factor_mapping_available = self._check_factor_mapping()

    def setup_style(self):
        """Setup consistent plotting style."""
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update(
            {
                "figure.figsize": (14, 10),
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    def _check_factor_mapping(self) -> bool:
        """Check if factor mapping module is available."""
        try:
            pass

            return True
        except ImportError:
            logger.warning("Factor mapping module not available")
            return False

    def create_plots(self, analysis_results: Dict, data: Dict, plot_dir: Path):
        """Create brain visualization plots."""
        logger.info("Creating brain visualization plots")

        # Create brain plots subdirectory
        brain_plot_dir = plot_dir / "brain"
        brain_plot_dir.mkdir(exist_ok=True)

        try:
            # Use existing brain visualization summary method
            if hasattr(analysis_results, 'get') and analysis_results.get('best_run'):
                # For analysis results with best_run structure
                best_run = analysis_results['best_run']
                results_summary = self.create_brain_visualization_summary(
                    results_dir=Path("./results")  # Fallback path
                )
                logger.info("Brain visualization summary created")
            else:
                # For other result structures, create basic brain analysis
                self._create_basic_brain_plots(analysis_results, data, brain_plot_dir)
                logger.info("Basic brain analysis plots created")

        except Exception as e:
            logger.warning(f"Failed to create brain visualizations: {e}")
            # Create a placeholder to indicate brain visualization was attempted
            placeholder_file = brain_plot_dir / "brain_visualization_attempted.txt"
            with open(placeholder_file, 'w') as f:
                f.write(f"Brain visualization attempted but failed: {e}\n")
                f.write("This is normal if neuroimaging-specific data is not available.\n")

    def _create_basic_brain_plots(self, analysis_results: Dict, data: Dict, plot_dir: Path):
        """Create basic brain analysis plots when full neuroimaging data is not available."""
        try:
            # Extract model results if available
            if 'W' in analysis_results and 'Z' in analysis_results:
                W = analysis_results['W']
                Z = analysis_results['Z']

                # Create factor loading distribution plot
                if isinstance(W, list) and len(W) > 0:
                    # For list-type W (multiple views), create summary plot for first view
                    feature_info = self._prepare_feature_info_for_plotting()
                    self._plot_factor_loading_distribution(W[0][:, 0], 0, plot_dir, feature_info)

                # Create basic spatial analysis plot
                self._plot_basic_factor_summary(W, Z, plot_dir)

            logger.info("Basic brain plots created successfully")

        except Exception as e:
            logger.warning(f"Failed to create basic brain plots: {e}")
            raise

    def _plot_basic_factor_summary(self, W: list, Z: np.ndarray, plot_dir: Path):
        """Create a basic factor summary plot."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Factor Analysis Summary", fontsize=16)

            # Plot 1: Factor loadings magnitude across views
            view_means = [np.mean(np.abs(w)) for w in W]
            bars1 = axes[0, 0].bar(range(len(view_means)), view_means, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title("Average Loading Strength by Brain Region", fontweight='bold')
            axes[0, 0].set_xlabel("Brain Region Index\n(Each bar = one brain region)")
            axes[0, 0].set_ylabel("Average |Loading| Strength")

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars1, view_means)):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Plot 2: Factor scores distribution
            if Z.shape[1] > 0:
                factor_vars = np.var(Z, axis=0)
                bars2 = axes[0, 1].bar(range(len(factor_vars)), factor_vars, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[0, 1].set_title("Factor Variability Across Subjects", fontweight='bold')
                axes[0, 1].set_xlabel("Factor Number\n(Each bar = one latent factor)")
                axes[0, 1].set_ylabel("Variance Across Subjects")

                # Add value labels
                for i, (bar, value) in enumerate(zip(bars2, factor_vars)):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Plot 3: View complexity (number of non-zero loadings)
            view_complexity = []
            for w in W:
                threshold = 0.01 * np.std(w)
                non_zero_ratio = np.mean(np.abs(w) > threshold)
                view_complexity.append(non_zero_ratio)

            bars3 = axes[1, 0].bar(range(len(view_complexity)), view_complexity, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title("Brain Region Activity Level", fontweight='bold')
            axes[1, 0].set_xlabel("Brain Region Index\n(Each bar = one brain region)")
            axes[1, 0].set_ylabel("Fraction of Active Features")

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars3, view_complexity)):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Add explanatory text
            axes[1, 0].text(0.02, 0.98, 'Higher bars = more features\nare actively involved',
                           transform=axes[1, 0].transAxes, fontsize=9, va='top', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))

            # Plot 4: Factor correlation heatmap
            if Z.shape[1] > 1:
                factor_corr = np.corrcoef(Z.T)
                im = axes[1, 1].imshow(factor_corr, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_title("Factor Correlation Matrix")
                axes[1, 1].set_xlabel("Factor Index")
                axes[1, 1].set_ylabel("Factor Index")
                plt.colorbar(im, ax=axes[1, 1])

            plt.tight_layout()
            save_plot(fig, plot_dir / "factor_summary", formats=["png", "pdf"])
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Failed to create factor summary plot: {e}")
            raise

    def create_brain_visualization_summary(
        self, results_dir: str, include_reconstructions: bool = True
    ) -> Dict:
        """
        Create comprehensive brain visualization summary.

        Parameters:
        -----------
        results_dir : str
            Directory containing analysis results
        include_reconstructions : bool
            Whether to include subject reconstructions

        Returns:
        --------
        dict
            Summary of created visualizations
        """
        results_dir = Path(results_dir)
        summary = {
            "brain_plots_created": [],
            "factor_maps": {},
            "reconstruction_summary": {},
            "spatial_analysis": {},
        }

        logger.info(f"Creating brain visualization summary for {results_dir}")

        # Create brain plots directory
        brain_plots_dir = results_dir / "brain_visualizations"
        brain_plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Factor loading brain maps
            if self.factor_mapping_available:
                factor_maps = self._create_factor_brain_maps(
                    results_dir, brain_plots_dir
                )
                summary["factor_maps"] = factor_maps

            # 2. Spatial coherence analysis
            spatial_analysis = self._analyze_spatial_coherence(
                results_dir, brain_plots_dir
            )
            summary["spatial_analysis"] = spatial_analysis

            # 3. Region-wise factor analysis
            region_analysis = self._create_region_factor_analysis(
                results_dir, brain_plots_dir
            )
            summary["region_analysis"] = region_analysis

            # 4. Subject reconstructions if requested
            if include_reconstructions:
                reconstruction_summary = self._create_reconstruction_summary(
                    results_dir, brain_plots_dir
                )
                summary["reconstruction_summary"] = reconstruction_summary

            # 5. Create comprehensive brain report
            self._create_brain_analysis_report(summary, brain_plots_dir)

        except Exception as e:
            logger.error(f"Error creating brain visualizations: {e}")
            summary["error"] = str(e)

        return summary

    def _create_factor_brain_maps(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Create brain maps for factor loadings."""
        if not self.factor_mapping_available:
            logger.warning("Factor mapping not available - skipping brain maps")
            return {}

        # Load robust parameters
        try:
            from core.utils import get_model_files, safe_pickle_load

            # Try to find factor loadings in modern format (CSV/numpy) first
            W = None

            # Look for modern format files (CSV or numpy)
            potential_files = [
                results_dir / "sparseGFA_factor_loadings.csv",
                results_dir / "sparseGFA_factor_loadings.npy",
                results_dir / "factor_loadings.csv",
                results_dir / "factor_loadings.npy"
            ]

            for file_path in potential_files:
                if file_path.exists():
                    try:
                        if file_path.suffix == '.csv':
                            import pandas as pd
                            W = pd.read_csv(file_path).values
                        else:  # .npy
                            W = np.load(file_path)
                        logger.info(f"Loaded factor loadings from modern format: {file_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

            # Fall back to legacy format if modern files not found
            if W is None:
                try:
                    # Try all available runs instead of just "best" run
                    robust_files = list(results_dir.glob("[*]Robust_params.dictionary"))
                    if robust_files:
                        # Try to load from any available robust params file
                        for robust_file in robust_files[:3]:  # Try first 3 files
                            try:
                                rob_params = safe_pickle_load(robust_file, description="Robust parameters")
                                if rob_params and "W" in rob_params:
                                    W = rob_params["W"]
                                    logger.info(f"Loaded factor loadings from {robust_file.name}")
                                    break
                            except Exception as e:
                                logger.debug(f"Could not load {robust_file.name}: {e}")
                                continue
                    else:
                        # Fallback to original method
                        best_run = self._find_best_run(results_dir)
                        files = get_model_files(results_dir, best_run)
                        rob_params = safe_pickle_load(files["robust_params"], description="Robust parameters")
                        if rob_params and "W" in rob_params:
                            W = rob_params["W"]
                            logger.info("Loaded factor loadings from legacy dictionary format")
                except Exception as e:
                    logger.warning(f"Failed to load legacy format: {e}")

            if W is None:
                logger.warning("No factor loadings found for brain mapping")
                return {}
            n_factors = W.shape[1]

            # Create brain maps for top factors
            factor_maps = {}
            n_maps_to_create = min(10, n_factors)  # Create maps for top 10 factors

            for factor_idx in range(n_maps_to_create):
                try:
                    # Create factor-specific brain map
                    factor_loadings = W[:, factor_idx]

                    # Create statistical summary plot
                    # Prepare feature information for interpretable plots
                    feature_info = self._prepare_feature_info_for_plotting()
                    self._plot_factor_loading_distribution(
                        factor_loadings, factor_idx, plot_dir, feature_info
                    )

                    factor_maps[f"factor_{factor_idx + 1}"] = {
                        "mean_loading": float(np.mean(np.abs(factor_loadings))),
                        "max_loading": float(np.max(np.abs(factor_loadings))),
                        "sparsity": float(
                            np.sum(np.abs(factor_loadings) > 0.01)
                            / len(factor_loadings)
                        ),
                    }

                except Exception as e:
                    logger.warning(
                        f"Failed to create brain map for factor {factor_idx + 1}: {e}"
                    )
                    continue

            logger.info(f"Created brain maps for {len(factor_maps)} factors")

            logger.info(f"Created brain maps for {len(factor_maps)} factors")

            return factor_maps

        except Exception as e:
            logger.error(f"Error in factor brain mapping: {e}")
            return {}

    def _plot_factor_loading_distribution(
        self, loadings: np.ndarray, factor_idx: int, plot_dir: Path,
        feature_info: Dict = None
    ):
        """Plot distribution of factor loadings with interpretable feature information.

        Args:
            loadings: Factor loading values
            factor_idx: Index of the factor
            plot_dir: Directory to save plots
            feature_info: Dictionary containing view_names, Dm, and feature_names for interpretation
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Factor {factor_idx + 1} Loading Analysis", fontsize=16, fontweight="bold"
        )

        # Distribution of loadings
        axes[0, 0].hist(
            loadings, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        axes[0, 0].set_title(f"Factor {factor_idx + 1} Loading Distribution")
        axes[0, 0].set_xlabel("Factor Loading Value")
        axes[0, 0].set_ylabel("Feature Count")

        # Absolute loadings
        axes[0, 1].hist(
            np.abs(loadings), bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        axes[0, 1].set_title(f"Factor {factor_idx + 1} Absolute Loading Distribution")
        axes[0, 1].set_xlabel("Absolute Factor Loading")
        axes[0, 1].set_ylabel("Feature Count")

        # Top loadings with interpretable feature information
        top_indices = np.argsort(np.abs(loadings))[-20:]
        top_loadings = loadings[top_indices]

        # Generate meaningful feature labels
        feature_labels = self._generate_feature_labels(top_indices, feature_info)

        # Create horizontal bar plot with clear color coding and explanations
        colors = ["red" if x < 0 else "blue" for x in top_loadings]
        bars = axes[1, 0].barh(
            range(len(top_loadings)),
            top_loadings,
            color=colors,
            alpha=0.7,
        )

        axes[1, 0].set_title(f"Factor {factor_idx + 1}: Top 20 Most Important Brain Features", fontweight='bold')
        axes[1, 0].set_xlabel("Factor Loading Strength\n(Blue = Positive influence, Red = Negative influence)")
        axes[1, 0].set_ylabel("Brain Region Features (ordered by importance)")

        # Set meaningful y-tick labels
        axes[1, 0].set_yticks(range(len(feature_labels)))
        axes[1, 0].set_yticklabels(feature_labels, fontsize=8)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_loadings)):
            axes[1, 0].text(value + 0.01 * np.sign(value), bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', va='center', ha='left' if value > 0 else 'right',
                           fontsize=7, fontweight='bold')

        # Add legend explaining colors
        axes[1, 0].axvline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].text(0.02, 0.98, 'Each bar = one brain feature\nBlue = positively associated\nRed = negatively associated',
                       transform=axes[1, 0].transAxes, fontsize=9, va='top', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Add grid for better readability
        axes[1, 0].grid(True, alpha=0.3, axis='x')

        # Loading contribution by brain region/view (more interpretable than cumulative)
        if feature_info and feature_info.get("view_names") and feature_info.get("Dm"):
            view_names = feature_info["view_names"]
            Dm = feature_info["Dm"]

            # Calculate mean absolute loading for each view
            view_contributions = []
            view_labels = []
            cumulative_start = 0

            for view_name, view_dim in zip(view_names, Dm):
                if cumulative_start + view_dim <= len(loadings):
                    view_loadings = loadings[cumulative_start:cumulative_start + view_dim]
                    mean_abs_loading = np.mean(np.abs(view_loadings))
                    view_contributions.append(mean_abs_loading)
                    view_labels.append(view_name)
                    cumulative_start += view_dim

            if view_contributions:
                colors = ['lightcoral' if 'clinical' in label.lower() else 'skyblue' for label in view_labels]
                bars = axes[1, 1].bar(view_labels, view_contributions, color=colors, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title(f"Factor {factor_idx + 1}: Average Importance by Brain Region", fontweight='bold')
                axes[1, 1].set_xlabel("Brain Region\n(Each bar = one brain region)")
                axes[1, 1].set_ylabel("Average Loading Strength")
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, value in zip(bars, view_contributions):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

                # Add explanatory text
                axes[1, 1].text(0.02, 0.98, 'Higher bars = more important\nfor this factor',
                               transform=axes[1, 1].transAxes, fontsize=9, va='top', ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            else:
                # Fallback to cumulative plot
                self._plot_cumulative_loadings(axes[1, 1], loadings)
        else:
            # Fallback to cumulative plot
            self._plot_cumulative_loadings(axes[1, 1], loadings)

        plt.tight_layout()
        plt.savefig(
            plot_dir / f"factor_{factor_idx + 1}_loading_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_cumulative_loadings(self, ax, loadings: np.ndarray):
        """Plot cumulative loading contribution as fallback."""
        sorted_abs_loadings = np.sort(np.abs(loadings))[::-1]
        cumulative_loadings = np.cumsum(sorted_abs_loadings**2)
        cumulative_loadings = cumulative_loadings / cumulative_loadings[-1]

        ax.plot(
            range(1, len(cumulative_loadings) + 1),
            cumulative_loadings,
            "b-",
            linewidth=2,
        )
        ax.axhline(
            y=0.8, color="red", linestyle="--", alpha=0.7, label="80% threshold"
        )
        ax.set_title("Cumulative Loading Contribution", fontsize=10)
        ax.set_xlabel("Feature Index (sorted by loading magnitude)")
        ax.set_ylabel("Cumulative Loading Contribution")
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=1)
        ax.grid(True, alpha=0.3)

    def _generate_feature_labels(self, feature_indices: np.ndarray, feature_info: Dict = None) -> List[str]:
        """Generate meaningful labels for features based on their indices and available metadata.

        Args:
            feature_indices: Array of feature indices to label
            feature_info: Dictionary containing view_names, Dm, and feature_names

        Returns:
            List of meaningful feature labels
        """
        if not feature_info:
            return [f"Feature_{i}" for i in feature_indices]

        view_names = feature_info.get("view_names", [])
        Dm = feature_info.get("Dm", [])
        feature_names = feature_info.get("feature_names", {})

        if not view_names or not Dm:
            return [f"Feature_{i}" for i in feature_indices]

        labels = []
        for idx in feature_indices:
            label = self._get_feature_label_for_index(idx, view_names, Dm, feature_names)
            labels.append(label)

        return labels

    def _get_feature_label_for_index(self, global_idx: int, view_names: List[str],
                                   Dm: List[int], feature_names: Dict) -> str:
        """Get a meaningful label for a specific feature index.

        Args:
            global_idx: Global feature index across all views
            view_names: List of view names
            Dm: List of dimensions for each view
            feature_names: Dictionary mapping view names to feature name lists

        Returns:
            Meaningful feature label
        """
        # Find which view this feature belongs to
        cumulative_start = 0
        for view_idx, (view_name, view_dim) in enumerate(zip(view_names, Dm)):
            if cumulative_start <= global_idx < cumulative_start + view_dim:
                local_idx = global_idx - cumulative_start

                # Use actual feature names if available
                if view_name in feature_names and local_idx < len(feature_names[view_name]):
                    actual_name = feature_names[view_name][local_idx]
                    return f"{actual_name} ({view_name})"

                # Generate meaningful names based on view type
                if 'clinical' in view_name.lower():
                    # Load clinical variable names
                    try:
                        import pandas as pd
                        from pathlib import Path
                        clinical_paths = [
                            Path("qMAP-PD_data/data_clinical/pd_motor_gfa_data.tsv"),
                            Path("../qMAP-PD_data/data_clinical/pd_motor_gfa_data.tsv"),
                            Path("./data_clinical/pd_motor_gfa_data.tsv")
                        ]
                        for path in clinical_paths:
                            if path.exists():
                                clinical_df = pd.read_csv(path, sep="\t")
                                clinical_names = [col for col in clinical_df.columns if col != 'sid']
                                if local_idx < len(clinical_names):
                                    return f"{clinical_names[local_idx]} (clinical)"
                                break
                    except Exception:
                        pass
                    return f"Clinical_var_{local_idx}"

                elif any(region in view_name.lower() for region in ['sn', 'substantia', 'putamen', 'lentiform']):
                    # Brain region voxels
                    region_map = {
                        'sn': 'SubstantiaNigra', 'substantia': 'SubstantiaNigra',
                        'putamen': 'Putamen', 'lentiform': 'Lentiform'
                    }
                    region_name = next((region_map[k] for k in region_map.keys()
                                      if k in view_name.lower()), view_name)
                    return f"{region_name}_voxel_{local_idx}"

                else:
                    # Generic view feature
                    return f"{view_name}_feature_{local_idx}"

            cumulative_start += view_dim

        # Fallback if index is out of bounds
        return f"Feature_{global_idx}"


    def _prepare_feature_info_for_plotting(self) -> Dict:
        """Prepare feature information for interpretable plotting.

        Returns:
            Dictionary containing view names, dimensions, and feature names
        """
        try:
            # Load position files to get brain region dimensions
            from pathlib import Path
            import pandas as pd

            data_dir = Path("qMAP-PD_data")

            # Check for filtered position lookups first (after preprocessing), fall back to original
            filtered_dir = data_dir / "position_lookup_filtered"
            original_dir = data_dir / "position_lookup"

            position_files = {}
            for region_name in ["sn", "putamen", "lentiform"]:
                # Try filtered version first
                filtered_file = filtered_dir / f"position_{region_name}_voxels_filtered.tsv"
                original_file = original_dir / f"position_{region_name}_voxels.tsv"

                if filtered_file.exists():
                    position_files[region_name] = filtered_file
                    logging.info(f"Using filtered position lookup for {region_name}")
                elif original_file.exists():
                    position_files[region_name] = original_file
                    logging.info(f"Using original position lookup for {region_name}")

            view_names = []
            Dm = []
            feature_names = {}

            # Load brain region dimensions
            for region_name, pos_file in position_files.items():
                positions = pd.read_csv(pos_file, sep="\t", header=None).values.flatten()
                view_names.append(region_name)
                Dm.append(len(positions))
                # Generate voxel names (could be enhanced with actual coordinates)
                feature_names[region_name] = [f"{region_name}_voxel_{i}" for i in range(len(positions))]

            # Load clinical variable names
            clinical_file = data_dir / "data_clinical" / "pd_motor_gfa_data.tsv"
            if clinical_file.exists():
                clinical_df = pd.read_csv(clinical_file, sep="\t")
                clinical_names = [col for col in clinical_df.columns if col != 'sid']
                view_names.append("clinical")
                Dm.append(len(clinical_names))
                feature_names["clinical"] = clinical_names

            return {
                "view_names": view_names,
                "Dm": Dm,
                "feature_names": feature_names
            }

        except Exception as e:
            logger.warning(f"Could not prepare feature info for plotting: {e}")
            return {}

    def _analyze_spatial_coherence(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Analyze spatial coherence of factors."""
        spatial_analysis = {
            "coherence_metrics": {},
            "region_specificity": {},
            "spatial_patterns": {},
        }

        try:
            # Load data and results
            from core.utils import safe_pickle_load

            # Check if preprocessing results contain spatial information
            # Try modern JSON format first, then fall back to legacy dictionary
            prep_json_path = results_dir / "preprocessing_results.json"
            prep_dict_path = results_dir / "preprocessing_results.dictionary"

            preprocessing_results = None
            if prep_json_path.exists():
                try:
                    with open(prep_json_path, 'r') as f:
                        import json
                        preprocessing_results = json.load(f)
                        logger.info("Loaded preprocessing results from JSON format")
                except Exception as e:
                    logger.warning(f"Failed to load JSON preprocessing results: {e}")

            if preprocessing_results is None and prep_dict_path.exists():
                preprocessing_results = safe_pickle_load(prep_dict_path, description="Preprocessing results")

            if preprocessing_results and "metadata" in preprocessing_results:
                metadata = preprocessing_results["metadata"]

                if metadata.get("spatial_processing_applied", False):
                    # Create spatial coherence plots
                    self._plot_spatial_processing_summary(metadata, plot_dir)

                    spatial_analysis["spatial_processing"] = {
                        "applied": True,
                        "position_lookups_loaded": metadata.get(
                            "position_lookups_loaded", []
                        ),
                        "harmonization_applied": metadata.get(
                            "harmonization_applied", False
                        ),
                    }

            # If no spatial processing, create basic spatial analysis
            if not spatial_analysis.get("spatial_processing", {}).get("applied", False):
                spatial_analysis = self._basic_spatial_analysis(results_dir, plot_dir)

        except Exception as e:
            logger.warning(f"Spatial coherence analysis failed: {e}")
            spatial_analysis["error"] = str(e)

        return spatial_analysis

    def _plot_spatial_processing_summary(self, metadata: Dict, plot_dir: Path):
        """Plot spatial preprocessing summary."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Spatial Processing Summary", fontsize=16, fontweight="bold")

        # Feature reduction by view
        if "feature_reduction" in metadata:
            views = list(metadata["feature_reduction"].keys())
            original_features = [
                metadata["feature_reduction"][v]["original"] for v in views
            ]
            processed_features = [
                metadata["feature_reduction"][v]["processed"] for v in views
            ]

            x = np.arange(len(views))
            width = 0.35

            axes[0, 0].bar(
                x - width / 2, original_features, width, label="Original", alpha=0.7
            )
            axes[0, 0].bar(
                x + width / 2, processed_features, width, label="Processed", alpha=0.7
            )
            axes[0, 0].set_title("Feature Reduction by View")
            axes[0, 0].set_xlabel("View")
            axes[0, 0].set_ylabel("Number of Features")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(views, rotation=45)
            axes[0, 0].legend()

        # Reduction ratios
        if "feature_reduction" in metadata:
            reduction_ratios = [
                metadata["feature_reduction"][v]["reduction_ratio"] for v in views
            ]

            axes[0, 1].bar(views, reduction_ratios, alpha=0.7, color="lightcoral")
            axes[0, 1].set_title("Feature Retention Ratios")
            axes[0, 1].set_xlabel("View")
            axes[0, 1].set_ylabel("Retention Ratio")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # Position data availability
        if metadata.get("position_lookups_loaded"):
            position_views = metadata["position_lookups_loaded"]
            axes[1, 0].bar(
                range(len(position_views)),
                [1] * len(position_views),
                alpha=0.7,
                color="lightgreen",
            )
            axes[1, 0].set_title("Views with Spatial Position Data")
            axes[1, 0].set_ylabel("Available")
            axes[1, 0].set_xticks(range(len(position_views)))
            axes[1, 0].set_xticklabels(position_views, rotation=45)

        # Processing summary text
        axes[1, 1].text(
            0.1,
            0.7,
            f"Spatial Processing: {'Applied' if metadata.get('spatial_processing_applied') else 'Not Applied'}",
            fontsize=12,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.1,
            0.5,
            f"Scanner Harmonization: {'Applied' if metadata.get('harmonization_applied') else 'Not Applied'}",
            fontsize=12,
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(
            0.1,
            0.3,
            f"Position Data Views: {len(metadata.get('position_lookups_loaded', []))}",
            fontsize=12,
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Processing Summary")

        plt.tight_layout()
        save_plot(plot_dir / "spatial_processing_summary.png")

    def _basic_spatial_analysis(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Perform basic spatial analysis without full preprocessing."""
        spatial_analysis = {
            "basic_analysis": True,
            "coherence_metrics": {},
            "spatial_processing": {"applied": False},
        }

        # Create placeholder spatial analysis plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Basic Spatial Analysis\n(Full spatial processing not applied)",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.3,
            "For enhanced spatial analysis, enable spatial processing\nwith --enable_spatial_processing",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Spatial Analysis Summary")

        plt.tight_layout()
        save_plot(plot_dir / "basic_spatial_analysis.png")

        return spatial_analysis

    def _create_region_factor_analysis(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Create region-wise factor analysis."""
        region_analysis = {
            "region_loadings": {},
            "factor_specificity": {},
            "cross_region_correlations": {},
        }

        try:
            # This would require region-specific data organization
            # For now, create a placeholder analysis
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Region-wise Factor Analysis\n(Requires region-specific data organization)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title("Region-wise Analysis")

            plt.tight_layout()
            save_plot(plot_dir / "region_factor_analysis.png")

        except Exception as e:
            logger.warning(f"Region factor analysis failed: {e}")
            region_analysis["error"] = str(e)

        return region_analysis

    def _create_reconstruction_summary(self, results_dir: Path, plot_dir: Path) -> Dict:
        """Create summary of subject reconstructions."""
        reconstruction_summary = {
            "reconstructions_available": False,
            "n_subjects_reconstructed": 0,
            "reconstruction_quality": {},
        }

        # Check for reconstruction directory
        reconstruction_dir = results_dir / "subject_reconstructions"
        if reconstruction_dir.exists():
            reconstruction_files = list(reconstruction_dir.glob("*.nii*"))
            reconstruction_summary["reconstructions_available"] = True
            reconstruction_summary["n_subjects_reconstructed"] = len(
                reconstruction_files
            )

            # Create reconstruction summary plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "Subject Reconstruction Summary", fontsize=16, fontweight="bold"
            )

            # Number of reconstructions
            axes[0, 0].bar(
                ["Reconstructed"],
                [len(reconstruction_files)],
                alpha=0.7,
                color="lightblue",
            )
            axes[0, 0].set_title("Number of Subject Reconstructions")
            axes[0, 0].set_ylabel("Count")

            # File size distribution (as proxy for data completeness)
            if reconstruction_files:
                file_sizes = [
                    f.stat().st_size / 1024 / 1024 for f in reconstruction_files[:50]
                ]  # MB, limit to 50
                axes[0, 1].hist(file_sizes, bins=20, alpha=0.7, color="lightgreen")
                axes[0, 1].set_title("Reconstruction File Sizes (MB)")
                axes[0, 1].set_xlabel("File Size (MB)")
                axes[0, 1].set_ylabel("Count")

            # Reconstruction availability over time (if timestamps available)
            axes[1, 0].text(
                0.5,
                0.5,
                f"Reconstructions Created\n{len(reconstruction_files)} subjects",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].axis("off")

            # Summary statistics
            summary_text = f"""Reconstruction Statistics:
Total files: {len(reconstruction_files)}
Directory: {reconstruction_dir.name}
File types: {set(f.suffix for f in reconstruction_files)}
"""
            axes[1, 1].text(
                0.1,
                0.5,
                summary_text,
                fontsize=11,
                transform=axes[1, 1].transAxes,
                verticalalignment="center",
            )
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Summary Statistics")

            plt.tight_layout()
            save_plot(plot_dir / "subject_reconstruction_summary.png")

        return reconstruction_summary

    def _create_brain_analysis_report(self, summary: Dict, plot_dir: Path):
        """Create comprehensive brain analysis report."""
        report_path = plot_dir / "brain_analysis_report.json"

        # Create JSON report
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Create visual summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Brain Analysis Summary Report", fontsize=16, fontweight="bold")

        # Factor maps summary
        if summary.get("factor_maps"):
            n_maps = len(summary["factor_maps"])
            axes[0, 0].bar(["Factor Maps"], [n_maps], alpha=0.7, color="lightblue")
            axes[0, 0].set_title("Brain Visualizations Created")
            axes[0, 0].set_ylabel("Count")

        # Spatial analysis status
        spatial_status = (
            "Applied"
            if summary.get("spatial_analysis", {})
            .get("spatial_processing", {})
            .get("applied", False)
            else "Basic"
        )
        status_colors = {"Applied": "green", "Basic": "orange"}
        axes[0, 1].bar(
            ["Spatial Analysis"],
            [1],
            alpha=0.7,
            color=status_colors.get(spatial_status, "gray"),
        )
        axes[0, 1].set_title("Spatial Processing Status")
        axes[0, 1].set_ylabel("Status")
        axes[0, 1].text(
            0, 0.5, spatial_status, ha="center", va="center", fontweight="bold"
        )

        # Reconstruction summary
        n_reconstructions = summary.get("reconstruction_summary", {}).get(
            "n_subjects_reconstructed", 0
        )
        axes[1, 0].bar(
            ["Reconstructions"], [n_reconstructions], alpha=0.7, color="lightgreen"
        )
        axes[1, 0].set_title("Subject Reconstructions")
        axes[1, 0].set_ylabel("Count")

        # Overall summary text
        summary_text = f"""Brain Analysis Summary:
• Factor Maps: {len(summary.get('factor_maps', { }))} created
• Spatial Processing: {spatial_status}
• Subject Reconstructions: {n_reconstructions}
• Analysis Complete: {'Yes' if not summary.get('error') else 'With Errors'}
"""
        axes[1, 1].text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            transform=axes[1, 1].transAxes,
            verticalalignment="center",
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Summary")

        plt.tight_layout()
        plt.savefig(
            plot_dir / "brain_analysis_overview.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(f"Brain analysis report saved to {report_path}")

    def _find_best_run(self, results_dir: Path) -> int:
        """Find the best run based on results.txt or default to 1."""
        results_file = results_dir / "results.txt"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    for line in f:
                        if line.startswith("Best run:"):
                            return int(line.split(":")[1].strip())
            except BaseException:
                pass
        return 1  # Default to run 1
