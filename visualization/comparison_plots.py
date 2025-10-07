"""Comparison and benchmarking visualization utilities.

This module provides reusable plotting functions for experiment comparisons,
benchmarking, and performance analysis. Used across model_comparison,
sgfa_hyperparameter_tuning, and clinical_validation experiments.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class ComparisonVisualizer:
    """Visualizer for experiment comparison and benchmarking plots.

    Provides consistent, reusable plotting functions for:
    - Performance comparisons (execution time, memory usage)
    - Quality comparisons (method-specific metrics)
    - Performance vs quality tradeoffs
    - Scalability analysis
    - Hyperparameter optimization results
    """

    def __init__(self, config=None):
        """Initialize comparison visualizer.

        Parameters
        ----------
        config : dict or ConfigAccessor, optional
            Configuration for plot styling and output
        """
        self.config = config or {}
        self.default_figsize = (12, 8)
        self.default_colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'pink']

    def plot_performance_comparison(
        self,
        methods: List[str],
        performance_metrics: Dict[str, Dict],
        title: str = "Performance Comparison",
        metrics_to_plot: List[str] = None,
    ) -> Figure:
        """Create bar plots comparing performance metrics across methods.

        Parameters
        ----------
        methods : List[str]
            List of method names
        performance_metrics : Dict[str, Dict]
            Dictionary mapping method names to their performance metrics
            Expected keys: 'execution_time', 'peak_memory_gb', 'convergence'
        title : str
            Overall plot title
        metrics_to_plot : List[str], optional
            Specific metrics to plot. Defaults to ['execution_time', 'peak_memory_gb']

        Returns
        -------
        Figure
            Matplotlib figure with performance comparison
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['execution_time', 'peak_memory_gb']

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16)

        for idx, metric in enumerate(metrics_to_plot):
            values = []
            valid_methods = []

            for method in methods:
                if method in performance_metrics:
                    val = performance_metrics[method].get(metric, 0)
                    if val is not None and not np.isnan(val) and val != float('inf'):
                        values.append(val)
                        valid_methods.append(method)

            if valid_methods:
                axes[idx].bar(valid_methods, values, color=self.default_colors[0], alpha=0.8)
                axes[idx].set_title(self._format_metric_name(metric))
                axes[idx].set_ylabel(self._get_metric_ylabel(metric))
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_quality_comparison(
        self,
        methods: List[str],
        quality_scores: Dict[str, float],
        title: str = "Quality Comparison",
        ylabel: str = "Quality Score",
        higher_is_better: bool = True,
    ) -> Figure:
        """Create bar plot comparing quality scores across methods.

        Parameters
        ----------
        methods : List[str]
            List of method names
        quality_scores : Dict[str, float]
            Dictionary mapping method names to quality scores
        title : str
            Plot title
        ylabel : str
            Y-axis label
        higher_is_better : bool
            If True, highest score is highlighted

        Returns
        -------
        Figure
            Matplotlib figure with quality comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        valid_methods = []
        valid_scores = []

        for method in methods:
            if method in quality_scores:
                score = quality_scores[method]
                if score is not None and not np.isnan(score) and score != float('inf') and score != float('-inf'):
                    valid_methods.append(method)
                    valid_scores.append(score)

        if not valid_methods:
            ax.text(0.5, 0.5, "No valid quality scores", ha='center', va='center')
            return fig

        # Color bars, highlighting best
        colors = [self.default_colors[1]] * len(valid_methods)
        if higher_is_better:
            best_idx = np.argmax(valid_scores)
        else:
            best_idx = np.argmin(valid_scores)
        colors[best_idx] = 'gold'

        bars = ax.bar(valid_methods, valid_scores, color=colors, alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(valid_methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, valid_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}' if abs(score) < 100 else f'{score:.1f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    def plot_performance_vs_quality(
        self,
        methods: List[str],
        performance_metrics: Dict[str, Dict],
        quality_scores: Dict[str, float],
        title: str = "Performance vs Quality Trade-off",
        performance_metric: str = "execution_time",
    ) -> Figure:
        """Create scatter plot showing performance vs quality tradeoff.

        Parameters
        ----------
        methods : List[str]
            List of method names
        performance_metrics : Dict[str, Dict]
            Performance metrics for each method
        quality_scores : Dict[str, float]
            Quality scores for each method
        title : str
            Plot title
        performance_metric : str
            Which performance metric to use (default: 'execution_time')

        Returns
        -------
        Figure
            Matplotlib figure with scatter plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        perf_values = []
        quality_values = []
        valid_methods = []

        for method in methods:
            if method in performance_metrics and method in quality_scores:
                perf = performance_metrics[method].get(performance_metric)
                qual = quality_scores[method]

                if (perf is not None and qual is not None and
                    not np.isnan(perf) and not np.isnan(qual) and
                    perf != float('inf') and qual != float('inf') and
                    qual != float('-inf')):
                    perf_values.append(perf)
                    quality_values.append(qual)
                    valid_methods.append(method)

        if not valid_methods:
            ax.text(0.5, 0.5, "No valid data for comparison", ha='center', va='center')
            return fig

        scatter = ax.scatter(perf_values, quality_values, s=100, alpha=0.7,
                            c=range(len(valid_methods)), cmap='viridis')

        # Add method labels
        for i, method in enumerate(valid_methods):
            ax.annotate(method, (perf_values[i], quality_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel(self._format_metric_name(performance_metric))
        ax.set_ylabel("Quality Score")
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_scalability_analysis(
        self,
        data_sizes: List[Union[int, str]],
        metrics_by_size: Dict[str, List[float]],
        title: str = "Scalability Analysis",
        xlabel: str = "Data Size",
        ylabel: str = "Execution Time (s)",
        log_x: bool = False,
        log_y: bool = False,
    ) -> Figure:
        """Create line plots showing how metrics scale with data size.

        Parameters
        ----------
        data_sizes : List[Union[int, str]]
            List of data sizes (can be integers or labels)
        metrics_by_size : Dict[str, List[float]]
            Dictionary mapping metric names to lists of values (one per data size)
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        log_x : bool
            Use logarithmic scale for x-axis
        log_y : bool
            Use logarithmic scale for y-axis

        Returns
        -------
        Figure
            Matplotlib figure with scalability plots
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)

        for idx, (metric_name, values) in enumerate(metrics_by_size.items()):
            color = self.default_colors[idx % len(self.default_colors)]
            ax.plot(data_sizes, values, marker='o', label=metric_name,
                   color=color, linewidth=2, markersize=8)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        plt.tight_layout()
        return fig

    def plot_hyperparameter_grid(
        self,
        param_name: str,
        param_values: List,
        scores: List[float],
        title: str = "Hyperparameter Optimization",
        ylabel: str = "Score",
    ) -> Figure:
        """Create plot showing how score varies with hyperparameter.

        Parameters
        ----------
        param_name : str
            Name of the hyperparameter
        param_values : List
            List of hyperparameter values tested
        scores : List[float]
            Scores corresponding to each parameter value
        title : str
            Plot title
        ylabel : str
            Y-axis label

        Returns
        -------
        Figure
            Matplotlib figure with hyperparameter plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(param_values, scores, marker='o', linewidth=2, markersize=8,
               color=self.default_colors[2])

        # Highlight best score
        best_idx = np.argmax(scores)
        ax.scatter([param_values[best_idx]], [scores[best_idx]],
                  s=200, color='gold', marker='*', zorder=5,
                  label=f'Best: {param_name}={param_values[best_idx]}')

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_method_characteristics(
        self,
        methods: List[str],
        characteristics: Dict[str, Dict],
        title: str = "Method Characteristics",
    ) -> Figure:
        """Create multi-panel plot showing various method characteristics.

        Parameters
        ----------
        methods : List[str]
            List of method names
        characteristics : Dict[str, Dict]
            Dictionary mapping method names to their characteristics
        title : str
            Overall plot title

        Returns
        -------
        Figure
            Matplotlib figure with method characteristics
        """
        # Extract common characteristics
        all_keys = set()
        for char_dict in characteristics.values():
            all_keys.update(char_dict.keys())

        # Filter to numeric characteristics
        numeric_keys = []
        for key in all_keys:
            sample_val = next((characteristics[m].get(key) for m in methods if m in characteristics and key in characteristics[m]), None)
            if sample_val is not None and isinstance(sample_val, (int, float, np.number)):
                numeric_keys.append(key)

        if not numeric_keys:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No numeric characteristics to plot", ha='center', va='center')
            return fig

        n_chars = min(len(numeric_keys), 4)  # Limit to 4 subplots
        fig, axes = plt.subplots(1, n_chars, figsize=(6 * n_chars, 5))
        if n_chars == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16)

        for idx, key in enumerate(numeric_keys[:n_chars]):
            values = []
            valid_methods = []

            for method in methods:
                if method in characteristics and key in characteristics[method]:
                    val = characteristics[method][key]
                    if not np.isnan(val) and val != float('inf') and val != float('-inf'):
                        values.append(val)
                        valid_methods.append(method)

            if valid_methods:
                axes[idx].bar(valid_methods, values,
                             color=self.default_colors[idx % len(self.default_colors)],
                             alpha=0.8)
                axes[idx].set_title(self._format_metric_name(key))
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_convergence_comparison(
        self,
        methods: List[str],
        convergence_status: Dict[str, bool],
        title: str = "Convergence Status",
    ) -> Figure:
        """Create bar plot showing convergence success across methods.

        Parameters
        ----------
        methods : List[str]
            List of method names
        convergence_status : Dict[str, bool]
            Dictionary mapping method names to convergence status
        title : str
            Plot title

        Returns
        -------
        Figure
            Matplotlib figure with convergence comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        status_values = [int(convergence_status.get(m, False)) for m in methods]
        colors = ['green' if status else 'red' for status in status_values]

        bars = ax.bar(methods, status_values, color=colors, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Converged (1=Yes, 0=No)")
        ax.set_ylim([0, 1.2])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add labels
        for bar, status in zip(bars, status_values):
            label = "✓" if status else "✗"
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                   label, ha='center', va='bottom', fontsize=16)

        plt.tight_layout()
        return fig

    def plot_information_criteria_comparison(
        self,
        methods: List[str],
        ic_metrics: Dict[str, Dict],
        title: str = "Model Selection: Information Criteria",
        criteria: List[str] = None,
    ) -> Figure:
        """Create bar plots comparing AIC/BIC across methods.

        Parameters
        ----------
        methods : List[str]
            List of method names
        ic_metrics : Dict[str, Dict]
            Dictionary mapping method names to IC metrics (aic, bic)
        title : str
            Plot title
        criteria : List[str], optional
            Which criteria to plot. Defaults to ['aic', 'bic']

        Returns
        -------
        Figure
            Matplotlib figure with IC comparison
        """
        if criteria is None:
            criteria = ['aic', 'bic']

        n_criteria = len(criteria)
        fig, axes = plt.subplots(1, n_criteria, figsize=(6 * n_criteria, 5))
        if n_criteria == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16)

        for idx, criterion in enumerate(criteria):
            values = []
            valid_methods = []

            for method in methods:
                if method in ic_metrics and criterion in ic_metrics[method]:
                    val = ic_metrics[method][criterion]
                    if np.isfinite(val):
                        values.append(val)
                        valid_methods.append(method)

            if valid_methods:
                # Color code: lower is better
                norm_values = np.array(values)
                min_val = np.min(norm_values)
                colors = ['green' if v == min_val else 'skyblue' for v in values]

                axes[idx].bar(valid_methods, values, color=colors, alpha=0.8)
                axes[idx].set_title(f"{criterion.upper()} (lower = better)")
                axes[idx].set_ylabel(criterion.upper())
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)

                # Highlight best model
                best_idx = np.argmin(values)
                axes[idx].axhline(y=values[best_idx], color='green',
                                 linestyle='--', alpha=0.5, label='Best model')
                axes[idx].legend()

        plt.tight_layout()
        return fig

    def plot_stability_comparison(
        self,
        methods: List[str],
        stability_metrics: Dict[str, Dict],
        title: str = "Factor Stability (Reproducibility)",
    ) -> Figure:
        """Create bar plot comparing factor stability across methods.

        Parameters
        ----------
        methods : List[str]
            List of method names
        stability_metrics : Dict[str, Dict]
            Dictionary mapping method names to stability metrics
        title : str
            Plot title

        Returns
        -------
        Figure
            Matplotlib figure with stability comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        values = []
        errors = []
        valid_methods = []

        for method in methods:
            if method in stability_metrics:
                mean_stab = stability_metrics[method].get('mean_stability', np.nan)
                std_stab = stability_metrics[method].get('std_stability', 0)

                if not np.isnan(mean_stab):
                    values.append(mean_stab)
                    errors.append(std_stab)
                    valid_methods.append(method)

        if valid_methods:
            # Color code: higher is better
            norm_values = np.array(values)
            max_val = np.max(norm_values)
            colors = ['green' if v == max_val else 'skyblue' for v in values]

            bars = ax.bar(valid_methods, values, yerr=errors, color=colors,
                         alpha=0.8, capsize=5, error_kw={'elinewidth': 2})

            ax.set_title(title, fontsize=14)
            ax.set_ylabel("Mean Factor Correlation (higher = more stable)")
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0.7, color='orange', linestyle='--',
                      alpha=0.5, label='Good stability threshold')
            ax.legend()

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., val + 0.05,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig

    def _format_metric_name(self, metric_name: str) -> str:
        """Format metric name for display."""
        return metric_name.replace('_', ' ').title()

    def _get_metric_ylabel(self, metric_name: str) -> str:
        """Get appropriate y-axis label for a metric."""
        ylabel_map = {
            'execution_time': 'Time (seconds)',
            'peak_memory_gb': 'Memory (GB)',
            'log_likelihood': 'Log-Likelihood',
            'reconstruction_error': 'Reconstruction Error',
            'convergence': 'Convergence Rate',
        }
        return ylabel_map.get(metric_name, self._format_metric_name(metric_name))
