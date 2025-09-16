# visualization/manager.py
"""Main visualization manager."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manages all visualization tasks."""

    def __init__(self, config):
        self.config = config
        self.plot_dir = None

        # Initialize specialized visualizers
        from visualization.brain_plots import BrainVisualizer
        from visualization.cv_plots import CrossValidationVisualizer
        from visualization.factor_plots import FactorVisualizer
        from visualization.preprocessing_plots import PreprocessingVisualizer
        from visualization.report_generator import ReportGenerator

        self.factor_viz = FactorVisualizer(config)
        self.preprocessing_viz = PreprocessingVisualizer(config)
        self.cv_viz = CrossValidationVisualizer(config)
        self.brain_viz = BrainVisualizer(config)
        self.report_gen = ReportGenerator(config)

    def create_all_visualizations(
        self,
        data: Dict,
        cv_results: Optional[Dict] = None,
        analysis_results: Optional[Dict] = None,
    ):
        """Create all visualizations based on available data."""

        # Setup plot directory
        self.plot_dir = self._setup_plot_directory()

        logger.info("=== Creating Visualizations ===")

        # Factor analysis plots
        if analysis_results:
            self.factor_viz.create_plots(analysis_results, data, self.plot_dir)

        # Preprocessing plots
        if data and "preprocessing" in data:
            self.preprocessing_viz.create_plots(data["preprocessing"], self.plot_dir)

        # Cross-validation plots
        if cv_results:
            self.cv_viz.create_plots(cv_results, self.plot_dir)

        # Brain visualizations (if applicable)
        if self.config.create_brain_viz and analysis_results:
            self.brain_viz.create_plots(analysis_results, data, self.plot_dir)

        # Generate comprehensive report
        self.report_gen.generate_html_report(
            self.plot_dir,
            data=data,
            cv_results=cv_results,
            analysis_results=analysis_results,
        )

        logger.info(f"Visualizations saved to {self.plot_dir}")

    def _setup_plot_directory(self) -> Path:
        """Setup directory for plots."""
        if self.config.should_run_standard():
            base_dir = self.config.standard_results_dir
        else:
            base_dir = self.config.cv_results_dir

        plot_dir = base_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["factors", "preprocessing", "cv", "brain", "reports"]:
            (plot_dir / subdir).mkdir(exist_ok=True)

        return plot_dir
