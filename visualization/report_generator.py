# visualization/report_generator.py
"""Report generation module."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive analysis reports."""

    def __init__(self, config):
        self.config = config

    def generate_html_report(
        self,
        plot_dir: Path,
        data: Dict,
        cv_results: Optional[Dict] = None,
        analysis_results: Optional[Dict] = None,
    ):
        """Generate comprehensive HTML report."""
        logger.info("Generating HTML report")

        html_content = self._create_html_header()
        html_content += self._create_data_section(data)

        if analysis_results:
            html_content += self._create_analysis_section(analysis_results)

        if cv_results:
            html_content += self._create_cv_section(cv_results)

        html_content += self._create_plots_section(plot_dir)
        html_content += self._create_html_footer()

        # Save report
        report_path = plot_dir / "reports" / "analysis_report.html"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Report saved to: {report_path}")

    def _create_html_header(self) -> str:
        """Create HTML header."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sparse Bayesian GFA Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }}
                .metric {{
                    background: white;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric strong {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    background: white;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .plot-container {{
                    background: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .plot-container img {{
                    width: 100%;
                    height: auto;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <h1>Sparse Bayesian Group Factor Analysis Report</h1>
            <p class="timestamp">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

    def _create_data_section(self, data: Dict) -> str:
        """Create data overview section."""
        if not data:
            return ""

        X_list = data.get("X_list", [])
        view_names = data.get("view_names", [])

        content = "<h2>Data Overview</h2>"
        content += "<div class='metric'>"
        content += f"<strong>Dataset:</strong> {self.config.dataset}<br>"
        content += f"<strong>Number of Subjects:</strong> {X_list[0].shape[0] if X_list else 'N/A'}<br>"
        content += f"<strong>Number of Views:</strong> {len(view_names)}<br>"
        content += f"<strong>Views:</strong> {', '.join(view_names)}<br>"

        if X_list:
            content += "<strong>Feature Dimensions:</strong><br>"
            content += "<table>"
            content += "<tr><th>View</th><th>Features</th></tr>"
            for name, X in zip(view_names, X_list):
                content += f"<tr><td>{name}</td><td>{X.shape[1]}</td></tr>"
            content += "</table>"

        content += "</div>"

        # Add preprocessing info if available
        if "preprocessing" in data:
            content += self._create_preprocessing_summary(data["preprocessing"])

        return content

    def _create_preprocessing_summary(self, preprocessing: Dict) -> str:
        """Create preprocessing summary section."""
        content = "<h3>Preprocessing Applied</h3>"
        content += "<div class='metric'>"

        if "feature_reduction" in preprocessing:
            content += "<strong>Feature Reduction:</strong><br>"
            content += "<table>"
            content += "<tr><th>View</th><th>Original</th><th>Processed</th><th>Retention</th></tr>"

            for view, stats in preprocessing["feature_reduction"].items():
                content += f"""
                <tr>
                    <td>{view}</td>
                    <td>{stats['original']}</td>
                    <td>{stats['processed']}</td>
                    <td>{stats['reduction_ratio']:.1%}</td>
                </tr>
                """
            content += "</table>"

        content += "</div>"
        return content

    def _create_analysis_section(self, analysis_results: Dict) -> str:
        """Create analysis results section."""
        content = "<h2>Analysis Results</h2>"

        # Find best run
        best_run_id = None
        best_score = -np.inf

        for run_id, run_data in analysis_results.items():
            if isinstance(run_data, dict) and "exp_logdensity" in run_data:
                if run_data["exp_logdensity"] > best_score:
                    best_score = run_data["exp_logdensity"]
                    best_run_id = run_id

        content += "<div class='metric'>"
        content += f"<strong>Number of Runs:</strong> {len(analysis_results)}<br>"
        content += f"<strong>Best Run:</strong> {best_run_id}<br>"
        content += f"<strong>Best Log Density:</strong> {best_score:.2f}<br>"
        content += f"<strong>Number of Factors:</strong> {self.config.K}<br>"
        content += f"<strong>Sparsity:</strong> {self.config.percW}%<br>"
        content += "</div>"

        return content

    def _create_cv_section(self, cv_results: Dict) -> str:
        """Create cross-validation section."""
        content = "<h2>Cross-Validation Results</h2>"
        content += "<div class='metric'>"

        content += f"<strong>CV Type:</strong> {self.config.cv_type}<br>"
        content += f"<strong>Number of Folds:</strong> {self.config.cv_folds}<br>"
        content += f"<strong>Mean Score:</strong> {
            cv_results.get(
                'mean_cv_score',
                'N/A'):.4f}<br>"
        content += f"<strong>Std Score:</strong> {
            cv_results.get(
                'std_cv_score',
                'N/A'):.4f}<br>"

        if "fold_results" in cv_results:
            converged = sum(
                1 for r in cv_results["fold_results"] if r.get("converged", False)
            )
            content += f"<strong>Converged Folds:</strong> {converged}/{
                len(
                    cv_results['fold_results'])}<br>"

        content += (
            f"<strong>Total Time:</strong> {cv_results.get('total_time', 0):.1f}s<br>"
        )
        content += "</div>"

        return content

    def _create_plots_section(self, plot_dir: Path) -> str:
        """Create plots section with embedded images."""
        content = "<h2>Visualizations</h2>"
        content += "<div class='plot-grid'>"

        # Find all PNG files
        plot_files = list(plot_dir.rglob("*.png"))

        for plot_file in sorted(plot_files)[:20]:  # Limit to 20 plots
            rel_path = plot_file.relative_to(plot_dir)
            content += f"""
            <div class='plot-container'>
                <h4>{rel_path.stem.replace('_', ' ').title()}</h4>
                <img src='{rel_path}' alt='{rel_path.stem}'>
            </div>
            """

        content += "</div>"
        return content

    def _create_html_footer(self) -> str:
        """Create HTML footer."""
        return """
        </body>
        </html>
        """
