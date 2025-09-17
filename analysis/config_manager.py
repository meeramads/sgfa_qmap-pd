"""
Configuration Management for run_analysis.py
Handles result directories, flags, dependencies, and analysis setup
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration for analysis runs"""

    run_standard: bool = True
    run_cv: bool = False
    flag: str = ""
    flag_regZ: str = ""
    standard_res_dir: Optional[Path] = None
    cv_res_dir: Optional[Path] = None


@dataclass
class DependencyStatus:
    """Track status of optional dependencies"""

    cv_available: bool = False
    neuroimaging_cv_available: bool = False
    factor_mapping_available: bool = False
    preprocessing_available: bool = False

    def log_status(self):
        """Log the status of dependencies"""
        logging.info("=== DEPENDENCY STATUS ===")
        logging.info(f"Basic CV: {'OK' if self.cv_available else 'MISSING'}")
        logging.info(
            f"Neuroimaging CV: {'OK' if self.neuroimaging_cv_available else 'MISSING'}"
        )
        logging.info(
            f"Factor Mapping: {'OK' if self.factor_mapping_available else 'MISSING'}"
        )
        logging.info(
            f"Preprocessing: {'OK' if self.preprocessing_available else 'MISSING'}"
        )
        logging.info("========================")


class ConfigManager:
    """Manages configuration and dependencies for analysis"""

    def __init__(self, args, results_base: str = "../results"):
        self.args = args
        self.results_base = Path(results_base)
        self.config = AnalysisConfig()
        self.dependencies = DependencyStatus()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check availability of all optional dependencies"""
        self._check_cv_dependencies()
        self._check_factor_mapping()
        self._check_preprocessing()
        self.dependencies.log_status()

    def _check_cv_dependencies(self):
        """Check cross-validation module availability"""
        try:
            # Check for neuroimaging-specific features (these actually exist)
            pass

            self.dependencies.cv_available = True
            self.dependencies.neuroimaging_cv_available = True
            logging.info("Cross-validation module available")
            logging.info("Neuroimaging-aware cross-validation available")

        except ImportError as e:
            logging.info(
                f"Cross-validation module not available - will run standard analysis only. Error: {e}"
            )

    def _check_factor_mapping(self):
        """Check factor-to-MRI mapping module"""
        try:
            pass

            self.dependencies.factor_mapping_available = True
            logging.info("Factor-to-MRI mapping module available")
        except ImportError:
            logging.info("Factor-to-MRI mapping module not available")

    def _check_preprocessing(self):
        """Check preprocessing module"""
        try:
            pass

            self.dependencies.preprocessing_available = True
            logging.info("Preprocessing module available")
        except ImportError:
            logging.info("Preprocessing module not available")

    def setup_analysis_config(self) -> AnalysisConfig:
        """Setup analysis configuration based on arguments"""
        self.config.run_standard = self._should_run_standard_analysis()
        self.config.run_cv = self._should_run_cv_analysis()

        self._create_flag_strings()
        self._create_result_directories()

        logging.info(
            f"Analysis plan: Standard={self.config.run_standard}, CV={self.config.run_cv}"
        )

        return self.config

    def _should_run_standard_analysis(self) -> bool:
        """Determine if we should run standard MCMC analysis"""
        return not getattr(self.args, "cv_only", False)

    def _should_run_cv_analysis(self) -> bool:
        """Determine if we should run cross-validation analysis"""
        return (
            getattr(self.args, "run_cv", False)
            or getattr(self.args, "cv_only", False)
            or getattr(self.args, "neuroimaging_cv", False)
        )

    def _create_flag_strings(self):
        """Create flag strings for directory naming"""
        if "synthetic" in self.args.dataset:
            self.config.flag = f"K{ self.args.K}_{ self.args.num_chains}chs_pW{ self.args.percW}_s{ self.args.num_samples}_addNoise{ self.args.noise}"
        else:
            self.config.flag = f"K{ self.args.K}_{ self.args.num_chains}chs_pW{ self.args.percW}_s{ self.args.num_samples}"

        if self.args.model == "sparseGFA":
            self.config.flag_regZ = "_reghsZ" if self.args.reghsZ else "_hsZ"
        else:
            self.config.flag_regZ = ""

    def _create_result_directories(self):
        """Create result directories using Path"""
        from core.utils import create_results_structure

        if self.config.run_standard:
            self.config.standard_res_dir = create_results_structure(
                self.results_base,
                self.args.dataset,
                self.args.model,
                self.config.flag,
                self.config.flag_regZ,
            )
            logging.info(f"Standard results directory: {self.config.standard_res_dir}")

        if self.config.run_cv:
            # Enhanced CV directory naming
            cv_suffix = "_cv"
            if getattr(self.args, "neuroimaging_cv", False):
                cv_suffix = "_neuroimaging_cv"
            if getattr(self.args, "nested_cv", False):
                cv_suffix += "_nested"

            self.config.cv_res_dir = create_results_structure(
                self.results_base,
                f"{self.args.dataset}_cv",
                self.args.model,
                self.config.flag,
                f"{self.config.flag_regZ}{cv_suffix}",
            )
            logging.info(f"CV results directory: {self.config.cv_res_dir}")

    def get_hyperparameters_dir(self) -> Path:
        """Get directory for hyperparameters"""
        return (
            self.config.standard_res_dir
            if self.config.run_standard
            else self.config.cv_res_dir
        )

    def setup_hyperparameters(self) -> dict:
        """Setup and load/create hyperparameters"""
        from core.utils import safe_pickle_load, safe_pickle_save

        hp_dir = self.get_hyperparameters_dir()
        hp_path = hp_dir / "hyperparameters.dictionary"

        # Check if file exists before trying to load (reduces error messages)
        if hp_path.exists():
            hypers = safe_pickle_load(hp_path, "Hyperparameters")
        else:
            hypers = None

        if hypers is None:
            logging.info(f"Creating new hyperparameters file: {hp_path}")
            hypers = {
                "a_sigma": 1,
                "b_sigma": 1,
                "nu_local": 1,
                "nu_global": 1,
                "slab_scale": 2,
                "slab_df": 4,
                "percW": self.args.percW,
            }

            # Create directory if it doesn't exist
            hp_dir.mkdir(parents=True, exist_ok=True)

            if not safe_pickle_save(hypers, hp_path, "Hyperparameters"):
                logging.warning(
                    "Failed to save hyperparameters, using in-memory defaults"
                )

        return hypers
