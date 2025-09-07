# analysis/cross_validation.py
"""Cross-validation orchestration module."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CVRunner:
    """Handles cross-validation analysis."""
    
    def __init__(self, config, results_dir):
        self.config = config
        self.results_dir = results_dir
    
    def run_cv_analysis(self, X_list, hypers, data):
        """Run cross-validation analysis based on available modules"""
        logger.info("=== RUNNING CROSS-VALIDATION ANALYSIS ===")
        
        # Check which CV modules are available
        cv_available = self._check_cv_availability()
        neuroimaging_cv_available = self._check_neuroimaging_cv_availability()
        
        if neuroimaging_cv_available and getattr(self.config, 'neuroimaging_cv', False):
            return self._run_neuroimaging_cv_analysis(X_list, hypers, data)
        elif cv_available:
            return self._run_basic_cv_analysis(X_list, hypers, data)
        else:
            logger.error("Cross-validation requested but no CV module available!")
            return None
    
    def _check_cv_availability(self):
        """Check if basic CV is available"""
        try:
            from .cross_validation_library import SparseBayesianGFACrossValidator, CVConfig
            return True
        except ImportError:
            return False
    
    def _check_neuroimaging_cv_availability(self):
        """Check if neuroimaging CV is available"""
        try:
            from .cross_validation_library import NeuroImagingCrossValidator, NeuroImagingCVConfig, ParkinsonsConfig
            return True
        except ImportError:
            return False
    
    def _run_neuroimaging_cv_analysis(self, X_list, hypers, data):
        """Run neuroimaging-aware cross-validation"""
        from .cross_validation_library import NeuroImagingCrossValidator, NeuroImagingCVConfig, ParkinsonsConfig
        
        logger.info("=== ORCHESTRATING NEUROIMAGING CROSS-VALIDATION ===")
        
        # Setup neuroimaging CV configuration
        config = NeuroImagingCVConfig()
        config.outer_cv_folds = getattr(self.config, 'cv_folds', 5)
        config.random_state = getattr(self.config, 'seed', 42)
        
        # Setup Parkinson's specific configuration
        pd_config = ParkinsonsConfig()
        
        # Initialize neuroimaging cross-validator
        cv = NeuroImagingCrossValidator(config, pd_config)
        
        # Run appropriate CV analysis
        if getattr(self.config, 'nested_cv', False):
            logger.info("Running nested neuroimaging cross-validation")
            results = cv.nested_neuroimaging_cv(X_list, self.config, hypers, data)
        else:
            logger.info("Running standard neuroimaging cross-validation")
            results = cv.neuroimaging_cross_validate(X_list, self.config, hypers, data)
        
        return results, cv
    
    def _run_basic_cv_analysis(self, X_list, hypers, data):
        """Run basic cross-validation analysis (fallback)"""
        from .cross_validation_library import SparseBayesianGFACrossValidator, CVConfig
        
        logger.info("=== ORCHESTRATING BASIC CROSS-VALIDATION ===")
        
        # Setup CV configuration
        config = CVConfig()
        config.outer_cv_folds = getattr(self.config, 'cv_folds', 5)
        config.n_jobs = getattr(self.config, 'cv_n_jobs', 1)
        config.random_state = getattr(self.config, 'seed', 42)
        
        # Initialize cross-validator
        cv = SparseBayesianGFACrossValidator(config)
        
        # Run cross-validation
        results = cv.standard_cross_validate(X_list, self.config, hypers)
        
        return results, cv


def should_run_standard_analysis(config):
    """Determine if we should run standard MCMC analysis"""
    return not getattr(config, 'cv_only', False)


def should_run_cv_analysis(config):
    """Determine if we should run cross-validation analysis"""
    return (getattr(config, 'run_cv', False) or 
            getattr(config, 'cv_only', False) or 
            getattr(config, 'neuroimaging_cv', False))