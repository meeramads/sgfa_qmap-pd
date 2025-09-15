"""Method comparison experiments for SGFA qMAP-PD analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import logging
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_decomposition import CCA
import warnings

from experiments.framework import ExperimentFramework, ExperimentConfig, ExperimentResult
from performance import PerformanceProfiler

class MethodComparisonExperiments(ExperimentFramework):
    """Comprehensive method comparison experiments for SGFA analysis."""
    
    def __init__(self, config: ExperimentConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.profiler = PerformanceProfiler()
        
        # Method configurations
        self.sgfa_variants = {
            'standard': {'use_sparse': True, 'use_group': True},
            'sparse_only': {'use_sparse': True, 'use_group': False},
            'group_only': {'use_sparse': False, 'use_group': True},
            'basic_fa': {'use_sparse': False, 'use_group': False}
        }
        
        # Traditional methods for comparison
        self.traditional_methods = ['pca', 'ica', 'fa', 'cca', 'kmeans']
        
    def run_sgfa_variant_comparison(self, X_list: List[np.ndarray], 
                                  hypers: Dict, args: Dict,
                                  **kwargs) -> ExperimentResult:
        """Compare different SGFA variants."""
        self.logger.info("Running SGFA variant comparison")
        
        results = {}
        performance_metrics = {}
        
        try:
            for variant_name, variant_config in self.sgfa_variants.items():
                self.logger.info(f"Testing SGFA variant: {variant_name}")
                
                # Profile variant performance
                with self.profiler.profile(f'sgfa_{variant_name}') as p:
                    # Update hyperparameters with variant config
                    variant_hypers = hypers.copy()
                    variant_hypers.update(variant_config)
                    
                    # Run analysis (would call your SGFA implementation)
                    variant_result = self._run_sgfa_variant(
                        X_list, variant_hypers, args, **kwargs
                    )
                    
                    results[variant_name] = variant_result
                
                # Store performance metrics
                metrics = self.profiler.get_current_metrics()
                performance_metrics[variant_name] = {
                    'execution_time': metrics.execution_time,
                    'peak_memory_gb': metrics.peak_memory_gb,
                    'convergence_iterations': variant_result.get('n_iterations', 0)
                }
                
            # Analyze results
            analysis = self._analyze_sgfa_variants(results, performance_metrics)
            
            # Generate plots
            plots = self._plot_sgfa_comparison(results, performance_metrics)
            
            return ExperimentResult(
                experiment_name="sgfa_variant_comparison",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"SGFA variant comparison failed: {str(e)}")
            return self._create_failure_result("sgfa_variant_comparison", str(e))
    
    def run_traditional_method_comparison(self, X_list: List[np.ndarray],
                                        sgfa_results: Dict = None,
                                        **kwargs) -> ExperimentResult:
        """Compare SGFA with traditional dimensionality reduction methods."""
        self.logger.info("Running traditional method comparison")
        
        results = {}
        performance_metrics = {}
        
        try:
            # Concatenate multi-view data for traditional methods
            X_concat = np.hstack(X_list) if len(X_list) > 1 else X_list[0]
            n_subjects = X_concat.shape[0]
            
            # Determine number of components
            n_components = kwargs.get('n_components', min(10, X_concat.shape[1] // 2))
            
            for method_name in self.traditional_methods:
                self.logger.info(f"Testing traditional method: {method_name}")
                
                with self.profiler.profile(f'traditional_{method_name}') as p:
                    method_result = self._run_traditional_method(
                        X_concat, method_name, n_components, **kwargs
                    )
                    results[method_name] = method_result
                
                # Store performance metrics
                metrics = self.profiler.get_current_metrics()
                performance_metrics[method_name] = {
                    'execution_time': metrics.execution_time,
                    'peak_memory_gb': metrics.peak_memory_gb
                }
            
            # Include SGFA results if provided
            if sgfa_results:
                results['sgfa'] = sgfa_results
                performance_metrics['sgfa'] = {
                    'execution_time': sgfa_results.get('execution_time', 0),
                    'peak_memory_gb': sgfa_results.get('peak_memory_gb', 0)
                }
            
            # Analyze method comparison
            analysis = self._analyze_traditional_comparison(results, X_list)
            
            # Generate plots
            plots = self._plot_traditional_comparison(results, performance_metrics)
            
            return ExperimentResult(
                experiment_name="traditional_method_comparison",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                performance_metrics=performance_metrics,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Traditional method comparison failed: {str(e)}")
            return self._create_failure_result("traditional_method_comparison", str(e))
    
    def run_multiview_capability_assessment(self, X_list: List[np.ndarray],
                                          **kwargs) -> ExperimentResult:
        """Assess multi-view capabilities of different methods."""
        self.logger.info("Running multi-view capability assessment")
        
        results = {}
        
        try:
            n_views = len(X_list)
            
            # Test with different numbers of views
            view_combinations = [
                list(range(i+1)) for i in range(n_views)
            ]
            
            for n_view_test in range(1, n_views + 1):
                view_subset = X_list[:n_view_test]
                
                # SGFA with subset of views
                sgfa_result = self._run_sgfa_multiview(view_subset, **kwargs)
                
                # Traditional methods (concatenated)
                X_concat = np.hstack(view_subset)
                traditional_results = {}
                
                for method in ['pca', 'fa', 'cca']:
                    if method == 'cca' and n_view_test < 2:
                        continue  # CCA needs at least 2 views
                        
                    traditional_results[method] = self._run_traditional_method(
                        X_concat, method, min(10, X_concat.shape[1] // 2)
                    )
                
                results[f'{n_view_test}_views'] = {
                    'sgfa': sgfa_result,
                    'traditional': traditional_results,
                    'view_dimensions': [X.shape[1] for X in view_subset]
                }
            
            # Analyze multi-view capabilities
            analysis = self._analyze_multiview_capabilities(results)
            
            # Generate plots
            plots = self._plot_multiview_comparison(results)
            
            return ExperimentResult(
                experiment_name="multiview_capability_assessment",
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Multi-view capability assessment failed: {str(e)}")
            return self._create_failure_result("multiview_capability_assessment", str(e))
    
    def run_scalability_comparison(self, X_list: List[np.ndarray],
                                 sample_sizes: List[int] = None,
                                 feature_sizes: List[int] = None,
                                 **kwargs) -> ExperimentResult:
        """Compare scalability of different methods."""
        self.logger.info("Running scalability comparison")
        
        if sample_sizes is None:
            sample_sizes = [100, 500, 1000, 2000]
        if feature_sizes is None:
            feature_sizes = [50, 100, 200, 500]
            
        results = {}
        
        try:
            # Sample size scalability
            self.logger.info("Testing sample size scalability")
            sample_results = {}
            
            for n_samples in sample_sizes:
                if n_samples > X_list[0].shape[0]:
                    continue
                    
                # Subsample data
                indices = np.random.choice(X_list[0].shape[0], n_samples, replace=False)
                X_subset = [X[indices] for X in X_list]
                
                sample_results[n_samples] = self._run_scalability_test(X_subset, **kwargs)
            
            results['sample_scalability'] = sample_results
            
            # Feature size scalability
            self.logger.info("Testing feature size scalability")
            feature_results = {}
            
            for n_features in feature_sizes:
                # Select subset of features from each view
                X_feature_subset = []
                for X in X_list:
                    if n_features >= X.shape[1]:
                        X_feature_subset.append(X)
                    else:
                        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)
                        X_feature_subset.append(X[:, feature_indices])
                
                feature_results[n_features] = self._run_scalability_test(X_feature_subset, **kwargs)
            
            results['feature_scalability'] = feature_results
            
            # Analyze scalability
            analysis = self._analyze_scalability(results)
            
            # Generate plots
            plots = self._plot_scalability_comparison(results)
            
            return ExperimentResult(
                experiment_name="scalability_comparison", 
                config=self.config,
                data=results,
                analysis=analysis,
                plots=plots,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Scalability comparison failed: {str(e)}")
            return self._create_failure_result("scalability_comparison", str(e))
    
    def _run_sgfa_variant(self, X_list: List[np.ndarray], 
                         hypers: Dict, args: Dict, **kwargs) -> Dict:
        """Run SGFA with specific variant configuration."""
        # This would call your actual SGFA implementation
        # For now, return mock results
        return {
            'W': [np.random.randn(X.shape[1], 5) for X in X_list],
            'Z': np.random.randn(X_list[0].shape[0], 5),
            'log_likelihood': np.random.randn(),
            'n_iterations': np.random.randint(100, 1000),
            'convergence': True
        }
    
    def _run_traditional_method(self, X: np.ndarray, method_name: str,
                              n_components: int, **kwargs) -> Dict:
        """Run traditional dimensionality reduction method."""
        results = {}
        
        try:
            if method_name == 'pca':
                model = PCA(n_components=n_components)
                Z = model.fit_transform(X)
                results = {
                    'components': model.components_,
                    'explained_variance_ratio': model.explained_variance_ratio_,
                    'Z': Z,
                    'model': model
                }
                
            elif method_name == 'fa':
                model = FactorAnalysis(n_components=n_components)
                Z = model.fit_transform(X)
                results = {
                    'components': model.components_,
                    'Z': Z,
                    'loglik': model.score(X),
                    'model': model
                }
                
            elif method_name == 'cca':
                # For CCA, split data in half
                n_features_1 = X.shape[1] // 2
                X1 = X[:, :n_features_1]
                X2 = X[:, n_features_1:]
                
                model = CCA(n_components=min(n_components, min(X1.shape[1], X2.shape[1])))
                Z1, Z2 = model.fit_transform(X1, X2)
                
                results = {
                    'Z1': Z1,
                    'Z2': Z2,
                    'x_weights': model.x_weights_,
                    'y_weights': model.y_weights_,
                    'model': model
                }
                
            elif method_name == 'kmeans':
                model = KMeans(n_clusters=n_components, random_state=42)
                labels = model.fit_predict(X)
                
                results = {
                    'labels': labels,
                    'centers': model.cluster_centers_,
                    'inertia': model.inertia_,
                    'model': model
                }
                
            elif method_name == 'ica':
                from sklearn.decomposition import FastICA
                model = FastICA(n_components=n_components, random_state=42)
                Z = model.fit_transform(X)
                
                results = {
                    'components': model.components_,
                    'Z': Z,
                    'model': model
                }
                
        except Exception as e:
            self.logger.warning(f"Method {method_name} failed: {str(e)}")
            results = {'error': str(e)}
            
        return results
    
    def _run_sgfa_multiview(self, X_list: List[np.ndarray], **kwargs) -> Dict:
        """Run SGFA on multi-view data."""
        # Mock SGFA results for multi-view data
        return {
            'W': [np.random.randn(X.shape[1], 5) for X in X_list],
            'Z': np.random.randn(X_list[0].shape[0], 5),
            'log_likelihood': np.random.randn(),
            'n_views_used': len(X_list)
        }
    
    def _run_scalability_test(self, X_list: List[np.ndarray], **kwargs) -> Dict:
        """Run scalability test for all methods."""
        results = {}
        
        # SGFA
        with self.profiler.profile('sgfa_scalability') as p:
            sgfa_result = self._run_sgfa_multiview(X_list, **kwargs)
        
        sgfa_metrics = self.profiler.get_current_metrics()
        results['sgfa'] = {
            'result': sgfa_result,
            'execution_time': sgfa_metrics.execution_time,
            'peak_memory_gb': sgfa_metrics.peak_memory_gb
        }
        
        # Traditional methods
        X_concat = np.hstack(X_list)
        n_components = min(5, X_concat.shape[1] // 2)
        
        for method in ['pca', 'fa']:
            with self.profiler.profile(f'{method}_scalability') as p:
                method_result = self._run_traditional_method(X_concat, method, n_components)
            
            method_metrics = self.profiler.get_current_metrics()
            results[method] = {
                'result': method_result,
                'execution_time': method_metrics.execution_time,
                'peak_memory_gb': method_metrics.peak_memory_gb
            }
        
        return results
    
    def _analyze_sgfa_variants(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Analyze SGFA variant comparison results."""
        analysis = {
            'variant_summary': {},
            'performance_ranking': {},
            'convergence_analysis': {},
            'recommendations': []
        }
        
        # Summarize each variant
        for variant_name, result in results.items():
            metrics = performance_metrics[variant_name]
            
            analysis['variant_summary'][variant_name] = {
                'converged': result.get('convergence', False),
                'log_likelihood': result.get('log_likelihood', 0),
                'execution_time': metrics['execution_time'],
                'memory_usage': metrics['peak_memory_gb'],
                'iterations': metrics['convergence_iterations']
            }
        
        # Performance ranking
        time_ranking = sorted(performance_metrics.items(), key=lambda x: x[1]['execution_time'])
        memory_ranking = sorted(performance_metrics.items(), key=lambda x: x[1]['peak_memory_gb'])
        
        analysis['performance_ranking'] = {
            'fastest': [name for name, _ in time_ranking],
            'memory_efficient': [name for name, _ in memory_ranking]
        }
        
        # Convergence analysis
        converged_variants = [name for name, result in results.items() 
                            if result.get('convergence', False)]
        
        analysis['convergence_analysis'] = {
            'converged_variants': converged_variants,
            'convergence_rate': len(converged_variants) / len(results)
        }
        
        # Recommendations
        if 'standard' in converged_variants:
            analysis['recommendations'].append("Standard SGFA converged successfully")
        
        fastest_variant = time_ranking[0][0]
        analysis['recommendations'].append(f"Fastest variant: {fastest_variant}")
        
        most_memory_efficient = memory_ranking[0][0]
        analysis['recommendations'].append(f"Most memory efficient: {most_memory_efficient}")
        
        return analysis
    
    def _analyze_traditional_comparison(self, results: Dict, X_list: List[np.ndarray]) -> Dict:
        """Analyze traditional method comparison results."""
        analysis = {
            'method_summary': {},
            'multi_view_handling': {},
            'dimensionality_reduction_quality': {},
            'computational_efficiency': {}
        }
        
        # Method summary
        for method_name, result in results.items():
            if 'error' in result:
                analysis['method_summary'][method_name] = {'status': 'failed', 'error': result['error']}
                continue
                
            analysis['method_summary'][method_name] = {
                'status': 'success',
                'output_dimensions': self._get_output_dimensions(result, method_name)
            }
        
        # Multi-view handling assessment
        total_features = sum(X.shape[1] for X in X_list)
        for method_name, result in results.items():
            if method_name == 'sgfa':
                analysis['multi_view_handling'][method_name] = 'native_support'
            elif method_name == 'cca':
                analysis['multi_view_handling'][method_name] = 'pairwise_only'
            else:
                analysis['multi_view_handling'][method_name] = 'concatenation_required'
        
        return analysis
    
    def _analyze_multiview_capabilities(self, results: Dict) -> Dict:
        """Analyze multi-view capabilities of different methods."""
        analysis = {
            'view_scaling': {},
            'information_preservation': {},
            'method_comparison': {}
        }
        
        # Analyze how methods scale with number of views
        for view_key, view_results in results.items():
            n_views = int(view_key.split('_')[0])
            
            sgfa_result = view_results['sgfa']
            traditional_results = view_results['traditional']
            
            analysis['view_scaling'][n_views] = {
                'sgfa_likelihood': sgfa_result.get('log_likelihood', 0),
                'traditional_methods_available': list(traditional_results.keys()),
                'total_features': sum(view_results['view_dimensions'])
            }
        
        return analysis
    
    def _analyze_scalability(self, results: Dict) -> Dict:
        """Analyze scalability comparison results."""
        analysis = {
            'sample_scaling': {},
            'feature_scaling': {},
            'efficiency_trends': {},
            'scalability_ranking': {}
        }
        
        # Sample scaling analysis
        sample_results = results['sample_scalability']
        for sample_size, sample_result in sample_results.items():
            analysis['sample_scaling'][sample_size] = {}
            for method, method_result in sample_result.items():
                analysis['sample_scaling'][sample_size][method] = {
                    'execution_time': method_result['execution_time'],
                    'memory_usage': method_result['peak_memory_gb']
                }
        
        # Feature scaling analysis
        feature_results = results['feature_scalability']
        for feature_size, feature_result in feature_results.items():
            analysis['feature_scaling'][feature_size] = {}
            for method, method_result in feature_result.items():
                analysis['feature_scaling'][feature_size][method] = {
                    'execution_time': method_result['execution_time'],
                    'memory_usage': method_result['peak_memory_gb']
                }
        
        return analysis
    
    def _get_output_dimensions(self, result: Dict, method_name: str) -> int:
        """Get output dimensions for a method result."""
        if method_name == 'sgfa':
            return result['Z'].shape[1] if 'Z' in result else 0
        elif 'Z' in result:
            return result['Z'].shape[1]
        elif 'Z1' in result and 'Z2' in result:  # CCA
            return result['Z1'].shape[1]
        elif 'labels' in result:  # K-means
            return len(np.unique(result['labels']))
        else:
            return 0
    
    def _plot_sgfa_comparison(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for SGFA variant comparison."""
        plots = {}
        
        try:
            # Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('SGFA Variant Comparison', fontsize=16)
            
            variants = list(results.keys())
            
            # Execution time
            times = [performance_metrics[v]['execution_time'] for v in variants]
            axes[0, 0].bar(variants, times)
            axes[0, 0].set_title('Execution Time')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory usage
            memory = [performance_metrics[v]['peak_memory_gb'] for v in variants]
            axes[0, 1].bar(variants, memory)
            axes[0, 1].set_title('Peak Memory Usage')
            axes[0, 1].set_ylabel('Memory (GB)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Convergence iterations
            iterations = [performance_metrics[v]['convergence_iterations'] for v in variants]
            axes[1, 0].bar(variants, iterations)
            axes[1, 0].set_title('Convergence Iterations')
            axes[1, 0].set_ylabel('Iterations')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Log likelihood
            likelihoods = [results[v].get('log_likelihood', 0) for v in variants]
            axes[1, 1].bar(variants, likelihoods)
            axes[1, 1].set_title('Log Likelihood')
            axes[1, 1].set_ylabel('Log Likelihood')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plots['sgfa_variant_comparison'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create SGFA comparison plots: {str(e)}")
            
        return plots
    
    def _plot_traditional_comparison(self, results: Dict, performance_metrics: Dict) -> Dict:
        """Generate plots for traditional method comparison."""
        plots = {}
        
        try:
            # Method performance comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Traditional Method Comparison', fontsize=16)
            
            methods = list(performance_metrics.keys())
            
            # Execution time comparison
            times = [performance_metrics[m]['execution_time'] for m in methods]
            axes[0].bar(methods, times)
            axes[0].set_title('Execution Time by Method')
            axes[0].set_ylabel('Time (seconds)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Memory usage comparison
            memory = [performance_metrics[m]['peak_memory_gb'] for m in methods]
            axes[1].bar(methods, memory)
            axes[1].set_title('Peak Memory Usage by Method')
            axes[1].set_ylabel('Memory (GB)')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plots['traditional_method_comparison'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create traditional comparison plots: {str(e)}")
            
        return plots
    
    def _plot_multiview_comparison(self, results: Dict) -> Dict:
        """Generate plots for multi-view comparison."""
        plots = {}
        
        try:
            # Multi-view scaling plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Multi-view Capability Assessment', fontsize=16)
            
            view_counts = sorted([int(k.split('_')[0]) for k in results.keys()])
            
            # SGFA likelihood vs number of views
            sgfa_likelihoods = [results[f'{n}_views']['sgfa'].get('log_likelihood', 0) 
                              for n in view_counts]
            axes[0].plot(view_counts, sgfa_likelihoods, 'o-', label='SGFA')
            axes[0].set_title('Log Likelihood vs Number of Views')
            axes[0].set_xlabel('Number of Views')
            axes[0].set_ylabel('Log Likelihood')
            axes[0].legend()
            
            # Feature dimensions handled
            total_features = [sum(results[f'{n}_views']['view_dimensions']) 
                            for n in view_counts]
            axes[1].plot(view_counts, total_features, 's-', color='red')
            axes[1].set_title('Total Features vs Number of Views')
            axes[1].set_xlabel('Number of Views')
            axes[1].set_ylabel('Total Features')
            
            plt.tight_layout()
            plots['multiview_comparison'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create multi-view comparison plots: {str(e)}")
            
        return plots
    
    def _plot_scalability_comparison(self, results: Dict) -> Dict:
        """Generate plots for scalability comparison."""
        plots = {}
        
        try:
            # Scalability plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Scalability Comparison', fontsize=16)
            
            # Sample size scalability - execution time
            sample_results = results['sample_scalability']
            sample_sizes = sorted(sample_results.keys())
            
            methods = list(next(iter(sample_results.values())).keys())
            
            for method in methods:
                times = [sample_results[size][method]['execution_time'] for size in sample_sizes]
                axes[0, 0].plot(sample_sizes, times, 'o-', label=method)
            
            axes[0, 0].set_title('Execution Time vs Sample Size')
            axes[0, 0].set_xlabel('Sample Size')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].legend()
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_yscale('log')
            
            # Sample size scalability - memory
            for method in methods:
                memory = [sample_results[size][method]['peak_memory_gb'] for size in sample_sizes]
                axes[0, 1].plot(sample_sizes, memory, 's-', label=method)
            
            axes[0, 1].set_title('Memory Usage vs Sample Size')
            axes[0, 1].set_xlabel('Sample Size')
            axes[0, 1].set_ylabel('Memory (GB)')
            axes[0, 1].legend()
            axes[0, 1].set_xscale('log')
            
            # Feature size scalability - execution time
            feature_results = results['feature_scalability']
            feature_sizes = sorted(feature_results.keys())
            
            for method in methods:
                times = [feature_results[size][method]['execution_time'] for size in feature_sizes]
                axes[1, 0].plot(feature_sizes, times, 'o-', label=method)
            
            axes[1, 0].set_title('Execution Time vs Feature Size')
            axes[1, 0].set_xlabel('Feature Size')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].legend()
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_yscale('log')
            
            # Feature size scalability - memory
            for method in methods:
                memory = [feature_results[size][method]['peak_memory_gb'] for size in feature_sizes]
                axes[1, 1].plot(feature_sizes, memory, 's-', label=method)
            
            axes[1, 1].set_title('Memory Usage vs Feature Size')
            axes[1, 1].set_xlabel('Feature Size')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].legend()
            axes[1, 1].set_xscale('log')
            
            plt.tight_layout()
            plots['scalability_comparison'] = fig
            
        except Exception as e:
            self.logger.warning(f"Failed to create scalability comparison plots: {str(e)}")

        return plots


def run_method_comparison(config):
    """Run method comparison experiments with remote workstation integration."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Method Comparison Experiments")

    try:
        # Add project root to path for imports
        import sys
        import os

        # Calculate the correct project root path
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))  # Go up from experiments/ to project root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from experiments.framework import ExperimentFramework, ExperimentConfig

        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )

        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_method_comparison",
            description="Compare SGFA model variants on remote workstation",
            dataset="qmap_pd",
            data_dir=config['data']['data_dir']
        )

        # COMPREHENSIVE MODELS FRAMEWORK INTEGRATION
        from models.models_integration import integrate_models_with_pipeline

        logger.info("🧠 Integrating comprehensive models framework...")
        model_type, model_instance, models_summary = integrate_models_with_pipeline(
            config=config
        )

        # COMPREHENSIVE ANALYSIS FRAMEWORK INTEGRATION
        from analysis.analysis_integration import integrate_analysis_with_pipeline

        logger.info("📊 Integrating comprehensive analysis framework...")
        data_manager, model_runner, analysis_summary = integrate_analysis_with_pipeline(
            config=config,
            data_dir=config['data']['data_dir']
        )

        # COMPREHENSIVE PERFORMANCE OPTIMIZATION INTEGRATION
        from performance.performance_integration import integrate_performance_with_pipeline

        logger.info("⚡ Integrating comprehensive performance optimization framework...")
        performance_manager, performance_summary = integrate_performance_with_pipeline(
            config=config,
            data_dir=config['data']['data_dir']
        )

        # Load data with structured analysis framework if available
        if data_manager and analysis_summary.get('integration_summary', {}).get('structured_analysis', False):
            logger.info("📊 Using structured DataManager for data loading...")
            from analysis.analysis_integration import _wrap_analysis_framework

            # Use structured data loading
            analysis_wrapper = _wrap_analysis_framework(data_manager, model_runner, analysis_summary)
            X_list, structured_data_info = analysis_wrapper.load_and_prepare_data()

            if structured_data_info.get('data_loaded', False):
                logger.info("✅ Data loaded with structured analysis framework")
                logger.info(f"   Loader: {structured_data_info.get('loader', 'unknown')}")
                if structured_data_info.get('preprocessing_applied', False):
                    logger.info(f"   Preprocessing: Applied via DataManager")

                # Store structured data info as preprocessing_info for compatibility
                preprocessing_info = {
                    'preprocessing_integration': True,
                    'loader_type': 'structured_analysis_framework',
                    'structured_data_info': structured_data_info,
                    'data_manager_used': True
                }
            else:
                logger.warning("⚠️ Structured data loading failed - falling back to preprocessing integration")
                # Fall back to preprocessing integration
                from data.preprocessing_integration import apply_preprocessing_to_pipeline
                X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                    config=config,
                    data_dir=config['data']['data_dir'],
                    auto_select_strategy=True
                )
        else:
            # Use preprocessing integration
            from data.preprocessing_integration import apply_preprocessing_to_pipeline

            logger.info("🔧 Applying comprehensive preprocessing integration...")
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config,
                data_dir=config['data']['data_dir'],
                auto_select_strategy=True
            )

        # Apply performance optimization to loaded data
        if performance_manager:
            logger.info("⚡ Applying performance optimization to data loading...")
            X_list = performance_manager.optimize_data_arrays(X_list)
        else:
            logger.info("⚡ Performance framework unavailable - using basic data loading")

        # Update models framework with data characteristics
        if X_list and models_summary:
            logger.info("🧠 Updating models framework with data characteristics...")
            data_characteristics = {
                'n_subjects': len(X_list[0]),
                'n_views': len(X_list),
                'total_features': sum(X.shape[1] for X in X_list),
                'view_dimensions': [X.shape[1] for X in X_list],
                'has_imaging_data': any(X.shape[1] > 1000 for X in X_list),
                'imaging_views': [i for i, X in enumerate(X_list) if X.shape[1] > 1000]
            }

            # Re-run model selection with data characteristics
            from models.models_integration import integrate_models_with_pipeline
            model_type, model_instance, updated_models_summary = integrate_models_with_pipeline(
                config=config,
                X_list=X_list,
                data_characteristics=data_characteristics
            )
            models_summary = updated_models_summary

        # Create data structure compatible with existing pipeline
        data = {
            'X_list': X_list,
            'view_names': preprocessing_info.get('data_summary', {}).get('view_names', [f'view_{i}' for i in range(len(X_list))]),
            'preprocessing_info': preprocessing_info
        }

        # Run the experiment
        def method_comparison_experiment(config, output_dir, **kwargs):
            import numpy as np

            logger.info("Running comprehensive method comparison...")

            # Log integration summaries
            logger.info("🧠 MODELS FRAMEWORK SUMMARY:")
            logger.info(f"   Framework: {models_summary.get('integration_summary', {}).get('framework_type', 'unknown')}")
            logger.info(f"   Model type: {models_summary.get('model_type', 'unknown')}")
            logger.info(f"   Model factory: {models_summary.get('integration_summary', {}).get('model_factory', 'unknown')}")
            logger.info(f"   Model instance: {models_summary.get('integration_summary', {}).get('model_instance', 'unknown')}")
            logger.info(f"   Available models: {', '.join(models_summary.get('integration_summary', {}).get('available_models', []))}")
            logger.info(f"   Features: {', '.join([f'{k}={v}' for k, v in models_summary.get('integration_summary', {}).get('features', {}).items()])}")

            logger.info("📊 ANALYSIS FRAMEWORK SUMMARY:")
            logger.info(f"   Framework: {analysis_summary.get('integration_summary', {}).get('framework_type', 'unknown')}")
            logger.info(f"   DataManager: {analysis_summary.get('integration_summary', {}).get('data_manager', 'unknown')}")
            logger.info(f"   ModelRunner: {analysis_summary.get('integration_summary', {}).get('model_runner', 'unknown')}")
            logger.info(f"   Components: {', '.join(analysis_summary.get('integration_summary', {}).get('components', []))}")
            logger.info(f"   Dependencies: {', '.join([f'{k}={v}' for k, v in analysis_summary.get('integration_summary', {}).get('dependencies', {}).items()])}")

            logger.info("⚡ PERFORMANCE OPTIMIZATION SUMMARY:")
            logger.info(f"   Strategy: {performance_summary.get('strategy_selection', {}).get('selected_strategy', 'unknown')}")
            logger.info(f"   Framework: {performance_summary.get('integration_summary', {}).get('framework_type', 'unknown')}")

            logger.info("🔧 PREPROCESSING INTEGRATION SUMMARY:")
            logger.info(f"   Strategy: {preprocessing_info.get('strategy_selection', {}).get('selected_strategy', 'unknown')}")
            logger.info(f"   Reason: {preprocessing_info.get('strategy_selection', {}).get('reason', 'not specified')}")

            # Return basic results for now
            return {
                'status': 'completed',
                'models_summary': models_summary,
                'analysis_summary': analysis_summary,
                'performance_summary': performance_summary,
                'data': data
            }

        # Run experiment using framework
        result = framework.run_experiment(
            experiment_function=method_comparison_experiment,
            config=exp_config,
            data=data
        )

        logger.info("Method comparison experiments completed")
        return result

    except Exception as e:
        logger.error(f"Method comparison failed: {e}")
        return None