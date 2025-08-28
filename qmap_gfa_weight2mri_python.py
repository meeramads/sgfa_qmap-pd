"""
Weight-to-MRI Integration for Sparse Bayesian Group Factor Analysis.

This module provides functionality to map factor loadings back to MRI space
for visualization and interpretation of neuroimaging results.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorToMRIMapper:
    """
    Maps factor loadings back to MRI space for neuroimaging visualization.
    
    This class handles the conversion of learned factor loadings (W matrix)
    back to 3D brain space using position lookup files.
    """
    
    def __init__(self, base_dir: str, reference_mri: str = None):
        """
        Initialize the mapper.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing position lookup files
        reference_mri : str, optional
            Path to reference MRI file. If None, will look in standard location.
        """
        self.base_dir = Path(base_dir)
        self.reference_mri = reference_mri
        
        # Standard paths
        self.position_dir = self.base_dir / "position_lookup"
        
        if reference_mri is None:
            # Look for standard reference MRI
            ref_path = self.base_dir / "mri_reference" / "average_space-qmap-shoot-384_pdw-brain.nii"
            if ref_path.exists():
                self.reference_mri = str(ref_path)
            else:
                logger.warning("No reference MRI specified and standard path not found")
        
        # Load position lookup files
        self._load_position_files()
    
    def _load_position_files(self):
        """Load position lookup files for each ROI."""
        self.position_files = {}
        self.roi_names = []
        
        # Standard ROI position files
        roi_files = [
            "position_sn_voxels.tsv",
            "position_putamen_voxels.tsv", 
            "position_lentiform_voxels.tsv"
        ]
        
        for roi_file in roi_files:
            roi_path = self.position_dir / roi_file
            if roi_path.exists():
                # Extract ROI name from filename
                roi_name = roi_file.replace("position_", "").replace("_voxels.tsv", "")
                
                # Load position indices
                positions = pd.read_csv(roi_path, sep='\t', header=None).values.flatten()
                self.position_files[roi_name] = positions
                self.roi_names.append(roi_name)
                
                logger.info(f"Loaded {len(positions)} positions for {roi_name}")
            else:
                logger.warning(f"Position file not found: {roi_path}")
        
        if not self.position_files:
            logger.error("No position files found!")
    
    def map_weights_to_mri(self, W: np.ndarray, view_names: List[str], 
                          Dm: List[int], factor_idx: int = 0,
                          output_dir: str = "factor_maps", 
                          fill_value: float = 0.0) -> Dict[str, str]:
        """
        Map factor loadings to MRI space.
        
        Parameters:
        -----------
        W : np.ndarray
            Factor loading matrix (features x factors)
        view_names : List[str]
            Names of data views
        Dm : List[int]
            Dimensions of each view
        factor_idx : int
            Which factor to map (default: 0)
        output_dir : str
            Directory to save output NIfTI files
        fill_value : float
            Value for voxels outside ROI
            
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping ROI names to output file paths
        """
        
        if self.reference_mri is None:
            raise ValueError("Reference MRI required for mapping")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract factor loadings for specified factor
        factor_loadings = W[:, factor_idx]
        
        output_files = {}
        feature_start = 0
        
        for view_idx, (view_name, dim) in enumerate(zip(view_names, Dm)):
            # Get loadings for this view
            view_loadings = factor_loadings[feature_start:feature_start + dim]
            
            # Check if this is an imaging view that we can map
            if self._is_imaging_view(view_name):
                # Map each ROI in this view
                roi_outputs = self._map_imaging_view(
                    view_loadings, view_name, factor_idx, output_dir, fill_value
                )
                output_files.update(roi_outputs)
            else:
                logger.info(f"Skipping non-imaging view: {view_name}")
            
            feature_start += dim
        
        return output_files
    
    def _is_imaging_view(self, view_name: str) -> bool:
        """Check if a view contains imaging data that can be mapped."""
        imaging_keywords = ['imaging', 'volume', 'sn', 'putamen', 'lentiform']
        return any(keyword in view_name.lower() for keyword in imaging_keywords)
    
    def _map_imaging_view(self, loadings: np.ndarray, view_name: str, 
                         factor_idx: int, output_dir: Path, 
                         fill_value: float) -> Dict[str, str]:
        """Map loadings for an imaging view to MRI space."""
        
        output_files = {}
        
        if 'imaging' in view_name.lower() and len(self.roi_names) > 1:
            # Multi-ROI imaging view - split loadings across ROIs
            loading_start = 0
            
            for roi_name in self.roi_names:
                if roi_name in self.position_files:
                    positions = self.position_files[roi_name]
                    roi_dim = len(positions)
                    
                    if loading_start + roi_dim <= len(loadings):
                        roi_loadings = loadings[loading_start:loading_start + roi_dim]
                        
                        output_file = self._create_roi_nifti(
                            roi_loadings, positions, roi_name, 
                            factor_idx, output_dir, fill_value
                        )
                        
                        if output_file:
                            output_files[f"{view_name}_{roi_name}"] = output_file
                        
                        loading_start += roi_dim
                    else:
                        logger.warning(f"Not enough loadings for {roi_name}")
        
        else:
            # Single ROI view - try to match by name
            roi_name = self._infer_roi_name(view_name)
            if roi_name and roi_name in self.position_files:
                positions = self.position_files[roi_name]
                
                if len(loadings) == len(positions):
                    output_file = self._create_roi_nifti(
                        loadings, positions, roi_name,
                        factor_idx, output_dir, fill_value
                    )
                    
                    if output_file:
                        output_files[view_name] = output_file
                else:
                    logger.warning(f"Loading dimension mismatch for {view_name}: "
                                 f"{len(loadings)} vs {len(positions)}")
        
        return output_files
    
    def _infer_roi_name(self, view_name: str) -> Optional[str]:
        """Infer ROI name from view name."""
        view_lower = view_name.lower()
        
        for roi_name in self.roi_names:
            if roi_name in view_lower:
                return roi_name
        
        # Try partial matches
        if 'putamen' in view_lower:
            return 'putamen'
        elif 'lentiform' in view_lower:
            return 'lentiform'  
        elif 'sn' in view_lower or 'substantia' in view_lower:
            return 'sn'
        
        return None
    
    def _create_roi_nifti(self, weights: np.ndarray, positions: np.ndarray,
                         roi_name: str, factor_idx: int, output_dir: Path,
                         fill_value: float) -> Optional[str]:
        """Create NIfTI file for ROI factor loadings."""
        
        try:
            # Load reference MRI
            ref_img = nib.load(self.reference_mri)
            ref_data = ref_img.get_fdata()
            
            # Initialize output image with fill value
            output_img = np.full(ref_data.shape, fill_value, dtype=np.float32)
            
            # Convert 1-based MATLAB indices to 0-based Python indices
            positions_python = positions.astype(int) - 1
            
            # Bounds check
            flat_n = np.prod(ref_data.shape)
            valid_indices = (positions_python >= 0) & (positions_python < flat_n)
            
            if not np.all(valid_indices):
                logger.warning(f"Some position indices for {roi_name} are out of bounds")
                positions_python = positions_python[valid_indices]
                weights = weights[valid_indices]
            
            # Convert linear indices to 3D coordinates
            coords = np.unravel_index(positions_python, ref_data.shape)
            
            # Assign weights to voxel positions
            output_img[coords] = weights.astype(np.float32)
            
            # Create output filename
            output_file = output_dir / f"factor_{factor_idx+1}_{roi_name}_loadings.nii"
            
            # Create and save NIfTI
            output_nii = nib.Nifti1Image(output_img, ref_img.affine, ref_img.header.copy())
            output_nii.header.set_data_dtype(np.float32)
            output_nii.header['descrip'] = f'Factor {factor_idx+1} loadings for {roi_name}'
            
            nib.save(output_nii, output_file)
            
            logger.info(f"Created factor map: {output_file}")
            logger.info(f"  Non-zero voxels: {np.sum(output_img != fill_value)}")
            logger.info(f"  Value range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create NIfTI for {roi_name}: {e}")
            return None
    
    def map_all_factors(self, W: np.ndarray, view_names: List[str],
                       Dm: List[int], output_dir: str = "factor_maps",
                       fill_value: float = 0.0) -> Dict[int, Dict[str, str]]:
        """
        Map all factors to MRI space.
        
        Returns:
        --------
        Dict[int, Dict[str, str]]
            Dictionary mapping factor indices to ROI output files
        """
        
        all_outputs = {}
        n_factors = W.shape[1]
        
        logger.info(f"Mapping {n_factors} factors to MRI space...")
        
        for factor_idx in range(n_factors):
            logger.info(f"Processing factor {factor_idx + 1}/{n_factors}")
            
            factor_outputs = self.map_weights_to_mri(
                W, view_names, Dm, factor_idx, output_dir, fill_value
            )
            
            all_outputs[factor_idx] = factor_outputs
        
        logger.info(f"Generated {sum(len(outputs) for outputs in all_outputs.values())} NIfTI files")
        
        return all_outputs


# Integration functions for existing workflow

def integrate_with_visualization(results_dir: str, data: Dict, W: np.ndarray, 
                                base_dir: str, factor_indices: List[int] = None):
    """
    Integrate factor-to-MRI mapping with existing visualization pipeline.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing analysis results  
    data : Dict
        Data dictionary from loader
    W : np.ndarray
        Factor loading matrix
    base_dir : str
        Base directory for position lookup files
    factor_indices : List[int], optional
        Specific factors to map (if None, maps all)
    """
    
    # Create output directory for factor maps
    factor_maps_dir = Path(results_dir) / "factor_maps"
    factor_maps_dir.mkdir(exist_ok=True)
    
    # Initialize mapper
    mapper = FactorToMRIMapper(base_dir)
    
    # Get view information from data
    view_names = data.get('view_names', [])
    Dm = [X.shape[1] for X in data.get('X_list', [])]
    
    # Map specified factors or all factors
    if factor_indices is None:
        factor_indices = list(range(W.shape[1]))
    
    all_factor_maps = {}
    
    for factor_idx in factor_indices:
        logger.info(f"Mapping factor {factor_idx + 1} to MRI space")
        
        factor_maps = mapper.map_weights_to_mri(
            W, view_names, Dm, factor_idx, 
            output_dir=str(factor_maps_dir)
        )
        
        all_factor_maps[factor_idx] = factor_maps
    
    # Save mapping summary
    summary_file = factor_maps_dir / "factor_mapping_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Factor Loading to MRI Mapping Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for factor_idx, factor_maps in all_factor_maps.items():
            f.write(f"Factor {factor_idx + 1}:\n")
            for roi_name, nii_file in factor_maps.items():
                f.write(f"  {roi_name}: {nii_file}\n")
            f.write("\n")
        
        f.write(f"Total NIfTI files created: {sum(len(maps) for maps in all_factor_maps.values())}\n")
    
    logger.info(f"Factor mapping summary saved to {summary_file}")
    
    return all_factor_maps


def add_to_qmap_visualization(data: Dict, res_dir: str, args, hypers, 
                             base_dir: str = None, brun: int = None):
    """
    Add factor-to-MRI mapping to the existing qMAP-PD visualization function.
    
    This function should be called from within the qmap_pd() function in visualization.py
    """
    
    if base_dir is None:
        base_dir = getattr(args, 'data_dir', 'qMAP-PD_data')
    
    # Load results if not provided
    if brun is None:
        # Find best run (simplified version)
        best_run = 1  # Default to first run
        for r in range(getattr(args, 'num_runs', 10)):
            rob_path = f"{res_dir}/[{r+1}]Robust_params.dictionary"
            if os.path.exists(rob_path) and os.path.getsize(rob_path) > 5:
                best_run = r + 1
                break
        brun = best_run
    
    # Load factor loadings
    rob_path = f"{res_dir}/[{brun}]Robust_params.dictionary"
    if os.path.exists(rob_path) and os.path.getsize(rob_path) > 5:
        import pickle
        with open(rob_path, 'rb') as f:
            rob_params = pickle.load(f)
        W = rob_params.get('W')
    else:
        logger.warning("Could not load robust parameters for factor mapping")
        return
    
    if W is not None:
        logger.info("Adding factor-to-MRI mapping to visualization...")
        
        try:
            factor_maps = integrate_with_visualization(
                res_dir, data, W, base_dir, 
                factor_indices=list(range(min(5, W.shape[1])))  # Map first 5 factors
            )
            
            logger.info("Factor-to-MRI mapping completed successfully")
            return factor_maps
            
        except Exception as e:
            logger.error(f"Factor-to-MRI mapping failed: {e}")
            return None
    else:
        logger.warning("No factor loadings (W) found for mapping")
        return None


# Standalone utility functions

def create_factor_overlay_images(factor_maps: Dict[int, Dict[str, str]], 
                                reference_mri: str, output_dir: str = "overlay_images"):
    """
    Create overlay images combining multiple ROIs for each factor.
    
    This creates composite images showing all ROIs for each factor in a single view.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_img = nib.load(reference_mri)
    
    for factor_idx, roi_maps in factor_maps.items():
        logger.info(f"Creating overlay for factor {factor_idx + 1}")
        
        # Initialize combined image
        combined_img = np.zeros(ref_img.shape, dtype=np.float32)
        
        # Load and combine all ROI maps for this factor
        for roi_name, nii_file in roi_maps.items():
            if os.path.exists(nii_file):
                roi_img = nib.load(nii_file)
                roi_data = roi_img.get_fdata()
                
                # Add ROI data to combined image (could also use max, etc.)
                combined_img += roi_data
        
        # Save combined image
        output_file = output_dir / f"factor_{factor_idx+1}_combined_overlay.nii"
        combined_nii = nib.Nifti1Image(combined_img, ref_img.affine, ref_img.header.copy())
        combined_nii.header.set_data_dtype(np.float32)
        combined_nii.header['descrip'] = f'Factor {factor_idx+1} combined ROI loadings'
        
        nib.save(combined_nii, output_file)
        logger.info(f"Created combined overlay: {output_file}")


def extract_top_voxels(factor_maps: Dict[int, Dict[str, str]], 
                      top_k: int = 100) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract coordinates and weights of top-k voxels for each factor and ROI.
    
    Returns:
    --------
    Dict mapping factor_idx -> roi_name -> array of (x, y, z, weight) coordinates
    """
    
    top_voxels = {}
    
    for factor_idx, roi_maps in factor_maps.items():
        factor_top = {}
        
        for roi_name, nii_file in roi_maps.items():
            if os.path.exists(nii_file):
                # Load NIfTI file
                img = nib.load(nii_file)
                data = img.get_fdata()
                
                # Find non-zero voxels
                nonzero_coords = np.nonzero(data)
                nonzero_weights = data[nonzero_coords]
                
                # Get top-k by absolute weight
                top_indices = np.argsort(np.abs(nonzero_weights))[-top_k:]
                
                # Extract coordinates and weights
                top_coords = np.column_stack([
                    nonzero_coords[0][top_indices],
                    nonzero_coords[1][top_indices], 
                    nonzero_coords[2][top_indices],
                    nonzero_weights[top_indices]
                ])
                
                factor_top[roi_name] = top_coords
                
        top_voxels[factor_idx] = factor_top
    
    return top_voxels


# Example usage and testing
def example_usage():
    """
    Example of how to use the FactorToMRIMapper.
    """
    
    # Setup (adjust paths as needed)
    base_dir = "qMAP-PD_data"
    results_dir = "../results/qmap_pd/sparseGFA_K20_4chs_pW33_s5000_reghsZ"
    
    # Initialize mapper
    mapper = FactorToMRIMapper(base_dir)
    
    # Example: Load results and create factor maps
    # This would typically be called from within your analysis pipeline
    
    # Dummy factor loadings for demonstration
    W_example = np.random.randn(1000, 5)  # 1000 features, 5 factors
    view_names = ["imaging", "clinical"]
    Dm = [950, 50]  # 950 imaging features, 50 clinical features
    
    # Map first factor
    factor_maps = mapper.map_weights_to_mri(
        W_example, view_names, Dm, factor_idx=0,
        output_dir="example_factor_maps"
    )
    
    print("Created factor maps:")
    for roi_name, nii_file in factor_maps.items():
        print(f"  {roi_name}: {nii_file}")


if __name__ == "__main__":
    example_usage()