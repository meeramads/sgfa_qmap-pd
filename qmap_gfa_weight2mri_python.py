"""
Weight-to-MRI Integration for Sparse Bayesian Group Factor Analysis.

This module provides functionality to map factor loadings back to MRI space
for visualization and interpretation of neuroimaging results.
"""

import json
import os
import time
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

        # Add volume matrix paths for reconstruction
        self.volume_files = {}
        self._load_volume_files()
        
    
    def _load_volume_files(self):
        """Load volume matrix file paths for reconstruction."""
        volumes_dir = self.base_dir / "volume_matrices"
        
        for roi_name in self.roi_names:
            vol_file = volumes_dir / f"volume_{roi_name}_voxels.tsv"
            if vol_file.exists():
                self.volume_files[roi_name] = vol_file
                logging.info(f"Found volume file for {roi_name}: {vol_file}")

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
    

    def reconstruct_subject_data(self, subject_ids: List[int], 
                               output_dir: str = "subject_reconstructions",
                               fill_value: float = 0.0,
                               create_overlays: bool = True) -> Dict[int, Dict[str, str]]:
        """
        Reconstruct original qMRI data for specific subjects.
        
        Parameters:
        -----------
        subject_ids : List[int]
            List of subject IDs to reconstruct (1-based indexing)
        output_dir : str
            Directory to save reconstructions
        fill_value : float
            Value for voxels outside ROI
        create_overlays : bool
            Whether to create overlay visualization images
        
        Returns:
        --------
        Dict[int, Dict[str, str]]
            Dictionary mapping subject_id -> roi_name -> nifti_file_path
        """
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if create_overlays:
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
        
        reconstructions = {}
        
        for subject_id in subject_ids:
            logging.info(f"Reconstructing subject {subject_id}")
            subject_files = {}
            roi_images = []
            
            for roi_name in self.roi_names:
                if roi_name in self.volume_files and roi_name in self.position_files:
                    output_file = output_dir / f"subject_{subject_id:02d}_{roi_name}.nii"
                    
                    # Reconstruct single ROI
                    img_data = self._reconstruct_roi_single(
                        vol_file=self.volume_files[roi_name],
                        pos_file=self.position_dir / f"position_{roi_name}_voxels.tsv",
                        output_file=output_file,
                        subject_id=subject_id,
                        fill_value=fill_value
                    )
                    
                    if img_data is not None:
                        subject_files[roi_name] = str(output_file)
                        roi_images.append((roi_name, img_data))
                        logging.info(f"Created reconstruction: {output_file}")
            
            # Create overlay visualization if requested
            if create_overlays and roi_images:
                overlay_file = overlay_dir / f"subject_{subject_id:02d}_overlay.png"
                self._create_overlay_plot(roi_images, overlay_file)
            
            reconstructions[subject_id] = subject_files
        
        logging.info(f"Completed reconstructions for {len(subject_ids)} subjects")
        return reconstructions
    
    def _reconstruct_roi_single(self, vol_file: Path, pos_file: Path, 
                              output_file: Path, subject_id: int, 
                              fill_value: float = 0.0) -> Optional[np.ndarray]:
        """
        Reconstruct a single ROI for one subject.
        
        Returns the image data for overlay creation.
        """
        try:
            # Load volume and position data
            V = pd.read_csv(vol_file, sep='\t', header=None).values
            pos = pd.read_csv(pos_file, sep='\t', header=None).values.flatten()
            
            # Validate inputs
            if V.shape[1] != len(pos):
                raise ValueError(f"Volume columns ({V.shape[1]}) != position rows ({len(pos)})")
            
            if subject_id < 1 or subject_id > V.shape[0]:
                raise ValueError(f"Subject ID {subject_id} out of range [1, {V.shape[0]}]")
            
            # Load reference image
            ref_img = nib.load(self.reference_mri)
            ref_data = ref_img.get_fdata()
            img_shape = ref_data.shape
            
            # Initialize output image
            if np.isnan(fill_value):
                img = np.full(img_shape, np.nan, dtype=np.float32)
            else:
                img = np.full(img_shape, fill_value, dtype=np.float32)
            
            # Convert 1-based MATLAB indices to 0-based Python indices
            pos_0based = pos.astype(int) - 1
            
            # Bounds check
            flat_n = np.prod(img_shape)
            if np.any((pos_0based < 0) | (pos_0based >= flat_n)):
                raise ValueError("Position indices fall outside reference image size")
            
            # Insert voxel values (subject_id is 1-based, convert to 0-based)
            vals = V[subject_id - 1, :].astype(np.float32)
            
            # Convert linear indices to subscripts and assign values
            img_flat = img.flatten()
            img_flat[pos_0based] = vals
            img = img_flat.reshape(img_shape)
            
            # Create and save NIfTI
            out_img = nib.Nifti1Image(img, ref_img.affine, ref_img.header.copy())
            out_img.header.set_data_dtype(np.float32)
            nib.save(out_img, output_file)
            
            return img
            
        except Exception as e:
            logging.error(f"Failed to reconstruct {vol_file.stem} for subject {subject_id}: {e}")
            return None
        
    def _create_overlay_plot(self, roi_images: List[Tuple[str, np.ndarray]], 
                           output_file: Path):
        """
        Create overlay plot showing multiple ROIs for one subject.
        """
        try:
            import matplotlib.pyplot as plt
            
            # Load reference image
            ref_img = nib.load(self.reference_mri)
            ref_data = ref_img.get_fdata()
            
            # Get middle slices for visualization
            mid_x, mid_y, mid_z = [s // 2 for s in ref_data.shape]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            # Colors for different ROIs
            roi_colors = ['Reds', 'Blues', 'Greens', 'Purples']
            
            for i, view_axis in enumerate(['sagittal', 'coronal', 'axial']):
                # Reference slice
                if view_axis == 'sagittal':
                    ref_slice = ref_data[mid_x, :, :]
                elif view_axis == 'coronal': 
                    ref_slice = ref_data[:, mid_y, :]
                else:  # axial
                    ref_slice = ref_data[:, :, mid_z]
                
                # Show reference
                axes[i].imshow(ref_slice.T, cmap='gray', alpha=0.7, origin='lower')
                axes[i].set_title(f'Reference - {view_axis.title()}')
                axes[i].axis('off')
                
                # Show overlay
                axes[i+3].imshow(ref_slice.T, cmap='gray', alpha=0.5, origin='lower')
                
                # Overlay ROIs
                for j, (roi_name, roi_data) in enumerate(roi_images):
                    if view_axis == 'sagittal':
                        roi_slice = roi_data[mid_x, :, :]
                    elif view_axis == 'coronal':
                        roi_slice = roi_data[:, mid_y, :]
                    else:  # axial
                        roi_slice = roi_data[:, :, mid_z]
                    
                    # Mask non-zero voxels
                    mask = roi_slice != 0
                    if np.any(mask):
                        axes[i+3].imshow(roi_slice.T, cmap=roi_colors[j % len(roi_colors)], 
                                       alpha=0.6, origin='lower')
                
                axes[i+3].set_title(f'Overlay - {view_axis.title()}')
                axes[i+3].axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=plt.cm.get_cmap(roi_colors[i % len(roi_colors)])(0.7), 
                                   label=roi_name) 
                             for i, (roi_name, _) in enumerate(roi_images)]
            fig.legend(handles=legend_elements, loc='upper right')
            
            plt.suptitle(f'Subject Reconstruction Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logging.debug(f"Created overlay plot: {output_file}")
            
        except Exception as e:
            logging.warning(f"Could not create overlay plot: {e}")
    
    def batch_reconstruct_subjects(self, n_subjects: int = None, 
                                 subject_range: Tuple[int, int] = None,
                                 output_dir: str = "batch_reconstructions") -> Dict:
        """
        Batch reconstruct multiple subjects with smart defaults.
        
        Parameters:
        -----------
        n_subjects : int, optional
            Number of subjects to reconstruct (from beginning)
        subject_range : Tuple[int, int], optional
            Range of subjects to reconstruct (start, end) - inclusive
        output_dir : str
            Output directory
        
        Returns:
        --------
        Dict with reconstruction results and statistics
        """
        
        # Determine subject list
        if subject_range is not None:
            start, end = subject_range
            subject_ids = list(range(start, end + 1))
        elif n_subjects is not None:
            subject_ids = list(range(1, n_subjects + 1))
        else:
            # Default: reconstruct first 10 subjects or all if fewer
            max_subjects = 87  # qMAP-PD default, could be made dynamic
            subject_ids = list(range(1, min(11, max_subjects + 1)))
        
        logging.info(f"Batch reconstructing {len(subject_ids)} subjects: {subject_ids}")
        
        # Perform reconstructions
        start_time = time.time()
        reconstructions = self.reconstruct_subject_data(
            subject_ids=subject_ids,
            output_dir=output_dir,
            create_overlays=True
        )
        elapsed_time = time.time() - start_time
        
        # Compile statistics
        stats = {
            'n_subjects_requested': len(subject_ids),
            'n_subjects_completed': len(reconstructions),
            'n_rois_per_subject': len(self.roi_names),
            'total_files_created': sum(len(files) for files in reconstructions.values()),
            'processing_time_seconds': elapsed_time,
            'success_rate': len(reconstructions) / len(subject_ids) if subject_ids else 0,
            'output_directory': str(Path(output_dir).resolve())
        }
        
        # Log summary
        logging.info(f"Batch reconstruction completed:")
        logging.info(f"  Subjects: {stats['n_subjects_completed']}/{stats['n_subjects_requested']}")
        logging.info(f"  Files created: {stats['total_files_created']}")
        logging.info(f"  Time: {stats['processing_time_seconds']:.1f}s")
        logging.info(f"  Success rate: {stats['success_rate']:.1%}")
        
        return {
            'reconstructions': reconstructions,
            'statistics': stats,
            'subject_ids': subject_ids
        }
    
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

# Add new integration function
def integrate_subject_reconstructions_with_factors(results_dir: str, data: Dict, 
                                                 W: np.ndarray, base_dir: str,
                                                 subject_ids: List[int] = None) -> Dict:
    """
    Create both factor maps and subject reconstructions in one integrated workflow.
    
    Parameters:
    -----------
    results_dir : str
        Results directory
    data : Dict
        Data dictionary from loader
    W : np.ndarray
        Factor loading matrix
    base_dir : str
        Base directory containing qMAP-PD data
    subject_ids : List[int], optional
        Specific subjects to reconstruct
    
    Returns:
    --------
    Dict with both factor maps and subject reconstructions
    """
    
    # Initialize mapper
    mapper = FactorToMRIMapper(base_dir)
    
    # Create comprehensive brain visualization directory
    brain_viz_dir = Path(results_dir) / "comprehensive_brain_visualization"
    brain_viz_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # 1. Create factor maps
    try:
        view_names = data.get('view_names', [])
        Dm = [X.shape[1] for X in data.get('X_list', [])]
        
        factor_maps = {}
        for factor_idx in range(min(10, W.shape[1])):  # First 10 factors
            factor_output = mapper.map_weights_to_mri(
                W, view_names, Dm, factor_idx,
                output_dir=str(brain_viz_dir / "factor_maps")
            )
            factor_maps[factor_idx] = factor_output
        
        results['factor_maps'] = factor_maps
        logging.info(f"Created factor maps for {len(factor_maps)} factors")
        
    except Exception as e:
        logging.error(f"Factor mapping failed: {e}")
        results['factor_maps'] = {}
    
    # 2. Create subject reconstructions
    try:
        if subject_ids is None:
            # Smart default: reconstruct first 5 subjects
            n_subjects = len(data.get('subject_ids', []))
            subject_ids = list(range(1, min(6, n_subjects + 1)))
        
        recon_results = mapper.batch_reconstruct_subjects(
            subject_range=(min(subject_ids), max(subject_ids)),
            output_dir=str(brain_viz_dir / "subject_reconstructions")
        )
        
        results['subject_reconstructions'] = recon_results
        logging.info(f"Created reconstructions for {len(subject_ids)} subjects")
        
    except Exception as e:
        logging.error(f"Subject reconstruction failed: {e}")
        results['subject_reconstructions'] = {}
    
    # 3. Create summary visualization
    try:
        _create_comprehensive_brain_summary(brain_viz_dir, results)
        results['summary_created'] = True
    except Exception as e:
        logging.warning(f"Could not create brain summary: {e}")
        results['summary_created'] = False
    
    # Save metadata
    metadata_file = brain_viz_dir / "brain_visualization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_factors_mapped': len(results.get('factor_maps', {})),
            'n_subjects_reconstructed': len(subject_ids) if subject_ids else 0,
            'subject_ids': subject_ids,
            'output_directory': str(brain_viz_dir)
        }, f, indent=2)
    
    logging.info(f"Comprehensive brain visualization completed: {brain_viz_dir}")
    return results

def _create_comprehensive_brain_summary(brain_viz_dir: Path, results: Dict):
    """Create an HTML summary of all brain visualizations."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Brain Visualization Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2E86C1; }}
            .section {{ margin: 20px 0; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Comprehensive Brain Visualization Summary</h1>
        
        <div class="section">
            <h2>Factor Maps</h2>
            <p>Number of factors mapped: {len(results.get('factor_maps', {}))}</p>
            <p>Location: <code>factor_maps/</code></p>
        </div>
        
        <div class="section">
            <h2>Subject Reconstructions</h2>
    """
    
    if 'subject_reconstructions' in results:
        stats = results['subject_reconstructions'].get('statistics', {})
        html_content += f"""
            <p>Subjects reconstructed: {stats.get('n_subjects_completed', 'N/A')}</p>
            <p>Total files created: {stats.get('total_files_created', 'N/A')}</p>
            <p>Processing time: {stats.get('processing_time_seconds', 0):.1f} seconds</p>
            <p>Location: <code>subject_reconstructions/</code></p>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Files Generated</h2>
            <ul>
                <li>Factor NIfTI files: <code>factor_maps/factor_*_*_loadings.nii</code></li>
                <li>Subject NIfTI files: <code>subject_reconstructions/subject_*_*.nii</code></li>
                <li>Overlay images: <code>subject_reconstructions/overlays/subject_*_overlay.png</code></li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    summary_file = brain_viz_dir / "brain_visualization_summary.html"
    with open(summary_file, 'w') as f:
        f.write(html_content)

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