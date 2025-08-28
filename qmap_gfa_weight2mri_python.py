import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Union


def qmap_gfa_weight2mri(fname: str, weights: np.ndarray, output: str, 
                       lookup: Union[str, np.ndarray], refmri: str) -> None:
    """
    Function to take calculated weight values from GFA or similar and map them to MRI space.
    
    Parameters:
    -----------
    fname : str
        Filename for image. Try to avoid whitespace, use - or _ instead
    weights : numpy.ndarray
        Calculated weight values from GFA or similar (1D array)
    output : str
        Output directory path
    lookup : str or numpy.ndarray
        Either path to corresponding position TSV file to map matrix pos to image space,
        or the voxel indices array directly
    refmri : str
        Path to reference MRI to base the image on
    """
    
    # Load voxel indices
    if isinstance(lookup, (str, Path)):
        # Load from file
        voxel_indices = pd.read_csv(lookup, sep='\t', header=None).values.flatten()
    else:
        # Use provided array
        voxel_indices = np.array(lookup).flatten()
    
    # Ensure weights is a numpy array
    weights = np.array(weights).flatten()
    
    # Validate input
    if len(weights) != len(voxel_indices):
        raise ValueError(f'Number of weights ({len(weights)}) must match '
                        f'number of voxel indices ({len(voxel_indices)})')
    
    # Load reference MRI
    ref_img = nib.load(refmri)
    ref_data = ref_img.get_fdata()
    
    # Initialize output array with zeros
    Z = np.zeros(ref_data.shape, dtype=np.float32)
    
    # Convert 1-based MATLAB indices to 0-based Python indices
    voxel_indices_python = voxel_indices.astype(int) - 1
    
    # Bounds check
    flat_n = np.prod(ref_data.shape)
    if np.any(voxel_indices_python < 0) or np.any(voxel_indices_python >= flat_n):
        raise ValueError('Voxel indices fall outside reference image dimensions')
    
    # Convert linear indices to 3D coordinates
    coords = np.unravel_index(voxel_indices_python, ref_data.shape)
    
    # Assign weights to voxel positions
    Z[coords] = weights.astype(np.float32)
    
    # Prepare output file path
    os.makedirs(output, exist_ok=True)
    newfile = os.path.join(output, f'{fname}.nii')
    
    # Create and save new NIfTI file
    # Copy header and affine from reference
    new_img = nib.Nifti1Image(Z, ref_img.affine, ref_img.header.copy())
    
    # Update data type in header
    new_img.header.set_data_dtype(np.float32)
    
    # Save the file
    nib.save(new_img, newfile)
    
    # Report
    print(f'Saved NIfTI image to {newfile}')


# Alternative version that handles NaN values more explicitly
def qmap_gfa_weight2mri_with_nan(fname: str, weights: np.ndarray, output: str, 
                                lookup: Union[str, np.ndarray], refmri: str,
                                fill_value: Union[float, int] = 0) -> None:
    """
    Enhanced version that explicitly handles NaN values and provides more control.
    
    Parameters:
    -----------
    fname : str
        Filename for image. Try to avoid whitespace, use - or _ instead
    weights : numpy.ndarray
        Calculated weight values from GFA or similar (1D array)
    output : str
        Output directory path
    lookup : str or numpy.ndarray
        Either path to corresponding position TSV file to map matrix pos to image space,
        or the voxel indices array directly
    refmri : str
        Path to reference MRI to base the image on
    fill_value : float or int, default=0
        Value to use for voxels outside the ROI
    """
    
    # Load voxel indices
    if isinstance(lookup, (str, Path)):
        # Try different separators and handle potential headers
        try:
            voxel_indices = pd.read_csv(lookup, sep='\t', header=None).values.flatten()
        except:
            # Try comma separator if tab fails
            voxel_indices = pd.read_csv(lookup, sep=',', header=None).values.flatten()
    else:
        voxel_indices = np.array(lookup).flatten()
    
    # Ensure weights is a numpy array
    weights = np.array(weights).flatten()
    
    # Remove any NaN indices (if present)
    valid_mask = ~(np.isnan(voxel_indices) | np.isnan(weights))
    voxel_indices = voxel_indices[valid_mask]
    weights = weights[valid_mask]
    
    # Validate input
    if len(weights) != len(voxel_indices):
        raise ValueError(f'Number of weights ({len(weights)}) must match '
                        f'number of voxel indices ({len(voxel_indices)})')
    
    # Load reference MRI
    ref_img = nib.load(refmri)
    ref_data = ref_img.get_fdata()
    
    # Initialize output array with fill value
    if np.isnan(fill_value):
        Z = np.full(ref_data.shape, np.nan, dtype=np.float32)
    else:
        Z = np.full(ref_data.shape, fill_value, dtype=np.float32)
    
    # Convert 1-based MATLAB indices to 0-based Python indices
    voxel_indices_python = voxel_indices.astype(int) - 1
    
    # Bounds check
    flat_n = np.prod(ref_data.shape)
    valid_indices = (voxel_indices_python >= 0) & (voxel_indices_python < flat_n)
    
    if not np.all(valid_indices):
        print(f'Warning: {np.sum(~valid_indices)} voxel indices fall outside image bounds and will be ignored')
        voxel_indices_python = voxel_indices_python[valid_indices]
        weights = weights[valid_indices]
    
    # Convert linear indices to 3D coordinates
    coords = np.unravel_index(voxel_indices_python, ref_data.shape)
    
    # Assign weights to voxel positions
    Z[coords] = weights.astype(np.float32)
    
    # Prepare output file path
    os.makedirs(output, exist_ok=True)
    newfile = os.path.join(output, f'{fname}.nii')
    
    # Create and save new NIfTI file
    new_img = nib.Nifti1Image(Z, ref_img.affine, ref_img.header.copy())
    new_img.header.set_data_dtype(np.float32)
    
    # Save the file
    nib.save(new_img, newfile)
    
    # Report
    print(f'Saved NIfTI image to {newfile}')
    print(f'Image shape: {Z.shape}')
    print(f'Non-zero/non-NaN voxels: {np.sum(~np.isnan(Z) & (Z != fill_value))}')
    if not np.isnan(fill_value):
        print(f'Value range: {np.nanmin(weights):.4f} to {np.nanmax(weights):.4f}')


# Example usage and testing function
def example_usage():
    """
    Example of how to use the function
    """
    # Example data
    weights = np.random.rand(1000) * 100  # 1000 random weights
    voxel_indices = np.random.randint(1, 100000, 1000)  # 1000 random voxel positions (1-based)
    
    # Example paths (adjust these to your actual paths)
    output_dir = './output_images'
    reference_mri = './reference_brain.nii'  # Path to your reference MRI
    
    # Call the function
    qmap_gfa_weight2mri('example_weights', weights, output_dir, voxel_indices, reference_mri)


if __name__ == "__main__":
    example_usage()