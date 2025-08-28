import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def reconstruct_and_overlay_all(base_dir, fill_val=0, save_png=True, subject_range=None):
    """
    Reconstruct + overlay SN/Putamen/Lentiform for subjects.
    
    Parameters:
    - base_dir : folder containing your qMAP-PD_data subfolder
    - fill_val : value outside ROI (0 or np.nan); default = 0
    - save_png : True/False to save a screenshot per subject; default = True
    - subject_range : tuple (start, end) for subject range, default (1, 87)
    """
    
    # Set subject range
    if subject_range is None:
        subject_range = (1, 87)
    start_subj, end_subj = subject_range
    
    # Reference NIfTI file
    ref_nii = os.path.join(base_dir, 'qMAP-PD_data', 'mri_reference', 
                          'average_space-qmap-shoot-384_pdw-brain.nii')
    
    # Volume and position files
    vols = [
        os.path.join(base_dir, 'qMAP-PD_data', 'volume_matrices', 'volume_sn_voxels.tsv'),
        os.path.join(base_dir, 'qMAP-PD_data', 'volume_matrices', 'volume_putamen_voxels.tsv'),
        os.path.join(base_dir, 'qMAP-PD_data', 'volume_matrices', 'volume_lentiform_voxels.tsv')
    ]
    
    poss = [
        os.path.join(base_dir, 'qMAP-PD_data', 'position_lookup', 'position_sn_voxels.tsv'),
        os.path.join(base_dir, 'qMAP-PD_data', 'position_lookup', 'position_putamen_voxels.tsv'),
        os.path.join(base_dir, 'qMAP-PD_data', 'position_lookup', 'position_lentiform_voxels.tsv')
    ]
    
    labels = ['sn', 'putamen', 'lentiform']
    
    # Create output directories
    out_nifti_dir = os.path.join(os.getcwd(), 'recon_out')
    os.makedirs(out_nifti_dir, exist_ok=True)
    
    out_png_dir = os.path.join(os.getcwd(), 'overlay_png')
    if save_png:
        os.makedirs(out_png_dir, exist_ok=True)
    
    # Loop through subjects
    n_subjects = end_subj - start_subj + 1
    for s in range(start_subj, end_subj + 1):
        print(f'Subject {s}/{end_subj} ({s-start_subj+1}/{n_subjects})')
        
        # Reconstruct three ROIs
        out_files = []
        roi_images = []
        
        for k, label in enumerate(labels):
            out_file = os.path.join(out_nifti_dir, f'recon_{label}_subj{s:02d}.nii')
            out_files.append(out_file)
            
            # Reconstruct single ROI and get the image data for visualization
            img_data = reconstruct_roi_single(vols[k], poss[k], ref_nii, out_file, s, fill_val)
            roi_images.append((label, img_data))
        
        # Create overlay visualization if requested
        if save_png:
            create_overlay_plot(ref_nii, roi_images, 
                              os.path.join(out_png_dir, f'overlay_subj{s:02d}.png'))
    
    print(f'Done. NIfTIs in: {out_nifti_dir}')
    if save_png:
        print(f'Overlay PNGs in: {out_png_dir}')


def reconstruct_roi_single(vol_tsv, pos_tsv, ref_nii, out_nii, row_idx, fill_val=0):
    """
    Rebuild a 3D NIfTI for one subject from TSV volume + position lookup.
    
    Returns:
    - img_data: The reconstructed 3D image data for visualization
    """
    
    # Load matrices (assuming no headers based on MATLAB readmatrix usage)
    V = pd.read_csv(vol_tsv, sep='\t', header=None).values  # [Nsubj × Nvox]
    pos = pd.read_csv(pos_tsv, sep='\t', header=None).values.flatten()  # [Nvox × 1]
    
    # Validate dimensions
    assert V.shape[1] == len(pos), 'Columns in volume must equal rows in position file.'
    assert 1 <= row_idx <= V.shape[0], 'row_idx out of range.'
    
    # Load reference NIfTI
    ref_img = nib.load(ref_nii)
    ref_data = ref_img.get_fdata()
    img_shape = ref_data.shape
    
    # Initialize output image
    if np.isnan(fill_val):
        img = np.full(img_shape, np.nan, dtype=np.float32)
    else:
        img = np.full(img_shape, fill_val, dtype=np.float32)
    
    # Bounds check for 1-based linear indices (convert to 0-based for Python)
    flat_n = np.prod(img_shape)
    pos_0based = pos - 1  # Convert from MATLAB 1-based to Python 0-based indexing
    
    if np.any((pos_0based < 0) | (pos_0based >= flat_n)):
        raise ValueError('Position indices fall outside reference image size.')
    
    # Insert voxel values for this subject (row_idx is 1-based, convert to 0-based)
    vals = V[row_idx - 1, :].astype(np.float32)
    
    # Convert linear indices to subscripts and assign values
    img_flat = img.flatten()
    img_flat[pos_0based.astype(int)] = vals
    img = img_flat.reshape(img_shape)
    
    # Create output NIfTI
    out_img = nib.Nifti1Image(img, ref_img.affine, ref_img.header)
    out_img.header.set_data_dtype(np.float32)
    
    # Save NIfTI
    nib.save(out_img, out_nii)
    
    return img


def create_overlay_plot(ref_nii, roi_images, out_png_path):
    """
    Create a simple overlay plot showing the reference image with ROI overlays.
    This is a basic replacement for SPM's spm_check_registration visualization.
    """
    
    # Load reference image
    ref_img = nib.load(ref_nii)
    ref_data = ref_img.get_fdata()
    
    # Get middle slices for visualization