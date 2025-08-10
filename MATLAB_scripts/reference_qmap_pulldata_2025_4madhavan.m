base = '/Users/meera/Desktop/03_MsC_MADHAVAN_MOTOR-GFA copy/04_data';

data = readtable(fullfile(base, 'data_clinical', 'pd_motor_gfa_data.tsv'), 'FileType', 'text', 'Delimiter', '\t');


% Load precomputed voxel-wise Jacobian volumes
sn        = readmatrix(fullfile(base, 'volume_matrices', 'volume_sn_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');
putamen   = readmatrix(fullfile(base, 'volume_matrices', 'volume_putamen_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');
lentiform = readmatrix(fullfile(base, 'volume_matrices', 'volume_lentiform_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');


% Load position lookup indices (for later visualization)
pos_sn        = readmatrix(fullfile(base, 'position_lookup', 'position_sn_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');
pos_putamen   = readmatrix(fullfile(base, 'position_lookup', 'position_putamen_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');
pos_lentiform = readmatrix(fullfile(base, 'position_lookup', 'position_lentiform_voxels.tsv'), 'FileType', 'text', 'Delimiter', '\t');

% Load ROI mask and reference image if visual reconstruction is needed
ref_mri = fullfile(base, 'mri_reference', 'average_space-qmap-shoot-384_pdw-brain.nii');
roi_mask = fullfile(base, 'mri_roi', 'average_space-qmap-384_roi-basal-ganglia.nii');


% Print information (for checks)
disp('Number of subjects: ');
disp(height(data));
disp('Size of SN voxel matrix: ');
disp(size(sn));


% Save cleaned data 
writetable(data, fullfile(base, 'data_clinical', 'pd_motor_gfa_data_cleaned.tsv'), 'Delimiter', '\t', 'FileType', 'text');
