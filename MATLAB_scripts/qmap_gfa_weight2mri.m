function qmap_gfa_weight2mri(fname,weights,output,lookup,refmri)
%% function qmap_gfa_weight2mri(weights,matrixpos,lookup,referencemri)
% Function to take the following inputs:
% weights = calculated weight values from gfa or similar (1xn vector)
% from gfa or similar (1xn vector), same order as weights.
% fname = Filename for image. Try to avoid whitespace, use - or _ instead
% lookup = corresponding position tsv files to map matrix pos to image space
% refmri = mri to base the image on

 % Load voxel indices
    if ischar(lookup) || isstring(lookup)
        voxel_indices = readmatrix(lookup, 'FileType', 'text');
    else
        voxel_indices = lookup;
    end

    % Validate input
    if numel(weights) ~= numel(voxel_indices)
        error('Number of weights (%d) must match number of voxel indices (%d)', ...
               numel(weights), numel(voxel_indices));
    end

    % Load reference MRI
    N = nifti(refmri);
    Z = zeros(N.dat.dim);

    % Assign weights to voxel positions
    Z(voxel_indices) = weights;

    % Prepare output file path
    if ~exist(output, 'dir')
        mkdir(output);
    end
    newfile = fullfile(output, [fname, '.nii']);

    % Save new NIfTI file
    N.dat.fname = newfile;
    N.dat.dtype = 'FLOAT32';
    N.dat(:,:,:) = Z;
    create(N);

    % Report
    fprintf('Saved NIfTI image to %s\n', newfile);
end

