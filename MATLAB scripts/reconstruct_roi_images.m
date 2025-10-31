function reconstruct_and_overlay_all(base_dir, fillVal, save_png)
% Reconstruct + overlay SN/Putamen/Lentiform for ALL subjects (1..87).
% - base_dir : folder containing your dataset subfolders
% - fillVal  : value outside ROI (0 or NaN); default = 0
% - save_png : true/false to save a screenshot per subject; default = true

    if nargin < 2 || isempty(fillVal), fillVal = 0; end
    if nargin < 3 || isempty(save_png), save_png = true; end

    % Ensure SPM is on the path before calling this script, e.g.:
    % addpath('/path/to/spm12'); spm('Defaults','fMRI'); spm_jobman('initcfg');

    ref_nii = fullfile(base_dir, 'mri_reference', 'average_space-qmap-shoot-384_pdw-brain.nii');

    vols = { ...
        fullfile(base_dir, 'volume_matrices',  'volume_sn_voxels.tsv'), ...
        fullfile(base_dir, 'volume_matrices',  'volume_putamen_voxels.tsv'), ...
        fullfile(base_dir, 'volume_matrices',  'volume_lentiform_voxels.tsv')};

    poss = { ...
        fullfile(base_dir, 'position_lookup',  'position_sn_voxels.tsv'), ...
        fullfile(base_dir, 'position_lookup',  'position_putamen_voxels.tsv'), ...
        fullfile(base_dir, 'position_lookup',  'position_lentiform_voxels.tsv')};

    labels = {'sn','putamen','lentiform'};

    % Output folders
    out_nifti_dir = fullfile(pwd, 'recon_out');
    if ~exist(out_nifti_dir, 'dir'), mkdir(out_nifti_dir); end
    out_png_dir = fullfile(pwd, 'overlay_png');
    if save_png && ~exist(out_png_dir, 'dir'), mkdir(out_png_dir); end

    % Loop through subjects 1..87
    Nsubjects = 87;
    for s = 1:Nsubjects
        fprintf('Subject %d/%d\n', s, Nsubjects);

        % Reconstruct three ROIs
        out_files = cell(1, numel(labels));
        for k = 1:numel(labels)
            out_files{k} = fullfile(out_nifti_dir, sprintf('recon_%s_subj%02d.nii', labels{k}, s));
            reconstruct_roi_single(vols{k}, poss{k}, ref_nii, out_files{k}, s, fillVal);
        end

        % Overlay in SPM
        spm_check_registration(char([string(ref_nii); string(out_files)']));

        % Optional: save a screenshot of the SPM graphics window
        if save_png
            F = spm_figure('GetWin','Graphics');
            drawnow;
            saveas(F, fullfile(out_png_dir, sprintf('overlay_subj%02d.png', s)));
        end

        % Close the figure to keep things tidy
        close(spm_figure('GetWin','Graphics'));
    end

    fprintf('Done. NIfTIs in: %s\n', out_nifti_dir);
    if save_png, fprintf('Overlay PNGs in: %s\n', out_png_dir); end
end


function reconstruct_roi_single(vol_tsv, pos_tsv, ref_nii, out_nii, row_idx, fillVal)
% Rebuild a 3D NIfTI for one subject from TSV volume + position lookup.
    if nargin < 6, fillVal = 0; end

    % Load matrices (no headers)
    V   = readmatrix(vol_tsv, 'FileType','text','Delimiter','\t');  % [Nsubj × Nvox]
    pos = readmatrix(pos_tsv,  'FileType','text','Delimiter','\t');  % [Nvox × 1]
    pos = pos(:);

    assert(size(V,2) == numel(pos), 'Columns in volume must equal rows in position file.');
    assert(row_idx>=1 && row_idx<=size(V,1), 'row_idx out of range.');

    % Reference geometry (SPM)
    Vref = spm_vol(ref_nii);
    img  = ones(Vref.dim, 'single') .* single(fillVal);

    % Bounds check for 1-based linear indices
    flatN = prod(Vref.dim);
    if any(pos < 1 | pos > flatN)
        error('Position indices fall outside reference image size.');
    end

    % Insert voxel values for this subject
    vals = single(V(row_idx, :));
    img(pos) = vals;

    % Save NIfTIs
    Vout = Vref;
    Vout.fname   = out_nii;
    Vout.descrip = sprintf('Reconstructed from %s (row %d)', vol_tsv, row_idx);
    spm_write_vol(Vout, img);
end

base = '/Users/meera/Documents/MATLAB/MSc Project';
reconstruct_and_overlay_all(base, NaN, true);