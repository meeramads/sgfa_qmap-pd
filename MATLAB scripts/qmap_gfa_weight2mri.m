function qmap_gfa_weight2mri(fname,weights,matrixpos,output,lookup,refmri)
%% function qmap_gfa_weight2mri(weights,matrixpos,lookup,referencemri)
% Function to take the following inputs:
% weights = calculated weight values from gfa or similar (1xn vector)
% matrixpos = column (i.e. voxel) position in original matrix weight values
% from gfa or similar (1xn vector), same order as weights.
% fname = Filename for image. Try to avoid whitespace, use - or _ instead
% lookup = corresponding position tsv files to map matrix pos to image space
% refmri = mri to base the image on

if nargin < 6
    refmri = spm_select(1,'nii','Please select reference MRI to base image on');
    if nargin < 5
        lookup = spm_select(1,'any','Please select corresponding "position" lookup file');
        if nargin < 4
            output = spm_select(1,'dir','output directory');
            if nargin < 3
                matrixpos = [];
                if nargin < 2
                    error('Insufficient inputs');
                end
            end
        end
    end
end

if isempty(matrixpos),matrixpos = 1:1:numel(weights);end %Assuming if not specified, same number of weights as original column vectors e.g., zero or nan entries for non-relevant

vox = spm_load(lookup);
vox = vox(matrixpos);

newfile = fullfile(output,[fname,'.nii']);

N=nifti(refmri);

Z=zeros(N.dat.dim);
Z(vox) = weights;

N.dat.fname=newfile;
N.dat.dtype = 'FLOAT32';
N.dat(:,:,:)=Z;
create(N);
end


