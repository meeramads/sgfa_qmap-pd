ppt = spm_load('/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/participant.tsv');
tiv = spm_load('/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/tiv_spm.csv');
mbrs = spm_load('/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/phenotype/ses-01/motor_mbrs.tsv');
updrs3 = spm_load('/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/phenotype/ses-01/motor_updrs_03.tsv');
mm = spm_load('/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/phenotype/ses-01/motor_mm.tsv');

sjac=cellstr(spm_select('FPListRec','/Volumes/lambert_group/01_DATA/01_QMAPLABDATA/02_qmap-pd-GROUP/03_data/derivatives/01-warpdata/anat/fgcs_jacobian_s2','^s.*.nii'));

mask='/Volumes/lambert_group/01_DATA/00_ANALYSIS/01_MSC_STUDENTS/2025_MADHAVAN_GFA/average_space-qmap-384_roi-basal-ganglia.nii';

%% Okay, let's start by finding the mask data locations

loc = {'sn'
    'putamen'
    'lentiform'};

side = {'left'
    'right'};

root=[];
count = 0;
N=nifti(mask);
for i = 1:3
    for ii = 1:2
        count = count+1;
        root.(loc{i}).(side{ii}) = find(N.dat(:,:,:)==count);
    end
end

%% Okay lets's run through PD baseline and build a matrix per structure and 
% variables

count = 0;

sid=ppt.sid(ppt.primary_diagnosis==3);
age=ppt.age(ppt.primary_diagnosis==3);
sex=ppt.sex(ppt.primary_diagnosis==3);

data.sid=[];
data.age=[];
data.sex=[];
data.tiv=[];

sn=[];
putamen=[];
lentiform=[];

for i = 1:numel(sid)
    i
    ses=cell2mat(mbrs.session(mbrs.sid==sid(i)));
    ses=ses(1,:);
    if ~contains(ses,'NaT')
    sestag=fullfile([ses(1:4),ses(6:7),ses(9:10)]);
    proc = false;
    ptiv=[];
    for ii = 1:numel(tiv.File)
        if contains(tiv.File{ii},num2str(sid(i))) && contains(tiv.File{ii},sestag)
            proc=true;
            ptiv = tiv.Volume1(ii)+tiv.Volume2(ii)+tiv.Volume3(ii);
        end
    end
    
    
    if proc


        %% Let's make sure we can get the data
        Nproc=false;
        Z=[];

            for ii = 1:numel(sjac)
        if contains(sjac{ii},num2str(sid(i))) && contains(sjac{ii},sestag)
            Nproc=true;
            N=nifti(sjac{ii});
            Z = N.dat(:,:,:);
        end
    end
    
    if Nproc
                count = count+1;
        %% First add data to matrix
        sn(count,:)=[Z(root.sn.left);Z(root.sn.right)];
         putamen(count,:)=[Z(root.putamen.left);Z(root.putamen.right)];
          lentiform(count,:)=[Z(root.lentiform.left);Z(root.lentiform.right)];
          upos=find(updrs3.sid==(sid(i)));  upos=upos(end);
            mpos=find(mbrs.sid==(sid(i)));  mpos=mpos(end); 
            mmpos = find(mm.sid==(sid(i)));  mmpos=mmpos(end);
          data.sid(:,count)=sid(i);
          data.age(:,count)=age(i);
          data.sex(:,count)=sex(i);
          data.tiv(:,count) = ptiv;
          data.rigidity_left(:,count) = updrs3.updrs3_03_lle(upos)+updrs3.updrs3_03_lue(upos);
          data.rigidity_right(:,count)= updrs3.updrs3_03_rle(upos)+updrs3.updrs3_03_rue(upos);
          data.tremor_rest_left(:,count) = updrs3.updrs3_17_lue(upos) + updrs3.updrs3_17_lle(upos) + updrs3.updrs3_18(upos); %Present and persistence 
          data.tremor_rest_right(:,count) = updrs3.updrs3_17_rue(upos) + updrs3.updrs3_17_rle(upos) + updrs3.updrs3_18(upos); %Present and persistence
          data.tremor_postural_left(:,count) = updrs3.updrs3_15_l(upos); %Present and persistence 
          data.tremor_postural_right(:,count) = updrs3.updrs3_15_r(upos);
          data.bradykinesia_speed_left(:,count) = mbrs.mbrs_l_speed(mpos);
          data.bradykinesia_speed_right(:,count) = mbrs.mbrs_r_speed(mpos);
         data.bradykinesia_amplitude_left(:,count) = mbrs.mbrs_l_amp(mpos);
          data.bradykinesia_amplitude_right(:,count) = mbrs.mbrs_r_amp(mpos);
          data.bradykinesia_rhythm_left(:,count) = mbrs.mbrs_l_rhythm(mpos);
          data.bradykinesia_rhythm_right(:,count) = mbrs.mbrs_r_rhythm(mpos);
           data.mirrormovement_left(:,count) = mm.mm_total_left(mmpos);
          data.mirrormovement_right(:,count) = mm.mm_total_left(mmpos);

    end
    end
    end
end

fx=fields(data)

for i = 1:numel(fx)
    data.(fx{i}) = data.(fx{i})(:);
end




