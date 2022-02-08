## PREPARE Forward Model for Mixed Source Space

import mne
import numpy as np
# from nilearn import plotting
import os.path as op

# directories
trans_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/trans/" # enter your special trans file folder here
meg_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
# subjects
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_06":"NEB26","MEM_09":"PAG48"}
nonill_dict = {"MEM_08":"WKI71","MEM_10":"NAL22","MEM_11":"KIY23_fa"}


# load the fsaverage source spaces for computing and saving source morphs from subjects
fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(meg_dir))
fs_vol_src = mne.read_source_spaces("{}fsaverage_vol-src.fif".format(meg_dir))

for meg,mri in sub_dict.items():
    # read source spaces and BEM solution (conductor model)
    trans = "{dir}{meg}_{mri}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    src = mne.read_source_spaces("{dir}{meg}-oct6-src.fif".format(dir=meg_dir,meg=meg))
    vol_src = mne.read_source_spaces("{dir}{meg}_vol-src.fif".format(dir=meg_dir,meg=meg))
    bem = mne.read_bem_solution("{dir}{meg}-bem.fif".format(dir=meg_dir,meg=meg))
    # load and prepare the MEG data
    info = mne.io.read_info("{}{}-analysis-epo.fif".format(meg_dir,meg))
    # for surface source spaces
    # build forward model from MRI and BEM  - for each experimental block
    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=3.0, n_jobs=8)
    mne.write_forward_solution("{dir}{meg}-fwd.fif".format(dir=meg_dir,meg=meg),fwd,overwrite=True)
    # compute and save source morph to fsaverage for later group analyses
    morph = mne.compute_source_morph(fwd['src'],subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir,src_to=fs_src)  ## it's important to use fwd['src'] to account for discarded vertices
    morph.save("{}{}_fs_oct6-morph.h5".format(meg_dir,meg))
    del fwd,morph
    # now for volume source space
    # build forward model from MRI and BEM  - for each experimental block
    vol_fwd = mne.make_forward_solution(info, trans=trans, src=vol_src, bem=bem, meg=True, eeg=False, mindist=3.0, n_jobs=8)
    mne.write_forward_solution("{dir}{meg}_vol-fwd.fif".format(dir=meg_dir,meg=meg),vol_fwd,overwrite=True)
    # compute and save source morph to fsaverage for later group analyses
    vol_morph = mne.compute_source_morph(vol_fwd['src'],subject_from=mri,subject_to="fsaverage",subjects_dir=mri_dir,src_to=fs_vol_src)  ## it's important to use fwd['src'] to account for discarded vertices
    vol_morph.save("{}{}_fs_vol-morph.h5".format(meg_dir,meg))
    del vol_fwd,vol_morph
