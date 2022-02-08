## PREPARE BEM-Model and Source Space(s) - Surface, Mixed, and/or Volume

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
sub_dict = {"MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_06":"NEB26","MEM_09":"PAG48"}

# # prep fsaverage
#
# # build BEM model for fsaverage (as boundary for source space creation) --- only needed for volume or mixed source spaces
# bem_model = mne.make_bem_model("fsaverage", subjects_dir=mri_dir, ico=4, conductivity=[0.3])
# bem = mne.make_bem_solution(bem_model)
# mne.write_bem_solution("{dir}fsaverage-bem.fif".format(dir=meg_dir),bem)
# mne.viz.plot_bem(subject="fsaverage", subjects_dir=mri_dir, brain_surfaces='white', orientation='coronal')
#
# # build fs_average 'oct6' surface source space & save (to use as morph target later)
# fs_src = mne.setup_source_space("fsaverage", spacing='oct6', surface="white", subjects_dir=mri_dir, n_jobs=8)
# print(fs_src)
# # print out the number of spaces and points
# n = sum(fs_src[i]['nuse'] for i in range(len(fs_src)))
# print('the fs_src space contains %d spaces and %d points' % (len(fs_src), n))
# fs_src.plot(subjects_dir=mri_dir)
# # save the surface source space
# fs_src.save("{}fsaverage_oct6-src.fif".format(meg_dir), overwrite=True)
# del fs_src
# # build fs_average volume source space & save (to use as morph target later)
# surface = op.join(mri_dir, 'fsaverage', 'bem', 'inner_skull.surf')
# fs_vol_src = mne.setup_volume_source_space(subject="fsaverage", pos=5.0, mri=None, bem=None, surface=surface,
#                                         mindist=5.0, exclude=0.0, subjects_dir=mri_dir)
# print(fs_vol_src)
# mne.viz.plot_bem(subject="fsaverage", subjects_dir=mri_dir,
#                  brain_surfaces='white', src=fs_vol_src, orientation='coronal')
# # save the volume source space
# fs_vol_src.save("{}fsaverage_vol-src.fif".format(meg_dir), overwrite=True)
# del fs_vol_src


# prep subjects
for meg,mri in sub_dict.items():

    # build BEM model from MRI, save and plot, along with sensor alignment
    bem_model = mne.make_bem_model(mri, subjects_dir=mri_dir, ico=5, conductivity=[0.3])
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("{dir}{meg}-bem.fif".format(dir=meg_dir,meg=meg),bem)
    mne.viz.plot_bem(subject=mri, subjects_dir=mri_dir, brain_surfaces='white', orientation='coronal')
    # load trans-file and plot coregistration alignment
    trans = "{dir}{meg}_{mri}-trans.fif".format(dir=trans_dir,mri=mri,meg=meg)
    info = mne.io.read_info("{}{}-analysis-epo.fif".format(meg_dir,meg))       # use your -epo.fif here
    mne.viz.plot_alignment(info, trans, subject=mri, dig='fiducials', meg=['helmet', 'sensors'], eeg=False,
                           subjects_dir=mri_dir, surfaces='head-dense', bem=bem)

    # build the surface source space for the subjects, with 'oct6' spacing
    src = mne.setup_source_space(mri, spacing='oct6', surface="white", subjects_dir=mri_dir, n_jobs=8)  ## uses 'oct6' as default, i.e. 4.9mm spacing appr.
    print(src)
    # print number of spaces and points, save
    n = sum(src[i]['nuse'] for i in range(len(src)))
    print('the src space contains %d spaces and %d points' % (len(src), n))
    # save the mixed source space
    src.save("{}{}-oct6-src.fif".format(meg_dir,meg), overwrite=True)
    # plot the source space with points
    src.plot(subjects_dir=mri_dir)
    mne.viz.plot_alignment(info, trans, subject=mri, dig='fiducials', meg=['helmet', 'sensors'], eeg=False,
                           subjects_dir=mri_dir, surfaces='head-dense', bem=bem, src=src)
    del src

    # build the volume source space for the subjects
    vol_src = mne.setup_volume_source_space(subject=mri, pos=5.0, mri=None, bem=bem, surface=None,
                                            mindist=5.0, exclude=0.0, subjects_dir=mri_dir)
    print(vol_src)
    mne.viz.plot_bem(subject=mri, subjects_dir=mri_dir,
                     brain_surfaces='white', src=vol_src, orientation='coronal')
    # save the volume source space
    vol_src.save("{}{}_vol-src.fif".format(meg_dir,meg), overwrite=True)
    del vol_src
