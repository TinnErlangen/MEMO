# Hilbert beamformer GA group analysis #

import numpy as np
import mne
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
import warnings

# directories
meg_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
# subjects
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_06":"NEB26","MEM_09":"PAG48"}
rank_problem = {"MEM_15":"KHA22_fa","MEM_09":"PAG48"}   # for these, noise_cov was rank deficient
nonill_dict = {"MEM_08":"WKI71","MEM_10":"NAL22","MEM_11":"KIY23_fa"}

# conditions x time windows for source loc
conds = ('cont', 'break')

# the frequencies passed as lists (for filter calculation)
freqs = [7, 8, 9, 10, 11]
fmin = 7
fmax = 11

# containers for GAs
GA_cont = []
GA_break = []
GA_diff = []

# read in & collect data
for meg, mri in sub_dict.items():
    stc_fs_c = mne.read_source_estimate("{}{}alpha_source_env_cont-vl.stc".format(meg_dir,meg))
    GA_cont.append(stc_fs_c)
    stc_fs_b = mne.read_source_estimate("{}{}alpha_source_env_break-vl.stc".format(meg_dir,meg))
    GA_break.append(stc_fs_b)
    stc_fs_d = mne.read_source_estimate("{}{}alpha_source_env_diff-vl.stc".format(meg_dir,meg))
    GA_diff.append(stc_fs_d)

# calc grand average & make stc
GA_cont_mean = np.mean([stc.data for stc in GA_cont],axis=0)
GA_break_mean = np.mean([stc.data for stc in GA_break],axis=0)
GA_diff_mean = np.mean([stc.data for stc in GA_diff],axis=0)
GA_cont_stc = GA_cont[0].copy()
GA_cont_stc.data = GA_cont_mean
GA_break_stc = GA_break[0].copy()
GA_break_stc.data = GA_break_mean
GA_diff_stc = GA_diff[0].copy()
GA_diff_stc.data = GA_diff_mean

# get fsaverage source space for plotting the GAs & plot
fs_vol_src = mne.read_source_spaces("{}fsaverage_vol-src.fif".format(meg_dir))
GA_diff_stc.plot(fs_vol_src, subjects_dir=mri_dir)

pos, lat = GA_diff_stc.get_peak(tmin=None, tmax=None, mode='pos')
GA_diff_stc.plot(fs_vol_src, subjects_dir=mri_dir, initial_time=1.82)
