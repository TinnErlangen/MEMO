# try source T-F difference with Hilbert beamformer #

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

for meg, mri in sub_dict.items():

    epo = mne.read_epochs("{}{}-analysis-epo.fif".format(meg_dir,meg))
    fwd = mne.read_forward_solution("{}{}_vol-fwd.fif".format(meg_dir,meg))
    epo.filter(fmin, fmax, n_jobs='cuda')
    data_cov = mne.compute_covariance(epo, tmin=0, tmax=None, rank=None, n_jobs=8)
    noise_cov = mne.compute_covariance(epo, tmin=None, tmax=0, rank=None, n_jobs=8)
    filters = make_lcmv(epo.info, fwd, data_cov=data_cov,
                        noise_cov=noise_cov, pick_ori='max-power',
                        weight_norm='unit-noise-gain', reg=0.05, rank=None)
    filters.save('{}{}_alpha_vol-lcmv.h5'.format(meg_dir,meg),overwrite=True)
    epo.apply_hilbert(n_jobs=1, envelope=False)
    stcs_cont = apply_lcmv_epochs(epo['cont'], filters, max_ori_out='signed')
    stcs_break = apply_lcmv_epochs(epo['break'], filters, max_ori_out='signed')
    del epo, noise_cov, data_cov
    # envelope data (absolute) for both conditions
    for stc in stcs_cont:
        stc.data[:, :] = np.abs(stc.data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stc.data = np.array(stc.data, 'float64')
    for stc in stcs_break:
        stc.data[:, :] = np.abs(stc.data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stc.data = np.array(stc.data, 'float64')
    # average across epochs per condition 
    mean_cont = np.mean([stc.data for stc in stcs_cont], axis=0)
    mean_break = np.mean([stc.data for stc in stcs_break], axis=0)
    mean_diff = mean_cont - mean_break
    # now make stcs and morph to fsaverage for GA collection
    morph = mne.read_source_morph("{}{}_fs_vol-morph.h5".format(meg_dir,meg))
    stc_c = stcs_cont[0].copy()
    stc_c.data = mean_cont
    stc_fs_c = morph.apply(stc_c)
    stc_fs_c.save("{}{}alpha_source_env_cont".format(meg_dir,meg))
    GA_cont.append(stc_fs_c)
    stc_b = stcs_cont[0].copy()
    stc_b.data = mean_break
    stc_fs_b = morph.apply(stc_b)
    stc_fs_b.save("{}{}alpha_source_env_break".format(meg_dir,meg))
    GA_break.append(stc_fs_b)
    stc_d = stcs_cont[0].copy()
    stc_d.data = mean_diff
    stc_fs_d = morph.apply(stc_d)
    stc_fs_d.save("{}{}alpha_source_env_diff".format(meg_dir,meg))
    GA_diff.append(stc_fs_d)
    del morph, stc_c, stc_fs_c, stc_b, stc_fs_b, stc_d, stc_fs_d
