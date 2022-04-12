# do spatiotemporal clustering on full source TFRs (fsaverage morphed)
import mne
import numpy as np
from mne.beamformer import tf_dics
from mne.viz import plot_source_spectrogram
import time
import datetime

# set directories
meg_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
proc_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
plot_dir = "/home/cora/hdd/MEG/MEMO_analyses/TF_dics/plots/"
save_dir = "/home/cora/hdd/MEG/MEMO_analyses/TF_dics/"
# subjects
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_06":"NEB26","MEM_09":"PAG48"}
nonill_dict = {"MEM_08":"WKI71","MEM_10":"NAL22","MEM_11":"KIY23_fa"}


# set parameters for TF-DICS beamformer
# frequency & time parameters
frequencies = [[7.,8.,9.,10.,11.]]
freq_bins = [(i[0],i[-1]) for i in frequencies]  # needed for plotting !!
cwt_n_cycles = [7.]
win_lengths = [1]
tmin = -0.1
tmax = 5
tstep = 0.05
tmin_plot = 0
tmax_plot = 5
n_jobs = 4


# GA collect bins
# GA_stcs_illu = []
# GA_stcs_break = []
GA_stcs_cont = []

# fsaverage SRC and adjacency
fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(meg_dir))
adjacency = mne.spatial_src_adjacency(fs_src)

# SUBJECT DATA COLLECTION and Morphing

for meg,mri in sub_dict.items():

    # load fwd['src'] and morph
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(meg_dir,meg))
    morph = mne.read_source_morph("{}{}_fs_oct6-morph.h5".format(meg_dir,meg))
    # load data, morph and collect them
    # stc_illu = mne.read_source_estimate("{}{}_TF_dics_illu_{}-{}-stc.h5".format(save_dir,meg,freq_bins[0][0],freq_bins[0][-1]))
    # stc_break = mne.read_source_estimate("{}{}_TF_dics_break_{}-{}-stc.h5".format(save_dir,meg,freq_bins[0][0],freq_bins[0][-1]))
    stc_cont = mne.read_source_estimate("{}{}_TF_dics_diff_I-B_{}-{}-stc.h5".format(save_dir,meg,freq_bins[0][0],freq_bins[0][-1]))
    stc_fs_cont = morph.apply(stc_cont)
    GA_stcs_cont.append(stc_fs_cont)

# GA calc and plotting
GA_plot_stc = GA_stcs_cont[0].copy()
GA_cont_meandat = np.mean([cont.data for cont in GA_stcs_cont], axis=0)
GA_plot_stc.data = GA_cont_meandat
# # plot
# brain = GA_plot_stc.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',clim={'kind':'value','pos_lims':(1e-27,2e-27,4e-27)},
#                          time_viewer=True,src=fs_src,show_traces=True,title="Source Alpha Difference")
# brain.add_annotation('aparc', borders=1, alpha=0.9)
# GA_plot_stc.save("{}GA_TF_dics_diff_I-B_{}-{}-stc.h5".format(save_dir,freq_bins[0][0],freq_bins[0][-1]))

# GA spatiotemporal clustering
# parameters
threshold = 2.781
# data array (obs x times x locs)
X = np.array([stc.data.T for stc in GA_stcs_cont])
X = X * 1e27    # otherwise variance too close to 0, get error
# cluster stats
t_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(X, n_permutations=1024, threshold = threshold, tail=1, adjacency=adjacency, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
# get significant clusters
good_cluster_inds = np.where(cluster_pv < 0.05)[0]
if len(good_cluster_inds):
    # then loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        GA_clu = GA_plot_stc.copy()
        time_inds, ch_inds = clusters[clu_idx]
        time_inds = np.unique(time_inds)
        print("Cluster {} extends over time indices {}".format(i_clu+1, time_inds))
        for t, v in zip(clusters[clu_idx][0], clusters[clu_idx][1]):
            GA_clu.data[v, t] = 1
        brain = GA_clu.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',clim={'kind':'value','pos_lims':(0.5,0.6,1)},
                            time_viewer=True,src=fs_src,show_traces=False,title="Source Alpha Difference")
else:
    print("No sign. clusters found")
