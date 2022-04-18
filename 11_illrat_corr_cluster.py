import numpy as np
import mne
import pandas as pd
import random
from scipy import stats
# from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

from sklearn.linear_model import LinearRegression

from mne.stats.cluster_level import _setup_adjacency, _find_clusters, \
    _reshape_clusters

# mne.viz.set_3d_backend('pyvista')

# setup files and folders, subject lists
proc_dir = "D:/MEMO_analyses/MEMO_preproc/"
mri_dir = "D:/freesurfer/subjects/"
tf_dir = "D:/MEMO_analyses/TF_dics/"
save_dir = "D:/MEMO_analyses/TF_dics/beh_corr/"

# subjects list
subjs = ['MEM_14', 'MEM_13', 'MEM_03', 'MEM_05', 'MEM_04', 'MEM_02', 'MEM_15', 'MEM_01', 'MEM_12', 'MEM_07', 'MEM_09', 'MEM_06']    # !! was _06, then _09 before, because _09 data missed, later added
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_09":"PAG48", "MEM_06":"NEB26"}
nonill_subjs = ["MEM_08","MEM_10","MEM_11"]
n_subjs = len(sub_dict)   # only count needed later

# behavioral variable data for correlations
ill_rats = [0.7656, 0.6875, 0.6172, 0.6172, 0.5547, 0.4531, 0.4219, 0.3828, 0.3594, 0.3359, 0.3359, 0.25]
noisecat = [10, 4, 16, 4, 7, 4, 12, 24, 17, 15, 13, 21]
# pick which one to do analysis for
Behav = ill_rats


# load the TF alpha diff (I-B) STCs, morph them, and collect to a data array
# GA collect bin
GA_stcs_cont = []
# fsaverage SRC and adjacency
fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(proc_dir))
spat_adjacency = mne.spatial_src_adjacency(fs_src)
temp_adjacency = 102    # = no. of time bins
adjacency = mne.stats.combine_adjacency(temp_adjacency, spat_adjacency)

# # SUBJECT DATA COLLECTION and Morphing
# for meg,mri in sub_dict.items():
#     # load fwd['src'] and morph
#     fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,meg))
#     morph = mne.read_source_morph("{}{}_fs_oct6-morph.h5".format(proc_dir,meg))
#     # load data, morph and collect them
#     stc_cont = mne.read_source_estimate("{}{}_TF_dics_diff_I-B_7.0-11.0-stc.h5".format(tf_dir,meg))
#     stc_fs_cont = morph.apply(stc_cont)
#     GA_stcs_cont.append(stc_fs_cont)

# GA calc and stc plotting prep
# GA_plot_stc = GA_stcs_cont[0].copy()
# GA_cont_meandat = np.mean([cont.data for cont in GA_stcs_cont], axis=0)
# GA_plot_stc.data = GA_cont_meandat
GA_plot_stc = mne.read_source_estimate("{}GA_source_TF_diff_plotting-stc.h5".format(save_dir))

# data array (obs x times x locs)
# X = np.array([stc.data.T for stc in GA_stcs_cont])
# X = X * 1e27    # otherwise variance too close to 0, get error
# print("Data array shape   ", X.shape)
# np.save("{}MEMO_corr_clustering_X.npy".format(save_dir), X)
X = np.load("{}MEMO_corr_clustering_X.npy".format(save_dir))

# GA spatiotemporal clustering prep
# parameters
threshold = 2.201            # 2.781 (for 1-tailed .01 threshold, df = 11)
def stat_fun(X):
    Behav = [0.7656, 0.6875, 0.6172, 0.6172, 0.5547, 0.4531, 0.4219, 0.3828, 0.3594, 0.3359, 0.3359, 0.25]
    tvals = []
    for i in range(X.shape[1]):
        r, p = stats.pearsonr(X[:,i], Behav)
        tval = (r * np.sqrt(12-2)) / np.sqrt(1 - r**2)
        tvals.append(tval)
    return np.array(tvals)

# cluster stats
t_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(X, n_permutations=1000, stat_fun=stat_fun, threshold = threshold,
                                                                                     tail=0, adjacency=adjacency, n_jobs=4, step_down_p=0.05,
                                                                                     t_power=1, out_type='indices')

# # START CALCULATIONS - Correlation R and T
# # calculate Pearson's r for each vertex-time-point to Behavioral variable of the subject
# X_Rval = np.empty((X.shape[1],X.shape[2]))
# X_R_Tval = np.empty((X.shape[1],X.shape[2]))
# for time_idx in range(X.shape[1]):
#     for vert_idx in range(X.shape[2]):
#         X_Rval[time_idx, vert_idx], p = stats.pearsonr(X[:, time_idx, vert_idx], Behav)
# # calculate an according t-value for each r
# X_R_Tval = (X_Rval * np.sqrt(n_subjs-2)) / np.sqrt(1 - X_Rval**2)
#
# # find initial clusters in the T-vals
# print("Looking for clusters...")
# clusters, cluster_stats = _find_clusters(X_R_Tval,threshold=threshold,
#                                                adjacency=adjacency,
#                                                tail=0)
# print("Found {} initial clusters...".format(len(clusters)))
# print("Max Cluster T-Sum is {}".format(cluster_stats.max()))
#
# # setup cluster permutation
# n_perms = 1000
# cluster_H0 = np.zeros(n_perms)
# # here comes the loop
# print("Now starting {} permutations".format(n_perms))
# for i in range(n_perms):
#     if i in [10,20,50,100,200,300,400,500,600,700,800,900]:
#         print("{} th iteration".format(i))
#     # permute the behavioral values over subjects
#     Beh_perm = random.sample(Behav,k=n_subjs)
#     # calculate Pearson's r for each time-vertex point to Behavioral variable of the permuted subj data
#     XP_Rval = np.empty((X.shape[1],X.shape[2]))
#     XP_R_Tval = np.empty((X.shape[1],X.shape[2]))
#     for time_idx in range(X.shape[1]):
#         for vert_idx in range(X.shape[2]):
#             XP_Rval[time_idx, vert_idx], p = stats.pearsonr(X[:, time_idx, vert_idx], Behav_perm)
#     # calculate an according t-value for each r
#     XP_R_Tval = (XP_Rval * np.sqrt(n_subjs-2)) / np.sqrt(1 - XP_Rval**2)
#     # now find clusters in the T-vals
#     perm_clusters, perm_cluster_stats = _find_clusters(XP_R_Tval,threshold=threshold,
#                                                    adjacency=adjacency,
#                                                    tail=0)
#     if len(perm_clusters):
#         cluster_H0[i] = np.abs(perm_cluster_stats).max()     # this should be changed to cluster_H0[i] = np.abs(perm_cluster_stats).max()
#     else:
#         cluster_H0[i] = np.nan
#
# # get upper CI bound from cluster mass H0
# clust_threshold = np.quantile(cluster_H0[~np.isnan(cluster_H0)], [.95])
# print("T-Sum Threshold after permutation is {}".format(clust_threshold))
# # good cluster inds
# good_cluster_inds = np.where(np.abs(cluster_stats) > clust_threshold)[0]
# print("{} good clusters remained.".format(len(good_cluster_inds)))
