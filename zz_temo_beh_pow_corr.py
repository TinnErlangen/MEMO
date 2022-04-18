import numpy as np
import mne
import pandas as pd
import random
from scipy import stats
from mayavi import mlab
import matplotlib.pyplot as plt
plt.ion()

from sklearn.linear_model import LinearRegression

from mne.stats.cluster_level import _setup_adjacency, _find_clusters, \
    _reshape_clusters

mne.viz.set_3d_backend('pyvista')

# setup files and folders, subject lists

proc_dir = "D:/TEMO_analyses/proc/"
mri_dir = "D:/freesurfer/subjects/"
acute = {"CAT_06":"ECN26","CAT_11":"SAN53","CAT_12":"NAO26","CAT_20":"ENC34"}
sub_dict = {"CAT_03":"IAU64_fa","CAT_05":"ZAA66_fa","CAT_07":"RWB50","CAT_09":"BEU80",
            "CAT_10":"MAU06","CAT_13":"BTE00_fa","CAT_14":"GIR89","CAT_16":"WAM42",
            "CAT_18":"EIP04","CAT_21":"MAM73","CAT_22":"AIR46","CAT_23":"RIN08_fa",
            "CAT_24":"NIP84","CAT_25":"ELL12","CAT_26":"NLF76","CAT_27":"UVC22",
            "CAT_28":"AOE32_fa","CAT_29":"SAT06_fa"}
# sub_dict = {"CAT_06":"ECN26"}
n_subjs = len(sub_dict)   # only count needed later
save_dir = "D:/TEMO_analyses_new/source_plots/beh_corr/"  # for saving plots

# values to choose from
freqs = {"theta_low":list(np.arange(4,5)),"theta_high":list(np.arange(5,7)),"alpha_low":list(np.arange(7,9)),"alpha_high":list(np.arange(9,14)),
         "beta_low":list(np.arange(14,21)),"beta_high":list(np.arange(21,32)),"gamma":list(np.arange(32,47))}
fmins = [4, 5, 7, 9, 14, 21, 32]
fmaxs = [5, 7, 9, 14, 21, 32, 46]
# change / update this part !!! :
freqs_ix = {"theta_low": 0, "theta_high": 1, "alpha_low": 2, "alpha_high": 3, "beta_low": 4, "beta_high": 5, "gamma": 6}
conds_for_contrasts = ['Att','Neg','Pos','Temo']
beh_vars = ['Ton_Laut', 'Ton_Ang','ER_ges', 'Angst_ges', 'Psycho_ges']

## SET THE VARIABLES : Behav, Contrast & Frequency
contrast = ('Neg','Pos')    # choose 2 conditions to contrast
beh_var = 'Laut'            # choose between 'Laut' and 'Ang'
freq = "gamma"          # choose freq band
if freq != "gamma_high":
    ix = freqs_ix[freq]
print("Correlation Cluster Analysis")
print("Contrast:   {}  vs. {}".format(contrast[0],contrast[1]))
print("Behav Variable:   {}".format(beh_var))
print("Frequency:   {}".format(freq))

# get the STC power data, subtract them, and reduce to the specified freq band
Con1 = np.load("{}TEMO_chronic_{}_7b_dics_dataX.npy".format(proc_dir,contrast[0]))
Con2 = np.load("{}TEMO_chronic_{}_7b_dics_dataX.npy".format(proc_dir,contrast[1]))
Diff = Con1 - Con2
Diff = np.squeeze(Diff[:,ix,:])

# get the behavioral data array ready & subtract the contrast variables
TEMO_df = pd.read_csv('{}TEMO_chronic_behav_data.csv'.format(proc_dir))
beh1 = contrast[0][0] + "_" + beh_var
beh2 = contrast[1][0] + "_" + beh_var
Behav = np.array(TEMO_df[beh1] - TEMO_df[beh2])

# START CALCULATIONS - Correlation R and T
# calculate Pearson's r for each vertex to Behavioral variable of the subject
X_Rval = np.empty(Diff.shape[1])
X_R_Tval = np.empty(Diff.shape[1])
for vert_idx in range(Diff.shape[1]):
    X_Rval[vert_idx], p = stats.pearsonr(Diff[:,vert_idx],Behav)
# calculate an according t-value for each r
X_R_Tval = (X_Rval * np.sqrt(n_subjs-2)) / np.sqrt(1 - X_Rval**2)

# setup for clustering -- t-thresholds for N=18: 2.11 (.05), 2.898 (.01), or 3.965 (.001)
threshold = 2.898
# load fsaverage source space; prepare fsaverage adjacency for cluster permutation analyses
fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(proc_dir))
adjacency = mne.spatial_src_adjacency(fs_src)

# find clusters in the T-vals
print("Looking for clusters...")
clusters, cluster_stats = _find_clusters(X_R_Tval,threshold=threshold,
                                               adjacency=adjacency,
                                               tail=0)
print("Found {} initial clusters...".format(len(clusters)))
print("Max Cluster T-Sum is {}".format(cluster_stats.max()))

# setup cluster permutation
n_perms = 1000
cluster_H0 = np.zeros(n_perms)
# here comes the loop
print("Now starting {} permutations".format(n_perms))
for i in range(n_perms):
    if i in [10,20,50,100,200,300,400,500,600,700,800,900]:
        print("{} th iteration".format(i))
    # permute the behavioral values over subjects
    Beh_perm = random.sample(list(Behav),k=n_subjs)
    # calculate Pearson's r for each vertex to Behavioral variable of the permuted subj data
    XP_Rval = np.empty(Diff.shape[1])
    XP_R_Tval = np.empty(Diff.shape[1])
    for vert_idx in range(Diff.shape[1]):
        XP_Rval[vert_idx], p = stats.pearsonr(Diff[:,vert_idx],Beh_perm)
    # calculate an according t-value for each r
    XP_R_Tval = (XP_Rval * np.sqrt(n_subjs-2)) / np.sqrt(1 - XP_Rval**2)
    # now find clusters in the T-vals
    perm_clusters, perm_cluster_stats = _find_clusters(XP_R_Tval,threshold=threshold,
                                                   adjacency=adjacency,
                                                   tail=0)
    if len(perm_clusters):
        cluster_H0[i] = np.abs(perm_cluster_stats).max()     # this should be changed to cluster_H0[i] = np.abs(perm_cluster_stats).max()
    else:
        cluster_H0[i] = np.nan

# get upper CI bound from cluster mass H0
clust_threshold = np.quantile(cluster_H0[~np.isnan(cluster_H0)], [.95])
print("T-Sum Threshold after permutation is {}".format(clust_threshold))
# good cluster inds
good_cluster_inds = np.where(np.abs(cluster_stats) > clust_threshold)[0]
print("{} good clusters remained.".format(len(good_cluster_inds)))

if len(good_cluster_inds):
    X_R_T = np.expand_dims(X_R_Tval, axis=0)
    cluster_pv = np.ones_like(cluster_stats)
    for ind in good_cluster_inds:
        cluster_pv[ind] = 0
    clusters_tv = [[np.zeros_like(x),x] for x in clusters]  # this creates t_inds arrays needed on top of v_inds
    stc_clu_summ = mne.stats.summarize_clusters_stc((X_R_T, clusters_tv, cluster_pv, cluster_H0), p_thresh=0.05, tstep=1, tmin=0, subject='fsaverage', vertices=fs_src)
    # brain = stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,show_traces=False,colormap='coolwarm')
    brain = stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',colormap='coolwarm',show_traces=False,clim={'kind':'value','pos_lims': (0,0.5,1)})
    brain.add_annotation('aparc', borders=1, alpha=0.9)
