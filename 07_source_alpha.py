## Power analyses

import mne
import numpy as np
mne.viz.set_3d_backend('pyvista')

# directories
trans_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/trans/" # enter your special trans file folder here
meg_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
# subjects
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_06":"NEB26","MEM_09":"PAG48"}
nonill_dict = {"MEM_08":"WKI71","MEM_10":"NAL22","MEM_11":"KIY23_fa"}

# conditions x time windows for source loc
conds = ('cont', 'break')
# tmins = [1.0, 1.3, 1.6, 1.9]
# tmaxs = [1.3, 1.6, 1.9, 2.2]
# TWs = ["TW0","TW1","TW2","TW3"]
new_tmins = [0.95, 1.55]
new_tmaxs = [1.45, 2.05]
TWs = ["before", "after"]

# the frequencies passed as lists (for filter calculation)
freqs = [7, 8, 9, 10, 11]
cycs = 7

## PREP PARAMETERS for Power Group Analyses

# list for collecting stcs for group average for plotting
all_diff = []

## POWER analyses

# load fsaverage source space to morph to; prepare fsaverage adjacency matrix for cluster permutation analyses
fs_src = mne.read_source_spaces("{}fsaverage_oct6-src.fif".format(meg_dir))
adjacency = mne.spatial_src_adjacency(fs_src)

## prep subject STCs, make Diff_STC and morph to 'fsaverage' -- collect for group analysis
for meg,mri in sub_dict.items():
    # load info and forward
    info = mne.io.read_info("{}{}-analysis-epo.fif".format(meg_dir,meg))
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(meg_dir,meg))
    # load 'big' CSD for common filters
    csd = mne.time_frequency.read_csd("{}{}_alpha-csd.h5".format(meg_dir,meg))
    # # prep filters and save them
    # filters = mne.beamformer.make_dics(info,fwd,csd.mean(),pick_ori='max-power',reduce_rank=False,depth=1.0,inversion='single')
    # filters.save('{}{}_alpha-dics.h5'.format(meg_dir,meg),overwrite=True)
    # del csd
    # load filters for DICS beamformer
    filters = mne.beamformer.read_beamformer('{}{}_alpha-dics.h5'.format(meg_dir,meg))
    tw_fs_diffs = []
    for tw in TWs:
        # load CSDs for conditions to compare, apply filters
        csd_c = mne.time_frequency.read_csd("{}{}_alpha_cont_{}-csd.h5".format(meg_dir,meg,tw))
        csd_b = mne.time_frequency.read_csd("{}{}_alpha_break_{}-csd.h5".format(meg_dir,meg,tw))
        stc_c, freqs_c = mne.beamformer.apply_dics_csd(csd_c.mean(),filters)
        stc_b, freqs_b = mne.beamformer.apply_dics_csd(csd_b.mean(),filters)
        # calculate the difference between conditions
        # stc_diff = (stc_c - stc_b) / stc_b
        stc_diff = stc_c - stc_b
        # morph diff to fsaverage
        morph = mne.read_source_morph("{}{}_fs_oct6-morph.h5".format(meg_dir,meg))
        stc_fs_diff = morph.apply(stc_diff)
        tw_fs_diffs.append(stc_fs_diff)
    all_diff.append(tw_fs_diffs)

for i in range(len(TWs)):
    stcs = [x[i] for x in all_diff]
    # create STC grand average for plotting
    stc_sum = stcs.pop()
    for stc in stcs:
        stc_sum = stc_sum + stc
    GA_stc_diff = stc_sum / len(all_diff)
    # plot GA difference on fsaverage
    brain = GA_stc_diff.plot(subjects_dir=mri_dir,subject='fsaverage',surface='inflated',hemi='both',clim={'kind':'value','pos_lims':(0.8e-27,1.2e-27,1.6e-27)},
                             time_viewer=False,src=fs_src,show_traces=False,title="Source Alpha Difference - TW {}".format(i))
    brain.add_annotation('aparc', borders=1, alpha=0.9)


## separate source level cluster perm statistic

threshold = None     ## choose initial T-threshold for clustering; based on p-value of .05 or .01 for df = (subj_n-1); with df=17 = 2.11, or 2.898

# now do cluster permutation analysis for each time window
for i in range(len(TWs)):
    stcs = [x[i] for x in all_diff]
    print("Performing cluster analysis for TW  {}".format(i+1))
    X = np.array([stc.data.T for stc in stcs])
    t_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(X, n_permutations=1024, threshold = threshold, tail=1, adjacency=adjacency, n_jobs=4, step_down_p=0.05, t_power=1, out_type='indices')
    # get significant clusters and plot
    good_cluster_inds = np.where(cluster_pv < 0.05)[0]
    if len(good_cluster_inds):
        stc_clu_summ = mne.stats.summarize_clusters_stc(clu, p_thresh=0.05, tstep=0.001, tmin=0, subject='fsaverage', vertices=fs_src)
        brain = stc_clu_summ.plot(subjects_dir=mri_dir,subject='fsaverage',surface='white',hemi='both',time_viewer=True,show_traces=False,colormap='coolwarm')       # if plotting problems, try adding: clim={'kind':'value','pos_lims':(0,0.0005,0.01)}
    else:
        print("No sign. clusters found")
