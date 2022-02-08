# look at Grand Averages of Evoked, and of TFR
import mne
import numpy as np

# setup directories
# beh_dir = "/Volumes/Windows/MEMO/MEMO_beh/"
proc_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
fig_dir = "/home/cora/hdd/MEG/MEMO_analyses/"

# subjects list
subjs = ["MEM_14","MEM_13","MEM_03","MEM_05","MEM_04","MEM_02","MEM_15","MEM_01",
         "MEM_12","MEM_07","MEM_06","MEM_09"]
nonill_subjs = ["MEM_08","MEM_10","MEM_11"]
#subjs = ["MEM_03","MEM_05","MEM_04","MEM_02","MEM_01",
#         "MEM_07","MEM_06"]


# load the epos
epos = []
for sub in subjs:
    epo = mne.read_epochs("{}{}-analysis-epo.fif".format(proc_dir,sub))
    epo.interpolate_bads()  # on sensor level, this makes sense
    epos.append(epo)
# get layout for plotting later
layout = mne.find_layout(epo.info)
mag_names = [epo.ch_names[p] for p in mne.pick_types(epo.info, meg=True)]
layout.names = mag_names

# # do the GA Evoked (break,cont, and difference cont-break)
# # get lists and Evoked-containers started
# contevs = [epos[0]['cont'].average()]
# breakevs = [epos[0]['break'].average()]
# diffevs = [contevs[0].copy()]
# diffevs[0].data = contevs[0].data - breakevs[0].data
# diffevs[0].comment = 'contrast cont-break'
# # then loop through remaining epos for averaging
# for epo in epos[1:]:
#     contev = epo['cont'].average()
#     breakev = epo['break'].average()
#     diffev = contev.copy()
#     diffev.data = contev.data - breakev.data
#     diffev.comment = 'contrast cont-break'
#     contevs.append(contev)
#     breakevs.append(breakev)
#     diffevs.append(diffev)
# # calc GAs and plot
# GA_cont = mne.grand_average(contevs)
# GA_cont.plot_joint()
# GA_break = mne.grand_average(breakevs)
# GA_break.plot_joint()
# GA_diff = mne.grand_average(diffevs)
# GA_diff.plot_joint()

# calculate TFRs per subject
freqs = np.arange(3,47,1)
n_cycles = 7
cont_TFRs = []
break_TFRs = []
diff_TFRs = []
for epo in epos:
    cont_TFR = mne.time_frequency.tfr_morlet(epo['cont'], freqs, n_cycles, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    cont_TFR.apply_baseline((None,0),mode='percent')
    cont_TFRs.append(cont_TFR)
    break_TFR = mne.time_frequency.tfr_morlet(epo['break'], freqs, n_cycles, use_fft=False, return_itc=False, decim=1, n_jobs=6, picks=None, zero_mean=True, average=True, output='power')
    break_TFR.apply_baseline((None,0),mode='percent')
    break_TFRs.append(break_TFR)
    diff_TFR = cont_TFR.copy()
    diff_TFR.data = cont_TFR.data - break_TFR.data
    diff_TFRs.append(diff_TFR)

#GA_cont_TFR = mne.grand_average(cont_TFRs)
#GA_cont_TFR.save("{}GA_MEM01-15_cont-tfr.h5".format(proc_dir))
#GA_cont_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black')

#GA_break_TFR = mne.grand_average(break_TFRs)
#GA_break_TFR.save("{}GA_MEM01-15_break-tfr.h5".format(proc_dir))
#GA_break_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black')
GA_diff_TFR = mne.grand_average(diff_TFRs)
#GA_diff_TFR.save("{}GA_MEM01-15_diff_C-B-tfr.h5".format(proc_dir) ,overwrite=True)
#GA_diff_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',vmin=-1,vmax=1)

# # re-plotting with different time or freq boundaries
# GA_diff_TFR.plot_topo(baseline=None,mode=None,layout=layout,fig_facecolor='white',font_color='black',tmin=None,tmax=5.0,fmin=5,fmax=35,vmin=-1,vmax=1)
# # or try a joint plot for 'peak tiles with title
#GA_cont_TFR.plot_joint(timefreqs={(1, 10): (0.1, 2)},baseline=None,mode=None,title="GA - TFR response Negative-Positive")  # set favorite topo peaks here, value tuple sets windows centered on time and freq

GA_diff_TFR.plot(baseline=None,mode=None,title="GA - TFR response Negative-Positive")

# STATISTICS - spatio-temporal cluster T-test cont vs. break for Alpha band (7-11 Hz)

# we crop the TFRs to the alpha band, and collect the data arrays into a list container
alpha_diff_TFRs = []
for tfr in diff_TFRs:
    copy = tfr.copy()
    copy.crop(fmin=7,fmax=11)
    alpha_diff_TFRs.append(copy.data)
# then we average over the frequencies (= axis 1 in a channel x freq x times array)
X_alpha_avg = [np.mean(x, axis=1) for x in alpha_diff_TFRs]
# and swap the axes from (channels x times) to (times x channels) as needed in the stats array
X_alpha_prep = [np.transpose(x, (1,0)) for x in X_alpha_avg]
# then we convert the list into our data array for the stats, where the 1st dimension is observations (i.e. here, subjects)
X_alpha_diff = np.array(X_alpha_prep)

# make spatio-temporal cluster T-test
# set parameters
threshold = None
# get channel connectivity for cluster permutation test
adjacency, ch_names = mne.channels.find_ch_adjacency(epo.info, ch_type='mag')
# MNE uses a 1sample T-test, i.e. one first subtracts the conditions, then tests against zero (that's why we used the diff_data directly)
# needed data form: X = array of shape observations x times/freqs x locs/chans
t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X_alpha_diff, threshold=threshold, n_permutations=1024,
                                                                               tail=0, adjacency=adjacency, n_jobs=4, step_down_p=0,
                                                                               t_power=1, out_type='indices')

# now explore and plot the clusters
# get indices of good clusters
good_cluster_inds = np.where(cluster_pv < .05)[0]   # it's a tuple, with [0] we grab the array therein
# if there are any, do more...
if good_cluster_inds.any():
    # then loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, ch_inds = clusters[clu_idx]
        ch_inds = np.unique(ch_inds)
        time_inds = np.unique(time_inds)




        # get topography for T stat (mean over cluster freqs)
        t_map = t_obs[time_inds, ...].mean(axis=0)
        # create spatial mask for plotting (setting cluster channels to "True")
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], epo.info, tmin=0)
        fig = t_evoked.plot_topomap(times=0, mask=mask, cmap='bwr',
                                    vmin=np.min, vmax=np.max,scalings=1.0,
                                    units="T_val", time_format= "",
                                    title="Alpha Power \n{} - {} ms".format(time_inds[0]*5-105,time_inds[-1]*5-105),
                                    mask_params=dict(markersize=4),
                                    size = 6, show=True)
        #fig.savefig("{d}GA_T-cluster_cont_minus_break_topo.png".format(d=fig_dir))

GA_diff_TFR.plot_joint(picks=ch_inds, timefreqs={(1, 9): (0.1, 2), (1.5, 9): (0.1, 2), (2, 9): (0.1, 2)}, baseline=None, mode=None, fmin=5, fmax= 20, title="GA - TFR response Negative-Positive")
GA_diff_TFR.plot_joint(picks=ch_inds, timefreqs={(0.05, 9): (0.1, 2),(0.25, 9): (0.1, 2),(0.45, 9): (0.1, 2),(0.65, 9): (0.1, 2),(0.85, 9): (0.1, 2),(1.05, 9): (0.1, 2),(1.25, 9): (0.1, 2),(1.45, 9): (0.1, 2),
                       (1.65, 9): (0.1, 2),(1.85, 9): (0.1, 2),(2.05, 9): (0.1, 2),(2.25, 9): (0.1, 2)}, baseline=None, mode=None, fmin=5, fmax= 20, title="GA - TFR response Negative-Positive")
GA_diff_TFR.plot_joint(picks=ch_inds, timefreqs={(1.15, 9): (0.15, 4), (1.45, 9): (0.15, 4), (1.75, 9): (0.15, 4), (2.05, 9): (0.15, 4)}, baseline=None, mode=None, fmin=5, fmax= 20, title="GA - TFR response Negative-Positive")

#GA_diff_TFR.plot(picks=ch_inds, title="GA - TFR response Negative-Positive Cluster")
