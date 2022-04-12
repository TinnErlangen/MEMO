# look at Grand Averages of Evoked, and of TFR
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


# collect dicts for GA and cluster stats
GA_illu_ltcs = {}
GA_break_ltcs = {}
GA_cont_ltcs = {}

# SUBJECT DATA COLLECTION and PLOTS

for meg,mri in sub_dict.items():

    # collect bins
    stcs_illu = []   # this would collect together the 7 freq bin stcs
    stcs_break = []
    stcs_cont = []
    ltcs_illu = {}   # this should collect, for each freq bin, a dict (nested!) with lab_name: ltc
    ltcs_break = {}
    ltcs_cont = {}
    # load fwd['src'] and labels
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(meg_dir,meg))
    labs = mne.read_labels_from_annot(mri, parc='aparc',subjects_dir=mri_dir,sort=True)
    lab_names = [l.name for l in labs]


    # loop over fbands to read and collect data and label time courses
    for fb in range(len(freq_bins)):    # for each frequency band
        # illusion
        # need dict container for ltc saving
        illu_ltc_dict = {}
        stc_illu = mne.read_source_estimate("{}{}_TF_dics_illu_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
        stcs_illu.append(stc_illu)
        ltc_illu = mne.extract_label_time_course(stc_illu, labs, mode='mean', src=fwd['src'], allow_empty=True)     # this will automatically extract all ltcs for ordered labels (alphabetical lh,rh), then subcortical labels (all lh, all rh)
        for i, ln in enumerate(lab_names):
            illu_ltc_dict[ln] = ltc_illu[i, :]
        ltcs_illu[fb] = illu_ltc_dict
        # break
        # need dict container for ltc saving
        break_ltc_dict = {}
        stc_break = mne.read_source_estimate("{}{}_TF_dics_break_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
        stcs_break.append(stc_break)
        ltc_break = mne.extract_label_time_course(stc_break, labs, mode='mean', src=fwd['src'], allow_empty=True)     # this will automatically extract all ltcs for ordered labels (alphabetical lh,rh), then subcortical labels (all lh, all rh)
        for i, ln in enumerate(lab_names):
            break_ltc_dict[ln] = ltc_break[i, :]
        ltcs_break[fb] = break_ltc_dict
        # contrast I-B
        # need dict container for ltc saving
        cont_ltc_dict = {}
        stc_cont = mne.read_source_estimate("{}{}_TF_dics_diff_I-B_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
        stcs_cont.append(stc_cont)
        ltc_cont = mne.extract_label_time_course(stc_cont, labs, mode='mean', src=fwd['src'], allow_empty=True)     # this will automatically extract all ltcs for ordered labels (alphabetical lh,rh), then subcortical labels (all lh, all rh)
        for i, ln in enumerate(lab_names):
            cont_ltc_dict[ln] = ltc_cont[i, :]
        ltcs_cont[fb] = cont_ltc_dict

    # collect subject data into GA collectors
    GA_illu_ltcs[mri] = ltcs_illu     # nested dict of 7 fb : 80 labels subject ltcs into GA dict
    GA_break_ltcs[mri] = ltcs_break
    GA_cont_ltcs[mri] = ltcs_cont

    # # SUBJECT PLOTS in LABELS     # note: add lines for closing figures in the background! memory overload!
    # # loop over labels:freq bands , collect stcs, make and save plots
    # # each time, i.e. for each label, we use the 7 fb stcs as containers to fill in the ltc data
    # for loi in all_labels:
    #     for fb in range(len(freq_bins)):
    #         stcs_neg[fb].data = np.tile(ltcs_neg[fb][loi], (stcs_neg[fb].data.shape[0], 1))     # we tile the same ltc for all vertices in the stc, to avoid an error (note!)
    #         stcs_pos[fb].data = np.tile(ltcs_pos[fb][loi], (stcs_pos[fb].data.shape[0], 1))
    #         stcs_cont[fb].data = np.tile(ltcs_cont[fb][loi], (stcs_cont[fb].data.shape[0], 1))
    #     fig_n = plot_source_spectrogram(stcs_neg, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
    #                                     cmap='hot_r', vmin=0, vmax=5e-26, title='{} - Source TFR in {} - NEG'.format(meg,loi), show=False)    # freq_bins! or error
    #     fig_n.savefig("{}{}_neg_{}.png".format(plot_dir,meg,loi))
    #     fig_p = plot_source_spectrogram(stcs_pos, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
    #                                     cmap='hot_r', vmin=0, vmax=5e-26, title='{} - Source TFR in {} - POS'.format(meg,loi), show=False)
    #     fig_p.savefig("{}{}_pos_{}.png".format(plot_dir,meg,loi))
    #     fig_c = plot_source_spectrogram(stcs_cont, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
    #                                     cmap='Spectral_r', vmin=-1e-26, vmax=1e-26, title='{} - Source TFR in {} - Contrast NEG-POS'.format(meg,loi), show=False)
    #     fig_c.savefig("{}{}_cont_N-P_{}.png".format(plot_dir,meg,loi))

#  GRAND AVERAGE PLOTS in LABELS
# need containers (lists of 7 freq band STCs) for GA plots, use from last subject
GA_illu_stcs = [stc.copy() for stc in stcs_cont]    # gotta copy each stc, otherwise just pointers, that's why we ended up with contrast values in GA neg/pos plots before...
GA_break_stcs = [stc.copy() for stc in stcs_cont]
GA_cont_stcs = [stc.copy() for stc in stcs_cont]
# loop over labels
# start a file for collecting cluster stats
filename = "{}MEMO_TF_DICS_label_clu_stats.txt".format(plot_dir)
with open(filename, "w") as file:
    for loi in lab_names:
        print("Doing GAs for Label:  {}".format(loi))
        file.write("Doing GAs for Label:  {}\n".format(loi))
        # loop over freq bands
        for fb in range(len(freq_bins)):
            illu_collect = np.array([GA_illu_ltcs[mri][fb][loi] for mri in sub_dict.values()])
            GA_illu_stcs[fb].data = np.tile(np.mean(illu_collect, axis=0), (GA_illu_stcs[fb].data.shape[0], 1))
            break_collect = np.array([GA_break_ltcs[mri][fb][loi] for mri in sub_dict.values()])
            GA_break_stcs[fb].data = np.tile(np.mean(break_collect, axis=0), (GA_break_stcs[fb].data.shape[0], 1))
            cont_collect = np.array([GA_cont_ltcs[mri][fb][loi] for mri in sub_dict.values()])
            GA_cont_stcs[fb].data = np.tile(np.mean(cont_collect, axis=0), (GA_cont_stcs[fb].data.shape[0], 1))
            # do cluster stat on freq band over time bins
            print("Cluster Stats for Label {} Frequency {}-{} Hz...".format(loi, freq_bins[fb][0], freq_bins[fb][-1]))
            file.write("Cluster Stats for Label {} Frequency {}-{} Hz...".format(loi, freq_bins[fb][0], freq_bins[fb][-1]))
            X = cont_collect * 1e27
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(X,threshold=None, n_permutations=1024, tail=0, stat_fun=None, adjacency=None, n_jobs=6, step_down_p=0.05, t_power=1, out_type='indices', seed=321)
            good_cluster_inds = np.where(cluster_pv < 0.05)[0]
            if len(good_cluster_inds):
                print("Total {} good clusters found:".format(len(good_cluster_inds)))
                file.write("Total {} good clusters found:\n".format(len(good_cluster_inds)))
                for c_ix in good_cluster_inds:
                    file.write("Cluster number {}:\n".format(c_ix))
                    file.write("P-value: {}\n".format(cluster_pv[c_ix]))
                tbix = np.where(np.abs(t_obs) > 2.201)
                file.write("Time bin indices: {}\n".format([i for i in tbix[0]]))
                file.write("T-value timecourse: {}\n".format(t_obs))
            else:
                print("No sign. clusters found")
                file.write("No sign. clusters found\n")
                marg_cluster_inds = np.where(cluster_pv < 0.10)[0]
                if len(marg_cluster_inds):
                    print("Total {} marginal clusters found:".format(len(marg_cluster_inds)))
                    file.write("Total {} marginal clusters found:\n".format(len(marg_cluster_inds)))
                    for c_ix in marg_cluster_inds:
                        file.write("Cluster number {}\n:".format(c_ix))
                        file.write("P-value: {}\n".format(cluster_pv[c_ix]))
                    tbix = np.where(np.abs(t_obs) > 2.201)              # !! adjusted for sample size = 12 !!
                    file.write("Time bin indices: {}\n".format([i for i in tbix[0]]))
                    file.write("T-value timecourse: {}\n".format(t_obs))
                else:
                    print("Not even marginally sign. clusters found")
                    file.write("Not even marginally sign. clusters found\n")
        # # make and save label GA plots
        # fig_i = plot_source_spectrogram(GA_illu_stcs, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
        #                                 cmap='hot_r', vmin=0, vmax=5e-26, title='GA - Source TFR in {} - NEG'.format(loi), show=False)    # freq_bins! or error
        # fig_i.savefig("{}GA_Illu_{}.png".format(plot_dir,loi))
        # fig_b = plot_source_spectrogram(GA_break_stcs, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
        #                                 cmap='hot_r', vmin=0, vmax=5e-26, title='GA - Source TFR in {} - POS'.format(loi), show=False)
        # fig_b.savefig("{}GA_Break_{}.png".format(plot_dir,loi))
        # fig_c = plot_source_spectrogram(GA_cont_stcs, freq_bins, tmin=tmin_plot, tmax=tmax_plot, source_index=None, colorbar=True,
        #                                 cmap='Spectral_r', vmin=-0.5e-26, vmax=0.5e-26, title='GA - Source TFR in {} - Contrast NEG-POS'.format(loi), show=False)
        # fig_c.savefig("{}GA_Diff_I-B_{}.png".format(plot_dir,loi))
