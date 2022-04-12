## Calculate Time-Frequency Beamformer in Label ##
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
sub_dict = {"MEM_15":"KHA22_fa","MEM_01":"ENR41",
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


for meg,mri in sub_dict.items():
    start = time.perf_counter()
    epo = mne.read_epochs("{}{}-analysis-epo.fif".format(proc_dir,meg))
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(meg_dir,meg))
    # labels = mne.read_labels_from_annot(mri, parc='aparc', hemi='both', surf_name='white', annot_fname=None, regexp=None, subjects_dir=mri_dir, sort=False, verbose=None)
    # label = [l for l in labels if l.name == loi][0]
    print("Calculating TF_DICS continuous/illusion for {}".format(meg))
    stcs_illu = tf_dics(epo['cont'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                       reg = 0.05, label=None, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    for fb in range(len(freq_bins)):
        stcs_illu[fb].save("{}{}_TF_dics_illu_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
    print("Calculating TF_DICS break/no-illusion for {}".format(meg))
    stcs_break = tf_dics(epo['break'], fwd, noise_csds = None, tmin=tmin, tmax=tmax, tstep=tstep, win_lengths=win_lengths, subtract_evoked=False, mode='cwt_morlet', frequencies=frequencies, cwt_n_cycles=cwt_n_cycles,
                         reg = 0.05, label=None, pick_ori='max-power', inversion='single', depth=1.0, n_jobs=n_jobs)
    for fb in range(len(freq_bins)):
        stcs_break[fb].save("{}{}_TF_dics_break_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
    # now calculate the difference between conditions
    stcs_cont = [(a[0]-a[1]) for a in zip(stcs_illu,stcs_break)]
    for fb in range(len(freq_bins)):
        stcs_cont[fb].save("{}{}_TF_dics_diff_I-B_{}-{}-stc.h5".format(save_dir,meg,freq_bins[fb][0],freq_bins[fb][-1]))
    end = time.perf_counter()
    dur = np.round(end-start)
    print("Finished TF_DICS calc for {}".format(meg))
    print("This took {}".format(str(datetime.timedelta(seconds=dur))))
