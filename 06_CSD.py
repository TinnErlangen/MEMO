## Compute the Cross-Spectral-Density ##
## Preparation for DICS Beamformer Source Localization ##

import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency import csd_morlet,csd_multitaper

# define file locations
proc_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
# subjects list
subjs = ["MEM_14","MEM_13","MEM_03","MEM_05","MEM_04","MEM_02","MEM_15","MEM_01",
         "MEM_12","MEM_07","MEM_06","MEM_09"]
nonill_subjs = ["MEM_08","MEM_10","MEM_11"]

# conditions x time windows for source loc
conds = ['cont', 'break']
tmins = [1.0, 1.3, 1.6, 1.9]
tmaxs = [1.3, 1.6, 1.9, 2.2]
new_tmins = [0.95, 1.55]
new_tmaxs = [1.45, 2.05]
new_tws = ["before", "after"]

# the frequencies passed as lists (for CSD calculation)
freqs = [7, 8, 9, 10, 11]
cycs = 7


# CSD over all conditions to calculate shared filters
for sub in subjs:
    # read in the epo files
    epo = mne.read_epochs("{}{}-analysis-epo.fif".format(proc_dir,sub))

    # # calculate "big" alpha CSDs for common filters
    # csd = csd_morlet(epo, frequencies=freqs, n_jobs=8, n_cycles=cycs, decim=1)
    # csd.save("{}{}_alpha-csd.h5".format(proc_dir,sub))
    #
    # # calculate "small" alpha CSDs for each condition and time window
    # for i, (tmin, tmax) in enumerate(zip(tmins,tmaxs)):
    #     epo_win = epo.copy()
    #     epo_win.crop(tmin=tmin, tmax=tmax)
    #     for cond in conds:
    #         csd_ct = csd_morlet(epo_win[cond], frequencies=freqs, n_jobs=8, n_cycles=cycs, decim=1)
    #         csd_ct.save("{}{}_alpha_{}_TW{}-csd.h5".format(proc_dir,sub,cond,i))

    # calculate new "small" alpha CSDs for each condition and time window (before, after)
    for (tmin, tmax, tw) in zip(new_tmins, new_tmaxs, new_tws):
        epo_win = epo.copy()
        epo_win.crop(tmin=tmin, tmax=tmax)
        for cond in conds:
            csd_ct = csd_morlet(epo_win[cond], frequencies=freqs, n_jobs=8, n_cycles=cycs, decim=1)
            csd_ct.save("{}{}_alpha_{}_{}-csd.h5".format(proc_dir,sub,cond,tw))
