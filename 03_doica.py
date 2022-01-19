import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #this keeps plots interactive

# define file locations
proc_dir = "/Volumes/Windows/MEMO/MEMO_preproc/"
# pass subject and run lists
#subjs = ['MEM_01', 'MEM_02', 'MEM_03', 'MEM_04', 'MEM_05', 'MEM_06', 'MEM_07', 'MEM_08', 'MEM_10', 'MEM_11', 'MEM_12', 'MEM_13', 'MEM_14', 'MEM_15']
subjs = ['MEM_09']
runs = ["3"]
# create new lists, if you want to try things on single subjects or runs

for sub in subjs:
    for run in runs:
        #load the annotated epoch file
        mraw = mne.io.Raw('{dir}{sub}_{run}_m-raw.fif'.format(dir=proc_dir,sub=sub,run=run))
        mraw.info["bads"] += ["MRyA","MRyaA"] # add broken reference channels to bad channels list

        #define the ica for reference channels and fit it onto raw file
        icaref = mne.preprocessing.ICA(n_components=8,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mraw.info,meg=False,ref_meg=True)
        icaref.fit(mraw,picks=picks)
        #save the reference ica result in its own file
        icaref.save('{dir}{sub}_{run}_ref-ica.fif'.format(dir=proc_dir,sub=sub,run=run))

        #define the ica for MEG channels and fit it onto raw file
        icameg = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard") #parameters for ica on MEG channels
        picks = mne.pick_types(mraw.info,meg=True,ref_meg=False)
        icameg.fit(mraw,picks=picks)
        #save the MEG ica result in its own file
        icameg.save('{dir}{sub}_{run}_meg-ica.fif'.format(dir=proc_dir,sub=sub,run=run))

        #define the combined ica for MEG and reference channels and fit it onto raw file
        icaall = mne.preprocessing.ICA(n_components=100,max_iter=10000,method="picard",allow_ref_meg=True) #parameters for ica on reference channels
        picks = mne.pick_types(mraw.info,meg=True,ref_meg=True)
        icaall.fit(mraw,picks=picks)
        icaall.save('{dir}{sub}_{run}-ica.fif'.format(dir=proc_dir,sub=sub,run=run))
