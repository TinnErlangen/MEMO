# create DataFrame with source power data for LMM analyses
import mne
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# set directories
beh_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_beh/"
proc_dir = "/home/cora/hdd/MEG/MEMO_analyses/MEMO_preproc/"
mri_dir = "/home/cora/hdd/MEG/freesurfer/subjects/"
save_dir = "/home/cora/hdd/MEG/MEMO_analyses/TF_dics/"


# subjects list
subjs = ['MEM_14', 'MEM_13', 'MEM_03', 'MEM_05', 'MEM_04', 'MEM_02', 'MEM_15', 'MEM_01', 'MEM_12', 'MEM_07', 'MEM_09', 'MEM_06']    # !! was _06, then _09 before, because _09 data missed, later added
sub_dict = {"MEM_14":"KOM47_fa","MEM_13":"EEE27_fa","MEM_03":"DIU11","MEM_05":"TGH11",
            "MEM_04":"MUN79","MEM_02":"DEN59","MEM_15":"KHA22_fa","MEM_01":"ENR41",
            "MEM_12":"GIZ04","MEM_07":"FAU35_fa","MEM_09":"PAG48", "MEM_06":"NEB26"}
nonill_subjs = ["MEM_08","MEM_10","MEM_11"]
ill_rats = [0.7656, 0.6875, 0.6172, 0.6172, 0.5547, 0.4531, 0.4219, 0.3828, 0.3594, 0.3359, 0.3359, 0.25]
noisecat = [10, 4, 16, 4, 7, 4, 12, 24, 17, 15, 13, 21]

# already done, just recording here:  Correlation of Illusion Ratio and Noise Catastrophizing & Linear Plot
# stats.pearsonr(ill_rats, noisecat)    # output: (-0.6131157559156297, 0.034009395131781174)
# dict = {"subjs": subjs, "Illusion_ratio": ill_rats, "Noise_catastrophizing": noisecat}
# df = pd.DataFrame(dict)
# g = sns.lmplot(x="Illusion_ratio",y="Noise_catastrophizing",data=df)
# plt.show()

# PARAMETER SETUP
# EVENTS and CONDITIONS
# dictionaries needed to read and write old/new conditions/triggers for epochs
tone_id = {'4000_b': 120, '7000_b': 140, '5500_b': 160, '8500_b': 180}
trig_id = {v: k for k,v in tone_id.items()}
event_id = {'4000_b/cont/sure': 121, '4000_b/cont/unsure': 122,'4000_b/break/unsure': 123, '4000_b/break/sure': 124,
            '7000_b/cont/sure': 141, '7000_b/cont/unsure': 142,'7000_b/break/unsure': 143, '7000_b/break/sure': 144,
            '5500_b/cont/sure': 161, '5500_b/cont/unsure': 162,'5500_b/break/unsure': 163, '5500_b/break/sure': 164,
            '8500_b/cont/sure': 181, '8500_b/cont/unsure': 182,'8500_b/break/unsure': 183, '8500_b/break/sure': 184}
# SOURCE POWER ROIs, times, and freq parameters
# frequencies passed as lists (for CSD calculation)
freqs = [7, 8, 9, 10, 11]
cycs = 7
# time period of interest
tmin = 0.600
tmax = 1.600
# labels of interest -- alphabetical since labels get read in sorted
lois = ["superiortemporal-rh", "transversetemporal-rh", "insula-rh", "lateralorbitofrontal-rh", "medialorbitofrontal-rh",
        "rostralanteriorcingulate-rh", "temporalpole-rh", "parsopercularis-rh", "entorhinal-rh"]
lois.sort()
l_cols = ["ENT_alpha", "INS_alpha", "LOF_alpha", "MOF_alpha", "OPC_alpha", "ACC_alpha", "STP_alpha", "TPO_alpha", "TTP_alpha"]
# DATAFRAME PREP DICTIONARY
MEMO_dict = {"Subject": [], "Illusion": [], "Sure": [], "ENT_alpha": [], "INS_alpha": [], "LOF_alpha": [],
             "MOF_alpha": [], "OPC_alpha": [], "ACC_alpha": [], "STP_alpha": [], "TPO_alpha": [], "TTP_alpha": []}

for sub,mri in sub_dict.items():

    # PREPARE EPO DATA
    # run EPO & rating collection like before, just keeping all trials and saving as -lmm-epo.fif
    # load the epoched data
    epo = mne.read_epochs("{}{}_3-epo.fif".format(proc_dir,sub))
    # load the ratings.txt and collect the ratings for target trials (break sounds only)
    beh_trials = []
    with open("{}MEMO_{}.txt".format(beh_dir,sub[-2:]),"r") as file:
        lines = file.readlines()
        del lines[:4]    # delete the header line "Subject recorded...", column headings, and 2 practice trials (!)
        x_lines = [l.split() for l in lines]    # get "words" as list per line
        for l in x_lines:
            beh_trials.append((l[1],l[3]))      # collect trigger and rating as tuple
    # then look into epoch drop log (containing all original trials) to delete dropped epochs from beh_trials
    # for this, you need some tricks:
    droplog = np.array([len(x) for x in epo.drop_log])
    drop_ixs = list(np.where(droplog != 0) [0])
    new_beh_trials = [x for (i,x) in enumerate(beh_trials) if i not in drop_ixs]
    # then check that n_epochs and n_beh_trials match
    if len(epo) != len(new_beh_trials):
        print("\n\n\nWARNING! Epochs and Ratings do not match! Check for error!\n\n\n")
    # to get the ratings into the epoch labels, we have to somehow
    # rewrite the event triggers and provide a new event_id dictionary (see event_id at the top)
    # we do this with a "hack", by "pointing" to the events array
    events = epo.events     # this is creating a pointer to change the epo.events -- it works here, but one usually shouldn't do this, not to mix up variables and lose control over what changes...
    # we then make a list for looping and change the triggers to include the rating info
    events = list(events)
    for ev,rat in zip(events,new_beh_trials):
        # check that triggers match, then add the rating value to the trigger value to yield the new event id
        if int(rat[0]) == ev[-1]:
            ev[-1] = ev[-1] + int(rat[1])
    events = np.array(events)
    # then we give epo its new dictionary
    epo.event_id = event_id
    # check that everything worked and save
    print(epo)
    epo.save("{}{}-lmm-epo.fif".format(proc_dir,sub))

    # CALC SOURCE POWER IN LABELS & COLLECT TO DATAFRAME
    # crop the epochs to the time interval of interest (this is done in_place)
    epo.crop(tmin=tmin, tmax=tmax)
    print(epo)
    # load files for source label power calc
    fwd = mne.read_forward_solution("{}{}-fwd.fif".format(proc_dir,sub))
    all_labs = mne.read_labels_from_annot(mri, parc='aparc', subjects_dir=mri_dir, sort=True)
    labs = [l for l in all_labs if l.name in lois]
    filters = mne.beamformer.read_beamformer('{}{}_alpha-dics.h5'.format(proc_dir,sub))
    # loop over epochs, calc, collect
    for i, e in enumerate(epo):
        epoch = epo[i].copy()
        csd = mne.time_frequency.csd_morlet(epoch, frequencies=freqs, n_jobs=8, n_cycles=cycs, decim=1)
        stc, f = mne.beamformer.apply_dics_csd(csd.mean(),filters)
        ltc = mne.extract_label_time_course(stc, labs, mode='mean', src=fwd['src'])
        # write behav trial info to DF
        MEMO_dict["Subject"].append(sub)
        ev = [key for key in epoch.event_id.keys()][0]
        if 'cont' in ev:
            MEMO_dict["Illusion"].append('cont')
        else:
            MEMO_dict["Illusion"].append('break')
        if 'un' in ev:
            MEMO_dict["Sure"].append('unsure')
        else:
            MEMO_dict["Sure"].append('sure')
        for i, (l,c) in enumerate(zip(lois,l_cols)):
            MEMO_dict[c].append(ltc[i][0])      # should be 1 numpy float64 value for label power

# MAKE DATAFRAME and SAVE
MEMO = pd.DataFrame(MEMO_dict)
# convert the power values for stats: multiply by e+30, to make fT^2 out of T^2 & then take the log10 to make linear; power columns are [10:-4]
MEMO.iloc[:, 3:] = MEMO.iloc[:, 3:].mul(1e+30)
MEMO.iloc[:, 3:] = np.log10(MEMO.iloc[:, 3:])
# save
MEMO.to_csv("{}MEMO_dataframe.csv".format(save_dir), index=False)
