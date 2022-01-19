#first step - code for reading events, filtering and resampling
import numpy as np
import mne

# define basic directories for file locations (adjust to your needs)
base_dir = "/Volumes/Windows/MEMO/"
raw_dir = base_dir+"MEMO_raw/"   # where to find the original raw files
proc_dir = base_dir+"MEMO_preproc/"    # location for saving processed data files

# set up your lists for subjects and runs (BTI raw files are saved in runs in separate folders)
#subjs = ['MEM_01', 'MEM_02', 'MEM_03', 'MEM_04', 'MEM_05', 'MEM_06', 'MEM_07', 'MEM_08', 'MEM_10', 'MEM_11', 'MEM_12', 'MEM_13', 'MEM_14', 'MEM_15']
subjs = ['MEM_09']
runs = ["3"]
# create new lists, if you want to try things on single subjects or runs

# define frequencies to apply notch filter to (electricity, monitor etc.)
notches = [50, 62, 100, 150, 200]
# define breadths for the notch filters (length must match notches above)
breadths = np.array([1.5, 0.5, 0.5, 0.5, 0.5])

# looping through the subjects...
for sub in subjs:
    # get the subject path to this subjects raw data
    sub_path = raw_dir+sub+"/"
    # then loop through the run folders for this subject
    for run in runs:
        run_path = sub_path+run+"/"
        # initial reading of 4D raw data
        rawmeg = mne.io.read_raw_bti(run_path+"c,rfhp1.0Hz",preload=True,rename_channels=False)   # file name depends on filter used during recording, here 1.0Hz
        # find events from trigger channel (named 'TRIGGER' in our system) - here still in the original measurement sample rate !
        # adjust parameters as needed (starts & stops?, consecutive?, dealing with overlaps?)
        rawmeg_events = mne.find_events(rawmeg,stim_channel="TRIGGER",initial_event=True,consecutive=True,min_duration=0.004)
        # events come out as an array with 3 columns (left one holds sample number (=time), right one holds trigger value)
        # now we adjust a 4D trigger error (odd numbered triggers got added 4095 to their value - we subtract it again to get original values)
        for i_idx in range(len(rawmeg_events)):
            if rawmeg_events[i_idx,2]>4000:
                rawmeg_events[i_idx,2]=rawmeg_events[i_idx,2]-4095
        # apply the notch filter (remember, we had an online 1.0Hz low-pass filter applied already during the measurement)
        picks = mne.pick_types(rawmeg.info, meg=True,ref_meg=True)  # we do it also on ref channels, so select ref_meg=True, (for combined ICA later)
        rawmeg.notch_filter(notches,picks=picks,
        #n_jobs="cuda",
        notch_widths=breadths) #cuda is for use of GPU for processing, if not available on your machine, leave out
        # resample, giving desired sample rate; pass the event list to keep original event time points (!)
        rawmeg,rawmeg_events = rawmeg.resample(200,events=rawmeg_events
        #,n_jobs="cuda"
        ) #see above for cuda
        meg_events = rawmeg_events
        # saving the event array as a numpy file
        np.save(proc_dir+sub+"_"+run+"_events.npy",meg_events)
        # save the filtered and resampled data as -raw.fif file
        rawmeg.save(proc_dir+sub+"_"+run+"-raw.fif",overwrite=True)
