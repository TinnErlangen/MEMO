#step 5 - inspect the ICA results and select components to exclude for all files
# see detailed instructions at the end of this script

import mne
import matplotlib
matplotlib.use('Qt5Agg')
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

#dictionary with conditions/triggers for plotting
event_id = {'rest': 220, "4000/break_2_0.wav":120, "5500/break_2_0.wav":160, "7000/break_2_0.wav":140, "8500/break_2_0.wav":180, "4000/contin_2_0.wav":110, "5500/contin_2_0.wav":170, "7000/contin_2_0.wav":130, "8500/contin_2_0.wav":190}   # change to fit your data

#collecting the files : triplets of annotated raw file and corresponding reference and MEG ica result files
filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(['{dir}{sub}_{run}_m-raw.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_ref-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}_meg-ica.fif'.format(dir=proc_dir,sub=sub,run=run),
                         '{dir}{sub}_{run}-ica.fif'.format(dir=proc_dir,sub=sub,run=run)])

ref_comp_num = 8   #number of reference components to be used as 'ground'

#definition of cycler object to go through the file list for component selection and exclusion
class Cycler():

    def __init__(self,filelist,ref_comp_num):
        self.filelist = filelist
        self.ref_comp_num = ref_comp_num

        #the go() method plots all components and sources for inspection/selection
    def go(self,idx=0):
        plt.close('all')
        # load the next raw/ICA files
        self.fn = self.filelist.pop(idx)
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        self.events = np.load(self.fn[0][:-10]+"_events.npy")
        self.icaref = mne.preprocessing.read_ica(self.fn[1])
        self.icameg = mne.preprocessing.read_ica(self.fn[2])
        self.ica = mne.preprocessing.read_ica(self.fn[3])

        #housekeeping on reference components, add them to raw data
        refcomps = self.icaref.get_sources(self.raw)
        for c in refcomps.ch_names[:self.ref_comp_num]: # they need to have REF_ prefix to be recognised by MNE algorithm
            refcomps.rename_channels({c:"REF_"+c})
        self.raw.add_channels([refcomps])

        self.comps = []

        # plot everything out for overview
        self.ica.plot_components(picks=list(range(40))) #wenn zu unübersichtlich, range ändern
        self.ica.plot_sources(self.raw)
        self.raw.plot(duration=15.0,n_channels=90,scalings=dict(mag=1e-12,ref_meg=3e-12,misc=10),events=self.events,event_id=event_id)
        self.raw.plot_psd(fmax=95)

    def show_file(self):
        print("Current Raw File: " + self.fn[0])

    def identify_bad(self,method,threshold=3):        # old corr threshold was 0.5  (this one is z scores)
        # search for components which correlate with noise
        if isinstance(method,str):
            method = [method]
        elif not isinstance(method,list):
            raise ValueError('"method" must be string or list.')
        for meth in method:
            print(meth)
            if meth == "eog":
                inds, scores = self.ica.find_bads_eog(self.raw)
            elif meth == "ecg":
                inds, scores = self.ica.find_bads_ecg(self.raw)
            elif meth == "ref":
                inds, scores = self.ica.find_bads_ref(self.raw, method="separate",threshold=threshold)       # method="separate" is needed since mne vers. 0.21
            else:
                raise ValueError("Unrecognised method.")
            print(inds)
            if inds:
                self.ica.plot_scores(scores, exclude=inds)
                self.comps += inds

    def plot_props(self,props=None):
        # in case you want to take a closer look at a component
        if not props:
            props = self.comps
        self.ica.plot_properties(self.raw,props)

    def without(self,comps=None,fmax=95):
        # see what the data would look like if we took comps out
        self.comps += self.ica.exclude
        if not comps:
            comps = self.comps
        test = self.raw.copy()
        test.load_data()
        test = self.ica.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax)
        test.plot(duration=15.0,n_channels=90,scalings=dict(mag=1e-12,ref_meg=3e-12,misc=10),events=self.events,event_id=event_id)
        self.test = test

        #when saving, enter the MEG components to be excluded, bad reference components are excluded automatically
    def save(self,comps=None):
        self.comps += self.ica.exclude
        if not comps:
            self.ica.apply(self.raw,exclude=self.comps).save(self.fn[0][:-10]+'_ica-raw.fif',overwrite=True)
        elif isinstance(comps,list):
            self.ica.apply(self.epo,exclude=self.comps+comps).save(self.fn[0][:-10]+'_ica-raw.fif',overwrite=True)
        else:
            print("No components applied, saving anyway for consistency")
            self.raw.save(self.fn[0][:-10]+'_ica-raw.fif',overwrite=True)


cyc = Cycler(filelist, ref_comp_num)

#run this file from command line with '-i' for interactive mode
#then use the cyc.go() command each time to pop the next file pair in the list for component inspection and selection
#and cyc.save() with "comps =" for the ones to be excluded when done -> it will save the 'cleaned' epochs in a new file
#then cyc.go() goes on to the next file pair again... until list is empty

## more NOTES:
# running cyc.identify_bad(method="ref") will automatically put the ICs that pass the threshold into cyc.comps, which is the list of components to delete when saving
# to add more components, the following works:
# a) from the plot of ICs (plot_sources), where they are displayed like raw data in channels, click on components to exclude (like eog,ecg etc) - they will turn red,
#    and after closing the plot(!) will be automatically added to cyc.comps
# b) add components by number manually with cyc.comps += [number(s)]
# very rarely, but sometimes, the reference channels can pick up a strong alpha signal from the brain, or the identify_bad method returns components that are not "bad" after inspection
# as they are automatically added to cyc.comps by the identify_bad method, you will want to remove them manually from the list:
# do that by number with cyc.comps.remove(number) (works only for one number at a time; if needed do it once for each component to remain/not remove)
# check with typing just cyc.comps to see, if all and only the components you want to delete from the signal are in the list, then do cyc.save
