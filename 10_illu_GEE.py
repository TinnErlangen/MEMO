## Try to do a GEE analysis on Illusion with precalculated 'random intercepts' (as centered Power Data per Subject)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# directories
save_dir = "/home/cora/hdd/MEG/MEMO_analyses/TF_dics/"

# load the dataframe
MEMO = pd.read_csv("{}MEMO_dataframe.csv".format(save_dir))

# Variable Lists
illu = "C(Illusion, Treatment('break'))"
sure = "C(Sure, Treatment('unsure'))"
pow_vars = ["ENT_alpha","INS_alpha","LOF_alpha","MOF_alpha","OPC_alpha","ACC_alpha","STP_alpha","TPO_alpha","TTP_alpha"]

# Stat prep
# aicd_thresh = 5
def aic_pval(a,b):
    return np.exp((a - b)/2)   # calculates the evidence ratio for 2 aic values
def pseudo_r2(m, y):    # calculates the pseudo r2 from model summary(m) and observed vals (y)
    fitvals = m.fittedvalues
    r , v = stats.pearsonr(fitvals, y)
    return r**2

MEMO_cent = MEMO.copy()
MEMO_cent.replace(to_replace={'break':0,'cont':1,'unsure':0,'sure':1}, inplace=True)
# we make a small DF containing the means per Subject for the Power Values, Subject Name becomes the Index
means = MEMO_cent.groupby('Subject').mean()
subjs = MEMO_cent.Subject.unique()
# now we can loop through Subjects and Power Variables, to subtract the respective means from the Subjects' Trial Values in the original DF
new_df = {}
for s in subjs:
    for pv in pow_vars:
        MEMO_cent.loc[MEMO_cent.Subject == s, pv] = MEMO_cent.loc[MEMO_cent.Subject == s, pv].sub(means.loc[s,pv])

model = sm.GEE.from_formula("Illusion ~ ENT_alpha + INS_alpha + LOF_alpha + MOF_alpha + OPC_alpha + ACC_alpha + STP_alpha + TPO_alpha + TTP_alpha",
                            groups="Subject", family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(), data=MEMO_cent)
