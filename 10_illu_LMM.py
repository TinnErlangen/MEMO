## Try to do a GEE analysis on Illusion with precalculated 'random intercepts' (as centered Power Data per Subject)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# directories
# save_dir = "/home/cora/hdd/MEG/MEMO_analyses/TF_dics/"
save_dir = "D:/MEMO/"

# load the dataframe
MEMO = pd.read_csv("{}MEMO_dataframe.csv".format(save_dir))

# Variable Lists
illu = "C(Illusion, Treatment('break'))"
sure = "C(Sure, Treatment('unsure'))"
pow_vars = ["ENT_alpha","INS_alpha","LOF_alpha","MOF_alpha","OPC_alpha","ACC_alpha","STP_alpha","TPO_alpha","TTP_alpha"]

d_var = "Illusion"

# Stat prep
aicd_thresh = 4
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
# to create an ordinal IllusionGrade variable, too:
illubas = [v if v == 0 else 2 for v in MEMO_cent.Illusion]
illgrad = [(i+s) if i==2 else i for i,s in zip(illubas,MEMO_cent.Sure)]
illgrad = [1 if (i==0 and s==0) else i for i,s in zip(illgrad,MEMO_cent.Sure)]
MEMO_cent["Illgrad"] = illgrad

d_var = "Illgrad"
obsvals = MEMO_cent[d_var]
p_vars = ["ENT_alpha","INS_alpha","LOF_alpha","MOF_alpha","OPC_alpha","ACC_alpha","STP_alpha","TPO_alpha","TTP_alpha"]

# Null model
print("ANALYSES FOR {}".format(d_var))
model = "{dv} ~ 1".format(dv=d_var)
res_0 = smf.mixedlm('{}'.format(model), data=MEMO_cent, groups=MEMO_cent['Subject']).fit(reml=False)
print("Null model AIC =  ", res_0.aic)
print("Null model PseudoR2 =  ", pseudo_r2(res_0,obsvals))
null_aic = res_0.aic
last_aic = res_0.aic    # for deltas and comparisons

# Finding the Optimal Model with Power Variables
print("Finding the Optimal Model with Power Variables..")
aic_p = 0
ix = 0
aic_dict = {}
aicd_dict = {}
# First Best Variable
ix = 1
print("Iteration/Variable {}".format(ix))
model_bef = "{dv} ~ ".format(dv=d_var)
for p_var in p_vars:
    model_now = "{mb} {pv}".format(mb=model_bef, pv=p_var)
    print(model_now)
    res = smf.mixedlm('{}'.format(model_now), data=MEMO_cent, groups=MEMO_cent['Subject']).fit(reml=False)
    aic_dict[p_var] = res.aic
    aicd_dict[p_var] = last_aic - res.aic
aic_dict_t = {v: k for k,v in aic_dict.items()}
print("AICs: ", aic_dict)
print("AIC Deltas: ", aicd_dict)
best_aic = np.array(list(aic_dict.values())).min()
best_var = aic_dict_t[best_aic]
print("Best Variable is: ", best_var)
aic_p = aic_pval(best_aic, last_aic)
if aicd_dict[best_var] > aicd_thresh :
    model_bef = "{mb} {bv}".format(mb=model_bef, bv=best_var)
    p_vars.remove(best_var)
    last_aic = best_aic
    print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)

    # now Iterate until Best Model
    while aicd_dict[best_var] > aicd_thresh :
        ix = ix + 1
        print("Iteration/Variable {}".format(ix))
        for p_var in p_vars:
            model_now = "{mb} + {pv}".format(mb=model_bef, pv=p_var)
            print(model_now)
            res = smf.mixedlm('{}'.format(model_now), data=MEMO_cent, groups=MEMO_cent['Subject']).fit(reml=False)
            aic_dict[p_var] = res.aic
            aicd_dict[p_var] = last_aic - res.aic
        aic_dict_t = {v: k for k,v in aic_dict.items()}
        print("AICs: ", aic_dict)
        print("AIC Deltas: ", aicd_dict)
        best_aic = np.array(list(aic_dict.values())).min()
        best_var = aic_dict_t[best_aic]
        print("Best Variable is: ", best_var)
        aic_p = aic_pval(best_aic, last_aic)
        if aicd_dict[best_var] > aicd_thresh :
            model_bef = "{mb} + {bv}".format(mb=model_bef, bv=best_var)
            p_vars.remove(best_var)
            last_aic = best_aic
            print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)


print("Optimization for {dv} complete.".format(dv=d_var))
print("Optimal model is: ", model_bef)
res_opt = smf.mixedlm('{}'.format(model_bef), data=MEMO_cent, groups=MEMO_cent['Subject']).fit(reml=False)
print(res_opt.summary())
print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
