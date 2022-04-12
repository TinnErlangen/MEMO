## Do Power Network Analysis with LMMs - Part2 - try full models to compare, Illu*Sure interaction,

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

# Building Full Models for each Power Node to assess
for pow_var in pow_vars:
    d_var = pow_var
    obsvals = MEMO[d_var]
    p_vars = ["ENT_alpha","INS_alpha","LOF_alpha","MOF_alpha","OPC_alpha","ACC_alpha","STP_alpha","TPO_alpha","TTP_alpha"]
    p_vars.remove(d_var)

    # Null model
    print("ANALYSES FOR {}".format(d_var))
    model = "{dv} ~ 1".format(dv=d_var)
    res_0 = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    print("Null model AIC =  ", res_0.aic)
    print("Null model PseudoR2 =  ", pseudo_r2(res_0,obsvals))
    null_aic = res_0.aic
    last_aic = res_0.aic    # for deltas and comparisons

    # Behavioral variables
    print("Testing Illusion Effect..")
    model = "{dv} ~ {i} * {s}".format(dv=d_var, i=illu, s=sure)
    res_illu = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    illu_aic = res_illu.aic
    print("Illu model results -- AIC = ", illu_aic, ", AIC_delta = ", null_aic - illu_aic, ", AIC_p = ", aic_pval(illu_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_illu,obsvals))
    print(res_illu.summary())

    # Doing the Full Model with Power Variables only
    print("Doing the Full Model with Power Variables..")
    model = "{dv} ~ {p1} + {p2} + {p3} + {p4} + {p5} + {p6} + {p7} + {p8}".format(dv=d_var, p1=p_vars[0], p2=p_vars[1], p3=p_vars[2], p4=p_vars[3],
                                                                                  p5=p_vars[4], p6=p_vars[5], p7=p_vars[6], p8=p_vars[7])
    res_full = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    full_aic = res_full.aic
    print("Full model results -- AIC = ", full_aic, ", AIC_delta = ", null_aic - full_aic, ", AIC_p = ", aic_pval(full_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_full,obsvals))
    print(res_full.summary())

    # Doing the Full Model with Behavioral and Power Variables
    print("Doing the Full Illu Model with Behavioral and  Power Variables..")
    model = "{dv} ~ {i} * {s} * ({p1} + {p2} + {p3} + {p4} + {p5} + {p6} + {p7} + {p8})".format(dv=d_var, i=illu, s=sure, p1=p_vars[0], p2=p_vars[1], p3=p_vars[2], p4=p_vars[3],
                                                                                                p5=p_vars[4], p6=p_vars[5], p7=p_vars[6], p8=p_vars[7])
    res_ifull = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    ifull_aic = res_ifull.aic
    print("Full Illu model results -- AIC = ", ifull_aic, ", AIC_delta_tonull = ", null_aic - ifull_aic, ", AIC_delta_tofull = ", full_aic - ifull_aic, ", AIC_p = ", aic_pval(ifull_aic,full_aic), ", PseudoR2 = ", pseudo_r2(res_ifull,obsvals))
    print(res_ifull.summary())
