## Do Power Network Analysis with LMMs and a GEE binomial analysis on illusion outcome

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

# Building Otimal Models for each Power Node
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

    # Experimental variables
    print("Testing Illusion Effect..")
    model = "{dv} ~ {i}".format(dv=d_var, i=illu)
    res_illu = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    illu_aic = res_illu.aic
    print("Illu model results -- AIC = ", illu_aic, ", AIC_delta = ", null_aic - illu_aic, ", AIC_p = ", aic_pval(illu_aic,null_aic), ", PseudoR2 = ", pseudo_r2(res_illu,obsvals))
    print("Trying to add Sureness..")
    model = "{dv} ~ {i} + {s}".format(dv=d_var, i=illu, s=sure)
    res_sure = smf.mixedlm('{}'.format(model), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    sure_aic = res_sure.aic
    print("Illu + Sure model results -- AIC = ", sure_aic, ", AIC_delta = ", illu_aic - sure_aic, ", AIC_p = ", aic_pval(sure_aic,illu_aic), ", PseudoR2 = ", pseudo_r2(res_sure,obsvals))

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
        res = smf.mixedlm('{}'.format(model_now), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
        aic_dict[p_var] = res.aic
        aicd_dict[p_var] = last_aic - res.aic
    aic_dict_t = {v: k for k,v in aic_dict.items()}
    print("AICs: ", aic_dict)
    print("AIC Deltas: ", aicd_dict)
    best_aic = np.array(list(aic_dict.values())).min()
    best_var = aic_dict_t[best_aic]
    print("Best Variable is: ", best_var)
    aic_p = aic_pval(best_aic, last_aic)
    if aic_p < 0.05 :
        model_bef = "{mb} {bv}".format(mb=model_bef, bv=best_var)
        p_vars.remove(best_var)
        last_aic = best_aic
        print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
        # TEST Illusion
        res_illu = smf.mixedlm('{mb} + {i}'.format(mb=model_bef, i=illu), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
        illu_aic = res_illu.aic
        illu_aicp = aic_pval(illu_aic,last_aic)
        print("Testing Illusion again -- AIC = ", illu_aic, ", AIC_delta = ", last_aic - illu_aic, ", AIC_p = ", illu_aicp)
        if illu_aicp < 0.05:
            print("Illusion stays in the model.")
        else:
            print("No independent illusion effect remains.")

        # now Iterate until Best Model
        while aic_p < 0.05 :
            ix = ix + 1
            print("Iteration/Variable {}".format(ix))
            for p_var in p_vars:
                model_now = "{mb} + {pv}".format(mb=model_bef, pv=p_var)
                print(model_now)
                res = smf.mixedlm('{}'.format(model_now), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
                aic_dict[p_var] = res.aic
                aicd_dict[p_var] = last_aic - res.aic
            aic_dict_t = {v: k for k,v in aic_dict.items()}
            print("AICs: ", aic_dict)
            print("AIC Deltas: ", aicd_dict)
            best_aic = np.array(list(aic_dict.values())).min()
            best_var = aic_dict_t[best_aic]
            print("Best Variable is: ", best_var)
            aic_p = aic_pval(best_aic, last_aic)
            if aic_p < 0.05 :
                model_bef = "{mb} + {bv}".format(mb=model_bef, bv=best_var)
                p_vars.remove(best_var)
                last_aic = best_aic
                print("Current best model: ", model_bef, ", AIC = ", best_aic, ", AIC_delta = ", aicd_dict[best_var], ", AIC_p = ", aic_p)
                # TEST Illusion
                res_illu = smf.mixedlm('{mb} + {i}'.format(mb=model_bef, i=illu), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
                illu_aic = res_illu.aic
                illu_aicp = aic_pval(illu_aic,last_aic)
                print("Testing Illusion again -- AIC = ", illu_aic, ", AIC_delta = ", last_aic - illu_aic, ", AIC_p = ", illu_aicp)
                if illu_aicp < 0.05:
                    print("Illusion stays in the model.")
                else:
                    print("No independent illusion effect remains.")

    print("Optimization for {dv} complete.".format(dv=d_var))
    print("Optimal model is: ", model_bef)
    res_opt = smf.mixedlm('{}'.format(model_bef), data=MEMO, groups=MEMO['Subject']).fit(reml=False)
    print(res_opt.summary())
    print("Opt Model PseudoR2 = ", pseudo_r2(res_opt,obsvals))
