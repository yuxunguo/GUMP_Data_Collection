############################/
############################/
## Parameterized Form Factor Central Value and Error
############################/
## ID = 1 for GEp, 2 for GMp, 3 for GEn, 4 for GMn,
## Q2 in GeV^2
##
# The parameterization formula returns the uncertainty devided by G(0)*GD, where
#  GD(Q2) = 1./(1+Q2/0.71)^2
# and GEp(0) = 1, GMp(0) = 2.79284356, GEn(0) = 1, GMn(0) = -1.91304272,
#
# The parameterization formula for the Form Factor value is:
#  $$ GN(z) = sum_{i=0}^{N=12}(a_i * z^i)
# Note that the return value has been divided by (G(Q2=0)*G_Dip)
#
# The parameterization formula for the Form Factor error is:
# $$ log_{10}\frac{\delta G}{G_D} = (L+c_0)\Theta_a(L_1-L)
#                                 +\sum_{i=1}^{N}(c_i+d_i L)[\Theta_a(L_i-L)-\Theta_a(L_{i+1}-L)]
#                                 +log_{10}(E_{\inf})\Theta_a(L-L_{N+1})$$
# where $L=log_{10}(Q^2)$, $\Theta_{a}(x)=[1+10^{-ax}]^{-1}$. $a=1$.
import numpy as np
import pandas as pd
from math import *
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def GetFF(kID, kQ2):# {{{
    ### GEp->kID=1, GMp->kID=2, GEn->kID=3, GMn->kID=4
    if kID<1 or kID>4:
        print ('*** ERROR***, kID is not any of [1->GEp, 2->GMp, 3->GEn, 4->GMn]')
        return -1000, -1000

    #################################################
    #### z-Expansion Parameters for Form Factor Values
    #################################################{{{
    GN_Coef_Fit = np.zeros((4,13), dtype=float)
    GN_Coef_Fit[0] = np.array([0.239163298067,  -1.10985857441,  1.44438081306,  0.479569465603,  -2.28689474187,  1.12663298498,  1.25061984354,  -3.63102047159,  4.08221702379,  0.504097346499,  -5.08512046051,  3.96774254395,  -0.981529071103]) #GEp
    GN_Coef_Fit[1] = np.array([0.264142994136, -1.09530612212, 1.21855378178, 0.661136493537, -1.40567892503, -1.35641843888, 1.44702915534, 4.2356697359, -5.33404565341, -2.91630052096, 8.70740306757, -5.70699994375, 1.28081437589]) #GMp
    GN_Coef_Fit[2] = np.array([0.048919981379,-0.064525053912,-0.240825897382,0.392108744873, 0.300445258602,-0.661888687179,-0.175639769687, 0.624691724461,-0.077684299367,-0.236003975259, 0.090401973470, 0.0, 0.0]) #GEn
    GN_Coef_Fit[3] = np.array([0.257758326959,-1.079540642058, 1.182183812195,0.711015085833,-1.348080936796,-1.662444025208, 2.624354426029, 1.751234494568,-4.922300878888, 3.197892727312,-0.712072389946, 0.0, 0.0]) #GMn
    #}}}

    #################################################
    #### Parameters for Form Factor Errors
    #################################################{{{
    parL = np.zeros((4,2), dtype=float)
    parM = np.zeros((4,15), dtype=float)
    parH = np.zeros((4,3), dtype=float)
    ## GEp:
    parL[0] = np.array([-0.97775297,  0.99685273]) #Low-Q2
    parM[0] = np.array([ -1.97750308e+00,  -4.46566998e-01,   2.94508717e-01,   1.54467525e+00,
        9.05268347e-01,  -6.00008111e-01,  -1.10732394e+00,  -9.85982716e-02,
        4.63035988e-01,   1.37729116e-01,  -7.82991627e-02,  -3.63056932e-02,
        2.64219326e-03,   3.13261383e-03,   3.89593858e-04 ]) #Mid-Q2:
    parH[0] = np.array([ 0.78584754,  1.89052183, -0.4104746]) #High-Q2

    #GMp:
    parL[1] = np.array([-0.68452707,  0.99709151]) #Low-Q2
    parM[1] = np.array([ -1.76549673e+00,   1.67218457e-01,  -1.20542733e+00,  -4.72244127e-01,
        1.41548871e+00,   6.61320779e-01,  -8.16422909e-01,  -3.73804477e-01,
        2.62223992e-01,   1.28886639e-01,  -3.90901510e-02,  -2.44995181e-02,
        8.34270064e-04,   1.88226433e-03,   2.43073327e-04]) #Mid-Q2:
    parH[1] = np.array([  0.80374002,  1.98005828, -0.69700928]) #High-Q2
    
    #GEn:
    parL[2] = np.array([-2.02311829, 1.00066282]) #Low-Q2
    parM[2] = np.array([-2.07343771e+00,   1.13218347e+00,   1.03946682e+00,  -2.79708561e-01,
        -3.39166129e-01,   1.98498974e-01,  -1.45403679e-01,  -1.21705930e-01,
        1.14234312e-01,   5.69989513e-02,  -2.33664051e-02,  -1.35740738e-02,
        7.84044667e-04,   1.19890550e-03,   1.55012141e-04,]) #Mid-Q2:
    parH[2] = np.array([0.4553596, 1.95063341, 0.32421279]) #High-Q2:

    #GMn:
    parL[3] = np.array([-0.20765505, 0.99767103]) #Low-Q2:
    parM[3] = np.array([  -2.07087611e+00,   4.32385770e-02,  -3.28705077e-01,   5.08142662e-01,
        1.89103676e+00,   1.36784324e-01,  -1.47078994e+00,  -3.54336795e-01,
        4.98368396e-01,   1.77178596e-01,  -7.34859451e-02,  -3.72184066e-02,
        1.97024963e-03,   2.88676628e-03,   3.57964735e-04]) #Mid-Q2:
    parH[3] = np.array([ 0.50859057, 1.96863291, 0.2321395]) #High-Q2
    ##}}}

    ## Apply the z-expansion formula
    tcut = 0.0779191396
    t0 = -0.7
    z = (sqrt(tcut+kQ2)-sqrt(tcut-t0))/(sqrt(tcut+kQ2)+sqrt(tcut-t0)) 
    GNQ2 = np.array([GN_Coef_Fit[kID-1][i]*(z**i) for i in range(0, len(GN_Coef_Fit[kID-1]))]).sum() 
    GDip= 1./(1. + kQ2/0.71)**2
    GNGD_Fit = GNQ2 / GDip #Note that the GN_Coef_Fit has been divided by mu_p or mu_n for GMp and GMn

    ## Apply the parameterization formula for error
    lnQ2 = log10(kQ2)
    lnGNGD_Err=0.0
    if kQ2<1e-3:
        lnGNGD_Err = parL[kID-1][0] + parL[kID-1][1]*lnQ2
    elif kQ2>1e2:
        lnGNGD_Err = parH[kID-1][0]*np.sqrt(lnQ2 - parH[kID-1][1]) + parH[kID-1][2]
    else:
        lnGNGD_Err = np.array([parM[kID-1][i]*(lnQ2**i) for i in range(0, len(parM[kID-1]))]).sum() 
    GNGD_Err = 10.**(lnGNGD_Err)    ##LOG10(dG/G(0)/GD)

    return GNGD_Fit, GNGD_Err
# }}}
# proton mass
Mp = 0.938272
# array of |t|=Q^2 that we want to tabulate
t_array = np.linspace(0.01,2.01,21)

def GEGM_to_F1F2_err_propg(GE, dGE, GM, dGM, Q2, M):
    
    GE = np.asarray(GE)
    dGE = np.asarray(dGE)
    GM = np.asarray(GM)
    dGM = np.asarray(dGM)
    Q2 = np.asarray(Q2)

    tau = Q2 / (4 * M**2)

    # Central values
    F1 = (GE + tau * GM) / (1 + tau)
    F2 = (GM - GE) / (1 + tau)

    # Uncertainties
    dF1 = 1 / (1 + tau) * np.sqrt(dGE**2 + (tau * dGM)**2)
    dF2 = 1 / (1 + tau) * np.sqrt(dGM**2 + dGE**2)

    return (F1, dF1), (F2, dF2)

import numpy as np

def FpFn_to_FSFNS_err_propg(Fp, dFp, Fn, dFn):
    """
    Compute F_u+F_d and F_u-F_d with uncertainties from proton and neutron form factors.

    Parameters:
        Fp, dFp: array-like or float
            Proton form factor and its uncertainty
        Fn, dFn: array-like or float
            Neutron form factor and its uncertainty

    Returns:
        (Fud, dFud), (Fud_diff, dFud_diff)
    """
    Fp = np.asarray(Fp)
    dFp = np.asarray(dFp)
    Fn = np.asarray(Fn)
    dFn = np.asarray(dFn)

    # Sum: 3(Fp + Fn)
    FS = 3 * (Fp + Fn)
    dFS = 3 * np.sqrt(dFp**2 + dFn**2)

    # Difference: Fp - Fn
    FNS = Fp - Fn
    dFNS = np.sqrt(dFp**2 + dFn**2)

    return (FS, dFS), (FNS, dFNS)

def GDprefact(kID, kQ2):
    if kID<1 or kID>4:
        print ('*** ERROR***, kID is not any of [1->GEp, 2->GMp, 3->GEn, 4->GMn]')
        return -1000
    
    if(kID == 1 or kID == 3):
        G0 = 1
    elif(kID == 2):
        G0 =  2.79284356
    elif(kID == 4):
        G0 = -1.91304272
        
    return G0 * 1./(1+kQ2/0.71)**2
        
GEplist =  [GDprefact(1,t_i)*GetFF(1, t_i)[0] for t_i in t_array]
dGEplist = [GDprefact(1,t_i)*GetFF(1, t_i)[1] for t_i in t_array]

GMplist =  [GDprefact(2,t_i)*GetFF(2, t_i)[0] for t_i in t_array]
dGMplist = [GDprefact(2,t_i)*GetFF(2, t_i)[1] for t_i in t_array]

GEnlist =  [GDprefact(3,t_i)*GetFF(3, t_i)[0] for t_i in t_array]
dGEnlist = [GDprefact(3,t_i)*GetFF(3, t_i)[1] for t_i in t_array]

GMnlist =  [GDprefact(4,t_i)*GetFF(4, t_i)[0] for t_i in t_array]
dGMnlist = [GDprefact(4,t_i)*GetFF(4, t_i)[1] for t_i in t_array]

'''
for i in range(len(t_array)):
    print(f"Q² = {t_array[i]:.2f} GeV²: GEp = {GEplist[i]:.5f} ± {dGEplist[i]:.5f}, GMp = {GMplist[i]:.5f} ± {dGMplist[i]:.5f}")
print('***********')
for i in range(len(t_array)):
    print(f"Q² = {t_array[i]:.2f} GeV²: GEn = {GEnlist[i]:.5f} ± {dGEnlist[i]:.5f}, GMn = {GMnlist[i]:.5f} ± {dGMnlist[i]:.5f}")
'''
(F1p, dF1p), (F2p, dF2p) = GEGM_to_F1F2_err_propg(GEplist, dGEplist, GMplist, dGMplist, t_array, Mp)
(F1n, dF1n), (F2n, dF2n) = GEGM_to_F1F2_err_propg(GEnlist, dGEnlist, GMnlist, dGMnlist, t_array, Mp)

(F1S, dF1S), (F1NS, dF1NS) = FpFn_to_FSFNS_err_propg(F1p, dF1p, F1n, dF1n)
(F2S, dF2S), (F2NS, dF2NS) = FpFn_to_FSFNS_err_propg(F2p, dF2p, F2n, dF2n)
'''
for i in range(len(F1p)):
    print(f"Q² = {t_array[i]:.2f} GeV²: F1S = {F1S[i]:.5f} ± {dF1S[i]:.5f}, F1NS = {F1NS[i]:.5f} ± {dF1NS[i]:.5f}")
print('***********')
for i in range(len(F1p)):
    print(f"Q² = {t_array[i]:.2f} GeV²: F2S = {F2S[i]:.5f} ± {dF2S[i]:.5f}, F2NS = {F2NS[i]:.5f} ± {dF2NS[i]:.5f}")
'''

dF1S_Mod  = np.maximum(F1S/20, dF1S)
dF1NS_Mod = np.maximum(F1NS/20,dF1NS)
dF2S_Mod  = np.maximum(F2S/20, dF2S)
dF2NS_Mod = np.maximum(F2NS/20,dF2NS)

def convert_to_std_form(f: np.array, df: np.array, gpdtype: int, flv: str):

    f_init = pd.DataFrame({"j": [0]* len(t_array), "t": -t_array, "mu": [2]*len(t_array)})
    f_init["f"] = f
    f_init["delta f"] = df
    f_init["GPD type"] = gpdtype
    f_init["flavor"] = flv
    return f_init

F1Sdf  = convert_to_std_form(F1S,  dF1S,  0, "S")
F1NSdf = convert_to_std_form(F1NS, dF1NS, 0, "NS")
F2Sdf  = convert_to_std_form(F2S,  dF2S,  1, "S")
F2NSdf = convert_to_std_form(F2NS, dF2NS, 1, "NS")

combine= pd.concat([F1Sdf, F1NSdf, F2Sdf,F2NSdf], axis=0)
combine.to_csv(os.path.join(dir_path,'ChargeFFdata_YAHL.csv'),index=None)

F1Sdf_Mod  = convert_to_std_form(F1S,  dF1S_Mod,  0, "S")
F1NSdf_Mod = convert_to_std_form(F1NS, dF1NS_Mod, 0, "NS")
F2Sdf_Mod  = convert_to_std_form(F2S,  dF2S_Mod,  1, "S")
F2NSdf_Mod = convert_to_std_form(F2NS, dF2NS_Mod, 1, "NS")

combine_Mod= pd.concat([F1Sdf_Mod, F1NSdf_Mod, F2Sdf_Mod,F2NSdf_Mod], axis=0)
combine_Mod.to_csv(os.path.join(dir_path,'ChargeFFdata_YAHL_Mod.csv'),index=None)