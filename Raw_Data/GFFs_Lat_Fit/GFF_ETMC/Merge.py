import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

EtFFData_ETMC_DF = pd.read_csv(os.path.join(dir_path,'EtFFdata_ETMC.csv'),header=None, names=['j','t','mu','f','delta f','GPD type','flavor'])

def DataFrame_ExtErr(DF: pd.DataFrame):
    
    ext_err = np.array(DF['f'])*0.30
    tot_err = np.array(DF['delta f'])
    Mod_err = np.sqrt(ext_err**2+tot_err**2)
    
    DF['delta f'] = Mod_err
    return DF

EtFFData_ETMC_DF_mod = DataFrame_ExtErr(EtFFData_ETMC_DF)

EtFFData_ETMC_DF_mod.to_csv(os.path.join(dir_path,'EtFFdata_ETMC_Mod.csv'),index=None,header=None)