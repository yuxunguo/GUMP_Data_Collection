import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

AgDF = pd.read_csv(os.path.join(dir_path,'AgData.csv'),header=0, names=['t','f','delta f'])


AgDF_Form = pd.DataFrame(columns=['j','t','mu','f','delta f','GPD type','flavor'])

AgDF_Form['t'] = -AgDF['t']
AgDF_Form['f'] = AgDF['f']
AgDF_Form['delta f'] = AgDF['delta f']
AgDF_Form['j'] =  1
AgDF_Form['mu'] =  2.
AgDF_Form['GPD type'] = 0
AgDF_Form['flavor'] = 'g'
 
def DataFrame_ExtErr(DF: pd.DataFrame):
    
    ext_err = np.array(DF['f'])*0.30
    tot_err = np.array(DF['delta f'])
    Mod_err = np.sqrt(ext_err**2+tot_err**2)
    
    DF['delta f'] = Mod_err
    return DF

AgDF_Form_Mod = DataFrame_ExtErr(AgDF_Form)

AgDF_Form_Mod.to_csv(os.path.join(dir_path,'AgData_MIT_Mod.csv'),index=None)