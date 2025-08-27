import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

Hprep =  pd.read_csv(os.path.join(dir_path,'Preprocess/H_prep.csv'))
Eprep =  pd.read_csv(os.path.join(dir_path,'Preprocess/E_prep.csv'))

Hprep = Hprep[(Hprep['x']>0.2) & (Hprep['x']<0.7) & (abs(Hprep['x']-Hprep['xi'])>0.2)]
Eprep = Eprep[(Eprep['x']>0.2) & (Eprep['x']<0.7) & (abs(Eprep['x']-Eprep['xi'])>0.2)]

def downsample_group(subdf, max_len):
    n = len(subdf)
    if n <= max_len:
        return subdf
    # choose evenly spaced indices
    indices = np.linspace(0, n-1, max_len, dtype=int)
    return subdf.iloc[indices]

max_len = 10

Hgrouped = Hprep.groupby(['xi', 't'], group_keys=False)  # group_keys=False to keep same DataFrame structure
H_downsampled = Hgrouped.apply(lambda g: downsample_group(g, max_len))

Egrouped = Eprep.groupby(['xi', 't'], group_keys=False)  # group_keys=False to keep same DataFrame structure
E_downsampled = Egrouped.apply(lambda g: downsample_group(g, max_len))

Hfin = pd.DataFrame({
    'x': H_downsampled['x'],
    'xi': H_downsampled['xi'],
    't': H_downsampled['t'],
    'Q': 2.0,
    'f': H_downsampled['f'],
     #'delta f': H_downsampled['delta f'],
    'delta f': np.sqrt(H_downsampled['delta f']**2 + (0.3 * H_downsampled['f'])**2),
    'GPD type': 0,
    'flavor': 'NS'
})

Efin = pd.DataFrame({
    'x': E_downsampled['x'],
    'xi': E_downsampled['xi'],
    't': E_downsampled['t'],
    'Q': 2.0,
    'f': E_downsampled['f'],
    #'delta f': E_downsampled['delta f'],
    'delta f': np.sqrt(E_downsampled['delta f']**2 + (0.3 * E_downsampled['f'])**2),
    'GPD type': 1,
    'flavor': 'NS'
})

final = pd.concat([Hfin, Efin], ignore_index=True)

final.to_csv(os.path.join(dir_path,'GPDdata.csv'), index=False)

