import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def split_dataframe_columns(df, sizes, names=None):
    """
    Splits a pandas DataFrame into column groups according to the specified sizes.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        sizes (list of int): A list of column counts for each group.
        names (list of str, optional): Optional names for the resulting groups.

    Returns:
        list or dict: A list of DataFrames (or a dict if names are provided).
    """
    splits = []
    start = 0
    for size in sizes:
        end = start + size
        splits.append(np.array(df.iloc[:, start:end]))
        start = end

    if names:
        return dict(zip(names, splits))
    return splits

# Reading the txt files into pandaframes

HSff   = pd.read_csv(os.path.join(dir_path,'Rawdata/H_An0_u+d.txt'), delim_whitespace=True, header = None)
HNSff  = pd.read_csv(os.path.join(dir_path,'Rawdata/H_An0_u-d.txt'), delim_whitespace=True, header = None)
ESff   = pd.read_csv(os.path.join(dir_path,'Rawdata/E_Bn0_u+d.txt'), delim_whitespace=True, header = None)
ENSff  = pd.read_csv(os.path.join(dir_path,'Rawdata/E_Bn0_u-d.txt'), delim_whitespace=True, header = None)
HtSff  = pd.read_csv(os.path.join(dir_path,'Rawdata/axial_H_An0_u+d.txt'), delim_whitespace=True, header = None)
HtNSff = pd.read_csv(os.path.join(dir_path,'Rawdata/axial_H_An0_u-d.txt'), delim_whitespace=True, header = None)

# Read the README.txt: the format are (|t|, A1_mean, A1_err_stat,A1_err_sys,A1_err_tot, A2...)
# Split the dataframe converted from the data file in 1, 4, 4, 4, 4, 4 for t, A1, A2, A3, A4, A5

sizes = [1,4,4,4,4,4]
names = ["t","A1","A2","A3","A4","A5"]

HSffsplit   =split_dataframe_columns(HSff, sizes, names)
HNSffsplit  =split_dataframe_columns(HNSff, sizes, names)
ESffsplit   =split_dataframe_columns(ESff, sizes, names)
ENSffsplit  =split_dataframe_columns(ENSff, sizes, names)
HtSffsplit  =split_dataframe_columns(HtSff, sizes, names)
HtNSffsplit =split_dataframe_columns(HtNSff, sizes, names)

# Convert each grouped dataframe into our standard form
# Noting that gpd type and flv need to be specified with the input moments

def DataFrame_Convert_std_Form(group, gpdtype: int, flv: str):
    
    # The array header should be in "j", "t", "mu", "f", "delta f", "GPD type", "flavor"
    # Noting a mis-match in j and n: j=0 <-> n=1
    
    tarray = - np.transpose(np.array(group["t"]))[0]
    DFj0 = pd.DataFrame({"j": [0]* len(tarray), "t": tarray, "mu": [2]*len(tarray)})
    DFj1 = pd.DataFrame({"j": [1]* len(tarray), "t": tarray, "mu": [2]*len(tarray)})
    DFj2 = pd.DataFrame({"j": [2]* len(tarray), "t": tarray, "mu": [2]*len(tarray)})
    DFj3 = pd.DataFrame({"j": [3]* len(tarray), "t": tarray, "mu": [2]*len(tarray)})
    DFj4 = pd.DataFrame({"j": [4]* len(tarray), "t": tarray, "mu": [2]*len(tarray)})
    
    DFj0["f"] = group["A1"][:,0]
    DFj0["delta f"] = group["A1"][:,-1]
    DFj0["GPD type"] = gpdtype
    DFj0["flavor"] = flv
    
    DFj1["f"] = group["A2"][:,0]
    DFj1["delta f"] = group["A2"][:,-1]
    DFj1["GPD type"] = gpdtype
    DFj1["flavor"] = flv
    
    DFj2["f"] = group["A3"][:,0]
    DFj2["delta f"] = group["A3"][:,-1]
    DFj2["GPD type"] = gpdtype
    DFj2["flavor"] = flv
    
    DFj3["f"] = group["A4"][:,0]
    DFj3["delta f"] = group["A4"][:,-1]
    DFj3["GPD type"] = gpdtype
    DFj3["flavor"] = flv
    
    DFj4["f"] = group["A5"][:,0]
    DFj4["delta f"] = group["A5"][:,-1]
    DFj4["GPD type"] = gpdtype
    DFj4["flavor"] = flv
    
    combined = pd.concat([DFj0, DFj1, DFj2,DFj3,DFj4], axis=0)
    
    return combined
    
    
stdHS   = DataFrame_Convert_std_Form(HSffsplit,  0,"S")
stdHNS  = DataFrame_Convert_std_Form(HNSffsplit, 0,"NS")
stdES   = DataFrame_Convert_std_Form(ESffsplit,  1,"S")
stdENS  = DataFrame_Convert_std_Form(ENSffsplit, 1,"NS")
stdHtS  = DataFrame_Convert_std_Form(HtSffsplit, 2,"S")
stdHtNS = DataFrame_Convert_std_Form(HtNSffsplit,2,"NS")

combine= pd.concat([stdHS, stdHNS, stdES,stdENS,stdHtS,stdHtNS], axis=0)

combine.to_csv(os.path.join(dir_path,'GFFDataLat_BNL.csv'),index=None)