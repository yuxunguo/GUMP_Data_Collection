#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 13:20:00 2025

@author: fpaslan
"""
                                       
        
import gpddatabase
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))



                                         ###### CROSS SECTION ########






"""

  DATASET 1
  uuid: TKhscLcB
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU  
  
"""
# Constants
E_lepton = 3.355  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('TKhscLcB')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/1_CS_TKhscLcB.csv"), index=False, header=False)

print("Final file written to 1_CS_TKhscLcB.csv")

#############################################################################################################

"""

  DATASET 2
  uuid: AtY8o7Ej
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU 
  
"""
# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AtY8o7Ej')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows_uu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat0_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat0_val = stat0_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst0_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst0_unc.get_unc_upper()
    b = syst0_unc.get_unc_lower()
    syst0_val = max(a, b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_uu = np.sqrt(stat0_val**2 + syst0_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row_uu = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f_uu": delta_f_uu,
        "pol": "UU"
    }

    rows_uu.append(row_uu)

# Saving to Excel
df_uu = pd.DataFrame(rows_uu)
df_uu.to_csv(os.path.join(dir_path,"DVCSoutput/2_CS_AtY8o7Ej_UU.csv"), index=False, header=False)

print("Final file written to 2_CS_AtY8o7Ej_UU.csv")

#############################################################################################################


"""

  DATASET 3
  uuid: msa6dh9v
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU 
  
"""
# Constants
E_lepton = 5.55  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('msa6dh9v')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/3_CS_msa6dh9v.csv"), index=False, header=False)

print("Final file written to 3_CS_msa6dh9v.csv")


#############################################################################################################

"""

  DATASET 4
  uuid: bmTzHHvg
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.55  # GeV
E_hadron = 0.9395654205  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('bmTzHHvg')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a,b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Compute y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/4_CS_bmTzHHvg.csv"), index=False, header=False)

print("Final file written to 4_CS_bmTzHHvg.csv")

#############################################################################################################


"""

  DATASET 5
  uuid: RQncbKtk
  collaboration: CLAS 
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.75  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('RQncbKtk')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU 
rows_uu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
    
    # UU
    stat0 = stat_unc.get_uncertainty(0)
    sys0 = sys_unc.get_uncertainty(0)
    delta_f_uu = np.sqrt(stat0.get_unc()**2 + sys0.get_unc()**2) 
    row_uu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "f": obs_dict[obs_names[0]],
            "delta_f": delta_f_uu,
            "pol": "UU"
        }
    rows_uu.append(row_uu)

    
# Exporting UU data
df_uu = pd.DataFrame(rows_uu)
df_uu.to_csv(os.path.join(dir_path,"DVCSoutput/5_CS_RQncbKtk_UU.csv"), index=False, header=False)

print("Final file written to 5_CS_RQncbKtk_UU.csv")

#############################################################################################################

"""

  DATASET 6
  uuid: mJXCLi4G
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
  
"""
# Constants
E_lepton = 4.455  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('mJXCLi4G')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "UU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/6_CS_mJXCLi4G.csv"), index=False, header=False)


print("Final file written to 6_CS_mJXCLi4G.csv")

#############################################################################################################

"""

  DATASET 7
  uuid: Cb6meE7Q
  collaboration: HallA
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 4.45  # GeV
E_hadron = 0.9395654205  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('Cb6meE7Q')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a, b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/7_CS_Cb6meE7Q.csv"), index=False, header=False)

print("Final file written to 7_CS_Cb6meE7Q.csv")

#############################################################################################################


"""

  DATASET 8
  uuid: ob8hLTm2
  collaboration: CLASS
  type: DVCS
  observables: CrossSectionUU
  
"""
# Constants
E_lepton = 5.88  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('ob8hLTm2')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Convert degrees to radians
        "f": obs_dict[obs_names[0]],   
        "delta_f": delta_f,
        "pol": "UU"  # Add polarization info
    }

    # Append to the list!
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/8_CS_ob8hLTm2.csv"), index=False, header=False)

print("Final file written to 8_CS_ob8hLTm2.csv")





#############################################################################################################





# List of all the file names
file_names = [
    "1_CS_TKhscLcB.csv",
    "2_CS_AtY8o7Ej_UU.csv",
    "3_CS_msa6dh9v.csv",
    "4_CS_bmTzHHvg.csv",
    "5_CS_RQncbKtk_UU.csv",
    "6_CS_mJXCLi4G.csv",
    "7_CS_Cb6meE7Q.csv",
    "8_CS_ob8hLTm2.csv"
]

dfs = []
for file in file_names:
    # Extract identifier (remove prefix and extension)
    identifier = file.split("_", 2)[-1].replace(".csv", "")
    
    # Load data
    df = pd.read_csv(os.path.join(dir_path, "DVCSoutput", file), header=None)
    
    # Add comment column
    df["comment"] = identifier
    
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to new Excel
output_path = os.path.join(dir_path, "DVCSoutput", "CS_Combined.csv")
combined_df.to_csv(output_path, index=False, header=False)

print(f"Combined Cross sections file saved as {output_path}")

#############################################################################################################
#############################################################################################################

                                               ###### t- dependent CROSS SECTION ########



"""

  DATASET 9
  uuid: 75ueQoQw
  collaboration: COMPASS
  type: DVCS
  observables: CrossSectionUUVirtualPhotoProduction
  
"""

# Constants
E_lepton = 160  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('75ueQoQw')
data = ob.get_data()
data_set = data.get_data_set('t_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = np.sqrt(a**2 + b**2)

    # Combining stat + syst uncertainty in quadrature
    delta_f = np.sqrt(stat_val**2 + syst_val**2) 

    # Computing y
    Q2 = kin_dict["Q2"]
    nu = kin_dict["nu"]
    xB =Q2 / (2* E_hadron* nu)
    y = Q2 / (2 * xB * E_lepton * E_hadron)
   

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "f": obs_dict[obs_names[0]],
        "delta_f": delta_f,
        "pol": "UU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/9_CS_75ueQoQw.csv"), index=False, header=False)

print("Final file written to 9_CS_75ueQoQw.csv")









                                                     ###### CROSS SECTION DIFFERENCE ########



"""

  DATASET 1
  uuid: EqbtDRkv
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 3.355  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('EqbtDRkv')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/1_CSD_EqbtDRkv.csv"), index=False, header=False)


print("Final file written to 1_CSD_EqbtDRkv.csv")

#############################################################################################################

"""

  DATASET 2
  uuid: nfPvTM2c
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 4.455  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('nfPvTM2c')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]] * 1e-3,   # Converting pb to nb],
        "delta_f": delta_f,
        "pol": "LU"  # Add polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/2_CSD_nfPvTM2c.csv"), index=False, header=False)


print("Final file written to 2_CSD_nfPvTM2c.csv")


#############################################################################################################



"""

  DATASET 3
  uuid: AtY8o7Ej
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""

# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AtY8o7Ej')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')



# Preparing data
rows_lu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat1_unc = point.get_observables_stat_uncertainties().get_uncertainty(1)
    stat1_val = stat1_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst1_unc = point.get_observables_sys_uncertainties().get_uncertainty(1)
    c = syst1_unc.get_unc_upper()
    d = syst1_unc.get_unc_lower()
    syst1_val = max(c, d)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f_lu = np.sqrt(stat1_val**2 + syst1_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row_lu = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[1]] * 1e-3,  # pb → nb
        "delta_f_lu": delta_f_lu,
        "pol": "LU"
    }

    rows_lu.append(row_lu)

# Saving to Excel
df_lu = pd.DataFrame(rows_lu)
df_lu.to_csv(os.path.join(dir_path,"DVCSoutput/3_CSD_AtY8o7Ej_LU.csv"), index=False, header=False)


print("Final file written to 3_CSD_AtY8o7Ej_LU.csv")






"""

  DATASET 4
  uuid: RQncbKtk
  collaboration: CLAS 
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""
# Constants
E_lepton = 5.75  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('RQncbKtk')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU and LU data separately

rows_lu = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
   
    # LU
    stat1 = stat_unc.get_uncertainty(1)
    sys1 = sys_unc.get_uncertainty(1)
    delta_f_lu = np.sqrt(stat1.get_unc()**2 + sys1.get_unc()**2)
    row_lu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "f": obs_dict[obs_names[1]],
            "delta_f": delta_f_lu,
            "pol": "LU"
        }
    rows_lu.append(row_lu)



# Exporting LU data
df_lu = pd.DataFrame(rows_lu)
df_lu.to_csv(os.path.join(dir_path,"DVCSoutput/4_CSD_RQncbKtk_LU.csv"), index=False, header=False)

print("Final file written to 4_CSD_RQncbKtk_LU.csv")


#############################################################################################################




"""

  DATASET 5
  uuid: BJ84iv8s
  collaboration: HallA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""

# Constants
E_lepton = 5.7572  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('BJ84iv8s')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))

    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Statistical uncertainty (symmetric)
    stat_unc = point.get_observables_stat_uncertainties().get_uncertainty(0)
    stat_val = stat_unc.get_unc()

    # Systematic uncertainty (asymmetric)
    syst_unc = point.get_observables_sys_uncertainties().get_uncertainty(0)
    a = syst_unc.get_unc_upper()
    b = syst_unc.get_unc_lower()
    syst_val = max(a,b)

    # Combining stat + syst uncertainty in quadrature, convert from pb → nb
    delta_f = np.sqrt(stat_val**2 + syst_val**2) * 1e-3

    # Computing y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating data row
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),
        "f": obs_dict[obs_names[0]] * 1e-3,  # pb → nb
        "delta_f": delta_f,
        "pol": "LU"
    }

    rows.append(row)

# Saving to Excel
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/5_CSD_BJ84iv8s.csv"), index=False, header=False)

print("Final file written to 5_CSD_BJ84iv8s.csv")




#############################################################################################################


"""

  DATASET 6
  uuid: EhPp8CP4
  collaboration: HALLA
  type: DVCS
  observables: CrossSectionDifferenceLU
  
"""


# Constants
E_lepton =  5.55  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('EhPp8CP4')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_f = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) * 1e-3 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "f": obs_dict[obs_names[0]]* 1e-3,   # Converting pb to nb],   
        "delta_f": delta_f,
        "pol": "LU"  # Add polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/6_CSD_EhPp8CP4.csv"), index=False, header=False)


print("Final file written to 6_CSD_EhPp8CP4.csv")












# List of all the Cross Section Difference LU Files
file_list = [
    "1_CSD_EqbtDRkv.csv",
    "2_CSD_nfPvTM2c.csv",
    "3_CSD_AtY8o7Ej_LU.csv",
    "4_CSD_RQncbKtk_LU.csv",
    "5_CSD_BJ84iv8s.csv",
    "6_CSD_EhPp8CP4.csv"
]

dfs = []
for file in file_list:
    # Extract identifier (everything after the second underscore, remove extension)
    identifier = file.split("_", 2)[-1].replace(".csv", "")
    
    # Load data
    df = pd.read_csv(os.path.join(dir_path, "DVCSoutput", file), header=None)
    
    # Add comment column
    df["comment"] = identifier
    
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to new Excel
output_path = os.path.join(dir_path, "DVCSoutput", "CSD_Combined.csv")
combined_df.to_csv(output_path, index=False, header=False)

print(f"Combined Cross section Difference file saved as {output_path}")






                                                                    ###### ASYMMETRIES ########

i


"""

  DATASET 1
  uuid: A6Pgo5TE
  collaboration: CLAS
  type: DVCS
  observables: ALU Q2 Dependent
  
"""

# Constants
E_lepton = 4.8  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('A6Pgo5TE')
data = ob.get_data()
data_set = data.get_data_set('Q2_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc() 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/1_ALU_A6Pgo5TE_Q2.csv"), index=False, header=False)


print("Final file written to 1_ALU_A6Pgo5TE_Q2.csv")

###########################################################################################################

"""

  DATASET 2
  uuid: A6Pgo5TE
  collaboration: CLAS
  type: DVCS
  observables: ALU t Dependent
  
"""
# Constants
E_lepton = 4.8  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('A6Pgo5TE')
data = ob.get_data()
data_set = data.get_data_set('t_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc() 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/2_ALU_A6Pgo5TE_t.csv"), index=False, header=False)


print("Final file written to 2_ALU_A6Pgo5TE_t.csv")


###########################################################################################################

"""

  DATASET 3
  uuid: NvMm42PD
  collaboration: HERMES
  type: DVCS
  observables: ALU
  
"""
# Constants
E_lepton = 27.6  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('NvMm42PD')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/3_ALU_NvMm42PD.csv"), index=False, header=False)


print("Final file written to 3_ALU_NvMm42PD.csv")

###########################################################################################################


"""

  DATASET 4
  uuid: QfefWWW2
  collaboration: CLAS
  type: DVCS
  observables: ALL
  
"""
# Constants
E_lepton = 10.6  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('QfefWWW2')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc()

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": kin_dict["phi"],  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/4_ALL_QfefWWW2.csv"), index=False, header=False)


print("Final file written to 4_ALL_QfefWWW2.csv")


###########################################################################################################


"""

  DATASET 5
  uuid: PusMstKs
  collaboration: CLAS
  type: DVCS
  observables: ALL
  
"""
# Constants
E_lepton = 10.2  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('PusMstKs')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    #syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = stat.get_unc()

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": kin_dict["phi"],  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and exporting
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/5_ALL_PusMstKs.csv"), index=False, header=False)


print("Final file written to 5_ALL_PusMstKs.csv")



###########################################################################################################


"""

  DATASET 6
  uuid: AvF5daeP
  collaboration: HERMES
  type: DVCS
  observables: ALU
  
"""
# Constants
E_lepton = 5.77  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset
ob = db.get_data_object('AvF5daeP')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Preparing data
rows = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)
    
    # Kinematics
    kin_names = point.get_kinematics_names()
    kin_vals = point.get_kinematics_values()
    kin_dict = dict(zip(kin_names, kin_vals))
    
    # Observables
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    # Calculating delta_f (combined stat + syst uncertainty)
    stat = point.get_observables_stat_uncertainties().get_uncertainty(0)
    syst = point.get_observables_sys_uncertainties().get_uncertainty(0)
    delta_A = np.sqrt(stat.get_unc()**2 + syst.get_unc()**2) 

    # Calculating y
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Creating ordered row 
    row = {
        "y": y,
        "xB": xB,
        "t": kin_dict["t"],
        "Q": np.sqrt(Q2),
        "phi": np.radians(kin_dict["phi"]),  # Converting degrees to radians
        "A": obs_dict[obs_names[0]],
        "delta_A": delta_A,
        "pol": "LU"  # Adding polarization info
    }

    # Appending to the list
    rows.append(row)

# Converting to DataFrame and export
df = pd.DataFrame(rows)
df.to_csv(os.path.join(dir_path,"DVCSoutput/6_ALU_AvF5daeP.csv"), index=False, header=False)


print("Final file written to 6_ALU_AvF5daeP.csv")


###########################################################################################################


"""

  DATASET 7
  uuid: vGAKAf7P
  collaboration: CLAS 
  type: DVCS
  observables: ALU, AUL, ALL
  
"""
# Constants
E_lepton = 5.932  # GeV
E_hadron = 0.93827208816  # GeV

# Creating database object
db = gpddatabase.ExclusiveDatabase()

# Loading the dataset from the database
ob = db.get_data_object('vGAKAf7P')
data = ob.get_data()
data_set = data.get_data_set('phi_dep')

# Listing to store UU and LU data separately
rows_lu = []
rows_ul = []
rows_ll = []

for i in range(data_set.get_number_of_data_points()):
    point = data_set.get_data_point(i)

    # Kinematics
    kin_dict = dict(zip(point.get_kinematics_names(), point.get_kinematics_values()))
    Q2 = kin_dict["Q2"]
    xB = kin_dict["xB"]
    y = Q2 / (2 * xB * E_lepton * E_hadron)

    # Observable values and uncertainties
    obs_names = point.get_observables_names()
    obs_vals = point.get_observables_values()
    obs_dict = dict(zip(obs_names, obs_vals))

    stat_unc = point.get_observables_stat_uncertainties()
    sys_unc = point.get_observables_sys_uncertainties()

    # Observable 0 is CrossSectionUU and 1 is CrossSectionDifferenceLU
    
    # LU
    stat0 = stat_unc.get_uncertainty(0)
    sys0 = sys_unc.get_uncertainty(0)
    delta_ALU = np.sqrt(stat0.get_unc()**2 + sys0.get_unc()**2) 
    row_lu = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "ALU": obs_dict[obs_names[0]],
            "delta_ALU": delta_ALU,
            "pol": "LU"
        }
    rows_lu.append(row_lu)

    # UL
    stat1 = stat_unc.get_uncertainty(1)
    sys1 = sys_unc.get_uncertainty(1)
    delta_AUL = np.sqrt(stat1.get_unc()**2 + sys1.get_unc()**2)
    row_ul = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "AUL": obs_dict[obs_names[1]],
            "delta_AUL": delta_AUL,
            "pol": "UL"
        }
    rows_ul.append(row_ul)
    
    # LL
    stat2 = stat_unc.get_uncertainty(2)
    sys2 = sys_unc.get_uncertainty(2)
    delta_ALL = np.sqrt(stat2.get_unc()**2 + sys2.get_unc()**2)
    row_ll = {
            "y": y,
            "xB": xB,
            "t": kin_dict["t"],
            "Q": np.sqrt(Q2),
            "phi": np.radians(kin_dict["phi"]),
            "ALL": obs_dict[obs_names[2]],
            "delta_ALL": delta_ALL,
            "pol": "LL"
        }
    rows_ll.append(row_ll)


# Exporting LU data
df_lu = pd.DataFrame(rows_lu)
df_lu.to_csv(os.path.join(dir_path,"DVCSoutput/7.1_ALU_vGAKAf7P.csv"), index=False, header=False)

# Exporting UL data
df_ul = pd.DataFrame(rows_ul)
df_ul.to_csv(os.path.join(dir_path,"DVCSoutput/7.2_AUL_vGAKAf7P.csv"), index=False, header=False)

# Exporting LL data
df_ll = pd.DataFrame(rows_ll)
df_ll.to_csv(os.path.join(dir_path,"DVCSoutput/7.3_ALL_vGAKAf7P.csv"), index=False, header=False)

print("ALU, AUL, ALL files written to 7.1_ALU_vGAKAf7P.csv and 7.2_AUL_vGAKAf7P.csv and 7.3_ALL_vGAKAf7P.csv")





file_list = [
    "1_ALU_A6Pgo5TE_Q2.csv",
    "2_ALU_A6Pgo5TE_t.csv",
    "3_ALU_NvMm42PD.csv",
    "4_ALL_QfefWWW2.csv",
    "5_ALL_PusMstKs.csv",
    "6_ALU_AvF5daeP.csv",
    "7.1_ALU_vGAKAf7P.csv",
    "7.2_AUL_vGAKAf7P.csv",
    "7.3_ALL_vGAKAf7P.csv"
]

dfs = []
for file in file_list:
    # Extract identifier (everything after the second underscore, remove .csv)
    identifier = file.split("_", 2)[-1].replace(".csv", "")
    
    # Load data
    df = pd.read_csv(os.path.join(dir_path, "DVCSoutput", file), header=None)
    
    # Add comment column
    df["comment"] = identifier
    
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save to new Excel
output_path = os.path.join(dir_path, "DVCSoutput", "ASYMMETRIES_combined.csv")
combined_df.to_csv(output_path, index=False, header=False)

print(f"Combined asymmetry file saved as {output_path}")