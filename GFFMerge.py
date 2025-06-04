import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# A script to merge different source of inputs
#ChargeFF_YAHL_DF = pd.read_csv(os.path.join(dir_path,'ChargeFFdata_YAHL.csv'))
#print(ChargeFF_YAHL_DF.shape)

# --------------------------------
# Fitted Charge Form Factor
# --------------------------------
ChargeFF_YAHL_Mod_DF = pd.read_csv(os.path.join(dir_path,'Raw_Data/ChargeFF_Fit/ChargeFFdata_YAHL_Mod.csv'))
print(ChargeFF_YAHL_Mod_DF.shape)

# --------------------------------
# lattice generalized form factors
# --------------------------------
GFFdatalat_DF = pd.read_csv(os.path.join(dir_path,'Raw_Data/GFFs_Lat_Fit/GFF_BNL/GFFDataLat_BNL.csv'))
print(GFFdatalat_DF.shape)

EtFFData_ETMC_DF = pd.read_csv(os.path.join(dir_path,'Raw_Data/GFFs_Lat_Fit/GFF_ETMC/EtFFdata_ETMC.csv'),header=None, names=['j','t','mu','f','delta f','GPD type','flavor'])
print(EtFFData_ETMC_DF.shape)

# ================================
# Merge All form factors
# ================================

merge = pd.concat([ChargeFF_YAHL_Mod_DF, GFFdatalat_DF,EtFFData_ETMC_DF], axis=0)
print(merge.shape)
merge.to_csv(os.path.join(dir_path,"GUMPDATA/GFFdata_Quark.csv"),index=None)

# ================================
# Merge Fitted PDFs and polarized PDF
# ================================

PDF_Fit_DF = pd.read_csv(os.path.join(dir_path,'Raw_Data/PDF_Fit/PDFdata50.csv'))
print(PDF_Fit_DF.shape)
PPDF_Fit_DF = pd.read_csv(os.path.join(dir_path,'Raw_Data/PDF_Fit/PPDFdata50.csv'))
print(PPDF_Fit_DF.shape)

mergePDF = pd.concat([PDF_Fit_DF, PPDF_Fit_DF], axis=0)
print(mergePDF.shape)
mergePDF.to_csv(os.path.join(dir_path,"GUMPDATA/PDFdata.csv"),index=None)