import numpy as np
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

Hpd =  pd.read_csv(os.path.join(dir_path,'Preprocess/H_LI.csv'))
Epd =  pd.read_csv(os.path.join(dir_path,'Preprocess/E_LI.csv'))
Htpd = pd.read_csv(os.path.join(dir_path,'Preprocess/Ht_LI.csv'))

syserrH = np.array(Hpd["f"]*0.3)
staterrH = np.array(Hpd["delta f"])

syserrE = np.array(Epd["f"]*0.3)
staterrE = np.array(Epd["delta f"])

syserrHt = np.array(Htpd["f"]*0.3)
staterrHt = np.array(Htpd["delta f"])

toterrH = np.sqrt(syserrH ** 2 + staterrH **2)
toterrE = np.sqrt(syserrE ** 2 + staterrE **2)
toterrHt = np.sqrt(syserrHt ** 2 + staterrHt **2)

Hpd["delta f"] = toterrH
Epd["delta f"] = toterrE
Htpd["delta f"] = toterrHt

combined = pd.concat([Hpd, Epd, Htpd], axis=0)

combined.to_csv(os.path.join(dir_path,"tPDFdata.csv"), index=False)