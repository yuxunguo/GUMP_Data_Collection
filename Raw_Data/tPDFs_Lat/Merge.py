import numpy as np
import pandas as pd

Hpd = pd.read_csv("H_LI.csv")
Epd = pd.read_csv("E_LI.csv")
Htpd = pd.read_csv("Ht_LI.csv")

syserrH = np.array(Hpd["f"]*0.25)
staterrH = np.array(Hpd["delta f"])

syserrE = np.array(Epd["f"]*0.25)
staterrE = np.array(Epd["delta f"])

syserrHt = np.array(Htpd["f"]*0.25)
staterrHt = np.array(Htpd["delta f"])

toterrH = np.sqrt(syserrH ** 2 + staterrH **2)
toterrE = np.sqrt(syserrE ** 2 + staterrE **2)
toterrHt = np.sqrt(syserrHt ** 2 + staterrHt **2)

Hpd["delta f"] = toterrH
Epd["delta f"] = toterrE
Htpd["delta f"] = toterrHt

combined = pd.concat([Hpd, Epd, Htpd], axis=0)

combined.to_csv("tPDFdata.csv", index=False)