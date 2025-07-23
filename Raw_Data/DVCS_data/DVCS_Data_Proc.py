import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
    
dir_path = os.path.dirname(os.path.realpath(__file__))
Q_threshold = 1.9
xB_Cut = 0.5
M=0.938

DVCS_names = ["y","xB","t","Q","phi","f","delta f","pol"]
cols_to_check = ['y', 'xB', 't', 'Q', 'phi', 'f', 'pol']

# ============================================================
# Read from the DVCS including cross-sections and asymmetries
# ============================================================

DVCS_xsec_unp = pd.read_excel(os.path.join(dir_path,'DVCSoutput/CS_Combined.xlsx'),names=DVCS_names)
DVCS_xsec_pol = pd.read_excel(os.path.join(dir_path,'DVCSoutput/CSD_Combined.xlsx'),names=DVCS_names)
DVCS_asymmetry = pd.read_excel(os.path.join(dir_path,'DVCSoutput/ASYMMETRIES_combined.xlsx'),names=DVCS_names)

# ============================================================
# Merge polarized and unpolaried cross-sections and output
# ============================================================

DVCS_xsec_comb = pd.concat([DVCS_xsec_unp, DVCS_xsec_pol], ignore_index=True)
DVCS_xsec_comb.to_csv(os.path.join(dir_path,"DVCSxsec_New.csv"),index=None)
DVCS_asymmetry.to_csv(os.path.join(dir_path,"DVCSAsym.csv"),index=None)

# ============================================================
# Merge with the previous DVCS data (seemingly not overlapping)
# ============================================================

DVCS_xsec_old = pd.read_csv(os.path.join(dir_path,'DVCSxsec_Old.csv'),names = ['y', 'xB', 't', 'Q', 'phi', 'f', 'delta f', 'pol'])

combined = pd.concat([DVCS_xsec_comb, DVCS_xsec_old], ignore_index=True)
combined_sel = combined[(combined['Q'] > Q_threshold) & (combined['xB'] < xB_Cut) & (combined['t']*(combined['xB']-1) - M ** 2 * combined['xB'] ** 2 > 0)]

lengtot = combined.shape[0]
lengsel = combined_sel.shape[0]

# ============================================================
# Drop the duplicated ones with some relative tolerance RT
# ============================================================

def drop_duplicates_rel_tol(df, compare_cols, rtol=1e-2):
    # Separate columns by type
    numeric_cols = df[compare_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in compare_cols if c not in numeric_cols]

    # Group by non-numeric columns (exact match)
    if non_numeric_cols:
        grouped = df.groupby(non_numeric_cols, sort=False)
    else:
        grouped = [((), df)]  # one group with all rows

    result_rows = []

    # Within each group, deduplicate by numeric similarity
    for _, group in grouped:
        if len(group) == 1 or not numeric_cols:
            result_rows.append(group.iloc[[0]])
            continue

        numeric_data = group[numeric_cols].to_numpy()
        keep = np.ones(len(group), dtype=bool)

        for i in range(len(group)):
            if not keep[i]:
                continue
            row_i = numeric_data[i]
            # Relative difference (max-relative)
            diffs = np.abs(numeric_data - row_i)
            max_abs = np.maximum(np.abs(numeric_data), np.abs(row_i))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diffs = np.where(max_abs == 0, 0, diffs / max_abs)

            # Mark rows as duplicates if all rel diffs are <= rtol
            similar = np.all(rel_diffs <= rtol, axis=1)
            keep &= ~similar | np.eye(len(group), dtype=bool)[i]

        result_rows.append(group[keep])

    return pd.concat(result_rows, ignore_index=True)

def Test_rtol(rt: float):
    
    combined_uni = drop_duplicates_rel_tol(combined, compare_cols=cols_to_check, rtol=rt)
    combined_uni_sel = combined_uni[(combined_uni['Q'] > Q_threshold) & (combined_uni['xB'] < xB_Cut) & (combined_uni['t']*(combined_uni['xB']-1) - M ** 2 * combined_uni['xB'] ** 2 > 0)]
    
    lenguni = combined_uni.shape[0]
    lengunisel = combined_uni_sel.shape[0]
    
    return [lengtot-lenguni, lengsel-lengunisel]

if __name__ == "__main__":
    
    RT = 0.01
    combined_uni = drop_duplicates_rel_tol(combined, compare_cols=cols_to_check, rtol=RT)
    combined_uni.to_csv(os.path.join(dir_path,'DVCSxsec_Merge.csv'), index=False)

    # ======================================================
    # Below test the choice of rtol
    # ======================================================
    '''
    rtollst = np.exp(np.linspace(np.log(0.0001),np.log(0.2),150))

    with Pool() as pool:
        Testrtol = pool.map(Test_rtol, rtollst)
        Testrtol = np.array(Testrtol)

    Test = np.hstack((rtollst.reshape(-1, 1), Testrtol))

    dftest = pd.DataFrame(Test, columns=['rtol', 'dup', 'dup2'])
    dftest.to_csv(os.path.join(dir_path,'DVCS_duplicate_test.csv'), index=False)
    '''
    '''
    dftest = pd.read_csv(os.path.join(dir_path,'DVCS_duplicate_test.csv'), header=0, names=['rtol', 'dup', 'dup2'])

    plt.plot(dftest['rtol'], dftest['dup'], label='duplicates')  # Replace as needed
    plt.plot(dftest['rtol'], dftest['dup2'], label='duplicates (large Q)')
    plt.xlabel('rtol')
    plt.ylabel('Num. of duplicates')
    plt.title('Plot of Num. of duplicates vs rtol')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''