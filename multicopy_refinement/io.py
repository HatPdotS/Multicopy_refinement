import pandas as pd


def read_crystfel_hkl(path):
    dtypes = {'h': int, 'k': int, 'l': int, 'I': float, 'phase': float, 'sigma': float, 'nmeas': int}
    ref = pd.read_csv(path, sep=r'\s+', names=['h','k','l','I','phase','sigma','nmeas'],skiprows=3,on_bad_lines='skip',dtype=dtypes).dropna()
    return ref

def read_mtz(path):
    import reciprocalspaceship as rs
    mtz = rs.read_mtz(path).reset_index()
    mtz.rename(columns={'H': 'h', 'K': 'k', 'L': 'l'}, inplace=True)
    return mtz
