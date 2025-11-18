import pandas as pd
from matplotlib import pyplot as plt
def plot_statistics_vs_parameter(df,parameter):
    df = df.groupby(parameter).mean()
    rtest = df['rtest'].values
    rwork = df['rwork'].values
    index = df.index.values
    return index, rtest, rwork



df =pd.read_csv('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/refine_bulk_solvent_results_2.csv')

print(df.keys())

param = ['dilation_radius','transition','radius']


for p in param:
    index, rtest, rwork = plot_statistics_vs_parameter(df,p)
    print(f'Parameter: {p}')
    plt.plot(index, rtest, label='R-test')
    plt.plot(index, rwork, label='R-work')
    plt.xlabel(p)
    plt.ylabel('R-factor')
    plt.legend()
    plt.savefig(f'test_statistics_vs_{p}.png')
    plt.close()