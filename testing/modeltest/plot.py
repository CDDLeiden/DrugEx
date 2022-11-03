import pandas as pd

df = pd.read_table('output/pretraining_fit.tsv')

ax = df.plot.line(x='epoch', y=['desire_ratio', 'valid_ratio'], ylim=(0,1))
fig = ax.get_figure()
fig.savefig('output/pretraining_fit.png')
