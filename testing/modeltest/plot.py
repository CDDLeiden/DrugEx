import pandas as pd
from settings import *

def plot():
    df = pd.read_table(f'{OUTPUT_FILE}_fit.tsv')
    ax = df.plot.line(x='epoch', y=['desire_ratio', 'valid_ratio'], ylim=(0,1))
    fig = ax.get_figure()
    fig.savefig(f'{OUTPUT_FILE}_fit.png')
