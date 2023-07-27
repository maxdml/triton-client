#!/usr/bin/python3

import yaml
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def debug_pctl(pctl, count, value):
    pass
    #print(f'found {pctl} at sample {count} with value {value}')

def compute_pctls(hist, load, model):
    pctls = {}
    pctls['load'] = load
    pctls['model'] = model
    pctls['MIN'] = int(hist['MIN'])
    pctls['MAX'] = int(hist['MAX'])
    count = int(hist['COUNT'])
    pctls['COUNT'] = count
    pctls['MEAN'] = (int(hist['TOTAL']) / count)
    pctls['GOODPUT'] = float(hist['GOODPUT'])
    hist = hist.drop(['GOODPUT', 'DURATION', 'MIN', 'MAX', 'COUNT', 'TOTAL'], axis=1).sum()
    c = 0
    for bucket in range(hist.shape[0]):
        b_count = int(hist.iloc[bucket])
        b_value = int(hist.index[bucket])
        if c < count * .25 and c + b_count >= count * .25:
            pctls['p25'] = b_value
            debug_pctl('p25', c, b_value)
        if c < count * .5 and c + b_count >= count * .5:
            pctls['MEDIAN'] = b_value
            debug_pctl('p50', c, b_value)
        if c < count * .75 and c + b_count >= count * .75:
            pctls['p75'] = b_value
            debug_pctl('p75', c, b_value)
        if c < count * .99 and c + b_count >= count * .99:
            pctls['p99'] = b_value
            debug_pctl('p99', c, b_value)
        if c < count * .999 and c + b_count >= count * .999:
            pctls['p99.9'] = b_value
            debug_pctl('p99.9', c, b_value)
        if c < count * .9999 and c + b_count >= count * .9999:
            pctls['p99.99'] = b_value
            debug_pctl('p99.99', c, b_value)
        c += b_count
    return pd.DataFrame(pctls, index=[0])


parser = argparse.ArgumentParser()
parser.add_argument('experiments')

args = parser.parse_args()

def parse_data(exps):
    data = {}
    with open(exps, 'r') as f:
        for line in f.readlines():
            fpath = line.strip()
            # Load hist
            print(f"loading {fpath}")
            # load = int(fpath.split('.')[0].split('_')[-1])
            load = int(fpath.split('/')[1].split('.')[0].split('_')[-2])
            data[load] = {}
            with open(fpath, 'r') as f2:
                lines = f2.readlines()
            for (header, values) in zip(lines[::2], lines[1::2]):
                t = values.split()[0]
                data[load][t] = pd.DataFrame(
                    {k:v for (k,v) in zip(header.split()[1:], values.split()[1:])},
                    index=[0]
                )

    pctls = []
    print(f"Computing percentiles")
    for load, hists in data.items():
        for model, hist in hists.items():
            if model == 'ALL':
                pctls.append(compute_pctls(hist, load, model))
    print(f"Concatenating dataframe")
    return pd.concat(pctls)

df = parse_data(args.experiments).sort_values('load')
print(df)

models = df.model.unique()
top = max(df['MAX'])
fig, axes = plt.subplots(2, len(models), squeeze=False, sharey=False, sharex=False, figsize=(15,5))
for i, model in enumerate(models):
    print(f'Plotting {model}')
    d = df[df.model == model]
    line, = axes[0][i].plot(d.load, d['p99'], label=model, marker='.')
    axes[0][i].set_ylim(bottom=0)
    axes[0][i].legend()
    axes[0][i].set_ylim(top=top, bottom=-10)
    axes[0][i].ticklabel_format(style='plain')
    axes[0][i].set_ylabel(f'Response time (us)')
    axes[0][i].set_xlabel(f'Offered load')

    line, = axes[1][i].plot(d.GOODPUT, d['p99'], label=model, marker='.')
    axes[1][i].set_ylim(bottom=0)
    axes[1][i].legend()
    axes[1][i].set_ylim(top=top, bottom=-10)
    axes[1][i].ticklabel_format(style='plain')
    axes[1][i].set_ylabel(f'Response time (us)')
    axes[1][i].set_xlabel(f'Jobs per seconds (all)')


plt.savefig(f'{args.experiments}.pdf', format='pdf')
plt.show()
