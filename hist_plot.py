#!/usr/bin/python

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('load_range', nargs=3, type=int, default=[10, 50])
parser.add_argument('experiment_dir', type=str)
parser.add_argument('experiment_label', type=str)
args = parser.parse_args()

cycles_per_us = 2195

models = ["densenet-9", "googlenet-9", "inceptionv3-kelvin", "mobilenetv2-7", "resnet18-v2-7", "resnet34-v2-7", "resnet50-v2-7", "squeezenet1.1-7"]
# models = ["squeezenet1.1-7", "inceptionv3"]

for s in ['1.5', '2']:
# for s in ['1.5']:
    s_df = {
        'OFFERED_LOAD': [],
        'model': [],
        'MIN': [],
        'MAX': [],
        'COUNT': [],
        'DURATION': [],
        'GOODPUT': [],
        'MEAN': [],
        'MEDIAN': [],
        'p95': [],
        'SIGMA': [],
    }
    for load in range(args.load_range[0], args.load_range[1], args.load_range[2]):
        results_file = f'{args.experiment_dir}/sched_all_nsdi23.yaml_{load}_{s}.csv'
        # print(f'processing {results_file}')
        df = pd.read_csv(results_file, delimiter='\t')
        s_df['SIGMA'].append(s)
        s_df['OFFERED_LOAD'].append(load)
        s_df['model'].append('ALL')
        s_df['MIN'].append(df.LATENCY.min() / 1e3)
        s_df['MAX'].append(df.LATENCY.max() / 1e3)
        s_df['COUNT'].append(df.shape[0])
        duration = ((max(df.RECEIVE) - min(df.SEND)) / cycles_per_us) / 1e6 # cycles to seconds
        s_df['DURATION'].append(duration)
        goodput = df.shape[0] / duration
        s_df['GOODPUT'].append(goodput)
        s_df['MEAN'].append(df.LATENCY.mean() / 1e3)
        s_df['MEDIAN'].append(df.LATENCY.quantile(q=.5) / 1e3) # in ms
        s_df['p95'].append(df.LATENCY.quantile(q=.95) / 1e3) # in ms

    df = pd.DataFrame(s_df)
    print(df)
    df.to_csv(f'{args.experiment_label}-{s}.csv')
    # sns.lineplot(data=df, x='OFFERED_LOAD', y='p95', label=f's{s}-p95', marker='^')
    # sns.lineplot(data=df, x='OFFERED_LOAD', y='MEDIAN', label=f's{s}-p50', marker='o')
    sns.lineplot(data=df, x='GOODPUT', y='p95', label=f's{s}-p95', marker='^')
    sns.lineplot(data=df, x='GOODPUT', y='MEDIAN', label=f's{s}-p50', marker='o')
##    ax2 = plt.twinx()
##    sns.lineplot(data=df.load, ax=ax2)
##    ax2.set_xlim(ax1.get_xlim())
##    ax2.set_xticks(range(args.load_range[0], args.load_range[1], args.load_range[2]))
##    ax2.set_xticklabels(range(args.load_range[0], args.load_range[1], args.load_range[2]))
##    ax2.set_xlabel("Offered load (requets/seconds)")

plt.legend()
plt.yscale('log')
plt.title(f'{args.experiment_label} lognormal')
plt.xlabel('goodput (requests/seconds)')
plt.ylabel('latency (ms)')

fname = f'{args.experiment_label}.pdf'

plt.savefig(fname)
