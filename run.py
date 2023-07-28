#!/usr/bin/env python3

import yaml
import time
import subprocess
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('load_range', nargs=3, type=int, default=[10, 50])
parser.add_argument('templates', nargs='+', type=str)
parser.add_argument('-b', '--binary', dest='client_binary');
parser.add_argument('-o', '--output_dir', dest='output_dir');
args = parser.parse_args()

for s in ['1.5', '2']:
    for template in args.templates:
        print(f'Running experiments for {template}')
        with open(template, 'r') as cfg_tpl:
            cfg_str = cfg_tpl.read()

        for load in range(args.load_range[0], args.load_range[1], args.load_range[2]):
            fname = template + '_' + str(load) + '.yaml'
            with open(fname, 'w') as f:
                f.write(cfg_str)
                f.write('rate: ' + str(load))

            results_file = args.output_dir + s + '/results_' + str(load) + '_' + s + '.csv'

            a = [args.client_binary, '--ip', '172.17.0.2', '--port', '8001', '-s', fname, '-o', results_file, '--sigma', s, '--num-jobs', '1000']

            print(f'Run with config {fname}, args {a}')
            p = subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            while(1):
                line = p.stdout.readline()
                print(line.decode('ascii'))
                if not line:
                    break
            p.wait()

            time.sleep(3)
