#!/usr/bin/env python3

import yaml
import time
import subprocess
from argparse import ArgumentParser

client_binary = '/home/maxdml/triton-client/build/cc-clients/examples/grpc_async_infer_client_mixed'

results_dir = '/home/maxdml/triton-client/sosp32_results/'

parser = ArgumentParser()
parser.add_argument('load_range', nargs=3, type=int, default=[10, 50])
parser.add_argument('templates', nargs='+', type=str)
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

            results_file = results_dir + s + '/results_' + str(load) + '_' + s + '.csv'

            a = [client_binary, '--ip', '172.17.0.2', '--port', '8001', '-s', fname, '-o', results_file, '--sigma', s, '--num-jobs', '1000']

            print(f'Run with config {fname}, args {a}')
            p = subprocess.Popen(a, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            while(1):
                line = p.stdout.readline()
                print(line.decode('ascii'))
                if not line:
                    break
            p.wait()

            time.sleep(3)
