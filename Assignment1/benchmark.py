from main import *
import json
from copy import deepcopy
import argparse


'''
python3 benchmark.py --run 0 --file benchmark_data.json


python3 benchmark.py --run 1 --file reduced_benchmark_data.json
'''

parser = argparse.ArgumentParser(description="Benchmark MRL98")
parser.add_argument('--file', default="benchmark_data.json", type=str, help=help, metavar='')
parser.add_argument('--run', default=1, type=int)
args = parser.parse_args()


filename = args.file[:-5]
out_filename = f"{filename}_res"

ext = ".json"
data = None
with open(filename+ext, 'r') as f:
    data = json.load(f)

N_list = data["N"]
num_exp = data["num_exp"]

new_data = deepcopy(data)

for i in range(num_exp):
    exp_name = f"experiment{i+1}"
    new_data[exp_name]["mem"] = []
    new_data[exp_name]["q"] = []
    new_data[exp_name]["time"] = []
    for j in range(len(N_list)):
        N = N_list[j]
        exp = data[exp_name]
        eps = exp["eps"]
        b = exp["b"][j]
        k = exp["k"][j]
        print("##########################################")
        print(f"N={N}\teps={eps}\tb={b}\tk={k}")
        quantile, memory_usage, time_usage = main(N,b,k, run=bool(args.run))
        new_data[exp_name]["mem"].append(memory_usage)
        new_data[exp_name]["q"].append(quantile)
        new_data[exp_name]["time"].append(time_usage)
        print("##########################################")
    #     break
    # break
with open(out_filename+ext, '+w') as f:
    json.dump(new_data, f,indent=4)

