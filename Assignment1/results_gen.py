import argparse
import json
from matplotlib import pyplot as plt
import numpy as np

'''
python3 results_gen.py 

python3 results_gen.py --time 1 --file reduced_benchmark_data_res.json

python3 results_gen.py --plot 0 
'''

parser = argparse.ArgumentParser(description="Benchmark MRL98")
parser.add_argument('--file', default="benchmark_data_res.json", type=str, help=help, metavar='')
parser.add_argument('--time', default=0, type=int)
parser.add_argument('--plot', default=1, type=int)

args = parser.parse_args()


filename = args.file[:-5]

ext = ".json"

with open(filename+ext, 'r') as f:
    data = json.load(f)


N_list = data["N"]
num_exp = data["num_exp"]


# Create plots with different epsilons
for i in range(num_exp):
    exp_name = f"experiment{i+1}"
    exp = data[exp_name]
    plt.plot(N_list[:min(len(N_list), len(exp["mem"]))],exp["mem"], label=f"eps={exp['eps']}")

plt.xscale('log', basex=10)
plt.xlabel("N")
plt.ylabel("Memory (KB)") 

plt.legend(loc="upper left")
if(args.plot):
    plt.show()

# Create numpy array with results
arr = [[-1]]

arr[0].extend(N_list)
arr[0].extend(N_list)
arr[0].extend(N_list)

for i in range(num_exp):
    exp_name = f"experiment{i+1}"
    exp = data[exp_name]
    arr.append([exp['eps']])
    l = exp['b']
    l.extend(exp['k'])
    l.extend(exp['mem'])
    arr[-1].extend(l)
# Create CSV from numpy
# Source: https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
a = np.asarray(arr)
np.savetxt(filename+".csv", a, delimiter=",", fmt='%f')


if(args.time):
    # Create plots with different epsilons
    for i in range(num_exp):
        exp_name = f"experiment{i+1}"
        exp = data[exp_name]
        plt.plot(N_list[:min(len(N_list), len(exp["time"]))],exp["time"], label=f"eps={exp['eps']}")

    # plt.plot(N_list[:len(Y[0])],Y)
    plt.xscale('log', basex=10)
    plt.xlabel("N")
    plt.ylabel("Time (s)") 
    plt.legend(loc="upper left")
    if(args.plot):
        plt.show()


