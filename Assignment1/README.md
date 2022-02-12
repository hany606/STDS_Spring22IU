# Assignment 1

In this directory, you can find the implementation of [MRL98] and the codes for benchmarking the performance of the algorithms under different parameters.


## Directory structure:

├── benchmark.py: script to run the testing for different experiments
├── DataStructure.py: includes Buffer class
├── main.py: script to run MRL98 algorithm
├── MRL98.py: includes the main implementation of the algorithm
├── README.md
├── results_gen.py: script to generate the plots and table in the report
├── shared.py: includes functions that are shared between different files
├── STDS_A1_HanyHamed.pdf: Report for submission
├── Streamer.py: includes Streamer class and Generator class that responsible for generating the data stream
├── benchmark_data.json: Experiments with different parameters as it is mentioned in [MRL98]
├── benchmark_data_res.csv: generated table from benchmark_data.json experiments
├── benchmark_data_res.json: results of running MRL98 with experiments in benchmark_data.json
├── reduced_benchmark_data.json: reduced version of experiments in benchmark_data.json
└── reduced_benchmark_data_res.json: results of running MRL98 with experiments in reduced_benchmark_data.json


## How to use?

```bash
python3 main.py
```

'''
python3 benchmark.py --run 0 --file benchmark_data.json

python3 benchmark.py --run 1 --file reduced_benchmark_data.json
'''

'''
python3 results_gen.py 

python3 results_gen.py --time 1 --file reduced_benchmark_data_res.json

python3 results_gen.py --plot 0 
'''


## References

[MRL98] Gurmeet Singh Manku, Sridhar Rajagopalan, and Bruce G Lindsay. “Approximate medians and
other quantiles in one pass and with limited memory”. In: ACM SIGMOD Record 27.2 (1998), pp. 426–435.