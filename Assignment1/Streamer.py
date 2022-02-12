from random import randint
import numpy as np
from random import sample as r_sample
from shared import *

class Generator:

    def __init__(self, static_range=True):
        self.count = 0 # for generator style generation
        self.range = lambda n: 1000 if static_range else n*1000

    def generate(self, N=1e6):
        # Generate a list of numbers with N length
        # l = [randint(0,N*100) for i in range(N)]
        printt(f"----\tRange of elements [0, {self.range(N)}]\t----", debug=True)
        N = int(N)
        l = np.random.randint(self.range(N), size=N).reshape((-1,1))    # N is multiplied with 1000 to decrease the probability of having two similar elements
        return l

    def generate_seq(self, N):
        self.range = lambda _: N
        return r_sample(range(1,N), N)
    
    def generate_sorted_seq(self, N):
        self.range = lambda _: N
        return np.array([i+1 for i in range(N)])#r_sample(range(N), N)

class Streamer:
    def __init__(self, *args, **kwargs):
        N = kwargs.pop("N", 1e6)
        self.generator = Generator(*args, **kwargs)
        self.gen = self.generator.generate(N=N)
        # self.gen = self.generator.generate_seq(N=N)
        # self.gen = self.generator.generate_sorted_seq(N=N)
        self.current_idx = 0

    def get_batch(self, k):
        slice_idx = min(self.current_idx + k, len(self.gen)) 
        ret = self.gen[self.current_idx:slice_idx]
        self.current_idx += k
        return ret

    def is_empty(self):
        return self.current_idx >= len(self.gen)

