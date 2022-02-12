import numpy as np
from shared import *
from copy import deepcopy

class Buffer(object):
    def __init__(self, k, sorted):
        self.k = k
        self.clear()
        self.sorted = sorted

    def clear(self):
        self.buffer = np.empty((self.k, 1), dtype=np.int32) 
        self.weight = 0
        self.level = 0 # Associate with each buffer X an integer L(X) that denotes its level.
        self.empty = 1  # 1: empty, 0: full (not empty)

    def store(self, k_elements):
        '''
        param k_elements: numpy array (kx1)
        '''
        self.buffer = k_elements
        if(self.sorted):
            self.buffer = np.sort(self.buffer)
        self.set_buffer_full()
        printt(f"In buffer: {self.buffer}")

    def copy(self, buffer):
        self.buffer = deepcopy(buffer.buffer)
        self.weight = buffer.weight
        self.level = buffer.level
        self.empty = buffer.empty

    def __getitem__(self,key):
        return self.buffer[key]

    def set_weight(self, w):
        self.weight = int(w)
    def set_level(self, l):
        self.level = l
    def set_buffer_empty(self):
        self.empty = 1
    def set_buffer_full(self):
        self.empty = 0

    def is_empty(self):
        # True: empty, False: Full
        return self.empty == 1

class FixedLengthFIFO:
    def __init__(self, fixed_length=2):
        self.length = fixed_length
        self.buffer = []

    def push(self, e):
        self.buffer.append(e)

        if(len(self.buffer) >= self.length):
            self.pop()

        return e

    def pop(self):
        e = self.buffer[0]
        del self.buffer[0]
        return  e

    def get_sum(self):
        return sum(self.buffer)

