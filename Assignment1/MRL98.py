import numpy as np
from math import ceil

from collections import Mapping, Container
from sys import getsizeof
import time
 
from DataStructure import *
from shared import *

# Source: https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object
 
    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.
 
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
 
    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0
 
    r = getsizeof(o)
    ids.add(id(o))
 
    if isinstance(o, str): #or isinstance(0, unicode):
        return r
 
    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())
 
    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)
 
    return r 


class MRL98:
    def __init__(self, N=1e6, b=10, k=50, buffers_sorted=True, max_range=1e6):
        self.N = int(N)
        self.b = b  # number of buffers
        self.k = k  # number of elements in each buffer
        self.num_collapsed = 0 # number of collapsed operations
        self.sum_weight_out_collapsed = 0 # sum of the weights of the output from the collapse operation
        self.sum_offset_collapsed = 0     # summ of the offsets from the collapse operation
        self.buffers_sorted = buffers_sorted
        self.buffers = [self._create_buffer() for _ in range(b)]#np.ndarray((10,),dtype=np.object)
        self.inf = int(max_range*1000) + 1 #np.inf
        printt(f"----\t-Inf: {-self.inf} -- +Inf: {self.inf}\t----", debug=True)
        self.last_collapse_types = FixedLengthFIFO()
        self.collapse_even_type = 1
        self.leftovers = 0
        self.history = [[]] # TODO (Improvement): Make history of the operations and print a tree 

        # self.buffers = np.empty((self.b, k)) # buffers memory bxk
        # # Intuitively, the weight of a buffer is the number of input elements represented by each element in the buffer. 
        # self.weights = np.zeros((self.b, 1))      # weights bx1
        # self.labels = np.zeros((self.b, 1))
        
    def get_memory_usage(self):
        # return deep_getsizeof(self.buffers)
        r = [getsizeof(buffer.buffer) for buffer in self.buffers]
        # r += 
        mem = sum(r)/1024
        printt(f"Memory consumption: {mem}KB")
        return mem

    def get_time_elapsed(self):
        t = self.final_time - self.initial_time
        printt(f"Time elapsed: {t}s")
        return t

    def _clear_buffer(self, idx):
        self.buffers[idx].clear()

    # Finished
    def _create_buffer(self):
        return Buffer(self.k, sorted=self.buffers_sorted)

    # Finished
    def run(self, s, phi):
        '''
            Runner for the algorithm with the metrics
            param s: Streamer
        '''
        self.initial_time = time.time()
        ret = self.collapse_policy(s, phi)
        self.final_time = time.time()
        return ret

    # Finished
    def collapse_policy(self, s, phi):
        """
            Based on the proposed New Algorithm in the paper MRL (Approximate Medians and other Quantiles in One Pass and with Limited Memory )
            Criteria when to use New/Collapse operations
            An algorithm for computing approximate quartiles consists of a series of invocations of NEW and COLLAPSE, terminating with OUTPUT.

            NEW populates empty buffers with input
            COLLAPSE reclaims some of them by collapsing a chosen subset of full buffers. 
            OUTPUT is invoked on the final set of full buffers. 

            param s: Streamer
        """
        iter = 0
        while not s.is_empty():
            printt(f"Iteration: {iter}")
            # smallest level for full buffers 
            levels = []
            empty_buffers_idx = []
            for i in range(self.b):
                buff = self.buffers[i]
                if(buff.is_empty()):
                    empty_buffers_idx.append(i)
                    continue
                levels.append(buff.level)
            # Let l be the smallest among the levels of currently full buffers
            min_level = 0
            if(len(levels) > 0):
                min_level = min(levels)
            num_empty_buffers = len(empty_buffers_idx)
            printt(f"Num. empty buffers: {num_empty_buffers}")
            # If there is exactly one empty buffer, invoke NEW and assign it level l
            if(num_empty_buffers == 1):
                buffer = self.buffers[empty_buffers_idx[0]]
                self.new(buffer, s.get_batch(self.k))
                buffer.set_level(min_level)
                self.history[min_level].append(buffer.weight)
            # If there are at least two empty buffers, invoke NEW on each and assign level 0 to each one. 
            elif(num_empty_buffers >= 2):
                for i in range(0, num_empty_buffers):
                    buffer = self.buffers[empty_buffers_idx[i]]
                    self.new(buffer, s.get_batch(self.k))
                    buffer.set_level(0)
                    self.history[0].append(buffer.weight)
                    
            # If there are no empty buffers, invoke COLLAPSE on the set of buffers with level l. Assign the output buffer, level l + 1.
            else:
                # TODO (Improvement): can be done better with numpy and filter
                buffers = [] # buffers with level l
                for i in range(self.b):
                    buff = self.buffers[i]
                    if(buff.level == min_level):
                        buffers.append(self.buffers[i])
                output_buffer = self.collapse(buffers, level=min_level+1)
                if(min_level+1 >= len(self.history)):
                    self.history.append([])
                self.history[min_level+1].append(output_buffer.weight)
            iter += 1

        non_empty_buffers = []
        for i in range(self.b):
            buff = self.buffers[i]
            if(not buff.is_empty()):
                non_empty_buffers.append(self.buffers[i])

        return self.output(non_empty_buffers, phi)
    # ----------------------------------------------------------------------------------------------------------------
    # Basic Operations
    # Finished
    def new(self, buffer, k_elements):
        '''
            [Basic Operation]
            - NEW takes as input an empty buffer.
            - It is invoked only if there is an empty buffer and at least one outstanding element in the input sequence.

        param buffer: Buffer 
        param k_elements: numpy.array (nx1) such that 0 < n <= k
        '''
        # If the buffer cannot be filled completely because there are less than k remaining elements in the input sequence, 
        leftovers = self.k - len(k_elements)

        if(leftovers > 0):
            # an equal number of -inf and +inf elements are added to make up for the deficit.
            neg_inf = np.array([-self.inf for i in range(leftovers//2)]).reshape((-1,1))
            pos_inf = np.array([self.inf for i in range(leftovers - (leftovers//2))]).reshape((-1,1))
            k_elements = np.append(k_elements, neg_inf, 0)
            k_elements = np.append(k_elements, pos_inf, 0)
            self.leftovers = leftovers # only set in the last not complete buffer
            printt(f"Leftovers: {leftovers}")

        # The operation simply populates the input buffer with the next k elements from the input sequence        
        printt(f"New->Store: {k_elements}")
        buffer.store(k_elements)
        # Then, labels the buffer as full, and assigns it a weight of 1.
        buffer.set_weight(1)
        buffer.set_buffer_full()


    # Finished
    def collapse(self, buffers, level=None):
        """
            [Basic Operation]
            COLLAPSE takes c > 2 full input buffers, X1, X2,. . . ,Xc, and outputs a buffer, Y, all of size k. 
            
            param buffers: list
        """
        # printt([self.buffers[i] for i in range(len(buffers))])
        printt(f"Collapse -> Input: {[buffers[i] for i in range(len(buffers))]}")

        c = len(buffers)
        # The weight of the output buffer w(Y) is the sum of weights of input buffers: \sum_{i=1}^c w(X_i) 
        output_weight = sum([b.weight for b in buffers])
        output_buffer = self._create_buffer()
        
        # Sort elements in buffers (Done inside the buffer.store())

        # TODO (Improvement): It can be elemented without materialization (storing really in the memory) ->  compute the offset and compute the indicies first 
        #           Then while calculating the elements instead of adding them to the array just check if it is should be added to the output buffer or not
        # Compute offset
        # Compute indicies
        # Initialize next output index = 0
        # for loop or while loop whil next_output_index < self.k*c
        # Check the minimum pointer and minimum element
        # Check its multiplication value if it is in the range of the next output index
        #   Then added to the buffer and increase the next output index

        # ---------------------------------
        # TODO (Improvement): implement it in a better way if the buffers are sorted anyway:
        #           - Combine all of them in matrix (CxK) then make k pointers that point to the selected element which is the minimum not added to the big matrix in the row
        # printt("-------------------------------")
        # printt(buffers[1])
        sorted_elements = self._merge_buffers(buffers)

        # Take k spaced elements
        # Let (w(Y)+1)/2 is the offset
        offset = None
        if(output_weight % 2): # odd
            printt("Collapse Odd type")
            offset = (output_weight+1)//2
            self.last_collapse_types.push(4) # As it is binary code (1,2,4) 1-> first choice of even collapse, 2 second choice of even collapse, and 4 odd collapse
            # positions = j*w(Y) +(w(Y)+1)/2 for j = 0,1,2,...k-1 -> [j*w+(w+1)//2 for j in range(k)]  indcies
            # pass
        else: # even
            # If w(Y) is even, we have two choices
            self.last_collapse_types.push(self.collapse_even_type)
            # 1. positions = jw(Y) + w(Y)/2
            if(self.collapse_even_type == 1):
                printt("Collapse Even 1 type")
                offset = (output_weight)//2

            # 2. positions = jw(Y) + (w(Y)+2)/2
            elif(self.collapse_even_type == 2):
                printt("Collapse Even 2 type")
                offset = (output_weight+2)//2

            # Alternate between the two choices when we have successive even collapses with the same choice
            if(self.last_collapse_types.get_sum() == 2 or self.last_collapse_types.get_sum() == 4):
                self.collapse_even_type = (self.collapse_even_type%2)+1 # change it to 2 if it is 1 and to 1 if it is 2
            # for j=0,1,...k-1
        indcies = [j*output_weight+offset-1 for j in range(self.k)]
        output_elements = sorted_elements[indcies]
        output_buffer.store(output_elements.reshape((-1,1)))
        output_buffer.set_weight(output_weight)
        printt(f"Collapse -> Output: {list(output_buffer.buffer.reshape((-1)))}")
        self.num_collapsed += 1
        self.sum_weight_out_collapsed += output_weight#np.sum(output_elements)
        self.sum_offset_collapsed += offset
        # printt(f"Collapse operation data: num_collapsed={self.num_collapsed} -> sum={self.sum_weight_out_collapsed}")
        lemma1_assertion = self.sum_offset_collapsed >= ((self.num_collapsed + self.sum_weight_out_collapsed -1)//2)
        printt(f"Lemma 1 ({lemma1_assertion}): {self.sum_offset_collapsed} >= {((self.num_collapsed + self.sum_weight_out_collapsed -1)//2)}", debug=False)
        # assert lemma1_assertion # Lemma 1
        if(level is not None):
            output_buffer.set_level(level)
        for buffer in buffers:
            buffer.clear()
        
        buffers[0].copy(output_buffer)
        return output_buffer

    # WIP
    def output(self, buffers, phi):
        '''
            [Basic Operation]
            OUTPUT is performed exactly once, just before termination.
            It takes c > 2 full input buffers, X1,X2,. . . ,X,, of size k. 
            It outputs a single element, corresponding to the approximate \phi'-quartile of the augmented dataset. 
            Recall that the \phi-quartile of the original dataset corresponds to the \phi' quartile of the augmented dataset, 
                consisting of the -inf and +inf elements added to the last buffer.
            
            param buffers: list
        '''
        # Similar to COLLAPSE, this operator makes w(Xi) copies of each element in Xi and sorts all the input buffers together, taking the multiple copies of each element into account. 
        sorted_elements = self._merge_buffers(buffers)
        # W = w(X1) + w(X2) + . . . + w(Xc). 
        W = sum([b.weight for b in buffers])
        self.history.append([])
        self.history[-1].append(W)
        # The output is the element in position ceil[\phi`kW]
        beta = (self.N+self.leftovers)/self.N # 1+self.leftovers/self.N -> N is too much and leftovers is too small most probably will be numerical errors here (I am not sure) 
        printt(f"Beta={beta}")
        phi_approx = (2*phi+beta-1)/(2*beta) #phi
        printt(f"\phi={phi}, \phi`={phi_approx}")
        idx = ceil(phi_approx * self.k * W) - 1 # Zero index and the formula was for 1-indexed
        return int(sorted_elements[idx])

    # ----------------------------------------------------------------------------------------------------------------

    # Finished and tested
    def _merge_buffers(self, buffers):
        c = len(buffers)
        all_elements = []        
        if(self.buffers_sorted):
            pointers = [0 for _ in range(c)]
            get_element = lambda i: buffers[i][pointers[i]][0]

            for i in range(self.k*c):
                # TODO (Improvement): implement it in a better and more compact way
                mn_pointer_idx = np.argmin(pointers) # get the pointer with the least value inside to avoid a bug if we started with a value of the mn_pointer_idx = 0
                for i_p in range(c):
                    if(pointers[i_p] >= self.k):
                        continue
                    # printt(f"Min: {mn_pointer_idx}")
                    mn_element = get_element(mn_pointer_idx)
                    # printt(i_p)
                    current_element = get_element(i_p)
                    if(current_element < mn_element):
                        mn_pointer_idx = i_p
                # printt("###########")
                element = get_element(mn_pointer_idx) #self.buffers[mn_pointer_idx][pointers[mn_pointer_idx]]
                all_elements.extend([element]*buffers[mn_pointer_idx].weight)
                # printt(f"Minnn: {mn_pointer_idx} -> {pointers[mn_pointer_idx]} -> {element}")

                pointers[mn_pointer_idx] += 1
                # printt(all_elements)
                # printt("-----")
            all_elements = np.array(all_elements).reshape((-1))
            sorted_elements = np.sort(all_elements)
            # printt(sorted_elements)
            return sorted_elements
    def print_history(self):
        for i in range(len(self.history)-1, -1, -1):
            for j in range(len(self.history[i])):
                print(f"{self.history[i][j]}\t",end="")
            print("")
        