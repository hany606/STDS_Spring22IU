from random import randint
import numpy as np
from math import ceil

class Generator:

    def __init__(self, static_range=False):
        self.count = 0 # for generator style generation
        self.range = lambda n: 1000 if static_range else n*1000

    def generate(self, N=1e6):
        # Generate a list of numbers with N length
        # l = [randint(0,N*100) for i in range(N)]
        N = int(N)
        l = np.random.randint(self.range(N), size=N)    # N is multiplied with 1000 to decrease the probability of having two similar elements
        return l

    def generate2(self, N=1e6):
        N = int(N)
        if(self.count == N):
            return -1
        yield randint(0,self.range(N))

class Buffer(object):
    def __init__(self, k, sorted):
        self.buffer = np.empty((k, 1)) 
        self.weight = 0
        self.label = 0
        self.level = 0 # Associate with esch buffer X an integer L(X) that denotes its level.
        self.sorted = sorted

    def store(self, k_elements):
        '''
        param k_elements: numpy array (kx1)
        '''
        self.buffer = k_elements
        if(self.sorted):
            self.buffer = np.sort(self.buffer)
        # print(f"In buffer: {self.buffer}")

    def set_weight(self, w):
        self.weight = w
    def set_label(self, l):
        self.label = l
    def set_level(self, l):
        self.level = l

class MRL98:
    # TODO: sort the elements when we store them into the buffers or not
    def __init__(self, N=1e6, b=10, k=50, buffers_sorted=True):
        self.N = int(N)
        self.b = b  # number of buffers
        self.k = k  # number of elements in each buffer
        self.num_collapsed = 0 # number of collapsed operations
        self.sum_weight_out_collapsed = 0 # sum of the weights of the output from the collapse operation
        self.sum_offset_collapsed = 0     # summ of the offsets from the collapse operation
        self.buffers_sorted = buffers_sorted
        self.buffers = [self._create_buffer() for _ in range(b)]#np.ndarray((10,),dtype=np.object)
        # self.buffers = np.empty((self.b, k)) # buffers memory bxk
        # # Intuitively, the weight of a buffer is the number of input elements represented by each element in the buffer. 
        # self.weights = np.zeros((self.b, 1))      # weights bx1
        # self.labels = np.zeros((self.b, 1))
        
        

    def run(self, l, phi):
        # An algorithm for computing approximate quartiles consists of a series of invocations of NEW and COLLAPSE, terminating with OUTPUT.

        # NEW populates empty buffers with input
        # COLLAPSE reclaims some of them by collapsing a chosen subset of full buffers. 
        # OUTPUT is invoked on the final set of full buffers. 

        # Pass a new element to the algorithm
        # Perform one-pass and returns the quantile
        q = None
        print(self.N, self.k)
        for i in range(0, self.N, self.k):
            # e = l[i*self.k:(i+1)*self.k]
            # Take k sequence from the list each time
            lim = min(i+self.k, len(l))
            e = l[i:lim]
            print(f"e:{e}")
            self.new(self.buffers[0], e) # TODO: not the correct place (Understand where it should be)
            q = self._pass(e, phi)
            # return -1
            # yield q
        return q

    def _pass(self, e, phi):
        q = None
        return q

    def new(self, buffer, k_elements):
        '''
            - NEW takes as input an empty buffer.
            - It is invoked only if there is an empty buffer and at least one outstanding element in the input sequence.

        param buffer: Buffer 
        param k_elements: numpy.array (nx1) such that 0 < n <= k
        '''
        # If the buffer cannot be filled completely because there are less than k remaining elements in the input sequence, 
        leftovers = self.k - len(k_elements)

        if(leftovers > 0):
            # an equal number of -inf and +inf elements are added to make up for the deficit.
            neg_inf = np.array([-np.inf for i in range(leftovers//2)])
            pos_inf = np.array([np.inf for i in range(leftovers - (leftovers//2))])
            # TODO: not sure if it is added to the end of the array or should be somewhere else
            k_elements = np.append(k_elements, neg_inf, 0)
            k_elements = np.append(k_elements, pos_inf, 0)

        # The operation simply populates the input buffer with the next k elements from the input sequence        
        print(f"New->Store: {k_elements}")
        buffer.store(k_elements)
        # Then, labels the buffer as full, and assigns it a weight of 1.
        buffer.set_label(1)
        buffer.set_weight(1.0)
        # self.
        pass

    def collapse_algorithm(self):
        """
            Based on the proposed New Algorithm in the paper MRL (Approximate Medians and other Quantiles in One Pass and with Limited Memory )
        """
        # l = ? # smallest level for full buffers # Let l be the smallest among the levels of currently full buffers
        num_empty_buffers = 0 # TODO

        # If there is exactly one empty buffer, invoke NEW and assign it level l
        if(num_empty_buffers == 1):
            buffer = [] # Empty buffer TODO from self.buffers
            self.new(buffer)
            buffer.set_level(l)
        # If there are at least two empty buffers, invoke NEW on each and assign level 0 to each one. 
        if(num_empty_buffers >= 2):
            for i in range(0, num_empty_buffers):
                buffer = [] # Empty buffer TODO from self.buffers
                self.new(buffer)
                buffer.set_level(0)
                
        # If there are no empty buffers, invoke COLLAPSE on the set of buffers with level l. Assign the output buffer, level l + 1.
        if(True): # TODO
            buffers = [] # buffers with level l TODO from self.buffers
            self._collapse(buffers)
        pass

    def _create_buffer(self):
        return Buffer(self.k, sorted=self.buffers_sorted)

    def _collapse(self, buffers):
        """
            COLLAPSE takes c > 2 full input buffers, X1, X2,. . . ,Xc, and outputs a buffer, Y, all of size k. 
            
            param buffers: list
        """
        c = len(buffers)
        # The weight of the output buffer w(Y) is the sum of weights of input buffers: \sum_{i=1}^c w(X_i) 
        output_weight = sum([b.weight for b in buffers])
        output_buffer = self._create_buffer()
        
        # Sort elements in buffers
        # TODO: It can be elemented without materialization (storing really in the memory) ->  compute the offset and compute the indicies first 
        #           Then while calculating the elements instead of adding them to the array just check if it is should be added to the output buffer or not
        # Compute offset
        # Compute indicies
        # Initialize next output index = 0
        # for loop or while loop whil next_output_index < self.k*c
        # Check the minimum pointer and minimum element
        # Check its multiplication value if it is in the range of the next output index
        #   Then added to the buffer and increase the next output index

        # ---------------------------------
        # TODO: implement it in a better way if the buffers are sorted anyway:
        #           - Combine all of them in matrix (CxK) then make k pointers that point to the selected element which is the minimum not added to the big matrix in the row
        # print("-------------------------------")
        # print(buffers[1])
        sorted_elements = self._merge_buffers(buffers)

        # Take k spaced elements
        # Let (w(Y)+1)/2 is the offset
        offset = None
        if(output_weight % 2): # odd
            offset = (output_weight+1)//2
            # positions = j*w(Y) +(w(Y)+1)/2 for j = 0,1,2,...k-1 -> [j*w+(w+1)//2 for j in range(k)]  indcies
            # pass
        else: # even
            # If w(Y) is even, we have two choices
            # 1. positions = jw(Y) + w(Y)/2
            offset = (output_weight)//2
            # 2. positions = jw(Y) + (w(Y)+2)/2
            offset = (output_weight+2)//2
            # for j=0,1,...k-1
        indcies = [j*output_weight+offset-1 for j in range(self.k)]
        output_elements = sorted_elements[indcies]
        output_buffer.store(output_elements)
        print(output_buffer.buffer)
        self.num_collapsed += 1
        self.sum_weight_out_collapsed += output_weight#np.sum(output_elements)
        self.sum_offset_collapsed += offset
        # print(f"Collapse operation data: num_collapsed={self.num_collapsed} -> sum={self.sum_weight_out_collapsed}")
        print(f"Lemma 1: {self.sum_offset_collapsed} >= {((self.num_collapsed + self.sum_weight_out_collapsed -1)/2)}")
        assert self.sum_offset_collapsed >= ((self.num_collapsed + self.sum_weight_out_collapsed -1)/2) # Lemma 1
        return output_buffer

    def _merge_buffers(self, buffers):
        c = len(buffers)
        all_elements = []        
        if(self.buffers_sorted):
            pointers = [0 for _ in range(c)]
            get_element = lambda i: buffers[i].buffer[pointers[i]]

            for i in range(self.k*c):
                # TODO: implement it in a better and more compact way
                mn_pointer_idx = np.argmin(pointers) # get the pointer with the least value inside to avoid a bug if we started with a value of the mn_pointer_idx = 0
                for i_p in range(c):
                    if(pointers[i_p] >= self.k):
                        continue
                    # print(f"Min: {mn_pointer_idx}")
                    mn_element = get_element(mn_pointer_idx)
                    # print(i_p)
                    current_element = get_element(i_p)
                    if(current_element < mn_element):
                        mn_pointer_idx = i_p
                # print("###########")
                element = get_element(mn_pointer_idx) #self.buffers[mn_pointer_idx][pointers[mn_pointer_idx]]
                all_elements.extend([element]*buffers[mn_pointer_idx].weight)
                # print(f"Minnn: {mn_pointer_idx} -> {pointers[mn_pointer_idx]} -> {element}")

                pointers[mn_pointer_idx] += 1
                # print(all_elements)
                # print("-----")
            all_elements = np.array(all_elements).reshape((-1))
            sorted_elements = np.sort(all_elements)
            print(sorted_elements)
            return sorted_elements

    def output(self, buffers, phi):
        '''
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
        # The output is the element in position ceil[\phi`kW]
        phi_approx = phi # TODO: Find out what is the relation between phi and phi_approx (phi: real dataset, phi_approx: dataset augmented with -inf and +inf added to the last buffer)
        idx = ceil(phi_approx * self.k * W)
        return sorted_elements[idx]

if __name__ == '__main__':
    gen = Generator(static_range=True)
    N = 26
    k = 5
    l = gen.generate(N=N)
    print(l)    
    print(len(l))
    algo = MRL98(N=N, b=3, k=k)
    # algo.run(l, 0.5)
    b1 = Buffer(k=k, sorted=True)
    b1.store(np.array([52,12,72,132,102]))
    b1.weight = 2
    b2 = Buffer(k=k, sorted=True)
    b2.store(np.array([143,83,33,153,23]))
    b2.weight = 3
    b3 = Buffer(k=k, sorted=True)
    b3.store(np.array([114,64,94,124,44]))
    b3.weight = 4


    algo._collapse([b1, b2, b3])
    # print(algo.buffers[0])