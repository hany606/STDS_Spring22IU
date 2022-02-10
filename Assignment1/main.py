from random import randint
import numpy as np

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
    def __init__(self, k):
        self.buffer = np.empty((k, 1)) 
        self.weight = 0
        self.label = 0
        self.level = 0 # Associate with esch buffer X an integer L(X) that denotes its level.

    def store(self, k_elements):
        self.buffer = k_elements

    def set_weight(self, w):
        self.weight = w
    def set_label(self, l):
        self.label = l
    def set_level(self, l):
        self.level = l

class MRL98:
    def __init__(self, N=1e6, b=10, k=50):
        self.N = int(N)
        self.b = b  # number of buffers
        self.k = k  # number of elements in each buffer
        self.buffers = [Buffer(self.k) for _ in range(b)]#np.ndarray((10,),dtype=np.object)

        # self.buffers = np.empty((self.b, k)) # buffers memory bxk
        # # Intuitively, the weight of a buffer is the number of input elements represented by each element in the buffer. 
        # self.weights = np.zeros((self.b, 1))      # weights bx1
        # self.labels = np.zeros((self.b, 1))


    def run(self, l, phi):
        print("Asdalsmd")

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

    def _collapse(self, buffers):
        """
            COLLAPSE takes c > 2 full input buffers, X1, X2,. . . ,Xc, and outputs a buffer, Y, all of size k. 
        """
        output_buffer = Buffer(self.k)
        # TODO
        return output_buffer


    def output(self):
        # TODO
        pass





if __name__ == '__main__':
    gen = Generator(static_range=True)
    N = 26
    l = gen.generate(N=N)
    print(l)    
    print(len(l))
    algo = MRL98(N=N, b=1, k=7)
    algo.run(l, 0.5)
    # print(algo.buffers[0])