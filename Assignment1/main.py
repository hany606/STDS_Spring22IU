from Streamer import *
from MRL98 import *

def main(N, b, k, phi=0.5, run=True):
    s = Streamer(N=N)

    algo = MRL98(N=N, b=b, k=k, max_range=s.generator.range(N))
    quantile = None
    time_usage = None
    if(run):
        quantile = algo.run(s, phi)
        time_usage = algo.get_time_elapsed()

    memory_usage = algo.get_memory_usage()
    print(f"Output: {quantile}")
    print(f"MEMORY: {memory_usage} KB")
    print(f"Time: {time_usage} seconds")
    return quantile, memory_usage, time_usage
    # algo.print_history()


if __name__ == '__main__':
    # Testing MRL98
    N = 10000
    k = 5
    b = 5

    main(N=N, b=b, k=k)
    # ---------------------------------------------------------
    # Testing Streamer
    # s = Streamer(N=N)
    # printt(s.gen, debug=True)
    # while not s.is_empty():
    #     printt(s.get_batch(k))

    # Testing Generator
    # gen = Generator(static_range=True)
    # l = gen.generate(N=N)
    # printt(l)    
    # printt(len(l))

    # Testing collapse
    # b1 = Buffer(k=k, sorted=True)
    # b1.store(np.array([52,12,72,132,102]))
    # b1.weight = 2
    # b2 = Buffer(k=k, sorted=True)
    # b2.store(np.array([143,83,33,153,23]))
    # b2.weight = 3
    # b3 = Buffer(k=k, sorted=True)
    # b3.store(np.array([114,64,94,124,44]))
    # b3.weight = 4

    # buffers = [b1, b2, b3]
    # buff = algo.collapse(buffers)
    # for buffer in buffers:
    #     print(buffer.buffer)
    # print(buff.buffer)
    # print(algo.buffers[0])