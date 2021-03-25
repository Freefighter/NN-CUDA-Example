import os
import time
import argparse
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# c = a + b (shape: [n])
n = 1024 * 1024
a = tf.random.normal([n])
b = tf.random.normal([n])

ntest = 10

def show_time(func):
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        # torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        # torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res

def run_cuda():
    if args.compiler == 'jit':
        raise NotImplementedError
    elif args.compiler == 'setup':
        raise NotImplementedError
    elif args.compiler == 'cmake':
        cuda_c = cuda_module.add2(a, b)
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    return cuda_c

def run_tf():
    # return None to avoid intermediate GPU memory application
    # for accurate time statistics
    c = a + b
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=str, choices=['jit', 'setup', 'cmake'], default='jit')
    args = parser.parse_args()

    if args.compiler == 'jit':
        raise NotImplementedError
    elif args.compiler == 'setup':
        raise NotImplementedError
    elif args.compiler == 'cmake':
        cuda_module = tf.load_op_library('build/libadd2.so')
    else:
        raise Exception("Type of cuda compiler must be one of jit/setup/cmake.")

    print("Running cuda...")
    cuda_time, _ = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running tensorflow...")
    tf_time, _ = show_time(run_tf)
    print("Tensorflow time:  {:.3f}us".format(np.mean(tf_time)))
