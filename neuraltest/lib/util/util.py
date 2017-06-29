import matplotlib.pyplot as plt
import numpy as np
import os
import random

def plot_loss(losses):
    plt.figure(figsize=(10,8))
    plt.plot(losses[0], label='Discriminative loss')
    plt.plot(losses[1], label='Generative loss')
    plt.legend()
    plt.show()

def write_to_file(filepath, filename, data):
    print("Writting to " + filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    f = open(filepath + "\\" + filename, "w+")
    f.write("\n".join("%s" % value for value in data))
    f.close()
    
def uniform_vector(minimum, maximum, size):
#    Return a random floating point number N such that a <= N <= b for a <= b 
#    and b <= N <= a for b < a.
#    The end-point value b may or may not be included in the range depending on 
#    floating-point rounding in the equation a + (b-a) * random().
    count = 0
    vector = []
    while count < size:
        count += 1
        vector.append(random.uniform(minimum, maximum))
    return vector

def sample_z(low, high, size):
    return np.random.uniform(low, high, size=size)