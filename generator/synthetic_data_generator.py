# Generating synthetic data for Neural Network Trianing
import numpy as np
import sys
from predicates import predicate_1

def data_generator_numpy(num_data):
    print("Generating data...")
    current_data = 0
    seen = set()
    vectors = []
    labels = []
    num_true = 0
    while(current_data < num_data):
        sample = np.random.randint(-sys.maxsize - 1, sys.maxsize, 3)
        x, y, z = sample[0], sample[1], sample[2]
        key = tuple(sorted((x,y)))
        if key in seen:
            continue
        current_data += 1
        seen.add(key)
        vectors.append(tuple((x,y,z)))
        result = predicate_1(x,y,z)
        if (num_true > current_data - num_true):
            continue
        elif(result): 
            labels.append(1)
            num_true += 1
        else:
            labels.append(0)

    print("Writing to file...")
    f = open("vectors.txt", "w+")
    f.write("\n".join("%s %s %s" % value for value in vectors))
    f.close()
    
    f = open("labels.txt", "w+")
    f.write("\n".join("%s" % label for label in labels))
    f.close()
    
    print("Number of True labels: %s" % num_true)
    print("Number of False labels: %s" % (num_data - num_true))
    return(vectors, labels)

data_generator_numpy(30000)