# Generating synthetic data for Neural Network Trianing
import numpy as np

from predicates import predicate_1

"""
NOTE: Adjust the digits parameters of the lambda function f and g to 
      balance the number of True / False data generated.
"""

def data_generator(num_data, digits):
    print("Generating data...")
    current_data = 0
    seen = set()
    vectors = []
    labels = []
    num_true = 0
    while(current_data < num_data):
        f = lambda: int(''.join(np.random.choice(list("0123456789"))
                        for i in range(np.random.randint(1, digits + 1))))
        g = lambda: int(''.join(np.random.choice(list("0123456789"))
                        for i in range(np.random.randint(1, digits + 3))))
        
        x, y, z = f(), f(), g()
        key = tuple(sorted((x,y)))
        if key in seen:
            continue
        current_data += 1
        seen.add(key)
        vectors.append(tuple((x,y,z)))
        result = predicate_1(x,y,z)
        if(result): 
            labels.append(1)
            num_true += 1
        else:
            labels.append(0)
    print("Data generation completed...")
    print("Writing to file...")
    f = open("vectors_50k_6d.txt", "w+")
    f.write("\n".join("%s %s %s" % value for value in vectors))
    f.close()
    
    f = open("labels_50k_6d.txt", "w+")
    f.write("\n".join("%s" % label for label in labels))
    f.close()
    
    print("Number of True labels: %s" % num_true)
    print("Number of False labels: %s" % (num_data - num_true))
    return(vectors, labels)