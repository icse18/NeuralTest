# Generating synthetic data for Neural Network Trianing
import numpy as np

# Predicate 1: x ** 2 + y **2 > z **2
def predicate_1(x, y, z):
    return True if ((x ** 2 + y ** 2) > z ** 2) else False
    

def data_generator(num_data, digits):
    print("Generating data...")
    current_data = 0
    seen = set()
    vectors = []
    labels = []
    while(current_data < num_data):
        f = lambda: int(''.join(np.random.choice(list("0123456789"))
                        for i in range(np.random.randint(1, digits + 1))))
        x, y, z = f(), f(), f()
        key = tuple(sorted((x,y)))
        if key in seen:
            continue
        current_data += 1
        seen.add(key)
        vectors.append(tuple((x,y,z)))
        labels.append(predicate_1(x, y, z))
    
    f = open("vectors.txt", "w+")
    f.write("\n".join("%s %s %s" % x for x in vectors))
    f.close()
    
    f = open("labels.txt", "w+")
    for i in range(len(labels)):
        f.write(str(labels[i]) + "\n")
    f.close()
    
    return(vectors, labels)