import numpy as np
import math

def generate(predicate, num_train_data, num_test_data):
    num_data = num_train_data + num_test_data
    current_data = 0
    seen = set()
    vectors = []
    labels = []
    num_true = 0
    while(current_data < num_data):
        sample = np.random.randint(-math.pow(10, 4), math.pow(10, 4) + 1, size = 3)
        x, y, z = sample[0], sample[1], sample[2]
        key = tuple(sorted((x,y)))
        if key in seen:
            continue
        result = predicate(x,y,z)
        if (num_true > current_data / 2 and result == 1):
            continue
        if (result == 1):
            num_true += result
        current_data += 1
        seen.add(key)
        vectors.append(tuple((x,y,z)))
        labels.append(result)

    f = open("datasets\\train_vectors.txt", "w+")
    f.write("\n".join("%s %s %s" % value for value in vectors[:num_train_data]))
    f.close()
    
    f = open("datasets\\train_labels.txt", "w+")
    f.write("\n".join("%s" % label for label in labels[:num_train_data]))
    f.close()
    
    f = open("datasets\\test_vectors.txt", "w+")
    f.write("\n".join("%s %s %s" % value for value in vectors[num_train_data:]))
    f.close()
    
    f = open("datasets\\test_labels.txt", "w+")
    f.write("\n".join("%s" % label for label in labels[num_train_data:]))
    f.close()
    
    print("Number of True labels: %s" % num_true)
    print("Number of False labels: %s" % (num_data - num_true))