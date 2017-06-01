"""
Helper functions for dataset
"""

# Author: Joel Ong

import random

def write_to_file(datatype, data):
#   Randomly generat a number and add it to the file name to decrease the odds ot overwritting existing datas
    fileName = datatype + str(random.randint(1, 100000)) + ".txt"
    print("Writting to " + fileName)
    f = open(fileName, "w+")
    f.write("\n".join("%s" % value for value in data))
    f.close()
    
def split_train_validation_test_set(data, train_size, validation_size):
#   Return training, validation, testing data set
    random.shuffle(data)
    return (data[:train_size])[validation_size:], (data[:train_size])[:validation_size], data[train_size:]
    
        
def uniform_vector(minimum, maximum, size=3):
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