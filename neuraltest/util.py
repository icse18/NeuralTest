import random

def write_to_file(filename, data):
    print("Writting to " + filename)
    f = open(filename, "w+")
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