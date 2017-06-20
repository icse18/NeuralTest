# x ^ 2 + y ^ 2 > z ^ 2
def predicate_1(sample):
    return 1 if ((sample[0] ** 2 + sample[1] ** 2 > sample[2] ** 2)) else 0

# (x * y ^ 2) / (x ^ 2 + y ^ 2) > z
def predicate_2(sample):
    return 1 if ((sample[0] * (sample[1] ** 2)) / ((sample[0] ** 2 + sample[1] ** 2))) > sample[3] else 0