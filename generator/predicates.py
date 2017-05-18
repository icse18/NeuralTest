# Predicate 1: x ** 2 + y **2 > z **2
def predicate_1(x, y, z):
    return 1 if ((x ** 2 + y ** 2) > z ** 2) else 0

# Predicate 2: (x * y) / (x ** 2 + y ** 2) > z
def predicate_2(x, y, z):
    return 1 if ((x * y) / (x ** 2 + y ** 2)) > z else 0
    
# Predicate 3: (x * y ** 2) / (x ** 2 + y ** 4) > z 
def predicate_3(x, y, z):
    return 1 if ((x * y ** 2) / (x ** 2 + y ** 4)) > z else 0