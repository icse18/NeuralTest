from util import uniform_vector
from predicates import predicate_1

def data_generator(predicate, total_data, vector_size, min_val, max_val):
    count = 0
    listOfVectors = []
    listOfLabels = []
    seen = set()
    while count < total_data:
        outputVector = uniform_vector(min_val, max_val, vector_size)
        key = tuple(sorted(outputVector))
        if key in seen:
            continue
        seen.add(key)
        label = predicate(outputVector)
        listOfVectors.append(outputVector)
        listOfLabels.append(label)
        count += 1    
    return listOfVectors, listOfLabels