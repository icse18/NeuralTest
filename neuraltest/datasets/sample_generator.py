"""
Generate samples of synthetic data sets.
"""

# Author: Joel Ong

from helper import uniform_vector, write_to_file
from predicates import predicate_1

def random_label_data(num_data, minimum, maximum):
    count = 0
    listOfVectors = []
    listOfLabels = []
    seen = set()
    while count < num_data:
        outputVector = uniform_vector(minimum, maximum)
        key = tuple(sorted(outputVector))
        if key in seen:
            continue
        label = predicate_1(outputVector)
        listOfVectors.append(outputVector)
        listOfLabels.append(label)
        count += 1
    write_to_file("vectors", listOfVectors)
    write_to_file("labels", listOfLabels)
    return listOfVectors, listOfLabels

def random_unlabel_data(num_data, minimum, maximum, boolean):
    count = 0
    listOfVectors = []
    seen = set()
    while count < num_data:
        outputVector = uniform_vector(minimum, maximum)
        key = tuple(sorted(outputVector))
        if key in seen:
            continue
        label = predicate_1(outputVector)
        if label != boolean:
            continue
        listOfVectors.append(outputVector)
        count += 1
    write_to_file("vectors", listOfVectors)
    return listOfVectors