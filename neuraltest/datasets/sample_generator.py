"""
Generate samples of synthetic data sets.
"""

# Author: Joel Ong

from helper import uniform_vector, write_to_file
from predicates import continuous_predicate_1
from predicates import recursive_predicate_1
import random

def generate_random_label_data(num_data, minimum, maximum):
    count = 0
    listOfVectors = []
    listOfLabels = []
    seen = set()
    while count < num_data:
        outputVector = uniform_vector(minimum, maximum)
        key = tuple(sorted(outputVector))
        if key in seen:
            continue
        label = continuous_predicate_1(outputVector)
        listOfVectors.append(outputVector)
        listOfLabels.append(label)
        count += 1
    write_to_file("vectors", listOfVectors)
    write_to_file("labels", listOfLabels)
    return listOfVectors, listOfLabels

def generate_random_unlabel_data(num_data, minimum, maximum, boolean):
    count = 0
    listOfVectors = []
    seen = set()
    while count < num_data:
        outputVector = uniform_vector(minimum, maximum)
        key = tuple(sorted(outputVector))
        if key in seen:
            continue
        label = continuous_predicate_1(outputVector)
        if label != boolean:
            continue
        listOfVectors.append(outputVector)
        count += 1
    write_to_file("vectors", listOfVectors)
    return listOfVectors

def generate_recursive_data(num_data, maximum):
    count = 0
    listOfVectors = []
    listOfLabels = []
    seen = set()
    while count < num_data:
        predicate_input = random.uniform(0, maximum)
        label = recursive_predicate_1(predicate_input)
        key = label
        if key in seen:
            continue
        listOfVectors.append(predicate_input)
        listOfLabels.append(label)
        count += 1
    write_to_file("vectors", listOfVectors)
    write_to_file("labels", listOfLabels)
    return listOfVectors, listOfLabels       