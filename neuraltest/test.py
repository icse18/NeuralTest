from synthetic_data_generator import data_generator
from util import write_to_file
from predicates import predicate_1

listOfVectors, listOfLabels = data_generator(predicate_1, 30000, 3, -40000, 40000)
write_to_file("data", "vectors.txt", listOfVectors)
write_to_file("data", "labels.txt", listOfLabels)