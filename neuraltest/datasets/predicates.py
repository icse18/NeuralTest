"""
List of predicates for data generation
"""

# Author: Joel Ong

def predicate_1(vector):
    #    x^2 + y^2 > z^2
    return 1 if ((vector[0] ** 2 + vector[1] ** 2 > vector[2] ** 2)) else 0