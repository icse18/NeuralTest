"""
List of predicates for data generation
"""

# Author: Joel Ong

from math import sqrt

def continuous_predicate_1(vector):
#    x^2 + y^2 > z^2
    return 1 if ((vector[0] ** 2 + vector[1] ** 2 > vector[2] ** 2)) else 0

def recursive_predicate_1(n):
#   Recursive function such as Fibonacci function
    return (((1 + sqrt(5)) ** n) - (1 - sqrt(5)) ** n) / ((2 ** n) * sqrt(5))