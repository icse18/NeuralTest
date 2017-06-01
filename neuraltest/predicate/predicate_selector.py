import continuous_predicates

def predicate_selector(predicate_type, num):
    if predicate_type == "continuous":
        if num == 1:
            return continuous_predicates.predicate_1, 3