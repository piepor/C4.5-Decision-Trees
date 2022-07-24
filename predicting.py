""" functions for the prediction phase """

def continuous_test_fn(threshold: float, attr_value: float) -> bool:
    """ Test if the attribute value is less than the threshold """
    return attr_value <= threshold
