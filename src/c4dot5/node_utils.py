import numpy as np

def get_distribution(classes_dictionary) -> dict:
    """ Returns the distribution over the classes inside a dictionary """
    distribution_dict = dict.fromkeys(classes_dictionary.keys())
    total_count = sum(classes_dictionary.values())
    for key in distribution_dict:
        distribution_dict[key] = np.round(classes_dictionary.copy()[key] / total_count, 4)
    return distribution_dict

def continuous_test_fn(threshold: float, attr_value: float) -> bool:
    """ Test if the attribute value is less than the threshold """
    return attr_value <= threshold
