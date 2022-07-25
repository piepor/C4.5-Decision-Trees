from dataclasses import dataclass
from predicting import get_distribution


@dataclass
class LeafClasses:
    """ class managing the final classes distribution in a leaf after the training """
    classes: dict

    def get_class_name(self) -> str:
        """ get the class with maximum number of samples """
        return max(zip(self.classes.values(), self.classes.keys()))[1]

    def get_classes(self) -> dict:
        """ return the classes dictionary """
        return self.classes

    def get_classes_distribution(self) -> dict:
        """ Returns the distribution over the classes inside the leaf """
        return get_distribution(self.classes)
