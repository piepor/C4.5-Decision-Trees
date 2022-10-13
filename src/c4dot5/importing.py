import pickle


def import_classifier(file_path: str):
    with open(file_path, 'rb') as file:
        classifier = pickle.load(file)
    return classifier
