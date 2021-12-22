import numpy as np


def accuracy_score(y_input, predicted):
    return np.mean(y_input == predicted)


def precision_score(y_input, predicted):
    return (y_input * predicted).sum() / predicted.sum()


def recall_score(y_input, predicted):
    return (y_input * predicted).sum() / y_input.sum()


def f1_score(y_input, predicted):
    return 2 * (y_input * predicted).sum() / (y_input + predicted).sum()
