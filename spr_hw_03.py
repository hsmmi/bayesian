import numpy as np
from pandas.core.frame import DataFrame
from bayesian import Bayesian_LDA
from my_io import generate_dataset


def linear_discriminant_analysis_runner():
    LDA1 = Bayesian_LDA('dataset/BC-Train1.csv')
    LDA1.runner()
    LDA1.find_test_accuracy('dataset/BC-Test1.csv')
    print(LDA1.accuracy)
    print(LDA1.test_accuracy)
    LDA2 = Bayesian_LDA('dataset/BC-Train2.csv')
    LDA2.runner()
    LDA2.find_test_accuracy('dataset/BC-Test2.csv')
    print(LDA2.accuracy)
    print(LDA2.test_accuracy)


# linear_discriminant_analysis_runner()  # uncomment to run LDA


def generate_dataset_QLA():
    mean_ds_1 = np.array([
        [3, 6],
        [5, 4],
        [6, 6]])

    cov_ds_1 = np.array([
        [[1.5, 0], [0, 1.5]],
        [[2, 0], [0, 2]],
        [[1, 0], [0, 1]]])

    mean_ds_2 = np.array([
        [3, 6],
        [5, 4],
        [6, 6]])

    cov_ds_2 = np.array([
        [[1.5, 0.1], [0.1, 1.5]],
        [[1, -0.20], [-0.20, 2]],
        [[2, -0.25], [-0.25, 1.5]]])

    generate_dataset('dataset/my_QDA_dataset1.csv', mean_ds_1, cov_ds_1, 500)
    generate_dataset('dataset/my_QDA_dataset2.csv', mean_ds_2, cov_ds_2, 500)


# generate_dataset_QLA()  # uncomment to create datasets again
