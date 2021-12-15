import numpy as np
from bayesian import Bayesian
from my_io import generate_dataset


def linear_discriminant_analysis_runner():
    LDA1 = Bayesian('dataset/BC-Train1.csv', 'dataset/BC-Test1.csv')
    LDA1.runner_LDA()
    print(
        f'In dataset BC-1 accuracy train is {LDA1.accuracy_train}',
        f' and accuracy test is {LDA1.accuracy_test}')
    LDA2 = Bayesian('dataset/BC-Train2.csv', 'dataset/BC-Test2.csv')
    LDA2.runner_LDA()
    print(
        f'In dataset BC-2 accuracy train is {LDA2.accuracy_train}',
        f' and accuracy test is {LDA2.accuracy_test}')


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


def quadratic_discriminant_analysis_runner():
    QDA1 = Bayesian('dataset/my_QDA_dataset1.csv')
    QDA1.runner_QDA()
    print(
        f'In dataset my_QDA_dataset1 accuracy train is {QDA1.accuracy_train}',
        f' and accuracy test is {QDA1.accuracy_test}')
    QDA2 = Bayesian('dataset/my_QDA_dataset2.csv')
    QDA2.runner_QDA()
    print(
        f'In dataset my_QDA_dataset2 accuracy train is {QDA2.accuracy_train}',
        f' and accuracy test is {QDA2.accuracy_test}')
    QDA3 = Bayesian('dataset/BC-Train1.csv', 'dataset/BC-Test1.csv')
    QDA3.runner_QDA()
    print(
        f'In dataset BC-1 accuracy train is {QDA3.accuracy_train}',
        f' and accuracy test is {QDA3.accuracy_test}')


# quadratic_discriminant_analysis_runner()  # uncomment to run QDA
