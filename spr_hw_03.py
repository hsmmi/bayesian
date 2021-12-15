import numpy as np
from bayesian import Bayesian
from my_io import generate_dataset
from plotter import plot_contour, plot_decision_boundary_LDA, plot_pdf


def linear_discriminant_analysis_runner():
    for i in range(1, 3):
        LDA = Bayesian(f'dataset/BC-Train{i}.csv', f'dataset/BC-Test{i}.csv')
        LDA.runner_LDA()
        print(
            f'In dataset BC-{i} accuracy train is {LDA.accuracy_train}',
            f' and accuracy test is {LDA.accuracy_test}')
        plot_decision_boundary_LDA(
            LDA.X_train, LDA.y_train, LDA.predicted_label_train,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Train{i} - Decision Boundary')
        plot_decision_boundary_LDA(
            LDA.X_test, LDA.y_test, LDA.predicted_label_test,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Test{i} - Decision Boundary')
        plot_pdf(
            LDA.X_train, LDA.y_train,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Train{i} - PDF')
        plot_pdf(
            LDA.X_test, LDA.y_test,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Test{i} - PDF')
        plot_contour(
            LDA.X_train, LDA.y_train,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Train{i} - Contour')
        plot_contour(
            LDA.X_test, LDA.y_test,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Test{i} - Contour')


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
