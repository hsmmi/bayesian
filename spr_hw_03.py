import numpy as np
from bayesian import Bayesian
from my_io import generate_dataset
from plotter import (
    plot_contour_LDA, plot_contour_QDA, plot_decision_boundary_LDA,
    plot_decision_boundary_QDA, plot_pdf_LDA, plot_pdf_QDA)


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
        plot_pdf_LDA(
            LDA.X_train, LDA.y_train,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Train{i} - PDF')
        plot_pdf_LDA(
            LDA.X_test, LDA.y_test,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Test{i} - PDF')
        plot_contour_LDA(
            LDA.X_train, LDA.y_train,
            LDA.diffrent_label, LDA.phi, LDA.mean_class,
            LDA.sigma, f'LDA dataset BC-Train{i} - Contour')
        plot_contour_LDA(
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
    for i in range(1, 3):
        QDA = Bayesian('dataset/my_QDA_dataset1.csv')
        QDA.runner_QDA()
        print(
            f'In dataset my_QDA_dataset{i} accuracy train is ',
            f'{QDA.accuracy_train} and accuracy test is {QDA.accuracy_test}')
        plot_decision_boundary_QDA(
            QDA.X_train, QDA.y_train, QDA.predicted_label_train,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Decision Boundary')
        plot_decision_boundary_QDA(
            QDA.X_test, QDA.y_test, QDA.predicted_label_test,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Decision Boundary')
        plot_pdf_QDA(
            QDA.X_train, QDA.y_train,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Decision Boundary')
        plot_pdf_QDA(
            QDA.X_test, QDA.y_test,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Decision Boundary')
        plot_contour_QDA(
            QDA.X_train, QDA.y_train,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Contour')
        plot_contour_QDA(
            QDA.X_test, QDA.y_test,
            QDA.diffrent_label, QDA.phi, QDA.mean_class,
            QDA.sigma, f'QDA dataset my_QDA_dataset{i} - Contour')


quadratic_discriminant_analysis_runner()  # uncomment to run QDA
