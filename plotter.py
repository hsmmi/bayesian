from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def plot_decision_boundary(X_input, classes_input, theta, title=None):
    min_x1_input = np.min(X_input[:, 1])
    max_x1_input = np.max(X_input[:, 1])
    class_0_input = np.array(classes_input[0])
    class_1_input = np.array(classes_input[1])
    plt.plot(class_0_input[:, 1], class_0_input[:, 2], '.', label='class 0')
    plt.plot(class_1_input[:, 1], class_1_input[:, 2], '.', label='class 1')
    plt.plot(
        [min_x1_input, max_x1_input], [
            -theta[1]/theta[2]*min_x1_input-theta[0]/theta[2],
            -theta[1]/theta[2]*max_x1_input-theta[0]/theta[2]],
        'r-', label='decision boundary')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_decision_boundary_LDA(
        X_input, y_input, predicted_input, label_input,
        phi, mean, sigma, title=None):
    classes = []
    classes_corr = []
    classes_miss = []
    for label in label_input:
        where_label_i = np.where(y_input == label)[0]
        classes.append(X_input[where_label_i])
        where_label_i_corr = np.where(
            np.logical_and(predicted_input == y_input, y_input == label))[0]
        classes_corr.append(X_input[where_label_i_corr])
        where_label_i_miss = np.where(
            np.logical_and(predicted_input != y_input, y_input == label))[0]
        classes_miss.append(X_input[where_label_i_miss])
        plt.plot(classes_corr[label][:, 0], classes_corr[label][:, 1], '.')
        plt.plot(classes_miss[label][:, 0], classes_miss[label][:, 1], '.')
    for label1 in label_input:
        for label2 in label_input[label_input < label1]:
            sigma_inverse = np.linalg.inv(sigma)
            b = (
                -1/2 * mean[label1] @ sigma_inverse @ mean[label1].T +
                1/2 * mean[label2] @ sigma_inverse @ mean[label2].T +
                np.log(phi[label1]/phi[label2]))
            a = np.linalg.inv(sigma) @ (mean[label1] - mean[label2]).T
            min_x0 = np.min(np.concatenate(
                (classes[label1][:, 0], classes[label2][:, 0])))
            max_x0 = np.max(np.concatenate(
                (classes[label1][:, 0], classes[label2][:, 0])))
            min_x1 = ((-b-a[0]*min_x0)/a[1])[0]
            max_x1 = ((-b-a[0]*max_x0)/a[1])[0]
            plt.plot([min_x0, max_x0], [min_x1, max_x1], '-')
    plt.title(
        title +
        f'\n{np.round(a[0][0], 6)}x0+{np.round(a[1][0], 6)}x1' +
        f'+{np.round(b[0][0], 6)}=0')
    plt.show()


def plot_pdf(
        X_input, y_input, label_input,
        phi, mean, sigma, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for label in label_input:
        class_label = X_input[(y_input == label).flatten()]
        x0_mesh, x1_mesh = np.mgrid[
            np.min(class_label[:, 0]):np.max(class_label[:, 0]):0.01,
            np.min(class_label[:, 1]):np.max(class_label[:, 1]):0.01]
        pos = np.dstack((x0_mesh, x1_mesh))
        prob = multivariate_normal(mean[label][0], sigma).pdf(pos)
        ax.plot_surface(x0_mesh, x1_mesh, prob)
    plt.title(title)
    plt.show()


def plot_contour(
        X_input, y_input, label_input,
        phi, mean, sigma, title=None):
    for label in label_input:
        class_label = X_input[(y_input == label).flatten()]
        x0_mesh, x1_mesh = np.mgrid[
            np.min(class_label[:, 0]):np.max(class_label[:, 0]):0.01,
            np.min(class_label[:, 1]):np.max(class_label[:, 1]):0.01]
        pos = np.dstack((x0_mesh, x1_mesh))
        plt.contour(
            x0_mesh, x1_mesh, multivariate_normal(
                mean[label][0], sigma).pdf(pos), levels=10)
    for label1 in label_input:
        for label2 in label_input[label_input < label1]:
            sigma_inverse = np.linalg.inv(sigma)
            b = (
                -1/2 * mean[label1] @ sigma_inverse @ mean[label1].T +
                1/2 * mean[label2] @ sigma_inverse @ mean[label2].T +
                np.log(phi[label1]/phi[label2]))
            a = np.linalg.inv(sigma) @ (mean[label1] - mean[label2]).T
            min_x0 = (np.min(np.concatenate(
                (X_input[(y_input == label1).flatten()][:, 0],
                 X_input[(y_input == label2).flatten()][:, 0]))))
            max_x0 = (np.max(np.concatenate(
                (X_input[(y_input == label1).flatten()][:, 0],
                 X_input[(y_input == label2).flatten()][:, 0]))))
            min_x1 = ((-b-a[0]*min_x0)/a[1])[0]
            max_x1 = ((-b-a[0]*max_x0)/a[1])[0]
            plt.plot([min_x0, max_x0], [min_x1, max_x1], '-')
    plt.title(title)
    plt.show()
