import numpy as np
from preprocessing import read_dataset_to_X_and_y


class Bayesian:
    def __init__(self, file, normalization_method=None):
        '''
        Read file and initial our baysian module
        '''
        self.X, self.y = read_dataset_to_X_and_y(
          file, normalization=normalization_method)

        self.diffrent_label, self.count_diffrent_label = np.unique(
            self.y, return_counts=True)
        self.number_of_sample = self.X.shape[0]
        self.number_of_feature = self.X.shape[1]
        self.number_of_class = self.diffrent_label.shape[0]
        self.mean_class = None
        self.covariance = None
        self.probabilities = None
        self.predicted_label = None
        self.accuracy = None
        self.test_accuracy = None

    def find_mean_class_i(self, label):
        '''
        Get label and find mean features of samples that has our label
        '''
        where_label = np.where(self.y == label)[0]
        sample_label = self.X[where_label]
        mean_sample_label = np.mean(sample_label, axis=0).reshape(-1, 1)
        return mean_sample_label

    def find_mean_classes(self):
        '''
        Find mean features of all classes
        '''
        return np.array(
            [self.find_mean_class_i(label).T for label in self.diffrent_label])

    def find_covariance(self):
        '''
        find sigma
        '''
        X_demean_in_class = np.copy(self.X)
        for label in self.diffrent_label.astype(int):
            where_label = np.where(self.y == label)[0]
            X_demean_in_class[where_label] -= self.mean_class[label]
        return np.cov(X_demean_in_class.T)

    def find_parameters(self):
        self.phi = self.count_diffrent_label / self.number_of_sample
        self.mean_class = self.find_mean_classes()
        self.covariance = self.find_covariance()

    def find_probabilities(self, X_input=None):
        if X_input is None:
            X_input = self.X
        number_of_sample = X_input.shape[0]
        probabilities = np.zeros((number_of_sample, self.number_of_class))
        coefficient = 1  # It's constant and doesn't change argmax
        covariance_inverse = np.linalg.inv(self.covariance)
        for label in self.diffrent_label.astype(int):
            X_demean = X_input - self.mean_class[label]
            P_X_given_y = coefficient * -0.5 * (
                    ((X_demean @ covariance_inverse) * X_demean)
                    @ np.ones((self.number_of_feature, 1)))
            prior = self.phi[label]
            probabilities[:, label:label+1] = P_X_given_y + prior
        return probabilities

    def find_prediction(self, probabilities=None):
        if probabilities is None:
            probabilities = self.probabilities
        return np.argmax(probabilities, axis=1).reshape(-1, 1)

    def find_accuracy(self, y_input=None, predicted_label=None):
        if predicted_label is None:
            predicted_label = self.predicted_label
        if y_input is None:
            y_input = self.y
        return sum(y_input == predicted_label) / y_input.shape[0]

    def runner(self):
        self.find_parameters()
        self.probabilities = self.find_probabilities()
        self.predicted_label = self.find_prediction()
        self.accuracy = self.find_accuracy()

    def find_test_accuracy(self, file, normalization_method=None):
        X_test, y_test = read_dataset_to_X_and_y(
          file, normalization=normalization_method)
        probabilities = self.find_probabilities(X_test)
        predicted_label = self.find_prediction(probabilities)
        self.test_accuracy = self.find_accuracy(y_test, predicted_label)
        return self.test_accuracy



