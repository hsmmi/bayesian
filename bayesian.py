import numpy as np
from preprocessing import read_dataset_to_X_and_y, train_test_split


class Bayesian:
    def __init__(self, file, file_test=None, normalization_method=None):
        '''
        Read file and initial our baysian module
        '''
        if file_test is not None:
            self.X_train, self.y_train = read_dataset_to_X_and_y(
                file, normalization=normalization_method)
            self.X_test, self.y_test = read_dataset_to_X_and_y(
                file_test, normalization=normalization_method)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(
                    'dataset/my_QDA_dataset1.csv', train_size=0.80))[0:4]

        self.diffrent_label, self.count_diffrent_label_train = np.unique(
            self.y_train, return_counts=True)
        self.number_of_sample_train = self.X_train.shape[0]
        self.number_of_sample_test = self.X_test.shape[0]
        self.number_of_feature = self.X_train.shape[1]
        self.number_of_class = self.diffrent_label.shape[0]
        self.mean_class = None
        self.sigma = None
        self.probabilities_train = None
        self.predicted_label_train = None
        self.accuracy_train = None
        self.probabilities_test = None
        self.predicted_label_test = None
        self.accuracy_test = None

    def find_mean_class_i(self, label):
        '''
        Get label and find mean features of samples that has our label
        '''
        where_label = np.where(self.y_train == label)[0]
        sample_label = self.X_train[where_label]
        mean_sample_label = np.mean(sample_label, axis=0).reshape(-1, 1)
        return mean_sample_label

    def find_mean_classes(self):
        '''
        Find mean features of all classes
        '''
        return np.array(
            [self.find_mean_class_i(label).T for label in self.diffrent_label])

    def find_sigma_LDA(self):
        '''
        find sigma
        '''
        X_demean_in_class = np.copy(self.X_train)
        for label in self.diffrent_label.astype(int):
            where_label = np.where(self.y_train == label)[0]
            X_demean_in_class[where_label] -= self.mean_class[label]
        return np.cov(X_demean_in_class.T)

    def find_parameters_LDA(self):
        self.phi = (
            self.count_diffrent_label_train / self.number_of_sample_train)
        self.mean_class = self.find_mean_classes()
        self.sigma = self.find_sigma_LDA()

    def find_probabilities_LDA(self, X_input):
        number_of_sample = X_input.shape[0]
        probabilities = np.zeros((number_of_sample, self.number_of_class))
        coefficient = 1  # It's constant and doesn't change argmax
        covariance_inverse = np.linalg.inv(self.sigma)
        for label in self.diffrent_label.astype(int):
            X_demean = X_input - self.mean_class[label]
            P_X_given_y = coefficient * -0.5 * (
                    ((X_demean @ covariance_inverse) * X_demean)
                    @ np.ones((self.number_of_feature, 1)))
            prior = self.phi[label]
            probabilities[:, label:label+1] = P_X_given_y + prior
        return probabilities

    def find_prediction(self, probabilities):
        return np.argmax(probabilities, axis=1).reshape(-1, 1)

    def find_accuracy(self, y_input, predicted_label):
        return sum(y_input == predicted_label) / y_input.shape[0]

    def runner_train_LDA(self):
        self.find_parameters_LDA()
        self.probabilities_train = self.find_probabilities_LDA(self.X_train)
        self.predicted_label_train = self.find_prediction(
            self.probabilities_train)
        self.accuracy_train = self.find_accuracy(
            self.y_train, self.predicted_label_train)

    def runner_test_LDA(self):
        self.probabilities_test = self.find_probabilities_LDA(self.X_test)
        self.predicted_label_test = self.find_prediction(
            self.probabilities_test)
        self.accuracy_test = self.find_accuracy(
            self.y_test, self.predicted_label_test)

    def runner_LDA(self):
        self.runner_train_LDA()
        self.runner_test_LDA()

    def find_sigma_QDA(self):
        '''
        find sigma
        '''
        sigma = []
        for label in self.diffrent_label.astype(int):
            where_label = np.where(self.y_train == label)[0]
            X_demean_class_i = (
                self.X_train[where_label] - self.mean_class[label])
            sigma.append(np.cov(X_demean_class_i.T))
        return sigma

    def find_parameters_QDA(self):
        self.phi = (
            self.count_diffrent_label_train / self.number_of_sample_train)
        self.mean_class = self.find_mean_classes()
        self.sigma = self.find_sigma_QDA()

    def find_probabilities_QDA(self, X_input):
        number_of_sample = X_input.shape[0]
        probabilities = np.zeros((number_of_sample, self.number_of_class))
        for label in self.diffrent_label.astype(int):
            coefficient = 1 / (
                (2 * np.pi) ** (self.number_of_feature/2)
                * np.sqrt(abs(np.linalg.det(self.sigma[label]))))
            covariance_i_inverse = np.linalg.inv(self.sigma[label])
            X_demean = X_input - self.mean_class[label]
            P_X_given_y = coefficient * -0.5 * np.exp(
                    ((X_demean @ covariance_i_inverse) * X_demean)
                    @ np.ones((self.number_of_feature, 1)))
            prior = self.phi[label]
            probabilities[:, label:label+1] = P_X_given_y * prior
        return probabilities

    def runner_train_QDA(self):
        self.find_parameters_QDA()
        self.probabilities_train = self.find_probabilities_QDA(self.X_train)
        self.predicted_label_train = self.find_prediction(
            self.probabilities_train)
        self.accuracy_train = self.find_accuracy(
            self.y_train, self.predicted_label_train)

    def runner_test_QDA(self):
        self.probabilities_test = self.find_probabilities_QDA(self.X_test)
        self.predicted_label_test = self.find_prediction(
            self.probabilities_test)
        self.accuracy_test = self.find_accuracy(
            self.y_test, self.predicted_label_test)

    def runner_QDA(self):
        self.runner_train_QDA()
        self.runner_test_QDA()
