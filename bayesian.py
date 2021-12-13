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
        print('hi')

    def probabilities(self):
        probabilities = np.zeros((self.number_of_sample, self.number_of_class))
        coefficient 
        
        print('hi')

b = Bayesian('dataset/BC-Train1.csv')
b.find_parameters()
b.probabilities()
print('hi')
