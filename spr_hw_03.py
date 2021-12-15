from bayesian import Bayesian
from preprocessing import read_dataset_to_X_and_y

# def linear_discriminant_analysis_runner ():
X_train, y_train = read_dataset_to_X_and_y('dataset/BC-Train1')
X_test, y_test = read_dataset_to_X_and_y('dataset/BC-Test1')

# linear_discriminant_analysis_runner() # uncomment to run GLDA
GLDA1 = Bayesian('dataset/BC-Train1.csv')
GLDA1.runner()
GLDA1.find_test_accuracy('dataset/BC-Test1.csv')
print(GLDA1.accuracy)
print(GLDA1.test_accuracy)
GLDA2 = Bayesian('dataset/BC-Train2.csv')
GLDA2.runner()
GLDA2.find_test_accuracy('dataset/BC-Test2.csv')
print(GLDA2.accuracy)
print(GLDA2.test_accuracy)
# This is a new line that ends the file.