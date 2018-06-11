from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import numpy as np
from models import World
from sklearn.datasets import load_boston, load_diabetes, make_regression, load_wine

def classification_errors_counter(predicted, known):
    diff = predicted - known
    errors = np.count_nonzero(diff)
    return errors

def main():
    dataset = make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1)
    objects, target = load_boston(return_X_y=True)
    #objects = objects[:,[3, 4, 5, 7, 8, 9, 10, 12]]
    x_train, x_test, y_train, y_test = train_test_split(objects, target, test_size=0.30,
                                                        random_state=10)

    world = World(x_train, x_test, y_train, y_test)
    world.evolve()
    world.draw()

if  __name__ ==  "__main__" :
    main()
