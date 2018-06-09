from models import World
from sklearn.datasets import load_boston, load_diabetes, make_regression


def main():
    # dataset = make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1)
    #features, target = make_regression(n_samples=1000, n_features=10, n_informative=4, n_targets=1, random_state=1)
    objects, target = load_boston(return_X_y=True)
    world = World(objects, target)
    world.evolve()

if  __name__ ==  "__main__" :
    main()
