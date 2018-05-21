from models import World
from sklearn.datasets import load_boston


def main():
    features, target = load_boston(return_X_y=True)
    world = World(features, target)
    world.evolve()

if  __name__ ==  "__main__" :
    main()
