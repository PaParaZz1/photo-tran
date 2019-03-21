from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cPickle as pickle


def main(path):
    feature = pickle.load(open(path))
    print(type(feature))


if __name__ == "__main__":
    path = 'mnist_feature/'
    main(path)
