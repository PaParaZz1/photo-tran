from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import cPickle as pickle
import pickle


def main(path):
    feature = pickle.load(open(path, 'rb'), encoding='bytes')
    print(feature.shape)

    


if __name__ == "__main__":
    #path = '~/icml2019/mnist_features/mmbk_0.4943844261876095.pkl'
    path = '/home/xinlei/icml2019/mnist_features/mmbk_0.4943844261876095.pkl'
    main(path)
