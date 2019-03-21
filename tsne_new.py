from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt 
from matplotlib import offsetbox
import pickle
import numpy as np
import torch
import datasets
from torchvision import transforms

def plot_embedding(X, label, origin_img, show_origin_image=False):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(label[i]),
                color=plt.cm.Set1(label[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9}) 
    if show_origin_image and hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
               offsetbox.OffsetImage(origin_img[i], cmap=plt.cm.gray_r),
               X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
   
   
def main(input_path, output_path, sample_num=500):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNISTInstance(root='./data', train=True, transform=transform)
    loader = torch.utils.data.DataLoader(mnist, batch_size=16, shuffle=False)
    total_img = []
    total_label = []
    for idx, data in enumerate(loader):
        img, label, index = data
        total_img.append(img.permute(0, 2, 3, 1).contiguous().numpy())
        total_label.append(label.numpy())
    total_img = np.concatenate(total_img, axis=0)
    total_label = np.concatenate(total_label, axis=0)
    print('data load ok')
    print('label', total_label.shape)
    feature = pickle.load(open(input_path, 'rb'), encoding='bytes')
    print('feature', feature.shape)
    print('img', total_img.shape)
    #feature = np.random.randn(label.shape[0], 1024)
    random_index = np.random.randint(feature.shape[0], size=sample_num)
    feature_sample = feature[random_index]
    label_sample = total_label[random_index]
    origin_img_sample = total_img[random_index]

    handle_tsne = TSNE(n_components=2, init='pca', random_state=0)
    feature_tsne = handle_tsne.fit_transform(feature_sample)
    plot_embedding(feature_tsne, label_sample, origin_img_sample)
    plt.show()
    plt.savefig(output_path)


if __name__ == "__main__":
    input_path = '/home/xinlei/icml2019/mnist_features/mmbk_0.4943844261876095.pkl'
    #input_path = '/home/xinlei/git/icml2019/mnist_method5_1024_new2/mmbk_1980.pkl'
    output_path = 'mnist_result.jpg'
    main(input_path, output_path)
