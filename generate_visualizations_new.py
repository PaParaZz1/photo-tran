import torch
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
import scipy.stats as stats
import datasets
import numpy as np
import models
import math
from torch.autograd import Variable
import pdb
import copy
import os
import pickle as pkl
import time
import torchvision
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from ggplot import *
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import datasets

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--low-dim', default=1024, type=int, metavar='D', help='feature dimension')
parser.add_argument('--mse', action='store_true')
parser.add_argument('--with-mmbk', action='store_true')
parser.add_argument('--with-regu', action='store_true')
parser.add_argument('--log-name', default='log_no_mse.txt')
parser.add_argument('--update-freq', default=1)
parser.add_argument('--normalize', action='store_true', default=True)
parser.add_argument('--new-loss', action='store_true')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--method2', action='store_true', help='L2 Loss, regular update')
parser.add_argument('--method3', action='store_true', help='L2 Loss, new update')
parser.add_argument('--method4', action='store_true', help='New loss, new update')
parser.add_argument('--method1', action='store_true', help='ZR')
parser.add_argument('--method5', action='store_true', help='updated method4')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--log-path', type=str, default='model')
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--with-mmbk2', action='store_true')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--k', type=int, default=200)
parser.add_argument('--num-class', type=int, default=10)
parser.add_argument('--proportion',  type=float, default=1.0)

args = parser.parse_args()
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def get_mmbk(ndata, low_dim=128, device='cuda', requires_grad=False):
    mmbk = np.random.uniform(low=-1.0, high=1.0, size=[ndata, low_dim])
    mmbk = Variable(torch.from_numpy(mmbk).to(device).float())
    mmbk = mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, low_dim)
    mmbk = Variable(mmbk.data, requires_grad=requires_grad)
    return mmbk

temp = args.temp
batch_size = args.batch_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.dataset == 'cifar10':
    trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train, proportion=args.proportion)
    testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'mnist':
    transform = transforms.Compose([transforms.RandomResizedCrop(size=28, scale=(0.2,1)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNISTInstance(root='./data', train=True, transform=transform, download=True, proportion=args.proportion)
    testset = datasets.MNISTInstance(root='./data', train=False, transform=transform_test, download=True)
elif args.dataset == 'cub':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=220, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = CUB_Dataset('CUB_200_2011/train.pkl', transform=transform)
    testset = CUB_Dataset('CUB_200_2011/test.pkl', transform=transform_test)
elif args.dataset == 'svhn':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.SVHNInstance(root='./data', split='train', transform=transform, download=True, proportion=args.proportion)
    testset = datasets.SVHNInstance(root='./data', split='test', transform=transform_test, download=True)
elif args.dataset == 'pong':
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = Pong_Dataset(file_path='pong_data', label_path='pong_label_data', transform=transform, train=True)
    testset = Pong_Dataset(file_path='pong_data', label_path='pong_label_data', transform=transform_test, train=False)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
ndata = trainset.__len__()
print(ndata)

net = models.__dict__[args.arch](low_dim=args.low_dim)#, l2norm=args.normalize)
if device == 'cuda':
    net = nn.DataParallel(net)
net = net.to(device)
mmbk = get_mmbk(ndata, args.low_dim, device, False)
mmbk2 = get_mmbk(ndata, args.low_dim, device, False)
mmbk_label = Variable(torch.zeros(ndata)).to(device).long()
old_mmbk = copy.copy(mmbk)
old_mmbk.requires_grad = False
if mmbk.requires_grad:
    opt2 = optim.Adam([mmbk], lr=args.lr, amsgrad=True)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

def test(net, trainloader, testloader, input_mmbk, k, num_class=10):
    net.eval()
    top1 = 0.
    top5 = 0.
    total = 0
    norm = input_mmbk.pow(2).sum(1, keepdim=True).pow(0.5)
    mmbk = input_mmbk.div(norm + 1e-16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
    try:
        mmbk_label = torch.LongTensor(temploader.dataset.train_labels).to(device)
    except:
        mmbk_label = torch.LongTensor(temploader.dataset.labels).to(device)
    # trainloader.dataset.transform = transform_bak
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(k, num_class).to(device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
            targets = targets.to(device)#cuda()#async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            dist = torch.mm(features, mmbk.t())
            yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
            candidates = mmbk_label.view(1, -1).expand(batchSize, -1)
            retrieval_one_hot = torch.zeros(batchSize * k, num_class).to(device)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(0.1).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, num_class), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
    print(top1*100./total)
    return top1/total


def calculate_newloss(args, net, inputs, targets, indexes, features, mmbk, mmbk2, device='cuda', ndata=50000, optimizer_net=None, update_net=True):
    with torch.no_grad():
        mmbk2[indexes.long(), :] = mmbk2[indexes.long(), :] * 0.5 + 0.5 * features.data
    tmp_mmbk = mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, args.low_dim)
    dist_mat = torch.bmm(features.unsqueeze(0), tmp_mmbk.detach().to(device).t().unsqueeze(0)).squeeze()
    dist_mat = torch.clamp(dist_mat, min=-1.0, max=1.0)
    dist_mat = 0.5 * (torch.acos(dist_mat)) ** 2.0
    II = Variable(torch.ones(dist_mat.size()), requires_grad=False).to(device)
    II[torch.arange(inputs.size(0)).long(), indexes.long()] *= 0.0
    dist_mat = torch.mul(dist_mat, II)
    loss1 = -1.0 * dist_mat.sum() /(inputs.size(0) * (ndata-1))

    mmbk_norm = mmbk.pow(2).sum(1, keepdim=True).pow(0.5)
    tmp_mmbk = mmbk.div(mmbk_norm)
    dist_mat2 = torch.bmm(features.unsqueeze(0), tmp_mmbk.detach().to(device).t().unsqueeze(0)).squeeze()
    dist_mat2 = torch.clamp(dist_mat2, min=-1.0, max=1.0)
    dist_mat2 = (torch.acos(dist_mat2))**2.0
    II = Variable(torch.zeros(dist_mat2.size()), requires_grad=False).to(device)
    II[torch.arange(inputs.size(0)).long(), indexes.long()] += 1.0
    dist_mat2 = torch.mul(dist_mat2, II)
    loss2 = dist_mat2.sum() / (inputs.size(0))
    loss = loss1 + loss2
    loss.backward()
    optimizer_net.step()

    tmp_mmbk = mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, args.low_dim)
    # calculate log_vi_xi
    tmp_mmbk = tmp_mmbk[indexes.long(), :].detach()#.cpu()
    dist_mat = torch.bmm(features.detach().unsqueeze(0), tmp_mmbk.t().unsqueeze(0)).squeeze()
    dist_mat = torch.clamp(dist_mat, min=-1.0, max=1.0)
    dist_mat = torch.acos(dist_mat).squeeze().detach()
    dist_mat_r = dist_mat[torch.arange(inputs.size(0)), torch.arange(inputs.size(0))].unsqueeze(-1).repeat(1, features.size(1))
    log_vi_xi = dist_mat_r/torch.sin(dist_mat_r) * (features.detach() - tmp_mmbk.detach() * torch.cos(dist_mat_r))
    # calculate log_vi_xj
    dist_mat = torch.bmm(tmp_mmbk.unsqueeze(0), mmbk2.detach().t().unsqueeze(0)).squeeze() #  batchsize * n
    dist_mat = torch.clamp(dist_mat, min=-1.0, max=1.0)
    dist_mat = torch.acos(dist_mat).unsqueeze(-1).repeat(1, 1, features.size(1)).cuda(1)
    vis = tmp_mmbk.unsqueeze(1).repeat(1, ndata, 1).cuda(1)
    xjs = mmbk2.detach().cpu().unsqueeze(0).repeat(features.size(0), 1, 1).cuda(1)
    log_vi_xj = -1.0 * dist_mat.detach()/torch.sin(dist_mat.detach()) * (xjs - vis.detach() * torch.cos(dist_mat.detach()))
    II = Variable(torch.ones(features.size(0), ndata)).to(device)
    II[torch.arange(inputs.size(0)).long(), indexes.long()] *= 0.0
    II = II.unsqueeze(-1).repeat(1,1,features.size(1))
    log_vi_xj = torch.mul(log_vi_xj.to(device), II)
    log_vi_xj = log_vi_xj.sum(-2) # batchsize * 128
    log_p_q = log_vi_xi + log_vi_xj
    v_norm = torch.norm(log_p_q, p=2, dim=1).squeeze().unsqueeze(1).repeat(1, args.low_dim)
    exp_p_v = torch.cos(v_norm) * tmp_mmbk + torch.sin(v_norm) * log_p_q/v_norm
    mmbk[indexes.long(), :] = mmbk[indexes.long(), :] * 0.5 + 0.5 * exp_p_v.to(device)
    return mmbk, mmbk2, loss

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



def get_visualization(mmbk, trainloader, net):
    device = 'cuda'
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        print(batch_idx)
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
        features = net(inputs)
        mmbk[indexes.long(), :] = features.data
    X = mmbk.data.cpu().numpy()
    y = np.array(trainloader.dataset.train_labels)
    X = X[:100, :]
    y = y[:100]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    plot_embedding(X, y, None)
    return
    n_sne = 7000
    f_dim = X.shape[1]
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))
    rndperm = np.random.permutation(df.shape[0])
    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]
    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
    pdb.set_trace()
    chart.save("cifar_vis.png")

net.load_state_dict(torch.load('cifar_method5_1024_manual_grad/model_1990.pt'), strict=False)
# mmbk = pkl.load(open('cifar_method5_1024_manual_grad/mmbk_1990.pkl','rb'))
get_visualization(mmbk, trainloader, net)
import sys
sys.exit("1")
