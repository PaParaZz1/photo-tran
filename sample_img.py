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
from cub_dataset import *
import cv2

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--low-dim', default=4096, type=int, metavar='D', help='feature dimension')
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
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--log-path', type=str, default='model')
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--with-mmbk2', action='store_true')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--k', type=int, default=200)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

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
    trainset = datasets.CIFAR10Instance(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10Instance(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'mnist':
    transform = transforms.Compose([transforms.RandomResizedCrop(size=28, scale=(0.2,1)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNISTInstance(root='./data', train=True, transform=transform, download=True)
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
ndata = trainset.__len__()
print(ndata)

# net = models.__dict__['resnet18'](low_dim=args.low_dim)#, l2norm=args.normalize)
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
optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)

def test(net, trainloader, testloader, input_mmbk, k=10, num_class=10, root='mnist_sample', sample_num=50):
    if not os.path.exists(root):
        os.mkdir(root)
    net.eval()
    top1 = 0.
    top5 = 0.
    total = 0
    norm = input_mmbk.pow(2).sum(1, keepdim=True).pow(0.5)
    mmbk = input_mmbk.div(norm + 1e-16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=1)
    mmbk_label = torch.LongTensor(temploader.dataset.train_labels).to(device)
    # trainloader.dataset.transform = transform_bak
    with torch.no_grad():
        # testloader batch_size 1
        retrieval_one_hot = torch.zeros(k, num_class).to(device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
            targets = targets.to(device)#cuda()#async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            dist = torch.mm(features, mmbk.t())
            yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
            print(yd)
            print(yi)
            top_k_imgs = trainloader[yi]
            print(top_k_imgs.shape)
            return
            if batch_idx<sample_num:
                img_dir_path = os.path.join(root, 'img_{}'.format(batch_idx))
                if not os.path.exists(img_dir_path):
                    os.mkdir(img_dir_path)
                origin_img = inputs.permute(0,2,3,1).contiugous().cpu().numpy()
                img_path = os.path.join(img_dir_path, 'query.png')
                cv2.imwrite(img_path, origin_img)
                for i in range(top_k_imgs.shape[0]):
                    similar_img = top_k_imgs[i].permute(1,2,0).cpu().numpy()
                    img_path = os.path.join(img_path, 'similar_{}.png'.format(i))
                    cv2.imwrite(img_path, similar_img)
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

device = 'cuda'
net.load_state_dict(torch.load('/home/xinlei/git/icml2019/mnist_1024_method5_k_200_mmbk/model_500.pt'), strict=False)
for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
    print(batch_idx)
    inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device)
    features = net(inputs)
    mmbk[indexes.long(), :] = features.data
test(net, trainloader, testloader, mmbk)
sys.exit(1)










import copy
best_acc = 0.

M1 = 0
M2 = 0
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
alpha = 0.001
temp = args.temp
# train
for epoch in range(2000):
    net.train()
    print('start epoch ', epoch)
    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        inputs, targets, indexes = inputs.to(device), targets.to(device), indexes.to(device).long()
        optimizer.zero_grad()
        features = net(inputs)

        if args.method1:
            with torch.no_grad():
                mmbk[indexes.long(), :] = mmbk[indexes.long(), :] * 0.5 + 0.5 * features.data
                mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, args.low_dim)
            dist_mat = torch.mm(features, mmbk.detach().to(device).t())
            dist_mat = dist_mat.div(temp)
            loss = nn.CrossEntropyLoss()(dist_mat, indexes.long())
            loss.backward()
            optimizer.step()
        elif args.method2:
            with torch.no_grad():
                mmbk[indexes.long(), :] = mmbk[indexes.long(), :] * 0.3 + 0.7 * features.data
                mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, args.low_dim)
            dist_mat = torch.mm(features, mmbk.detach().to(device).t())
            I = Variable(torch.ones(dist_mat.size())).to(device)
            I[torch.arange(inputs.size(0)).long(), indexes.long()] *= 0.0
            dist_mat1 = torch.mul(dist_mat, I)
            loss1 = dist_mat1.sum()/(inputs.size(0) * (ndata-1))

            II = Variable(torch.zeros(dist_mat.size()), requires_grad=False).to(device)
            II[torch.arange(inputs.size(0)).long(), indexes.long()] = 1.0
            dist_mat2 = torch.mm(features, mmbk.detach().to(device).t())
            dist_mat2 = torch.mul(dist_mat2, II)
            loss2 = -1.0 * dist_mat2.sum()/(inputs.size(0))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
        elif args.method3:
            # update network
            tmp_mmbk = mmbk/torch.norm(mmbk, p=2, dim=1).unsqueeze(1).repeat(1, args.low_dim)
            dist_mat = torch.mm(features, tmp_mmbk.detach().to(device).t())
            I = Variable(torch.ones(dist_mat.size())).to(device)
            I[torch.arange(inputs.size(0)).long(), indexes.long()] *= 0.0
            dist_mat1 = torch.mul(dist_mat, I)
            dist_mat1 = torch.exp(dist_mat1/temp)
            loss1 = dist_mat1.sum()/(inputs.size(0) * (ndata-1))

            II = Variable(torch.zeros(dist_mat.size()), requires_grad=False).to(device)
            II[torch.arange(inputs.size(0)).long(), indexes.long()] = 1.0
            dist_mat2 = torch.mm(features, tmp_mmbk.detach().to(device).t())
            dist_mat2 = torch.mul(dist_mat2, II)
            dist_mat2 = torch.exp(dist_mat2/temp)
            loss2 = -1.0 * dist_mat2.sum()/(inputs.size(0))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()


            # update mmbk
            dist_mat = torch.mm(features.detach(), tmp_mmbk.to(device).t())
            I = Variable(torch.ones(dist_mat.size())).to(device)
            I[torch.arange(inputs.size(0)).long(), indexes.long()] *= 0.0
            dist_mat1 = torch.mul(dist_mat, I)
            dist_mat1 = torch.exp(dist_mat1/temp)
            loss1 = dist_mat1.sum()/(inputs.size(0) * (ndata-1))

            II = Variable(torch.zeros(dist_mat.size()), requires_grad=False).to(device)
            II[torch.arange(inputs.size(0)).long(), indexes.long()] = 1.0
            dist_mat2 = torch.mm(features.detach(), tmp_mmbk.to(device).t())
            dist_mat2 = torch.mul(dist_mat2, II)
            dist_mat2 = torch.exp(dist_mat2/temp)
            loss2 = -1.0 * dist_mat2.sum()/(inputs.size(0))
            loss = loss1 + loss2
            loss.backward()
            III = Variable(torch.zeros(mmbk.size())).to(device)
            III[indexes.long(), :] += 1.0
            mmbk.grad = mmbk.grad * III
            opt2.step()
        elif args.method4:
            mmbk, mmbk2, loss = calculate_newloss(args, net, inputs, targets, indexes, features, mmbk, mmbk2, device, ndata, optimizer)
        elif args.method5:
            with torch.no_grad():
                if args.with_mmbk2:
                    mmbk2[indexes.long(), :] = mmbk2[indexes.long(), :]*0.5+0.5*features.data
                else:
                    mmbk[indexes.long(), :] = mmbk[indexes.long(), :] * 0.5 + 0.5 * features.data
            mmbk_norm = mmbk.pow(2).sum(1, keepdim=True).pow(0.5)
            mmbk = mmbk.div(mmbk_norm)
            d_xi_vj = torch.acos(torch.clamp(torch.mm(features, mmbk.t()), min=-1.0, max=1.0))**2.0
            I = Variable(torch.zeros(d_xi_vj.size())).to(device)
            I[torch.arange(inputs.size(0)).long(), indexes.long()] += 1.0
            exp_d_xi_vj = torch.exp(-1.0 * d_xi_vj/temp)
            exp_d_xi_vi = torch.mul(exp_d_xi_vj, I).sum(-1).squeeze(-1)
            loss = -0.5 * torch.log(exp_d_xi_vi/exp_d_xi_vj.sum(-1)).sum()/(inputs.size(0))
            loss.backward()
            optimizer.step()
            # update mmbk
            if args.with_mmbk2:
                with torch.no_grad():
                    vis = mmbk[indexes.long(), :]
                    dist = torch.acos(torch.clamp(torch.mm(features, vis.t()), min=-1.0, max=1.0))/temp
                    dist = dist[torch.arange(inputs.size(0)).long(), torch.arange(inputs.size(0)).long()].squeeze().unsqueeze(-1)#.repeat(1, args.low_dim)
                    log_vi_xi = dist/torch.sin(dist) * (features - vis * torch.cos(dist)) # batch * D

                    d_xk_vj = torch.exp(-1.0 * torch.acos(torch.clamp(torch.mm(features, mmbk.t()),min=-1.0, max=1.0))**2.0/temp) # batch * n
                    d_xk_vi = d_xk_vj[torch.arange(inputs.size(0)).long(), indexes.long()] # batch
                    dist_vi_xk = torch.acos(torch.clamp(torch.mm(vis, features.t()), min=-1.0, max=1.0))/temp
                    vis_r = vis.unsqueeze(1)#.repeat(1, inputs.size(0), 1) # batch * 1 * D
                    features_r = features.unsqueeze(0)#.repeat(inputs.size(0), 1, 1) # 1 * batch * D
                    dist_vi_xk_r = dist_vi_xk.unsqueeze(-1)#.repeat(1, 1, args.low_dim) # batch * batch * 1
                    log_vi_xk = dist_vi_xk_r/torch.sin(dist_vi_xk_r) * (features_r - vis_r * torch.cos(dist_vi_xk_r)) # batch * batch * D
                    norm_term = 1.0/d_xk_vj.sum(-1).squeeze().unsqueeze(0)# .repeat(inputs.size(0), 1) # 1 * batch
                    d_vi_xk = torch.exp(-1.0 * torch.acos(torch.clamp(torch.mm(vis, features.t()), min=-1.0, max=1.0))**2.0/temp) # vi * xk
                    whole_term = (norm_term * d_vi_xk).unsqueeze(-1)
                    final_term = (whole_term * log_vi_xk).sum(1).squeeze()
                    v_grad = log_vi_xi - final_term
                    v_grad_norm = v_grad.pow(2).sum(-1, keepdim=True).pow(0.5)
                    exp_v_grad = torch.cos(v_grad_norm) * vis + torch.sin(v_grad_norm) * v_grad.div(v_grad_norm)
                    mmbk[indexes.long(), :] = mmbk[indexes.long(), :] * 0.5 + 0.5 * exp_v_grad.to(device)
    if batch_idx % 50 == 0:
        print('batch ', batch_idx, ' loss is ', loss)#, loss1, loss2)
    if os.path.isdir(args.log_path) == False:
        os.mkdir(args.log_path)
    if epoch % 10 == 0 and epoch > 0:
        torch.save(net.state_dict(), args.log_path+'/model_'+str(epoch)+'.pt')
        pkl.dump(mmbk, open(args.log_path+'/mmbk_'+str(epoch)+'.pkl', 'wb'))
        pkl.dump(mmbk2, open(args.log_path+'/mmbk2_'+str(epoch)+'.pkl','wb'))
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        num_class = 10
    elif args.dataset == 'cub':
        num_class = 200
    acc1 = test(net, trainloader, testloader, mmbk2, args.k, num_class=num_class)
    acc2 = test(net, trainloader, testloader, mmbk, args.k, num_class=num_class)
    acc = max(acc1, acc2)
    if acc > best_acc:
        best_acc = acc
    with open(args.log_path+'/'+args.log_name, 'a') as f:
        f.write('epoch '+str(epoch)+' acc '+str(acc)+'\n')
    print('best acc is ', best_acc)
