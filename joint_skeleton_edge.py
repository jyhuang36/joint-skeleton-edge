import os
import numpy as np
import scipy.misc
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


parser = argparse.ArgumentParser(description='Deep Skeleton')
parser.add_argument('-epochs', default=6, type=int, help='number of epochs')
parser.add_argument('-itersize', default=10, type=int, help='iteration size')
parser.add_argument('-printfreq', default=100, type=int, help='printing frequency')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-decay', default=0.0002, type=float, help='weight decay')
parser.add_argument('-mode', default='cpu', type=str, help='mode')
parser.add_argument('-gpuid', default=0, type=int, help='gpu id')
parser.add_argument('-train', default=False, action='store_true')
parser.add_argument('-visualize', default=False, action='store_true')
parser.add_argument('-test', default=False, action='store_true')



class Skeleton(nn.Module):
    def __init__(self):
        super(Skeleton, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),               
        )        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),               
        )               
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),   
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),              
        )                        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),                
        )                               
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),                 
        )
        self.sk2 = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.sk3 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear")
        )
        self.sk4 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear")
        )
        self.sk5 = nn.Sequential(
            nn.Conv2d(512, 5, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear")
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear")
        )
        self.scale4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear")
        )
        self.scale5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear")
        )
        
        
        self.skfus1 = nn.Conv2d(4, 1, kernel_size=1)
        self.skfus2 = nn.Conv2d(4, 1, kernel_size=1)
        self.skfus3 = nn.Conv2d(3, 1, kernel_size=1)
        self.skfus4 = nn.Conv2d(2, 1, kernel_size=1)
        self.skfus5 = nn.Conv2d(1, 1, kernel_size=1)
        
        
        self.edge1 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
        )
        self.edge2 = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Sigmoid()
        )
        self.edge3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Upsample(scale_factor=4, mode="bilinear"),
            nn.Sigmoid()
        )
        self.edge4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Upsample(scale_factor=8, mode="bilinear"),
            nn.Sigmoid()
        )
        self.edge5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Upsample(scale_factor=16, mode="bilinear"),
            nn.Sigmoid()
        )
        self.edgefus = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        
        self.init_weights()
     
    def forward(self, x):
        x_ = self.conv1(x)
        y1edge = self.edge1(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv2(x_)
        y2sk = self.sk2(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y2scale = self.scale2(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y2edge = self.edge2(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv3(x_)
        y3sk = self.sk3(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y3scale = self.scale3(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y3edge = self.edge3(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv4(x_)
        y4sk = self.sk4(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y4scale = self.scale4(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y4edge = self.edge4(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        x_ = self.conv5(x_)
        y5sk = self.sk5(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y5scale = self.scale5(x_)[:,:,0:x.shape[2],0:x.shape[3]]
        y5edge = self.edge5(x_)[:,:,0:x.shape[2],0:x.shape[3]]

        skf1 = self.skfus1(torch.cat((y2sk[:,0:1,:,:], y3sk[:,0:1,:,:], y4sk[:,0:1,:,:], y5sk[:,0:1,:,:]), 1))
        skf2 = self.skfus2(torch.cat((y2sk[:,1:2,:,:], y3sk[:,1:2,:,:], y4sk[:,1:2,:,:], y5sk[:,1:2,:,:]), 1))
        skf3 = self.skfus3(torch.cat((y3sk[:,2:3,:,:], y4sk[:,2:3,:,:], y5sk[:,2:3,:,:]), 1))
        skf4 = self.skfus4(torch.cat((y4sk[:,3:4,:,:], y5sk[:,3:4,:,:]), 1))
        skf5 = self.skfus5(y5sk[:,4:5,:,:])
                
        skf = torch.cat((skf1, skf2, skf3, skf4, skf5), 1)
        
        edgef = self.edgefus(torch.cat((y1edge, y2edge, y3edge, y4edge, y5edge), 1))
        
        y1edge = torch.squeeze(y1edge)
        y2edge = torch.squeeze(y2edge)
        y3edge = torch.squeeze(y3edge)        
        y4edge = torch.squeeze(y4edge)
        y5edge = torch.squeeze(y5edge)
        
        return y1edge, y2sk, y2scale, y2edge, y3sk, y3scale, y3edge, y4sk, y4scale, y4edge, y5sk, y5scale, y5edge, skf, edgef
    


    def init_weights(self):
        #load VGG weights
        for i, (param, pretrained) in enumerate(zip(self.parameters(), 
                                                torchvision.models.vgg16(pretrained=True).parameters())):
            if i < 26:
                param.data = pretrained.data
        
        #initialize other parameters        
        nn.init.normal_(self.sk2[0].weight, 0, 0.01)
        nn.init.normal_(self.sk3[0].weight, 0, 0.01)
        nn.init.normal_(self.sk4[0].weight, 0, 0.01)
        nn.init.normal_(self.sk5[0].weight, 0, 0.01)
        nn.init.normal_(self.scale2[0].weight, 0, 0.01)
        nn.init.normal_(self.scale3[0].weight, 0, 0.01)
        nn.init.normal_(self.scale4[0].weight, 0, 0.01)
        nn.init.normal_(self.scale5[0].weight, 0, 0.01)
                                
        nn.init.constant_(self.skfus1.weight, 0.25)
        nn.init.constant_(self.skfus2.weight, 0.25)
        nn.init.constant_(self.skfus3.weight, 0.33)
        nn.init.constant_(self.skfus4.weight, 0.5)
        nn.init.constant_(self.skfus5.weight, 1)
        
        nn.init.constant_(self.edgefus[0].weight, 0.2)
        nn.init.normal_(self.edge1[0].weight, 0, 0.01)
        nn.init.normal_(self.edge2[0].weight, 0, 0.01)
        nn.init.normal_(self.edge3[0].weight, 0, 0.01)
        nn.init.normal_(self.edge4[0].weight, 0, 0.01)
        nn.init.normal_(self.edge5[0].weight, 0, 0.01)
        
class SkeletonTrainingSet(Dataset):
    def __init__(self, lst_file, root_dir='', resize=None, transform=None, fieldsize=None, threshold=None):
        with open(lst_file) as f:
            self.lst = f.read().splitlines()
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        self.fieldsize = fieldsize
        self.threshold = threshold
        
    def __len__(self):
        #return len(self.lst)
        return 1
    
    def __getitem__(self, index):
        filenames = self.lst[index].split(" ")
        image = Image.open(self.root_dir + filenames[0])
        sk = Image.open(self.root_dir + filenames[1])
        edge = Image.open(self.root_dir + filenames[2])
        
        if self.resize:
            image = self.resize(image)
            sk = self.resize(sk)
            edge = self.resize(edge)
            
        if self.transform:
            image = self.transform(image)

        #only use single channel for label
        sk = np.array(sk)   
        if sk.ndim == 3:
            sk = sk[:,:,0]
            
        if self.fieldsize:
            sk_bin_lst = []
            for i in range(len(self.fieldsize) - 1):
                    sk_bin = ((1.2 * sk > self.fieldsize[i]) & (1.2 * sk < self.fieldsize[i+1])).astype(int)
                    sk_bin_lst.append(sk_bin * (i + 1))
                    
            l2sk = sk_bin_lst[0]
            l3sk = l2sk + sk_bin_lst[1]
            l4sk = l3sk + sk_bin_lst[2]
            l5sk = l4sk + sk_bin_lst[3] 
            
            l2scale = sk * (l2sk > 0).astype(int) * 2 / self.fieldsize[1] - 1
            l3scale = sk * (l3sk > 0).astype(int) * 2 / self.fieldsize[2] - 1
            l4scale = sk * (l4sk > 0).astype(int) * 2 / self.fieldsize[3] - 1
            l5scale = sk * (l5sk > 0).astype(int) * 2 / self.fieldsize[4] - 1
            
        edge = np.array(edge)   
        if edge.ndim == 3:
            edge = edge[:,:,0]
            
        if self.threshold:
            edge = edge > self.threshold
            edge = edge.astype(float)
        edge = torch.from_numpy(edge)

        return image, l2sk, l3sk, l4sk, l5sk, l2scale, l3scale, l4scale, l5scale, edge


class SkeletonTestSet(Dataset):
    def __init__(self, im_dir, root_dir='', resize=None, transform=None):
        self.lst = os.listdir(root_dir + im_dir)
        self.root_dir = root_dir
        self.im_dir = im_dir
        self.resize = resize
        self.transform = transform
        
    def __len__(self):
        return len(self.lst)
    
    def __getitem__(self, index):
        image = Image.open(self.root_dir + self.im_dir + self.lst[index])
            
        if self.resize:
            image = self.resize(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, self.lst[index]
   


def main():
    global args

    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    model = Skeleton()
    
    if args.mode == 'gpu':           
        torch.cuda.set_device(args.gpuid) 
        torch.cuda.manual_seed(0)               
        model.cuda()
    
    train_dataset = SkeletonTrainingSet(lst_file="aug_data/train_pair.lst", 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])]),
                                        fieldsize=[0, 14, 40, 92, 196],
                                        threshold=10)
    
    test_dataset = SkeletonTestSet(im_dir='images/test/', 
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])]))
        
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam([{'params': model.conv1.parameters()},
                                  {'params': model.conv2.parameters()},
                                  {'params': model.conv3.parameters()},
                                  {'params': model.conv4.parameters()},
                                  {'params': model.conv5.parameters()}, 
                                  {'params': model.sk2.parameters()},
                                  {'params': model.sk3.parameters()},
                                  {'params': model.sk4.parameters()},
                                  {'params': model.sk5.parameters()}, 
                                  {'params': model.skfus1.parameters()}, 
                                  {'params': model.skfus2.parameters()},
                                  {'params': model.skfus3.parameters()},
                                  {'params': model.skfus4.parameters()},
                                  {'params': model.skfus5.parameters()},
                                  {'params': model.edge1.parameters()},
                                  {'params': model.edge2.parameters()},
                                  {'params': model.edge3.parameters()},
                                  {'params': model.edge4.parameters()}, 
                                  {'params': model.edge5.parameters()}, 
                                  {'params': model.edgefus.parameters()}], 
                                 lr=args.lr, weight_decay=args.decay) 
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    if args.train:        
        train(model, train_loader, optimizer, scheduler)        
        torch.save(model.state_dict(), 'joint_sk_edge.pt')
    
    if args.visualize:    
        visualize(model, test_dataset)
    
    if args.test:
        test(model, test_dataset)

    
def loss_sk(y, label):
    label_ = label.cpu().data.numpy()
    count_lst = []       
    for i in range(y.shape[1]):
        n = (label_ == i).sum()
        if n != 0:
            count_lst.append(1/n)
        else:
            count_lst.append(0)
    s = sum(count_lst)
    for i in range(len(count_lst)):
        count_lst[i] = count_lst[i]/s
        
    if args.mode == 'gpu':
        loss = nn.CrossEntropyLoss(torch.cuda.FloatTensor(count_lst))
    else:
        loss = nn.CrossEntropyLoss(torch.FloatTensor(count_lst))
    return loss(y, label)

def loss_scale(y, label):
    label_ = label.cpu().data.numpy()
    label_bin = (label_ > 0).astype(int)
    n = label_bin.sum()
    
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
        
    if n == 0:
        mult = 0
    else:
        mult = 1/n
        
    label_weight = torch.from_numpy(label_bin * mult).type(dtype_float)

    return (((y - label) ** 2) * label_weight).sum()

def loss_edge(y, label):
    loss = - 0.962 * label * torch.log(y) - 0.038 * (1 - label) * torch.log(1 - y) 
    return loss.mean()

   
def train(model, train_loader, optimizer, scheduler): 
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
            
    for epoch in range(args.epochs):
        
        optimizer.zero_grad()
        loss_value = 0
    
        for i, (image, l2sk, l3sk, l4sk, l5sk, l2scale, l3scale, l4scale, l5scale, edge) in enumerate(train_loader):
            
            image = image.type(dtype_float)
            l2sk = l2sk.type(dtype_long)
            l3sk = l3sk.type(dtype_long)
            l4sk = l4sk.type(dtype_long)
            l5sk = l5sk.type(dtype_long)
            l2scale = l2scale.type(dtype_float)
            l3scale = l3scale.type(dtype_float)
            l4scale = l4scale.type(dtype_float)
            l5scale = l5scale.type(dtype_float)
            edge = edge.type(dtype_float)            
            
            
            y1edge, y2sk, y2scale, y2edge, y3sk, y3scale, y3edge, y4sk, y4scale, y4edge, y5sk, y5scale, y5edge, skf, edgef = model(image)
            
            loss = (loss_sk(skf, l5sk) + loss_sk(y2sk, l2sk) + loss_sk(y3sk, l3sk) + \
                    loss_sk(y4sk, l4sk) + loss_sk(y5sk, l5sk) + loss_scale(y2scale, l2scale) + \
                    loss_scale(y3scale, l3scale) + loss_scale(y4scale, l4scale) + loss_scale(y5scale, l5scale) + \
                    loss_edge(y1edge, edge) + loss_edge(y2edge, edge) + loss_edge(y3edge, edge) + \
                    loss_edge(y4edge, edge) + loss_edge(y5edge, edge) +loss_edge(edgef, edge))/args.itersize
            
            loss_value += loss.cpu().data.numpy()
                    
            loss.backward()
            
            if (i+1) % (args.printfreq * args.itersize) == 0:
                print("epoch: %d    iteration: %d    loss: %.3f" 
                      %(epoch, i//args.itersize, loss_value))
            
            if (i+1) % args.itersize == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss_value = 0
                
        #scheduler.step()

def visualize(model, visualize_dataset):
    if args.mode == 'cpu':
        model.load_state_dict(torch.load('joint_sk_edge.pt', 
                                     map_location={'cuda:0':'cpu', 'cuda:1':'cpu',                                                    
                                                   'cuda:2':'cpu', 'cuda:3':'cpu'}))
    else:
        model.load_state_dict(torch.load('joint_sk_edge.pt'))
        
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
        
    image = visualize_dataset[198][0]
    image = image.unsqueeze(0) 
    image = image.type(dtype_float)
    
    y1edge, y2sk, y2scale, y2edge, y3sk, y3scale, y3edge, y4sk, y4scale, y4edge, y5sk, y5scale, y5edge, skf, edgef = model(image)
    
    sk_out = 1 - F.softmax(skf[0], 0)[0].cpu().data.numpy()
    sk_out = sk_out/sk_out.max()
    
    edge_out = (5 * edgef + y1edge + y2edge + y3edge + y4edge + y5edge)/10
    edge_out = edge_out[0][0].data.numpy()
    edge_out = edge_out/edge_out.max()
    
    scale_lst = [sk_out, edge_out]
    
    plot_single_scale(scale_lst, 22)

def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(scale_lst[i], cmap = plt.cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


def test(model, test_dataset):
    if args.mode == 'cpu':
        model.load_state_dict(torch.load('joint_sk_edge.pt', 
                                     map_location={'cuda:0':'cpu', 'cuda:1':'cpu',                                                    
                                                   'cuda:2':'cpu', 'cuda:3':'cpu'}))
    else:
        model.load_state_dict(torch.load('joint_sk_edge.pt'))
        
    if args.mode == 'gpu':
        dtype_float = torch.cuda.FloatTensor
    else:
        dtype_float = torch.FloatTensor
        
    for i in range(len(test_dataset)):
        image, name = test_dataset[i]
        image = image.unsqueeze(0) 
        if image.shape[1] == 1:
            image = torch.cat((image, image, image), 1)
        
        image = image.type(dtype_float)   
        y1edge, y2sk, y2scale, y2edge, y3sk, y3scale, y3edge, y4sk, y4scale, y4edge, y5sk, y5scale, y5edge, skf, edgef = model(image)
    
        sk_out = 1 - F.softmax(skf[0], 0)[0].cpu().data.numpy()
        sk_out = sk_out/sk_out.max()
        
        edge_out = (5 * edgef + y1edge + y2edge + y3edge + y4edge + y5edge)/10
        edge_out = edge_out/edge_out.max()
        
        scipy.misc.imsave('results/sk/' + name[0:-4] + '.png', sk_out)
        scipy.misc.imsave('results/edge/' + name[0:-4] + '.png', edge_out)
        print('%d of %d images saved' %(i+1, len(test_dataset)))
            
       
if __name__ == '__main__':
    main()                
