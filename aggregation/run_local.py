import syft as sy
from syft.grid.public_grid import PublicGridNetwork
import numpy as np
import torch as th

import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from syft.federated.floptimizer import Optims
from mnist_loader import read_mnist_data

##-------------parameter setting--------------

parties = 2
AGG_EPOCH = 1
EPOCHS = 10
TAG_NAME = "mnist_test_"+str(parties)+"nodes_ns"
grid_address = "http://203.145.221.20:80"  # address


INIT_WEIGHT = True
path_checkpoint ='model/initial_weight.pth'
SAVE_MODEL = False
#N_EPOCHS = AGG_EPOCH*EPOCHS  # number of epochs to train
N_TEST   = 128   # number of test
train_batch_size = 16
N_LOG = 100

LR = 0.01
momentum = 0.9
weight_decay = 1e-5

node_name = ["gridnode01","gridnode02","gridnode03","gridnode04","gridnode05","gridnode06","gridnode07","gridnode08"]
output_model_folder = 'model'



class Arguments():
    def __init__(self):
        self.test_batch_size = N_TEST
        
        self.lr = LR
        self.log_interval = N_LOG
        self.momentum = momentum
        self.weight_decay = weight_decay
        #self.device = th.device("cpu")
        
args = Arguments()

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(5408, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)        
        x = F.relu(x)        
        x = F.max_pool2d(x, 2)        
        x = th.flatten(x, 1)        
        x = self.fc1(x)        
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    

##-----------------load data------------------
def load_data():
    transform = transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.1307,), (0.3081,)),  #  mean and std 
                                  ])


##---------load 30000 (2Parties/party0) as training data----------------

    npz_path = '../2Parties/data_party0.npz'
    mnist_train_loader0,mnist_test_loader0 = read_mnist_data(npz_path, batch = train_batch_size )
    
    npz_path = '../2Parties/data_party1.npz'
    mnist_train_loader1,mnist_test_loader1 = read_mnist_data(npz_path, batch = train_batch_size )
    


##-------------load original mnist, 10000 test data-----------------
    testset = datasets.MNIST('../8node/dataset2', download=False, train=False, transform=transform)
    testloader = th.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    trainset = datasets.MNIST('../8node/dataset2', download=False, train=True, transform=transform)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False)
    
    return mnist_train_loader0, testloader
    #return trainloader, testloader
    

def init_model():

    model_list = Net()
    model_list.to(device)
    if INIT_WEIGHT:
        print("use initial weight")
        checkpoint = th.load(path_checkpoint)      
        model_list.load_state_dict(checkpoint) 
    #optims_list[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr,weight_decay=args.weight_decay))
    #optims_list = Optims(workers, optim=optim.Adam(params=model_list.parameters(),lr=args.lr))
    #optims_list[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum,weight_decay=args.weight_decay))
    #optims_list[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum))
    
    optims_list = optim.Adam(model_list.parameters(), lr=LR)

    return model_list, optims_list



def get_syft_optimizer(curr_optims):
    hook = sy.TorchHook(th)
    my_grid = PublicGridNetwork(hook, grid_address)
    data = my_grid.search("#X_"+TAG_NAME) 
    data = list(data.values()) 
    worker = data[0][0].location    
    return curr_optims.get_optim(worker.id)   


def train(curr_model, optimizer, train_loader, args,fo):
#    shuffle_list = np.arange(len(datalist))
#    np.random.shuffle(shuffle_list)
#    datalist = datalist[shuffle_list]
#    targetlist = targetlist[shuffle_list]
    
    curr_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = curr_model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(pred, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))   
    fo.write(",{:.6f}".format(loss.item()))   
    return curr_model

def test(testloader,test_model, args,fo):
    
    
    
    test_model.eval()
    test_loss = 0
    correct = 0
    with th.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = test_model(data)
            loss = criterion(output, target)
            test_loss += loss  #F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    fo.write(",{:.2f}".format(100. * correct / len(testloader.dataset)))   
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))




workers =node_name[:parties]
train_loader, test_loader = load_data()
criterion = nn.CrossEntropyLoss()
model_list, optims_list = init_model()
#optimizer = get_syft_optimizer(optims_list)
optimizer = optims_list


output_file_name = 'local_Adam_wodecay.csv'
fo = open(output_file_name, "w")

for epoch in range(EPOCHS):
    fo.write(str(epoch))  
    
    current_model = train(model_list, optimizer, train_loader, args,fo)        
    test(test_loader,current_model, args,fo)
    fo.write("\n")          
fo.close()           
            
            
