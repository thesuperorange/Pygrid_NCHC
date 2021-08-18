import syft as sy
from syft.grid.public_grid import PublicGridNetwork
import numpy as np
import torch as th

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms




parties = 1
AGG_EPOCH = 1
EPOCHS = 10
TAG_NAME = "mnist_test_"+str(parties)+"nodes"
#TAG_NAME = "mnist_test_small2"
# mnist_test_8nodes  mnist_test_4nodes  mnist_test  mnist_test_small(4)   mnist_test_small2 (2node)
grid_address = "http://203.145.221.20:80"  # address



N_EPOCHS = AGG_EPOCH*EPOCHS  # number of epochs to train
N_TEST   = 128   # number of test
train_batch_size = 16
N_LOG = 1

LR = 0.01
momentum = 0.9
weight_decay = 1e-5

node_name = ["gridnode01","gridnode02","gridnode03","gridnode04","gridnode05","gridnode06","gridnode07","gridnode08"]
output_model_folder = 'model'

hook = sy.TorchHook(th)


# Connect direcly to grid nodes
my_grid = PublicGridNetwork(hook, grid_address)

class Arguments():
    def __init__(self):
        self.test_batch_size = N_TEST
        self.epochs = N_EPOCHS
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
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(5408, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)        
        x = F.relu(x)        
        #x = self.conv2(x)
        #x = F.relu(x)
        x = F.max_pool2d(x, 2)        
        #x = self.dropout1(x)
        x = th.flatten(x, 1)        
        x = self.fc1(x)        
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
data = my_grid.search("#X_"+TAG_NAME)  # images
target = my_grid.search("#Y_"+TAG_NAME)  # labels

data = list(data.values())  # returns a pointer
target = list(target.values())  # returns a pointer


print(data)
print(target)


from mnist_loader import read_mnist_data
transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),  #  mean and std 
                              ])
#npz_path = '../'+str(parties)+'Parties/data_party0.npz'
#trainloader,testloader = read_mnist_data(npz_path, batch = args.test_batch_size )
testset = datasets.MNIST('../8node/dataset2', download=False, train=False, transform=transform)
testloader = th.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

trainset = datasets.MNIST('../8node/dataset2', download=False, train=True, transform=transform)
trainloader = th.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=True)


def init_model(parties):
    model_list=[None] * parties
    optims_list=[None] * parties
    for i in range(parties):
        model_list[i] = Net()
        model_list[i].to(device)
        optims_list[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr,weight_decay=args.weight_decay))
        #optims_list[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr))
        #optims_list[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum,weight_decay=args.weight_decay))
        #optims_list[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum))
    return model_list, optims_list


def avgWeight(model_list):
    model_tmp=[None] * parties
    optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        
        model_tmp[idx] = my_model.state_dict()


    for key in model_tmp[0]:    
        print(key)
        model_sum = 0
        for model_tmp_content in model_tmp:        
            model_sum += model_tmp_content[key]
            #print(model_tmp_content[key])
        for i in range(len(model_tmp)):
            #print("model_sum={}".format(model_sum))
            #print("len:{}".format(len(model_tmp)))
            model_avg = model_sum/len(model_tmp)
            #print("model_avg={}".format(model_avg))
            model_tmp[i][key] = model_sum/len(model_tmp)
    for i in range(len(model_list)):    
        model_list[i].load_state_dict(model_tmp[i])
        #optims_tmp[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum,weight_decay=args.weight_decay))
        #optims_tmp[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum))
        optims_tmp[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr,weight_decay=args.weight_decay))
        #optims_tmp[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr))
    return model_list, optims_tmp

def train(curr_model, curr_optims, args):
#    shuffle_list = np.arange(len(datalist))
#    np.random.shuffle(shuffle_list)
#    datalist = datalist[shuffle_list]
#    targetlist = targetlist[shuffle_list]

    
    

    for i in range(len(data)):
        
        curr_model[i].train()
        # This loop is for "a bunch of data" searched on the node.
        # Equals to an epoch for a node if there is only "one bunch of data" for a node. 
        loss_epoch = 0
        for j in range(len(data[i])):




            worker = data[i][j].location  # worker hosts data
            print(worker.id)
            if worker.id not in workers:
                print("not in worker list")
                continue


            data_device = data[i][j].to(device)
            target_device = target[i][j].to(device)

            curr_model[i].send(worker)  # send model to PyGridNode worker

            batch_remainder = len(data[i][j])%train_batch_size


            for k in range(len(data[i][j])//train_batch_size):
                optimizer = curr_optims[i].get_optim(worker.id)   

                optimizer.zero_grad()  
                pred = curr_model[i](data_device[k*train_batch_size:(k+1)*train_batch_size])
                loss = criterion(pred, target_device[k*train_batch_size:(k+1)*train_batch_size])
                #loss = F.nll_loss(pred, target[i][j])
                loss.backward()

                optimizer.step()
                loss_epoch += loss.get().item()
                
            k+=1

            if batch_remainder != 0:
                optimizer = curr_optims[i].get_optim(worker.id)   
                optimizer.zero_grad()  
                pred = curr_model[i](data_device[k*train_batch_size:k*train_batch_size+batch_remainder])
                loss = criterion(pred, target_device[k*train_batch_size:k*train_batch_size+batch_remainder])
                #loss = F.nll_loss(pred, target[i][j])
                loss.backward()

                optimizer.step()
                loss_epoch += loss.get().item()

            curr_model[i].get()  # get back the model


        th.save(curr_model[i].state_dict(), f'{output_model_folder}/checkpoint_{epoch}_{i}.pth')    

        if epoch % args.log_interval == 0:

            print('Train Epoch: {} | With {} data |: \tLoss: {:.6f}'.format(
                      epoch, worker.id,  loss_epoch))

    return curr_model

def test(test_model, args,fo):
    
    if epoch % args.log_interval == 0:
    
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


from syft.federated.floptimizer import Optims

workers =node_name[:parties]
criterion = nn.CrossEntropyLoss()
model_list, optims_list = init_model(parties)



output_file_name = TAG_NAME+'_AGG'+str(AGG_EPOCH)+'_Adam_decay.csv'
fo = open(output_file_name, "w")

for epoch in range(N_EPOCHS):
    fo.write(str(epoch))
    
    current_model = train(model_list,optims_list, args)
    
    print("----before aggregation----")
    for test_model in current_model:
           
        test(test_model, args,fo)
        
     
    if (epoch+1) % AGG_EPOCH ==0:
        ## model avg
        model_list,optims_list = avgWeight(current_model)
        print("----after aggregation----")
        for test_model in model_list:
            
            test(test_model, args,fo)
    fo.write("\n")          
fo.close()           
            
            
