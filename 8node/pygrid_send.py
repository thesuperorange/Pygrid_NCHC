import syft as sy
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient  # websocket client. It sends commands to the node servers
from syft.grid.public_grid import PublicGridNetwork

import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import requests
import argparse

from syft.federated.floptimizer import Optims


def syft_conf(hook):
    

    # Connect direcly to grid nodes
    compute_nodes = {}

    compute_nodes["gridnode01"] = DataCentricFLClient(hook, gridnode01)
    compute_nodes["gridnode02"]   = DataCentricFLClient(hook, gridnode02) 
    #compute_nodes["Charlie"]   = DataCentricFLClient(hook, charlie_address) 

    # Check if they are connected
    for key, value in compute_nodes.items(): 
        print("Is " + key + " connected?: " + str(value.ws.connected))
    
    return compute_nodes
    

def load_data(MNIST_PATH, compute_nodes, transform):
    # Define a transformation.


    # Download and load MNIST dataset
    trainset = datasets.MNIST(MNIST_PATH, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_SAMPLES, shuffle=True)

    dataiter = iter(trainloader)
    images_train_mnist, labels_train_mnist = dataiter.next()  # Train images and their labels

    if useGPU == True:
        images_train_mnist = images_train_mnist.to(device)
        labels_train_mnist = labels_train_mnist.to(device)
    
    
    
    #split data
    images_train_mnist = torch.split(images_train_mnist, int(len(images_train_mnist) / len(compute_nodes)), dim=0 ) #tuple of chunks (dataset / number of nodes)
    labels_train_mnist   = torch.split(labels_train_mnist, int(len(labels_train_mnist) / len(compute_nodes)), dim=0 )  #tuple of chunks (labels / number of nodes)
    
    return images_train_mnist, labels_train_mnist

def load_testdata(MNIST_PATH, transform):
    testset = datasets.MNIST(MNIST_PATH, download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    return testloader

def send_data(compute_nodes, images_train_mnist, labels_train_mnist, TAG_NAME):
    
    for index, _ in enumerate(compute_nodes):

        images_train_mnist[index]\
            .tag("#X", "#"+TAG_NAME, "#dataset")\
            .describe("The input datapoints to the MNIST dataset.") 


        labels_train_mnist[index]\
            .tag("#Y", "#"+TAG_NAME, "#dataset") \
            .describe("The input labels to the MNIST dataset.")


    for index, key in enumerate(compute_nodes):

        print("Sending data to", key)

        images_train_mnist[index].send(compute_nodes[key], garbage_collect_data=False)
        labels_train_mnist[index].send(compute_nodes[key], garbage_collect_data=False)



    print("Alice's tags: ", requests.get(gridnode01 + "/data-centric/dataset-tags").json())
    print("Bob's tags: ",   requests.get(gridnode02   + "/data-centric/dataset-tags").json())
    #print("charlie's tags: ",   requests.get(charlie_address   + "/data-centric/dataset-tags").json())

class Arguments():
    def __init__(self,N_TEST, N_EPOCHS, device):
        self.test_batch_size = N_TEST
        self.epochs = N_EPOCHS
        self.lr = 0.01
        self.log_interval = 5
        #self.device = th.device("cuda")
        self.device = device
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

def search_grid(my_grid,tag_name):
    data = my_grid.search("#X", "#"+tag_name, "#dataset")  # images
    target = my_grid.search("#Y", "#"+tag_name, "#dataset")  # labels

    data = list(data.values())  # returns a pointer
    target = list(target.values())  # returns a pointer
    print(data)
    print(target)
    return data, target

    
# epoch size
def epoch_total_size(data):
    total = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            total += data[i][j].shape[0]
            
    return total

def train(args, model,data, target, optims, epoch):
    
    model.train()
    epoch_total = epoch_total_size(data)
    
    current_epoch_size = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            
            current_epoch_size += len(data[i][j])
            worker = data[i][j].location  # worker hosts data
            
            model.send(worker)  # send model to PyGridNode worker
            
            optimizer = optims.get_optim(worker.id)
            
            optimizer.zero_grad()  
            
            pred = model(data[i][j])
            loss = F.nll_loss(pred, target[i][j])
            loss.backward()
            
            optimizer.step()
            model.get()  # get back the model
            
            loss = loss.get()
            
        if epoch % args.log_interval == 0:

            print('Train Epoch: {} | With {} data |: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, worker.id, current_epoch_size, epoch_total,
                            100. *  current_epoch_size / epoch_total, loss.item()))

def test(testloader, args,model, fo, epoch):
    
    if epoch % args.log_interval == 0:
    
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)
        fo.write("{},{:.4f},{:.0f}\n".format(epoch, test_loss,100. * correct / len(testloader.dataset)))
 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
    
def main(args):
    

    hook = sy.TorchHook(torch)
    
    nodes = syft_conf(hook)    
    
    
    transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),  #  mean and std 
                              ])
        
        
    img, label = load_data(MNIST_PATH,nodes, transform)
    send_data(nodes, img, label, TAG_NAME)
    
    # Connect direcly to grid nodes
    my_grid = PublicGridNetwork(hook, grid_address)
    
    model = Net()
    model.to(args.device)
    
    optims = Optims(nodes, optim=optim.Adam(params=model.parameters(),lr=0.01))
    

    #optimizer = optim.SGD(model.parameters(), lr=0.01)

    
    testloader = load_testdata(MNIST_PATH, transform)
    data, target = search_grid(my_grid,TAG_NAME)
    
## start training
    output_file_name = TAG_NAME+'.csv'
    fo = open(output_file_name, "w")
    for epoch in range(args.epochs):
        train(args, model, data, target, optims, epoch)
        test(testloader, args, model, fo, epoch)
    fo.close()
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_tag',  help='data tag name')
    parser.add_argument('--epoch', default=100, help='epoch')
    parser.add_argument('--test_batch_size', default=10, help='test batch size')
    parser.add_argument('--n_sample', default=10000, help='n sample (training data batch size)')
    args = vars(parser.parse_args())
    EPOCH = int(args['epoch'])
    TAG_NAME = args['data_tag']
    N_SAMPLES = int(args['n_sample'])
    TEST_BATCH_SIZE = int(args['test_batch_size'])
    
    
    
    
    MNIST_PATH = './dataset2'  # Path to save MNIST dataset
    
    #-----------------setting--------------------

    # address setting
#     alice_address = "http://alice.libthomas.org:80" 
#     bob_address   = "http://bob.libthomas.org:80"
#     grid_address = "http://203.145.218.196:80"  # address

        
#     alice_address = "http://alice:5000" 
#     bob_address   = "http://bob:5001"
#     charlie_address = "http://charlie:5002"
#     grid_address = "http://network:7000"  # address

    gridnode01 = "http://203.145.219.187:53980"
    gridnode02 = "http://203.145.219.187:53946"
    gridnode03 = "http://203.145.219.187:53359"
    gridnode04 = "http://203.145.219.187:56716"
    gridnode05 = "http://203.145.219.187:57096"
    gridnode06 = "http://203.145.219.187:55194"
    gridnode07 = "http://203.145.219.187:57574"
    gridnode08 = "http://203.145.219.187:52228"
        
    grid_address = "http://203.145.221.20:80"
    
    # device setting
    useGPU = False

    if useGPU == True:
        if not torch.cuda.is_available():
            print("no cuda available")
            exit()
        else:
            device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    else:

        device = torch.device("cpu")
        
    args = Arguments(TEST_BATCH_SIZE, EPOCH, device)    
        
    main(args)
    
    