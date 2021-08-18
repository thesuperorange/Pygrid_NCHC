
import torch
import numpy as np
import torch.utils.data as datautil


def load_data(train_data, train_label,BATCH_SIZE):
    x_train_tensor = torch.from_numpy(train_data).float()
    y_train_tensor = torch.from_numpy(train_label).long() 

    train_dataset = datautil.TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = datautil.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2,
    )

    return train_loader

def read_mnist_data(npz_filepath,batch):
       
    # load npz file
    with np.load(npz_filepath) as data:
        x_train=data['x_train']
        y_train=data['y_train']
        x_test=data['x_test']
        y_test=data['y_test']
        X_train = x_train.reshape(len(x_train),1,28,28)
        X_test = x_test.reshape(len(x_test),1,28,28)

    train_loader = load_data(X_train, y_train,batch)
    test_loader = load_data(X_test, y_test,batch)
    return train_loader,test_loader

