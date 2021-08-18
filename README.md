
# AetherAI
* path: Pygrid_aetherAI
* 01-FL-mnist-populate-a-grid-node.ipynb 由雲象送出資料
* 02-FL-mnist-train-model.ipynb由NCHC來train

# mnist aggregation experiments

## multi-party benchmark
* Path: mnist_benchmark/8node
* using pygrid
* execution
```python pygrid_send.py --data_tag mnist_1000_alice_same --n_sample 1000```
* output filepath: output/xxx.csv


## Aggregation (FedAvg)
* Path: mnist_benchmark/aggregation
* Pygrid + FedAvg自行implement
* 執行
  ```run_agg_test.py```
* 01-FL-mnist-populate-a-grid-node.ipynb  送檔案
* 02-FL-mnist-train-model.ipynb training (可用run_agg_test.py取代)

## FedAvg non-iid local(by 日本人)
* https://github.com/thesuperorange/fedavg.pytorch
* from: https://github.com/katsura-jp/fedavg.pytorch
* 執行方法: 參考4/7 notion筆記 (跑resnet)
* 改code
  * 加入transform 224*224
  * fedavg.pytorch/src/datasets/mnist.py

  ```
  self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
  ```
  * model改resnet18
  * fedavg.pytorch/train.py
    ```
    model = torchvision.models.resnet18(pretrained=False)        
    model.conv1 = nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    #batch = torch.rand(4, 1, 224, 224) # in the NCHW format
    #model(batch).size()
    in_feature =  model.fc.in_features
    model.fc = nn.Linear(in_feature, 10)  # ch
    ```
 
  * 加入mGPU
  ```
  if cfg.mGPU == True:    
        model = nn.DataParallel(model)
  ```
* setting
  ```
  device: 4GPU
  n_round: 100
  E: 5
  K: 100
  C: 1
  B: 10
  iid: true
  mGPU: true
  seed: 2020
  root: ${env:PWD}
  savedir: ${env:PWD}/output/${fed.type}/${now:%Y-%m-%d_%H-%M-%S}
  model: ResNet18
  args:
  in_features: 784
  num_classes: 10
  hidden_dim: 200
  fed:
  type: fedavg
  classname: FedAvg
  optim:
  type: sgd
  classname: SGD
  args:
  lr: 0.01
  momentum: 0.9
  weight_decay: 1.0e-05
  ```
* 執行
  * iid
  ```python train.py iid=True mGPU=True```  
  * non-iid
  ```fedavg.pytorch$ python train.py iid=False mGPU=True```
  
## FedSGD (2021.5.6)
### FedSGD_LeiDu
* from: https://github.com/LeiDu-dev/FedSGD
* 說明：原本cifar10 +LeNet, 改成tti版本network +mnist
* 執行 FedSGD_LeiDu/FedSGD.ipynb

### local
* Path: FL_pygrid2/mnist_benchmark/mnist_FedSGD.ipynb
* 說明：tti版本network+mnist (Adam改SGD才會收斂)
