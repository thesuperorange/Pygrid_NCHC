

# mnist aggregation experiments

## multi-party benchmark
### Setting
* Path: mnist_benchmark/8node
* using pygrid
* execution
```python pygrid_send.py --data_tag mnist_1000_alice_same --n_sample 1000```
* output filepath: output/xxx.csv

### Experimental Results

* Test accurcy of different node number

![image](https://user-images.githubusercontent.com/8772677/129832988-d3999317-6cb8-4550-aac2-76d82b885b29.png)

* Training time of different node number
 
![image](https://user-images.githubusercontent.com/8772677/129832997-4e868939-1ceb-4622-864a-14a782ad337b.png)


## Aggregation (FedAvg)
### Setting
* Path: mnist_benchmark/aggregation
* Pygrid + FedAvg自行implement
* 執行
  ```run_agg_test.py```
* 01-FL-mnist-populate-a-grid-node.ipynb  送檔案
* 02-FL-mnist-train-model.ipynb training (可用run_agg_test.py取代)

### Experimental Results
* aggregation with different parties
  * 2 parties
  
  ![image](https://user-images.githubusercontent.com/8772677/129833162-20c0b66b-a4ab-4751-a602-fa5a8f9cc866.png)

  * 4 parties
  
  ![image](https://user-images.githubusercontent.com/8772677/129833181-d9bb0ca1-0dac-446e-ab23-f935013fb2f3.png)

  * 8 parties

  ![image](https://user-images.githubusercontent.com/8772677/129833227-b524ab1b-4607-4d54-9ade-aefded1851e4.png)

* Accuracy of global/ local model


  ![image](https://user-images.githubusercontent.com/8772677/129833372-ff207d54-dd47-4565-9edf-4a36fcee0f7f.png)



