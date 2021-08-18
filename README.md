

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

## Experimental Results
