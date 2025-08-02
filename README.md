# Efficient External 1-WL Coloring via GPU Acceleration
Tests code for the paper "Efficient External 1-WL Coloring via GPU Acceleration". PyTorch datasets are not present for space limitations while datasets from the LAW collection are downloaded by the test scripts.

# Requirements
- CUDA version >= 12
- Cmake version >= 3.9
- g++
- python3
- cargo (to compile [webgraph-rs](https://github.com/vigna/webgraph-rs), used to decompress the LAW datasets)
- sqlite3 (to read results)
  
# Installation
Run 
`./install.sh`
# Running normal tests
Run `python3 bench.py number_of_gpus_to_be_used`  
experiment times will be saved in the results table of the `bench.db` sqlite3 database. Each row corresponds to one dataset and have the following fields:

- **name** name of the dataset
- **tool** name of the tool to use for the decompression of the dataset
- **nodes** number of nodes of the dataset
- **edges** number of edges of the dataset
- **cpu** result of the RandomBE CPU implementation
- **valmari** result of the partition refinement CPU implementation
- **cuda** result of the GPU BatchedBE implementation with a batch size of 100%
- **cuda_xx** result of the GPU BatchedBE implementation with a batch size of xx%
- **status** has a value of 1 if all the experiments on this dataset have been completed

results for each of the implementations is saved as a JSON object having the following keys:

- **nodes** number of nodes of the resulting graph, if the experiment have not been completed successfully (e.g. timeout) this field have a value of 0
- **edges** number of edges of the resulting graph, if the experiment have not been completed successfully (e.g. timeout) this field have a value of 0
- **time** runtime in seconds of the experiment capped at the timeout time
- **done** is equal to 1 if the result is the coarsest possible BE reduction of the original graph otherwise equal to 0
  
# Running large scale tests
Run `python3 benchlarge.py number_of_gpus_to_be_used`  
times will be saved in the results table of `benchlarge.db` sqlite3 database. Each row corresponds to one dataset and have the following fields:

- **name** name of the dataset
- **tool** name of the tool to use for the decompression of the dataset
- **nodes** number of nodes of the dataset
- **edges** number of edges of the dataset
- **cpu** result of the RandomBE CPU implementation
- **valmari** result of the partition refinement CPU implementation
- **cuda** result of the GPU BatchedBE implementation with a batch size of 1.8 billion edges
- **status** has a value of 1 if all the experiments on this dataset have been completed

results for each of the implementations are saved in the same way of the ones of the normal tests.
