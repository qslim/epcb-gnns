# Breaking the Expressive Bottlenecks of Graph Neural Networks

The code is built upon the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).

## Prerequisites

python 3.6.9

pytorch 1.5.1

pytorch-geometric 1.5.0

ogb 1.2.1

Additional modules: numpy, easydict, tensorboardx

## Running the tests

#### Running the ogbg-ppa test.

   - ```cd ogbg/ppa```
   - Set hyper-parameters in [ogbg-ppa.json](ogbg/ppa/ogbg-ppa.json).
   - Set CUDA_VISIBLE_DEVICES and output directory in [run_script.sh](ogbg/ppa/run_script.sh).
   - ```chmod +x run_script.sh```
   - ```./run_script.sh```
   
####  Running the ogbg-code test.

   - ```cd ogbg/code```
   - Set hyper-parameters in [ogbg-code.json](ogbg/code/ogbg-code.json).
   - Set CUDA_VISIBLE_DEVICES and output directory in [run_script.sh](ogbg/code/run_script.sh).
   - ```chmod +x run_script.sh```
   - `````./run_script.sh`````
   
####  Running the ogbg-molhiv test.

   - ```cd ogbg/mol```
   - Set hyper-parameters in [ogbg-mol.json](ogbg/mol/ogbg-mol.json).
   - Set CUDA_VISIBLE_DEVICES and output directory in [run_script.sh](ogbg/mol/run_script.sh).
   - ```chmod +x run_script.sh```
   - ```./run_script.sh```
   
####  Running the QM9 test.

   - ```cd ogbg/qm9```
   - Set hyper-parameters in [QM9.json](qm9/QM9.json).
   - Set CUDA_VISIBLE_DEVICES and output directory in [run_script.sh](qm9/run_script.sh).
   - ```chmod +x run_script.sh```
   - ```./run_script.sh```

