# Breaking the Expression Bottleneck of Graph Neural Networks

[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

This is the code of the paper
*Breaking the Expression Bottleneck of Graph Neural Networks*.

## Table of Contents <!-- omit in toc -->

- [Requirements](#requirements)
- [Run Experiments](#run-experiments)
  - [ogbg-ppa](#ogbg-ppa)
  - [ogbg-code](#ogbg-code)
  - [ogbg-molhiv & ogbg-molpcba](#ogbg-molhiv--ogbg-molpcba)
  - [QM9](#qm9)
  - [TU](#tu)
- [License](#license)

## Requirements

The code is built upon the
[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric),
which has been tested running under Python 3.6.9.

The following packages need to be installed manually:

- `torch==1.5.1`
- `torch-geometric==1.5.0`

The following packages can be installed by running

```sh
python3 -m pip install -r requirements.txt
```

- `ogb==1.2.1`
- `numpy`
- `easydict`
- `tensorboardX`

## Run Experiments

You can run experiments on multiple datasets as follows.

### ogbg-ppa

To run experiments on `ogbg-ppa`, change directory to [ogbg/ppa](ogbg/ppa):

```sh
cd ogbg/ppa
```

You can set hyper-parameters in
[ogbg-ppa.json](ogbg/ppa/ogbg-ppa.json).

You can change `CUDA_VISIBLE_DEVICES` and output directory in
[run_script.sh](ogbg/ppa/run_script.sh).

Then, run the following script:

```sh
./run_script.sh
```

### ogbg-code

To run experiments on `ogbg-code`, change directory to [ogbg/code](ogbg/code):

```sh
cd ogbg/code
```

You can set hyper-parameters in
[ogbg-code.json](ogbg/code/ogbg-code.json).

You can change `CUDA_VISIBLE_DEVICES` and output directory in
[run_script.sh](ogbg/code/run_script.sh).

Then, run the following script:

```sh
./run_script.sh
```

### ogbg-molhiv & ogbg-molpcba

To run experiments on `ogbg-mol*`, change directory to [ogbg/mol](ogbg/mol):

```sh
cd ogbg/mol
```

You can set dataset name and hyper-parameters in
[ogbg-mol.json](ogbg/mol/ogbg-mol.json).
Dataset name should be either `ogbg-molhiv` or `ogbg-molpcba`.

You can change `CUDA_VISIBLE_DEVICES` and output directory in
[run_script.sh](ogbg/mol/run_script.sh).

Then, run the following script:

```sh
./run_script.sh
```

### QM9

To run experiments on `QM9`, change directory to [qm9](qm9):

```sh
cd qm9
```

You can set hyper-parameters in [QM9.json](qm9/QM9.json).

You can change `CUDA_VISIBLE_DEVICES` and output directory in
[run_script.sh](qm9/run_script.sh).

Then, run the following script:

```sh
./run_script.sh
```

### TU

To run experiments on `TU`, change directory to [tu](tu):

```sh
cd tu
```

There are several datesets in TU.
You can set dataset name in [run_script.sh](tu/run_script.sh)
and set hyper-parameters in [configs/\<dataset\>.json](tu/configs).

You can change `CUDA_VISIBLE_DEVICES` and output directory in
[run_script.sh](tu/run_script.sh).

Then, run the following script:

```sh
./run_script.sh
```

## Reference
```
@ARTICLE {yang2022breaking,
	author = {Yang, Mingqi and Wang, Renjian and Shen, Yanming and Qi, Heng and Yin, Baocai},
	journal = {IEEE Transactions on Knowledge & Data Engineering},
	title = {Breaking the Expression Bottleneck of Graph Neural Networks},
	year = {2022},
	doi = {10.1109/TKDE.2022.3168070},
	address = {Los Alamitos, CA, USA}
}
```

## License

[MIT License](LICENSE)
