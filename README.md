# [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks] Implementation in PyTorch and additional features


**Authors**: [Valentin Exbrayat](https://github.com/valdo92), [Tom Mariani](https://github.com/t-mariani), [Hugo Pavy](https://github.com/hpavy)

---

## Overview
This repository is a PyTorch implementation of the paper:
**[Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks]** ([arXiv:1704.06803](https://arxiv.org/pdf/1704.06803))

We aim to provide a clean, modular, and well-documented implementation of the original paper, along with additional features and improvements.


```
@misc{monti2017geometricmatrixcompletionrecurrent,
      title={Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks}, 
      author={Federico Monti and Michael M. Bronstein and Xavier Bresson},
      year={2017},
      eprint={1704.06803},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1704.06803}, 
}
```

## Table of Contents
- [\[Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks\] Implementation in PyTorch and additional features](#geometric-matrix-completion-with-recurrent-multi-graph-neural-networks-implementation-in-pytorch-and-additional-features)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [TODO](#todo)
  - [Roadmap](#roadmap)
  - [Installation](#installation)
    - [Setup](#setup)

---

## Features
- Faithful PyTorch implementation of the original paper
- Modular and extensible codebase
- Additional features (see [Roadmap](#roadmap))

---

## TODO

- [ ] Importing the data
- [ ] Splitting the data train, test, val
- [ ] Creating the dataset for pytorch
- [ ] Coding the model
- [ ] Coding the test loop
- [ ] Filling the code of the train loop
- [ ] Creating visualization
- [ ] Saving the model (saving the best model ?)

## Roadmap

## Installation

### Setup
- Using poetry ```pip install poetry```
- ```poetry install```
