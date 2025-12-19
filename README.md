# [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks] Implementation in PyTorch and additional features


**Authors**: [Valentin Exbrayat](https://github.com/valdo92), [Tom Mariani](https://github.com/t-mariani), [Hugo Pavy](https://github.com/hpavy)

---

## Overview
This repository is a PyTorch implementation of the paper:
**[Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks]** ([arXiv:1704.06803](https://arxiv.org/pdf/1704.06803))

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

We aim to provide a clean, modular, and well-documented implementation of the original paper, along with additional features and improvements.

You can find the report of our work in the file [report.pdf](report.pdf).


## Table of Contents
- [\[Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks\] Implementation in PyTorch and additional features](#geometric-matrix-completion-with-recurrent-multi-graph-neural-networks-implementation-in-pytorch-and-additional-features)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Code of the original paper](#code-of-the-original-paper)
    - [Minimal run snippet](#minimal-run-snippet)
  - [Our additional features](#our-additional-features)
    - [Setup](#setup)

---

## Code of the original paper
This repository reimplements the core components from "Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks" in a modular PyTorch codebase. Below is a quick map from the paper’s formulation to the implemented modules and scripts.

- **Data + Laplacians:** [src/multi_graph_cnn/data.py](src/multi_graph_cnn/data.py)


- **Model (MGCNN):** [src/multi_graph_cnn/model.py](src/multi_graph_cnn/model.py)


- **Losses and Metrics:** [src/multi_graph_cnn/loss.py](src/multi_graph_cnn/loss.py)
 

- **Training loop:** [find_impact_graph/training.py](find_impact_graph/training.py)
  
### Minimal run snippet
Below is a minimal run sketch to train and evaluate on one dataset. Adjust paths and configs as needed.

1) Configure `config.yaml` (or `config_MGCNN.yaml`) with dataset, training rates, and model hyperparameters.
2) Run training (example):

```bash
python main.py
```

Outputs include TensorBoard logs (under `saved_models/.../runs/`), figures in `result_dir`, and best model weights in `output_dir`.

## Our additional features


**Graph/Filter insights**: [src/multi_graph_cnn/graph_insights.py](src/multi_graph_cnn/graph_insights.py)

  **Visualization:** [src/multi_graph_cnn/visualization.py](src/multi_graph_cnn/visualization.py)
  - Heatmaps for matrix completion, prediction vs. ground-truth, error distributions, and filter influence.




**Understanding the impact of the graph**: 

**Representation of the learned user and movie embeddings**:


### Setup
- Using poetry ```pip install poetry```
- ```poetry install```
