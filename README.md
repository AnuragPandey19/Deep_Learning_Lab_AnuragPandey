# 🧠 Deep Learning Experiments

<div align="center">

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Experiments-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**A comprehensive collection of deep learning experiments from fundamentals to advanced applications**

[View Experiments](#-experiments) • [Setup](#️-setup--installation) • [Technologies](#-technologies-used)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [Repository Structure](#-repository-structure)
- [Experiments](#-experiments)
- [Setup & Installation](#️-setup--installation)
- [Technologies Used](#-technologies-used)

---

## 🎯 About

This repository contains hands-on implementations of deep learning concepts, covering everything from basic neural network components to advanced transfer learning techniques. Each experiment is designed to build practical understanding through implementation and analysis.

---

## 📁 Repository Structure

```
📦 Deep-Learning-Experiments
┣ 📂 Exp_1
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_2
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_3
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_4
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_5
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_6
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_7
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┣ 📂 Exp_8
┃ ┣ 📓 experiment.ipynb
┃ ┣ 📂 datasets
┃ ┗ 📂 images
┗ 📄 README.md
```

> Each experiment folder is self-contained with its notebook, datasets, and generated visualizations.

---

## 🔬 Experiments

<table>
<tr>
<td width="50%">

### 📊 Experiment 1
**Comparative Study of Deep Learning Frameworks**

```
Topics:
├── TensorFlow Implementation
├── Keras Implementation
├── PyTorch Implementation
└── Framework Comparison
```

Compare TensorFlow, Keras, and PyTorch by implementing linear regression. Analyze code verbosity, API design patterns, and debugging capabilities across frameworks.

</td>
<td width="50%">

### 🔧 Experiment 2
**Neural Networks from Scratch**

```
Topics:
├── Single Neuron (AND Gate)
├── Feedforward Network (XOR)
├── MLP with Backpropagation
└── Activation & Loss Functions
```

Build neural network components from ground up without high-level libraries. Implement forward propagation, backpropagation, and training mechanisms.

</td>
</tr>

<tr>
<td width="50%">

### 🎯 Experiment 3
**Classification with DL Frameworks**

```
Topics:
├── Dataset: MNIST/Fashion-MNIST
├── Data Preprocessing
├── Model Training & Validation
└── Performance Evaluation
```

End-to-end classification pipeline using deep learning frameworks. Includes data normalization, model building, training curves, and confusion matrix analysis.

</td>
<td width="50%">

### 🖼️ Experiment 4
**Transfer Learning for Image Classification**

```
Topics:
├── Pretrained Models
├── Feature Extraction
├── Fine-Tuning Strategies
└── Cats vs Dogs / CIFAR-10
```

Leverage pretrained models (ResNet, EfficientNet, MobileNet) for image classification. Implement both feature extraction and fine-tuning approaches.

</td>
</tr>

<tr>
<td width="50%">

### ⚡ Experiment 5
**Training Deep Networks**

```
Topics:
├── Activation Functions Visualization
├── Loss Functions Implementation
├── Backpropagation Algorithm
└── Optimizer Comparison
```

Deep dive into training mechanisms. Visualize activation functions (Sigmoid, ReLU, Tanh, Softmax) and loss functions. Compare SGD, Momentum, and Adam optimizers.

</td>
<td width="50%">

### 🔷 Experiment 6
**Multi-Layer Perceptron**

```
Topics:
├── MLP Architecture Design
├── Layer Configuration
├── Hyperparameter Tuning
└── Classification Tasks
```

Build and train MLP architectures with various configurations. Explore different layer depths, neuron counts, and activation strategies.

</td>
</tr>

<tr>
<td colspan="2">

### 🖥️ Experiment 7
**Convolutional Neural Networks**

```
Topics:
├── Convolution Operations          ├── Pooling Layers (Max, Average)
├── Feature Map Extraction          └── CNN Architecture Design
```

Implement CNN components from scratch. Visualize learned features through feature maps and understand how convolution and pooling operations work.

</td>
</tr>

<tr>
<td colspan="2">

### 🎨 Experiment 8
**CNN with Data Augmentation for Image Classification**

```
Topics:
├── Data Augmentation Techniques    ├── Image Transformations (Rotation, Flip, Zoom)
├── CNN Model Training              └── Performance Comparison with/without Augmentation
```

Implement CNN with data augmentation strategies to improve model generalization. Apply various image transformations and analyze their impact on classification accuracy.

</td>
</tr>
</table>

---

## 🛠️ Setup & Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
GPU (Optional, for faster training)
```

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd deep-learning-experiments

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow keras torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook
```

### Running an Experiment

```bash
# Navigate to experiment folder
cd Exp_1

# Open the notebook
jupyter notebook experiment.ipynb

# Or use JupyterLab
jupyter lab experiment.ipynb
```

---

## 🔧 Technologies Used

<div align="center">

| Framework | Version | Purpose |
|-----------|---------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow) | 2.x | Deep Learning Framework |
| ![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras) | 2.x | High-level Neural Networks API |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch) | 2.x | Deep Learning Framework |
| ![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy) | 1.x | Numerical Computing |
| ![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas) | 2.x | Data Manipulation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c?logo=python) | 3.x | Data Visualization |
| ![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-1.x-F7931E?logo=scikit-learn) | 1.x | Machine Learning Tools |

</div>

---

<div align="center">

### 🌟 Star this repository if you find it helpful!

**Anurag Pandey**

</div>
