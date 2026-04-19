# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that learns to prune its own weights during training.
Instead of removing weights after training, the model uses learnable gate parameters to decide which connections are important and which can be ignored.

---

## Idea

Each weight is associated with a gate value. During the forward pass:

Effective Weight = Weight × Sigmoid(Gate Score)

If the gate value becomes very small, the corresponding weight effectively gets removed.

---

## Loss Function

The training objective combines classification performance with sparsity:

Total Loss = CrossEntropy Loss + λ × Sparsity Loss

The sparsity term uses L1 regularization on gate values, which pushes many of them toward zero.

---

## Model

* Convolutional layers for feature extraction
* Custom linear layers with learnable gates
* Dropout and batch normalization for stability

---

## Results

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 0.5    | 81.06%        | 69.37%   |
| 2.0    | 79.97%        | 82.48%   |
| 8.0    | 79.05%        | 94.25%   |

---

## Observations

* Lower lambda keeps more connections and gives better accuracy
* Higher lambda removes more weights but slightly reduces accuracy
* A reasonable balance is achieved around lambda = 2.0

The final classification layer is not pruned much, which suggests it is important for prediction.

---

## Outputs

* Distribution of gate values (histogram)
* Training curves (accuracy, loss, sparsity)
* JSON file with experiment results

---

## How to Run

pip install -r requirements.txt
python main.py

---

## Folder Structure

self-pruning-network/

* main.py
* README.md
* requirements.txt
* outputs/

  * gate_histograms.png
  * training_curves.png
  * experiment_results.json

---

## Conclusion

The model is able to automatically reduce the number of active parameters during training.
This shows the trade-off between model size and performance, where higher sparsity leads to a smaller but slightly less accurate model.

---

## Author

Abhay Tiwari

