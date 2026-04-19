"""
Self-Pruning Neural Network on CIFAR-10
========================================
A neural network that learns to prune its own weights during training
using learnable gate parameters with L1 sparsity regularization.

Key Concepts:
    - PrunableLinear: Custom linear layer with learnable gate scores
    - Sparsity Loss: Normalized L1 penalty on gate values to encourage pruning
    - Automatic Pruning: Gates converge to 0 -> weights are effectively removed

FIXES vs naive implementation:
    1. gate_scores init = +3.0 (sigmoid(3)=0.95) not 0.0 (sigmoid(0)=0.5)
       -> L1 has real distance to push gates down to zero
    2. Sparsity loss NORMALIZED by total gate count -> lambda is scale-invariant
    3. Lambda values tuned for normalized loss: [0.5, 2.0, 8.0]
    4. Gate LR = 10x weight LR so gates respond faster to sparsity pressure
    5. Sparsity threshold = 0.05 (tighter, more accurate reporting)
"""

import os
import sys
import time
import json

# sys.stdout.reconfigure(line_buffering=True)  # not needed in Jupyter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ============================================================================
# Part 1: Custom PrunableLinear Layer
# ============================================================================

class PrunableLinear(nn.Module):
    """
    A linear layer with learnable gate scores for automatic pruning.

    Each weight w_ij has a corresponding gate score g_ij. During the forward
    pass, the effective weight is: w_ij * sigmoid(g_ij).

    The L1 sparsity penalty pushes gate_scores toward large negative values,
    driving sigmoid(g_ij) -> 0, which effectively prunes that connection.

    KEY DESIGN CHOICE — gate_scores initialized to +3.0:
        sigmoid(+3.0) ≈ 0.95  -> gates start near 1 (all connections active)
        The L1 penalty then has real "distance" to push each gate down to 0.

        If initialized to 0.0: sigmoid(0) = 0.5, gradient of sigmoid = 0.25
        -> very slow convergence, gates barely move -> sparsity stays near 0%.

    Parameters:
        in_features  (int): Input dimension
        out_features (int): Output dimension
        bias         (bool): Whether to include bias term
        gate_init    (float): Initial value for gate_scores (default +3.0)
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, gate_init: float = 3.0):
        super(PrunableLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Standard learnable weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Gate scores initialized to gate_init so sigmoid ≈ 0.95 at start
        # (not 0.0 which gives sigmoid = 0.5 and very slow sparsification)
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(gate_init))
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
            gates         = sigmoid(gate_scores)         # in (0, 1)
            pruned_weights = weight * gates              # soft masking
            output        = x @ pruned_weights.T + bias  # linear transform

        Gradients flow through both weight and gate_scores via autograd.
        """
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self) -> torch.Tensor:
        """Return detached gate values for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def get_sparsity(self, threshold: float = 0.05) -> float:
        """% of connections whose gate value is below threshold."""
        g = self.get_gate_values()
        return 100.0 * (g < threshold).sum().item() / g.numel()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ============================================================================
# Part 2: Self-Pruning Network Architecture
# ============================================================================

class SelfPruningNetwork(nn.Module):
    """
    CNN + Prunable FC layers for CIFAR-10.

    Architecture:
        Conv Block 1 : Conv2d(3,32)  -> BN -> ReLU -> Conv2d(32,32) -> BN -> ReLU -> MaxPool -> Dropout
        Conv Block 2 : Conv2d(32,64) -> BN -> ReLU -> Conv2d(64,64) -> BN -> ReLU -> MaxPool -> Dropout
        FC Classifier: PrunableLinear(4096->512) -> ReLU -> Dropout
                    -> PrunableLinear(512->256)  -> ReLU -> Dropout
                    -> PrunableLinear(256->10)
    """

    def __init__(self, gate_init: float = 3.0):
        super(SelfPruningNetwork, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        # 64 channels * 8x8 spatial = 4096 after two 2x2 maxpools on 32x32 input
        self.fc1 = PrunableLinear(64 * 8 * 8, 512, gate_init=gate_init)
        self.fc2 = PrunableLinear(512, 256,        gate_init=gate_init)
        self.fc3 = PrunableLinear(256, 10,         gate_init=gate_init)

        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x));  x = self.dropout_fc(x)
        x = F.relu(self.fc2(x));  x = self.dropout_fc(x)
        x = self.fc3(x)
        return x

    def get_prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def total_gate_count(self) -> int:
        return sum(l.gate_scores.numel() for l in self.get_prunable_layers())

    def get_all_gate_values(self) -> np.ndarray:
        return np.concatenate([
            l.get_gate_values().cpu().numpy().flatten()
            for l in self.get_prunable_layers()
        ])

    def get_overall_sparsity(self, threshold: float = 0.05) -> float:
        gates = np.concatenate([
            l.get_gate_values().cpu().numpy().flatten()
            for l in self.get_prunable_layers()
        ])
        return 100.0 * (gates < threshold).mean()

    def get_layer_sparsities(self, threshold: float = 0.05) -> dict:
        return {
            name: module.get_sparsity(threshold)
            for name, module in self.named_modules()
            if isinstance(module, PrunableLinear)
        }

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        pw    = sum(l.weight.numel()      for l in self.get_prunable_layers())
        pg    = sum(l.gate_scores.numel() for l in self.get_prunable_layers())
        return {"total": total, "prunable_weights": pw,
                "prunable_gate_params": pg, "non_prunable": total - pw - pg}


# ============================================================================
# Part 3: Sparsity-Regularized Loss
# ============================================================================

class SparsityLoss(nn.Module):
    """
    Total Loss = CrossEntropy + lambda * NormalizedSparsityLoss

    NormalizedSparsityLoss = sum(all gate values) / total_gate_count

    WHY NORMALIZE?
        Raw L1 = sum of ~1.3M gates ≈ huge number.
        Without normalization, lambda=1e-4 is effectively ~130 after scaling,
        completely overwhelming CrossEntropy and collapsing accuracy.
        Dividing by N_gates keeps the sparsity loss in [0, 1] range,
        making lambda directly interpretable as a [0,1] penalty weight.

    WHY L1 (not L2)?
        L1 gradient = constant (lambda/N) regardless of gate magnitude.
        This uniform pull drives gates to EXACTLY zero (true sparsity).
        L2 gradient -> 0 as gate -> 0, so weights shrink but never zero out.

    Args:
        lambda_sparse (float): Sparsity penalty weight (tuned for normalized loss)
    """

    def __init__(self, lambda_sparse: float):
        super(SparsityLoss, self).__init__()
        self.lambda_sparse = lambda_sparse
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, model: SelfPruningNetwork):
        ce_loss = self.cross_entropy(outputs, targets)

        # Sum all gate values across all PrunableLinear layers
        raw_sparsity = sum(
            torch.sigmoid(l.gate_scores).sum()
            for l in model.get_prunable_layers()
        )

        # CRITICAL: normalize by total gate count so lambda is scale-invariant
        n_gates       = model.total_gate_count()
        norm_sparsity = raw_sparsity / n_gates   # in [0, 1]

        total_loss = ce_loss + self.lambda_sparse * norm_sparsity

        return total_loss, ce_loss.item(), norm_sparsity.item()


# ============================================================================
# Data Loading
# ============================================================================

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 0):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    run_loss = run_ce = run_sp = correct = total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, ce_val, sp_val = criterion(outputs, targets, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        run_loss += loss.item()
        run_ce   += ce_val
        run_sp   += sp_val
        correct  += outputs.argmax(1).eq(targets).sum().item()
        total    += targets.size(0)

    n = len(train_loader)
    return {
        "loss": run_loss/n, "ce_loss": run_ce/n,
        "sparse_loss": run_sp/n, "accuracy": 100.*correct/total,
    }


def evaluate(model, test_loader, device, threshold=0.05):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            correct += model(inputs).argmax(1).eq(targets).sum().item()
            total   += targets.size(0)
    return {
        "accuracy": 100.*correct/total,
        "sparsity": model.get_overall_sparsity(threshold),
        "layer_sparsities": model.get_layer_sparsities(threshold),
    }


def train_model(lambda_sparse, epochs=30, lr=0.001, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print(f"  TRAINING  lambda = {lambda_sparse}")
    print(f"  Device: {device} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print("="*70 + "\n")

    train_loader, test_loader = get_cifar10_loaders(batch_size)

    model = SelfPruningNetwork(gate_init=3.0).to(device)
    info  = model.count_parameters()
    print(f"  Total params:         {info['total']:,}")
    print(f"  Prunable weights:     {info['prunable_weights']:,}")
    print(f"  Prunable gate params: {info['prunable_gate_params']:,}")
    print(f"  Non-prunable params:  {info['non_prunable']:,}\n")

    criterion = SparsityLoss(lambda_sparse)

    # Separate param groups: gate_scores get 10x higher LR
    weight_p = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    gate_p   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    optimizer = optim.Adam([
        {"params": weight_p, "lr": lr,       "weight_decay": 1e-4},
        {"params": gate_p,   "lr": lr * 10,  "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss":[], "train_acc":[], "test_acc":[],
               "sparsity":[], "ce_loss":[], "sparse_loss":[]}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, epochs+1):
        tm = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        em = evaluate(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(tm["loss"])
        history["train_acc"].append(tm["accuracy"])
        history["test_acc"].append(em["accuracy"])
        history["sparsity"].append(em["sparsity"])
        history["ce_loss"].append(tm["ce_loss"])
        history["sparse_loss"].append(tm["sparse_loss"])

        if em["accuracy"] > best_acc:
            best_acc = em["accuracy"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | Loss:{tm['loss']:.4f} | "
                  f"CE:{tm['ce_loss']:.4f} | TrainAcc:{tm['accuracy']:.2f}% | "
                  f"TestAcc:{em['accuracy']:.2f}% | Sparsity:{em['sparsity']:.2f}%"
                  f" | Time:{time.time()-t0:.0f}s")

    total_time = time.time() - t0
    final = evaluate(model, test_loader, device)

    print(f"\n  --- Final Results  lambda={lambda_sparse} ---")
    print(f"  Best  Test Acc : {best_acc:.2f}%")
    print(f"  Final Test Acc : {final['accuracy']:.2f}%")
    print(f"  Overall Sparse : {final['sparsity']:.2f}%")
    print(f"  Time           : {total_time:.1f}s")
    for ln, sp in final["layer_sparsities"].items():
        print(f"    {ln}: {sp:.2f}%")

    return {
        "model": model, "history": history,
        "final_accuracy": final["accuracy"],
        "best_accuracy": best_acc,
        "final_sparsity": final["sparsity"],
        "layer_sparsities": final["layer_sparsities"],
        "training_time": total_time,
        "lambda": lambda_sparse,
        "gate_values": model.get_all_gate_values(),
    }


# ============================================================================
# Part 4: Visualization & Reporting
# ============================================================================

def plot_gate_histograms(results, save_path="gate_histograms.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    colors = ["#2196F3", "#FF9800", "#F44336"]

    for ax, result, color in zip(axes, results, colors):
        ax.hist(result["gate_values"], bins=100, color=color,
                alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(0.05, color="red", ls="--", lw=1.5, label="Threshold (0.05)")
        ax.set_title(
            f"λ = {result['lambda']}\n"
            f"Acc: {result['final_accuracy']:.2f}%  |  "
            f"Sparsity: {result['final_sparsity']:.2f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        pct = 100.*(result["gate_values"] < 0.05).mean()
        ax.text(0.55, 0.88, f"Pruned: {pct:.1f}%",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    plt.suptitle("Gate Value Distributions — Self-Pruning Network (CIFAR-10)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [OK] Gate histogram saved: {save_path}")


def plot_training_curves(results, save_path="training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#2196F3", "#FF9800", "#F44336"]

    for result, color in zip(results, colors):
        lam  = result["lambda"]
        h    = result["history"]
        eps  = range(1, len(h["train_acc"])+1)
        axes[0,0].plot(eps, h["test_acc"],    color=color, label=f"λ={lam}", lw=2)
        axes[0,1].plot(eps, h["train_loss"],  color=color, label=f"λ={lam}", lw=2)
        axes[1,0].plot(eps, h["sparsity"],    color=color, label=f"λ={lam}", lw=2)
        axes[1,1].plot(eps, h["ce_loss"],     color=color, label=f"λ={lam}", lw=2)

    titles  = ["Test Accuracy vs Epoch", "Total Training Loss vs Epoch",
               "Sparsity (% gates < 0.05) vs Epoch", "Cross-Entropy Loss vs Epoch"]
    ylabels = ["Accuracy (%)", "Loss", "Sparsity (%)", "CE Loss"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics — Self-Pruning Neural Network on CIFAR-10",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Training curves saved: {save_path}")


def print_results_table(results):
    print("\n" + "="*70)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("="*70 + "\n")

    if HAS_TABULATE:
        headers = ["Lambda", "Test Accuracy (%)", "Sparsity (%)", "Training Time (s)"]
        rows = [[f"{r['lambda']:.1e}", f"{r['final_accuracy']:.2f}",
                 f"{r['final_sparsity']:.2f}", f"{r['training_time']:.1f}"]
                for r in results]
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="center"))

        layer_names = list(results[0]["layer_sparsities"].keys())
        lh = ["Layer"] + [f"λ={r['lambda']:.1e}" for r in results]
        lr = [[n]+[f"{r['layer_sparsities'][n]:.2f}%" for r in results] for n in layer_names]
        print(f"\n  Per-Layer Sparsity:")
        print(tabulate(lr, headers=lh, tablefmt="grid", stralign="center"))
    else:
        print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14} {'Time (s)':>12}")
        print("-"*56)
        for r in results:
            print(f"  {r['lambda']:<12.1e} {r['final_accuracy']:>14.2f} "
                  f"{r['final_sparsity']:>14.2f} {r['training_time']:>12.1f}")


def save_results_json(results, save_path="experiment_results.json"):
    data = [{
        "lambda": r["lambda"],
        "final_accuracy": r["final_accuracy"],
        "best_accuracy": r["best_accuracy"],
        "final_sparsity": r["final_sparsity"],
        "layer_sparsities": r["layer_sparsities"],
        "training_time": r["training_time"],
        "history": r["history"],
    } for r in results]
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [OK] Results saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  SELF-PRUNING NEURAL NETWORK — CIFAR-10")
    print("="*70)

    # Lambda values are tuned for NORMALIZED sparsity loss (loss in [0,1])
    # Low=0.5, Medium=2.0, High=8.0
    LAMBDA_VALUES  = [0.5, 2.0, 8.0]
    EPOCHS         = 30
    LEARNING_RATE  = 0.001
    BATCH_SIZE     = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Lambda : {LAMBDA_VALUES}")
    print(f"  Epochs : {EPOCHS}")

    all_results = []
    for lam in LAMBDA_VALUES:
        result = train_model(
            lambda_sparse=lam, epochs=EPOCHS,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, device=device,
        )
        all_results.append(result)

    print_results_table(all_results)
    plot_gate_histograms(all_results,  save_path="gate_histograms.png")
    plot_training_curves(all_results,  save_path="training_curves.png")
    save_results_json(all_results,     save_path="experiment_results.json")

    print("\n" + "="*70)
    print("  DONE — Outputs: gate_histograms.png, training_curves.png, experiment_results.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()