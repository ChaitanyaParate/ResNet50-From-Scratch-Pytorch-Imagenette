# ResNet50 From Scratch -- Imagenette Classification (PyTorch)

This project implements a **fully custom ResNet-50 architecture from scratch** using **PyTorch**, trained and evaluated on the **Imagenette** dataset.
The goal is to build and train a high-performance convolutional neural network **without relying on torchvision's ResNet implementation**, showcasing full understanding of CNN architecture design, training engineering, and deep learning best practices.

## 🚀 Project Highlights

### 🧠 Custom ResNet50 Implementation

A complete ResNet-50 replication including: - Conv1 stem (7×7 + MaxPool) - 4 stages of Bottleneck blocks: 3, 4, 6, 3 - Identity &
projection skip connections - AdaptiveAvgPool - Dynamic fully-connected classifier head

## 🎨 Advanced Data Augmentation

Using Albumentations: - Resize to 224×224 - Random rotation (±15°) - Horizontal & vertical flips - Normalization - ToTensorV2

## ⚙️ Training Pipeline

-   Mixed precision (`torch.amp.GradScaler`)
-   Adam + weight decay
-   Label smoothing (0.1)
-   ReduceLROnPlateau scheduler
-   Checkpointing
-   Softmax accuracy evaluation
-   TQDM progress bars

## 📂 Project Structure

    dataset.py
    model.py
    train.py
    utils.py
    README.md

## 🧩 Model Architecture Summary

### Bottleneck Block

-   1×1 → 3×3 → 1×1 convs
-   BatchNorm + ReLU
-   Identity / projection skip

### Dynamic Classifier

Input dim inferred at first forward pass.

## 🏋️ Training

    python train.py

## 📊 Performance

High accuracy on Imagenette using augmentations + label smoothing + LR scheduling.

## 🔧 Requirements

    torch
    torchvision
    albumentations
    tqdm

## 🧠 Demonstrated Skills

-   Custom deep learning architecture engineering
-   PyTorch internals
-   Data augmentation pipelines
-   Training optimization techniques
-   Model evaluation & debugging

## 🚀 Future Improvements

-   MixUp/CutMix
-   Cosine Annealing
-   EMA weights
-   TensorBoard
-   ONNX export
