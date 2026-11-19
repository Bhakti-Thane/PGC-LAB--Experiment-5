# Experiment 5: Neural Network Layers in CUDA

## 📌 Project Overview

This experiment implements basic neural network layers using custom CUDA kernels. It compares CPU and GPU execution, validates numerical correctness, measures execution time, and calculates GPU speedup.

The implementation is performed using **Google Colab with an NVIDIA Tesla T4 GPU**.

## 🎯 Aim

To implement and evaluate basic neural network operations using CUDA:

- Matrix Multiplication with Bias
- ReLU Activation
- Sigmoid Activation

To compare CPU and GPU performance and verify numerical correctness.

## 🛠️ Technologies Used

- **Programming Language:** C++ / CUDA C++
- **Platform:** Google Colab
- **GPU:** NVIDIA Tesla T4
- **CUDA Toolkit:** CUDA 13.0 (runtime compiler version may vary)
- **Compiler:** NVCC

## 🧠 Neural Network Architecture

The experiment implements a two-layer feed-forward neural network.

| Layer | Dimensions | Activation |
|---|---|---|
| Input Layer | 256 × 128 | None |
| Hidden Layer | 256 × 64 | ReLU |
| Output Layer | 256 × 10 | Sigmoid |

### Forward Propagation

**Layer 1:**

Z1 = X × W1 + b1

A1 = ReLU(Z1)

**Layer 2:**

Z2 = A1 × W2 + b2

Y = Sigmoid(Z2)

## ⚙️ CUDA Kernels Implemented

### 1. Matrix Multiplication with Bias
Performs matrix multiplication and adds bias to each output element.

### 2. ReLU Activation
ReLU(x) = max(0, x)

Converts negative values to zero while keeping positive values unchanged.

### 3. Sigmoid Activation
Sigmoid(x) = 1 / (1 + exp(-x))

Converts input values into a range between 0 and 1.

## 💻 Implementation Steps

1. Detect the NVIDIA GPU.
2. Verify CUDA compiler using NVCC.
3. Define neural network dimensions and constants.
4. Implement CUDA kernels.
5. Implement CPU functions.
6. Implement GPU forward propagation.
7. Test individual CUDA kernels.
8. Allocate GPU memory and transfer data.
9. Measure CPU and GPU execution time.
10. Compare outputs and calculate speedup.

## 📊 Performance Evaluation

The experiment measures:

- CPU Average Compute Time
- GPU Kernel-only Execution Time
- GPU End-to-End Execution Time
- Kernel-only Speedup
- End-to-End Speedup
- Maximum Absolute Output Difference

### Speedup Formula

Kernel-only Speedup:

CPU Time / GPU Kernel-only Time

End-to-End Speedup:

CPU Time / GPU End-to-End Time

## ✅ Validation

The program validates the three CUDA kernels using test inputs with known expected outputs.

Expected validation results:

- Matrix Multiplication + Bias: PASSED
- ReLU: PASSED
- Sigmoid: PASSED
- CPU and GPU Outputs: MATCH (within tolerance)

## 📁 Project Structure

```text
Experiment_5_CUDA/
│
├── README.md
├── neural_network.cu
├── Experiment_5_Neural_Network_Layers_CUDA.ipynb
│
└── Screenshots/
    ├── 01_GPU_Detection.png
    ├── 02_CUDA_Compiler.png
    ├── 03_CUDA_Kernels.png
    ├── 04_CPU_GPU_Implementation.png
    ├── 05_Kernel_Test_Results.png
    └── 06_Final_Output.png
```

## ▶️ How to Run

1. Open Google Colab.
2. Select an NVIDIA T4 GPU runtime (if available).
3. Upload or open the Jupyter Notebook.
4. Execute the implementation cells in sequence.
5. Compile the CUDA program:

```bash
!nvcc -O2 neural_network.cu -o neural_network
```

6. Execute the program:

```bash
!./neural_network
```

## 📌 Important Note

This experiment performs forward propagation (inference) only. It does not include model training, loss calculation, optimization, or backpropagation.

Execution timings may vary depending on GPU hardware, compiler, runtime and system conditions.

## 🎓 Learning Outcomes

- Understanding CUDA kernels, blocks and threads.
- Implementing neural network operations using CUDA.
- Understanding CPU and GPU memory transfers.
- Measuring GPU execution performance.
- Comparing CPU and GPU numerical results.
- Calculating GPU speedup.

---

**Experiment:** 5 – Neural Network Layers in CUDA  
**Subject:** Cloud / GPU Computing Laboratory  
**Platform:** Google Colab  
**GPU:** NVIDIA Tesla T4
