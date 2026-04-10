# Iron-Linalg: High-Performance Linear Algebra for Systems-Level AI

## Project Vision
My goal is to build a high-performance, system-level AI library in Rust, specifically optimized for **STM32 (Nucleo) microcontrollers**. I want to move away from the "luxury" of high-level abstractions like NumPy to understand the fundamental "atoms" of computation. By rebuilding everything from scratch—from memory management to advanced matrix decompositions—I am preparing to deploy real-time signal processing and AI models directly onto bare-metal hardware.

---

## What I Have Built So Far

### 1. Core Data Structures
*   **Vector Engine**: Implemented a robust `Vector` structure supporting L1, L2, and Infinity norms, dot products, and orthogonal projections.
*   **Matrix Engine**: Designed a `Matrix` structure using **Row-Major Format (RMF)** for contiguous memory access.
*   **Zero-Copy Operations**: Implemented matrix multiplication (`matmul_accumulate`) with virtual transposition flags to save memory on embedded devices.
*   **Operator Overloading**: Fully implemented Rust traits (`Add`, `Sub`, `Mul`, `PartialEq`) for intuitive mathematical syntax (e.g., `&A * &B`).

### 2. Fundamental Algorithms
*   **Gram-Schmidt Process**: A stable implementation to transform any set of vectors into an orthonormal basis.
*   **QR Decomposition ($A = QR$)**: The gold standard for numerical stability, decomposing matrices into orthogonal ($Q$) and upper triangular ($R$) components.
*   **Linear System Solver ($Ax = b$)**: A complete solver combining QR decomposition and **Back-substitution** to find solutions even for non-square matrices (Least Squares).

---

## Lessons Learned & Challenges

### 1. The Battle with Floating Point Precision
One of my biggest "Aha!" moments was realizing that `assert_eq!(12.0, 12.000002)` fails. I learned that in numerical computing, strict equality is a myth. I had to manually implement `PartialEq` using a custom **Epsilon** tolerance to account for cumulative round-off errors during complex transformations like QR.

### 2. Ownership and Memory Gymnastics
Coming from high-level languages, managing Rust's ownership was a challenge. I had to carefully design my operators to work with **references** (`&Vector`, `&Matrix`) to avoid destroying my objects during intermediate calculations—a critical skill for memory-constrained STM32 environments.

### 3. Indexing Logic (Row-Major)
Implementing back-substitution and matrix-vector bridges required intense focus on indexing math. Mapping a 2D matrix onto a flat 1D `Vec<f32>` taught me exactly how data flows through the CPU cache, which is the foundation of performance optimization.

---

## Where I Can Still Improve

*   **Cache Locality**: My current matrix multiplication uses the standard $(i, j, k)$ loop. To reach "Senior" status, I need to explore $(i, k, j)$ patterns or "tiling" to better utilize CPU caches.
*   **Memory Allocation**: Currently, I rely heavily on `Vec` (Heap). For STM32 bare-metal, I need to transition toward **stack-based allocation** or fixed-size arrays to achieve `no_std` compatibility.
*   **Advanced Stability**: While Gram-Schmidt is great, I want to eventually implement **Householder Reflections** for even greater numerical stability in extreme AI scenarios.

---

## Roadmap: The Path to Bare-Metal AI

### Phase 1: Algorithmic Mastery
- [ ] **SVD (Singular Value Decomposition)**: The "final boss" of linear algebra for dimensionality reduction.
- [ ] **LU Decomposition**: For faster solving of square systems.
- [ ] **Automatic Differentiation**: Building the engine for backpropagation.

### Phase 2: Embedded Transition (`no_std`)
- [ ] **`no_std` compatibility**: Stripping away the Rust standard library.
- [ ] **Static Memory Management**: Replacing `Vec` with fixed-size buffers for Nucleo boards.
- [ ] **ARM CMSIS-DSP Integration**: Utilizing the STM32's hardware-specific instructions (SIMD) to accelerate math.

### Phase 3: Real-World Applications
- [ ] **Kalman Filters**: For real-time sensor fusion on my Nucleo board.
- [ ] **On-Device Inference**: Loading and running quantized neural networks.

---
*Developed with the mindset that NumPy is a luxury, but precision is a necessity.*
