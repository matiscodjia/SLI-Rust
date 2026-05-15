# Ferrite

**A no_std machine learning framework for microcontrollers, written in Rust.**

Ferrite brings high-performance linear algebra and machine learning to resource-constrained embedded systems — with zero heap allocation, compile-time memory safety, and hardware-accelerated f32 arithmetic on Cortex-M4/M7.

```toml
[dependencies]
ferrite = { version = "0.1", default-features = false }
```

---

## Why Ferrite

Every existing ML framework assumes a heap, an OS, and megabytes of RAM. On a microcontroller, none of that exists.

Ferrite is built around a different contract: **all memory is allocated at compile time, all dimensions are type-checked at compile time, and no standard library is required**. A matrix multiplication that would silently panic at runtime in other libraries simply won't compile in Ferrite if the dimensions don't match.

The goal is a complete ML stack — from linear algebra primitives to autodiff and on-device training — that runs on a bare-metal STM32F4 with 192KB of RAM and no operating system.

---

## Design Principles

- **`no_std` first** — runs on any Cortex-M target without an OS or allocator
- **Const generics throughout** — matrix and vector dimensions are part of the type, checked at compile time
- **Zero heap** — every buffer lives on the stack; no `Vec`, no `Box`, no allocator
- **f32 hardware acceleration** — designed for Cortex-M4/M7 FPU (single precision)
- **Portable** — the same code runs on STM32F4, STM32H7, nRF, RP2040, and desktop

---

## Current Capabilities

### Linear Algebra Primitives

```rust
use ferrite::{Vector, Matrix};

// Compile-time dimension safety
let v: Vector<3> = Vector::new([1.0, 2.0, 3.0]);
let a: Matrix<3, 3> = Matrix::new();

// Ergonomic owned arithmetic (no & required)
let result = v * 2.0;
let product = a * b; // dimension mismatch = compile error
```

**Vector\<N\>**
- L1, L2, infinity norms
- Dot product, orthogonal projection
- Add, Sub, Mul (owned, zero-copy for Copy types)

**Matrix\<ROWS, COLS\>**
- Row-major stack allocation
- Transpose, scale
- `mat[(i, j)]` indexing via `Index` / `IndexMut`
- Matrix multiplication with compile-time dimension checking

### Algorithms

| Algorithm | Function | Status |
|---|---|---|
| Gram-Schmidt orthogonalization | `gram_schmidt` | ✅ |
| QR decomposition | `qr_decomposition` | ✅ |
| Back-substitution | `solve_upper_triangular` | ✅ |
| Linear system solver (Ax = b) | `solve_linear_system` | ✅ |
| Jacobi SVD (2×2, closed-form) | `svd_2x2` | ✅ |
| Jacobi one-sided SVD (N×N) | `svd` | ✅ |

### SVD Example

```rust
use ferrite::algorithms::svd;

let (u, sigma, v) = svd(&a);
// sigma is sorted: sigma[0] >= sigma[1] >= ... >= sigma[N-1]
// U and V are orthogonal: U^T * U = I, V^T * V = I
// Reconstruction: U * diag(sigma) * V^T = A
```

---

## Target Hardware

| Board | MCU | RAM | FPU | Status |
|---|---|---|---|---|
| Nucleo-F446RE / F447 | Cortex-M4 @ 180MHz | 192KB | Single | Reference target |
| Nucleo-H743ZI | Cortex-M7 @ 480MHz | 1MB | Double | Planned |
| Any Cortex-M4/M7 | — | ≥ 64KB | Single/Double | Supported |

---

## Roadmap

### v0.2 — Autodiff Engine
- Computational graph with static memory layout
- Forward and backward pass for scalar and tensor operations
- No dynamic dispatch, no heap

### v0.3 — Neural Network Primitives
- `Linear<IN, OUT>` layer
- Activation functions: ReLU, sigmoid, tanh
- Loss functions: MSE, cross-entropy

### v0.4 — On-Device Training
- SGD and Adam optimizers
- Full MLP training loop on STM32F4
- Gradient checkpointing for memory-constrained targets

### v0.5 — Applications
- PCA and dimensionality reduction
- Kalman filter
- Learned inverse kinematics for robotic arms

---

## Benchmarks (STM32F447, release build)

| Operation | Time |
|---|---|
| Matrix multiply 4×4 | 3.5 ns |
| SVD 3×3 | 241 ns |
| SVD 4×4 | 539 ns |

*Measured on host (aarch64, Apple Silicon). STM32 figures coming in v0.2.*

---

## License

MIT OR Apache-2.0
