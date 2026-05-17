#![no_std] // We are officially an embedded library now.

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod algorithms;
pub mod autodiff;
pub mod matrix;
pub mod vector;
pub use algorithms::{gram_schmidt, qr_decomposition, solve_linear_system};
pub use matrix::Matrix;
/// Re-export main types for simplified usage
pub use vector::Vector;

#[cfg(test)]
mod tests {
    use crate::autodiff::{activations::{ReLU, Tanh}, linear::Linear};
    use super::*;

    #[test]
    fn test_integration_vector_matrix() {
        let v = Vector::new([1.0, 2.0]);
        assert_eq!(v.dim(), 2);

        let m = Matrix::<2, 2>::new();
        assert_eq!(m.get_rows(), 2);
    }

    #[test]
    fn test_mlp_training_step() {
        use crate::autodiff::loss::mse;
        use crate::autodiff::module::Module;
        use crate::autodiff::optims::sgd::sgd;

        let mut network = (Linear::<2, 4>::from_seed(42), ReLU::<4> {}, Linear::<4, 1>::from_seed(137));

        let input = Vector::new([1.0, 0.5]);
        let target = Vector::new([1.0]);

        let (output, ctx) = network.forward(input);
        let (loss, loss_grad) = mse(output, target);

        let (_, grads) = network.backward(loss_grad, &ctx);
        sgd(&mut network, &grads, 0.01);

        let (output2, _) = network.forward(input);
        let (loss2, _) = mse(output2, target);

        assert!(loss2 < loss, "loss should decrease after one SGD step: {loss2} >= {loss}");
    }

    // cargo test test_convergence -- --nocapture
    #[test]
    fn test_convergence() {
        use crate::autodiff::loss::mse;
        use crate::autodiff::module::Module;
        use crate::autodiff::optims::sgd::sgd;

        // Fonction cible : f(x1, x2) = x1 * x2
        // Requiert de la non-linéarité — un réseau linéaire ne peut pas l'apprendre.
        let dataset: [(Vector<2>, Vector<1>); 4] = [
            (Vector::new([ 1.0,  1.0]), Vector::new([ 1.0])),
            (Vector::new([ 1.0, -1.0]), Vector::new([-1.0])),
            (Vector::new([-1.0,  1.0]), Vector::new([-1.0])),
            (Vector::new([-1.0, -1.0]), Vector::new([ 1.0])),
        ];

        let mut network = (
            Linear::<2, 8>::from_seed(42),
            Tanh::<8> {},
            Linear::<8, 1>::from_seed(137),
        );

        let lr = 0.05;
        let epochs = 2000;

        for epoch in 0..=epochs {
            let mut total_loss = 0.0;
            for &(input, target) in &dataset {
                let (output, ctx) = network.forward(input);
                let (loss, loss_grad) = mse(output, target);
                total_loss += loss;
                let (_, grads) = network.backward(loss_grad, &ctx);
                sgd(&mut network, &grads, lr);
            }
            if epoch % 200 == 0 {
                println!("epoch {:4} | loss {:.6}", epoch, total_loss / dataset.len() as f32);
            }
        }

        let mut final_loss = 0.0;
        for &(input, target) in &dataset {
            let (output, _) = network.forward(input);
            let (loss, _) = mse(output, target);
            final_loss += loss;
        }
        final_loss /= dataset.len() as f32;

        println!("loss finale : {final_loss:.6}");
        assert!(final_loss < 0.05, "réseau n'a pas convergé : loss finale = {final_loss:.6}");
    }

    // cargo test bench_autodiff -- --nocapture
    #[test]
    fn bench_autodiff() {
        use crate::autodiff::loss::mse;
        use crate::autodiff::module::Module;
        use crate::autodiff::optims::sgd::sgd;

        let input = Vector::new([1.0, 0.5]);
        let target_2 = Vector::new([1.0]);
        let _target_4 = Vector::new([1.0, 0.0, 0.5, -1.0]);

        // Réseau small : 2→4→1
        let mut net_s = (Linear::<2, 4>::from_seed(42), Tanh::<4> {}, Linear::<4, 1>::from_seed(99));
        // Réseau medium : 2→8→4→1
        let mut net_m = (Linear::<2, 8>::from_seed(42), Tanh::<8> {}, Linear::<8, 4>::from_seed(99), Tanh::<4> {}, Linear::<4, 1>::from_seed(7));

        let n = 10_000u32;

        // --- Forward seul ---
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = net_s.forward(input);
        }
        println!("forward  2→4→1   x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = net_m.forward(input);
        }
        println!("forward  2→8→4→1 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        // --- Forward + Backward ---
        let t = std::time::Instant::now();
        for _ in 0..n {
            let (output, ctx) = net_s.forward(input);
            let (_, grad) = mse(output, target_2);
            let _ = net_s.backward(grad, &ctx);
        }
        println!("fwd+bwd  2→4→1   x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n {
            let (output, ctx) = net_m.forward(input);
            let (_, grad) = mse(output, target_2);
            let _ = net_m.backward(grad, &ctx);
        }
        println!("fwd+bwd  2→8→4→1 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        // --- Step complet : forward + backward + update ---
        let t = std::time::Instant::now();
        for _ in 0..n {
            let (output, ctx) = net_s.forward(input);
            let (_, grad) = mse(output, target_2);
            let (_, grads) = net_s.backward(grad, &ctx);
            sgd(&mut net_s, &grads, 0.01);
        }
        println!("step     2→4→1   x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);

        let t = std::time::Instant::now();
        for _ in 0..n {
            let (output, ctx) = net_m.forward(input);
            let (_, grad) = mse(output, target_2);
            let (_, grads) = net_m.backward(grad, &ctx);
            sgd(&mut net_m, &grads, 0.01);
        }
        println!("step     2→8→4→1 x{n}: {:?} ({:.1}ns/iter)", t.elapsed(), t.elapsed().as_nanos() as f64 / n as f64);
    }
}
