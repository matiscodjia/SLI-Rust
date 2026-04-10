#![no_std] // We are officially an embedded library now.

#[cfg(any(feature = "std", test))]
#[macro_use]
extern crate std; // Use full std for tests and dev on MacOS.

pub mod algorithms;
pub mod matrix;
pub mod vector;

pub use algorithms::{gram_schmidt, qr_decomposition, solve_linear_system};
pub use matrix::Matrix;
/// Re-export main types for simplified usage
pub use vector::Vector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_vector_matrix() {
        let v = Vector::new([1.0, 2.0]);
        assert_eq!(v.dim(), 2);

        let m = Matrix::<2, 2>::new();
        assert_eq!(m.get_rows(), 2);
    }
}
