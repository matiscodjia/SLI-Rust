//! # Linear Algebra Library
//! 
//! A robust linear algebra library implemented in Rust.
//! This library provides structures for vector and matrix manipulation,
//! along with fundamental algorithms such as the Gram-Schmidt orthonormalization process.

pub mod vector;
pub mod matrix;
pub mod algorithms;

/// Re-exporting main types for simplified usage
pub use vector::Vector;
pub use matrix::Matrix;
pub use algorithms::gram_schmidt;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_vector_matrix() {
        // A small integration test to verify communication between modules
        let v = Vector::new(vec![1.0, 2.0]);
        assert_eq!(v.dim(), 2);
        
        let m = Matrix::new(2, 2);
        assert_eq!(m.get_rows(), 2);
    }
}
