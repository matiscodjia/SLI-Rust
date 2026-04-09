use crate::matrix::{Matrix, from_cols};
use crate::vector::Vector;

/// Transforms a set of vectors into an orthonormal basis using the Gram-Schmidt process.
///
/// The algorithm stabilizes the vectors by subtracting their projections onto
/// previously computed vectors, then normalizes each result.
///
/// # Arguments
/// * `base` - A slice of vectors to orthonormalize.
///
/// # Returns
/// A `Vec<Vector>` containing the resulting orthonormal basis.
pub fn gram_schmidt(base: &[Vector]) -> Vec<Vector> {
    let mut orthogonal_basis: Vec<Vector> = Vec::new();

    for v in base {
        let mut q = v.clone();
        for u in &orthogonal_basis {
            // Subtract the projection of the original vector 'v' onto the basis vector 'u'
            let proj = v.orthogonal_projection(u);
            q = &q - &proj;
        }

        let norm = q.l2_norm();
        // Tolerance threshold to avoid division by zero on near-null vectors
        if norm > 1e-6 {
            q = &q * (1.0 / norm);
            orthogonal_basis.push(q);
        }
    }
    orthogonal_basis
}

/// QR decomposition
pub fn qr_decomposition(mat: &Matrix) -> (Matrix, Matrix) {
    let mut cols: Vec<Vector> = Vec::new();
    for j in 0..mat.get_cols() {
        cols.push(mat.get_col(j).expect("Empty matrix"));
    }
    let cols = gram_schmidt(&cols);
    let q = from_cols(&cols);
    let mut r = Matrix::new(q.get_cols(), mat.get_cols());
    r.matmul_accumulate(&q, mat, true, false);
    (q, r)
}

/// Solve triangular upper
//pub fn solve_triangular_upper(r: &Matrix, b: &Vector) -> Vector {}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let v1 = Vector::new(vec![1.0, 1.0, 0.0]);
        let v2 = Vector::new(vec![1.0, 0.0, 1.0]);
        let v3 = Vector::new(vec![0.0, 1.0, 1.0]);
        let base = vec![v1, v2, v3];

        let ortho = gram_schmidt(&base);

        assert_eq!(ortho.len(), 3);

        // Verify orthonormality
        let epsilon = 1e-5;
        for i in 0..ortho.len() {
            // Norm = 1
            assert!((ortho[i].l2_norm() - 1.0).abs() < epsilon);
            for j in (i + 1)..ortho.len() {
                // Dot product = 0
                assert!(ortho[i].dot(&ortho[j]).abs() < epsilon);
            }
        }
    }

    #[test]
    fn test_gram_schmidt_empty() {
        let base: Vec<Vector> = Vec::new();
        let result = gram_schmidt(&base);
        assert!(result.is_empty());
    }
}

#[test]
fn test_qr_decomposition() {
    let mut a = Matrix::new(3, 2);
    a.set(0, 0, 12.0);
    a.set(0, 1, -51.0);
    a.set(1, 0, 6.0);
    a.set(1, 1, 167.0);
    a.set(2, 0, -4.0);
    a.set(2, 1, 24.0);

    let (q, r) = qr_decomposition(&a);

    // 1. Check if Q * R == A
    let reconstructed = &q * &r;
    assert_eq!(reconstructed, a);

    // 2. Check if Q is orthogonal (Q^T * Q == I)
    let mut identity_check = Matrix::new(q.get_cols(), q.get_cols());
    identity_check.matmul_accumulate(&q, &q, true, false);

    // Create an actual identity matrix for comparison
    let mut expected_i = Matrix::new(q.get_cols(), q.get_cols());
    for i in 0..q.get_cols() {
        expected_i.set(i, i, 1.0);
    }

    assert_eq!(identity_check, expected_i);
}
