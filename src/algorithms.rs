use crate::matrix::Matrix;
use crate::vector::Vector;
use libm::fabsf;

/// Transforms a set of N vectors of size M into an orthonormal basis.
///
/// In this static version, we return an array of exactly N vectors.
/// If a vector is linearly dependent, it will result in a null vector.
pub fn gram_schmidt<const M: usize, const N: usize>(base: &[Vector<M>; N]) -> [Vector<M>; N] {
    let mut orthogonal_basis = [Vector::<M>::new([0.0; M]); N];

    for i in 0..N {
        let v = &base[i];
        let mut q = *v; // Copy the original vector

        // Subtract projections on all previously computed basis vectors
        for j in 0..i {
            let u = &orthogonal_basis[j];
            let proj = v.orthogonal_projection(u);
            q = &q - &proj;
        }

        let norm = q.l2_norm();
        if norm > 1e-6 {
            q = &q * (1.0 / norm);
            orthogonal_basis[i] = q;
        } else {
            // Keep it as a null vector if it's not linearly independent
            orthogonal_basis[i] = Vector::<M>::new([0.0; M]);
        }
    }
    orthogonal_basis
}

/// QR decomposition of a M x N matrix.
///
/// Returns (Q, R) where:
/// - Q is an M x N orthogonal matrix.
/// - R is an N x N upper triangular matrix.
pub fn qr_decomposition<const M: usize, const N: usize>(
    mat: &Matrix<M, N>,
) -> (Matrix<M, N>, Matrix<N, N>) {
    // 1. Extract columns into a fixed-size array
    let mut cols = [Vector::<M>::new([0.0; M]); N];
    for j in 0..N {
        cols[j] = mat.get_col(j).unwrap();
    }

    // 2. Perform Gram-Schmidt
    let ortho_cols = gram_schmidt::<M, N>(&cols);

    // 3. Create Q from the orthonormal columns
    let q = Matrix::<M, N>::from_cols(ortho_cols);

    // 4. Calculate R = Q^T * mat
    // Resulting R is N x N
    let mut r = Matrix::<N, N>::new();
    r.matmul_accumulate(&q.transpose(), mat);

    (q, r)
}

/// Solves an upper triangular system Rx = b using back-substitution.
///
/// R is an N x N matrix, b is a Vector of size N.
pub fn solve_upper_triangular<const N: usize>(
    r: &Matrix<N, N>,
    b: &Vector<N>,
) -> Option<Vector<N>> {
    let mut x_data = [0.0; N];
    let b_data = b.get_data();

    for i in (0..N).rev() {
        let diag = r.get(i, i).unwrap_or(0.0);

        if fabsf(diag) < 1e-10 {
            return None; // Singular matrix
        }

        let mut sum = 0.0;
        for j in (i + 1)..N {
            sum += r.get(i, j).unwrap_or(0.0) * x_data[j];
        }

        x_data[i] = (b_data[i] - sum) / diag;
    }

    Some(Vector::new(x_data))
}

/// Solves a linear system Ax = b using QR decomposition.
///
/// A is M x N, b is size M, result x is size N.
pub fn solve_linear_system<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
    b: &Vector<M>,
) -> Option<Vector<N>> {
    let (q, r) = qr_decomposition(a);

    // Compute c = Q^T * b (Vector of size N)
    let mut c_data = [0.0; N];
    for i in 0..N {
        let q_col = q.get_col(i).unwrap();
        c_data[i] = q_col.dot(b);
    }
    let c = Vector::new(c_data);

    // Solve Rx = c
    solve_upper_triangular(&r, &c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_schmidt_2d() {
        let v1 = Vector::new([1.0, 1.0]);
        let v2 = Vector::new([0.0, 1.0]);
        let ortho = gram_schmidt::<2, 2>(&[v1, v2]);

        // Check orthogonality
        assert!(ortho[0].dot(&ortho[1]).abs() < 1e-6);
        // Check normality
        assert!((ortho[0].l2_norm() - 1.0).abs() < 1e-6);
        assert!((ortho[1].l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_qr_decomposition_simple() {
        let mut a = Matrix::<2, 2>::new();
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 0.0);
        a.set(1, 1, 1.0);

        let (q, r) = qr_decomposition(&a);

        // A = QR
        assert_eq!(&q * &r, a);
        // Q is orthogonal
        let i_check = &q.transpose() * &q;
        assert_eq!(i_check, crate::matrix::identity::<2>());
    }

    #[test]
    fn test_solve_upper_triangular() {
        let mut r = Matrix::<2, 2>::new();
        r.set(0, 0, 2.0);
        r.set(0, 1, 1.0);
        r.set(1, 1, 1.0);
        let b = Vector::new([5.0, 1.0]);

        let x = solve_upper_triangular(&r, &b).unwrap();
        // 2x + 1y = 5, 1y = 1 => y=1, 2x=4 => x=2
        assert_eq!(x, Vector::new([2.0, 1.0]));
    }

    #[test]
    fn test_solve_linear_system_2d() {
        let mut a = Matrix::<2, 2>::new();
        a.set(0, 0, 1.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, -1.0);
        let b = Vector::new([3.0, 1.0]);

        let x = solve_linear_system(&a, &b).unwrap();
        assert_eq!(x, Vector::new([2.0, 1.0]));
    }

    #[test]
    fn test_singular_system() {
        let a = Matrix::<2, 2>::new(); // All zeros
        let b = Vector::new([1.0, 1.0]);
        assert!(solve_linear_system(&a, &b).is_none());
    }

    #[test]
    fn test_gram_schmidt_dependent() {
        let v1 = Vector::new([1.0, 0.0]);
        let v2 = Vector::new([2.0, 0.0]); // Linearly dependent
        let ortho = gram_schmidt::<2, 2>(&[v1, v2]);
        assert_eq!(ortho[1], Vector::new([0.0, 0.0]));
    }

    #[test]
    fn test_qr_3x2_matrix() {
        let mut a = Matrix::<3, 2>::new();
        a.set(0, 0, 12.0);
        a.set(0, 1, -51.0);
        a.set(1, 0, 6.0);
        a.set(1, 1, 167.0);
        a.set(2, 0, -4.0);
        a.set(2, 1, 24.0);

        let (q, r) = qr_decomposition(&a);
        assert_eq!(&q * &r, a);
    }

    #[test]
    fn test_back_substitution_3d() {
        let mut r = Matrix::<3, 3>::new();
        r.set(0, 0, 1.0);
        r.set(0, 1, 2.0);
        r.set(0, 2, 3.0);
        r.set(1, 1, 1.0);
        r.set(1, 2, 2.0);
        r.set(2, 2, 1.0);
        let b = Vector::new([6.0, 3.0, 1.0]);
        let x = solve_upper_triangular(&r, &b).unwrap();
        assert_eq!(x, Vector::new([1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_identity_solver() {
        let a = crate::matrix::identity::<3>();
        let b = Vector::new([1.0, 2.0, 3.0]);
        let x = solve_linear_system(&a, &b).unwrap();
        assert_eq!(x, b);
    }

    #[test]
    fn test_orthogonal_projection_consistency() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        let u = Vector::new([1.0, 0.0, 0.0]);
        let proj = v.orthogonal_projection(&u);
        assert_eq!(proj, Vector::new([1.0, 0.0, 0.0]));
    }
}
