use crate::vector::Vector;
use core::cmp::PartialEq;
use core::ops::{Add, Mul, Sub};
use libm::fabsf;

/// A Static Matrix of ROWS x COLS, stored entirely on the stack.
/// Uses a 2D array to remain 100% static and compatible with stable Rust.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    data: [[f32; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Creates a new matrix filled with zeros.
    pub const fn new() -> Self {
        Self {
            data: [[0.0; COLS]; ROWS],
        }
    }

    /// Returns the number of rows.
    pub const fn get_rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns.
    pub const fn get_cols(&self) -> usize {
        COLS
    }

    /// Retrieves a value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < ROWS && col < COLS {
            Some(self.data[row][col])
        } else {
            None
        }
    }

    /// Sets a value at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        if row < ROWS && col < COLS {
            self.data[row][col] = val;
        } else {
            panic!("Matrix index out of bounds: ({}, {})", row, col);
        }
    }

    /// Extracts a column as a Static Vector of size ROWS.
    pub fn get_col(&self, col: usize) -> Option<Vector<ROWS>> {
        if col >= COLS {
            return None;
        }
        let mut col_data = [0.0; ROWS];
        for i in 0..ROWS {
            col_data[i] = self.data[i][col];
        }
        Some(Vector::new(col_data))
    }

    /// Injects a Static Vector into a matrix column.
    pub fn set_col(&mut self, col: usize, vec: &Vector<ROWS>) {
        if col >= COLS {
            panic!("Column index out of bounds");
        }
        let vec_data = vec.get_data();
        for i in 0..ROWS {
            self.data[i][col] = vec_data[i];
        }
    }

    /// Creates a Matrix from an array of column vectors.
    pub fn from_cols(cols: [Vector<ROWS>; COLS]) -> Self {
        let mut mat = Self::new();
        for j in 0..COLS {
            mat.set_col(j, &cols[j]);
        }
        mat
    }

    /// Performs the matrix product: self (M x N) * other (N x P) -> Result (M x P).
    /// Dimensions are checked at compile-time!
    pub fn multiply<const OTHER_COLS: usize>(
        &self,
        other: &Matrix<COLS, OTHER_COLS>,
    ) -> Matrix<ROWS, OTHER_COLS> {
        let mut result = Matrix::<ROWS, OTHER_COLS>::new();
        result.matmul_accumulate(self, other);
        result
    }

    /// Accumulates the product of A * B into the current matrix (self).
    /// self = self + (A * B)
    /// All dimensions are checked at compile-time.
    pub fn matmul_accumulate<const K: usize>(&mut self, a: &Matrix<ROWS, K>, b: &Matrix<K, COLS>) {
        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum = 0.0;
                for k in 0..K {
                    sum += a.data[i][k] * b.data[k][j];
                }
                self.data[i][j] += sum;
            }
        }
    }

    /// Returns the transpose as a new static matrix.
    pub fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut result = Matrix::<COLS, ROWS>::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.set(j, i, self.data[i][j]);
            }
        }
        result
    }

    /// Scales the matrix by a coefficient.
    pub fn scale(&self, coef: f32) -> Self {
        let mut result = Self::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.set(i, j, self.data[i][j] * coef);
            }
        }
        result
    }
}
///Create identity matrix
pub fn identity<const SIZE: usize>() -> Matrix<SIZE, SIZE> {
    let mut result = Matrix::<SIZE, SIZE>::new();
    for i in 0..SIZE {
        result.set(i, i, 1.0)
    }
    result
}

// Operators for Static Matrices
impl<const ROWS: usize, const COLS: usize> Add<&Matrix<ROWS, COLS>> for &Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn add(self, rhs: &Matrix<ROWS, COLS>) -> Self::Output {
        let mut res = Matrix::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                res.set(i, j, self.data[i][j] + rhs.data[i][j]);
            }
        }
        res
    }
}

impl<const ROWS: usize, const COLS: usize> Sub<&Matrix<ROWS, COLS>> for &Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn sub(self, rhs: &Matrix<ROWS, COLS>) -> Self::Output {
        let mut res = Matrix::new();
        for i in 0..ROWS {
            for j in 0..COLS {
                res.set(i, j, self.data[i][j] - rhs.data[i][j]);
            }
        }
        res
    }
}

// Global Mul operator for Matrix * Matrix
impl<const M: usize, const N: usize, const P: usize> Mul<&Matrix<N, P>> for &Matrix<M, N> {
    type Output = Matrix<M, P>;
    fn mul(self, rhs: &Matrix<N, P>) -> Self::Output {
        self.multiply(rhs)
    }
}

impl<const ROWS: usize, const COLS: usize> PartialEq for Matrix<ROWS, COLS> {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-5;
        for i in 0..ROWS {
            for j in 0..COLS {
                if fabsf(self.data[i][j] - other.data[i][j]) >= epsilon {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_and_size() {
        let m = Matrix::<2, 3>::new();
        assert_eq!(m.get_rows(), 2);
        assert_eq!(m.get_cols(), 3);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m = Matrix::<2, 2>::new();
        m.set(0, 1, 42.0);
        assert_eq!(m.get(0, 1), Some(42.0));
        assert_eq!(m.get(1, 1), Some(0.0));
    }

    #[test]
    fn test_matrix_out_of_bounds() {
        let m = Matrix::<2, 2>::new();
        assert_eq!(m.get(2, 0), None);
    }

    #[test]
    fn test_matrix_addition() {
        let mut m1 = Matrix::<2, 2>::new();
        m1.set(0, 0, 1.0);
        let mut m2 = Matrix::<2, 2>::new();
        m2.set(0, 0, 2.0);
        assert_eq!(&m1 + &m2, {
            let mut res = Matrix::<2, 2>::new();
            res.set(0, 0, 3.0);
            res
        });
    }

    #[test]
    fn test_matrix_multiplication() {
        let mut m1 = Matrix::<2, 2>::new();
        m1.set(0, 0, 1.0);
        m1.set(0, 1, 2.0);
        m1.set(1, 0, 3.0);
        m1.set(1, 1, 4.0);

        let mut m2 = Matrix::<2, 1>::new();
        m2.set(0, 0, 5.0);
        m2.set(1, 0, 6.0);

        let res = &m1 * &m2;
        assert_eq!(res.get(0, 0), Some(17.0)); // 1*5 + 2*6
        assert_eq!(res.get(1, 0), Some(39.0)); // 3*5 + 4*6
    }

    #[test]
    fn test_matmul_accumulate() {
        let mut res = Matrix::<1, 1>::new();
        res.set(0, 0, 10.0);
        let m1 = identity::<1>();
        let m2 = identity::<1>();
        res.matmul_accumulate(&m1, &m2);
        assert_eq!(res.get(0, 0), Some(11.0));
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::<1, 2>::new();
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        let t = m.transpose();
        assert_eq!(t.get_rows(), 2);
        assert_eq!(t.get_cols(), 1);
        assert_eq!(t.get(1, 0), Some(2.0));
    }

    #[test]
    fn test_matrix_col_extraction() {
        let mut m = Matrix::<2, 2>::new();
        m.set(0, 1, 5.0);
        m.set(1, 1, 10.0);
        let col = m.get_col(1).unwrap();
        assert_eq!(col, Vector::new([5.0, 10.0]));
    }

    #[test]
    fn test_matrix_from_cols() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        let m = Matrix::from_cols([v1, v2]);
        assert_eq!(m.get(1, 0), Some(2.0));
        assert_eq!(m.get(1, 1), Some(4.0));
    }

    #[test]
    fn test_matrix_identity() {
        let id = identity::<3>();
        assert_eq!(id.get(0, 0), Some(1.0));
        assert_eq!(id.get(0, 1), Some(0.0));
        assert_eq!(id.get(1, 1), Some(1.0));
        assert_eq!(id.get(2, 2), Some(1.0));
    }
}
