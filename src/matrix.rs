use crate::vector::Vector;
use std::cmp::PartialEq;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
/// A structure representing a matrix of floating-point numbers (f32).
/// Data is stored contiguously in memory using Row-Major Format (RMF).
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    rmf: Vec<f32>,
}

impl Matrix {
    /// Creates a new matrix filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - The number of rows.
    /// * `cols` - The number of columns.
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            rmf: vec![0.0; rows * cols],
        }
    }

    /// Returns the number of rows in the matrix.
    pub fn get_rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn get_cols(&self) -> usize {
        self.cols
    }

    /// Retrieves the value at a given position (row, column).
    /// Returns `Some(value)` if indices are valid, otherwise `None`.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.rmf.get(index).copied()
        } else {
            None
        }
    }

    pub fn get_col(&self, col: usize) -> Option<Vector> {
        if col >= self.cols {
            return None;
        }
        let data: Vec<f32> = (0..self.rows)
            .map(|i| self.get(i, col).expect("Index guaranteed to be safe"))
            .collect();
        Some(Vector::new(data))
    }
    pub fn set_col(&mut self, col: usize, vector: &Vector) {
        assert_eq!(
            vector.dim(),
            self.rows,
            "Vector dimension must match matrix rows"
        );
        let vec_data = vector.get_data();
        for i in 0..self.rows {
            self.set(i, col, vec_data[i]);
        }
    }

    /// Modifies the value at a given position.
    ///
    /// # Panics
    /// Panics if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.rmf[index] = val;
        } else {
            panic!(
                "Invalid index: ({}, {}) for a {}x{} matrix",
                row, col, self.rows, self.cols
            );
        }
    }

    /// Consumes the matrix and returns the raw data as a Vec.
    pub fn into_vec(self) -> Vec<f32> {
        self.rmf
    }

    /// Performs matrix multiplication: self * other.
    /// Returns a new matrix or an error if dimensions are incompatible.
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Incompatible dimensions for multiplication: {}x{} * {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        let mut result = Matrix::new(self.rows, other.cols);
        result.matmul_accumulate(self, other, false, false);
        Ok(result)
    }

    /// Applies the ReLU activation function to each element.
    pub fn relu(&self) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        result.rmf = self
            .rmf
            .iter()
            .map(|&x| if x < 0.0 { 0.0 } else { x })
            .collect();
        result
    }

    /// Adds two matrices.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Incompatible dimensions for addition".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j).unwrap() + other.get(i, j).unwrap());
            }
        }
        Ok(result)
    }

    /// Subtracts two matrices.
    pub fn sub(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Incompatible dimensions for subtraction".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j).unwrap() - other.get(i, j).unwrap());
            }
        }
        Ok(result)
    }

    /// Adds a matrix to the current instance (in-place).
    pub fn assign_add(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Incompatible dimensions for assign_add");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j).unwrap() + other.get(i, j).unwrap();
                self.set(i, j, val);
            }
        }
    }

    /// Subtracts a matrix from the current instance (in-place).
    pub fn assign_sub(&mut self, other: &Matrix) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Incompatible dimensions for assign_sub");
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                let val = self.get(i, j).unwrap() - other.get(i, j).unwrap();
                self.set(i, j, val);
            }
        }
    }

    /// Is upper triangular
    pub fn is_triangular_upper(&self) -> bool {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if i > j {
                    if self.get(i, j).expect("Valid indexes") > f32::EPSILON {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j).unwrap());
            }
        }
        result
    }

    /// Multiplies all elements of the matrix by a scalar.
    pub fn scale(&self, coef: f32) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        result.rmf = self.rmf.iter().map(|&x| x * coef).collect();
        result
    }

    /// Retrieves a value considering a virtual transposition.
    pub fn get_with_transpose(&self, row: usize, col: usize, is_transposed: bool) -> Option<f32> {
        let (r, c) = if is_transposed {
            (col, row)
        } else {
            (row, col)
        };
        self.get(r, c)
    }

    /// Accumulates the matrix product A * B into the current instance.
    /// Supports virtual transposition of operands.
    pub fn matmul_accumulate(
        &mut self,
        a: &Matrix,
        b: &Matrix,
        a_transpose: bool,
        b_transpose: bool,
    ) {
        let (a_rows, a_cols) = if a_transpose {
            (a.cols, a.rows)
        } else {
            (a.rows, a.cols)
        };
        let (b_rows, b_cols) = if b_transpose {
            (b.cols, b.rows)
        } else {
            (b.rows, b.cols)
        };

        assert_eq!(a_cols, b_rows, "Incompatible dimensions for matmul");
        assert_eq!(self.rows, a_rows, "Destination row count mismatch");
        assert_eq!(self.cols, b_cols, "Destination column count mismatch");

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    let v1 = a.get_with_transpose(i, k, a_transpose).unwrap();
                    let v2 = b.get_with_transpose(k, j, b_transpose).unwrap();
                    sum += v1 * v2;
                }
                let current = self.get(i, j).unwrap();
                self.set(i, j, current + sum);
            }
        }
    }
}

/// Creates a matrix filled with 1.0.
pub fn ones_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix {
        rows,
        cols,
        rmf: vec![1.0; rows * cols],
    }
}

/// Creates a new Matrix from a slice of Vectors (each vector is a column)
pub fn from_cols(cols: &[Vector]) -> Matrix {
    if cols.is_empty() {
        return Matrix::new(0, 0);
    }
    let n_rows = cols[0].dim();

    assert!(
        cols.iter().all(|vector| vector.dim() == n_rows),
        "All column vectors must have the same dimension ({})",
        n_rows
    );
    let n_cols = cols.len();
    let mut mat = Matrix::new(n_rows, n_cols);
    for (j, col_vec) in cols.iter().enumerate() {
        mat.set_col(j, col_vec)
    }
    mat
}

/// Creates a matrix filled with 0.0.
pub fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    Matrix::new(rows, cols)
}

// Operator trait implementations
impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        self.add(rhs).expect("Error during matrix addition")
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        self.sub(rhs).expect("Error during matrix subtraction")
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.multiply(rhs)
            .expect("Error during matrix multiplication")
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        self.assign_add(rhs)
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        self.assign_sub(rhs)
    }
}

impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        self.scale(-1.0)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        let epsilon = 1e-5;
        self.rmf
            .iter()
            .zip(other.rmf.iter())
            .all(|(&a, &b)| (a - b).abs() < epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.get_rows(), 2);
        assert_eq!(m.get_cols(), 3);
        assert_eq!(m.get(0, 0), Some(0.0));
    }

    #[test]
    fn test_matrix_set_get() {
        let mut m = Matrix::new(2, 2);
        m.set(0, 1, 5.0);
        assert_eq!(m.get(0, 1), Some(5.0));
        assert_eq!(m.get(1, 1), Some(0.0));
    }

    #[test]
    fn test_matrix_add() {
        let mut m1 = ones_matrix(2, 2);
        let m2 = ones_matrix(2, 2);
        let res = &m1 + &m2;
        assert_eq!(res.get(0, 0), Some(2.0));

        m1 += &m2;
        assert_eq!(m1.get(1, 1), Some(2.0));
    }

    #[test]
    fn test_matrix_multiply() {
        // [1 2] * [5 6] = [1*5 + 2*7  1*6 + 2*8] = [19 22]
        // [3 4]   [7 8]   [3*5 + 4*7  3*6 + 4*8]   [43 50]
        let mut m1 = Matrix::new(2, 2);
        m1.set(0, 0, 1.0);
        m1.set(0, 1, 2.0);
        m1.set(1, 0, 3.0);
        m1.set(1, 1, 4.0);

        let mut m2 = Matrix::new(2, 2);
        m2.set(0, 0, 5.0);
        m2.set(0, 1, 6.0);
        m2.set(1, 0, 7.0);
        m2.set(1, 1, 8.0);

        let res = &m1 * &m2;
        assert_eq!(res.get(0, 0), Some(19.0));
        assert_eq!(res.get(0, 1), Some(22.0));
        assert_eq!(res.get(1, 0), Some(43.0));
        assert_eq!(res.get(1, 1), Some(50.0));
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix::new(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(0, 2, 3.0);
        let t = m.transpose();
        assert_eq!(t.get_rows(), 3);
        assert_eq!(t.get_cols(), 2);
        assert_eq!(t.get(1, 0), Some(2.0));
    }

    #[test]
    fn test_relu() {
        let mut m = Matrix::new(1, 2);
        m.set(0, 0, -5.0);
        m.set(0, 1, 3.0);
        let res = m.relu();
        assert_eq!(res.get(0, 0), Some(0.0));
        assert_eq!(res.get(0, 1), Some(3.0));
    }
}
