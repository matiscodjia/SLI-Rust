use std::ops::{Add, Mul, Sub};

/// Represents a mathematical vector in a real vector space.
/// Operations are implemented for f32 floating-point numbers.
#[derive(Debug, Clone)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    /// Creates a new vector from a Vec<f32>.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Returns the dimension (size) of the vector.
    pub fn dim(&self) -> usize {
        self.data.len()
    }
    /// Calculates the infinity norm (max of absolute values).
    pub fn inf_norm(&self) -> f32 {
        self.data.iter().map(|x| x.abs()).fold(0.0, |a, b| a.max(b))
    }

    /// Calculates the L2 norm (Euclidean norm).
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Calculates the L1 norm (sum of absolute values).
    pub fn l1_norm(&self) -> f32 {
        self.data.iter().map(|x| x.abs()).sum()
    }

    /// Calculates the dot product between two vectors.
    ///
    /// # Panics
    /// Panics if the dimensions do not match.
    pub fn dot(&self, other: &Vector) -> f32 {
        if self.dim() != other.dim() {
            panic!("Dimensions mismatch: {} != {}", self.dim(), other.dim());
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Calculates the orthogonal projection of `self` onto `other`.
    ///
    /// # Panics
    /// Panics if `other` is a null vector.
    pub fn orthogonal_projection(&self, other: &Vector) -> Vector {
        let scale_factor = other.dot(other);
        if scale_factor.abs() < 1e-8 {
            panic!(
                "Attempted projection onto a null vector (norm squared: {})",
                scale_factor
            );
        }
        let ratio = other.dot(self) / scale_factor;
        other * ratio
    }

    /// Provides access to the internal data (read-only).
    pub fn get_data(&self) -> &[f32] {
        &self.data
    }
}

// Operator trait implementations
impl Mul<f32> for &Vector {
    type Output = Vector;
    fn mul(self, rhs: f32) -> Self::Output {
        let data = self.data.iter().map(|x| x * rhs).collect();
        Vector::new(data)
    }
}

impl Sub<&Vector> for &Vector {
    type Output = Vector;
    fn sub(self, rhs: &Vector) -> Self::Output {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Vector::new(data)
    }
}

impl Add<&Vector> for &Vector {
    type Output = Vector;
    fn add(self, rhs: &Vector) -> Self::Output {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Vector::new(data)
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        if self.dim() != other.dim() {
            return false;
        }
        let epsilon = 1e-5;
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() < epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_norms() {
        let v = Vector::new(vec![3.0, -4.0]);
        assert_eq!(v.l1_norm(), 7.0);
        assert_eq!(v.l2_norm(), 5.0);
        assert_eq!(v.inf_norm(), 4.0);
    }

    #[test]
    fn test_vector_ops() {
        let v1 = Vector::new(vec![1.0, 2.0]);
        let v2 = Vector::new(vec![3.0, 4.0]);

        let v_add = &v1 + &v2;
        assert_eq!(v_add, Vector::new(vec![4.0, 6.0]));

        let v_sub = &v2 - &v1;
        assert_eq!(v_sub, Vector::new(vec![2.0, 2.0]));

        let v_mul = &v1 * 2.0;
        assert_eq!(v_mul, Vector::new(vec![2.0, 4.0]));
    }

    #[test]
    fn test_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(v1.dot(&v2), 4.0 + 10.0 + 18.0);
    }

    #[test]
    fn test_projection() {
        let v = Vector::new(vec![1.0, 1.0]);
        let d = Vector::new(vec![1.0, 0.0]);
        let proj = v.orthogonal_projection(&d);
        assert_eq!(proj, Vector::new(vec![1.0, 0.0]));
    }
}
