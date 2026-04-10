use core::ops::{Add, Mul, Sub};
use libm::{fabsf, sqrtf};

/// A Static Vector of size N, stored entirely on the stack.
/// No heap allocation, 100% no_std compatible.
#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize> {
    data: [f32; N],
}

impl<const N: usize> Vector<N> {
    /// Creates a new vector from a static array.
    pub const fn new(data: [f32; N]) -> Self {
        Self { data }
    }

    /// Returns the dimension of the vector.
    pub const fn dim(&self) -> usize {
        N
    }

    /// Calculates the infinity norm (max of absolute values).
    pub fn inf_norm(&self) -> f32 {
        let mut max = 0.0;
        for &val in &self.data {
            let abs_val = fabsf(val);
            if abs_val > max {
                max = abs_val;
            }
        }
        max
    }

    /// Calculates the L2 norm (Euclidean norm).
    pub fn l2_norm(&self) -> f32 {
        sqrtf(self.dot(self))
    }

    /// Calculates the L1 norm (sum of absolute values).
    pub fn l1_norm(&self) -> f32 {
        let mut sum = 0.0;
        for &val in &self.data {
            sum += fabsf(val);
        }
        sum
    }

    /// Calculates the dot product between two vectors of the same size N.
    /// The size is checked at compile-time.
    pub fn dot(&self, other: &Vector<N>) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// Calculates the orthogonal projection of `self` onto `other`.
    pub fn orthogonal_projection(&self, other: &Vector<N>) -> Vector<N> {
        let scale_factor = other.dot(other);
        if fabsf(scale_factor) < 1e-8 {
            // Returns a null vector if other is nearly zero
            return Vector::new([0.0; N]);
        }
        let ratio = self.dot(other) / scale_factor;
        other * ratio
    }

    /// Access the raw data array.
    pub const fn get_data(&self) -> &[f32; N] {
        &self.data
    }
}

// Operator implementations for Static Vectors
impl<const N: usize> Mul<f32> for &Vector<N> {
    type Output = Vector<N>;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] * rhs;
        }
        Vector::new(data)
    }
}

impl<const N: usize> Sub<&Vector<N>> for &Vector<N> {
    type Output = Vector<N>;
    fn sub(self, rhs: &Vector<N>) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] - rhs.data[i];
        }
        Vector::new(data)
    }
}

impl<const N: usize> Add<&Vector<N>> for &Vector<N> {
    type Output = Vector<N>;
    fn add(self, rhs: &Vector<N>) -> Self::Output {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = self.data[i] + rhs.data[i];
        }
        Vector::new(data)
    }
}

impl<const N: usize> PartialEq for Vector<N> {
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-6;
        for i in 0..N {
            if fabsf(self.data[i] - other.data[i]) >= epsilon {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation_and_dim() {
        let v = Vector::new([1.0, 2.0, 3.0]);
        assert_eq!(v.dim(), 3);
    }

    #[test]
    fn test_vector_l1_norm() {
        let v = Vector::new([1.0, -2.0, 3.0]);
        assert_eq!(v.l1_norm(), 6.0);
    }

    #[test]
    fn test_vector_l2_norm() {
        let v = Vector::new([3.0, 4.0]);
        assert_eq!(v.l2_norm(), 5.0);
    }

    #[test]
    fn test_vector_inf_norm() {
        let v = Vector::new([-10.0, 2.0, 5.0]);
        assert_eq!(v.inf_norm(), 10.0);
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new([1.0, 2.0]);
        let v2 = Vector::new([3.0, 4.0]);
        assert_eq!(&v1 + &v2, Vector::new([4.0, 6.0]));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::new([5.0, 7.0]);
        let v2 = Vector::new([2.0, 3.0]);
        assert_eq!(&v1 - &v2, Vector::new([3.0, 4.0]));
    }

    #[test]
    fn test_vector_scalar_mul() {
        let v = Vector::new([1.0, -2.0]);
        assert_eq!(&v * 3.0, Vector::new([3.0, -6.0]));
    }

    #[test]
    fn test_vector_projection() {
        let v = Vector::new([1.0, 1.0]);
        let target = Vector::new([1.0, 0.0]);
        assert_eq!(v.orthogonal_projection(&target), Vector::new([1.0, 0.0]));
    }

    #[test]
    fn test_vector_null_projection() {
        let v = Vector::new([1.0, 2.0]);
        let null_v = Vector::<2>::new([0.0, 0.0]);
        assert_eq!(v.orthogonal_projection(&null_v), Vector::new([0.0, 0.0]));
    }
}
