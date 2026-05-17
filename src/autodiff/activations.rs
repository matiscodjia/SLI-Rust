use crate::autodiff::module::Module;
use crate::autodiff::update::Update;
use crate::vector::Vector;
use libm::{expf, tanhf};

#[derive(Clone, Copy)]
pub struct ReLU<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for ReLU<N> {
    type Output = Vector<N>;
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = x[i];
            }
        }
        (Vector::new(result), x)
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Vector<N>, Self::Gradients) {
        let x = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            if x[i] > 0.0 {
                result[i] = grad_out[i];
            }
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for ReLU<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: f32) {}
}

#[derive(Clone, Copy)]
pub struct Sigmoid<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for Sigmoid<N> {
    type Output = Vector<N>;
    // Context stocke σ(x) directement — utile pour le backward : σ'(x) = σ(x) * (1 - σ(x))
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = 1.0 / (1.0 + expf(-x[i]));
        }
        let output = Vector::new(result);
        (output, output)
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Vector<N>, Self::Gradients) {
        let sigmoid = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = grad_out[i] * sigmoid[i] * (1.0 - sigmoid[i]);
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for Sigmoid<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: f32) {}
}

#[derive(Clone, Copy)]
pub struct Tanh<const N: usize> {}

impl<const N: usize> Module<Vector<N>> for Tanh<N> {
    type Output = Vector<N>;
    // Context stocke tanh(x) — utile pour le backward : tanh'(x) = 1 - tanh(x)²
    type Context = Vector<N>;
    type Gradients = ();

    fn forward(&self, x: Vector<N>) -> (Self::Output, Self::Context) {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = tanhf(x[i]);
        }
        let output = Vector::new(result);
        (output, output)
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Vector<N>, Self::Gradients) {
        let tanh = ctx;
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = grad_out[i] * (1.0 - tanh[i] * tanh[i]);
        }
        (Vector::new(result), ())
    }
}

impl<const N: usize> Update for Tanh<N> {
    type Gradients = ();
    fn update(&mut self, _grads: &Self::Gradients, _lr: f32) {}
}
