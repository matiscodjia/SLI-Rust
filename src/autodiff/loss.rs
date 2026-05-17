use crate::vector::Vector;

/// Mean Squared Error — retourne (loss, gradient).
/// loss = mean((output - target)²), grad[i] = 2 * (output[i] - target[i]) / N
pub fn mse<const N: usize>(output: Vector<N>, target: Vector<N>) -> (f32, Vector<N>) {
    let mut loss = 0.0;
    let mut grad_data = [0.0; N];
    for i in 0..N {
        let diff = output[i] - target[i];
        loss += diff * diff;
        grad_data[i] = 2.0 * diff / N as f32;
    }
    (loss / N as f32, Vector::new(grad_data))
}

/// Mean Absolute Error — retourne (loss, gradient).
/// loss = mean(|output - target|), grad[i] = sign(output[i] - target[i]) / N
pub fn mae<const N: usize>(output: Vector<N>, target: Vector<N>) -> (f32, Vector<N>) {
    let mut loss = 0.0;
    let mut grad_data = [0.0; N];
    for i in 0..N {
        let diff = output[i] - target[i];
        loss += if diff >= 0.0 { diff } else { -diff };
        grad_data[i] = diff.signum() / N as f32;
    }
    (loss / N as f32, Vector::new(grad_data))
}
