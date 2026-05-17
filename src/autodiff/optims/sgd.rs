use crate::autodiff::update::Update;

pub fn sgd<N: Update>(network: &mut N, grads: &N::Gradients, lr: f32) {
    network.update(grads, lr);
}
