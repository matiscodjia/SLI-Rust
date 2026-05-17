pub trait Update {
    type Gradients;
    fn update(&mut self, grads: &Self::Gradients, lr: f32);
}

impl<L1, L2> Update for (L1, L2)
where
    L1: Update,
    L2: Update,
{
    type Gradients = (L1::Gradients, L2::Gradients);
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
    }
}

impl<L1, L2, L3> Update for (L1, L2, L3)
where
    L1: Update,
    L2: Update,
    L3: Update,
{
    type Gradients = (L1::Gradients, L2::Gradients, L3::Gradients);
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
    }
}

impl<L1, L2, L3, L4> Update for (L1, L2, L3, L4)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
{
    type Gradients = (L1::Gradients, L2::Gradients, L3::Gradients, L4::Gradients);
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
    }
}

impl<L1, L2, L3, L4, L5> Update for (L1, L2, L3, L4, L5)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
    }
}

impl<L1, L2, L3, L4, L5, L6> Update for (L1, L2, L3, L4, L5, L6)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
    L6: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
        self.5.update(&grads.5, lr);
    }
}

impl<L1, L2, L3, L4, L5, L6, L7> Update for (L1, L2, L3, L4, L5, L6, L7)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
    L6: Update,
    L7: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
        L7::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
        self.5.update(&grads.5, lr);
        self.6.update(&grads.6, lr);
    }
}

impl<L1, L2, L3, L4, L5, L6, L7, L8> Update for (L1, L2, L3, L4, L5, L6, L7, L8)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
    L6: Update,
    L7: Update,
    L8: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
        L7::Gradients,
        L8::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
        self.5.update(&grads.5, lr);
        self.6.update(&grads.6, lr);
        self.7.update(&grads.7, lr);
    }
}

impl<L1, L2, L3, L4, L5, L6, L7, L8, L9> Update for (L1, L2, L3, L4, L5, L6, L7, L8, L9)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
    L6: Update,
    L7: Update,
    L8: Update,
    L9: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
        L7::Gradients,
        L8::Gradients,
        L9::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
        self.5.update(&grads.5, lr);
        self.6.update(&grads.6, lr);
        self.7.update(&grads.7, lr);
        self.8.update(&grads.8, lr);
    }
}

impl<L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> Update for (L1, L2, L3, L4, L5, L6, L7, L8, L9, L10)
where
    L1: Update,
    L2: Update,
    L3: Update,
    L4: Update,
    L5: Update,
    L6: Update,
    L7: Update,
    L8: Update,
    L9: Update,
    L10: Update,
{
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
        L7::Gradients,
        L8::Gradients,
        L9::Gradients,
        L10::Gradients,
    );
    fn update(&mut self, grads: &Self::Gradients, lr: f32) {
        self.0.update(&grads.0, lr);
        self.1.update(&grads.1, lr);
        self.2.update(&grads.2, lr);
        self.3.update(&grads.3, lr);
        self.4.update(&grads.4, lr);
        self.5.update(&grads.5, lr);
        self.6.update(&grads.6, lr);
        self.7.update(&grads.7, lr);
        self.8.update(&grads.8, lr);
        self.9.update(&grads.9, lr);
    }
}
