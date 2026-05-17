pub trait Module<Input> {
    type Output;
    type Context;
    type Gradients;

    fn forward(&self, x: Input) -> (Self::Output, Self::Context);
    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients);
}
