// Module est implémenté sur des tuples de couches (L1, L2, ..., LN).
// Chaque impl est écrit à la main car Rust n'a pas de variadic generics.
// Le forward chaîne les couches dans l'ordre, le backward les parcourt à l'envers.
// Supporte jusqu'à 10 couches — largement suffisant pour un réseau embarqué.

use crate::autodiff::module::Module;

impl<Input, L1, L2> Module<Input> for (L1, L2)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
{
    type Output = L2::Output;
    type Context = (L1::Context, L2::Context);
    type Gradients = (L1::Gradients, L2::Gradients);

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        (out2, (ctx1, ctx2))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2) = ctx;
        let (grad1, grads2) = self.1.backward(grad_out, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (grad0, (grads1, grads2))
    }
}

impl<Input, L1, L2, L3> Module<Input> for (L1, L2, L3)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
{
    type Output = L3::Output;
    type Context = (L1::Context, L2::Context, L3::Context);
    type Gradients = (L1::Gradients, L2::Gradients, L3::Gradients);

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        (out3, (ctx1, ctx2, ctx3))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3) = ctx;
        let (grad2, grads3) = self.2.backward(grad_out, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (grad0, (grads1, grads2, grads3))
    }
}

impl<Input, L1, L2, L3, L4> Module<Input> for (L1, L2, L3, L4)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
{
    type Output = L4::Output;
    type Context = (L1::Context, L2::Context, L3::Context, L4::Context);
    type Gradients = (L1::Gradients, L2::Gradients, L3::Gradients, L4::Gradients);

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        (out4, (ctx1, ctx2, ctx3, ctx4))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4) = ctx;
        let (grad3, grads4) = self.3.backward(grad_out, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (grad0, (grads1, grads2, grads3, grads4))
    }
}

impl<Input, L1, L2, L3, L4, L5> Module<Input> for (L1, L2, L3, L4, L5)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
{
    type Output = L5::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
    );
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
    );

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        (out5, (ctx1, ctx2, ctx3, ctx4, ctx5))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5) = ctx;
        let (grad4, grads5) = self.4.backward(grad_out, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (grad0, (grads1, grads2, grads3, grads4, grads5))
    }
}

impl<Input, L1, L2, L3, L4, L5, L6> Module<Input> for (L1, L2, L3, L4, L5, L6)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
    L6: Module<L5::Output>,
{
    type Output = L6::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
        L6::Context,
    );
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
    );

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        let (out6, ctx6) = self.5.forward(out5);
        (out6, (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6) = ctx;
        let (grad5, grads6) = self.5.backward(grad_out, ctx6);
        let (grad4, grads5) = self.4.backward(grad5, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (grad0, (grads1, grads2, grads3, grads4, grads5, grads6))
    }
}

impl<Input, L1, L2, L3, L4, L5, L6, L7> Module<Input> for (L1, L2, L3, L4, L5, L6, L7)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
    L6: Module<L5::Output>,
    L7: Module<L6::Output>,
{
    type Output = L7::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
        L6::Context,
        L7::Context,
    );
    type Gradients = (
        L1::Gradients,
        L2::Gradients,
        L3::Gradients,
        L4::Gradients,
        L5::Gradients,
        L6::Gradients,
        L7::Gradients,
    );

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        let (out6, ctx6) = self.5.forward(out5);
        let (out7, ctx7) = self.6.forward(out6);
        (out7, (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7) = ctx;
        let (grad6, grads7) = self.6.backward(grad_out, ctx7);
        let (grad5, grads6) = self.5.backward(grad6, ctx6);
        let (grad4, grads5) = self.4.backward(grad5, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (
            grad0,
            (grads1, grads2, grads3, grads4, grads5, grads6, grads7),
        )
    }
}

impl<Input, L1, L2, L3, L4, L5, L6, L7, L8> Module<Input> for (L1, L2, L3, L4, L5, L6, L7, L8)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
    L6: Module<L5::Output>,
    L7: Module<L6::Output>,
    L8: Module<L7::Output>,
{
    type Output = L8::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
        L6::Context,
        L7::Context,
        L8::Context,
    );
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

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        let (out6, ctx6) = self.5.forward(out5);
        let (out7, ctx7) = self.6.forward(out6);
        let (out8, ctx8) = self.7.forward(out7);
        (out8, (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8) = ctx;
        let (grad7, grads8) = self.7.backward(grad_out, ctx8);
        let (grad6, grads7) = self.6.backward(grad7, ctx7);
        let (grad5, grads6) = self.5.backward(grad6, ctx6);
        let (grad4, grads5) = self.4.backward(grad5, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (
            grad0,
            (
                grads1, grads2, grads3, grads4, grads5, grads6, grads7, grads8,
            ),
        )
    }
}

impl<Input, L1, L2, L3, L4, L5, L6, L7, L8, L9> Module<Input>
    for (L1, L2, L3, L4, L5, L6, L7, L8, L9)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
    L6: Module<L5::Output>,
    L7: Module<L6::Output>,
    L8: Module<L7::Output>,
    L9: Module<L8::Output>,
{
    type Output = L9::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
        L6::Context,
        L7::Context,
        L8::Context,
        L9::Context,
    );
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

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        let (out6, ctx6) = self.5.forward(out5);
        let (out7, ctx7) = self.6.forward(out6);
        let (out8, ctx8) = self.7.forward(out7);
        let (out9, ctx9) = self.8.forward(out8);
        (out9, (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8, ctx9))
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8, ctx9) = ctx;
        let (grad8, grads9) = self.8.backward(grad_out, ctx9);
        let (grad7, grads8) = self.7.backward(grad8, ctx8);
        let (grad6, grads7) = self.6.backward(grad7, ctx7);
        let (grad5, grads6) = self.5.backward(grad6, ctx6);
        let (grad4, grads5) = self.4.backward(grad5, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (
            grad0,
            (
                grads1, grads2, grads3, grads4, grads5, grads6, grads7, grads8, grads9,
            ),
        )
    }
}

impl<Input, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> Module<Input>
    for (L1, L2, L3, L4, L5, L6, L7, L8, L9, L10)
where
    L1: Module<Input>,
    L2: Module<L1::Output>,
    L3: Module<L2::Output>,
    L4: Module<L3::Output>,
    L5: Module<L4::Output>,
    L6: Module<L5::Output>,
    L7: Module<L6::Output>,
    L8: Module<L7::Output>,
    L9: Module<L8::Output>,
    L10: Module<L9::Output>,
{
    type Output = L10::Output;
    type Context = (
        L1::Context,
        L2::Context,
        L3::Context,
        L4::Context,
        L5::Context,
        L6::Context,
        L7::Context,
        L8::Context,
        L9::Context,
        L10::Context,
    );
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

    fn forward(&self, x: Input) -> (Self::Output, Self::Context) {
        let (out1, ctx1) = self.0.forward(x);
        let (out2, ctx2) = self.1.forward(out1);
        let (out3, ctx3) = self.2.forward(out2);
        let (out4, ctx4) = self.3.forward(out3);
        let (out5, ctx5) = self.4.forward(out4);
        let (out6, ctx6) = self.5.forward(out5);
        let (out7, ctx7) = self.6.forward(out6);
        let (out8, ctx8) = self.7.forward(out7);
        let (out9, ctx9) = self.8.forward(out8);
        let (out10, ctx10) = self.9.forward(out9);
        (
            out10,
            (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8, ctx9, ctx10),
        )
    }

    fn backward(&self, grad_out: Self::Output, ctx: &Self::Context) -> (Input, Self::Gradients) {
        let (ctx1, ctx2, ctx3, ctx4, ctx5, ctx6, ctx7, ctx8, ctx9, ctx10) = ctx;
        let (grad9, grads10) = self.9.backward(grad_out, ctx10);
        let (grad8, grads9) = self.8.backward(grad9, ctx9);
        let (grad7, grads8) = self.7.backward(grad8, ctx8);
        let (grad6, grads7) = self.6.backward(grad7, ctx7);
        let (grad5, grads6) = self.5.backward(grad6, ctx6);
        let (grad4, grads5) = self.4.backward(grad5, ctx5);
        let (grad3, grads4) = self.3.backward(grad4, ctx4);
        let (grad2, grads3) = self.2.backward(grad3, ctx3);
        let (grad1, grads2) = self.1.backward(grad2, ctx2);
        let (grad0, grads1) = self.0.backward(grad1, ctx1);
        (
            grad0,
            (
                grads1, grads2, grads3, grads4, grads5, grads6, grads7, grads8, grads9, grads10,
            ),
        )
    }
}
