use talos_xii::prelude::*;

#[test]
fn public_prelude_builds_and_backpropagates_a_model() {
    let input = Tensor::new_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let layer = Linear::new(2, 3, true, 42);

    let output = layer.forward(&input).gelu();
    assert_eq!(output.shape, vec![2, 3]);
    assert_eq!(output.device, Device::Cpu);
    assert_eq!(output.dtype, Dtype::F32);

    output.mean().backward();
    let gradient = input.grad_to_f32_vec();
    assert_eq!(gradient.len(), input.numel());
    assert!(gradient.iter().all(|value| value.is_finite()));
}
