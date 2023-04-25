#![feature(error_in_core)]
#![feature(vec_push_within_capacity)]
#![feature(const_trait_impl)]
#![feature(stmt_expr_attributes)]
#![feature(type_name_of_val)]
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(variant_count)]

pub mod context;
pub use context::{Context, TensorData, TensorOp, TensorType};

pub mod tensor_ops;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*;
    // const ERROR_MARGIN: f32 = f32::EPSILON;
    const ERROR_MARGIN: f32 = 0.001;

    /// Graph and compute a single operation to ensure it is properly working
    macro_rules! impl_test_forward_op {
        ($test:ident, $op:ident, $first_op:expr, $res:expr) => {
            #[test]
            fn $test() -> eyre::Result<()> {
                // Create the context to create tensors and the graph
                let mut ctx = Context::new(8);

                // Initialize the first operand
                let arg1_data = $first_op;
                let arg1 = ctx
                    .new_tensor_1dim_data(TensorData::F32(arg1_data.clone()))
                    .unwrap();

                // Ensure the data for the tensor was inserted properly
                let data = ctx.datas[arg1].get_f32_slice();
                assert!(data
                    .iter()
                    .zip(arg1_data.iter())
                    .all(|(x, y)| (x - y).abs() < ERROR_MARGIN));

                // Add a mul operation
                let out = ctx.$op(&arg1).unwrap();

                // Build the forward graph
                let graph = ctx.build_forward(&out)?;

                // Calculate the forward pass
                ctx.compute_forward(&graph)?;

                // Confirm the results
                let data = ctx.datas[out].get_f32_slice();
                let result = $res;
                assert!(
                    data.iter()
                        .zip(result.iter())
                        .all(|(x, y)| (x - y).abs() < ERROR_MARGIN),
                    "{data:?} {result:?} {:?}",
                    data.iter()
                        .zip(result.iter())
                        .map(|(x, y)| ((x - y).abs() * 100.0).round() / 100.0)
                        .collect::<Vec<_>>()
                );

                Ok(())
            }
        };
        ($test:ident, $op:ident, $arg1:expr, $arg2:expr, $res:expr) => {
            #[test]
            fn $test() -> eyre::Result<()> {
                let mut ctx = Context::new(8);
                let arg1_data = $arg1;
                let arg1 = ctx
                    .new_tensor_1dim_data(TensorData::F32(arg1_data.clone()))
                    .unwrap();

                let arg2_data = $arg2;
                let arg2 = ctx
                    .new_tensor_1dim_data(TensorData::F32(arg2_data.clone()))
                    .unwrap();

                // Ensure the data for the tensor was inserted properly
                let data = ctx.datas[arg1].get_f32_slice();
                assert!(data
                    .iter()
                    .zip(arg1_data.iter())
                    .all(|(x, y)| (x - y).abs() < ERROR_MARGIN));

                // Ensure the data for the tensor was inserted properly
                let data = ctx.datas[arg2].get_f32_slice();
                assert!(data
                    .iter()
                    .zip(arg2_data.iter())
                    .all(|(x, y)| (x - y).abs() < ERROR_MARGIN));

                // Add a mul operation
                let out = ctx.$op(&arg1, &arg2).unwrap();

                // Build the forward graph
                let graph = ctx.build_forward(&out)?;

                // Calculate the forward pass
                ctx.compute_forward(&graph)?;

                // Confirm the results
                let data = ctx.datas[out].get_f32_slice();
                let result = $res;
                assert!(
                    data.iter()
                        .zip(result.iter())
                        .all(|(x, y)| (x - y).abs() < ERROR_MARGIN),
                    "{data:?} {result:?}"
                );

                Ok(())
            }
        };
    }

    impl_test_forward_op!(
        test_mul,
        mul,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, 2.0, -3.0],
        vec![0.0, 2.0, 6.0, -12.0]
    );

    impl_test_forward_op!(
        test_add,
        add,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, 2.0, -5.0],
        vec![1.0, 3.0, 5.0, -1.0]
    );

    impl_test_forward_op!(
        test_relu,
        relu,
        vec![-1.0, -2.0, 3.0, 4.0],
        vec![0.0, 0.0, 3.0, 4.0]
    );

    impl_test_forward_op!(
        test_softmax,
        soft_max,
        vec![-1.0, 0.0, 3.0, 5.0],
        vec![0.002, 0.006, 0.118, 0.874]
    );
}
