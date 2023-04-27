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

    #[test]
    fn test_matrix_mul() -> eyre::Result<()> {
        use rand::Rng;

        // Create the context to create tensors and the graph
        let mut ctx = Context::new(64);
        let mut rng = rand::thread_rng();

        let rows = 16 * 20 + 2;
        let cols = 16 * 20 + 7;

        // Initialize the first operand
        let arg1_data = vec![rng.gen_range(-1.0..1.0); rows * cols];
        let arg1 = ctx
            .new_tensor_2dim_data(rows, cols, TensorData::F32(arg1_data.clone()))
            .unwrap();

        let arg2_data = vec![rng.gen_range(-1.0..1.0); rows * cols];
        let arg2 = ctx
            .new_tensor_2dim_data(cols, rows, TensorData::F32(arg2_data.clone()))
            .unwrap();

        // Add a mul operation
        let out = ctx.matrix_mul(&arg1, &arg2).unwrap();

        // Build the forward graph
        let graph = ctx.build_forward(&out)?;

        // Calculate the forward pass
        ctx.compute_forward(&graph)?;

        // Call the easy, naive result to check the result
        let mut naive_result = vec![rng.gen_range(-1.0..1.0); rows * rows];
        naive_matrix_mul(
            &mut naive_result,
            &[rows, rows],
            &arg1_data,
            &[rows, cols],
            &arg2_data,
            &[cols, rows],
        );

        // Confirm the results
        let data = ctx.datas[out].get_f32_slice();

        const ERROR: f32 = 0.01;
        assert!(
            data.iter()
                .zip(naive_result.iter())
                .map(|(x, y)| (x - y).abs())
                .all(|x| x <= ERROR),
            "Errors: {:?}",
            data.iter()
                .zip(naive_result.iter())
                .map(|(x, y)| (x - y).abs())
                .filter(|x| x >= &ERROR)
                .collect::<Vec<_>>()
        );

        Ok(())
    }
}

pub fn naive_matrix_mul(
    sum: &mut [f32],
    sum_dims: &[usize; 2],
    arg1: &[f32],
    arg1_dims: &[usize; 2],
    arg2: &[f32],
    arg2_dims: &[usize; 2],
) {
    let [arg1_rows, arg1_cols] = arg1_dims.clone();
    assert!(arg1.len() == arg1_rows * arg1_cols);

    let [arg2_rows, arg2_cols] = arg2_dims.clone();
    assert!(
        arg2.len() == arg2_rows * arg2_cols,
        "{} != {}",
        arg2.len(),
        arg2_rows * arg2_cols
    );

    let [sum_rows, sum_cols] = sum_dims.clone();
    assert!(sum.len() == sum_rows * sum_cols);

    assert!(arg1_cols == arg2_rows);
    assert!(sum_rows == arg1_rows);
    assert!(sum_cols == arg2_cols);

    for row in 0..arg1_rows {
        for col in 0..arg2_cols {
            let mut curr_sum = 0.0;

            for index in 0..arg1_cols {
                curr_sum += arg1[row * arg1_cols + index] * arg2[arg2_cols * index + col];
            }

            sum[row * sum_cols + col] = curr_sum;
        }
    }
}
