use core::simd::{LaneCount, Simd, SupportedLaneCount};
use std::ops::{Add, Mul};
use std::simd::SimdFloat;

use eyre::Result;

use crate::context::{Context, TensorError, TensorId, TensorOp};

macro_rules! impl_operation_2ops {
    ($func:ident, $op:ident, $operation:ident) => {
        impl Context {
            /// Compute the multiplication for this tensor
            pub fn $func(&mut self, node: &TensorId) -> Result<()> {
                // Get the first operand
                let Some(first_op) = self.first_op(node) else {
                    return Err(TensorError::FirstOperandNeededForOp { tensor: *node, operation: TensorOp::$op}.into());
                };

                // Get the second operand
                let Some(second_op) = self.second_op(node) else {
                    return Err(TensorError::SecondOperandNeededForOp { tensor: *node, operation: TensorOp::$op}.into());
                };

                let dest_type_name = self.type_name(node);
                let op1_type_name = self.type_name(&first_op);
                let op2_type_name = self.type_name(&second_op);

                assert!(dest_type_name == "f32", "Only can mul by f32 for now");
                assert!(dest_type_name == op1_type_name);
                assert!(dest_type_name == op2_type_name);

                fn work<const LANES: usize>(ctx: &mut Context, operands: [&TensorId; 3])
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    let [dest, op1, op2] = operands;

                    // Ensure the input slices are the same size
                    assert!(ctx.datas[dest].len() == ctx.datas[op1].len());
                    assert!(ctx.datas[dest].len() == ctx.datas[op2].len());
                    let elems = ctx.datas[dest].len();
                    let mut index = 0;

                    while elems - index >= LANES {
                        // Get the SIMD values for the current slice
                        let x_chunk =
                            Simd::<f32, LANES>::from_slice(&ctx.datas[op1].get_f32_slice()[index..]);
                        let y_chunk =
                            Simd::<f32, LANES>::from_slice(&ctx.datas[op2].get_f32_slice()[index..]);

                        // out[i] = x[i] * y[i]
                        let sum_chunk = x_chunk.$operation(y_chunk);
                        let dest = ctx.datas[dest].get_f32_slice_mut();
                        dest[index..index + LANES].copy_from_slice(sum_chunk.as_array());

                        index += LANES;
                    }

                    // Add out[i] = x[i] + [y] for the remainder of the chunks
                    // naive(ctx, operands)
                    for i in index..elems {
                        *ctx.datas[dest].get_f32_mut(i) = ctx.datas[op1].get_f32(i).$operation(ctx.datas[op2].get_f32(i));
                    }
                }

                log::debug!(
                    "Before {:?} = {:?} {} {:?}",
                    self.datas[node], self.datas[first_op], stringify!($operation), self.datas[second_op]
                );

                #[cfg(feature = "avx512")]
                work::<16>(self, [node, &first_op, &second_op]);
                #[cfg(not(feature = "avx512"))]
                work::<8>(self, [node, &first_op, &second_op]);

                log::debug!(
                    "After  {:?} = {:?} {} {:?}",
                    self.datas[node], self.datas[first_op], stringify!($operation), self.datas[second_op]
                );

                Ok(())
            }
        }
    }
}

impl_operation_2ops!(compute_forward_mul, Mul, mul);
impl_operation_2ops!(compute_forward_add, Add, add);

macro_rules! impl_operation_1op {
    ($func:ident, $work:ident) => {
        impl Context {
            /// Compute the multiplication for this tensor
            pub fn $func(&mut self, node: &TensorId) -> Result<()> {
                // Get the first operand
                let Some(first_op) = self.first_op(node) else {
                    return Err(TensorError::FirstOperandNeededForOp { tensor: *node, operation: TensorOp::Relu}.into());
                };

                let dest_type_name = self.type_name(node);
                let op1_type_name = self.type_name(&first_op);

                assert!(dest_type_name == "f32", "Only can mul by f32 for now");
                assert!(dest_type_name == op1_type_name);


                log::debug!(
                    "Before {:?} = {:?} relu",
                    self.datas[node],
                    self.datas[first_op]
                );

                #[cfg(feature = "avx512")]
                $work::<16>(self, [node, &first_op]);
                #[cfg(not(feature = "avx512"))]
                $work::<8>(self, [node, &first_op]);

                log::debug!(
                    "After  {:?} = {:?} relu",
                    self.datas[node],
                    self.datas[first_op]
                );

                Ok(())
            }
        }
    }
}

/// SIMD implementation using stdsimd with a variety number of possible lanes
fn relu_work<const LANES: usize>(ctx: &mut Context, operands: [&TensorId; 2])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let [dest, op1] = operands;

    // Ensure the input slices are the same size
    assert!(ctx.datas[dest].len() == ctx.datas[op1].len());
    let elems = ctx.datas[dest].len();
    let mut index = 0;

    while elems - index >= LANES {
        // Get the SIMD values for the current slice
        let x_chunk = Simd::<f32, LANES>::from_slice(&ctx.datas[op1].get_f32_slice()[index..]);
        let min_chunk = Simd::<f32, LANES>::splat(0.0);

        // Perform the ReLU
        // out[i] = x[i] > 0.0 ? x[i] : 0.0
        let max_chunk = x_chunk.simd_max(min_chunk);
        let dest = ctx.datas[dest].get_f32_slice_mut();
        dest[index..index + LANES].copy_from_slice(max_chunk.as_array());

        index += LANES;
    }

    // Add out[i] = x[i] + [y] for the remainder of the chunks
    for i in index..elems {
        *ctx.datas[dest].get_f32_mut(i) = ctx.datas[op1].get_f32(i).max(0.0);
    }
}

impl_operation_1op!(compute_forward_relu, relu_work);

fn softmax_work<const LANES: usize>(ctx: &mut Context, operands: [&TensorId; 2])
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let [dest, op1] = operands;

    let len = ctx.datas[dest].get_f32_slice().len();

    for i in 0..len {
        ctx.datas[dest].get_f32_slice_mut()[i] = ctx.datas[op1].get_f32_slice()[i];
    }

    // Get the number of lanes
    let num_chunks = ctx.datas[dest]
        .get_f32_slice_mut()
        .chunks_exact_mut(LANES)
        .len();

    // e^self for each item
    ctx.datas[dest]
        .get_f32_slice_mut()
        .chunks_exact_mut(LANES)
        .for_each(|chunk| {
            for x in chunk.iter_mut() {
                *x = x.exp();
            }
        });

    // Normalize the remaining items non-divisible by LANES
    ctx.datas[dest]
        .get_f32_slice_mut()
        .iter_mut()
        .skip(num_chunks * LANES)
        .for_each(|x| *x = x.exp());

    // Calculate the sum of the exponentials
    let sum: f32 = ctx.datas[dest].get_f32_slice().iter().sum();

    // Normalize all of the exponentials
    ctx.datas[dest]
        .get_f32_slice_mut()
        .chunks_exact_mut(16)
        .for_each(|chunk| {
            for x in chunk.iter_mut() {
                *x /= sum;
            }
        });

    // Normalize the remaining items non-divisible by LANES
    ctx.datas[dest]
        .get_f32_slice_mut()
        .iter_mut()
        .skip(num_chunks * LANES)
        .for_each(|x| *x /= sum);
}
impl_operation_1op!(compute_forward_softmax, softmax_work);
