use color_eyre::eyre::Result;
use ggml_rs::{Context, TensorType};

fn main() -> Result<()> {
    color_eyre::install()?;
    env_logger::init();

    let mut ctx = Context::new(1024);

    let x = ctx.new_tensor_1dim(TensorType::F32, 4)?;
    ctx.set_root_node(&x)?;

    // Create the graph
    let a = ctx.new_tensor_1dim(TensorType::F32, 4)?;
    let b = ctx.new_tensor_1dim(TensorType::F32, 4)?;
    let x2 = ctx.mul(&x, &x)?;
    let m1 = ctx.mul(&a, &x2)?;
    let f = ctx.add(&m1, &b)?;
    let out = ctx.relu(&f)?;
    let out = ctx.soft_max(&out)?;

    // let weights = ctx.new_tensor_2dim(TensorType::F32, 2, 3)?;
    // let bias = ctx.new_tensor_2dim(TensorType::F32, 2, 4)?;
    let out = ctx.matrix_mul(&a, &b)?;

    // Build the forward graph
    let graph = ctx.build_forward(&out)?;

    // Set input values
    ctx.set_value_f32(&x, 2.0)?;
    ctx.set_value_f32(&a, 3.0)?;
    ctx.set_value_f32(&b, 4.0)?;

    // ctx.dump_dot(&mut graph);
    // ctx.print_graph(&graph);

    for _ in 0..0x1fff {
        ctx.compute_forward(&graph)?;
    }

    ctx.print_stats();

    Ok(())
}
