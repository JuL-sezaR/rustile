use cutile::prelude::*;

// The macro captures this module's AST into the host binary. The kernel is
// JIT-compiled through CUDA Tile IR the first time it is actually launched.
#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like(x, z);
        let ty = load_tile_like(y, z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let c: Vec<f32> = kernel::add(z, x, y)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;

    let errors = c.iter().filter(|&&v| (v - 2.0).abs() > 1e-5).count();
    if errors == 0 {
        println!("PASSED: all {} elements correct", c.len());
    } else {
        eprintln!("FAILED: {errors} errors");
    }
    Ok(())
}
