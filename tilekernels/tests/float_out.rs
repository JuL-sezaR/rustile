//! The output type of a `Float` can be named from outside the crate. This is
//! an integration test so that it compiles the way a user's code does; a
//! unit test inside the crate could see the sealed traits and would not
//! notice if the documented form stopped working (issue #15). Needs no GPU.

use cutile::half::bf16;
use tilekernels::Float;

/// The qualified form the documentation promises.
fn size_of_out<E: Float>() -> usize {
    size_of::<<E as Float>::Out>()
}

/// The shorthand for generic code.
fn size_of_out_shorthand<E: Float>() -> usize {
    size_of::<E::Out>()
}

#[test]
fn the_output_type_is_nameable_in_generic_code() {
    assert_eq!(size_of_out::<bf16>(), 4);
    assert_eq!(size_of_out::<f32>(), 4);
    assert_eq!(size_of_out::<f64>(), 8);
    assert_eq!(size_of_out_shorthand::<bf16>(), 4);
    assert_eq!(size_of_out_shorthand::<f64>(), 8);
}

#[test]
fn the_output_type_of_a_concrete_type_is_nameable() {
    let from_bf16: <bf16 as Float>::Out = 1.5f32;
    let from_f32: <f32 as Float>::Out = 1.5f32;
    let from_f64: <f64 as Float>::Out = 1.5f64;
    assert_eq!((from_bf16, from_f32, from_f64), (1.5f32, 1.5f32, 1.5f64));
}
