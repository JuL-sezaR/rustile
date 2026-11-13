//! The tile kernels. Each entry is launched from a safe wrapper in this
//! crate, which checks the shapes and picks the tile from [`crate::tiles`].

#[cutile::module]
pub(crate) mod entries {
    use cutile::core::*;

    /// Softmax along the columns. The tile spans the whole row, so the
    /// columns past the end of the tensor, which read as 0, are replaced by
    /// `-inf` before the maximum and the sum are taken. Rows past the end
    /// compute garbage that the clipped store drops.
    #[cutile::entry()]
    fn softmax_f32<E: ElementType, const BM: i32, const BN: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<E, { [-1, -1] }>,
    ) {
        let tx: Tile<E, { [BM, BN] }> = x.load_like(y);
        let tx: Tile<f32, { [BM, BN] }> = convert_tile(tx);
        let n: i32 = x.shape()[1];
        let idx: Tile<i32, { [BN] }> = iota(shape![BN]);
        let nn: Tile<i32, { [BN] }> = n.broadcast(shape![BN]);
        let mask: Tile<bool, { [BN] }> = lt_tile(idx, nn);
        let mask: Tile<bool, { [BM, BN] }> = mask.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let ninf: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
        let tx: Tile<f32, { [BM, BN] }> = select(mask, tx, ninf);
        let mx: Tile<f32, { [BM] }> = reduce_max(tx, 1i32);
        let mx: Tile<f32, { [BM, BN] }> = mx.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let num: Tile<f32, { [BM, BN] }> = exp(tx - mx);
        let den: Tile<f32, { [BM] }> = reduce_sum(num, 1i32);
        let den: Tile<f32, { [BM, BN] }> = den.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        y.store(num / den);
    }

    /// The same in `f64`.
    #[cutile::entry()]
    fn softmax_f64<const BM: i32, const BN: i32>(
        y: &mut Tensor<f64, { [BM, BN] }>,
        x: &Tensor<f64, { [-1, -1] }>,
    ) {
        let tx: Tile<f64, { [BM, BN] }> = x.load_like(y);
        let n: i32 = x.shape()[1];
        let idx: Tile<i32, { [BN] }> = iota(shape![BN]);
        let nn: Tile<i32, { [BN] }> = n.broadcast(shape![BN]);
        let mask: Tile<bool, { [BN] }> = lt_tile(idx, nn);
        let mask: Tile<bool, { [BM, BN] }> = mask.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let ninf: Tile<f64, { [BM, BN] }> = constant(f64::NEG_INFINITY, shape![BM, BN]);
        let tx: Tile<f64, { [BM, BN] }> = select(mask, tx, ninf);
        let mx: Tile<f64, { [BM] }> = reduce_max(tx, 1i32);
        let mx: Tile<f64, { [BM, BN] }> = mx.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        let num: Tile<f64, { [BM, BN] }> = exp(tx - mx);
        let den: Tile<f64, { [BM] }> = reduce_sum(num, 1i32);
        let den: Tile<f64, { [BM, BN] }> = den.reshape(shape![BM, 1]).broadcast(shape![BM, BN]);
        y.store(num / den);
    }
}
