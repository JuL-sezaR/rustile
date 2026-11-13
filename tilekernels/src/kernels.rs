//! The tile kernels. Each entry is launched from a safe wrapper in this
//! crate, which checks the shapes and picks the tile from [`crate::tiles`].

#[cutile::module]
pub(crate) mod entries {
    use cutile::core::*;
    use cutile::tileir::*;

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

    /// RMSNorm along the columns. The tile spans the whole row. Columns past
    /// the end read as 0, so they add nothing to the sum of squares, and the
    /// mean divides by the real column count. `g` is loaded as one tile,
    /// zero-padded the same way.
    #[cutile::entry()]
    fn rmsnorm_f32<E: ElementType, const BM: i32, const BD: i32>(
        y: &mut Tensor<f32, { [BM, BD] }>,
        x: &Tensor<E, { [-1, -1] }>,
        g: &Tensor<E, { [-1] }>,
        eps: f32,
    ) {
        let tx: Tile<E, { [BM, BD] }> = x.load_like(y);
        let tx: Tile<f32, { [BM, BD] }> = convert_tile(tx);
        let d: i32 = x.shape()[1];
        let df: f32 = convert_scalar(d);
        let ss: Tile<f32, { [BM] }> = reduce_sum(tx * tx, 1i32);
        let ss: Tile<f32, { [BM, BD] }> = ss.reshape(shape![BM, 1]).broadcast(y.shape());
        let dt: Tile<f32, { [BM, BD] }> = df.broadcast(y.shape());
        let et: Tile<f32, { [BM, BD] }> = eps.broadcast(y.shape());
        let inv: Tile<f32, { [BM, BD] }> = rsqrt(ss / dt + et, ftz::Disabled);
        let tg: Tile<E, { [BD] }> = g.partition(shape![BD]).load([0i32]);
        let tg: Tile<f32, { [BD] }> = convert_tile(tg);
        let tg: Tile<f32, { [BM, BD] }> = tg.reshape(shape![1, BD]).broadcast(y.shape());
        y.store(tx * inv * tg);
    }

    /// The same in `f64`.
    #[cutile::entry()]
    fn rmsnorm_f64<const BM: i32, const BD: i32>(
        y: &mut Tensor<f64, { [BM, BD] }>,
        x: &Tensor<f64, { [-1, -1] }>,
        g: &Tensor<f64, { [-1] }>,
        eps: f64,
    ) {
        let tx: Tile<f64, { [BM, BD] }> = x.load_like(y);
        let d: i32 = x.shape()[1];
        let df: f64 = convert_scalar(d);
        let ss: Tile<f64, { [BM] }> = reduce_sum(tx * tx, 1i32);
        let ss: Tile<f64, { [BM, BD] }> = ss.reshape(shape![BM, 1]).broadcast(y.shape());
        let dt: Tile<f64, { [BM, BD] }> = df.broadcast(y.shape());
        let et: Tile<f64, { [BM, BD] }> = eps.broadcast(y.shape());
        let inv: Tile<f64, { [BM, BD] }> = rsqrt(ss / dt + et, ftz::Disabled);
        let tg: Tile<f64, { [BD] }> = g.partition(shape![BD]).load([0i32]);
        let tg: Tile<f64, { [BM, BD] }> = tg.reshape(shape![1, BD]).broadcast(y.shape());
        y.store(tx * inv * tg);
    }

    /// Embedding gather. Block `(p0, p1)` loads `BM` ids and copies columns
    /// `p1 * BD ..` of those rows. The view pads with zeros, so an id outside
    /// the table reads a row of zeros. Ids past the end of `ids` read as 0
    /// (a row that exists) and the rows they produce are clipped, as are
    /// columns past the end.
    ///
    /// # Safety
    /// `table`, `ids` and `y` must stay alive and unwritten by anything else
    /// until the launch has finished.
    #[cutile::entry()]
    unsafe fn gather<E: ElementType, const BM: i32, const BD: i32>(
        y: &mut Tensor<E, { [BM, BD] }>,
        table: &Tensor<E, { [-1, -1] }>,
        ids: &Tensor<i32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tid: Tile<i32, { [BM] }> = ids.partition(shape![BM]).load([pid.0]);
        let view: GatherScatterView<E, { [BM, BD] }, 0> =
            unsafe { make_gather_scatter_view(table, shape![BM, BD], padding::Zero) };
        let (t, _tok): (Tile<E, { [BM, BD] }>, Token) = unsafe {
            load_gather_scatter_view_tko(
                &view,
                tid,
                pid.1 * BD,
                new_token_unordered(),
                ordering::Relaxed,
                scope::TileBlock,
            )
        };
        y.store(t);
    }

    /// A linear layer, `y = x @ w^T + b`, with `w` of shape `[n, k]`. Each
    /// weight tile is transposed as it is loaded, so no transposed copy of
    /// the weight is ever made. The loop over `k` runs `ceil(k / BK)` times;
    /// the last `x` and `w` tiles are both zero-padded along `k`, so the
    /// padding adds 0. Padded output rows and columns are clipped.
    #[cutile::entry()]
    fn linear_f32<E: ElementType, const BM: i32, const BN: i32, const BK: i32>(
        y: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<E, { [-1, -1] }>,
        w: &Tensor<E, { [-1, -1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_w = w.partition(shape![BN, BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let k: i32 = x.shape()[1];
        let nk: i32 = ceil_div(k, BK);
        let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        for i in 0i32..nk {
            let tx: Tile<E, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tw: Tile<E, { [BN, BK] }> = part_w.load([pid.1, i]);
            let twt: Tile<E, { [BK, BN] }> = tw.transpose();
            acc = mma(tx, twt, acc);
        }
        let tb: Tile<f32, { [BN] }> = b.partition(shape![BN]).load([pid.1]);
        let tb: Tile<f32, { [BM, BN] }> = tb.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        y.store(acc + tb);
    }

    /// The same in `f64`.
    #[cutile::entry()]
    fn linear_f64<const BM: i32, const BN: i32, const BK: i32>(
        y: &mut Tensor<f64, { [BM, BN] }>,
        x: &Tensor<f64, { [-1, -1] }>,
        w: &Tensor<f64, { [-1, -1] }>,
        b: &Tensor<f64, { [-1] }>,
    ) {
        let part_x = x.partition(shape![BM, BK]);
        let part_w = w.partition(shape![BN, BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let k: i32 = x.shape()[1];
        let nk: i32 = ceil_div(k, BK);
        let mut acc: Tile<f64, { [BM, BN] }> = constant(0.0f64, shape![BM, BN]);
        for i in 0i32..nk {
            let tx: Tile<f64, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tw: Tile<f64, { [BN, BK] }> = part_w.load([pid.1, i]);
            let twt: Tile<f64, { [BK, BN] }> = tw.transpose();
            acc = mma(tx, twt, acc);
        }
        let tb: Tile<f64, { [BN] }> = b.partition(shape![BN]).load([pid.1]);
        let tb: Tile<f64, { [BM, BN] }> = tb.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        y.store(acc + tb);
    }
}
