//! CPU references: plain single-threaded loops over row-major slices,
//! accumulating in `f64`. They are the oracle for the kernel tests and the
//! naive baseline for the benchmark. The caller converts stored values to
//! `f64` first, so the reference sees exactly the values the device sees.
//!
//! Slice lengths that do not match the stated sizes are a bug in the caller
//! and panic.

/// Row-wise softmax of a `rows × cols` matrix. Subtracts the row maximum
/// before `exp`, like the kernel. A row that is all `-inf` is not defined.
pub fn softmax(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(x.len(), rows * cols, "x is not rows × cols");
    let mut y = vec![0.0; x.len()];
    for (src, dst) in x
        .chunks_exact(cols.max(1))
        .zip(y.chunks_exact_mut(cols.max(1)))
    {
        let max = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = (v - max).exp();
            sum += *d;
        }
        for d in dst.iter_mut() {
            *d /= sum;
        }
    }
    y
}

/// `y[i][j] = x[i][j] / sqrt(mean_j(x[i][j]^2) + eps) * gain[j]`.
pub fn rmsnorm(x: &[f64], gain: &[f64], rows: usize, cols: usize, eps: f64) -> Vec<f64> {
    assert_eq!(x.len(), rows * cols, "x is not rows × cols");
    assert_eq!(
        gain.len(),
        cols,
        "gain does not have one element per column"
    );
    let mut y = vec![0.0; x.len()];
    for (src, dst) in x
        .chunks_exact(cols.max(1))
        .zip(y.chunks_exact_mut(cols.max(1)))
    {
        let mean_square = src.iter().map(|v| v * v).sum::<f64>() / cols as f64;
        let inv = 1.0 / (mean_square + eps).sqrt();
        for ((d, &v), &g) in dst.iter_mut().zip(src).zip(gain) {
            *d = v * inv * g;
        }
    }
    y
}

/// `out[i] = table[ids[i]]`, where `table` has `d` columns. A row of `zero`
/// for an id that is negative or not less than the number of table rows
/// (`table.len() / d`).
pub fn gather<T: Copy>(table: &[T], ids: &[i32], d: usize, zero: T) -> Vec<T> {
    assert!(d > 0, "d is 0");
    assert_eq!(table.len() % d, 0, "table is not a whole number of rows");
    let v = table.len() / d;
    let mut out = Vec::with_capacity(ids.len() * d);
    for &id in ids {
        match usize::try_from(id).ok().filter(|&row| row < v) {
            Some(row) => out.extend_from_slice(&table[row * d..(row + 1) * d]),
            None => out.extend(std::iter::repeat_n(zero, d)),
        }
    }
    out
}

/// `y[i][j] = sum_p x[i][p] * w[j][p] + bias[j]` for `x: m × k` and
/// `w: n × k` (one row per output), and with it
/// `scale[i][j] = sum_p |x[i][p] * w[j][p]| + |bias[j]|`, which the test
/// tolerance for a rounding error in the sum is proportional to.
pub fn linear(
    x: &[f64],
    w: &[f64],
    bias: Option<&[f64]>,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(x.len(), m * k, "x is not m × k");
    assert_eq!(w.len(), n * k, "w is not n × k");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n, "bias does not have one element per output");
    }
    let mut y = vec![0.0; m * n];
    let mut scale = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            let mut abs = 0.0;
            for p in 0..k {
                let product = x[i * k + p] * w[j * k + p];
                sum += product;
                abs += product.abs();
            }
            let b = bias.map_or(0.0, |bias| bias[j]);
            y[i * n + j] = sum + b;
            scale[i * n + j] = abs + b.abs();
        }
    }
    (y, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: &[f64], want: &[f64], tolerance: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tolerance, "element {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn softmax_of_zero_and_ln2_is_a_third_and_two_thirds() {
        let y = softmax(&[0.0, std::f64::consts::LN_2], 1, 2);
        close(&y, &[1.0 / 3.0, 2.0 / 3.0], 1e-15);
    }

    #[test]
    fn softmax_rows_are_independent_and_sum_to_one() {
        let y = softmax(&[1.0, 2.0, 3.0, 100.0, 100.0, 100.0], 2, 3);
        close(&y[3..], &[1.0 / 3.0; 3], 1e-15);
        for row in y.chunks(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        }
        close(&y[..1], &[1.0 / (1.0 + 1f64.exp() + 2f64.exp())], 1e-15);
    }

    #[test]
    fn softmax_does_not_overflow_on_large_values() {
        let y = softmax(&[1e4, 1e4 - 1.0, -1e4], 1, 3);
        let e = (-1f64).exp();
        close(&y, &[1.0 / (1.0 + e), e / (1.0 + e), 0.0], 1e-15);
    }

    #[test]
    fn softmax_gives_zero_to_minus_infinity() {
        let y = softmax(&[f64::NEG_INFINITY, 0.0, 0.0], 1, 3);
        assert_eq!(y, vec![0.0, 0.5, 0.5]);
        assert_eq!(softmax(&[7.0], 1, 1), vec![1.0]);
    }

    #[test]
    fn rmsnorm_of_3_and_4_divides_by_the_root_mean_square() {
        // mean square is (9 + 16) / 2 = 12.5
        let y = rmsnorm(&[3.0, 4.0], &[1.0, 1.0], 1, 2, 0.0);
        let rms = 12.5f64.sqrt();
        close(&y, &[3.0 / rms, 4.0 / rms], 1e-15);
    }

    #[test]
    fn rmsnorm_applies_the_gain_per_column_and_eps_to_the_mean() {
        let y = rmsnorm(&[3.0, 4.0], &[2.0, -1.0], 1, 2, 1.5);
        let rms = (12.5f64 + 1.5).sqrt(); // 14 ^ 0.5
        close(&y, &[6.0 / rms, -4.0 / rms], 1e-15);
    }

    #[test]
    fn rmsnorm_of_a_zero_row_is_zero_and_rows_are_independent() {
        let y = rmsnorm(&[0.0, 0.0, 3.0, 4.0], &[1.0, 1.0], 2, 2, 1e-6);
        assert_eq!(&y[..2], &[0.0, 0.0]);
        let rms = (12.5f64 + 1e-6).sqrt();
        close(&y[2..], &[3.0 / rms, 4.0 / rms], 1e-15);
    }

    #[test]
    fn gather_copies_rows_and_zeroes_ids_outside_the_table() {
        let table = [10, 11, 20, 21, 30, 31]; // 3 rows, d = 2
        let out = gather(&table, &[2, -1, 3, 0, 0, i32::MIN, i32::MAX], 2, 0);
        assert_eq!(out, vec![30, 31, 0, 0, 0, 0, 10, 11, 10, 11, 0, 0, 0, 0]);
    }

    #[test]
    fn gather_with_no_ids_is_empty_and_works_for_any_element() {
        assert!(gather(&[1.0f64, 2.0], &[], 2, 0.0).is_empty());
        assert_eq!(gather(&[1u8, 2, 3, 4], &[1, 1], 2, 9), vec![3, 4, 3, 4]);
    }

    // x = [[1, 2, 3], [4, 5, 6]], w = [[1, 0, -1], [2, 1, 0]], bias = [10, -1]
    const X: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const W: [f64; 6] = [1.0, 0.0, -1.0, 2.0, 1.0, 0.0];

    #[test]
    fn linear_by_hand_without_a_bias() {
        // y[0][0] = 1 - 3 = -2        y[0][1] = 2 + 2 = 4
        // y[1][0] = 4 - 6 = -2        y[1][1] = 8 + 5 = 13
        let (y, scale) = linear(&X, &W, None, 2, 3, 2);
        assert_eq!(y, vec![-2.0, 4.0, -2.0, 13.0]);
        assert_eq!(scale, vec![4.0, 4.0, 10.0, 13.0]);
    }

    #[test]
    fn linear_by_hand_with_a_bias() {
        let (y, scale) = linear(&X, &W, Some(&[10.0, -1.0]), 2, 3, 2);
        assert_eq!(y, vec![8.0, 3.0, 8.0, 12.0]);
        assert_eq!(scale, vec![14.0, 5.0, 20.0, 14.0]);
    }

    #[test]
    fn linear_with_an_identity_weight_returns_x() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (y, _) = linear(&X, &identity, None, 2, 3, 3);
        assert_eq!(y, X.to_vec());
    }

    #[test]
    #[should_panic(expected = "gain does not have one element per column")]
    fn a_wrong_length_panics() {
        rmsnorm(&[1.0, 2.0], &[1.0], 1, 2, 0.0);
    }
}
