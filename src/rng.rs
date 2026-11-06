//! Seedable pseudo-random numbers with a fixed, documented algorithm.
//!
//! The generator is xoshiro256++ 1.0 (Blackman and Vigna), with its 256-bit
//! state filled by four outputs of SplitMix64 started at the seed, exactly as
//! the reference implementation at <https://prng.di.unimi.it/> recommends.
//! Bounded integers use Lemire's multiply-and-reject method, so they are
//! unbiased. Both are implemented here rather than taken from a crate so that a
//! seed reproduces the same numbers in every future version of ireval; the
//! unit tests pin outputs produced by the reference C code.

/// SplitMix64 step, used for seeding and for deriving per-stream seeds.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// xoshiro256++ generator.
#[derive(Clone, Debug)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self { s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` (Lemire 2019, "Fast random integer generation
    /// in an interval"). `n` must be positive.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0)");
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// 64-bit FNV-1a hash, used to derive a stream seed from names so that a
/// comparison's random numbers do not depend on the order of comparisons.
pub fn fnv1a(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            h ^= 0xff;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for b in part.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Seed for one named stream: SplitMix64 applied to `seed ^ fnv1a(parts)`.
pub fn stream_seed(seed: u64, parts: &[&str]) -> u64 {
    let mut s = seed ^ fnv1a(parts);
    splitmix64(&mut s)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors printed by the reference C code (splitmix64.c and
    // xoshiro256plusplus.c from prng.di.unimi.it), seeding s[0..4] with four
    // splitmix64 outputs.
    #[test]
    fn matches_reference_c_implementation() {
        let cases: [(u64, [u64; 6]); 3] = [
            (
                0,
                [
                    0x53175d61490b23df,
                    0x61da6f3dc380d507,
                    0x5c0fdf91ec9a7bfc,
                    0x02eebf8c3bbe5e1a,
                    0x7eca04ebaf4a5eea,
                    0x0543c37757f08d9a,
                ],
            ),
            (
                42,
                [
                    0xd0764d4f4476689f,
                    0x519e4174576f3791,
                    0xfbe07cfb0c24ed8c,
                    0xb37d9f600cd835b8,
                    0xcb231c3874846a73,
                    0x968d9f004e50de7d,
                ],
            ),
            (
                20261107,
                [
                    0x3968f7e332c6e425,
                    0xf5b6e65fbfa6b127,
                    0x7986cc8d771f30ae,
                    0x8c56e4e55c0e63f3,
                    0xad275e8a838da5b6,
                    0x838d11901f77f822,
                ],
            ),
        ];
        for (seed, expected) in cases {
            let mut rng = Xoshiro256pp::from_seed(seed);
            for want in expected {
                assert_eq!(rng.next_u64(), want, "seed {seed}");
            }
        }
        let mut sm = 0;
        assert_eq!(splitmix64(&mut sm), 0xe220a8397b1dcdaf);
        assert_eq!(splitmix64(&mut sm), 0x6e789e6aa1b965f4);
        assert_eq!(splitmix64(&mut sm), 0x06c45d188009454f);
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut rng = Xoshiro256pp::from_seed(7);
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            counts[rng.below(7) as usize] += 1;
        }
        for c in counts {
            assert!((9_400..10_600).contains(&c), "{counts:?}");
        }
        assert_eq!(rng.below(1), 0);
    }
}
