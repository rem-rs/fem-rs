//! MFEM `miniapps/spde/util.{hpp,cpp}` port.
//!
//! `FillWithRandomNumbers` / `FillWithRandomRotations` use `std::mt19937`
//! seeded from `std::random_device` in C++. The Rust port keeps the exact
//! MT19937 + libstdc++ `uniform_real_distribution` algorithm (so a fixed seed
//! reproduces the C++ stream bit-for-bit) but seeds from wall-clock time since
//! the standard library has no `random_device` equivalent.

/// MT19937 (32-bit Mersenne Twister), matching `std::mt19937`.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;
    const MATRIX_A: u32 = 0x9908_B0DF;
    const UPPER_MASK: u32 = 0x8000_0000;
    const LOWER_MASK: u32 = 0x7FFF_FFFF;

    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; Self::N];
        state[0] = seed;
        for i in 1..Self::N {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: Self::N }
    }

    fn generate(&mut self) {
        for i in 0..Self::N {
            let y = (self.state[i] & Self::UPPER_MASK) | (self.state[(i + 1) % Self::N] & Self::LOWER_MASK);
            let mut next = self.state[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= Self::MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= Self::N {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    /// libstdc++ `generate_canonical<double, 53>` for MT19937: 2 draws per
    /// value (r = 2^32, log2r = 32, k = ceil(53/32) = 2).
    pub fn generate_canonical_f64(&mut self) -> f64 {
        const R: f64 = 4294967296.0; // 2^32
        let d1 = self.next_u32() as f64;
        let d2 = self.next_u32() as f64;
        let sum = d1 + d2 * R;
        let ret = sum / (R * R);
        if ret >= 1.0 {
            f64::from_bits(0x3FEFFFFFFFFFFFFF)
        } else {
            ret
        }
    }

    /// libstdc++ `uniform_real_distribution<double>(a, b)`:
    /// `generate_canonical * (b - a) + a`.
    pub fn uniform_real(&mut self, a: f64, b: f64) -> f64 {
        self.generate_canonical_f64() * (b - a) + a
    }
}

/// Fills the vector x with random numbers between a and b (util.cpp).
pub fn fill_with_random_numbers(x: &mut [f64], a: f64, b: f64) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u32 ^ (d.as_secs() as u32))
        .unwrap_or(5489);
    let mut gen = Mt19937::new(nanos);
    for v in x.iter_mut() {
        *v = gen.uniform_real(a, b);
    }
}

/// Creates random rotation matrices (3×3) via uniform Euler angles (util.cpp).
/// `x.len()` must be a multiple of 9.
pub fn fill_with_random_rotations(x: &mut [f64]) {
    assert_eq!(x.len() % 9, 0, "rotation vector size must be a multiple of 9");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u32 ^ (d.as_secs() as u32))
        .unwrap_or(4357);
    let mut gen = Mt19937::new(nanos);
    let two_pi = std::f64::consts::TAU;
    for chunk in x.chunks_exact_mut(9) {
        // Get a random rotation matrix via uniform Euler angles.
        let e1 = two_pi * gen.uniform_real(0.0, 1.0);
        let e2 = two_pi * gen.uniform_real(0.0, 1.0);
        let e3 = two_pi * gen.uniform_real(0.0, 1.0);
        let (c1, s1) = (e1.cos(), e1.sin());
        let (c2, s2) = (e2.cos(), e2.sin());
        let (c3, s3) = (e3.cos(), e3.sin());

        // Fill the rotation matrix R with the Euler angles.
        chunk[0] = c1 * c3 - c2 * s1 * s3;
        chunk[1] = -c1 * s3 - c2 * c3 * s1;
        chunk[2] = s1 * s2;
        chunk[3] = c3 * s1 + c1 * c2 * s3;
        chunk[4] = c1 * c2 * c3 - s1 * s3;
        chunk[5] = -c1 * s2;
        chunk[6] = s2 * s3;
        chunk[7] = c3 * s2;
        chunk[8] = c2;
    }
}
