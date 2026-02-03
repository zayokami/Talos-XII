use rand_core::{Error, RngCore};
use std::time::{SystemTime, UNIX_EPOCH};

// --- Pseudo-Random Number Generator (Hand-rolled) ---
// Algorithm: xoshiro256** (StarStar)
// Reference: https://prng.di.unimi.it/
// This is a high-quality PRNG suitable for simulations, far better than simple LCG or Xorshift.

pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    pub fn from_seed(mut seed: u64) -> Self {
        // 2. Use SplitMix64 to initialize the 4 states of Xoshiro256 from one seed
        // SplitMix64 is the standard way to seed Xoshiro from a 64-bit value.
        let sm64 = |s: &mut u64| -> u64 {
            *s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };

        Rng {
            state: [
                sm64(&mut seed),
                sm64(&mut seed),
                sm64(&mut seed),
                sm64(&mut seed),
            ],
        }
    }

    pub fn new() -> Self {
        // 1. Get a seed from system time
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        // ENHANCEMENT 1: Better Seeding
        // Mix SystemTime (nanoseconds) with the memory address of a heap allocation.
        // This utilizes ASLR (Address Space Layout Randomization) as an additional entropy source,
        // making the seed much harder to predict than just using time.
        let heap_var = Box::new(0);
        let ptr_val = &*heap_var as *const i32 as u64;
        let seed = (since_the_epoch.as_nanos() as u64) ^ ptr_val;

        Self::from_seed(seed)
    }

    // Generate next u64 using xoshiro256**
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;

        self.state[3] = self.state[3].rotate_left(45);

        result
    }

    // Generate float in [0.0, 1.0)
    // Standard conversion: (u64 >> 11) * 2^-53
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / 9007199254740992.0)
    }

    // Standard Normal Distribution using Box-Muller Transform
    // Returns a value normally distributed with mean 0.0 and std_dev 1.0
    #[inline]
    #[allow(dead_code)]
    pub fn next_f64_normal(&mut self) -> f64 {
        // Box-Muller transform generates two independent standard normal random variables
        // from two independent uniform random variables in (0, 1].
        // z0 = sqrt(-2 ln u1) cos(2 pi u2)
        // z1 = sqrt(-2 ln u1) sin(2 pi u2)
        // We only return z0 here for simplicity (stateful caching of z1 is possible but adds complexity)

        let u1 = loop {
            let u = self.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.next_f64();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;

        r * theta.cos()
    }

    // ENHANCEMENT 2: Unbiased Range Generation
    // Generates a random number in [0, range) using Rejection Sampling.
    // This eliminates "Modulo Bias" which occurs when the range is not a power of 2.
    // Reference: https://lemire.me/blog/2019/06/06/nearly-divisionless-random-integer-generation-on-various-systems/
    pub fn next_u64_bounded(&mut self, range: u64) -> u64 {
        let threshold = (0u64.wrapping_sub(range)) % range;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % range;
            }
        }
    }
}

impl RngCore for Rng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i < dest.len() {
            let v = self.next_u64();
            let bytes = v.to_le_bytes();
            let n = std::cmp::min(dest.len() - i, 8);
            dest[i..i + n].copy_from_slice(&bytes[..n]);
            i += n;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}
