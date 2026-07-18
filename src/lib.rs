//! Talos-XII is a Rust deep-learning framework with autograd, neural-network
//! modules, Transformer components, ACHF layers, and optional CUDA execution.
//!
//! The command-line simulator is one application built on top of this library.

#![allow(clippy::wrong_self_convention)]

pub mod achf;
pub mod autograd;
#[doc(hidden)]
#[cfg(feature = "application")]
pub mod bench;
#[doc(hidden)]
pub mod binary_codec;
#[doc(hidden)]
pub mod calibrate;
#[doc(hidden)]
#[cfg(feature = "application")]
pub mod chart;
#[doc(hidden)]
pub mod collect;
pub mod config;
pub mod cuda;
#[doc(hidden)]
pub mod dbn;
pub mod diagnostics;
#[doc(hidden)]
pub mod dqn;
pub mod dtype;
#[doc(hidden)]
pub mod env_net;
#[doc(hidden)]
pub mod gacha_env;
#[cfg(test)]
mod grad_check;
#[doc(hidden)]
pub mod i18n;
#[doc(hidden)]
pub mod model_init;
#[doc(hidden)]
pub mod model_io;
#[doc(hidden)]
pub mod neural;
pub mod nn;
#[doc(hidden)]
pub mod panic_guard;
#[doc(hidden)]
pub mod policy_eval;
#[doc(hidden)]
pub mod ppo;
#[cfg(feature = "python-bindings")]
pub mod python_bridge;
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod sim;
#[doc(hidden)]
pub mod simd;
#[doc(hidden)]
pub mod strategy;
#[doc(hidden)]
pub mod trainer;
pub mod training_error;
#[doc(hidden)]
pub mod training_metrics;
pub mod transformer;
#[doc(hidden)]
pub mod utils;
#[doc(hidden)]
pub mod worker;

pub use achf::AchfLayer;
pub use autograd::{Device, Tensor};
pub use config::{AchfConfig, ComputeDevice, Config, ConfigError};
pub use dtype::Dtype;
pub use nn::{Linear, Module, RMSNorm};
pub use transformer::{
    KVCache, LuckTransformer, MLAConfig, MultiHeadLatentAttention, RoPE, TransformerBlock,
};

/// Framework-oriented imports for model and training code.
pub mod prelude {
    pub use crate::achf::AchfLayer;
    pub use crate::autograd::{Device, Tensor};
    pub use crate::config::{AchfConfig, ComputeDevice};
    pub use crate::dtype::Dtype;
    pub use crate::nn::{Linear, Module, RMSNorm};
    pub use crate::transformer::{
        KVCache, LuckTransformer, MLAConfig, MultiHeadLatentAttention, RoPE, TransformerBlock,
    };
}
