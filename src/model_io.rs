use crate::binary_codec;
use crate::env_net::EnvNet;
use crate::neural::NeuralLuckOptimizer;
use log::info;
use std::path::{Component, Path};

fn safe_parent_fallback(path: &str) -> Option<String> {
    let requested = Path::new(path);
    if requested.is_absolute() || requested.components().any(|c| c == Component::ParentDir) {
        return None;
    }
    Some(
        Path::new("..")
            .join("..")
            .join(requested)
            .to_string_lossy()
            .into_owned(),
    )
}

pub fn read_cache_bytes(path: &str) -> Option<Vec<u8>> {
    if let Ok(bytes) = std::fs::read(path) {
        return Some(bytes);
    }
    let alt = safe_parent_fallback(path)?;
    std::fs::read(alt).ok()
}

pub fn load_neural_cache(path: &str) -> Option<NeuralLuckOptimizer> {
    let bytes = read_cache_bytes(path)?;
    NeuralLuckOptimizer::from_bytes(&bytes)
}

pub fn save_neural_cache(path: &str, net: &NeuralLuckOptimizer) -> bool {
    let bytes = net.to_bytes();
    if std::fs::write(path, &bytes).is_ok() {
        return true;
    }
    match safe_parent_fallback(path) {
        Some(alt) => std::fs::write(alt, &bytes).is_ok(),
        None => false,
    }
}

pub fn save_model<T: serde::Serialize>(model: &T, path: &str, label: &str) {
    let bin_path = format!("{}.bin", path);
    match std::fs::File::create(&bin_path) {
        Ok(file) => {
            let writer = std::io::BufWriter::new(file);
            if let Err(e) = binary_codec::serialize_into(writer, model) {
                log::warn!(
                    "[{}] Failed to serialize model to {}: {}",
                    label,
                    bin_path,
                    e
                );
            } else {
                info!("[{}] Model saved to {} (Binary)", label, bin_path);
            }
        }
        Err(e) => {
            log::warn!("[{}] Failed to create {}: {}", label, bin_path, e);
        }
    }

    // JSON debug dump disabled — binary format is authoritative and JSON serialization
    // of large neural network models (PPO/DQN with millions of f64 weights) takes
    // tens of seconds, causing the program to appear frozen after training.
}

pub fn load_env_net_cache(path: &str) -> Option<EnvNet> {
    let bytes = match read_cache_bytes(path) {
        Some(b) => b,
        None => {
            log::debug!("[EnvNet] Cache file not found: {}", path);
            return None;
        }
    };
    let json_str = match std::str::from_utf8(&bytes) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("[EnvNet] Cache file is not valid UTF-8: {}. Rebuilding.", e);
            return None;
        }
    };
    let mut rng = crate::rng::Rng::from_seed(0);
    match EnvNet::from_json(json_str, &mut rng) {
        Some(net) => Some(net),
        None => {
            log::warn!(
                "[EnvNet] Cache deserialization failed (version/arch/dim mismatch). Rebuilding."
            );
            None
        }
    }
}

/// Atomically save EnvNet cache to avoid corruption on crash.
/// Writes to a temp file first, then renames.
pub fn save_env_net_cache(path: &str, env_net: &EnvNet) -> bool {
    let json = env_net.to_json();
    let tmp_path = format!("{}.tmp", path);

    // Write to temp file
    if let Err(e) = std::fs::write(&tmp_path, json.as_bytes()) {
        log::warn!("[EnvNet] Failed to write temp cache {}: {}", tmp_path, e);
        return false;
    }

    // Atomic rename
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        log::warn!("[EnvNet] Failed to rename temp cache to {}: {}", path, e);
        // Clean up temp file
        let _ = std::fs::remove_file(&tmp_path);
        return false;
    }

    true
}

pub fn load_model<T: serde::de::DeserializeOwned>(path: &str, label: &str) -> Option<T> {
    let bin_path = format!("{}.bin", path);
    match std::fs::File::open(&bin_path) {
        Ok(file) => {
            let reader = std::io::BufReader::new(file);
            match binary_codec::deserialize_from(reader) {
                Ok(model) => {
                    info!("[{}] Loaded model from {} (Binary)", label, bin_path);
                    return Some(model);
                }
                Err(e) => {
                    log::warn!(
                        "[{}] Failed to deserialize model from {}: {}",
                        label,
                        bin_path,
                        e
                    );
                }
            }
        }
        Err(e) => {
            log::warn!("[{}] Failed to open {}: {}", label, bin_path, e);
        }
    }

    match std::fs::File::open(path) {
        Ok(file) => {
            let reader = std::io::BufReader::new(file);
            match serde_json::from_reader(reader) {
                Ok(model) => {
                    info!("[{}] Loaded model from {} (JSON)", label, path);
                    return Some(model);
                }
                Err(e) => {
                    log::warn!(
                        "[{}] Failed to deserialize model from {}: {}",
                        label,
                        path,
                        e
                    );
                }
            }
        }
        Err(e) => {
            log::warn!("[{}] Failed to open {}: {}", label, path, e);
        }
    }
    None
}
