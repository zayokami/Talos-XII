use crate::binary_codec;
use crate::neural::NeuralLuckOptimizer;
use log::info;

pub fn read_cache_bytes(path: &str) -> Option<Vec<u8>> {
    if let Ok(bytes) = std::fs::read(path) {
        return Some(bytes);
    }
    let alt = format!("../../{}", path);
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
    let alt = format!("../../{}", path);
    std::fs::write(alt, &bytes).is_ok()
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

    if cfg!(debug_assertions) {
        match std::fs::File::create(path) {
            Ok(file) => {
                let writer = std::io::BufWriter::new(file);
                if let Err(e) = serde_json::to_writer(writer, model) {
                    log::warn!("[{}] Failed to serialize model to {}: {}", label, path, e);
                } else {
                    info!("[{}] Model saved to {} (JSON)", label, path);
                }
            }
            Err(e) => {
                log::warn!("[{}] Failed to create {}: {}", label, path, e);
            }
        }
    }
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
