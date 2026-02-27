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
    if let Ok(file) = std::fs::File::create(&bin_path) {
        let writer = std::io::BufWriter::new(file);
        if binary_codec::serialize_into(writer, model).is_ok() {
            info!("[{}] Model saved to {} (Binary)", label, bin_path);
        }
    }

    if cfg!(debug_assertions) {
        if let Ok(file) = std::fs::File::create(path) {
            let writer = std::io::BufWriter::new(file);
            if serde_json::to_writer(writer, model).is_ok() {
                info!("[{}] Model saved to {} (JSON)", label, path);
            }
        }
    }
}

pub fn load_model<T: serde::de::DeserializeOwned>(path: &str, label: &str) -> Option<T> {
    let bin_path = format!("{}.bin", path);
    if let Ok(file) = std::fs::File::open(&bin_path) {
        let reader = std::io::BufReader::new(file);
        if let Ok(model) = binary_codec::deserialize_from(reader) {
            info!("[{}] Loaded model from {} (Binary)", label, bin_path);
            return Some(model);
        }
    }

    if let Ok(file) = std::fs::File::open(path) {
        let reader = std::io::BufReader::new(file);
        if let Ok(model) = serde_json::from_reader(reader) {
            info!("[{}] Loaded model from {} (JSON)", label, path);
            return Some(model);
        }
    }
    None
}
