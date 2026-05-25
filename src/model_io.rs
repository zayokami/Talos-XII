use crate::binary_codec;
use crate::env_net::EnvNet;
use crate::neural::NeuralLuckOptimizer;
use log::info;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static ATOMIC_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    match read_file_bytes(path) {
        Ok(Some(bytes)) => return Some(bytes),
        Ok(None) => {}
        Err(err) => log::warn!("[Cache] Failed to read {}: {}", path, err),
    }
    let alt = safe_parent_fallback(path)?;
    match read_file_bytes(&alt) {
        Ok(Some(bytes)) => Some(bytes),
        Ok(None) => None,
        Err(err) => {
            log::warn!("[Cache] Failed to read fallback {}: {}", alt, err);
            None
        }
    }
}

fn read_file_bytes(path: &str) -> Result<Option<Vec<u8>>, String> {
    let path_ref = Path::new(path);
    let metadata = match fs::metadata(path_ref) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.to_string()),
    };
    if !metadata.is_file() {
        return Err("path is not a regular file".to_string());
    }
    if metadata.len() == 0 {
        return Err("file is empty".to_string());
    }
    fs::read(path_ref).map(Some).map_err(|err| err.to_string())
}

pub fn load_neural_cache(path: &str) -> Option<NeuralLuckOptimizer> {
    let bytes = read_cache_bytes(path)?;
    match NeuralLuckOptimizer::from_bytes(&bytes) {
        Some(cache) => Some(cache),
        None => {
            log::warn!(
                "[Neural Core] Cache deserialization failed for {}. Rebuilding.",
                path
            );
            None
        }
    }
}

pub fn save_neural_cache(path: &str, net: &NeuralLuckOptimizer) -> bool {
    let bytes = net.to_bytes();
    save_bytes_with_fallback(path, &bytes, "Neural Core")
}

pub fn save_model<T: serde::Serialize>(model: &T, path: &str, label: &str) {
    let bin_path = format!("{}.bin", path);
    match write_file_atomically(&bin_path, |writer| {
        binary_codec::serialize_into(writer, model)
    }) {
        Ok(()) => info!("[{}] Model saved to {} (Binary)", label, bin_path),
        Err(err) => {
            log::warn!("[{}] Failed to save model to {}: {}", label, bin_path, err);
        }
    };

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
    save_bytes_with_fallback(path, json.as_bytes(), "EnvNet")
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

fn save_bytes_with_fallback(path: &str, bytes: &[u8], label: &str) -> bool {
    match write_bytes_atomically(path, bytes) {
        Ok(()) => return true,
        Err(err) => log::warn!("[{}] Failed to save {}: {}", label, path, err),
    }

    let Some(alt) = safe_parent_fallback(path) else {
        return false;
    };
    match write_bytes_atomically(&alt, bytes) {
        Ok(()) => true,
        Err(err) => {
            log::warn!("[{}] Failed to save fallback cache {}: {}", label, alt, err);
            false
        }
    }
}

fn write_bytes_atomically(path: &str, bytes: &[u8]) -> Result<(), String> {
    write_file_atomically(path, |writer| writer.write_all(bytes))
}

fn write_file_atomically<E, F>(path: &str, write_fn: F) -> Result<(), String>
where
    E: std::fmt::Display,
    F: FnOnce(&mut BufWriter<File>) -> Result<(), E>,
{
    let target = Path::new(path);
    if target.as_os_str().is_empty() {
        return Err("target path is empty".to_string());
    }
    if let Some(parent) = target.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create parent directory: {}", err))?;
    }

    let tmp_path = temp_sibling_path(target, "tmp")?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)
        .map_err(|err| format!("failed to create temp file {}: {}", tmp_path.display(), err))?;

    let mut writer = BufWriter::new(file);
    if let Err(err) = write_fn(&mut writer) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("failed while writing temp file: {}", err));
    }
    if let Err(err) = writer.flush() {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("failed to flush temp file: {}", err));
    }

    let file = match writer.into_inner() {
        Ok(file) => file,
        Err(err) => {
            let _ = fs::remove_file(&tmp_path);
            return Err(format!(
                "failed to finalize temp file: {}",
                err.into_error()
            ));
        }
    };
    if let Err(err) = file.sync_all() {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("failed to sync temp file: {}", err));
    }
    drop(file);

    replace_file(&tmp_path, target).map_err(|err| {
        let _ = fs::remove_file(&tmp_path);
        format!("failed to replace {}: {}", target.display(), err)
    })
}

fn replace_file(tmp_path: &Path, target: &Path) -> std::io::Result<()> {
    match fs::rename(tmp_path, target) {
        Ok(()) => return Ok(()),
        Err(err) if !target.exists() => return Err(err),
        Err(_) => {}
    }

    let backup_path = temp_sibling_path(target, "bak").map_err(std::io::Error::other)?;
    fs::rename(target, &backup_path)?;
    match fs::rename(tmp_path, target) {
        Ok(()) => {
            let _ = fs::remove_file(&backup_path);
            Ok(())
        }
        Err(err) => {
            let _ = fs::rename(&backup_path, target);
            Err(err)
        }
    }
}

fn temp_sibling_path(target: &Path, suffix: &str) -> Result<PathBuf, String> {
    let file_name = target
        .file_name()
        .ok_or_else(|| "target path has no file name".to_string())?
        .to_string_lossy();
    let counter = ATOMIC_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!(
        ".{}.{}.{}.{}",
        file_name,
        std::process::id(),
        counter,
        suffix
    );
    Ok(target.with_file_name(tmp_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct TestModel {
        value: u32,
        name: String,
    }

    fn temp_stem(prefix: &str) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir()
            .join(format!("{}_{}_{}", prefix, std::process::id(), now))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn save_model_replaces_existing_binary() {
        let path = temp_stem("talos_model_io_replace");
        let first = TestModel {
            value: 1,
            name: "first".to_string(),
        };
        let second = TestModel {
            value: 2,
            name: "second".to_string(),
        };

        save_model(&first, &path, "Test");
        assert_eq!(load_model::<TestModel>(&path, "Test"), Some(first));
        save_model(&second, &path, "Test");
        assert_eq!(load_model::<TestModel>(&path, "Test"), Some(second));

        let _ = fs::remove_file(format!("{}.bin", path));
    }

    #[test]
    fn load_model_rejects_empty_binary_cache() {
        let path = temp_stem("talos_model_io_empty");
        let bin_path = format!("{}.bin", path);
        File::create(&bin_path).unwrap();

        assert_eq!(load_model::<TestModel>(&path, "Test"), None);

        let _ = fs::remove_file(bin_path);
    }

    #[test]
    fn atomic_write_leaves_existing_file_on_write_error() {
        let path = temp_stem("talos_model_io_atomic_error");
        fs::write(&path, b"original").unwrap();

        let result = write_file_atomically(&path, |writer| -> std::io::Result<()> {
            writer.write_all(b"partial")?;
            Err(std::io::Error::other("injected failure"))
        });

        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");

        let _ = fs::remove_file(path);
    }
}
