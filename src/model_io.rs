use crate::binary_codec;
use crate::config::Config;
use crate::env_net::EnvNet;
use crate::neural::{NeuralLuckOptimizer, DIM};
use log::info;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static ATOMIC_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);
const CACHE_MANIFEST_SCHEMA: u32 = 1;
const FEATURE_SPEC_VERSION: u32 = 2;

fn executable_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

fn is_cache_relative_path(path: &Path) -> bool {
    !path.is_absolute()
        && !path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
}

fn cache_primary_path(path: &str) -> PathBuf {
    let requested = Path::new(path);
    if requested.as_os_str().is_empty() {
        return requested.to_path_buf();
    }
    if is_cache_relative_path(requested) {
        if let Some(base) = executable_dir() {
            return base.join(requested);
        }
    }
    requested.to_path_buf()
}

fn legacy_parent_fallback(path: &str) -> Option<PathBuf> {
    let requested = Path::new(path);
    if !is_cache_relative_path(requested) {
        return None;
    }
    Some(Path::new("..").join("..").join(requested))
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn cache_read_candidates(path: &str) -> Vec<PathBuf> {
    let mut paths = Vec::with_capacity(3);
    push_unique_path(&mut paths, cache_primary_path(path));

    let requested = PathBuf::from(path);
    if !requested.is_absolute() {
        push_unique_path(&mut paths, requested);
    }
    if let Some(fallback) = legacy_parent_fallback(path) {
        push_unique_path(&mut paths, fallback);
    }
    paths
}

fn cache_write_candidates(path: &str) -> Vec<PathBuf> {
    let mut paths = Vec::with_capacity(3);
    push_unique_path(&mut paths, cache_primary_path(path));

    let requested = PathBuf::from(path);
    if !requested.is_absolute() {
        push_unique_path(&mut paths, requested);
    }
    if let Some(fallback) = legacy_parent_fallback(path) {
        push_unique_path(&mut paths, fallback);
    }
    paths
}

pub fn read_cache_bytes(path: &str) -> Option<Vec<u8>> {
    for candidate in cache_read_candidates(path) {
        match read_file_bytes(&candidate) {
            Ok(Some(bytes)) => return Some(bytes),
            Ok(None) => {}
            Err(err) => log::warn!("[Cache] Failed to read {}: {}", candidate.display(), err),
        }
    }
    None
}

fn read_file_bytes(path_ref: &Path) -> Result<Option<Vec<u8>>, String> {
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

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheProvenance {
    #[default]
    Unknown,
    OfflineTrained,
    OnlineBootstrap,
    OnlineUpdated,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct CacheQualitySummary {
    #[serde(default)]
    pub provenance: CacheProvenance,
    #[serde(default)]
    pub training_steps: Option<usize>,
    #[serde(default)]
    pub final_loss: Option<f64>,
    #[serde(default)]
    pub final_reward: Option<f64>,
    #[serde(default)]
    pub note: Option<String>,
}

impl CacheQualitySummary {
    pub fn training_steps(steps: usize) -> Self {
        Self {
            provenance: CacheProvenance::OfflineTrained,
            training_steps: Some(steps),
            ..Self::default()
        }
    }

    pub fn note(note: impl Into<String>) -> Self {
        Self {
            note: Some(note.into()),
            ..Self::default()
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    pub fn online_bootstrap(note: impl Into<String>) -> Self {
        Self {
            provenance: CacheProvenance::OnlineBootstrap,
            note: Some(note.into()),
            ..Self::default()
        }
    }

    pub fn online_updated(note: impl Into<String>) -> Self {
        Self {
            provenance: CacheProvenance::OnlineUpdated,
            note: Some(note.into()),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CacheManifest {
    pub schema_version: u32,
    pub model_kind: String,
    pub role: String,
    pub config_fingerprint: String,
    pub feature_spec_version: u32,
    pub architecture: String,
    pub source_hash: Option<String>,
    pub artifact_hash: Option<String>,
    pub quality: CacheQualitySummary,
    pub created_unix_secs: u64,
}

impl CacheManifest {
    pub fn expected(model_kind: &str, role: &str, config: &Config, architecture: String) -> Self {
        Self {
            schema_version: CACHE_MANIFEST_SCHEMA,
            model_kind: model_kind.to_string(),
            role: role.to_string(),
            config_fingerprint: config_fingerprint(config),
            feature_spec_version: FEATURE_SPEC_VERSION,
            architecture,
            source_hash: None,
            artifact_hash: None,
            quality: CacheQualitySummary::default(),
            created_unix_secs: 0,
        }
    }

    pub fn with_source_hash(mut self, source_hash: Option<String>) -> Self {
        self.source_hash = source_hash;
        self
    }

    pub fn with_quality(mut self, quality: CacheQualitySummary) -> Self {
        self.quality = quality;
        self
    }

    fn for_saved_artifact(mut self, artifact_hash: Option<String>) -> Self {
        self.artifact_hash = artifact_hash;
        self.created_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self
    }

    fn compatibility_error(&self, expected: &CacheManifest) -> Option<String> {
        if self.schema_version != expected.schema_version {
            return Some(format!(
                "schema {} != {}",
                self.schema_version, expected.schema_version
            ));
        }
        if self.model_kind != expected.model_kind {
            return Some(format!(
                "model_kind {} != {}",
                self.model_kind, expected.model_kind
            ));
        }
        if self.role != expected.role {
            return Some(format!("role {} != {}", self.role, expected.role));
        }
        if self.config_fingerprint != expected.config_fingerprint {
            return Some("config fingerprint mismatch".to_string());
        }
        if self.feature_spec_version != expected.feature_spec_version {
            return Some(format!(
                "feature spec {} != {}",
                self.feature_spec_version, expected.feature_spec_version
            ));
        }
        if self.architecture != expected.architecture {
            return Some("architecture signature mismatch".to_string());
        }
        if self.role == "inference_bf16" && self.source_hash.is_none() {
            return Some("inference manifest missing source artifact hash".to_string());
        }
        if expected.role == "inference_bf16" && expected.source_hash.is_none() {
            return Some("expected inference source artifact hash unavailable".to_string());
        }
        if expected.source_hash.is_some() && self.source_hash != expected.source_hash {
            return Some("source artifact hash mismatch".to_string());
        }
        if let Some(reason) = self.quality.compatibility_error(&expected.quality) {
            return Some(reason);
        }
        None
    }
}

impl CacheQualitySummary {
    fn compatibility_error(&self, expected: &CacheQualitySummary) -> Option<String> {
        match expected.provenance {
            CacheProvenance::Unknown => {}
            CacheProvenance::OfflineTrained => {
                if self.provenance != CacheProvenance::OfflineTrained {
                    return Some(format!(
                        "quality provenance {:?} is not offline_trained",
                        self.provenance
                    ));
                }
            }
            CacheProvenance::OnlineUpdated => {
                if !matches!(
                    self.provenance,
                    CacheProvenance::OfflineTrained
                        | CacheProvenance::OnlineBootstrap
                        | CacheProvenance::OnlineUpdated
                ) {
                    return Some(format!(
                        "quality provenance {:?} is not usable for online training",
                        self.provenance
                    ));
                }
            }
            CacheProvenance::OnlineBootstrap => {
                if self.provenance != CacheProvenance::OnlineBootstrap {
                    return Some(format!(
                        "quality provenance {:?} is not online_bootstrap",
                        self.provenance
                    ));
                }
            }
        }

        if let Some(min_steps) = expected.training_steps {
            let actual_steps = self.training_steps.unwrap_or(0);
            if actual_steps < min_steps {
                return Some(format!(
                    "training_steps {} < required {}",
                    actual_steps, min_steps
                ));
            }
        }
        None
    }
}

pub fn env_net_cache_manifest(config: &Config) -> CacheManifest {
    CacheManifest::expected(
        "env_net",
        "master",
        config,
        "envnet:5_64_32_16_2".to_string(),
    )
}

pub fn neural_cache_manifest(config: &Config) -> CacheManifest {
    CacheManifest::expected(
        "neural_luck_optimizer",
        "master",
        config,
        format!("neural:v2:dim={DIM}:residual=2xdense_layernorm"),
    )
}

pub fn dqn_master_cache_manifest(config: &Config, quality: CacheQualitySummary) -> CacheManifest {
    CacheManifest::expected("dqn", "master", config, dqn_architecture(config)).with_quality(quality)
}

pub fn dqn_inference_cache_manifest(config: &Config, source_hash: Option<String>) -> CacheManifest {
    CacheManifest::expected("dqn", "inference_bf16", config, dqn_architecture(config))
        .with_source_hash(source_hash)
}

pub fn ppo_master_cache_manifest(config: &Config, quality: CacheQualitySummary) -> CacheManifest {
    CacheManifest::expected("ppo", "master", config, ppo_architecture(config)).with_quality(quality)
}

pub fn ppo_inference_cache_manifest(config: &Config, source_hash: Option<String>) -> CacheManifest {
    CacheManifest::expected("ppo", "inference_bf16", config, ppo_architecture(config))
        .with_source_hash(source_hash)
}

fn dqn_architecture(config: &Config) -> String {
    format!(
        "dqn:v2:input={DIM}:hidden={}:actions={}:achf={}:achf_dqn={}:rank={}",
        config.model_hidden_dim,
        crate::utils::ACTION_SPACE,
        config.achf.enabled,
        config.achf.apply_dqn,
        config.achf.rank
    )
}

fn ppo_architecture(config: &Config) -> String {
    format!(
        // v3: apply_attn now structurally attaches an AchfLayer to the MLA w_o
        // projection and rank carries real low-rank semantics, so caches written
        // by v2 binaries are incompatible and must be retrained.
        "ppo:v3:input={DIM}:model_dim={}:hidden={}:layers={}:heads={}:kv_rank={}:rope={}:multi_stream={}:stream_factor={}:actions={}:achf={}:attn={}:ffn={}:rank={}",
        config.model_dim,
        config.model_hidden_dim,
        config.model_num_layers,
        config.model_num_heads,
        config.model_kv_lora_rank,
        config.model_qk_rope_dim,
        config.use_multi_stream,
        config.multi_stream_factor,
        crate::utils::ACTION_SPACE,
        config.achf.enabled,
        config.achf.apply_attn,
        config.achf.apply_ffn,
        config.achf.rank
    )
}

fn config_fingerprint(config: &Config) -> String {
    let payload = format!(
        "v2|p6={:.17}|p5={:.17}|p4={:.17}|up={:.17}|soft_start={}|soft_slope={:.17}|small={}|big={}|up_soft={}|five={}|always5={}|big_requires_not_up={}|fast={}|ppo_mode={}|ppo_steps={}|ppo_update={}|ppo_epochs={}|ppo_batch={}|ppo_ctx={}|ppo_envs={}|ppo_topk={}|luck_cost={:.17}|luck_budget_enabled={}|luck_budget_max={:.17}|luck_budget_initial={:.17}|luck_budget_recovery={:.17}|luck_budget_refund={:.17}|distill={}|distill_decay={:.17}|distill_kl={:.17}|distill_warmup={}|model_dim={}|hidden={}|layers={}|heads={}|kv={}|rope={}|multi_stream={}|stream_factor={}|achf={:?}",
        config.prob_6_base,
        config.prob_5_base,
        config.prob_4_base,
        config.up_rate,
        config.soft_pity_start,
        config.soft_pity_slope,
        config.small_pity_guarantee,
        config.big_pity_cumulative,
        config.up_pity_soft,
        config.five_star_pity,
        config.always_5_star,
        config.big_pity_requires_not_up,
        config.fast_init,
        config.ppo_mode,
        config.ppo_total_steps,
        config.ppo_steps_per_update,
        config.ppo_k_epochs,
        config.ppo_batch_size,
        config.ppo_context_len,
        config.ppo_num_envs,
        config.ppo_top_k,
        config.luck_action_cost,
        config.luck_budget_enabled,
        config.luck_budget_max,
        config.luck_budget_initial,
        config.luck_budget_recovery_per_pull,
        config.luck_budget_negative_refund,
        config.distill_enabled,
        config.distill_ema_decay,
        config.distill_kl_coef,
        config.distill_warmup_steps,
        config.model_dim,
        config.model_hidden_dim,
        config.model_num_layers,
        config.model_num_heads,
        config.model_kv_lora_rank,
        config.model_qk_rope_dim,
        config.use_multi_stream,
        config.multi_stream_factor,
        config.achf
    );
    fnv1a_hex(payload.as_bytes())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    fnv1a_update(&mut hash, bytes);
    format!("{hash:016x}")
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn fnv1a_reader_hex<R: Read>(mut reader: R) -> std::io::Result<String> {
    let mut hash = 0xcbf29ce484222325u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buf)?;
        if read == 0 {
            break;
        }
        fnv1a_update(&mut hash, &buf[..read]);
    }
    Ok(format!("{hash:016x}"))
}

pub fn model_artifact_hash(path: &str) -> Option<String> {
    let bin_path = format!("{}.bin", path);
    artifact_hash_for_path(&bin_path)
}

pub fn serialized_model_hash<T: serde::Serialize>(model: &T) -> Option<String> {
    match binary_codec::to_vec(model) {
        Ok(bytes) => Some(fnv1a_hex(&bytes)),
        Err(err) => {
            log::warn!("[Cache] Failed to hash serialized model: {}", err);
            None
        }
    }
}

fn artifact_hash_for_path(path: &str) -> Option<String> {
    for candidate in cache_read_candidates(path) {
        match File::open(&candidate) {
            Ok(file) => match fnv1a_reader_hex(BufReader::new(file)) {
                Ok(hash) => return Some(hash),
                Err(err) => {
                    log::warn!("[Cache] Failed to hash {}: {}", candidate.display(), err);
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                log::warn!(
                    "[Cache] Failed to open {} for hashing: {}",
                    candidate.display(),
                    err
                );
            }
        }
    }
    None
}

pub fn cache_artifact_hash(path: &str) -> Option<String> {
    artifact_hash_for_path(path)
}

fn cache_manifest_path(path: &str) -> String {
    format!("{}.manifest.json", path)
}

fn load_manifest(path: &str) -> Option<CacheManifest> {
    let manifest_path = cache_manifest_path(path);
    let bytes = read_cache_bytes(&manifest_path)?;
    match serde_json::from_slice::<CacheManifest>(&bytes) {
        Ok(manifest) => Some(manifest),
        Err(err) => {
            log::warn!(
                "[Cache] Failed to parse manifest {}: {}",
                manifest_path,
                err
            );
            None
        }
    }
}

fn save_manifest(path: &str, manifest: &CacheManifest) -> bool {
    let manifest_path = cache_manifest_path(path);
    match serde_json::to_vec_pretty(manifest) {
        Ok(bytes) => save_bytes_with_fallback(&manifest_path, &bytes, "Cache Manifest"),
        Err(err) => {
            log::warn!(
                "[Cache] Failed to serialize manifest {}: {}",
                manifest_path,
                err
            );
            false
        }
    }
}

fn manifest_allows_load(path: &str, expected: &CacheManifest, artifact_path: Option<&str>) -> bool {
    let Some(manifest) = load_manifest(path) else {
        log::warn!("[Cache] Missing manifest for {}. Rebuilding.", path);
        return false;
    };
    if let Some(reason) = manifest.compatibility_error(expected) {
        log::warn!(
            "[Cache] Manifest mismatch for {}: {}. Rebuilding.",
            path,
            reason
        );
        return false;
    }
    if let Some(artifact_path) = artifact_path {
        let Some(expected_hash) = &manifest.artifact_hash else {
            log::warn!(
                "[Cache] Manifest for {} is missing artifact hash. Rebuilding.",
                path
            );
            return false;
        };
        match cache_artifact_hash(artifact_path) {
            Some(actual_hash) if &actual_hash == expected_hash => {}
            Some(_) => {
                log::warn!("[Cache] Artifact hash mismatch for {}. Rebuilding.", path);
                return false;
            }
            None => {
                log::warn!("[Cache] Missing artifact for {}. Rebuilding.", path);
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
fn cache_manifest_is_compatible(path: &str, expected: &CacheManifest) -> bool {
    manifest_allows_load(path, expected, Some(path))
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

pub fn load_neural_cache_with_manifest(
    path: &str,
    expected: &CacheManifest,
) -> Option<NeuralLuckOptimizer> {
    if !manifest_allows_load(path, expected, Some(path)) {
        return None;
    }
    load_neural_cache(path)
}

pub fn save_neural_cache_with_manifest(
    path: &str,
    net: &NeuralLuckOptimizer,
    manifest: CacheManifest,
) -> bool {
    let saved = save_neural_cache(path, net);
    if saved {
        let Some(artifact_hash) = cache_artifact_hash(path) else {
            log::warn!(
                "[Neural Core] Cache saved, but artifact hash is unavailable for {}",
                path
            );
            return false;
        };
        let manifest = manifest.for_saved_artifact(Some(artifact_hash));
        return save_manifest(path, &manifest);
    }
    saved
}

pub fn save_model<T: serde::Serialize>(model: &T, path: &str, label: &str) -> bool {
    let bin_path = format!("{}.bin", path);
    let display_path = cache_primary_path(&bin_path);
    let saved = match write_file_atomically(&bin_path, |writer| {
        binary_codec::serialize_into(writer, model)
    }) {
        Ok(()) => {
            info!(
                "[{}] Model saved to {} (Binary)",
                label,
                display_path.display()
            );
            true
        }
        Err(err) => {
            log::warn!(
                "[{}] Failed to save model to {}: {}",
                label,
                display_path.display(),
                err
            );
            false
        }
    };

    // JSON debug dump disabled — binary format is authoritative and JSON serialization
    // of large neural network models (PPO/DQN with millions of f64 weights) takes
    // tens of seconds, causing the program to appear frozen after training.
    saved
}

pub fn save_model_with_manifest<T: serde::Serialize>(
    model: &T,
    path: &str,
    label: &str,
    manifest: CacheManifest,
) -> bool {
    if !save_model(model, path, label) {
        return false;
    }
    let Some(artifact_hash) = model_artifact_hash(path) else {
        log::warn!(
            "[{}] Model saved, but artifact hash is unavailable for {}",
            label,
            path
        );
        return false;
    };
    let manifest = manifest.for_saved_artifact(Some(artifact_hash));
    save_manifest(path, &manifest)
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

pub fn load_env_net_cache_with_manifest(path: &str, expected: &CacheManifest) -> Option<EnvNet> {
    if !manifest_allows_load(path, expected, Some(path)) {
        return None;
    }
    load_env_net_cache(path)
}

pub fn save_env_net_cache_with_manifest(
    path: &str,
    env_net: &EnvNet,
    manifest: CacheManifest,
) -> bool {
    let saved = save_env_net_cache(path, env_net);
    if saved {
        let Some(artifact_hash) = cache_artifact_hash(path) else {
            log::warn!(
                "[EnvNet] Cache saved, but artifact hash is unavailable for {}",
                path
            );
            return false;
        };
        let manifest = manifest.for_saved_artifact(Some(artifact_hash));
        return save_manifest(path, &manifest);
    }
    saved
}

pub fn load_model<T: serde::de::DeserializeOwned>(path: &str, label: &str) -> Option<T> {
    let bin_path = format!("{}.bin", path);
    if let Some((file, actual_path)) = open_cache_file(&bin_path, label) {
        let reader = std::io::BufReader::new(file);
        match binary_codec::deserialize_from(reader) {
            Ok(model) => {
                info!(
                    "[{}] Loaded model from {} (Binary)",
                    label,
                    actual_path.display()
                );
                return Some(model);
            }
            Err(e) => {
                log::warn!(
                    "[{}] Failed to deserialize model from {}: {}",
                    label,
                    actual_path.display(),
                    e
                );
            }
        };
    }

    if let Some((file, actual_path)) = open_cache_file(path, label) {
        let reader = std::io::BufReader::new(file);
        match serde_json::from_reader(reader) {
            Ok(model) => {
                info!(
                    "[{}] Loaded model from {} (JSON)",
                    label,
                    actual_path.display()
                );
                return Some(model);
            }
            Err(e) => {
                log::warn!(
                    "[{}] Failed to deserialize model from {}: {}",
                    label,
                    actual_path.display(),
                    e
                );
            }
        };
    }
    None
}

fn open_cache_file(path: &str, label: &str) -> Option<(File, PathBuf)> {
    for candidate in cache_read_candidates(path) {
        match File::open(&candidate) {
            Ok(file) => return Some((file, candidate)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                log::warn!(
                    "[{}] Failed to open {}: {}",
                    label,
                    candidate.display(),
                    err
                );
            }
        }
    }
    None
}

pub fn load_model_with_manifest<T: serde::de::DeserializeOwned>(
    path: &str,
    label: &str,
    expected: &CacheManifest,
) -> Option<T> {
    let bin_path = format!("{}.bin", path);
    if !manifest_allows_load(path, expected, Some(&bin_path)) {
        return None;
    }
    load_model(path, label)
}

pub fn load_model_with_manifest_allow_source_mismatch<T: serde::de::DeserializeOwned>(
    path: &str,
    label: &str,
    expected: &CacheManifest,
) -> Option<T> {
    let Some(manifest) = load_manifest(path) else {
        log::warn!("[Cache] Missing manifest for {}. Rebuilding.", path);
        return None;
    };
    let mut relaxed_expected = expected.clone();
    relaxed_expected.source_hash = manifest.source_hash.clone();
    let bin_path = format!("{}.bin", path);
    if !manifest_allows_load(path, &relaxed_expected, Some(&bin_path)) {
        return None;
    }
    if expected.source_hash.is_some() && manifest.source_hash != expected.source_hash {
        log::warn!(
            "[Cache] Source artifact hash mismatch for {}. Using cached {} because config and architecture still match; use force retrain to rebuild it.",
            path,
            label
        );
    } else if expected.source_hash.is_none() && manifest.source_hash.is_some() {
        log::warn!(
            "[Cache] Source artifact hash for {} was not recomputed during startup. Using cached {} after artifact/config validation.",
            path,
            label
        );
    }
    load_model(path, label)
}

fn save_bytes_with_fallback(path: &str, bytes: &[u8], label: &str) -> bool {
    let mut last_error = None;
    for candidate in cache_write_candidates(path) {
        match write_bytes_atomically_at(&candidate, bytes) {
            Ok(()) => return true,
            Err(err) => {
                log::warn!(
                    "[{}] Failed to save {}: {}",
                    label,
                    candidate.display(),
                    err
                );
                last_error = Some(err);
            }
        }
    }
    if last_error.is_none() {
        log::warn!("[{}] Failed to save {}: no candidate path", label, path);
    }
    false
}

fn write_bytes_atomically_at(path: &Path, bytes: &[u8]) -> Result<(), String> {
    write_file_atomically_at(path, |writer| writer.write_all(bytes))
}

fn write_file_atomically<E, F>(path: &str, write_fn: F) -> Result<(), String>
where
    E: std::fmt::Display,
    F: FnOnce(&mut BufWriter<File>) -> Result<(), E>,
{
    let target = cache_primary_path(path);
    write_file_atomically_at(&target, write_fn)
}

fn write_file_atomically_at<E, F>(target: &Path, write_fn: F) -> Result<(), String>
where
    E: std::fmt::Display,
    F: FnOnce(&mut BufWriter<File>) -> Result<(), E>,
{
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

    fn unique_relative_name(prefix: &str) -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("{}_{}_{}", prefix, std::process::id(), now)
    }

    #[test]
    fn relative_cache_names_default_to_executable_dir() {
        let name = unique_relative_name("talos_model_io_relative_bytes.cache");
        let primary = cache_primary_path(&name);
        let exe_dir = executable_dir().expect("test executable path should be available");
        assert_eq!(primary, exe_dir.join(&name));

        let cwd_path = PathBuf::from(&name);
        let _ = fs::remove_file(&primary);
        let _ = fs::remove_file(&cwd_path);

        assert!(save_bytes_with_fallback(&name, b"cached", "Test Cache"));
        assert_eq!(fs::read(&primary).unwrap(), b"cached");
        assert_eq!(read_cache_bytes(&name).unwrap(), b"cached");

        let _ = fs::remove_file(primary);
        let _ = fs::remove_file(cwd_path);
    }

    #[test]
    fn relative_model_cache_loads_from_executable_dir() {
        let path = unique_relative_name("talos_model_io_relative_model");
        let bin_path = cache_primary_path(&format!("{}.bin", path));
        let cwd_bin_path = PathBuf::from(format!("{}.bin", path));
        let _ = fs::remove_file(&bin_path);
        let _ = fs::remove_file(&cwd_bin_path);

        let model = TestModel {
            value: 7,
            name: "relative".to_string(),
        };
        assert!(save_model(&model, &path, "Test"));
        assert!(bin_path.exists());
        assert_eq!(load_model::<TestModel>(&path, "Test"), Some(model));

        let _ = fs::remove_file(bin_path);
        let _ = fs::remove_file(cwd_bin_path);
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

    #[test]
    fn manifest_rejects_config_fingerprint_mismatch() {
        let path = temp_stem("talos_model_io_manifest");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = env_net_cache_manifest(&config);
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"artifact")))
        ));
        assert!(cache_manifest_is_compatible(
            &path,
            &env_net_cache_manifest(&config)
        ));

        let mut changed = config.clone();
        changed.model_hidden_dim += 1;
        assert!(!cache_manifest_is_compatible(
            &path,
            &env_net_cache_manifest(&changed)
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn manifest_rejects_artifact_hash_mismatch() {
        let path = temp_stem("talos_model_io_artifact_hash");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = env_net_cache_manifest(&config);
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"different")))
        ));

        assert!(!cache_manifest_is_compatible(
            &path,
            &env_net_cache_manifest(&config)
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn manifest_rejects_inference_without_source_hash() {
        let path = temp_stem("talos_model_io_missing_source");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = dqn_inference_cache_manifest(&config, None);
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"artifact")))
        ));

        assert!(!cache_manifest_is_compatible(
            &path,
            &dqn_inference_cache_manifest(&config, Some("master".to_string()))
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn manifest_rejects_source_hash_mismatch() {
        let path = temp_stem("talos_model_io_source_hash");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = dqn_inference_cache_manifest(&config, Some("old".to_string()));
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"artifact")))
        ));

        assert!(!cache_manifest_is_compatible(
            &path,
            &dqn_inference_cache_manifest(&config, Some("new".to_string()))
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn relaxed_model_load_allows_only_source_hash_mismatch() {
        let path = temp_stem("talos_model_io_relaxed_source_hash");
        let model = TestModel {
            value: 42,
            name: "cached".to_string(),
        };

        let config = crate::config::Config::default();
        let saved_manifest =
            dqn_master_cache_manifest(&config, CacheQualitySummary::training_steps(1))
                .with_source_hash(Some("old-source".to_string()));
        assert!(save_model_with_manifest(
            &model,
            &path,
            "Test",
            saved_manifest
        ));

        let expected_manifest =
            dqn_master_cache_manifest(&config, CacheQualitySummary::training_steps(1))
                .with_source_hash(Some("new-source".to_string()));
        assert_eq!(
            load_model_with_manifest::<TestModel>(&path, "Test", &expected_manifest),
            None
        );
        assert_eq!(
            load_model_with_manifest_allow_source_mismatch::<TestModel>(
                &path,
                "Test",
                &expected_manifest
            ),
            Some(TestModel {
                value: 42,
                name: "cached".to_string(),
            })
        );

        let mut changed = config.clone();
        changed.model_hidden_dim += 1;
        let incompatible_manifest =
            dqn_master_cache_manifest(&changed, CacheQualitySummary::training_steps(1))
                .with_source_hash(Some("new-source".to_string()));
        assert_eq!(
            load_model_with_manifest_allow_source_mismatch::<TestModel>(
                &path,
                "Test",
                &incompatible_manifest
            ),
            None
        );

        let _ = fs::remove_file(format!("{}.bin", path));
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn ppo_manifest_rejects_multi_stream_shape_change() {
        let path = temp_stem("talos_model_io_ppo_shape");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = ppo_master_cache_manifest(&config, CacheQualitySummary::training_steps(1));
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"artifact")))
        ));

        let mut changed = config.clone();
        changed.multi_stream_factor += 1;
        assert!(!cache_manifest_is_compatible(
            &path,
            &ppo_master_cache_manifest(&changed, CacheQualitySummary::training_steps(1))
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }

    #[test]
    fn offline_manifest_rejects_online_bootstrap_quality() {
        let path = temp_stem("talos_model_io_bootstrap_quality");
        fs::write(&path, b"artifact").unwrap();

        let config = crate::config::Config::default();
        let manifest = dqn_master_cache_manifest(
            &config,
            CacheQualitySummary::online_bootstrap("random init"),
        );
        assert!(save_manifest(
            &path,
            &manifest.for_saved_artifact(Some(fnv1a_hex(b"artifact")))
        ));

        assert!(!cache_manifest_is_compatible(
            &path,
            &dqn_master_cache_manifest(&config, CacheQualitySummary::training_steps(1))
        ));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(cache_manifest_path(&path));
    }
}
